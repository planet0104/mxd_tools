//! NEAT 训练计分：YOLO 可见拾取（主分）+ 视觉动作 shaping + 可选内存 shaping。

use super::observation::{obs_has_drop, obs_has_enemy, OBS_DIM};
use crate::yolo::Detection;

use super::types::DropKind;

const MESO_LABEL: &str = "金币";
const POTION_LABEL: &str = "药水";

/// 视觉动作 shaping 分值（与部署观测一致，权重低）。
const PTS_ATTACK_ALIGN: f32 = 0.5;
const PTS_PICKUP_ALIGN: f32 = 2.5;
/// 每局视觉 shaping 总分上限（防止站桩刷分）。
const VISION_SHAPING_CAP: f32 = 60.0;
/// 内存 shaping 原始分（最终 × `memory_weight` 计入总分）。
const PTS_MOB_HIT: f32 = 5.0;
const PTS_MOB_KILL: f32 = 25.0;
/// 击杀后短时内捡到 YOLO 可见掉落物的额外奖励（鼓励砍怪→捡币链路）。
const PTS_KILL_LOOT_CHAIN: f32 = 18.0;
const KILL_LOOT_CHAIN_TICKS: u64 = 180;
/// 有击杀但整局未捡到 YOLO 可见金币时的局末惩罚。
const NO_PICKUP_AFTER_KILL_PENALTY: f32 = 25.0;
/// 触发局末「有杀无捡」惩罚所需最少击杀数。
const NO_PICKUP_KILL_THRESHOLD: u32 = 2;
/// YOLO 多次看到地上金币却未捡到时的局末惩罚。
const IGNORE_VISIBLE_LOOT_PENALTY: f32 = 15.0;
/// 触发「见币不捡」惩罚所需 YOLO 可见金币帧数（约 pace=12 时 30 帧 ≈ 6s）。
const IGNORE_VISIBLE_LOOT_FRAMES: u32 = 30;
/// 视觉攻击 shaping：仅当 obs 有敌人且 `Attack` 后，在此窗口内实际命中怪才加分。
const ATTACK_HIT_SHAPING_TICKS: u64 = 30;
/// 连续停滞达到此时长（逻辑 tick，60Hz）且期间无新拾取/击杀则扣分或早停。
pub const STAGNATION_TICKS: u32 = 300;
/// 相对锚点 **水平** 位移超过该值（px）视为有移动，重置停滞计时（跳跃不计）。
const STAGNATION_MOVE_THRESHOLD: f32 = 48.0;
/// 命中怪后暂缓无产出早停的时长（逻辑 tick，60Hz，15s）；地上有可见金币时不生效。
pub const MOB_HIT_FORFEIT_EXEMPT_TICKS: u32 = 900;
/// 每次触发停滞惩罚扣分（拾取主分远大于此，避免站桩仍划算）。
const STAGNATION_PENALTY: f32 = 15.0;
/// 单局停滞惩罚累计上限（防止早期全负适应度）。
const STAGNATION_PENALTY_CAP: f32 = 90.0;
/// 开局宽限：此 tick 数内不触发无产出早停（给首帧视觉与随机探索留时间）。
pub const IDLE_FORFEIT_GRACE_TICKS: u32 = 600;

/// 训练适应度 shaping 配置（仅影响计分，不进入 NEAT 观测）。
#[derive(Debug, Clone, Copy)]
pub struct FitnessShapingConfig {
    /// 内存事件分（命中/击杀）权重；0=仅视觉拾取+视觉 shaping。
    pub memory_weight: f32,
}

impl Default for FitnessShapingConfig {
    fn default() -> Self {
        Self {
            memory_weight: 0.25,
        }
    }
}

impl FitnessShapingConfig {
    pub fn disabled() -> Self {
        Self {
            memory_weight: 0.0,
        }
    }
}

/// 上一帧 YOLO 可见的掉落框（屏幕坐标 xyxy）。
#[derive(Debug, Clone, Default)]
pub struct VisibleLoot {
    pub meso: Vec<[f32; 4]>,
    pub potions: Vec<[f32; 4]>,
}

/// 训练个体累计得分。
#[derive(Debug, Clone)]
pub struct TrainingFitness {
    /// 总分（拾取 + 视觉 shaping + memory_weight × 内存 shaping）。
    pub score: f32,
    pub pickup_score: f32,
    pub vision_shaping_score: f32,
    pub memory_shaping_score: f32,
    pub meso_events: u32,
    pub potion_events: u32,
    pub meso_units: u32,
    pub attack_align_events: u32,
    pub pickup_align_events: u32,
    pub mob_hit_events: u32,
    pub mob_kill_events: u32,
    /// 本局累计停滞惩罚（正数表示已扣分数）。
    pub stagnation_penalty: f32,
    pub stagnation_penalty_events: u32,
    /// 局末「有杀无捡 / 见币不捡」等惩罚合计。
    pub episode_penalty: f32,
    /// 是否因「站桩且无拾取/命中」被提前结束本局。
    pub idle_forfeit: bool,
    shaping: FitnessShapingConfig,
    last_obs: [f32; OBS_DIM],
    last_visible: VisibleLoot,
    /// 本 tick 观测是否刚由视觉帧更新（仅此时计 shaping 分）。
    observation_fresh: bool,
    stagnation_anchor_x: f32,
    stagnation_ticks: u32,
    stagnation_initialized: bool,
    last_x: f32,
    last_y: f32,
    last_pickup_tick: u64,
    last_mob_hit_tick: u64,
    last_mob_kill_tick: u64,
    /// 最近一次「obs 有敌人 + Attack」的视觉帧 tick；命中窗口内实际打到怪才给攻击 shaping。
    pending_attack_align_tick: Option<u64>,
    /// YOLO 帧中检测到地上金币的次数（用于局末见币不捡惩罚）。
    visible_meso_frames: u32,
    episode_finalized: bool,
}

impl Default for TrainingFitness {
    fn default() -> Self {
        Self::with_shaping(FitnessShapingConfig::default())
    }
}

impl TrainingFitness {
    pub fn with_shaping(shaping: FitnessShapingConfig) -> Self {
        Self {
            score: 0.0,
            pickup_score: 0.0,
            vision_shaping_score: 0.0,
            memory_shaping_score: 0.0,
            meso_events: 0,
            potion_events: 0,
            meso_units: 0,
            attack_align_events: 0,
            pickup_align_events: 0,
            mob_hit_events: 0,
            mob_kill_events: 0,
            stagnation_penalty: 0.0,
            stagnation_penalty_events: 0,
            episode_penalty: 0.0,
            idle_forfeit: false,
            shaping,
            last_obs: [0.0; OBS_DIM],
            last_visible: VisibleLoot::default(),
            observation_fresh: false,
            stagnation_anchor_x: 0.0,
            stagnation_ticks: 0,
            stagnation_initialized: false,
            last_x: 0.0,
            last_y: 0.0,
            last_pickup_tick: 0,
            last_mob_hit_tick: 0,
            last_mob_kill_tick: 0,
            pending_attack_align_tick: None,
            visible_meso_frames: 0,
            episode_finalized: false,
        }
    }

    pub fn configure_shaping(&mut self, shaping: FitnessShapingConfig) {
        self.shaping = shaping;
    }

    pub fn shaping_config(&self) -> FitnessShapingConfig {
        self.shaping
    }

    /// 每帧 `perceive` 后调用，记录 YOLO 可见掉落与上一帧观测。
    pub fn record_visible_drops(&mut self, detections: &[Detection]) {
        let mut vis = VisibleLoot::default();
        for d in detections {
            let b = [d.x1, d.y1, d.x2, d.y2];
            match d.label {
                MESO_LABEL => vis.meso.push(b),
                POTION_LABEL => vis.potions.push(b),
                _ => {}
            }
        }
        if !vis.meso.is_empty() {
            self.visible_meso_frames = self.visible_meso_frames.saturating_add(1);
        }
        self.last_visible = vis;
    }

    pub fn set_last_observation(&mut self, obs: &[f32]) {
        let n = obs.len().min(OBS_DIM);
        self.last_obs[..n].copy_from_slice(&obs[..n]);
        self.observation_fresh = true;
    }

    /// 本帧输入与观测一致时给小分（仅视觉帧、且未超上限）。
    /// 攻击 shaping 仅登记待命中窗口，实际加分在 `record_mob_hit`。
    pub fn try_score_input(&mut self, input: &super::InputFrame, episode_tick: u64) {
        self.expire_pending_attack_align(episode_tick);
        if !self.observation_fresh || self.vision_shaping_score >= VISION_SHAPING_CAP {
            return;
        }
        self.observation_fresh = false;
        if input.attack && obs_has_enemy(&self.last_obs) {
            self.pending_attack_align_tick = Some(episode_tick);
        }
        if input.pick_up && obs_has_drop(&self.last_obs) {
            self.vision_shaping_score += PTS_PICKUP_ALIGN;
            self.score += PTS_PICKUP_ALIGN;
            self.pickup_align_events += 1;
        }
    }

    fn expire_pending_attack_align(&mut self, episode_tick: u64) {
        if let Some(p) = self.pending_attack_align_tick {
            if episode_tick.saturating_sub(p) > ATTACK_HIT_SHAPING_TICKS {
                self.pending_attack_align_tick = None;
            }
        }
    }

    fn try_grant_attack_hit_shaping(&mut self, episode_tick: u64) {
        if self.vision_shaping_score >= VISION_SHAPING_CAP {
            self.pending_attack_align_tick = None;
            return;
        }
        let Some(p) = self.pending_attack_align_tick else {
            return;
        };
        if episode_tick < p || episode_tick.saturating_sub(p) > ATTACK_HIT_SHAPING_TICKS {
            return;
        }
        self.vision_shaping_score += PTS_ATTACK_ALIGN;
        self.score += PTS_ATTACK_ALIGN;
        self.attack_align_events += 1;
        self.pending_attack_align_tick = None;
    }

    pub fn record_mob_hit(&mut self, episode_tick: u64) {
        self.try_grant_attack_hit_shaping(episode_tick);
        self.mob_hit_events += 1;
        self.last_mob_hit_tick = episode_tick;
        self.note_stagnation_activity();
        if self.shaping.memory_weight <= 0.0 {
            return;
        }
        self.memory_shaping_score += PTS_MOB_HIT;
        self.score += self.shaping.memory_weight * PTS_MOB_HIT;
    }

    pub fn record_mob_kill(&mut self, episode_tick: u64) {
        self.mob_kill_events += 1;
        self.last_mob_kill_tick = episode_tick;
        self.note_stagnation_activity();
        if self.shaping.memory_weight <= 0.0 {
            return;
        }
        self.memory_shaping_score += PTS_MOB_KILL;
        self.score += self.shaping.memory_weight * PTS_MOB_KILL;
    }

    /// 每逻辑 tick 调用：水平位移不足且期间无新拾取/击杀时，满 5s 扣一次分；否则早停。
    pub fn tick_stagnation(&mut self, x: f32, y: f32, episode_tick: u64) -> bool {
        self.last_x = x;
        self.last_y = y;
        if !self.stagnation_initialized {
            self.reset_stagnation_anchor(x);
            self.stagnation_initialized = true;
            return false;
        }

        let dx = x - self.stagnation_anchor_x;
        let thresh = STAGNATION_MOVE_THRESHOLD;
        if dx * dx > thresh * thresh {
            self.reset_stagnation_anchor(x);
            return false;
        }

        self.stagnation_ticks += 1;
        if self.stagnation_ticks < STAGNATION_TICKS {
            return false;
        }

        if episode_tick >= IDLE_FORFEIT_GRACE_TICKS as u64 && !self.blocks_idle_forfeit(episode_tick) {
            self.idle_forfeit = true;
            return true;
        }

        if self.stagnation_penalty < STAGNATION_PENALTY_CAP {
            let applied = STAGNATION_PENALTY.min(STAGNATION_PENALTY_CAP - self.stagnation_penalty);
            self.stagnation_penalty += applied;
            self.score -= applied;
            self.stagnation_penalty_events += 1;
        }
        self.stagnation_ticks = 0;
        false
    }

    /// 停滞窗口内的新拾取/击杀，或 15s 内命中（且地上无可见金币），暂缓无产出早停。
    fn blocks_idle_forfeit(&self, episode_tick: u64) -> bool {
        let window = u64::from(STAGNATION_TICKS);
        if self.last_pickup_tick > 0
            && episode_tick.saturating_sub(self.last_pickup_tick) < window
        {
            return true;
        }
        if self.last_mob_kill_tick > 0
            && episode_tick.saturating_sub(self.last_mob_kill_tick) < window
        {
            return true;
        }
        // 地上有可见金币时，命中怪不再豁免早停，避免站桩砍怪刷 shaping。
        if self.last_visible.meso.is_empty()
            && self.last_mob_hit_tick > 0
            && episode_tick.saturating_sub(self.last_mob_hit_tick)
                < u64::from(MOB_HIT_FORFEIT_EXEMPT_TICKS)
        {
            return true;
        }
        false
    }

    /// 局末结算（可重复调用，仅生效一次）：惩罚「只杀不捡 / 见币不捡」。
    pub fn finalize_episode(&mut self) {
        if self.episode_finalized {
            return;
        }
        self.episode_finalized = true;

        if self.meso_events == 0
            && self.mob_kill_events >= NO_PICKUP_KILL_THRESHOLD
        {
            self.episode_penalty += NO_PICKUP_AFTER_KILL_PENALTY;
            self.score -= NO_PICKUP_AFTER_KILL_PENALTY;
        }

        if self.meso_events == 0 && self.visible_meso_frames >= IGNORE_VISIBLE_LOOT_FRAMES {
            self.episode_penalty += IGNORE_VISIBLE_LOOT_PENALTY;
            self.score -= IGNORE_VISIBLE_LOOT_PENALTY;
        }
    }

    fn note_stagnation_activity(&mut self) {
        if self.stagnation_initialized {
            self.reset_stagnation_anchor(self.last_x);
        }
    }

    fn reset_stagnation_anchor(&mut self, x: f32) {
        self.stagnation_anchor_x = x;
        self.stagnation_ticks = 0;
    }

    /// 拾取成功时调用；仅当掉落物中心落在上一帧 YOLO 框内才加分。
    pub fn try_score_pickup(
        &mut self,
        kind: DropKind,
        x: f32,
        y: f32,
        meso_amount: u32,
        episode_tick: u64,
    ) -> f32 {
        let gained = match kind {
            DropKind::Meso => {
                if !point_in_any_box(x, y, &self.last_visible.meso) {
                    return 0.0;
                }
                let pts = meso_amount as f32;
                self.meso_events += 1;
                self.meso_units += meso_amount;
                pts
            }
            DropKind::RedPotion => {
                if !point_in_any_box(x, y, &self.last_visible.potions) {
                    return 0.0;
                }
                self.potion_events += 1;
                50.0
            }
        };
        if gained > 0.0 {
            self.last_pickup_tick = episode_tick;
            self.note_stagnation_activity();
            if self.last_mob_kill_tick > 0
                && episode_tick.saturating_sub(self.last_mob_kill_tick) <= KILL_LOOT_CHAIN_TICKS
            {
                self.score += PTS_KILL_LOOT_CHAIN;
            }
        }
        self.pickup_score += gained;
        self.score += gained;
        gained
    }

    pub fn reset(&mut self) {
        let shaping = self.shaping;
        *self = Self::with_shaping(shaping);
    }
}

fn point_in_any_box(x: f32, y: f32, boxes: &[[f32; 4]]) -> bool {
    boxes.iter().any(|b| x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::input::InputFrame;
    use crate::game::observation::{OBS_DROP_SLOTS, OBS_ENEMY_SLOTS, OBS_FLOOR_SLOTS, OBS_SELF, OBS_SLOT_DIM};
    use crate::yolo::CLASS_NAMES;
    use crate::yolo::Detection;

    fn det(class_id: usize, x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection {
            class_id,
            label: CLASS_NAMES[class_id],
            conf: 0.9,
            x1,
            y1,
            x2,
            y2,
        }
    }

    fn obs_with_enemy() -> [f32; OBS_DIM] {
        let mut v = [0.0_f32; OBS_DIM];
        let base = OBS_SELF + OBS_FLOOR_SLOTS * OBS_SLOT_DIM;
        v[base + 2] = 0.05;
        v[base + 3] = 0.05;
        v
    }

    fn obs_with_drop() -> [f32; OBS_DIM] {
        let mut v = [0.0_f32; OBS_DIM];
        let base = OBS_SELF
            + (OBS_FLOOR_SLOTS + OBS_ENEMY_SLOTS) * OBS_SLOT_DIM;
        v[base + 2] = 0.04;
        v[base + 3] = 0.04;
        let _ = OBS_DROP_SLOTS;
        v
    }

    #[test]
    fn scores_only_yolo_visible_meso() {
        let mut f = TrainingFitness::default();
        f.record_visible_drops(&[det(11, 100.0, 200.0, 130.0, 230.0)]);
        let g1 = f.try_score_pickup(DropKind::Meso, 115.0, 215.0, 3, 10);
        assert!(g1 > 0.0);
        let g2 = f.try_score_pickup(DropKind::Meso, 500.0, 500.0, 3, 11);
        assert_eq!(g2, 0.0);
    }

    #[test]
    fn vision_shaping_only_on_fresh_obs() {
        let mut f = TrainingFitness::default();
        f.set_last_observation(&obs_with_enemy());
        let mut inp = InputFrame::default();
        inp.attack = true;
        f.try_score_input(&inp, 1);
        assert_eq!(f.vision_shaping_score, 0.0);
        f.record_mob_hit(1);
        assert!(f.vision_shaping_score >= PTS_ATTACK_ALIGN);
        // 同一观测未刷新时不再登记
        f.try_score_input(&inp, 2);
        assert!((f.vision_shaping_score - PTS_ATTACK_ALIGN).abs() < 1e-3);
    }

    #[test]
    fn attack_shaping_requires_hit_within_window() {
        let mut f = TrainingFitness::default();
        f.set_last_observation(&obs_with_enemy());
        let mut inp = InputFrame::default();
        inp.attack = true;
        f.try_score_input(&inp, 10);
        assert_eq!(f.vision_shaping_score, 0.0);
        f.record_mob_hit(10);
        assert!((f.vision_shaping_score - PTS_ATTACK_ALIGN).abs() < 1e-3);
    }

    #[test]
    fn attack_shaping_no_score_without_hit() {
        let mut f = TrainingFitness::default();
        f.set_last_observation(&obs_with_enemy());
        let mut inp = InputFrame::default();
        inp.attack = true;
        f.try_score_input(&inp, 10);
        f.try_score_input(&inp, 10 + ATTACK_HIT_SHAPING_TICKS + 1);
        assert_eq!(f.vision_shaping_score, 0.0);
        assert_eq!(f.attack_align_events, 0);
    }

    #[test]
    fn vision_shaping_cap() {
        let mut f = TrainingFitness::default();
        let mut inp = InputFrame::default();
        inp.attack = true;
        for i in 0..200u64 {
            let tick = i * 2 + 1;
            f.set_last_observation(&obs_with_enemy());
            f.try_score_input(&inp, tick);
            f.record_mob_hit(tick);
        }
        assert!(f.vision_shaping_score <= VISION_SHAPING_CAP);
    }

    #[test]
    fn memory_shaping_scaled_by_weight() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig {
            memory_weight: 0.5,
        });
        f.record_mob_kill(100);
        assert!((f.score - 0.5 * PTS_MOB_KILL).abs() < 1e-3);
        assert_eq!(f.mob_kill_events, 1);
    }

    #[test]
    fn memory_shaping_off_when_weight_zero() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        f.record_mob_hit(100);
        assert_eq!(f.score, 0.0);
        assert_eq!(f.mob_hit_events, 1);
    }

    #[test]
    fn stagnation_penalty_after_idle_window() {
        let mut f = TrainingFitness::default();
        f.tick_stagnation(100.0, 200.0, 0);
        for t in 1..=STAGNATION_TICKS {
            f.tick_stagnation(100.0, 200.0, u64::from(t));
        }
        assert_eq!(f.stagnation_penalty_events, 1);
        assert!((f.stagnation_penalty - STAGNATION_PENALTY).abs() < 1e-3);
        assert!(f.score < 0.0);
    }

    #[test]
    fn stagnation_resets_on_movement() {
        let mut f = TrainingFitness::default();
        f.tick_stagnation(100.0, 200.0, 0);
        for _ in 1..STAGNATION_TICKS {
            f.tick_stagnation(100.0, 200.0, 100);
        }
        f.tick_stagnation(200.0, 200.0, 100);
        for _ in 0..STAGNATION_TICKS {
            f.tick_stagnation(200.0, 200.0, 100);
        }
        assert_eq!(f.stagnation_penalty_events, 1);
    }

    #[test]
    fn stagnation_resets_on_pickup() {
        let mut f = TrainingFitness::default();
        f.record_visible_drops(&[det(11, 100.0, 200.0, 130.0, 230.0)]);
        f.tick_stagnation(100.0, 200.0, 0);
        for _ in 1..STAGNATION_TICKS {
            f.tick_stagnation(100.0, 200.0, 100);
        }
        assert_eq!(f.stagnation_penalty_events, 0);
        f.try_score_pickup(DropKind::Meso, 115.0, 215.0, 3, 100);
        for _ in 0..STAGNATION_TICKS - 1 {
            f.tick_stagnation(100.0, 200.0, 101);
        }
        assert_eq!(f.stagnation_penalty_events, 0);
    }

    #[test]
    fn idle_forfeit_when_stagnant_and_unproductive() {
        let mut f = TrainingFitness::default();
        f.tick_stagnation(100.0, 200.0, 0);
        let grace = u64::from(IDLE_FORFEIT_GRACE_TICKS);
        for t in 1..grace {
            f.tick_stagnation(100.0, 200.0, t);
        }
        for t in 0..STAGNATION_TICKS - 1 {
            f.tick_stagnation(100.0, 200.0, grace + u64::from(t));
        }
        let forfeit = f.tick_stagnation(100.0, 200.0, grace + u64::from(STAGNATION_TICKS - 1));
        assert!(forfeit);
        assert!(f.idle_forfeit);
    }

    #[test]
    fn idle_forfeit_blocked_during_grace() {
        let mut f = TrainingFitness::default();
        f.tick_stagnation(100.0, 200.0, 0);
        for t in 1..=STAGNATION_TICKS {
            let forfeit = f.tick_stagnation(100.0, 200.0, u64::from(t));
            assert!(!forfeit);
        }
    }

    #[test]
    fn idle_forfeit_blocked_after_mob_hit_within_exempt_window() {
        let mut f = TrainingFitness::default();
        f.tick_stagnation(100.0, 200.0, 0);
        let grace = u64::from(IDLE_FORFEIT_GRACE_TICKS);
        let hit_tick = grace + 100;
        for t in 1..hit_tick {
            f.tick_stagnation(100.0, 200.0, t);
        }
        f.record_mob_hit(hit_tick);
        for t in 0..STAGNATION_TICKS {
            let forfeit = f.tick_stagnation(100.0, 200.0, hit_tick + 1 + u64::from(t));
            assert!(!forfeit, "mob hit within 30s exempt should block forfeit");
        }
    }

    #[test]
    fn idle_forfeit_after_mob_hit_exempt_expires() {
        let mut f = TrainingFitness::default();
        f.tick_stagnation(100.0, 200.0, 0);
        let hit_tick = u64::from(IDLE_FORFEIT_GRACE_TICKS) + 10;
        for t in 1..=hit_tick {
            f.tick_stagnation(100.0, 200.0, t);
        }
        f.record_mob_hit(hit_tick);
        let exempt_end = hit_tick + u64::from(MOB_HIT_FORFEIT_EXEMPT_TICKS);
        for t in hit_tick + 1..=exempt_end {
            f.tick_stagnation(100.0, 200.0, t);
        }
        let base = exempt_end + 1;
        for t in 0..STAGNATION_TICKS - 1 {
            f.tick_stagnation(100.0, 200.0, base + u64::from(t));
        }
        let forfeit = f.tick_stagnation(100.0, 200.0, base + u64::from(STAGNATION_TICKS - 1));
        assert!(forfeit);
        assert!(f.idle_forfeit);
    }

    #[test]
    fn vertical_jump_does_not_reset_stagnation() {
        let mut f = TrainingFitness::default();
        f.tick_stagnation(100.0, 200.0, 0);
        for t in 1..STAGNATION_TICKS {
            f.tick_stagnation(100.0, 150.0, u64::from(t));
        }
        f.tick_stagnation(100.0, 150.0, u64::from(STAGNATION_TICKS));
        assert_eq!(f.stagnation_penalty_events, 1);
    }

    #[test]
    fn kill_loot_chain_bonus_after_recent_kill() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        f.record_visible_drops(&[det(11, 100.0, 200.0, 130.0, 230.0)]);
        f.record_mob_kill(100);
        let meso = f.try_score_pickup(DropKind::Meso, 115.0, 215.0, 3, 150);
        assert!(meso > 0.0);
        assert!((f.score - meso - PTS_KILL_LOOT_CHAIN).abs() < 1e-3);
    }

    #[test]
    fn kill_loot_chain_no_bonus_after_window() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        f.record_visible_drops(&[det(11, 100.0, 200.0, 130.0, 230.0)]);
        f.record_mob_kill(100);
        let tick = 100 + KILL_LOOT_CHAIN_TICKS + 1;
        let meso = f.try_score_pickup(DropKind::Meso, 115.0, 215.0, 3, tick);
        assert!(meso > 0.0);
        assert!((f.score - meso).abs() < 1e-3);
    }

    #[test]
    fn mob_hit_exempt_blocked_when_meso_visible() {
        let mut f = TrainingFitness::default();
        f.record_visible_drops(&[det(11, 100.0, 200.0, 130.0, 230.0)]);
        f.tick_stagnation(100.0, 200.0, 0);
        let grace = u64::from(IDLE_FORFEIT_GRACE_TICKS);
        let hit_tick = grace + 100;
        for t in 1..=hit_tick {
            f.tick_stagnation(100.0, 200.0, t);
        }
        f.record_mob_hit(hit_tick);
        for t in 0..STAGNATION_TICKS - 1 {
            let forfeit = f.tick_stagnation(100.0, 200.0, hit_tick + 1 + u64::from(t));
            assert!(!forfeit);
        }
        let forfeit = f.tick_stagnation(
            100.0,
            200.0,
            hit_tick + u64::from(STAGNATION_TICKS),
        );
        assert!(forfeit, "visible meso should prevent mob-hit forfeit exempt");
        assert!(f.idle_forfeit);
    }

    #[test]
    fn mob_hit_exempt_when_no_meso_visible() {
        let mut f = TrainingFitness::default();
        f.tick_stagnation(100.0, 200.0, 0);
        let grace = u64::from(IDLE_FORFEIT_GRACE_TICKS);
        let hit_tick = grace + 100;
        for t in 1..=hit_tick {
            f.tick_stagnation(100.0, 200.0, t);
        }
        f.record_mob_hit(hit_tick);
        for t in 0..STAGNATION_TICKS {
            let forfeit = f.tick_stagnation(100.0, 200.0, hit_tick + 1 + u64::from(t));
            assert!(!forfeit, "mob hit exempt should block forfeit when no visible meso");
        }
    }

    #[test]
    fn finalize_penalizes_kills_without_pickup() {
        let mut f = TrainingFitness::default();
        f.record_mob_kill(10);
        f.record_mob_kill(20);
        f.finalize_episode();
        assert!((f.episode_penalty - NO_PICKUP_AFTER_KILL_PENALTY).abs() < 1e-3);
        assert!(f.score <= -NO_PICKUP_AFTER_KILL_PENALTY + 0.5 * PTS_MOB_KILL * 2.0);
    }

    #[test]
    fn finalize_penalizes_ignored_visible_loot() {
        let mut f = TrainingFitness::default();
        for _ in 0..IGNORE_VISIBLE_LOOT_FRAMES {
            f.record_visible_drops(&[det(11, 100.0, 200.0, 130.0, 230.0)]);
        }
        f.finalize_episode();
        assert!((f.episode_penalty - IGNORE_VISIBLE_LOOT_PENALTY).abs() < 1e-3);
    }

    #[test]
    fn finalize_idempotent() {
        let mut f = TrainingFitness::default();
        f.record_mob_kill(10);
        f.record_mob_kill(20);
        f.finalize_episode();
        let s1 = f.score;
        f.finalize_episode();
        assert!((f.score - s1).abs() < 1e-3);
    }
}
