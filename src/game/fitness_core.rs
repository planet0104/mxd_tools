//! NEAT 寻路计分：只奖励「到没去过的地方」。
//!
//! 网络不再负责砍怪（由 CombatFsm 接管），训练时怪物也不掉血，因此击杀/金币/挨打全部
//! 不计入分数——那是状态机的产出，混进来只会给寻路策略加噪声。分数构成：
//! 新网格 + 新高度带 + 离开出生台 + 每多到一个平台 + 局末按覆盖高度带加成。没有惩罚项：
//! 顶墙/乱跳由动作掩码在结构上兜底，任何基于视觉位移的惩罚都会把里程计误差变成「不动最优」。
//!
//! 结束条件用「探索停滞」而不是「位置停滞」：连续 EXPLORE_STALL_TICKS 没踩到新网格就认输，
//! 这同时覆盖原地不动、顶墙、两台之间来回乒乓三种无产出行为。
//!
//! 位置来自 sim 仅用于计分；网络输入与动作宏判定仍是纯 YOLO/OCR。

use std::collections::HashSet;

use super::explore_memory::ExploreHints;
use super::macro_action::MacroAction;
use super::observation::OBS_DIM;
use super::types::DropKind;
use crate::yolo::Detection;

const MESO_LABEL: &str = "金币";
const POTION_LABEL: &str = "药水";

const PTS_NEW_CELL: f32 = 1.0;
const NEW_CELL_CAP: f32 = 60.0;
const PTS_NEW_Y_BAND: f32 = 8.0;
const PTS_LEAVE_SPAWN_BAND: f32 = 55.0;
/// 离开出生点至少水平移动这么多，才算 leave_spawn（防原地换带刷分）。
const LEAVE_SPAWN_MIN_DX_PX: f32 = 96.0;
const PTS_EXTRA_PLATFORM: f32 = 40.0;
const PLATFORM_CHANGE_CAP: f32 = 200.0;
const PTS_MULTI_BAND: f32 = 35.0;
const MULTI_BAND_BONUS_CAP: f32 = 105.0;

/// 连续这么多 tick 没踩到新网格 → 认输结束。正常走路每 ~40 tick 一格，跳/爬 <90 tick。
pub const EXPLORE_STALL_TICKS: u32 = 300;

/// 与 rule_bot 探索网格一致（80×120 px）。
const X_CELL_PX: f32 = 80.0;
const ALTITUDE_BAND_PX: f32 = 120.0;

/// 前 `CURRICULUM_VERTICAL_GEN` 个虚拟世代只奖水平新格；之后才开启换层/离台大奖。
pub const CURRICULUM_VERTICAL_GEN: u32 = 8;

const PTS_HINT_ALIGN: f32 = 0.5;
const PTS_HINT_VERTICAL: f32 = 0.9;
const HINT_SCORE_CAP: f32 = 30.0;

/// 保留给 trainer CLI 的 `--fitness-shaping`；寻路训练下击杀不发生，仅在开状态机预览时体现。
const PTS_MOB_KILL: f32 = 25.0;
const DEFAULT_MEMORY_WEIGHT: f32 = 0.5;
const DEFAULT_HINT_WEIGHT: f32 = 0.35;

#[derive(Debug, Clone, Copy)]
pub struct FitnessShapingConfig {
    pub memory_weight: f32,
    /// 决策与 unvisited 提示对齐的稠密 shaping 权重；0=关闭。
    pub hint_weight: f32,
    /// false=课程第一阶段，只奖水平新格。
    pub vertical_rewards: bool,
}

impl Default for FitnessShapingConfig {
    fn default() -> Self {
        Self {
            memory_weight: DEFAULT_MEMORY_WEIGHT,
            hint_weight: DEFAULT_HINT_WEIGHT,
            vertical_rewards: true,
        }
    }
}

impl FitnessShapingConfig {
    pub fn disabled() -> Self {
        Self {
            memory_weight: 0.0,
            hint_weight: 0.0,
            vertical_rewards: true,
        }
    }

    pub fn with_curriculum(self, generation: u32) -> Self {
        Self {
            vertical_rewards: generation >= CURRICULUM_VERTICAL_GEN,
            ..self
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct VisibleLoot {
    pub meso: Vec<[f32; 4]>,
    pub potions: Vec<[f32; 4]>,
}

#[derive(Debug, Clone, Copy)]
pub struct FitnessPreviewDiag {
    pub score: f32,
    pub explore_score: f32,
    pub explore_stall_ticks: u32,
    pub cells: u32,
    pub y_bands: u32,
    pub left_spawn: bool,
    pub idle_forfeit: bool,
    pub kills: u32,
    pub meso_events: u32,
}

#[derive(Debug, Clone)]
pub struct TrainingFitness {
    pub score: f32,
    pub explore_score: f32,
    pub platform_change_score: f32,
    pub leave_spawn_bonus: f32,
    pub meso_events: u32,
    pub potion_events: u32,
    pub meso_units: u32,
    pub mob_hit_events: u32,
    pub mob_kill_events: u32,
    pub idle_forfeit: bool,
    shaping: FitnessShapingConfig,
    hint_score: f32,
    last_obs: [f32; OBS_DIM],
    last_visible: VisibleLoot,
    explore_stall_ticks: u32,
    episode_finalized: bool,
    visited_cells: HashSet<(i32, i32)>,
    visited_y_bands: HashSet<i32>,
    spawn_y_band: Option<i32>,
    spawn_x: Option<f32>,
    left_spawn_band: bool,
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
            explore_score: 0.0,
            platform_change_score: 0.0,
            leave_spawn_bonus: 0.0,
            meso_events: 0,
            potion_events: 0,
            meso_units: 0,
            mob_hit_events: 0,
            mob_kill_events: 0,
            idle_forfeit: false,
            shaping,
            hint_score: 0.0,
            last_obs: [0.0; OBS_DIM],
            last_visible: VisibleLoot::default(),
            explore_stall_ticks: 0,
            episode_finalized: false,
            visited_cells: HashSet::new(),
            visited_y_bands: HashSet::new(),
            spawn_y_band: None,
            spawn_x: None,
            left_spawn_band: false,
        }
    }

    pub fn configure_shaping(&mut self, shaping: FitnessShapingConfig) {
        self.shaping = shaping;
    }

    pub fn shaping_config(&self) -> FitnessShapingConfig {
        self.shaping
    }

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
        self.last_visible = vis;
    }

    pub fn set_last_observation(&mut self, obs: &[f32]) {
        let n = obs.len().min(OBS_DIM);
        self.last_obs[..n].copy_from_slice(&obs[..n]);
    }

    /// 按键不计分；保留接口以便 sim 调用点不变。
    pub fn try_score_input(&mut self, _input: &super::InputFrame, _episode_tick: u64) {}

    /// 稠密 shaping：宏决策与 explore hints 对齐时给小分，便于 NEAT 建立因果。
    pub fn score_nav_decision(&mut self, action: MacroAction, hints: &ExploreHints) {
        let w = self.shaping.hint_weight;
        if w <= 0.0 || self.hint_score >= HINT_SCORE_CAP {
            return;
        }
        let mut pts = 0.0_f32;
        match action {
            MacroAction::WalkLeft if hints.unvisited_left > 0.5 => pts = PTS_HINT_ALIGN,
            MacroAction::WalkRight if hints.unvisited_right > 0.5 => pts = PTS_HINT_ALIGN,
            MacroAction::Climb if hints.unvisited_band_up > 0.5 => pts = PTS_HINT_VERTICAL,
            MacroAction::JumpLeft | MacroAction::JumpRight
                if hints.unvisited_band_up > 0.5 || hints.unvisited_band_down > 0.5 =>
            {
                pts = PTS_HINT_VERTICAL * 0.85
            }
            _ => {}
        }
        if hints.stall_pressure > 0.55
            && matches!(
                action,
                MacroAction::Climb | MacroAction::JumpLeft | MacroAction::JumpRight
            )
        {
            pts = pts.max(PTS_HINT_VERTICAL * 0.7);
        }
        if pts > 0.0 {
            let applied = (pts * w).min(HINT_SCORE_CAP - self.hint_score);
            self.hint_score += applied;
            self.score += applied;
        }
    }

    pub fn record_mob_hit(&mut self, _episode_tick: u64) {
        self.mob_hit_events += 1;
    }

    pub fn record_mob_kill(&mut self, _episode_tick: u64) {
        self.mob_kill_events += 1;
        if self.shaping.memory_weight > 0.0 {
            self.score += self.shaping.memory_weight * PTS_MOB_KILL;
        }
    }

    pub fn record_player_hurt(&mut self, _damage: i32) {}

    /// 每 tick 调用；返回 true 表示探索停滞认输，本局应结束。
    pub fn tick_stagnation(
        &mut self,
        x: f32,
        y: f32,
        _episode_tick: u64,
        on_ground: bool,
        climbing: bool,
    ) -> bool {
        let new_cell = self.tick_exploration(x, y, on_ground, climbing);
        if new_cell {
            self.explore_stall_ticks = 0;
            return false;
        }
        self.explore_stall_ticks += 1;
        if self.explore_stall_ticks >= EXPLORE_STALL_TICKS {
            self.idle_forfeit = true;
            return true;
        }
        false
    }

    /// 分数单调不减，peak 与终局一致；保留接口给训练日志。
    pub fn allows_peak_update(&self) -> bool {
        true
    }

    pub fn preview_diag(&self) -> FitnessPreviewDiag {
        FitnessPreviewDiag {
            score: self.score,
            explore_score: self.explore_score,
            explore_stall_ticks: self.explore_stall_ticks,
            cells: self.visited_cells.len() as u32,
            y_bands: self.visited_y_bands.len() as u32,
            left_spawn: self.left_spawn_band,
            idle_forfeit: self.idle_forfeit,
            kills: self.mob_kill_events,
            meso_events: self.meso_events,
        }
    }

    /// 返回是否踩到了新网格。
    fn tick_exploration(&mut self, x: f32, y: f32, on_ground: bool, climbing: bool) -> bool {
        if self.spawn_x.is_none() {
            self.spawn_x = Some(x);
        }
        // 空中不记探索：否则跳跃途中的高度带也会被算成换平台。
        if !(on_ground || climbing) {
            return false;
        }
        let key = visit_key(x, y);
        let new_cell = self.visited_cells.insert(key);
        if new_cell && self.explore_score < NEW_CELL_CAP {
            let applied = PTS_NEW_CELL.min(NEW_CELL_CAP - self.explore_score);
            self.explore_score += applied;
            self.score += applied;
        }
        let y_band = key.1;
        if self.spawn_y_band.is_none() {
            self.spawn_y_band = Some(y_band);
        }
        let is_new_band = self.visited_y_bands.insert(y_band);
        if is_new_band && self.shaping.vertical_rewards {
            self.explore_score += PTS_NEW_Y_BAND;
            self.score += PTS_NEW_Y_BAND;
        }
        let left_horizontally = self
            .spawn_x
            .map(|sx| (x - sx).abs() >= LEAVE_SPAWN_MIN_DX_PX)
            .unwrap_or(false);
        let Some(spawn) = self.spawn_y_band else {
            return new_cell;
        };
        let changed_band = y_band != spawn;
        let left_by_climb = climbing && changed_band;
        let left_by_platform = on_ground && changed_band && left_horizontally;
        if self.shaping.vertical_rewards
            && (left_by_climb || left_by_platform)
            && !self.left_spawn_band
        {
            self.left_spawn_band = true;
            self.leave_spawn_bonus += PTS_LEAVE_SPAWN_BAND;
            self.score += PTS_LEAVE_SPAWN_BAND;
        }
        if self.shaping.vertical_rewards
            && self.left_spawn_band
            && is_new_band
            && changed_band
            && (climbing || left_horizontally)
            && self.platform_change_score < PLATFORM_CHANGE_CAP
        {
            let applied =
                PTS_EXTRA_PLATFORM.min(PLATFORM_CHANGE_CAP - self.platform_change_score);
            self.platform_change_score += applied;
            self.score += applied;
        }
        new_cell
    }

    pub fn finalize_episode(&mut self) {
        if self.episode_finalized {
            return;
        }
        self.episode_finalized = true;
        let bands = self.visited_y_bands.len() as u32;
        if bands >= 2 && self.left_spawn_band && self.shaping.vertical_rewards {
            let extra = (bands - 1).min(3) as f32 * PTS_MULTI_BAND;
            let applied = extra.min(MULTI_BAND_BONUS_CAP);
            self.platform_change_score += applied;
            self.score += applied;
        }
    }

    /// 拾取只计数不计分（寻路训练下掉落多来自 NPC 击杀，计分会诱导守尸）。
    pub fn try_score_pickup(
        &mut self,
        kind: DropKind,
        x: f32,
        y: f32,
        meso_amount: u32,
        _episode_tick: u64,
    ) -> f32 {
        match kind {
            DropKind::Meso => {
                if point_in_any_box(x, y, &self.last_visible.meso) {
                    self.meso_events += 1;
                    self.meso_units += meso_amount;
                }
            }
            DropKind::RedPotion => {
                if point_in_any_box(x, y, &self.last_visible.potions) {
                    self.potion_events += 1;
                }
            }
        }
        0.0
    }

    pub fn reset(&mut self) {
        let shaping = self.shaping;
        *self = Self::with_shaping(shaping);
    }
}

fn visit_key(x: f32, y: f32) -> (i32, i32) {
    (
        (x / X_CELL_PX).floor() as i32,
        (y / ALTITUDE_BAND_PX).floor() as i32,
    )
}

fn point_in_any_box(x: f32, y: f32, boxes: &[[f32; 4]]) -> bool {
    boxes
        .iter()
        .any(|b| x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::InputFrame;

    #[test]
    fn idle_genome_forfeits_after_stall_with_tiny_score() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        let mut forfeited_at = None;
        for t in 1..=EXPLORE_STALL_TICKS as u64 + 5 {
            if f.tick_stagnation(100.0, 1000.0, t, true, false) {
                forfeited_at = Some(t);
                break;
            }
        }
        assert_eq!(forfeited_at, Some(EXPLORE_STALL_TICKS as u64 + 1));
        f.finalize_episode();
        assert!(f.score < 15.0, "原地不动只有首格分 score={}", f.score);
    }

    #[test]
    fn ping_pong_between_two_cells_forfeits() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        let mut forfeited = false;
        for t in 1..=(EXPLORE_STALL_TICKS as u64 * 2) {
            let x = if (t / 20) % 2 == 0 { 100.0 } else { 200.0 };
            if f.tick_stagnation(x, 1000.0, t, true, false) {
                forfeited = true;
                break;
            }
        }
        assert!(forfeited, "两格之间来回没有新格子，必须认输");
    }

    #[test]
    fn steady_walker_keeps_going() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        for t in 1..=2000u64 {
            // 每 40 tick 前进一格
            let x = 100.0 + (t / 40) as f32 * X_CELL_PX;
            assert!(!f.tick_stagnation(x, 1000.0, t, true, false));
        }
        assert!(f.score > 40.0);
    }

    #[test]
    fn explorer_beats_walker_on_one_platform() {
        let mut flat = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        let mut multi = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        for t in 1..=400u64 {
            let x = 100.0 + (t / 10) as f32 * X_CELL_PX;
            flat.tick_stagnation(x, 1000.0, t, true, false);
            let y = if t < 200 { 1000.0 } else { 1240.0 };
            multi.tick_stagnation(x, y, t, true, false);
        }
        flat.finalize_episode();
        multi.finalize_episode();
        assert!(multi.score > flat.score + PTS_LEAVE_SPAWN_BAND);
    }

    #[test]
    fn airborne_jump_does_not_grant_leave_spawn() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        f.tick_stagnation(100.0, 1100.0, 1, true, false);
        f.tick_stagnation(100.0, 900.0, 2, false, false);
        assert!(!f.left_spawn_band);
        f.tick_stagnation(100.0 + LEAVE_SPAWN_MIN_DX_PX + 1.0, 1100.0, 3, true, false);
        assert!(!f.left_spawn_band);
        f.tick_stagnation(100.0 + LEAVE_SPAWN_MIN_DX_PX + 1.0, 900.0, 4, true, false);
        assert!(f.left_spawn_band);
    }

    #[test]
    fn curriculum_horizontal_skips_leave_spawn_bonus() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig {
            memory_weight: 0.0,
            hint_weight: 0.0,
            vertical_rewards: false,
        });
        f.tick_stagnation(100.0, 1100.0, 1, true, false);
        f.tick_stagnation(100.0 + LEAVE_SPAWN_MIN_DX_PX + 1.0, 900.0, 2, true, false);
        assert!(!f.left_spawn_band);
        assert!(f.score < PTS_LEAVE_SPAWN_BAND);
    }

    #[test]
    fn hint_shaping_rewards_unvisited_walk() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig {
            memory_weight: 0.0,
            hint_weight: 1.0,
            vertical_rewards: true,
        });
        let hints = ExploreHints {
            unvisited_right: 1.0,
            ..ExploreHints::default()
        };
        f.score_nav_decision(MacroAction::WalkRight, &hints);
        assert!(f.score > 0.0);
    }

    #[test]
    fn inputs_pickups_and_hurt_do_not_change_score() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        f.last_visible.meso.push([0.0, 0.0, 100.0, 100.0]);
        let gained = f.try_score_pickup(DropKind::Meso, 50.0, 50.0, 5, 1);
        assert_eq!(gained, 0.0);
        assert_eq!(f.meso_events, 1);
        f.record_player_hurt(10);
        f.record_mob_hit(1);
        f.try_score_input(
            &InputFrame {
                left: true,
                ..Default::default()
            },
            1,
        );
        assert_eq!(f.score, 0.0);
    }
}
