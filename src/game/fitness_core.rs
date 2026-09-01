//! NEAT 训练计分：YOLO 可见拾取（主分）+ 视觉动作 shaping。

use std::collections::{HashSet, VecDeque};

use super::observation::{
    obs_has_drop, obs_has_same_level_enemy, obs_nearest_same_level_enemy_px, OBS_DIM,
};
use super::types::{DropKind, WINDOW_H, WINDOW_W};
use crate::yolo::Detection;

const MESO_LABEL: &str = "金币";
const POTION_LABEL: &str = "药水";

const PTS_MESO_UNIT: f32 = 8.0;
const PTS_POTION_PICKUP: f32 = 18.0;
const PTS_ATTACK_ALIGN: f32 = 0.5;
const PTS_PICKUP_ALIGN: f32 = 2.5;
const VISION_SHAPING_CAP: f32 = 60.0;
const PTS_MOB_HIT: f32 = 5.0;
const PTS_MOB_KILL: f32 = 25.0;
const PTS_KILL_LOOT_CHAIN: f32 = 18.0;
const KILL_LOOT_CHAIN_TICKS: u64 = 180;
const NO_PICKUP_AFTER_KILL_PENALTY: f32 = 25.0;
const NO_PICKUP_KILL_THRESHOLD: u32 = 2;
const IGNORE_VISIBLE_LOOT_PENALTY: f32 = 15.0;
const IGNORE_VISIBLE_LOOT_FRAMES: u32 = 30;
const ATTACK_HIT_SHAPING_TICKS: u64 = 30;
pub const STAGNATION_TICKS: u32 = 300;
const STAGNATION_MOVE_THRESHOLD: f32 = 48.0;
pub const MOB_HIT_FORFEIT_EXEMPT_TICKS: u32 = 900;
const STAGNATION_PENALTY: f32 = 15.0;
const STAGNATION_PENALTY_CAP: f32 = 90.0;
pub const IDLE_FORFEIT_GRACE_TICKS: u32 = 600;

const PTS_NEW_CELL: f32 = 2.0;
const PTS_NEW_Y_BAND: f32 = 5.0;
const EXPLORE_REWARD_CAP: f32 = 80.0;
const PTS_MOVE_TOWARD_ENEMY: f32 = 0.3;
const MOVE_TOWARD_ENEMY_CAP: f32 = 40.0;
const PENALTY_EMPTY_ATTACK: f32 = 0.25;
const EMPTY_ATTACK_PENALTY_CAP: f32 = 30.0;
const PING_PONG_PENALTY: f32 = 8.0;
const PING_PONG_PENALTY_CAP: f32 = 64.0;
const PING_PONG_WINDOW: usize = 12;
const PING_PONG_MIN_ALTERNATIONS: usize = 5;
const PING_PONG_COOLDOWN_TICKS: u64 = 120;
const MOVE_TOWARD_ENEMY_DX_PX: f32 = 8.0;

/// 与 rule_bot 探索网格一致（80×120 px）。
const X_CELL_PX: f32 = 80.0;
const ALTITUDE_BAND_PX: f32 = 120.0;

/// 默认 sim 记忆 shaping 权重（击杀/命中辅助信号，主分仍为 YOLO 可见拾取）。
const DEFAULT_MEMORY_WEIGHT: f32 = 0.35;

#[derive(Debug, Clone, Copy)]
pub struct FitnessShapingConfig {
    pub memory_weight: f32,
}

impl Default for FitnessShapingConfig {
    fn default() -> Self {
        Self {
            memory_weight: DEFAULT_MEMORY_WEIGHT,
        }
    }
}

impl FitnessShapingConfig {
    pub fn disabled() -> Self {
        Self { memory_weight: 0.0 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct VisibleLoot {
    pub meso: Vec<[f32; 4]>,
    pub potions: Vec<[f32; 4]>,
}

#[derive(Debug, Clone)]
pub struct TrainingFitness {
    pub score: f32,
    pub pickup_score: f32,
    pub vision_shaping_score: f32,
    pub memory_shaping_score: f32,
    pub explore_score: f32,
    pub meso_events: u32,
    pub potion_events: u32,
    pub meso_units: u32,
    pub attack_align_events: u32,
    pub pickup_align_events: u32,
    pub mob_hit_events: u32,
    pub mob_kill_events: u32,
    pub stagnation_penalty: f32,
    pub stagnation_penalty_events: u32,
    pub empty_attack_penalty: f32,
    pub ping_pong_penalty: f32,
    pub episode_penalty: f32,
    pub idle_forfeit: bool,
    shaping: FitnessShapingConfig,
    last_obs: [f32; OBS_DIM],
    last_visible: VisibleLoot,
    observation_fresh: bool,
    stagnation_anchor_x: f32,
    stagnation_anchor_y: f32,
    stagnation_ticks: u32,
    stagnation_initialized: bool,
    last_x: f32,
    last_y: f32,
    last_pickup_tick: u64,
    last_mob_hit_tick: u64,
    last_mob_kill_tick: u64,
    pending_attack_align_tick: Option<u64>,
    visible_meso_frames: u32,
    episode_finalized: bool,
    visited_cells: HashSet<(i32, i32)>,
    visited_y_bands: HashSet<i32>,
    recent_cells: VecDeque<(i32, i32)>,
    move_toward_enemy_score: f32,
    last_ping_pong_penalty_tick: u64,
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
            explore_score: 0.0,
            meso_events: 0,
            potion_events: 0,
            meso_units: 0,
            attack_align_events: 0,
            pickup_align_events: 0,
            mob_hit_events: 0,
            mob_kill_events: 0,
            stagnation_penalty: 0.0,
            stagnation_penalty_events: 0,
            empty_attack_penalty: 0.0,
            ping_pong_penalty: 0.0,
            episode_penalty: 0.0,
            idle_forfeit: false,
            shaping,
            last_obs: [0.0; OBS_DIM],
            last_visible: VisibleLoot::default(),
            observation_fresh: false,
            stagnation_anchor_x: 0.0,
            stagnation_anchor_y: 0.0,
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
            visited_cells: HashSet::new(),
            visited_y_bands: HashSet::new(),
            recent_cells: VecDeque::new(),
            move_toward_enemy_score: 0.0,
            last_ping_pong_penalty_tick: 0,
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

    pub fn try_score_input(&mut self, input: &super::InputFrame, episode_tick: u64) {
        self.expire_pending_attack_align(episode_tick);
        if !self.observation_fresh || self.vision_shaping_score >= VISION_SHAPING_CAP {
            return;
        }
        self.observation_fresh = false;
        if input.attack {
            if obs_has_same_level_enemy(&self.last_obs) {
                self.pending_attack_align_tick = Some(episode_tick);
            } else if self.empty_attack_penalty < EMPTY_ATTACK_PENALTY_CAP {
                let applied =
                    PENALTY_EMPTY_ATTACK.min(EMPTY_ATTACK_PENALTY_CAP - self.empty_attack_penalty);
                self.empty_attack_penalty += applied;
                self.score -= applied;
            }
        }
        if input.pick_up && obs_has_drop(&self.last_obs) {
            self.vision_shaping_score += PTS_PICKUP_ALIGN;
            self.score += PTS_PICKUP_ALIGN;
            self.pickup_align_events += 1;
        }
        self.try_score_move_toward_enemy(input);
    }

    fn try_score_move_toward_enemy(&mut self, input: &super::InputFrame) {
        if self.move_toward_enemy_score >= MOVE_TOWARD_ENEMY_CAP {
            return;
        }
        if !obs_has_same_level_enemy(&self.last_obs) {
            return;
        }
        let Some((dx_px, _)) =
            obs_nearest_same_level_enemy_px(&self.last_obs, WINDOW_W, WINDOW_H)
        else {
            return;
        };
        let toward = (dx_px > MOVE_TOWARD_ENEMY_DX_PX && input.right && !input.left)
            || (dx_px < -MOVE_TOWARD_ENEMY_DX_PX && input.left && !input.right);
        if toward {
            self.move_toward_enemy_score += PTS_MOVE_TOWARD_ENEMY;
            self.score += PTS_MOVE_TOWARD_ENEMY;
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

    pub fn tick_stagnation(&mut self, x: f32, y: f32, episode_tick: u64) -> bool {
        self.last_x = x;
        self.last_y = y;
        self.tick_exploration(x, y, episode_tick);
        if !self.stagnation_initialized {
            self.reset_stagnation_anchor(x, y);
            self.stagnation_initialized = true;
            return false;
        }
        let dx = x - self.stagnation_anchor_x;
        let dy = y - self.stagnation_anchor_y;
        if dx * dx + dy * dy > STAGNATION_MOVE_THRESHOLD * STAGNATION_MOVE_THRESHOLD {
            self.reset_stagnation_anchor(x, y);
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

    fn tick_exploration(&mut self, x: f32, y: f32, episode_tick: u64) {
        let key = visit_key(x, y);
        if self.visited_cells.insert(key) && self.explore_score < EXPLORE_REWARD_CAP {
            let applied = PTS_NEW_CELL.min(EXPLORE_REWARD_CAP - self.explore_score);
            self.explore_score += applied;
            self.score += applied;
        }
        let y_band = key.1;
        if self.visited_y_bands.insert(y_band) && self.explore_score < EXPLORE_REWARD_CAP {
            let applied = PTS_NEW_Y_BAND.min(EXPLORE_REWARD_CAP - self.explore_score);
            self.explore_score += applied;
            self.score += applied;
        }
        self.recent_cells.push_back(key);
        while self.recent_cells.len() > PING_PONG_WINDOW {
            self.recent_cells.pop_front();
        }
        if detect_ping_pong(&self.recent_cells) {
            let cooldown_ok = self.last_ping_pong_penalty_tick == 0
                || episode_tick.saturating_sub(self.last_ping_pong_penalty_tick)
                    >= PING_PONG_COOLDOWN_TICKS;
            if cooldown_ok && self.ping_pong_penalty < PING_PONG_PENALTY_CAP {
                let applied = PING_PONG_PENALTY.min(PING_PONG_PENALTY_CAP - self.ping_pong_penalty);
                self.ping_pong_penalty += applied;
                self.score -= applied;
                self.last_ping_pong_penalty_tick = episode_tick;
            }
        }
    }

    fn blocks_idle_forfeit(&self, episode_tick: u64) -> bool {
        let window = u64::from(STAGNATION_TICKS);
        if self.last_pickup_tick > 0 && episode_tick.saturating_sub(self.last_pickup_tick) < window {
            return true;
        }
        if self.last_mob_kill_tick > 0 && episode_tick.saturating_sub(self.last_mob_kill_tick) < window {
            return true;
        }
        if self.last_visible.meso.is_empty()
            && self.last_mob_hit_tick > 0
            && episode_tick.saturating_sub(self.last_mob_hit_tick) < u64::from(MOB_HIT_FORFEIT_EXEMPT_TICKS)
        {
            return true;
        }
        false
    }

    pub fn finalize_episode(&mut self) {
        if self.episode_finalized {
            return;
        }
        self.episode_finalized = true;
        if self.meso_events == 0 && self.mob_kill_events >= NO_PICKUP_KILL_THRESHOLD {
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
            self.reset_stagnation_anchor(self.last_x, self.last_y);
        }
    }

    fn reset_stagnation_anchor(&mut self, x: f32, y: f32) {
        self.stagnation_anchor_x = x;
        self.stagnation_anchor_y = y;
        self.stagnation_ticks = 0;
    }

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
                self.meso_events += 1;
                self.meso_units += meso_amount;
                meso_amount as f32 * PTS_MESO_UNIT
            }
            DropKind::RedPotion => {
                if !point_in_any_box(x, y, &self.last_visible.potions) {
                    return 0.0;
                }
                self.potion_events += 1;
                PTS_POTION_PICKUP
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

fn visit_key(x: f32, y: f32) -> (i32, i32) {
    (
        (x / X_CELL_PX).floor() as i32,
        (y / ALTITUDE_BAND_PX).floor() as i32,
    )
}

fn detect_ping_pong(cells: &VecDeque<(i32, i32)>) -> bool {
    if cells.len() < PING_PONG_MIN_ALTERNATIONS + 1 {
        return false;
    }
    let slice: Vec<(i32, i32)> = cells.iter().copied().collect();
    let n = slice.len();
    let a = slice[n - 2];
    let b = slice[n - 1];
    if a == b {
        return false;
    }
    let start = n.saturating_sub(PING_PONG_MIN_ALTERNATIONS + 1);
    let mut expect_a = ((n - 1 - start) % 2) == 1;
    for i in start..n {
        let expected = if expect_a { a } else { b };
        if slice[i] != expected {
            return false;
        }
        expect_a = !expect_a;
    }
    true
}

fn point_in_any_box(x: f32, y: f32, boxes: &[[f32; 4]]) -> bool {
    boxes.iter().any(|b| x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::observation::{OBS_ENEMY_START, OBS_FLOOR_START};

    fn floor_slot(values: &mut [f32], dx: f32, dy: f32) {
        values[OBS_FLOOR_START] = dx;
        values[OBS_FLOOR_START + 1] = dy;
        values[OBS_FLOOR_START + 2] = 0.3;
        values[OBS_FLOOR_START + 3] = 0.05;
    }

    fn same_level_enemy_slot(values: &mut [f32], dx: f32) {
        values[OBS_ENEMY_START] = dx;
        values[OBS_ENEMY_START + 1] = 0.0;
        values[OBS_ENEMY_START + 2] = 0.05;
        values[OBS_ENEMY_START + 3] = 0.05;
    }

    #[test]
    fn meso_pickup_scores_more_than_potion_per_unit() {
        let mut f = TrainingFitness::default();
        f.last_visible.meso.push([0.0, 0.0, 100.0, 100.0]);
        let gained = f.try_score_pickup(DropKind::Meso, 50.0, 50.0, 5, 1);
        assert_eq!(gained, 40.0);
        assert!(gained > PTS_POTION_PICKUP * 0.5);
    }

    #[test]
    fn empty_attack_without_same_level_enemy_penalized() {
        let mut f = TrainingFitness::default();
        let mut obs = [0.0_f32; OBS_DIM];
        floor_slot(&mut obs, 0.0, 0.02);
        f.set_last_observation(&obs);
        f.try_score_input(
            &super::super::InputFrame {
                attack: true,
                ..Default::default()
            },
            1,
        );
        assert!(f.empty_attack_penalty > 0.0);
        assert!(f.score < 0.0);
    }

    #[test]
    fn attack_with_same_level_enemy_no_penalty() {
        let mut f = TrainingFitness::default();
        let mut obs = [0.0_f32; OBS_DIM];
        floor_slot(&mut obs, 0.0, 0.02);
        same_level_enemy_slot(&mut obs, 0.1);
        f.set_last_observation(&obs);
        f.try_score_input(
            &super::super::InputFrame {
                attack: true,
                ..Default::default()
            },
            1,
        );
        assert_eq!(f.empty_attack_penalty, 0.0);
        assert_eq!(f.pending_attack_align_tick, Some(1));
    }

    #[test]
    fn move_toward_enemy_rewards_aligned_direction() {
        let mut f = TrainingFitness::default();
        let mut obs = [0.0_f32; OBS_DIM];
        floor_slot(&mut obs, 0.0, 0.02);
        same_level_enemy_slot(&mut obs, 0.15);
        f.set_last_observation(&obs);
        let score0 = f.score;
        f.try_score_input(
            &super::super::InputFrame {
                right: true,
                ..Default::default()
            },
            1,
        );
        assert!(f.score > score0);
    }

    #[test]
    fn exploration_rewards_new_cell_and_y_band() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        f.tick_stagnation(100.0, 1225.0, 1);
        assert!(f.explore_score >= PTS_NEW_CELL + PTS_NEW_Y_BAND);
    }

    #[test]
    fn stagnation_uses_2d_distance() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        f.tick_stagnation(100.0, 1000.0, 1);
        f.tick_stagnation(100.0, 1000.0, 2);
        for t in 3u64..=STAGNATION_TICKS as u64 + 2 {
            f.tick_stagnation(100.0, 1000.0, t);
        }
        assert!(f.stagnation_penalty > 0.0 || f.stagnation_ticks > 0);
        f.tick_stagnation(200.0, 1000.0, STAGNATION_TICKS as u64 + 3);
        assert_eq!(f.stagnation_ticks, 0);
    }

    #[test]
    fn ping_pong_detected_and_penalized() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        for (i, x) in [100.0_f32, 180.0, 100.0, 180.0, 100.0, 180.0].into_iter().enumerate() {
            f.tick_stagnation(x, 800.0, (i + 1) as u64);
        }
        assert!(f.ping_pong_penalty > 0.0);
    }

    #[test]
    fn detect_ping_pong_alternating_cells() {
        let mut cells = VecDeque::new();
        for key in [(1, 10), (2, 10), (1, 10), (2, 10), (1, 10), (2, 10)] {
            cells.push_back(key);
        }
        assert!(detect_ping_pong(&cells));
    }
}
