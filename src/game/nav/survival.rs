//! 保命状态机：血量≤50%强制换安全台回血；回血到较高比例再战。

use super::super::input::InputFrame;
use super::super::observation::{
    obs_has_platform_enemy, obs_platform_edge, obs_slot_active, ENEMY_PLATFORM_DY, OBS_ENEMY_START,
    OBS_SLOT_DIM,
};
use super::super::types::{WINDOW_H, WINDOW_W};
use super::types::SubGoal;

/// 血量 ≤ 此比例 → 强制撤离找安全台回血。
pub const HEAL_ENTER_RATIO: f32 = 0.50;
/// 回血超过此比例才恢复砍怪（避免 51% 就下场又被刷怪压残）。
pub const HEAL_EXIT_RATIO: f32 = 0.85;
/// 相对撤离起点升高超过该像素，视为已到安全高度。
const SAFE_CLIMB_DY: f32 = 50.0;
/// 或绝对高度高于该 y（地图 y 越小越高）也视为中高台。
const SAFE_ABS_Y: f32 = 1120.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvivalMode {
    Fight,
    FleeClimb,
    HealWait,
}

#[derive(Debug)]
pub struct SurvivalFsm {
    pub mode: SurvivalMode,
    heal_band_y: Option<f32>,
}

impl Default for SurvivalFsm {
    fn default() -> Self {
        Self {
            mode: SurvivalMode::Fight,
            heal_band_y: None,
        }
    }
}

impl SurvivalFsm {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn mode(&self) -> SurvivalMode {
        self.mode
    }

    pub fn heal_band_y(&self) -> Option<f32> {
        self.heal_band_y
    }

    /// `hp_ratio`：YOLO 血条或 sim 真值，0~1。
    pub fn observe(&mut self, obs: &[f32], hp_ratio: f32, nav_y: f32) {
        let platform_mobs = obs_has_platform_enemy(obs);
        let low_hp = hp_ratio <= HEAL_ENTER_RATIO;
        let can_fight = hp_ratio >= HEAL_EXIT_RATIO;

        match self.mode {
            SurvivalMode::Fight => {
                if low_hp {
                    // 低血一律撤离：当前台暂时无怪也会刷回来，禁止原地站桩等死。
                    self.mode = SurvivalMode::FleeClimb;
                    self.heal_band_y = Some(nav_y);
                }
            }
            SurvivalMode::FleeClimb => {
                if can_fight {
                    self.mode = SurvivalMode::Fight;
                    self.heal_band_y = None;
                    return;
                }
                if platform_mobs {
                    // 仍被贴身：继续逃，刷新起点高度
                    self.heal_band_y = Some(
                        self.heal_band_y
                            .map(|y0| y0.min(nav_y))
                            .unwrap_or(nav_y),
                    );
                    return;
                }
                if self.is_safe_heal_spot(nav_y) {
                    self.mode = SurvivalMode::HealWait;
                }
            }
            SurvivalMode::HealWait => {
                if can_fight {
                    self.mode = SurvivalMode::Fight;
                    self.heal_band_y = None;
                } else if platform_mobs || !self.is_safe_heal_spot(nav_y) {
                    self.mode = SurvivalMode::FleeClimb;
                    self.heal_band_y = Some(nav_y);
                }
            }
        }
    }

    fn is_safe_heal_spot(&self, nav_y: f32) -> bool {
        let climbed = self
            .heal_band_y
            .map(|y0| nav_y + SAFE_CLIMB_DY < y0)
            .unwrap_or(false);
        climbed || nav_y < SAFE_ABS_Y
    }

    pub fn suppress_chase(&self) -> bool {
        matches!(self.mode, SurvivalMode::FleeClimb | SurvivalMode::HealWait)
    }

    /// 必须主动换台/上楼（不可 Idle）。
    pub fn force_seek_safe_platform(&self) -> bool {
        self.mode == SurvivalMode::FleeClimb
    }

    pub fn force_climb_escape(&self) -> bool {
        self.force_seek_safe_platform()
    }

    /// 仅在已到安全台且无贴身怪时允许原地回血。
    pub fn prefer_idle_heal(&self, obs: &[f32], nav_y: f32) -> bool {
        self.mode == SurvivalMode::HealWait
            && !obs_has_platform_enemy(obs)
            && self.is_safe_heal_spot(nav_y)
    }
}

pub fn preferred_fight_side(obs: &[f32]) -> Option<f32> {
    let mut left_n = 0u32;
    let mut right_n = 0u32;
    for i in 0..6 {
        let b = OBS_ENEMY_START + i * OBS_SLOT_DIM;
        if !obs_slot_active(obs, b, 1) {
            continue;
        }
        let dx = obs[b] * WINDOW_W;
        let dy = obs[b + 1] * WINDOW_H;
        if dy.abs() > ENEMY_PLATFORM_DY * WINDOW_H {
            continue;
        }
        if dx < -8.0 {
            left_n += 1;
        } else if dx > 8.0 {
            right_n += 1;
        }
    }
    if left_n > 0 && right_n > 0 {
        return None;
    }
    if left_n > 0 {
        return Some(-1.0);
    }
    if right_n > 0 {
        return Some(1.0);
    }
    None
}

pub fn side_approach_override(obs: &[f32], toward: f32) -> Option<InputFrame> {
    let side = preferred_fight_side(obs)?;
    let safe_dir = -side;
    if obs_platform_edge(obs, safe_dir) {
        return None;
    }
    if (toward - side).abs() < 0.1 && !obs_platform_edge(obs, toward) {
        let mut f = InputFrame::default();
        if safe_dir > 0.0 {
            f.right = true;
        } else {
            f.left = true;
        }
        return Some(f);
    }
    None
}

pub fn escape_climb_goal(rope_x: Option<f32>) -> Option<SubGoal> {
    rope_x.map(|rope_x| SubGoal::ClimbUp { rope_x })
}
