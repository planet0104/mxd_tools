//! 保命状态机：有怪时血量>50%优先砍怪，≤50%优先逃跑回血。

use super::super::input::InputFrame;
use super::super::observation::{
    obs_has_platform_enemy, obs_platform_edge, obs_slot_active, ENEMY_PLATFORM_DY, OBS_ENEMY_START,
    OBS_SLOT_DIM,
};
use super::super::types::{WINDOW_H, WINDOW_W};
use super::types::SubGoal;

/// 有怪且血量 ≤ 此比例 → 撤离回血；> 此比例 → 优先砍怪。
pub const HEAL_ENTER_RATIO: f32 = 0.50;
/// 与 HEAL_ENTER_RATIO 相同：血量超过 50% 即恢复砍怪。
pub const HEAL_EXIT_RATIO: f32 = 0.50;

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

    /// `hp_ratio`：YOLO 血条或 sim 真值，0~1。
    /// 有怪时：>50% 优先砍怪；≤50% 优先逃跑，回血到 >50% 再战。
    pub fn observe(&mut self, obs: &[f32], hp_ratio: f32, nav_y: f32) {
        let platform_mobs = obs_has_platform_enemy(obs);
        let low_hp = hp_ratio <= HEAL_ENTER_RATIO;
        let can_fight = hp_ratio > HEAL_ENTER_RATIO;

        match self.mode {
            SurvivalMode::Fight => {
                if !low_hp {
                    // 血量 >50%：优先砍怪，不撤离
                    return;
                }
                // 血量 ≤50%：优先逃跑回血
                if platform_mobs {
                    self.mode = SurvivalMode::FleeClimb;
                    self.heal_band_y = Some(nav_y);
                } else {
                    // 低血但当前台无怪：原地等回血
                    self.mode = SurvivalMode::HealWait;
                    self.heal_band_y = Some(nav_y);
                }
            }
            SurvivalMode::FleeClimb => {
                if can_fight {
                    // 已回到可战血量：立刻恢复砍怪
                    self.mode = SurvivalMode::Fight;
                    self.heal_band_y = None;
                    return;
                }
                let climbed = self
                    .heal_band_y
                    .map(|y0| nav_y < y0 - 50.0)
                    .unwrap_or(false);
                if climbed {
                    self.mode = SurvivalMode::HealWait;
                }
            }
            SurvivalMode::HealWait => {
                if can_fight {
                    self.mode = SurvivalMode::Fight;
                    self.heal_band_y = None;
                } else if platform_mobs {
                    // 仍低血且台又有怪：继续换台
                    self.mode = SurvivalMode::FleeClimb;
                    self.heal_band_y = Some(nav_y);
                }
            }
        }
    }

    pub fn suppress_chase(&self) -> bool {
        matches!(self.mode, SurvivalMode::FleeClimb | SurvivalMode::HealWait)
    }

    pub fn force_climb_escape(&self) -> bool {
        self.mode == SurvivalMode::FleeClimb
    }

    pub fn prefer_idle_heal(&self, obs: &[f32]) -> bool {
        self.mode == SurvivalMode::HealWait && !obs_has_platform_enemy(obs)
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
