//! 保命状态机：防夹击、低血爬梯回血、侧面接战。

use std::time::Instant;

use super::super::input::InputFrame;
use super::super::observation::{
    obs_assess_enemy_contact, obs_has_platform_enemy, obs_platform_edge, obs_slot_active,
    ENEMY_PLATFORM_DY, OBS_ENEMY_START, OBS_SLOT_DIM,
};
use super::super::types::{WINDOW_H, WINDOW_W};
use super::types::SubGoal;

pub const HEAL_ENTER_RATIO: f32 = 0.50;
pub const HEAL_EXIT_RATIO: f32 = 1.0;
pub const COMBAT_RETREAT_SECS: f32 = 14.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvivalMode {
    Fight,
    FleeClimb,
    HealWait,
}

#[derive(Debug)]
pub struct SurvivalFsm {
    pub mode: SurvivalMode,
    combat_started: Option<Instant>,
    heal_band_y: Option<f32>,
}

impl Default for SurvivalFsm {
    fn default() -> Self {
        Self {
            mode: SurvivalMode::Fight,
            combat_started: None,
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

    pub fn observe(&mut self, obs: &[f32], hp_ratio: f32, nav_y: f32) {
        let contact = obs_assess_enemy_contact(obs);
        let sandwiched = contact.left > 0 && contact.right > 0;
        let platform_mobs = obs_has_platform_enemy(obs);

        match self.mode {
            SurvivalMode::Fight => {
                if platform_mobs || contact.total > 0 {
                    if self.combat_started.is_none() {
                        self.combat_started = Some(Instant::now());
                    }
                } else {
                    self.combat_started = None;
                }
                let combat_secs = self
                    .combat_started
                    .map(|t| t.elapsed().as_secs_f32())
                    .unwrap_or(0.0);
                if sandwiched || hp_ratio <= HEAL_ENTER_RATIO {
                    self.mode = SurvivalMode::FleeClimb;
                    self.heal_band_y = Some(nav_y);
                    self.combat_started = None;
                } else if combat_secs >= COMBAT_RETREAT_SECS && platform_mobs {
                    self.mode = SurvivalMode::FleeClimb;
                    self.heal_band_y = Some(nav_y);
                    self.combat_started = None;
                }
            }
            SurvivalMode::FleeClimb => {
                let climbed = self
                    .heal_band_y
                    .map(|y0| nav_y < y0 - 50.0)
                    .unwrap_or(false);
                if climbed && !sandwiched {
                    self.mode = SurvivalMode::HealWait;
                }
                if hp_ratio >= HEAL_EXIT_RATIO && !sandwiched && !platform_mobs {
                    self.mode = SurvivalMode::Fight;
                    self.heal_band_y = None;
                }
            }
            SurvivalMode::HealWait => {
                if sandwiched {
                    self.mode = SurvivalMode::FleeClimb;
                    self.heal_band_y = Some(nav_y);
                } else if hp_ratio >= HEAL_EXIT_RATIO {
                    self.mode = SurvivalMode::Fight;
                    self.heal_band_y = None;
                    self.combat_started = None;
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
