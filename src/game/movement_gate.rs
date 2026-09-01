//! ÃÂÃÂ¨ÃÂÃÂ§ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂ bot ÃÂÃÂ¨ÃÂÃÂ¾ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂºÃÂÃÂ§ÃÂÃÂ»ÃÂÃÂÃÂÃÂ¦ÃÂÃÂ­ÃÂÃÂ¤ÃÂÃÂ©ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ§ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂÃÂÃÂ¦ÃÂÃÂ»ÃÂÃÂ¤ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ¥ÃÂÃÂ´ÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¹ÃÂÃÂ±ÃÂÃÂ¨ÃÂÃÂµÃÂÃÂ°ÃÂÃÂ£ÃÂÃÂÃÂÃÂÃÂÃÂ§ÃÂÃÂ©ÃÂÃÂºÃÂÃÂ§ÃÂÃÂ ÃÂÃÂÃÂÃÂ£ÃÂÃÂÃÂÃÂÃÂÃÂ§ÃÂÃÂ©ÃÂÃÂºÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¡ÃÂÃÂ§ÃÂÃÂ­ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¤ÃÂÃÂ½ÃÂÃÂÃÂÃÂ£ÃÂÃÂÃÂÃÂ

use super::input::InputFrame;
use super::observation::{
    obs_climb_grab_ready, obs_enemy_in_attack_range, obs_enemy_in_attack_range_platform,
    obs_floor_ahead, obs_has_drop, obs_has_floor_signal, obs_has_same_level_enemy,
    obs_jump_allowed, obs_vertical_nav_allowed, OBS_DIM,
};

/// ÃÂÃÂ©ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ§ÃÂÃÂ¤ÃÂÃÂ¸ÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¸ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂÃÂÃÂ§ÃÂÃÂÃÂÃÂ± sim ÃÂÃÂ¦ÃÂÃÂ¯ÃÂÃÂ tick ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ©ÃÂÃÂÃÂÃÂ ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂÃÂÃÂ£ÃÂÃÂÃÂÃÂ
#[derive(Debug, Clone, Copy)]
pub struct MovementGateCtx {
    pub facing: f32,
    pub on_ground: bool,
    pub climbing: bool,
    pub can_use_potion: bool,
    pub physics_right_ok: Option<bool>,
    pub physics_left_ok: Option<bool>,
    /// sim ÃÂÃÂ¥ÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¥ÃÂÃÂ®ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ¨ÃÂÃÂ¾ÃÂÃÂ¹ÃÂÃÂ§ÃÂÃÂ¼ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂ¯ÃÂÃÂ¨ÃÂÃÂÃÂÃÂ½ÃÂÃÂ¤ÃÂÃÂ¸ÃÂÃÂÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¸ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ´ÃÂÃÂ¤ÃÂÃÂ½ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¹ÃÂÃÂ³ÃÂÃÂ¥ÃÂÃÂÃÂÃÂ°ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂÃÂÃÂ£ÃÂÃÂÃÂÃÂ
    pub physics_drop_right: Option<bool>,
    pub physics_drop_left: Option<bool>,
    pub sim_mob_in_melee: bool,
    /// Ã§ÂÂ©Ã¥Â®Â¶Ã¥ÂÂ¨Ã¦ÂÂªÃ¨ÂÂÃ¥ÂÂÃ¦ÂÂ¶Ã¥ÂÂÃ¨Â®Â¸Ã¥Â¹Â³Ã¥ÂÂ°Ã¨Â·Â³Ã¨Â·ÂÃ§Â»ÂÃ¥ÂÂÃ¯Â¼ÂÃ¤Â¸ÂÃ¤Â¾ÂÃ¨ÂµÂ YOLO Ã¦ÂÂÃ¦Â§Â½Ã¯Â¼ÂÃ£ÂÂ
    pub allow_combat_leap: bool,
    /// Ã§Â´Â§Ã©ÂÂ»Ã¥Â½ÂÃ¥ÂÂÃ¥Â±ÂÃ§ÂÂÃ§Â»Â³/Ã¦Â¢Â¯Ã¥ÂÂ¯Ã§ÂÂ¨Ã¯Â¼ÂÃ¤Â¸ÂÃ§ÂÂ¬Ã¦ÂÂÃ¤Â¸ÂÃ§ÂÂ¬Ã¯Â¼ÂÃ£ÂÂ
    pub adjacent_climb: bool,
    /// Ã¥Â½ÂÃ¥ÂÂÃ¨ÂÂÃ§ÂÂ¹Ã¦ÂÂÃ¥ÂÂ¯Ã¨Â·Â³Ã¤Â¸ÂÃ§ÂÂÃ¤Â¸ÂÃ¥Â±ÂÃ¥ÂÂ°Ã©ÂÂ¶Ã£ÂÂ
    pub allow_step_up: bool,
}

/// ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ§ÃÂÃÂÃÂÃÂ¦ÃÂÃÂµÃÂÃÂÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¯ÃÂÃÂ¹ bot ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂ¾ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂ®ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¨ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ¦ÃÂÃÂ»ÃÂÃÂ¤ÃÂÃÂ£ÃÂÃÂÃÂÃÂ
#[derive(Debug, Clone)]
pub struct MovementGate {
    last_obs: [f32; OBS_DIM],
}

impl Default for MovementGate {
    fn default() -> Self {
        Self {
            last_obs: [0.0; OBS_DIM],
        }
    }
}

impl MovementGate {
    pub fn set_last_observation(&mut self, obs: &[f32]) {
        let n = obs.len().min(OBS_DIM);
        self.last_obs[..n].copy_from_slice(&obs[..n]);
    }

    pub fn last_observation(&self) -> &[f32; OBS_DIM] {
        &self.last_obs
    }

    fn walk_allowed(&self, direction: f32, ctx: MovementGateCtx) -> bool {
        let phys = if direction > 0.0 {
            ctx.physics_right_ok
        } else {
            ctx.physics_left_ok
        };
        let drop_ok = if direction > 0.0 {
            ctx.physics_drop_right == Some(true)
        } else {
            ctx.physics_drop_left == Some(true)
        };
        match phys {
            // å¯è½ä¸ç¼ï¼åè®¸èµ°ä¸å»ï¼ä¸å¿èµ·è·³ï¼ï¼çèç©ºè¾¹ä»æ¡æ­»ã
            Some(false) => drop_ok,
            Some(true) => {
                if obs_has_floor_signal(&self.last_obs) {
                    obs_floor_ahead(&self.last_obs, direction)
                } else {
                    true
                }
            }
            None => {
                if drop_ok {
                    return true;
                }
                if !obs_has_floor_signal(&self.last_obs) {
                    return true;
                }
                obs_floor_ahead(&self.last_obs, direction)
            }
        }
    }

    pub fn filter_input(&self, input: &InputFrame, ctx: MovementGateCtx) -> InputFrame {
        let mut out = *input;

        let both_cliffs = ctx.physics_right_ok == Some(false) && ctx.physics_left_ok == Some(false);
        let either_cliff =
            ctx.physics_right_ok == Some(false) || ctx.physics_left_ok == Some(false);

        // ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ¥ÃÂÃÂ´ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ·ÃÂÃÂ³ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ§ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂ¾ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂÃÂÃÂ¨ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂ¸ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ left/rightÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂcliff_jump ÃÂÃÂ¦ÃÂÃÂ°ÃÂÃÂ¸ÃÂÃÂ¨ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¤ÃÂÃÂ¸ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ£ÃÂÃÂÃÂÃÂ
        let jump_left_intent = out.jump && out.left && !out.right;
        let jump_right_intent = out.jump && out.right && !out.left;

        if !both_cliffs {
            if out.left && !self.walk_allowed(-1.0, ctx) {
                let keep = jump_left_intent
                    && (ctx.physics_drop_left == Some(true) || ctx.physics_left_ok == Some(false));
                if !keep {
                    out.left = false;
                }
            }
            if out.right && !self.walk_allowed(1.0, ctx) {
                let keep = jump_right_intent
                    && (ctx.physics_drop_right == Some(true)
                        || ctx.physics_right_ok == Some(false));
                if !keep {
                    out.right = false;
                }
            }
        }

        if out.jump {
            if ctx.climbing {
                // ÃÂÃÂ§ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ§ÃÂÃÂ»ÃÂÃÂ³ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¶ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¨ÃÂÃÂ®ÃÂÃÂ¸ÃÂÃÂ¨ÃÂÃÂ·ÃÂÃÂ³ÃÂÃÂ§ÃÂÃÂ¦ÃÂÃÂ»
            } else if !ctx.on_ground {
                out.jump = false;
            } else if both_cliffs || either_cliff {
                let jump_left = jump_left_intent;
                let jump_right = jump_right_intent;
                let phys_drop = (jump_left && ctx.physics_drop_left == Some(true))
                    || (jump_right && ctx.physics_drop_right == Some(true));
                let cliff_jump = (jump_left && ctx.physics_left_ok == Some(false))
                    || (jump_right && ctx.physics_right_ok == Some(false));
                let combat = obs_has_same_level_enemy(&self.last_obs)
                    && (out.attack || obs_enemy_in_attack_range(&self.last_obs, ctx.facing));
                if phys_drop || (cliff_jump && !combat && !ctx.sim_mob_in_melee) {
                    // sim ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ¥ÃÂÃÂ´ÃÂÃÂ/ÃÂÃÂ¨ÃÂÃÂÃÂÃÂ½ÃÂÃÂ§ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ¨ÃÂÃÂ·ÃÂÃÂ³ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¶ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¨ÃÂÃÂ®ÃÂÃÂ¸ÃÂÃÂ¨ÃÂÃÂ¶ÃÂÃÂÃÂÃÂ§ÃÂÃÂ¼ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¢ÃÂÃÂ¥ÃÂÃÂÃÂÃÂº
                } else if combat {
                    out.jump = false;
                } else if !(ctx.adjacent_climb || ctx.allow_step_up)
                    && !obs_climb_grab_ready(&self.last_obs)
                    && !obs_jump_allowed(&self.last_obs, ctx.facing, ctx.climbing)
                {
                    out.jump = false;
                }
            } else if ctx.adjacent_climb {
                // adjacent rope/ladder: allow jump+up
            } else if ctx.allow_step_up {
                // one-step ledge: allow jump
            } else if obs_has_floor_signal(&self.last_obs)
                && obs_jump_allowed(&self.last_obs, ctx.facing, ctx.climbing)
            {
                // platform edge in move dir
            } else if (out.left || out.right)
                && !out.attack
                && !ctx.sim_mob_in_melee
                && (ctx.allow_combat_leap || obs_has_same_level_enemy(&self.last_obs))
            {
                // combat leap from behind
            } else {
                out.jump = false;
            }
        }

        // ÃÂÃÂ¨ÃÂÃÂ·ÃÂÃÂ³ÃÂÃÂ¨ÃÂÃÂ¢ÃÂÃÂ«ÃÂÃÂ¦ÃÂÃÂ»ÃÂÃÂ¤ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¸ÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¸ÃÂÃÂºÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ¥ÃÂÃÂ´ÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ§ÃÂÃÂÃÂÃÂÃÂÃÂ§ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ©ÃÂÃÂÃÂÃÂ®ÃÂÃÂ¤ÃÂÃÂ¸ÃÂÃÂÃÂÃÂ¥ÃÂÃÂºÃÂÃÂÃÂÃÂ§ÃÂÃÂ»ÃÂÃÂ§ÃÂÃÂ§ÃÂÃÂ»ÃÂÃÂ­ÃÂÃÂ¨ÃÂÃÂµÃÂÃÂ°ÃÂÃÂ¨ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ¨ÃÂÃÂÃÂÃÂÃÂÃÂ§ÃÂÃÂ©ÃÂÃÂºÃÂÃÂ£ÃÂÃÂÃÂÃÂ
        if !out.jump && !both_cliffs {
            if out.left && !self.walk_allowed(-1.0, ctx) {
                out.left = false;
            }
            if out.right && !self.walk_allowed(1.0, ctx) {
                out.right = false;
            }
        }

        if !ctx.climbing
            && (out.up || out.down)
            && !ctx.adjacent_climb
            && !obs_vertical_nav_allowed(&self.last_obs, ctx.climbing)
        {
            out.up = false;
            out.down = false;
        }

        if out.attack {
            if !ctx.sim_mob_in_melee
                && !obs_enemy_in_attack_range_platform(&self.last_obs, ctx.facing)
            {
                out.attack = false;
            }
        }
        if out.pick_up && !obs_has_drop(&self.last_obs) {
            out.pick_up = false;
        }
        if out.use_potion && !ctx.can_use_potion {
            out.use_potion = false;
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::observation::{OBS_ENEMY_START, OBS_FLOOR_START};

    fn floor_slot(values: &mut [f32], dx: f32, dy: f32, w: f32) {
        values[OBS_FLOOR_START] = dx;
        values[OBS_FLOOR_START + 1] = dy;
        values[OBS_FLOOR_START + 2] = w;
        values[OBS_FLOOR_START + 3] = 0.05;
    }

    #[test]
    fn blocks_attack_when_enemy_out_of_range() {
        let mut gate = MovementGate::default();
        let mut obs = [0.0_f32; OBS_DIM];
        floor_slot(&mut obs, 0.0, 0.02, 0.3);
        obs[OBS_ENEMY_START] = 0.4;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
        gate.set_last_observation(&obs);
        let ctx = MovementGateCtx {
            facing: 1.0,
            on_ground: true,
            climbing: false,
            can_use_potion: false,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            allow_combat_leap: false,
            adjacent_climb: false,
            allow_step_up: false,
        };
        let mut inp = InputFrame::default();
        inp.attack = true;
        let out = gate.filter_input(&inp, ctx);
        assert!(!out.attack);
    }

    #[test]
    fn blocks_jump_during_flat_combat() {
        let mut gate = MovementGate::default();
        let mut obs = [0.0_f32; OBS_DIM];
        floor_slot(&mut obs, 0.0, 0.02, 0.3);
        obs[OBS_ENEMY_START] = 0.02;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
        gate.set_last_observation(&obs);
        let ctx = MovementGateCtx {
            facing: 1.0,
            on_ground: true,
            climbing: false,
            can_use_potion: false,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            allow_combat_leap: false,
            adjacent_climb: false,
            allow_step_up: false,
        };
        let mut inp = InputFrame::default();
        inp.jump = true;
        inp.attack = true;
        let out = gate.filter_input(&inp, ctx);
        assert!(!out.jump);
        assert!(out.attack);
    }

    #[test]
    fn platform_edge_jump_allowed_in_gate() {
        let mut gate = MovementGate::default();
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_FLOOR_START + 2] = 8.0 / 1368.0;
        obs[OBS_FLOOR_START + 3] = 0.04;
        gate.set_last_observation(&obs);
        let ctx = MovementGateCtx {
            facing: 1.0,
            on_ground: true,
            climbing: false,
            can_use_potion: false,
            physics_right_ok: Some(false),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            allow_combat_leap: false,
            adjacent_climb: false,
            allow_step_up: false,
        };
        let mut inp = InputFrame::default();
        inp.jump = true;
        let out = gate.filter_input(&inp, ctx);
        assert!(out.jump);
    }

    #[test]
    fn physics_cliff_allows_jump_when_not_in_combat() {
        let mut gate = MovementGate::default();
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_FLOOR_START + 2] = 200.0 / 1368.0;
        obs[OBS_FLOOR_START + 3] = 0.04;
        gate.set_last_observation(&obs);
        let ctx = MovementGateCtx {
            facing: 1.0,
            on_ground: true,
            climbing: false,
            can_use_potion: false,
            physics_right_ok: Some(false),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            allow_combat_leap: false,
            adjacent_climb: false,
            allow_step_up: false,
        };
        let mut inp = InputFrame::default();
        inp.jump = true;
        inp.right = true;
        let out = gate.filter_input(&inp, ctx);
        assert!(out.jump);
    }

    #[test]
    fn blocks_attack_when_neither_sim_nor_vision_melee() {
        let mut gate = MovementGate::default();
        let mut obs = [0.0_f32; OBS_DIM];
        // 远处同层怪：不在可砍带
        obs[OBS_ENEMY_START] = 0.25;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
        gate.set_last_observation(&obs);
        let ctx = MovementGateCtx {
            facing: 1.0,
            on_ground: true,
            climbing: false,
            can_use_potion: false,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            allow_combat_leap: false,
            adjacent_climb: false,
            allow_step_up: false,
        };
        let mut inp = InputFrame::default();
        inp.attack = true;
        let out = gate.filter_input(&inp, ctx);
        assert!(!out.attack);
    }

    #[test]
    fn allows_attack_when_vision_melee_even_if_sim_misses() {
        let mut gate = MovementGate::default();
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_ENEMY_START] = 0.04;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
        gate.set_last_observation(&obs);
        let ctx = MovementGateCtx {
            facing: 1.0,
            on_ground: true,
            climbing: false,
            can_use_potion: false,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            allow_combat_leap: false,
            adjacent_climb: false,
            allow_step_up: false,
        };
        let mut inp = InputFrame::default();
        inp.attack = true;
        let out = gate.filter_input(&inp, ctx);
        assert!(out.attack);
    }

    #[test]
    fn physics_drop_allows_cliff_jump_without_yolo() {
        let mut gate = MovementGate::default();
        let obs = [0.0_f32; OBS_DIM];
        gate.set_last_observation(&obs);
        let ctx = MovementGateCtx {
            facing: 1.0,
            on_ground: true,
            climbing: false,
            can_use_potion: false,
            physics_right_ok: Some(false),
            physics_left_ok: Some(true),
            physics_drop_right: Some(true),
            physics_drop_left: None,
            sim_mob_in_melee: false,
            allow_combat_leap: false,
            adjacent_climb: false,
            allow_step_up: false,
        };
        let mut inp = InputFrame::default();
        inp.jump = true;
        inp.right = true;
        let out = gate.filter_input(&inp, ctx);
        assert!(out.jump);
    }
}
