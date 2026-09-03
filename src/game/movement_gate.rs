//! 脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗搂脙聝脗聜脙聜脗聞脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聢脙聝脗聜脙聜脗聶 bot 脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗戮脙聝脗聜脙聜脗聯脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聡脙聝脗聜脙聜脗潞脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗禄脙聝脗聜脙聜脗聫脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗颅脙聝脗聜脙聜脗陇脙聝脗聝脙聜脗漏脙聝脗聜脙聜脗聴脙聝脗聜脙聜脗篓脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聨脙聝脗聜脙聜脗搂脙聝脗聝脙聜脗炉脙聝脗聜脙聜脗录脙聝脗聜脙聜脗聦脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗禄脙聝脗聜脙聜脗陇脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聨脙聝脗聜脙聜脗聣脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聜脙聝脗聜脙聜脗卢脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗麓脙聝脗聜脙聜脗聳脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗鹿脙聝脗聜脙聜脗卤脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗碌脙聝脗聜脙聜脗掳脙聝脗聝脙聜脗拢脙聝脗聜脙聜脗聙脙聝脗聜脙聜脗聛脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗漏脙聝脗聜脙聜脗潞脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗聽脙聝脗聜脙聜脗聧脙聝脗聝脙聜脗拢脙聝脗聜脙聜脗聙脙聝脗聜脙聜脗聛脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗漏脙聝脗聜脙聜脗潞脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聧脙聝脗聜脙聜脗隆脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗颅脙聝脗聜脙聜脗聣脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聴脙聝脗聜脙聜脗聽脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聲脙聝脗聜脙聜脗聢脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聤脙聝脗聜脙聜脗篓脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗陆脙聝脗聜脙聜脗聹脙聝脗聝脙聜脗拢脙聝脗聜脙聜脗聙脙聝脗聜脙聜脗聜

use super::input::InputFrame;
use super::observation::{
    obs_climb_grab_ready, obs_enemy_in_attack_range, obs_enemy_in_attack_range_platform,
    obs_floor_ahead, obs_has_drop, obs_has_floor_signal, obs_has_same_level_enemy,
    obs_jump_allowed, obs_same_level_gap_ahead, obs_vertical_nav_allowed, OBS_DIM,
};
use super::types::{WINDOW_H, WINDOW_W};

/// 脙聝脗聝脙聜脗漏脙聝脗聜脙聜脗聴脙聝脗聜脙聜脗篓脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聨脙聝脗聜脙聜脗搂脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗赂脙聝脗聜脙聜脗聤脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗赂脙聝脗聜脙聜脗聥脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聳脙聝脗聜脙聜脗聡脙聝脗聝脙聜脗炉脙聝脗聜脙聜脗录脙聝脗聜脙聜脗聢脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗聰脙聝脗聜脙聜脗卤 sim 脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗炉脙聝脗聜脙聜脗聫 tick 脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聻脙聝脗聜脙聜脗聞脙聝脗聝脙聜脗漏脙聝脗聜脙聜脗聙脙聝脗聜脙聜脗聽脙聝脗聝脙聜脗炉脙聝脗聜脙聜脗录脙聝脗聜脙聜脗聣脙聝脗聝脙聜脗拢脙聝脗聜脙聜脗聙脙聝脗聜脙聜脗聜
#[derive(Debug, Clone, Copy)]
pub struct MovementGateCtx {
    pub facing: f32,
    pub on_ground: bool,
    pub climbing: bool,
    pub can_use_potion: bool,
    pub physics_right_ok: Option<bool>,
    pub physics_left_ok: Option<bool>,
    /// sim 脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聢脙聝脗聜脙聜脗陇脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗庐脙聝脗聜脙聜脗職脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聣脙聝脗聜脙聜脗聧脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聳脙聝脗聜脙聜脗鹿脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗戮脙聝脗聜脙聜脗鹿脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗录脙聝脗聜脙聜脗聵脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聫脙聝脗聜脙聜脗炉脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗聬脙聝脗聜脙聜脗陆脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗赂脙聝脗聜脙聜脗聥脙聝脗聝脙聜脗炉脙聝脗聜脙聜脗录脙聝脗聜脙聜脗聢脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗赂脙聝脗聜脙聜脗聥脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聳脙聝脗聜脙聜脗鹿脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聹脙聝脗聜脙聜脗聣脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聸脙聝脗聜脙聜脗麓脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗陆脙聝脗聜脙聜脗聨脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗鹿脙聝脗聜脙聜脗鲁脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聫脙聝脗聜脙聜脗掳脙聝脗聝脙聜脗炉脙聝脗聜脙聜脗录脙聝脗聜脙聜脗聣脙聝脗聝脙聜脗拢脙聝脗聜脙聜脗聙脙聝脗聜脙聜脗聜
    pub physics_drop_right: Option<bool>,
    pub physics_drop_left: Option<bool>,
    pub sim_mob_in_melee: bool,
    /// 脙搂脗聨脗漏脙楼脗庐脗露脙楼脗聹脗篓脙娄脗聙脗陋脙篓脗聝脗聦脙楼脗聬脗聨脙娄脗聴脗露脙楼脗聟脗聛脙篓脗庐脗赂脙楼脗鹿脗鲁脙楼脗聹脗掳脙篓脗路脗鲁脙篓脗路脗聝脙搂脗禄脗聲脙楼脗聣脗聧脙炉脗录脗聢脙陇脗赂脗聧脙陇脗戮脗聺脙篓脗碌脗聳 YOLO 脙娄脗聲脗聦脙娄脗搂脗陆脙炉脗录脗聣脙拢脗聙脗聜
    pub allow_combat_leap: bool,
    /// 脙搂脗麓脗搂脙漏脗聜脗禄脙楼脗陆脗聯脙楼脗聣脗聧脙楼脗卤脗聜脙搂脗職脗聞脙搂脗禄脗鲁/脙娄脗垄脗炉脙楼脗聫脗炉脙搂脗聰脗篓脙炉脗录脗聢脙陇脗赂脗聤脙搂脗聢脗卢脙娄脗聢脗聳脙陇脗赂脗聥脙搂脗聢脗卢脙炉脗录脗聣脙拢脗聙脗聜
    pub adjacent_climb: bool,
    /// 褰撳墠鑴氱偣鏈夊彲璺充笂鐨勪竴灞傚彴闃躲€?
    pub allow_step_up: bool,
    /// 鍥捐鍒掕蛋璺紙GoTo/Patrol锛夛細鐗╃悊鍙蛋鏃跺嬁琚?YOLO 纰庡湴鏉挎尅姝汇€?
    pub allow_nav_walk: bool,
}

/// 脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聦脙聝脗聜脙聜脗聛脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聹脙聝脗聜脙聜脗聣脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聹脙聝脗聜脙聜脗聙脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗驴脙聝脗聜脙聜脗聭脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗搂脙聝脗聜脙聜脗聜脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗碌脙聝脗聜脙聜脗聥脙聝脗聝脙聜脗炉脙聝脗聜脙聜脗录脙聝脗聜脙聜脗聦脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗炉脙聝脗聜脙聜脗鹿 bot 脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聞脙聝脗聜脙聜脗聫脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聸脙聝脗聜脙聜脗戮脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聛脙聝脗聜脙聜脗職脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗庐脙聝脗聜脙聜脗聣脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聟脙聝脗聜脙聜脗篓脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗驴脙聝脗聜脙聜脗聡脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗禄脙聝脗聜脙聜脗陇脙聝脗聝脙聜脗拢脙聝脗聜脙聜脗聙脙聝脗聜脙聜脗聜
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
        // 瀵艰埅璧拌矾锛氱墿鐞嗙‘璁ゅ彲璧板垯鏀捐锛圷OLO 鍦版澘甯哥鎴愮煭娈碉紝obs_floor_ahead 浼氳鎸★級銆?
        if ctx.allow_nav_walk {
            match phys {
                Some(true) => return true,
                Some(false) => return drop_ok,
                None => {}
            }
        }
        match phys {
            // 鍙惤涓嬬紭锛氬厑璁歌蛋涓嬪幓锛堜笉蹇呰捣璺筹級锛涚湡铏氱┖杈逛粛鎸℃銆?
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

        // 脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聜脙聝脗聜脙聜脗卢脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗麓脙聝脗聜脙聜脗聳脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗路脙聝脗聜脙聜脗鲁脙聝脗聝脙聜脗漏脙聝脗聜脙聜脗聹脙聝脗聜脙聜脗聙脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗驴脙聝脗聜脙聜脗聺脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗聲脙聝脗聜脙聜脗聶脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聳脙聝脗聜脙聜脗鹿脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聬脙聝脗聜脙聜脗聭脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聞脙聝脗聜脙聜脗聫脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聸脙聝脗聜脙聜脗戮脙聝脗聝脙聜脗炉脙聝脗聜脙聜脗录脙聝脗聜脙聜脗聸脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗聥脙聝脗聜脙聜脗楼脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聟脙聝脗聜脙聜脗聢脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗赂脙聝脗聜脙聜脗聟脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聨脙聝脗聜脙聜脗聣 left/right脙聝脗聝脙聜脗炉脙聝脗聜脙聜脗录脙聝脗聜脙聜脗聦cliff_jump 脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗掳脙聝脗聜脙聜脗赂脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗驴脙聝脗聜脙聜脗聹脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聢脙聝脗聜脙聜脗陇脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗赂脙聝脗聜脙聜脗聧脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗驴脙聝脗聜脙聜脗聡脙聝脗聝脙聜脗拢脙聝脗聜脙聜脗聙脙聝脗聜脙聜脗聜
        let jump_left_intent = out.jump && out.left && !out.right;
        let jump_right_intent = out.jump && out.right && !out.left;

        if !both_cliffs {
            if out.left && !self.walk_allowed(-1.0, ctx) {
                let keep = jump_left_intent
                    && (ctx.allow_step_up
                        || ctx.physics_drop_left == Some(true)
                        || ctx.physics_left_ok == Some(false)
                        || gap_hop_intent(&self.last_obs, true, false));
                if !keep {
                    out.left = false;
                }
            }
            if out.right && !self.walk_allowed(1.0, ctx) {
                let keep = jump_right_intent
                    && (ctx.allow_step_up
                        || ctx.physics_drop_right == Some(true)
                        || ctx.physics_right_ok == Some(false)
                        || gap_hop_intent(&self.last_obs, false, true));
                if !keep {
                    out.right = false;
                }
            }
        }

        if out.jump {
            if ctx.climbing {
                // 挂绳时 jump 会脱离绳子，禁止。
                out.jump = false;
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
                    // sim 脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聜脙聝脗聜脙聜脗卢脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗麓脙聝脗聜脙聜脗聳/脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗聬脙聝脗聜脙聜脗陆脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗聜脙聝脗聜脙聜脗鹿脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗路脙聝脗聜脙聜脗鲁脙聝脗聝脙聜脗炉脙聝脗聜脙聜脗录脙聝脗聜脙聜脗職脙聝脗聝脙聜脗漏脙聝脗聜脙聜脗聺脙聝脗聜脙聜脗聻脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聢脙聝脗聜脙聜脗聵脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聳脙聝脗聜脙聜脗聴脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聴脙聝脗聜脙聜脗露脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聟脙聝脗聜脙聜脗聛脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗庐脙聝脗聜脙聜脗赂脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗露脙聝脗聜脙聜脗聤脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗录脙聝脗聜脙聜脗聺脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聧脙聝脗聜脙聜脗垄脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聦脙聝脗聜脙聜脗潞
                } else if combat {
                    out.jump = false;
                } else if !(ctx.adjacent_climb || ctx.allow_step_up)
                    && !obs_climb_grab_ready(&self.last_obs)
                    && !obs_jump_allowed(&self.last_obs, ctx.facing, ctx.climbing)
                    && !gap_hop_intent(&self.last_obs, jump_left, jump_right)
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
                // platform edge / same-level gap in facing
            } else if gap_hop_intent(&self.last_obs, jump_left_intent, jump_right_intent) {
                // 同层缝隙 hop（按按键方向，不依赖 facing 滞后）
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

        // 脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗路脙聝脗聜脙聜脗鲁脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗垄脙聝脗聜脙聜脗芦脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗禄脙聝脗聜脙聜脗陇脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聨脙聝脗聜脙聜脗聣脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聬脙聝脗聜脙聜脗聨脙聝脗聝脙聜脗炉脙聝脗聜脙聜脗录脙聝脗聜脙聜脗聦脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗赂脙聝脗聜脙聜脗聧脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗赂脙聝脗聜脙聜脗潞脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聜脙聝脗聜脙聜脗卢脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗麓脙聝脗聜脙聜脗聳脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗驴脙聝脗聜脙聜脗聺脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗聲脙聝脗聜脙聜脗聶脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗職脙聝脗聜脙聜脗聞脙聝脗聝脙聜脗娄脙聝脗聜脙聜脗聳脙聝脗聜脙聜脗鹿脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗聬脙聝脗聜脙聜脗聭脙聝脗聝脙聜脗漏脙聝脗聜脙聜脗聰脙聝脗聜脙聜脗庐脙聝脗聝脙聜脗陇脙聝脗聜脙聜脗赂脙聝脗聜脙聜脗聧脙聝脗聝脙聜脗楼脙聝脗聜脙聜脗潞脙聝脗聜脙聜脗聰脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗禄脙聝脗聜脙聜脗搂脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗禄脙聝脗聜脙聜脗颅脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗碌脙聝脗聜脙聜脗掳脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗驴脙聝脗聜脙聜脗聸脙聝脗聝脙聜脗篓脙聝脗聜脙聜脗聶脙聝脗聜脙聜脗職脙聝脗聝脙聜脗搂脙聝脗聜脙聜脗漏脙聝脗聜脙聜脗潞脙聝脗聝脙聜脗拢脙聝脗聜脙聜脗聙脙聝脗聜脙聜脗聜
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

fn gap_hop_intent(obs: &[f32], jump_left: bool, jump_right: bool) -> bool {
    (jump_right && obs_same_level_gap_ahead(obs, 1.0, WINDOW_W, WINDOW_H))
        || (jump_left && obs_same_level_gap_ahead(obs, -1.0, WINDOW_W, WINDOW_H))
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
            allow_nav_walk: false,
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
            allow_nav_walk: false,
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
            allow_nav_walk: false,
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
            allow_nav_walk: false,
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
        // 杩滃鍚屽眰鎬細涓嶅湪鍙爫甯?
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
            allow_nav_walk: false,
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
            allow_nav_walk: false,
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
            allow_nav_walk: false,
        };
        let mut inp = InputFrame::default();
        inp.jump = true;
        inp.right = true;
        let out = gate.filter_input(&inp, ctx);
        assert!(out.jump);
    }

    #[test]
    fn allow_step_up_keeps_jump_and_dir_when_yolo_cliff() {
        let mut gate = MovementGate::default();
        let mut obs = [0.0_f32; OBS_DIM];
        // 鑴氫笅鏈夊彴锛屽墠鏂规棤鍦版澘 鈫?walk 浼氳鍓ワ紱allow_step_up 鏃跺簲淇濈暀 jump+right
        floor_slot(&mut obs, 0.0, 0.02, 0.2);
        gate.set_last_observation(&obs);
        let ctx = MovementGateCtx {
            facing: 1.0,
            on_ground: true,
            climbing: false,
            can_use_potion: false,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: Some(false),
            physics_drop_left: Some(false),
            sim_mob_in_melee: false,
            allow_combat_leap: false,
            adjacent_climb: false,
            allow_step_up: true,
            allow_nav_walk: false,
        };
        let mut inp = InputFrame::default();
        inp.jump = true;
        inp.right = true;
        let out = gate.filter_input(&inp, ctx);
        assert!(out.jump, "step_up must keep jump");
        assert!(out.right, "step_up must keep right toward ledge");
    }

    #[test]
    fn allow_nav_walk_trusts_physics_when_yolo_floor_short() {
        let mut gate = MovementGate::default();
        let mut obs = [0.0_f32; OBS_DIM];
        // 地板全在左侧：无 allow_nav_walk 时右走会被剥。
        floor_slot(&mut obs, -0.2, 0.02, 0.15);
        gate.set_last_observation(&obs);
        let mut ctx = MovementGateCtx {
            facing: 1.0,
            on_ground: true,
            climbing: false,
            can_use_potion: false,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: Some(false),
            physics_drop_left: Some(false),
            sim_mob_in_melee: false,
            allow_combat_leap: false,
            adjacent_climb: false,
            allow_step_up: false,
            allow_nav_walk: false,
        };
        let mut inp = InputFrame::default();
        inp.right = true;
        assert!(!gate.filter_input(&inp, ctx).right);
        ctx.allow_nav_walk = true;
        assert!(gate.filter_input(&inp, ctx).right);
    }

    #[test]
    fn strips_jump_while_climbing() {
        let mut gate = MovementGate::default();
        let mut obs = [0.0_f32; OBS_DIM];
        floor_slot(&mut obs, 0.0, 0.02, 0.3);
        gate.set_last_observation(&obs);
        let ctx = MovementGateCtx {
            facing: 1.0,
            on_ground: false,
            climbing: true,
            can_use_potion: false,
            physics_right_ok: None,
            physics_left_ok: None,
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            allow_combat_leap: false,
            adjacent_climb: true,
            allow_step_up: false,
            allow_nav_walk: false,
        };
        let mut inp = InputFrame::default();
        inp.up = true;
        inp.jump = true;
        let out = gate.filter_input(&inp, ctx);
        assert!(out.up, "climbing must keep up");
        assert!(!out.jump, "jump while climbing detaches rope");
    }

    #[test]
    fn allows_gap_hop_jump_when_loose_floor_ahead() {
        // YOLO 松散看到对面台（非边缘），但缝不衔接 → 仍应放行 jump+right。
        let mut gate = MovementGate::default();
        let mut obs = [0.0_f32; OBS_DIM];
        floor_slot(&mut obs, 0.0, 0.01, 12.0 / 1368.0);
        let under_half = 6.0 / 1368.0;
        let gap = 50.0 / 1368.0;
        let opp_half = 40.0 / 1368.0;
        obs[OBS_FLOOR_START + 4] = under_half + gap + opp_half;
        obs[OBS_FLOOR_START + 5] = 0.01;
        obs[OBS_FLOOR_START + 6] = 80.0 / 1368.0;
        obs[OBS_FLOOR_START + 7] = 0.02;
        gate.set_last_observation(&obs);
        let ctx = MovementGateCtx {
            facing: 1.0,
            on_ground: true,
            climbing: false,
            can_use_potion: false,
            physics_right_ok: Some(false),
            physics_left_ok: Some(true),
            physics_drop_right: Some(false),
            physics_drop_left: Some(false),
            sim_mob_in_melee: false,
            allow_combat_leap: false,
            adjacent_climb: false,
            allow_step_up: false,
            allow_nav_walk: false,
        };
        let mut inp = InputFrame::default();
        inp.right = true;
        inp.jump = true;
        let out = gate.filter_input(&inp, ctx);
        assert!(out.jump && out.right, "gap hop must keep jump+right, out={out:?}");
    }
}
