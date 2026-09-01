//! NEAT 网络输入压缩：102 维原始槽位 → 24 维事实摘要。
//!
//! 只做降维，不做决策：摘要里全是「有什么、在哪、多远、动没动」这类事实，
//! 不含「该往哪走」的结论，否则网络无可学。数据源仍是纯 YOLO 槽位 + OCR 脚点。

use crate::game::map::ClimbDir;
use crate::game::observation::{
    obs_assess_enemy_contact, obs_climb_grab_ready, obs_climb_hint,
    obs_enemy_in_attack_range_platform, obs_floor_ahead_connected, obs_floor_drop_ahead,
    obs_floor_underfoot, obs_has_nearby_platform_enemy, obs_has_platform_enemy,
    obs_nearest_same_level_enemy_px, obs_platform_edge, obs_same_level_enemy_count, obs_step_up_dx,
    OBS_PROPRIO_START,
};
use crate::game::types::{WINDOW_H, WINDOW_W};

pub const NEAT_OBS_DIM: usize = 24;

const ENEMY_DX_SCALE: f32 = 300.0;
const ENEMY_DY_SCALE: f32 = 120.0;
const NAV_DX_SCALE: f32 = 120.0;

/// `last_macro_failed`：上一个动作宏是否失败（撞墙/抓不上绳），让网络能学会换动作。
pub fn compact_obs(obs: &[f32], last_macro_failed: bool) -> [f32; NEAT_OBS_DIM] {
    let mut v = [0.0_f32; NEAT_OBS_DIM];
    let facing = facing_from_proprio(obs);

    if let Some((dx, dy)) = obs_nearest_same_level_enemy_px(obs, WINDOW_W, WINDOW_H) {
        v[0] = (dx / ENEMY_DX_SCALE).clamp(-1.0, 1.0);
        v[1] = (dy / ENEMY_DY_SCALE).clamp(-1.0, 1.0);
    }
    v[2] = flag(obs_has_platform_enemy(obs));
    v[3] = flag(obs_has_nearby_platform_enemy(obs));
    v[4] = (obs_same_level_enemy_count(obs) as f32 / 3.0).min(1.0);
    v[5] = flag(obs_enemy_in_attack_range_platform(obs, facing));
    let contact = obs_assess_enemy_contact(obs);
    v[6] = (contact.left as f32 / 2.0).min(1.0);
    v[7] = (contact.right as f32 / 2.0).min(1.0);

    v[8] = flag(obs_floor_ahead_connected(obs, -1.0));
    v[9] = flag(obs_floor_ahead_connected(obs, 1.0));
    v[10] = flag(obs_floor_underfoot(obs));
    v[11] = flag(obs_platform_edge(obs, -1.0));
    v[12] = flag(obs_platform_edge(obs, 1.0));
    v[13] = flag(obs_floor_drop_ahead(obs, -1.0));
    v[14] = flag(obs_floor_drop_ahead(obs, 1.0));

    v[15] = flag(obs_climb_grab_ready(obs));
    if let Some(hint) = obs_climb_hint(obs, WINDOW_W, WINDOW_H) {
        v[16] = match hint.dir {
            ClimbDir::Up => 1.0,
            ClimbDir::Down => -1.0,
        };
        v[17] = (hint.dx / NAV_DX_SCALE).clamp(-1.0, 1.0);
    }
    if let Some(dx) = obs_step_up_dx(obs, WINDOW_W, WINDOW_H) {
        v[18] = (dx / NAV_DX_SCALE).clamp(-1.0, 1.0);
    }

    v[19] = obs.get(OBS_PROPRIO_START).copied().unwrap_or(0.0);
    v[20] = obs.get(OBS_PROPRIO_START + 1).copied().unwrap_or(0.0);
    v[21] = obs.get(OBS_PROPRIO_START + 2).copied().unwrap_or(0.0);
    v[22] = obs.get(OBS_PROPRIO_START + 3).copied().unwrap_or(0.0);
    v[23] = flag(last_macro_failed);
    v
}

fn flag(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// 朝向由 OCR 反馈的「上一帧按了哪个方向」推出，不读 sim 的 facing。
fn facing_from_proprio(obs: &[f32]) -> f32 {
    let last_left = obs.get(OBS_PROPRIO_START + 4).copied().unwrap_or(0.0) >= 0.5;
    let last_right = obs.get(OBS_PROPRIO_START + 5).copied().unwrap_or(0.0) >= 0.5;
    if last_left && !last_right {
        -1.0
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::observation::{OBS_DIM, OBS_ENEMY_START, OBS_FLOOR_START};

    fn floor_under(obs: &mut [f32; OBS_DIM]) {
        obs[OBS_FLOOR_START] = 0.0;
        obs[OBS_FLOOR_START + 1] = 0.02;
        obs[OBS_FLOOR_START + 2] = 0.3;
        obs[OBS_FLOOR_START + 3] = 0.05;
    }

    #[test]
    fn compact_keeps_enemy_geometry() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_under(&mut obs);
        obs[OBS_ENEMY_START] = -60.0 / WINDOW_W;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
        let v = compact_obs(&obs, false);
        assert!(v[0] < 0.0, "怪在左侧应为负 dx");
        assert_eq!(v[2], 1.0);
    }

    #[test]
    fn compact_reports_underfoot_floor_and_failure_flag() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_under(&mut obs);
        let v = compact_obs(&obs, true);
        assert_eq!(v[10], 1.0);
        assert_eq!(v[23], 1.0);
    }

    #[test]
    fn compact_is_all_zero_without_detections() {
        let obs = [0.0_f32; OBS_DIM];
        let v = compact_obs(&obs, false);
        assert!(v.iter().all(|x| *x == 0.0));
    }
}
