//! NEAT 寻路网络输入：102 维原始槽位 → 21 维几何事实摘要。
//!
//! 地板 / 边缘 / 绳梯 / 本体感 / 探索记忆提示（未访问方向 + 停滞压力）。
//! 不含怪物——砍怪由状态机负责。数据源仍是纯 YOLO 槽位 + OCR 脚点 + 里程计维护的记忆。

use crate::game::explore_memory::ExploreHints;
use crate::game::map::ClimbDir;
use crate::game::observation::{
    obs_climb_grab_ready, obs_climb_hint, obs_floor_ahead_connected, obs_floor_drop_ahead,
    obs_floor_underfoot, obs_platform_edge, obs_step_up_dx, OBS_PROPRIO_START,
};
use crate::game::types::{WINDOW_H, WINDOW_W};

pub const NEAT_OBS_DIM: usize = 21;

const NAV_DX_SCALE: f32 = 120.0;

/// `last_macro_failed`：上一个动作宏是否失败（撞墙/抓不上绳），让网络能学会换动作。
pub fn compact_obs(
    obs: &[f32],
    last_macro_failed: bool,
    hints: &ExploreHints,
) -> [f32; NEAT_OBS_DIM] {
    let mut v = [0.0_f32; NEAT_OBS_DIM];

    v[0] = flag(obs_floor_ahead_connected(obs, -1.0));
    v[1] = flag(obs_floor_ahead_connected(obs, 1.0));
    v[2] = flag(obs_floor_underfoot(obs));
    v[3] = flag(obs_platform_edge(obs, -1.0));
    v[4] = flag(obs_platform_edge(obs, 1.0));
    v[5] = flag(obs_floor_drop_ahead(obs, -1.0));
    v[6] = flag(obs_floor_drop_ahead(obs, 1.0));

    v[7] = flag(obs_climb_grab_ready(obs));
    if let Some(hint) = obs_climb_hint(obs, WINDOW_W, WINDOW_H) {
        v[8] = match hint.dir {
            ClimbDir::Up => 1.0,
            ClimbDir::Down => -1.0,
        };
        v[9] = (hint.dx / NAV_DX_SCALE).clamp(-1.0, 1.0);
    }
    if let Some(dx) = obs_step_up_dx(obs, WINDOW_W, WINDOW_H) {
        v[10] = (dx / NAV_DX_SCALE).clamp(-1.0, 1.0);
    }

    v[11] = obs.get(OBS_PROPRIO_START).copied().unwrap_or(0.0);
    v[12] = obs.get(OBS_PROPRIO_START + 1).copied().unwrap_or(0.0);
    v[13] = obs.get(OBS_PROPRIO_START + 2).copied().unwrap_or(0.0);
    v[14] = obs.get(OBS_PROPRIO_START + 3).copied().unwrap_or(0.0);
    v[15] = flag(last_macro_failed);

    v[16] = hints.unvisited_left;
    v[17] = hints.unvisited_right;
    v[18] = hints.unvisited_band_up;
    v[19] = hints.unvisited_band_down;
    v[20] = hints.stall_pressure;
    v
}

fn flag(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::explore_memory::ExploreHints;
    use crate::game::observation::{OBS_DIM, OBS_ENEMY_START, OBS_FLOOR_START};

    fn floor_under(obs: &mut [f32; OBS_DIM]) {
        obs[OBS_FLOOR_START] = 0.0;
        obs[OBS_FLOOR_START + 1] = 0.02;
        obs[OBS_FLOOR_START + 2] = 0.3;
        obs[OBS_FLOOR_START + 3] = 0.05;
    }

    #[test]
    fn compact_reports_underfoot_floor_and_failure_flag() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_under(&mut obs);
        let hints = ExploreHints::default();
        let v = compact_obs(&obs, true, &hints);
        assert_eq!(v[2], 1.0);
        assert_eq!(v[15], 1.0);
    }

    #[test]
    fn compact_includes_explore_hints() {
        let obs = [0.0_f32; OBS_DIM];
        let hints = ExploreHints {
            unvisited_right: 1.0,
            stall_pressure: 0.5,
            ..ExploreHints::default()
        };
        let v = compact_obs(&obs, false, &hints);
        assert_eq!(v[16], 0.0);
        assert_eq!(v[17], 1.0);
        assert_eq!(v[20], 0.5);
    }

    #[test]
    fn compact_ignores_enemies() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_under(&mut obs);
        let hints = ExploreHints::default();
        let base = compact_obs(&obs, false, &hints);
        obs[OBS_ENEMY_START] = -60.0 / WINDOW_W;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
        assert_eq!(compact_obs(&obs, false, &hints), base, "怪物不进寻路网络");
    }

    #[test]
    fn compact_is_all_zero_without_detections() {
        let obs = [0.0_f32; OBS_DIM];
        let hints = ExploreHints::default();
        let v = compact_obs(&obs, false, &hints);
        assert!(v.iter().all(|x| *x == 0.0));
    }
}
