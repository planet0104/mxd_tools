//! 从 sim 真值合成 YOLO 观测（headless bot 探针，无需渲染/YOLO）。

use crate::player_name::NamedPlayerHit;
use crate::yolo::Detection;
use crate::yolo::CLASS_NAMES;

use super::camera::WorldCamera;
use super::map::GameMap;
use super::observation::{VisionObservation, OBS_DIM};
use super::sim::GameSim;
use super::types::{WINDOW_H, WINDOW_W, WORLD_VIEW_H};

const SYNTH_CONF: f32 = 0.92;
const VIEW_MARGIN: f32 = 96.0;
const FLOOR_THICK: f32 = 40.0;

/// 由当前 sim 状态构建观测向量（与 `VisionObservation::from_detections` 布局一致）。
pub fn observation_from_sim(sim: &GameSim) -> [f32; OBS_DIM] {
    let p = &sim.state.player;
    let cam = WorldCamera {
        cam_x: sim.state.cam_x,
        cam_y: sim.state.cam_y,
    };
    let ax = cam.player_screen_x(p.x);
    let ay = cam.player_screen_y(p.y);

    let mut dets = Vec::new();
    collect_floors(&sim.map, &cam, &mut dets);
    collect_mobs(sim, &cam, &mut dets);
    collect_drops(sim, &cam, &mut dets);
    collect_ropes(&sim.map, &cam, &mut dets);

    let self_hit = NamedPlayerHit {
        x: ax,
        y: ay,
        ocr_text: String::new(),
        match_score: 1.0,
        partial: false,
        player_conf: SYNTH_CONF,
        roi: (0, 0, 0, 0),
    };

    let obs = VisionObservation::from_detections(
        &dets,
        Some(&self_hit),
        WINDOW_W as u32,
        WINDOW_H as u32,
    );
    let mut out = [0.0_f32; OBS_DIM];
    let n = obs.values.len().min(OBS_DIM);
    out[..n].copy_from_slice(&obs.values[..n]);
    out
}

fn collect_floors(map: &GameMap, cam: &WorldCamera, out: &mut Vec<Detection>) {
    for plat in &map.platforms {
        let (sx1, sy) = world_to_screen(cam, plat.x1, plat.y1);
        let (sx2, _) = world_to_screen(cam, plat.x2, plat.y2);
        let x1 = sx1.min(sx2);
        let x2 = sx1.max(sx2);
        let y2 = sy;
        let y1 = sy - FLOOR_THICK;
        if !box_on_screen(x1, y1, x2, y2) {
            continue;
        }
        push_det(out, "地板", x1, y1, x2, y2);
    }
}

fn collect_mobs(sim: &GameSim, cam: &WorldCamera, out: &mut Vec<Detection>) {
    for mob in &sim.state.mobs {
        if !mob.alive {
            continue;
        }
        let (w, h) = mob_bbox_wh(mob.mob_id);
        let (sx, sy) = world_to_screen(cam, mob.x, mob.y);
        let x1 = sx - w * 0.5;
        let x2 = sx + w * 0.5;
        let y1 = sy - h;
        let y2 = sy;
        if !box_on_screen(x1, y1, x2, y2) {
            continue;
        }
        push_det(out, mob_enemy_label(mob.mob_id), x1, y1, x2, y2);
    }
}

fn collect_drops(sim: &GameSim, cam: &WorldCamera, out: &mut Vec<Detection>) {
    for drop in &sim.state.drops {
        if !drop.alive {
            continue;
        }
        let (sx, sy) = world_to_screen(cam, drop.x, drop.y - 8.0);
        let w = 28.0;
        let h = 28.0;
        let x1 = sx - w * 0.5;
        let x2 = sx + w * 0.5;
        let y1 = sy - h * 0.5;
        let y2 = sy + h * 0.5;
        if !box_on_screen(x1, y1, x2, y2) {
            continue;
        }
        let label = match drop.kind {
            super::types::DropKind::Meso => "金币",
            super::types::DropKind::RedPotion => "药水",
        };
        push_det(out, label, x1, y1, x2, y2);
    }
}

fn collect_ropes(map: &GameMap, cam: &WorldCamera, out: &mut Vec<Detection>) {
    for rope in &map.ropes {
        let (sx, sy1) = world_to_screen(cam, rope.x, rope.y1);
        let (_, sy2) = world_to_screen(cam, rope.x, rope.y2);
        let half = rope.width * 0.5;
        let x1 = sx - half;
        let x2 = sx + half;
        let y1 = sy1.min(sy2);
        let y2 = sy1.max(sy2);
        if !box_on_screen(x1, y1, x2, y2) {
            continue;
        }
        let label = if rope.kind.contains("ladder") {
            "梯子"
        } else {
            "绳子"
        };
        push_det(out, label, x1, y1, x2, y2);
    }
}

fn world_to_screen(cam: &WorldCamera, wx: f32, wy: f32) -> (f32, f32) {
    (wx - cam.cam_x, wy - cam.cam_y)
}

fn box_on_screen(x1: f32, y1: f32, x2: f32, y2: f32) -> bool {
    let sx1 = -VIEW_MARGIN;
    let sy1 = -VIEW_MARGIN;
    let sx2 = WINDOW_W + VIEW_MARGIN;
    let sy2 = WORLD_VIEW_H + VIEW_MARGIN;
    x1 <= sx2 && x2 >= sx1 && y1 <= sy2 && y2 >= sy1
}

fn push_det(out: &mut Vec<Detection>, label: &'static str, x1: f32, y1: f32, x2: f32, y2: f32) {
    let class_id = CLASS_NAMES.iter().position(|&n| n == label).unwrap_or(0);
    out.push(Detection {
        class_id,
        label,
        conf: SYNTH_CONF,
        x1,
        y1,
        x2,
        y2,
    });
}

fn mob_enemy_label(mob_id: u32) -> &'static str {
    match mob_id {
        100101 => "蓝蜗牛",
        100100 => "绿蜗牛",
        130101 => "红蜗牛",
        1210102 => "花蘑菇",
        130100 => "树怪",
        _ => "蓝蜗牛",
    }
}

fn mob_bbox_wh(mob_id: u32) -> (f32, f32) {
    match mob_id {
        130100 => (54.0, 62.0),
        1210102 => (50.0, 52.0),
        _ => (44.0, 48.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::load_default_map;
    use crate::game::observation::{obs_has_enemy, obs_has_floor_signal};
    use crate::game::GameSim;

    #[test]
    fn synth_obs_has_floor_and_mobs_at_start() {
        let map = load_default_map().expect("map");
        let sim = GameSim::new_preview(map, 0);
        let obs = observation_from_sim(&sim);
        assert!(obs_has_floor_signal(&obs));
        assert!(obs_has_enemy(&obs));
    }
}
