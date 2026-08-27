//! 训练用：用 sim 投影脚点匹配 YOLO「玩家」框，替代 OCR 定位自身。

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::player_name::NamedPlayerHit;
use crate::yolo::Detection;

const PLAYER_LABEL: &str = "玩家";
/// 最近 YOLO 玩家框与 sim 脚点距离超过此值则视为未匹配。
const MAX_MATCH_DIST_PX: f32 = 220.0;

/// 每局固定的锚点偏移（由 episode_seed 决定，整局不变）。
pub fn episode_anchor_offset(seed: u64, max_px: f32) -> (f32, f32) {
    if max_px <= 0.0 {
        return (0.0, 0.0);
    }
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (
        rng.gen_range(-max_px..=max_px),
        rng.gen_range(-max_px..=max_px),
    )
}

fn detection_foot(det: &Detection) -> (f32, f32) {
    ((det.x1 + det.x2) * 0.5, det.y2)
}

/// 在 YOLO 玩家框中选取与 sim 投影脚点最近的一个，锚点坐标与 OCR 路径一致（框脚点 + 偏移）。
pub fn find_self_player_by_sim(
    detections: &[Detection],
    player_x: f32,
    player_y: f32,
    cam_x: f32,
    cam_y: f32,
    min_player_conf: f32,
    offset_x: f32,
    offset_y: f32,
) -> Option<NamedPlayerHit> {
    let screen_x = player_x - cam_x;
    let screen_y = player_y - cam_y;

    let mut best: Option<(&Detection, f32)> = None;
    for det in detections {
        if det.label != PLAYER_LABEL || det.conf < min_player_conf {
            continue;
        }
        let (fx, fy) = detection_foot(det);
        let dx = fx - screen_x;
        let dy = fy - screen_y;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq > MAX_MATCH_DIST_PX * MAX_MATCH_DIST_PX {
            continue;
        }
        if best
            .map(|(_, d)| dist_sq < d)
            .unwrap_or(true)
        {
            best = Some((det, dist_sq));
        }
    }

    let (det, _) = best?;
    let (fx, fy) = detection_foot(det);
    Some(NamedPlayerHit {
        x: fx + offset_x,
        y: fy + offset_y,
        ocr_text: String::from("sim-match"),
        match_score: 1.0,
        partial: false,
        player_conf: det.conf,
        roi: (
            det.x1.max(0.0) as u32,
            det.y1.max(0.0) as u32,
            (det.x2 - det.x1).max(1.0) as u32,
            (det.y2 - det.y1).max(1.0) as u32,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yolo::CLASS_NAMES;

    fn player_det(x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection {
            class_id: 10,
            label: CLASS_NAMES[10],
            conf: 0.9,
            x1,
            y1,
            x2,
            y2,
        }
    }

    #[test]
    fn picks_nearest_player_box() {
        let dets = vec![
            player_det(100.0, 200.0, 140.0, 300.0),
            player_det(500.0, 200.0, 540.0, 300.0),
        ];
        let hit = find_self_player_by_sim(&dets, 520.0, 310.0, 0.0, 10.0, 0.7, 0.0, 0.0)
            .expect("match");
        assert!((hit.x - 520.0).abs() < 0.01);
        assert!((hit.y - 300.0).abs() < 0.01);
    }

    #[test]
    fn offset_is_deterministic_per_seed() {
        let a = episode_anchor_offset(42, 10.0);
        let b = episode_anchor_offset(42, 10.0);
        assert_eq!(a, b);
    }
}
