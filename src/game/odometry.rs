//! 视觉里程计：用相邻两帧静态地标（地板/梯子/绳子）的相对位移反推自身世界位移。
//!
//! 相机居中跟随时 OCR 名牌的屏幕坐标几乎不变，直接拿名牌位移当「走了多远」在地图中部恒为 0，
//! 只有相机贴边停住时才有值——这就是旧版 blocked 判定在地图中央从不生效的原因。
//! 静态地标相对名牌的偏移则总是随自身世界位移等量反向变化（相机跟随时地标屏幕位移 = -Δ自身，
//! 相机贴边时名牌屏幕位移 = +Δ自身，两种情况下 Δ(地标-名牌) 都等于 -Δ自身），所以对相邻两帧
//! 按几何匹配同类地标，取相对偏移变化量的中位数取反即为自身位移。只用 YOLO 框，可直接用于真机。

use super::observation::{
    OBS_FLOOR_SLOTS, OBS_FLOOR_START, OBS_LADDER_SLOTS, OBS_LADDER_START, OBS_ROPE_SLOTS,
    OBS_ROPE_START, OBS_SLOT_DIM,
};
use super::types::{WINDOW_H, WINDOW_W};

/// 相邻两帧同一地标允许的最大屏幕位移（跳跃峭壁段一帧约 40px）。
const MATCH_RADIUS_PX: f32 = 90.0;
/// 同一地标两帧框尺寸差容忍（YOLO 抖动 + 轻微遮挡）。
const SIZE_TOL_PX: f32 = 24.0;
/// 框贴到屏幕边缘即视为被裁切：该轴的中心位移不再反映真实位移。
const EDGE_PX: f32 = 2.0;

#[derive(Debug, Clone, Copy)]
struct Landmark {
    class: u8,
    /// 相对名牌的中心偏移（px）。
    rx: f32,
    ry: f32,
    w: f32,
    h: f32,
    clipped_x: bool,
    clipped_y: bool,
}

fn collect(obs: &[f32]) -> Vec<Landmark> {
    let ax = obs.first().copied().unwrap_or(0.5) * WINDOW_W;
    let ay = obs.get(1).copied().unwrap_or(0.5) * WINDOW_H;
    let groups = [
        (0u8, OBS_FLOOR_START, OBS_FLOOR_SLOTS),
        (1u8, OBS_LADDER_START, OBS_LADDER_SLOTS),
        (2u8, OBS_ROPE_START, OBS_ROPE_SLOTS),
    ];
    let mut out = Vec::new();
    for (class, start, count) in groups {
        for i in 0..count {
            let base = start + i * OBS_SLOT_DIM;
            if base + OBS_SLOT_DIM > obs.len() {
                break;
            }
            let w = obs[base + 2] * WINDOW_W;
            let h = obs[base + 3] * WINDOW_H;
            if w <= 0.1 && h <= 0.1 {
                continue;
            }
            let rx = obs[base] * WINDOW_W;
            let ry = obs[base + 1] * WINDOW_H;
            let sx = ax + rx;
            let sy = ay + ry;
            out.push(Landmark {
                class,
                rx,
                ry,
                w,
                h,
                clipped_x: sx - w * 0.5 <= EDGE_PX || sx + w * 0.5 >= WINDOW_W - EDGE_PX,
                clipped_y: sy - h * 0.5 <= EDGE_PX || sy + h * 0.5 >= WINDOW_H - EDGE_PX,
            });
        }
    }
    out
}

fn median(v: &mut Vec<f32>) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

/// 自身世界位移（px，右/下为正）。无任何可匹配地标时返回 None。
pub fn estimate_world_delta_px(prev: &[f32], cur: &[f32]) -> Option<(f32, f32)> {
    let prev_marks = collect(prev);
    let cur_marks = collect(cur);
    let mut dx_all = Vec::new();
    let mut dy_all = Vec::new();
    let mut dx_clean = Vec::new();
    let mut dy_clean = Vec::new();
    for c in &cur_marks {
        let mut best: Option<(f32, &Landmark)> = None;
        for p in &prev_marks {
            if p.class != c.class
                || (p.w - c.w).abs() > SIZE_TOL_PX
                || (p.h - c.h).abs() > SIZE_TOL_PX
            {
                continue;
            }
            let d = ((p.rx - c.rx).powi(2) + (p.ry - c.ry).powi(2)).sqrt();
            if d <= MATCH_RADIUS_PX && best.map_or(true, |(bd, _)| d < bd) {
                best = Some((d, p));
            }
        }
        let Some((_, p)) = best else {
            continue;
        };
        let dx = p.rx - c.rx;
        let dy = p.ry - c.ry;
        dx_all.push(dx);
        dy_all.push(dy);
        if !p.clipped_x && !c.clipped_x {
            dx_clean.push(dx);
        }
        if !p.clipped_y && !c.clipped_y {
            dy_clean.push(dy);
        }
    }
    if dx_all.is_empty() {
        return None;
    }
    let dx = median(&mut dx_clean).or_else(|| median(&mut dx_all))?;
    let dy = median(&mut dy_clean).or_else(|| median(&mut dy_all))?;
    Some((dx, dy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::observation::OBS_DIM;

    fn obs_with_anchor(ax: f32, ay: f32) -> [f32; OBS_DIM] {
        let mut o = [0.0_f32; OBS_DIM];
        o[0] = ax / WINDOW_W;
        o[1] = ay / WINDOW_H;
        o
    }

    fn put(o: &mut [f32; OBS_DIM], base: usize, rx: f32, ry: f32, w: f32, h: f32) {
        o[base] = rx / WINDOW_W;
        o[base + 1] = ry / WINDOW_H;
        o[base + 2] = w / WINDOW_W;
        o[base + 3] = h / WINDOW_H;
    }

    #[test]
    fn following_camera_landmarks_shift_opposite_to_motion() {
        let mut prev = obs_with_anchor(684.0, 400.0);
        put(&mut prev, OBS_FLOOR_START, 100.0, 40.0, 300.0, 20.0);
        put(&mut prev, OBS_ROPE_START, 250.0, -60.0, 12.0, 120.0);
        let mut cur = obs_with_anchor(684.0, 400.0);
        // 自身向右走 12px：地标相对偏移全部 -12。
        put(&mut cur, OBS_FLOOR_START, 88.0, 40.0, 300.0, 20.0);
        put(&mut cur, OBS_ROPE_START, 238.0, -60.0, 12.0, 120.0);
        let (dx, dy) = estimate_world_delta_px(&prev, &cur).unwrap();
        assert!((dx - 12.0).abs() < 0.5, "dx={dx}");
        assert!(dy.abs() < 0.5);
    }

    #[test]
    fn clamped_camera_anchor_moves_instead() {
        // 相机贴边：地标屏幕位置不动，名牌向右走 12px → 相对偏移同样 -12。
        let mut prev = obs_with_anchor(200.0, 400.0);
        put(&mut prev, OBS_FLOOR_START, 100.0, 40.0, 300.0, 20.0);
        let mut cur = obs_with_anchor(212.0, 400.0);
        put(&mut cur, OBS_FLOOR_START, 88.0, 40.0, 300.0, 20.0);
        let (dx, _) = estimate_world_delta_px(&prev, &cur).unwrap();
        assert!((dx - 12.0).abs() < 0.5, "dx={dx}");
    }

    #[test]
    fn edge_clipped_floor_does_not_pollute_horizontal() {
        let mut prev = obs_with_anchor(684.0, 400.0);
        // 横跨整屏的地板：两侧都被裁，中心不随相机动。
        put(&mut prev, OBS_FLOOR_START, 0.0, 300.0, WINDOW_W, 20.0);
        put(&mut prev, OBS_ROPE_START, 250.0, -60.0, 12.0, 120.0);
        let mut cur = obs_with_anchor(684.0, 400.0);
        put(&mut cur, OBS_FLOOR_START, 0.0, 300.0, WINDOW_W, 20.0);
        put(&mut cur, OBS_ROPE_START, 238.0, -60.0, 12.0, 120.0);
        let (dx, _) = estimate_world_delta_px(&prev, &cur).unwrap();
        assert!((dx - 12.0).abs() < 0.5, "dx={dx}");
    }

    #[test]
    fn vertical_motion_while_climbing() {
        let mut prev = obs_with_anchor(684.0, 400.0);
        put(&mut prev, OBS_FLOOR_START, 100.0, 40.0, 300.0, 20.0);
        put(&mut prev, OBS_FLOOR_START + OBS_SLOT_DIM, -200.0, -150.0, 260.0, 20.0);
        let mut cur = obs_with_anchor(684.0, 400.0);
        put(&mut cur, OBS_FLOOR_START, 100.0, 50.0, 300.0, 20.0);
        put(&mut cur, OBS_FLOOR_START + OBS_SLOT_DIM, -200.0, -140.0, 260.0, 20.0);
        let (dx, dy) = estimate_world_delta_px(&prev, &cur).unwrap();
        assert!(dx.abs() < 0.5);
        assert!((dy + 10.0).abs() < 0.5, "向上爬 10px → dy=-10, got {dy}");
    }

    #[test]
    fn slot_reordering_is_irrelevant() {
        let mut prev = obs_with_anchor(684.0, 400.0);
        put(&mut prev, OBS_FLOOR_START, 100.0, 40.0, 300.0, 20.0);
        put(&mut prev, OBS_FLOOR_START + OBS_SLOT_DIM, -200.0, -150.0, 260.0, 20.0);
        let mut cur = obs_with_anchor(684.0, 400.0);
        put(&mut cur, OBS_FLOOR_START, -212.0, -150.0, 260.0, 20.0);
        put(&mut cur, OBS_FLOOR_START + OBS_SLOT_DIM, 88.0, 40.0, 300.0, 20.0);
        let (dx, _) = estimate_world_delta_px(&prev, &cur).unwrap();
        assert!((dx - 12.0).abs() < 0.5, "dx={dx}");
    }

    #[test]
    fn no_landmarks_gives_none() {
        let prev = obs_with_anchor(684.0, 400.0);
        let cur = obs_with_anchor(684.0, 400.0);
        assert!(estimate_world_delta_px(&prev, &cur).is_none());
    }
}
