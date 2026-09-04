//! 训练用：自身脚点整局固定抖动（域随机化）。

use crate::game::self_track::SelfPlayerHit;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

/// 在自身脚点上叠加整局固定抖动（仅改观测原点，不改 YOLO 框）。
pub fn apply_anchor_jitter(hit: &mut SelfPlayerHit, episode_seed: u64, max_px: f32) {
    if max_px <= 0.0 {
        return;
    }
    let (ox, oy) = episode_anchor_offset(episode_seed, max_px);
    hit.x += ox;
    hit.y += oy;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_is_deterministic_per_seed() {
        let a = episode_anchor_offset(42, 10.0);
        let b = episode_anchor_offset(42, 10.0);
        assert_eq!(a, b);
    }

    #[test]
    fn apply_jitter_mutates_foot_point() {
        let mut hit = SelfPlayerHit {
            x: 520.0,
            y: 300.0,
            conf: 0.9,
            x1: 500.0,
            y1: 200.0,
            x2: 540.0,
            y2: 300.0,
        };
        let base_y = hit.y;
        apply_anchor_jitter(&mut hit, 42, 10.0);
        assert!((hit.y - base_y).abs() > 1e-3 || (hit.x - 520.0).abs() > 1e-3);
    }
}
