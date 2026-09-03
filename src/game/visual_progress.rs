use super::observation::{
    OBS_DIM, OBS_FLOOR_SLOTS, OBS_FLOOR_START, OBS_LADDER_SLOTS, OBS_LADDER_START, OBS_ROPE_SLOTS,
    OBS_ROPE_START, OBS_SLOT_DIM,
};
use super::types::{WINDOW_H, WINDOW_W};

/// 低置信度时拒绝的单帧最大位移（px），防止 180px 级误配来回跳。
const LOW_CONF_MAX_SHIFT: f32 = 48.0;
const LOW_CONF_MATCHES: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocationNode {
    pub x: i32,
    pub y: i32,
    pub terrain: u16,
}

#[derive(Debug, Clone, Copy)]
struct Landmark {
    group: u8,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

#[derive(Debug, Clone)]
struct FrameFeatures {
    self_x: f32,
    self_y: f32,
    landmarks: Vec<Landmark>,
}

#[derive(Debug, Clone)]
pub struct VisualMotionEstimator {
    previous: Option<FrameFeatures>,
    pub x: f32,
    pub y: f32,
    pub dx: f32,
    pub dy: f32,
    pub confidence: u8,
    pub terrain: u16,
}

impl Default for VisualMotionEstimator {
    fn default() -> Self {
        Self {
            previous: None,
            x: 0.0,
            y: 0.0,
            dx: 0.0,
            dy: 0.0,
            confidence: 0,
            terrain: 0,
        }
    }
}

impl VisualMotionEstimator {
    pub fn update(&mut self, obs: &[f32; OBS_DIM]) {
        let current = frame_features(obs);
        self.dx = 0.0;
        self.dy = 0.0;
        self.confidence = 0;
        self.terrain = terrain_fingerprint(&current);

        if let Some(previous) = &self.previous {
            if let Some((dx, dy, matches)) =
                estimate_rigid_shift(&previous.landmarks, &current.landmarks)
            {
                let shift = dx.abs().max(dy.abs());
                if matches >= LOW_CONF_MATCHES
                    || shift <= LOW_CONF_MAX_SHIFT
                    || (shift <= 80.0 && matches >= 2)
                {
                    self.dx = dx;
                    self.dy = dy;
                    self.confidence = matches;
                }
            }
            if self.confidence == 0 {
                let screen_dx = current.self_x - previous.self_x;
                let screen_dy = current.self_y - previous.self_y;
                let screen_shift = screen_dx.abs().max(screen_dy.abs());
                if screen_shift <= LOW_CONF_MAX_SHIFT
                    && (screen_dx.abs() >= 2.0 || screen_dy.abs() >= 2.0)
                {
                    self.dx = screen_dx;
                    self.dy = screen_dy;
                    self.confidence = 1;
                }
            }
        }

        if self.dx.abs() < 1.5 {
            self.dx = 0.0;
        }
        if self.dy.abs() < 1.5 {
            self.dy = 0.0;
        }
        self.x += self.dx;
        self.y += self.dy;
        self.previous = Some(current);
    }

    pub fn node(&self) -> LocationNode {
        LocationNode {
            x: (self.x / 64.0).floor() as i32,
            y: (self.y / 64.0).floor() as i32,
            terrain: self.terrain,
        }
    }
}

pub fn location_node(x: f32, y: f32, obs: &[f32; OBS_DIM]) -> LocationNode {
    LocationNode {
        x: (x / 64.0).floor() as i32,
        y: (y / 64.0).floor() as i32,
        terrain: terrain_fingerprint(&frame_features(obs)),
    }
}

fn frame_features(obs: &[f32; OBS_DIM]) -> FrameFeatures {
    let mut landmarks = Vec::new();
    collect_group(obs, OBS_FLOOR_START, OBS_FLOOR_SLOTS, 0, &mut landmarks);
    collect_group(obs, OBS_LADDER_START, OBS_LADDER_SLOTS, 1, &mut landmarks);
    collect_group(obs, OBS_ROPE_START, OBS_ROPE_SLOTS, 2, &mut landmarks);
    FrameFeatures {
        self_x: obs[0] * WINDOW_W,
        self_y: obs[1] * WINDOW_H,
        landmarks,
    }
}

fn collect_group(
    obs: &[f32; OBS_DIM],
    start: usize,
    count: usize,
    group: u8,
    out: &mut Vec<Landmark>,
) {
    for i in 0..count {
        let base = start + i * OBS_SLOT_DIM;
        let w = obs[base + 2] * WINDOW_W;
        let h = obs[base + 3] * WINDOW_H;
        if w <= 1.0 || h <= 1.0 {
            continue;
        }
        out.push(Landmark {
            group,
            x: obs[base] * WINDOW_W,
            y: obs[base + 1] * WINDOW_H,
            w,
            h,
        });
    }
}

fn terrain_fingerprint(frame: &FrameFeatures) -> u16 {
    let mut hash = frame.landmarks.len() as u32;
    for p in frame.landmarks.iter().take(6) {
        let qx = (p.x / 32.0).round() as i32;
        let qy = (p.y / 24.0).round() as i32;
        let qw = (p.w / 32.0).round() as i32;
        hash = hash
            .wrapping_mul(16777619)
            .wrapping_add((p.group as i32 * 31 + qx * 17 + qy * 7 + qw) as u32);
    }
    (hash ^ (hash >> 16)) as u16
}

/// 用「候选刚性位移」代替逐点最近邻：`fill_nearest_slots` 按距玩家远近排序，
/// 帧间同一地形块的槽位下标会跳动，逐点匹配容易把不同地形块错配到一起。
/// 这里改为枚举「旧->新同组候选位移」，选支持匹配数最多（残差最小作为平局判据）
/// 的单一整体位移，天然贴合摄像机跟随玩家时整个地形一起平移的物理约束。
fn estimate_rigid_shift(old: &[Landmark], new: &[Landmark]) -> Option<(f32, f32, u8)> {
    if old.is_empty() || new.is_empty() {
        return None;
    }
    const MAX_SIZE_DIFF: f32 = 40.0;
    const MATCH_TOL: f32 = 26.0;
    const MAX_SHIFT_X: f32 = 220.0;
    const MAX_SHIFT_Y: f32 = 160.0;

    let mut best: Option<(f32, f32, u32, f32)> = None;
    for old_pt in old {
        for new_pt in new.iter().filter(|p| p.group == old_pt.group) {
            let size_diff = (old_pt.w - new_pt.w).abs() + (old_pt.h - new_pt.h).abs();
            if size_diff > MAX_SIZE_DIFF {
                continue;
            }
            let cx = old_pt.x - new_pt.x;
            let cy = old_pt.y - new_pt.y;
            if cx.abs() > MAX_SHIFT_X || cy.abs() > MAX_SHIFT_Y {
                continue;
            }

            let mut match_count = 0u32;
            let mut residual = 0.0f32;
            for o in old {
                let pred_x = o.x - cx;
                let pred_y = o.y - cy;
                let mut nearest = f32::MAX;
                for n in new.iter().filter(|p| p.group == o.group) {
                    let d = ((n.x - pred_x).powi(2) + (n.y - pred_y).powi(2)).sqrt();
                    if d < nearest {
                        nearest = d;
                    }
                }
                if nearest <= MATCH_TOL {
                    match_count += 1;
                    residual += nearest;
                }
            }

            let better = match best {
                None => true,
                Some((_, _, best_count, best_residual)) => {
                    match_count > best_count
                        || (match_count == best_count && residual < best_residual)
                }
            };
            if better {
                best = Some((cx, cy, match_count, residual));
            }
        }
    }

    best.and_then(|(dx, dy, count, _)| {
        if count >= 2 {
            Some((dx, dy, count.min(u8::MAX as u32) as u8))
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_floor_slot(obs: &mut [f32; OBS_DIM], slot: usize, x: f32, y: f32, w: f32, h: f32) {
        let base = OBS_FLOOR_START + slot * OBS_SLOT_DIM;
        obs[base] = x / WINDOW_W;
        obs[base + 1] = y / WINDOW_H;
        obs[base + 2] = w / WINDOW_W;
        obs[base + 3] = h / WINDOW_H;
    }

    #[test]
    fn rigid_shift_survives_slot_reordering() {
        // fill_nearest_slots 按距玩家远近排序，帧间同一地形块的槽位下标会跳动；
        // 这里模拟槽位 0/1 互换顺序，估计器仍需给出与真实位移一致的 dx。
        let mut estimator = VisualMotionEstimator::default();
        let mut frame1 = [0.0_f32; OBS_DIM];
        set_floor_slot(&mut frame1, 0, 300.0, 500.0, 200.0, 50.0);
        set_floor_slot(&mut frame1, 1, 700.0, 460.0, 90.0, 50.0);
        estimator.update(&frame1);

        // 玩家向右走 60px：地形在屏幕上整体左移 60px；同时把两块地形的槽位顺序互换。
        let mut frame2 = [0.0_f32; OBS_DIM];
        set_floor_slot(&mut frame2, 0, 640.0, 460.0, 90.0, 50.0);
        set_floor_slot(&mut frame2, 1, 240.0, 500.0, 200.0, 50.0);
        estimator.update(&frame2);

        assert!(
            (estimator.dx - 60.0).abs() < 4.0,
            "dx should track true rightward motion despite slot reorder, got {}",
            estimator.dx
        );
        assert!(estimator.dy.abs() < 4.0);
        assert!(estimator.confidence >= 2);
    }

    #[test]
    fn rigid_shift_rejects_mismatched_single_landmark() {
        // 仅 1 个候选匹配（含尺寸差过大/唯一地形块变化过猛）时不应武断给出位移，
        // 避免单点误配把噪声当成大幅位移。
        let mut estimator = VisualMotionEstimator::default();
        let mut frame1 = [0.0_f32; OBS_DIM];
        set_floor_slot(&mut frame1, 0, 300.0, 500.0, 200.0, 50.0);
        estimator.update(&frame1);

        let mut frame2 = [0.0_f32; OBS_DIM];
        set_floor_slot(&mut frame2, 0, 850.0, 500.0, 40.0, 50.0);
        estimator.update(&frame2);

        assert_eq!(estimator.confidence, 0);
        assert_eq!(estimator.dx, 0.0);
    }

    fn node(x: f32, y: f32) -> LocationNode {
        LocationNode {
            x: (x / 64.0) as i32,
            y: (y / 64.0) as i32,
            terrain: 1,
        }
    }





    #[test]
    fn rejects_low_confidence_large_shift() {
        let mut estimator = VisualMotionEstimator::default();
        let mut frame1 = [0.0_f32; OBS_DIM];
        set_floor_slot(&mut frame1, 0, 300.0, 500.0, 200.0, 50.0);
        set_floor_slot(&mut frame1, 1, 700.0, 460.0, 90.0, 50.0);
        estimator.update(&frame1);

        // 仅 1 个弱匹配却给出 180px 位移 → 应拒绝。
        let mut frame2 = [0.0_f32; OBS_DIM];
        set_floor_slot(&mut frame2, 0, 120.0, 500.0, 40.0, 50.0);
        estimator.update(&frame2);

        assert!(
            estimator.dx.abs() < 20.0,
            "weak match large shift should be rejected, got dx={}",
            estimator.dx
        );
    }


}
