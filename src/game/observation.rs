//! 从 YOLO 检测结果构建 NEAT 视觉观测向量。
//!
//! 每槽固定 4 维几何量 `(Δx/W, Δy/H, w/W, h/H)`，**不含类别 one-hot**。
//! YOLO 类别名仅用于筛选进哪组槽位（敌人/地板/梯绳等），编码内容一律是位置+大小。

use crate::player_name::NamedPlayerHit;
use crate::yolo::Detection;

/// NEAT 训练默认检测置信度阈值（与文档一致）。
pub const NEAT_CONF_THRESH: f32 = 0.70;

/// 观测向量布局（固定长度，不足槽位填 0）。
pub const OBS_SELF: usize = 2;
pub const OBS_FLOOR_SLOTS: usize = 8;
pub const OBS_ENEMY_SLOTS: usize = 6;
pub const OBS_DROP_SLOTS: usize = 4;
pub const OBS_LADDER_SLOTS: usize = 2;
pub const OBS_ROPE_SLOTS: usize = 3;
pub const OBS_SLOT_DIM: usize = 4;
pub const OBS_DIM: usize = OBS_SELF
    + (OBS_FLOOR_SLOTS + OBS_ENEMY_SLOTS + OBS_DROP_SLOTS + OBS_LADDER_SLOTS + OBS_ROPE_SLOTS)
        * OBS_SLOT_DIM;

const FLOOR: &str = "地板";
const LADDER: &str = "梯子";
const ROPE: &str = "绳子";
const ENEMY_LABELS: [&str; 5] = ["花蘑菇", "蓝蜗牛", "绿蜗牛", "红蜗牛", "树怪"];
/// 敌人槽位：五类怪物在 YOLO 侧合并为「敌人」，NEAT 只看框的几何，不区分种类。
const DROP_LABELS: [&str; 2] = ["金币", "药水"];

#[derive(Debug, Clone)]
pub struct VisionObservation {
    pub values: Vec<f32>,
}

impl VisionObservation {
    pub fn zeros() -> Self {
        Self {
            values: vec![0.0; OBS_DIM],
        }
    }

    /// 从**同一次** YOLO 推理结果 + OCR 自身位置构建观测（不再二次推理）。
    ///
    /// 坐标系：以自身脚点为原点；每槽 4 维为相对偏移与框宽高（归一化）。
    /// 敌人/地板/梯子/绳子/掉落均只编码位置+大小，供攻击、逃跑、跳跃、攀爬等决策。
    pub fn from_detections(
        detections: &[Detection],
        self_player: Option<&NamedPlayerHit>,
        img_w: u32,
        img_h: u32,
    ) -> Self {
        let mut obs = Self::zeros();
        let (ax, ay) = self_player
            .map(|p| (p.x, p.y))
            .unwrap_or((img_w as f32 * 0.5, img_h as f32 * 0.5));

        obs.values[0] = ax / img_w as f32;
        obs.values[1] = ay / img_h as f32;

        let mut offset = OBS_SELF;
        offset = fill_nearest_slots(
            &mut obs.values,
            offset,
            detections,
            |d| d.label == FLOOR,
            ax,
            ay,
            img_w,
            img_h,
            OBS_FLOOR_SLOTS,
            SlotAnchor::Center,
        );
        offset = fill_nearest_slots(
            &mut obs.values,
            offset,
            detections,
            |d| ENEMY_LABELS.contains(&d.label),
            ax,
            ay,
            img_w,
            img_h,
            OBS_ENEMY_SLOTS,
            SlotAnchor::Foot,
        );
        offset = fill_nearest_slots(
            &mut obs.values,
            offset,
            detections,
            |d| DROP_LABELS.contains(&d.label),
            ax,
            ay,
            img_w,
            img_h,
            OBS_DROP_SLOTS,
            SlotAnchor::Center,
        );
        offset = fill_nearest_slots(
            &mut obs.values,
            offset,
            detections,
            |d| d.label == LADDER,
            ax,
            ay,
            img_w,
            img_h,
            OBS_LADDER_SLOTS,
            SlotAnchor::Center,
        );
        let _ = fill_nearest_slots(
            &mut obs.values,
            offset,
            detections,
            |d| d.label == ROPE,
            ax,
            ay,
            img_w,
            img_h,
            OBS_ROPE_SLOTS,
            SlotAnchor::Center,
        );

        obs
    }
}

/// 槽位内 w/W 或 h/H 非零视为 YOLO 检出有效框。
pub fn obs_slot_active(values: &[f32], slot_start: usize, slot_count: usize) -> bool {
    for i in 0..slot_count {
        let base = slot_start + i * OBS_SLOT_DIM;
        if values.get(base + 2).copied().unwrap_or(0.0) > 1e-4
            || values.get(base + 3).copied().unwrap_or(0.0) > 1e-4
        {
            return true;
        }
    }
    false
}

pub fn obs_has_enemy(values: &[f32]) -> bool {
    let start = OBS_SELF + OBS_FLOOR_SLOTS * OBS_SLOT_DIM;
    obs_slot_active(values, start, OBS_ENEMY_SLOTS)
}

pub fn obs_has_drop(values: &[f32]) -> bool {
    let start = OBS_SELF + (OBS_FLOOR_SLOTS + OBS_ENEMY_SLOTS) * OBS_SLOT_DIM;
    obs_slot_active(values, start, OBS_DROP_SLOTS)
}

#[derive(Clone, Copy)]
enum SlotAnchor {
    Center,
    Foot,
}

fn fill_nearest_slots(
    values: &mut [f32],
    offset: usize,
    detections: &[Detection],
    pred: impl Fn(&Detection) -> bool,
    ax: f32,
    ay: f32,
    img_w: u32,
    img_h: u32,
    slots: usize,
    anchor: SlotAnchor,
) -> usize {
    let mut picked: Vec<&Detection> = detections.iter().filter(|d| pred(d)).collect();
    picked.sort_by(|a, b| {
        let da = dist2(a, ax, ay, anchor);
        let db = dist2(b, ax, ay, anchor);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut next = offset;
    for det in picked.into_iter().take(slots) {
        encode_slot(values, next, det, ax, ay, img_w, img_h, anchor);
        next += OBS_SLOT_DIM;
    }
    offset + slots * OBS_SLOT_DIM
}

fn dist2(det: &Detection, ax: f32, ay: f32, anchor: SlotAnchor) -> f32 {
    let (cx, cy) = anchor_point(det, anchor);
    let dx = cx - ax;
    let dy = cy - ay;
    dx * dx + dy * dy
}

fn anchor_point(det: &Detection, anchor: SlotAnchor) -> (f32, f32) {
    match anchor {
        SlotAnchor::Center => ((det.x1 + det.x2) * 0.5, (det.y1 + det.y2) * 0.5),
        SlotAnchor::Foot => ((det.x1 + det.x2) * 0.5, det.y2),
    }
}

fn encode_slot(
    values: &mut [f32],
    offset: usize,
    det: &Detection,
    ax: f32,
    ay: f32,
    img_w: u32,
    img_h: u32,
    anchor: SlotAnchor,
) {
    if offset + OBS_SLOT_DIM > values.len() {
        return;
    }
    let (cx, cy) = anchor_point(det, anchor);
    let w = img_w as f32;
    let h = img_h as f32;
    values[offset] = (cx - ax) / w;
    values[offset + 1] = (cy - ay) / h;
    values[offset + 2] = (det.x2 - det.x1) / w;
    values[offset + 3] = (det.y2 - det.y1) / h;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yolo::CLASS_NAMES;

    fn det(class_id: usize, conf: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection {
            class_id,
            label: CLASS_NAMES[class_id],
            conf,
            x1,
            y1,
            x2,
            y2,
        }
    }

    #[test]
    fn obs_dim_fixed() {
        let obs = VisionObservation::from_detections(&[], None, 1368, 768);
        assert_eq!(obs.values.len(), OBS_DIM);
    }

    #[test]
    fn nearest_floor_first() {
        let dets = vec![
            det(0, 0.9, 100.0, 400.0, 200.0, 450.0),
            det(0, 0.9, 500.0, 400.0, 600.0, 450.0),
        ];
        let self_hit = NamedPlayerHit {
            x: 120.0,
            y: 450.0,
            ocr_text: "test".into(),
            match_score: 1.0,
            partial: false,
            player_conf: 0.9,
            roi: (0, 0, 10, 10),
        };
        let obs = VisionObservation::from_detections(&dets, Some(&self_hit), 1368, 768);
        let near_dx = obs.values[OBS_SELF];
        let far_dx = obs.values[OBS_SELF + OBS_SLOT_DIM];
        assert!(near_dx.abs() < far_dx.abs(), "nearest floor should be first slot");
    }
}
