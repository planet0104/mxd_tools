//! 从 YOLO 检测结果构建视觉观测向量。
//!
//! 每槽固定 4 维几何量 `(Δx/W, Δy/H, w/W, h/H)`，**不含类别 one-hot**。
//! YOLO 类别名仅用于筛选进哪组槽位（敌人/地板/梯绳等），编码内容一律是位置+大小。

use crate::player_name::NamedPlayerHit;
use crate::yolo::Detection;

/// YOLO 检测置信度阈值。
pub const VISION_CONF_THRESH: f32 = 0.55;

/// 观测向量布局（固定长度，不足槽位填 0）。
pub const OBS_SELF: usize = 2;
pub const OBS_FLOOR_SLOTS: usize = 8;
pub const OBS_ENEMY_SLOTS: usize = 6;
pub const OBS_DROP_SLOTS: usize = 4;
pub const OBS_LADDER_SLOTS: usize = 2;
pub const OBS_ROPE_SLOTS: usize = 3;
pub const OBS_SLOT_DIM: usize = 4;
/// 物理前方同层可走（右、左），与 YOLO 无关；训练/预览一致，专治顶边站桩。
pub const OBS_PHYSICS: usize = 2;
pub const OBS_DIM: usize = OBS_SELF
    + (OBS_FLOOR_SLOTS + OBS_ENEMY_SLOTS + OBS_DROP_SLOTS + OBS_LADDER_SLOTS + OBS_ROPE_SLOTS)
        * OBS_SLOT_DIM
    + OBS_PHYSICS;

/// 各组槽位在观测向量中的起始下标。
pub const OBS_FLOOR_START: usize = OBS_SELF;
pub const OBS_ENEMY_START: usize = OBS_SELF + OBS_FLOOR_SLOTS * OBS_SLOT_DIM;
pub const OBS_DROP_START: usize = OBS_ENEMY_START + OBS_ENEMY_SLOTS * OBS_SLOT_DIM;
pub const OBS_LADDER_START: usize = OBS_DROP_START + OBS_DROP_SLOTS * OBS_SLOT_DIM;
pub const OBS_ROPE_START: usize = OBS_LADDER_START + OBS_LADDER_SLOTS * OBS_SLOT_DIM;
pub const OBS_PHYSICS_START: usize = OBS_ROPE_START + OBS_ROPE_SLOTS * OBS_SLOT_DIM;

const FLOOR: &str = "地板";
const LADDER: &str = "梯子";
const ROPE: &str = "绳子";
const ENEMY_LABELS: [&str; 5] = ["花蘑菇", "蓝蜗牛", "绿蜗牛", "红蜗牛", "树怪"];
/// 敌人槽位：五类怪物在 YOLO 侧合并为「敌人」，只编码框的几何，不区分种类。
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
    obs_slot_active(values, OBS_ENEMY_START, OBS_ENEMY_SLOTS)
}

/// 敌人槽位是否与玩家在同一条可普攻的平台（按脚点垂直差判定）。
pub fn obs_enemy_same_level(values: &[f32], slot_index: usize) -> bool {
    if slot_index >= OBS_ENEMY_SLOTS {
        return false;
    }
    enemy_slot_same_level(values, OBS_ENEMY_START + slot_index * OBS_SLOT_DIM)
}

fn enemy_slot_same_level(values: &[f32], base: usize) -> bool {
    let Some((_, dy, _, _)) = read_slot(values, base) else {
        return false;
    };
    dy.abs() <= ENEMY_SAME_LEVEL_DY
}

/// 视野内是否存在与玩家同层的敌人。
pub fn obs_has_same_level_enemy(values: &[f32]) -> bool {
    for i in 0..OBS_ENEMY_SLOTS {
        let base = OBS_ENEMY_START + i * OBS_SLOT_DIM;
        if obs_slot_active(values, base, 1) && enemy_slot_same_level(values, base) {
            return true;
        }
    }
    false
}

/// 与 `sim::try_attack_mobs` 攻击框对齐的归一化距离阈值（1368×768 视口）。
const ATTACK_GATE_DX_FWD: f32 = 90.0 / 1368.0;
const ATTACK_GATE_DX_BACK: f32 = 8.0 / 1368.0;
const ATTACK_GATE_DY: f32 = 80.0 / 768.0;
/// 敌人与玩家同层才可普攻；对齐 sim 攻击框并留 YOLO 容差。
pub const ENEMY_SAME_LEVEL_DY: f32 = 70.0 / 768.0;

/// 与 `sim::check_mob_touch` 对齐的玩家接触盒（归一化，以自身脚点为原点）。
const PLAYER_CONTACT_HALF_W: f32 = 28.0 / 1368.0;
const PLAYER_CONTACT_HALF_H: f32 = 36.0 / 768.0;
/// YOLO 框偏差：略放大接触判定，避免「视觉上已贴脸但未重叠」。
const CONTACT_YOLO_PAD_X: f32 = 12.0 / 1368.0;
const CONTACT_YOLO_PAD_Y: f32 = 10.0 / 768.0;
const CONTACT_SIDE_DX: f32 = 6.0 / 1368.0;

/// 敌人 YOLO 框与玩家接触盒重叠统计（用于贴身躲怪，非远距计数）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EnemyContactAssessment {
    pub left: u32,
    pub right: u32,
    pub total: u32,
}

pub fn obs_enemy_touching_player(values: &[f32]) -> bool {
    obs_assess_enemy_contact(values).total > 0
}

/// 统计与玩家接触盒重叠的敌人（左右侧各几只）。
pub fn obs_assess_enemy_contact(values: &[f32]) -> EnemyContactAssessment {
    let mut out = EnemyContactAssessment::default();
    for i in 0..OBS_ENEMY_SLOTS {
        let base = OBS_ENEMY_START + i * OBS_SLOT_DIM;
        if !enemy_slot_overlaps_player_contact(values, base) {
            continue;
        }
        out.total += 1;
        let dx = values[base];
        if dx < -CONTACT_SIDE_DX {
            out.left += 1;
        } else if dx > CONTACT_SIDE_DX {
            out.right += 1;
        } else {
            out.left += 1;
            out.right += 1;
        }
    }
    out
}

fn enemy_slot_overlaps_player_contact(values: &[f32], base: usize) -> bool {
    let Some((dx, dy, w, h)) = read_slot(values, base) else {
        return false;
    };
    let pl = -PLAYER_CONTACT_HALF_W - CONTACT_YOLO_PAD_X;
    let pr = PLAYER_CONTACT_HALF_W + CONTACT_YOLO_PAD_X;
    let pt = -PLAYER_CONTACT_HALF_H - CONTACT_YOLO_PAD_Y;
    let pb = PLAYER_CONTACT_HALF_H + CONTACT_YOLO_PAD_Y;
    let el = dx - w * 0.5;
    let er = dx + w * 0.5;
    let et = dy - h;
    let eb = dy;
    pl < er && pr > el && pt < eb && pb > et
}

/// 行走：前方同层地板水平/垂直容差（归一化）。
/// 注意：真实 YOLO 地板常是大框，框心在脚下；门控必须看**框体覆盖**而非框心是否落在前方带。
const WALK_FLOOR_DX_MIN: f32 = 8.0 / 1368.0;
const WALK_FLOOR_DX_MAX: f32 = 160.0 / 1368.0;
/// YOLO 地板框心常偏下，略放宽垂直容差。
const WALK_FLOOR_DY: f32 = 90.0 / 768.0;
/// 脚下方/platform 判定。
const FLOOR_UNDER_DX: f32 = 48.0 / 1368.0;
const FLOOR_UNDER_DY: f32 = 60.0 / 768.0;
/// 梯/绳抓取范围。
const CLIMB_GATE_DX: f32 = 28.0 / 1368.0;
const CLIMB_GATE_DY: f32 = 110.0 / 768.0;

fn read_slot(values: &[f32], base: usize) -> Option<(f32, f32, f32, f32)> {
    if base + OBS_SLOT_DIM > values.len() {
        return None;
    }
    if values[base + 2].abs() <= 1e-4 && values[base + 3].abs() <= 1e-4 {
        return None;
    }
    Some((
        values[base],
        values[base + 1],
        values[base + 2],
        values[base + 3],
    ))
}

pub fn obs_has_floor_signal(values: &[f32]) -> bool {
    obs_slot_active(values, OBS_FLOOR_START, OBS_FLOOR_SLOTS)
}

pub fn obs_has_ladder_or_rope_signal(values: &[f32]) -> bool {
    obs_slot_active(values, OBS_LADDER_START, OBS_LADDER_SLOTS)
        || obs_slot_active(values, OBS_ROPE_START, OBS_ROPE_SLOTS)
}

/// 指定方向前方是否有可行走的地板（YOLO 地板槽）。
///
/// 用框的水平覆盖 `[dx±w/2]` 与前方走廊相交判定，避免「大平台框心在脚下 → 永远判无前方地板」。
pub fn obs_floor_ahead(values: &[f32], direction: f32) -> bool {
    if direction.abs() <= f32::EPSILON {
        return false;
    }
    let right = direction > 0.0;
    for i in 0..OBS_FLOOR_SLOTS {
        let base = OBS_FLOOR_START + i * OBS_SLOT_DIM;
        let Some((dx, dy, w, _)) = read_slot(values, base) else {
            continue;
        };
        if dy.abs() > WALK_FLOOR_DY {
            continue;
        }
        let half = (w * 0.5).max(0.0);
        let box_l = dx - half;
        let box_r = dx + half;
        let overlaps = if right {
            box_r >= WALK_FLOOR_DX_MIN && box_l <= WALK_FLOOR_DX_MAX
        } else {
            box_l <= -WALK_FLOOR_DX_MIN && box_r >= -WALK_FLOOR_DX_MAX
        };
        if overlaps {
            return true;
        }
    }
    false
}

pub fn obs_floor_underfoot(values: &[f32]) -> bool {
    for i in 0..OBS_FLOOR_SLOTS {
        let base = OBS_FLOOR_START + i * OBS_SLOT_DIM;
        let Some((dx, dy, w, _)) = read_slot(values, base) else {
            continue;
        };
        if dy.abs() > FLOOR_UNDER_DY {
            continue;
        }
        let half = (w * 0.5).max(0.0);
        // 脚下：框覆盖 x≈0，或框心足够近。
        if box_l_r_covers_zero(dx, half) || dx.abs() <= FLOOR_UNDER_DX {
            return true;
        }
    }
    false
}

fn box_l_r_covers_zero(dx: f32, half: f32) -> bool {
    let box_l = dx - half;
    let box_r = dx + half;
    box_l <= 0.0 && box_r >= 0.0
}

fn obs_ladder_or_rope_near(values: &[f32]) -> bool {
    for start in [OBS_LADDER_START, OBS_ROPE_START] {
        let count = if start == OBS_LADDER_START {
            OBS_LADDER_SLOTS
        } else {
            OBS_ROPE_SLOTS
        };
        for i in 0..count {
            let base = start + i * OBS_SLOT_DIM;
            let Some((dx, dy, _, _)) = read_slot(values, base) else {
                continue;
            };
            if dx.abs() <= CLIMB_GATE_DX && dy.abs() <= CLIMB_GATE_DY {
                return true;
            }
        }
    }
    false
}

/// 站在平台边缘：脚下有地板但行进方向前方无地板。
pub fn obs_platform_edge(values: &[f32], facing: f32) -> bool {
    obs_floor_underfoot(values) && !obs_floor_ahead(values, facing.signum())
}

/// 允许跳跃：仅平台边缘（行进方向前方无地板）。上方/backdrop 平台不算。
pub fn obs_jump_allowed(values: &[f32], facing: f32, climbing: bool) -> bool {
    if climbing {
        return true;
    }
    obs_platform_edge(values, facing)
}

/// 绳/梯已对齐到可抓取距离（才应 jump+up，避免远处绳梯误跳）。
pub fn obs_climb_grab_ready(values: &[f32]) -> bool {
    const GRAB_DX: f32 = 20.0 / 1368.0;
    const GRAB_DY: f32 = 90.0 / 768.0;
    for start in [OBS_LADDER_START, OBS_ROPE_START] {
        let count = if start == OBS_LADDER_START {
            OBS_LADDER_SLOTS
        } else {
            OBS_ROPE_SLOTS
        };
        for i in 0..count {
            let base = start + i * OBS_SLOT_DIM;
            let Some((dx, dy, _, _)) = read_slot(values, base) else {
                continue;
            };
            if dx.abs() <= GRAB_DX && dy.abs() <= GRAB_DY {
                return true;
            }
        }
    }
    false
}

/// 允许 up/down：近梯绳（抓绳）或已在攀爬中由 sim 处理。
pub fn obs_vertical_nav_allowed(values: &[f32], climbing: bool) -> bool {
    climbing || obs_ladder_or_rope_near(values)
}

/// 同层敌人是否在攻击距离内（需结合朝向：敌人在身前且够近）。
pub fn obs_enemy_in_attack_range(values: &[f32], facing: f32) -> bool {
    let facing_right = facing >= 0.0;
    for i in 0..OBS_ENEMY_SLOTS {
        let base = OBS_ENEMY_START + i * OBS_SLOT_DIM;
        if !obs_slot_active(values, base, 1) || !enemy_slot_same_level(values, base) {
            continue;
        }
        let dx = values[base];
        let dy = values[base + 1];
        let horiz_ok = if facing_right {
            dx >= -ATTACK_GATE_DX_BACK && dx <= ATTACK_GATE_DX_FWD
        } else {
            dx <= ATTACK_GATE_DX_BACK && dx >= -ATTACK_GATE_DX_FWD
        };
        if horiz_ok && dy.abs() <= ATTACK_GATE_DY {
            return true;
        }
    }
    false
}

pub fn obs_has_drop(values: &[f32]) -> bool {
    obs_slot_active(values, OBS_DROP_START, OBS_DROP_SLOTS)
}

/// 掉落槽是否在拾取半径内（与 `sim::tick_drops` 约 40px 对齐，略放宽以容 YOLO 框心偏差）。
pub fn obs_drop_in_pickup_range(values: &[f32]) -> bool {
    const PICKUP_GATE_DX: f32 = 56.0 / 1368.0;
    const PICKUP_GATE_DY: f32 = 56.0 / 768.0;
    for i in 0..OBS_DROP_SLOTS {
        let base = OBS_DROP_START + i * OBS_SLOT_DIM;
        let Some((dx, dy, _, _)) = read_slot(values, base) else {
            continue;
        };
        let nx = dx / PICKUP_GATE_DX;
        let ny = dy / PICKUP_GATE_DY;
        if nx * nx + ny * ny <= 1.0 {
            return true;
        }
    }
    false
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

/// 写入物理「右/左前方同层可走」标志（1=可走，0=悬崖/挡墙）。
pub fn inject_physics_walk_flags(values: &mut [f32], right_ok: Option<bool>, left_ok: Option<bool>) {
    if values.len() < OBS_DIM {
        return;
    }
    values[OBS_PHYSICS_START] = if right_ok.unwrap_or(true) { 1.0 } else { 0.0 };
    values[OBS_PHYSICS_START + 1] = if left_ok.unwrap_or(true) { 1.0 } else { 0.0 };
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

    #[test]
    fn attack_range_requires_same_level_enemy() {
        let mut v = [0.0_f32; OBS_DIM];
        v[OBS_ENEMY_START + 2] = 0.05;
        v[OBS_ENEMY_START + 3] = 0.05;
        v[OBS_ENEMY_START] = 0.03;
        v[OBS_ENEMY_START + 1] = 0.0;
        assert!(obs_enemy_in_attack_range(&v, 1.0));
        v[OBS_ENEMY_START + 1] = 0.45;
        assert!(!obs_enemy_in_attack_range(&v, 1.0));
        assert!(!obs_has_same_level_enemy(&v));
    }

    #[test]
    fn attack_range_requires_near_enemy_ahead() {
        let mut v = [0.0_f32; OBS_DIM];
        v[OBS_ENEMY_START + 2] = 0.05;
        v[OBS_ENEMY_START + 3] = 0.05;
        v[OBS_ENEMY_START] = 0.03;
        assert!(obs_enemy_in_attack_range(&v, 1.0));
        v[OBS_ENEMY_START] = 0.2;
        assert!(!obs_enemy_in_attack_range(&v, 1.0));
        v[OBS_ENEMY_START] = -0.03;
        assert!(obs_enemy_in_attack_range(&v, -1.0));
        v[OBS_ENEMY_START] = 0.03;
        assert!(!obs_enemy_in_attack_range(&v, -1.0));
    }

    #[test]
    fn walk_requires_floor_ahead() {
        let mut v = [0.0_f32; OBS_DIM];
        v[OBS_FLOOR_START + 2] = 0.08;
        v[OBS_FLOOR_START + 3] = 0.02;
        v[OBS_FLOOR_START] = 0.04;
        assert!(obs_floor_ahead(&v, 1.0));
        assert!(!obs_floor_ahead(&v, -1.0));
        v[OBS_FLOOR_START] = -0.04;
        assert!(obs_floor_ahead(&v, -1.0));
    }

    #[test]
    fn walk_ahead_accepts_wide_floor_centered_underfoot() {
        // 真实 YOLO：大平台框心在脚下，但框体向右延伸 → 应允许右走。
        let mut v = [0.0_f32; OBS_DIM];
        v[OBS_FLOOR_START] = 0.0; // center under feet
        v[OBS_FLOOR_START + 1] = 0.02;
        v[OBS_FLOOR_START + 2] = 400.0 / 1368.0; // wide
        v[OBS_FLOOR_START + 3] = 0.03;
        assert!(obs_floor_underfoot(&v));
        assert!(obs_floor_ahead(&v, 1.0));
        assert!(obs_floor_ahead(&v, -1.0));
    }

    #[test]
    fn walk_ahead_rejects_true_platform_edge() {
        // 框只覆盖脚下偏左，右前方无覆盖 → 右走应判无地板。
        let mut v = [0.0_f32; OBS_DIM];
        v[OBS_FLOOR_START] = -20.0 / 1368.0;
        v[OBS_FLOOR_START + 1] = 0.01;
        v[OBS_FLOOR_START + 2] = 40.0 / 1368.0;
        v[OBS_FLOOR_START + 3] = 0.02;
        assert!(obs_floor_underfoot(&v));
        assert!(!obs_floor_ahead(&v, 1.0));
        assert!(obs_floor_ahead(&v, -1.0));
    }

    #[test]
    fn jump_not_allowed_for_unreachable_floor_above() {
        let mut v = [0.0_f32; OBS_DIM];
        v[OBS_FLOOR_START] = 0.05;
        v[OBS_FLOOR_START + 1] = -0.12;
        v[OBS_FLOOR_START + 2] = 0.08;
        v[OBS_FLOOR_START + 3] = 0.03;
        v[OBS_FLOOR_START + OBS_SLOT_DIM] = 0.0;
        v[OBS_FLOOR_START + OBS_SLOT_DIM + 1] = 0.02;
        v[OBS_FLOOR_START + OBS_SLOT_DIM + 2] = 0.3;
        v[OBS_FLOOR_START + OBS_SLOT_DIM + 3] = 0.03;
        assert!(obs_floor_underfoot(&v));
        assert!(obs_floor_ahead(&v, 1.0));
        assert!(!obs_jump_allowed(&v, 1.0, false));
    }

    #[test]
    fn climb_grab_ready_requires_alignment() {
        let mut v = [0.0_f32; OBS_DIM];
        v[OBS_ROPE_START] = 0.08;
        v[OBS_ROPE_START + 1] = -0.05;
        v[OBS_ROPE_START + 2] = 0.02;
        v[OBS_ROPE_START + 3] = 0.15;
        assert!(!obs_climb_grab_ready(&v));
        v[OBS_ROPE_START] = 0.01;
        v[OBS_ROPE_START + 1] = 0.02;
        assert!(obs_climb_grab_ready(&v));
    }

    #[test]
    fn jump_allowed_at_platform_edge() {
        let mut v = [0.0_f32; OBS_DIM];
        // 窄地板仅盖住脚下，不延伸到前方走廊 → 平台边，允许跳。
        v[OBS_FLOOR_START] = 0.0;
        v[OBS_FLOOR_START + 2] = 8.0 / 1368.0;
        v[OBS_FLOOR_START + 3] = 0.02;
        assert!(obs_floor_underfoot(&v));
        assert!(!obs_floor_ahead(&v, 1.0));
        assert!(obs_jump_allowed(&v, 1.0, false));
        // 宽地板延伸到前方 → 非边缘，禁止无意义跳。
        v[OBS_FLOOR_START + 2] = 200.0 / 1368.0;
        assert!(obs_floor_ahead(&v, 1.0));
        assert!(!obs_jump_allowed(&v, 1.0, false));
    }

    #[test]
    fn drop_pickup_range_near_vs_far() {
        let mut near = [0.0_f32; OBS_DIM];
        near[OBS_DROP_START + 2] = 0.04;
        near[OBS_DROP_START + 3] = 0.04;
        assert!(obs_has_drop(&near));
        assert!(obs_drop_in_pickup_range(&near));

        let mut far = near;
        far[OBS_DROP_START] = 200.0 / 1368.0;
        assert!(obs_has_drop(&far));
        assert!(!obs_drop_in_pickup_range(&far));
    }

    #[test]
    fn enemy_contact_requires_bbox_overlap_not_just_nearby() {
        let mut touching = [0.0_f32; OBS_DIM];
        touching[OBS_ENEMY_START] = 0.018;
        touching[OBS_ENEMY_START + 1] = 0.0;
        touching[OBS_ENEMY_START + 2] = 0.05;
        touching[OBS_ENEMY_START + 3] = 0.07;
        assert!(obs_enemy_touching_player(&touching));
        let c = obs_assess_enemy_contact(&touching);
        assert_eq!(c.total, 1);
        assert_eq!(c.right, 1);

        let mut distant = touching;
        distant[OBS_ENEMY_START] = 0.10;
        assert!(!obs_enemy_touching_player(&distant));
        assert_eq!(obs_assess_enemy_contact(&distant).total, 0);
    }
}
