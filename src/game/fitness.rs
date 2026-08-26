//! NEAT 训练计分：YOLO 可见拾取（主分）+ 视觉动作 shaping + 可选内存 shaping。

use crate::game::action::Action;
use super::observation::{obs_has_drop, obs_has_enemy, OBS_DIM};
use crate::yolo::Detection;

use super::types::DropKind;

const MESO_LABEL: &str = "金币";
const POTION_LABEL: &str = "药水";

/// 视觉动作 shaping 分值（与部署观测一致，权重低）。
const PTS_ATTACK_ALIGN: f32 = 1.0;
const PTS_PICKUP_ALIGN: f32 = 1.0;
/// 内存 shaping 原始分（最终 × `memory_weight` 计入总分）。
const PTS_MOB_HIT: f32 = 3.0;
const PTS_MOB_KILL: f32 = 15.0;

/// 训练适应度 shaping 配置（仅影响计分，不进入 NEAT 观测）。
#[derive(Debug, Clone, Copy)]
pub struct FitnessShapingConfig {
    /// 内存事件分（命中/击杀）权重；0=仅视觉拾取+视觉 shaping。
    pub memory_weight: f32,
}

impl Default for FitnessShapingConfig {
    fn default() -> Self {
        Self {
            memory_weight: 0.2,
        }
    }
}

impl FitnessShapingConfig {
    pub fn disabled() -> Self {
        Self {
            memory_weight: 0.0,
        }
    }
}

/// 上一帧 YOLO 可见的掉落框（屏幕坐标 xyxy）。
#[derive(Debug, Clone, Default)]
pub struct VisibleLoot {
    pub meso: Vec<[f32; 4]>,
    pub potions: Vec<[f32; 4]>,
}

/// 训练个体累计得分。
#[derive(Debug, Clone)]
pub struct TrainingFitness {
    /// 总分（拾取 + 视觉 shaping + memory_weight × 内存 shaping）。
    pub score: f32,
    pub pickup_score: f32,
    pub vision_shaping_score: f32,
    pub memory_shaping_score: f32,
    pub meso_events: u32,
    pub potion_events: u32,
    pub meso_units: u32,
    pub attack_align_events: u32,
    pub pickup_align_events: u32,
    pub mob_hit_events: u32,
    pub mob_kill_events: u32,
    shaping: FitnessShapingConfig,
    last_obs: [f32; OBS_DIM],
    last_visible: VisibleLoot,
}

impl Default for TrainingFitness {
    fn default() -> Self {
        Self::with_shaping(FitnessShapingConfig::default())
    }
}

impl TrainingFitness {
    pub fn with_shaping(shaping: FitnessShapingConfig) -> Self {
        Self {
            score: 0.0,
            pickup_score: 0.0,
            vision_shaping_score: 0.0,
            memory_shaping_score: 0.0,
            meso_events: 0,
            potion_events: 0,
            meso_units: 0,
            attack_align_events: 0,
            pickup_align_events: 0,
            mob_hit_events: 0,
            mob_kill_events: 0,
            shaping,
            last_obs: [0.0; OBS_DIM],
            last_visible: VisibleLoot::default(),
        }
    }

    pub fn configure_shaping(&mut self, shaping: FitnessShapingConfig) {
        self.shaping = shaping;
    }

    pub fn shaping_config(&self) -> FitnessShapingConfig {
        self.shaping
    }

    /// 每帧 `perceive` 后调用，记录 YOLO 可见掉落与上一帧观测。
    pub fn record_visible_drops(&mut self, detections: &[Detection]) {
        let mut vis = VisibleLoot::default();
        for d in detections {
            let b = [d.x1, d.y1, d.x2, d.y2];
            match d.label {
                MESO_LABEL => vis.meso.push(b),
                POTION_LABEL => vis.potions.push(b),
                _ => {}
            }
        }
        self.last_visible = vis;
    }

    pub fn set_last_observation(&mut self, obs: &[f32]) {
        let n = obs.len().min(OBS_DIM);
        self.last_obs[..n].copy_from_slice(&obs[..n]);
    }

    /// 本帧动作与观测一致时给小分（Attack+敌人框 / PickUp+掉落框）。
    pub fn try_score_action(&mut self, action: Action) {
        match action {
            Action::Attack if obs_has_enemy(&self.last_obs) => {
                self.vision_shaping_score += PTS_ATTACK_ALIGN;
                self.score += PTS_ATTACK_ALIGN;
                self.attack_align_events += 1;
            }
            Action::PickUp if obs_has_drop(&self.last_obs) => {
                self.vision_shaping_score += PTS_PICKUP_ALIGN;
                self.score += PTS_PICKUP_ALIGN;
                self.pickup_align_events += 1;
            }
            _ => {}
        }
    }

    pub fn record_mob_hit(&mut self) {
        if self.shaping.memory_weight <= 0.0 {
            return;
        }
        self.memory_shaping_score += PTS_MOB_HIT;
        self.score += self.shaping.memory_weight * PTS_MOB_HIT;
        self.mob_hit_events += 1;
    }

    pub fn record_mob_kill(&mut self) {
        if self.shaping.memory_weight <= 0.0 {
            return;
        }
        self.memory_shaping_score += PTS_MOB_KILL;
        self.score += self.shaping.memory_weight * PTS_MOB_KILL;
        self.mob_kill_events += 1;
    }

    /// 拾取成功时调用；仅当掉落物中心落在上一帧 YOLO 框内才加分。
    pub fn try_score_pickup(&mut self, kind: DropKind, x: f32, y: f32, meso_amount: u32) -> f32 {
        let gained = match kind {
            DropKind::Meso => {
                if !point_in_any_box(x, y, &self.last_visible.meso) {
                    return 0.0;
                }
                let pts = meso_amount as f32;
                self.meso_events += 1;
                self.meso_units += meso_amount;
                pts
            }
            DropKind::RedPotion => {
                if !point_in_any_box(x, y, &self.last_visible.potions) {
                    return 0.0;
                }
                self.potion_events += 1;
                50.0
            }
        };
        self.pickup_score += gained;
        self.score += gained;
        gained
    }

    pub fn reset(&mut self) {
        let shaping = self.shaping;
        *self = Self::with_shaping(shaping);
    }
}

fn point_in_any_box(x: f32, y: f32, boxes: &[[f32; 4]]) -> bool {
    boxes.iter().any(|b| x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::observation::{OBS_DROP_SLOTS, OBS_ENEMY_SLOTS, OBS_FLOOR_SLOTS, OBS_SELF, OBS_SLOT_DIM};
    use crate::yolo::CLASS_NAMES;
    use crate::yolo::Detection;

    fn det(class_id: usize, x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection {
            class_id,
            label: CLASS_NAMES[class_id],
            conf: 0.9,
            x1,
            y1,
            x2,
            y2,
        }
    }

    fn obs_with_enemy() -> [f32; OBS_DIM] {
        let mut v = [0.0_f32; OBS_DIM];
        let base = OBS_SELF + OBS_FLOOR_SLOTS * OBS_SLOT_DIM;
        v[base + 2] = 0.05;
        v[base + 3] = 0.05;
        v
    }

    fn obs_with_drop() -> [f32; OBS_DIM] {
        let mut v = [0.0_f32; OBS_DIM];
        let base = OBS_SELF
            + (OBS_FLOOR_SLOTS + OBS_ENEMY_SLOTS) * OBS_SLOT_DIM;
        v[base + 2] = 0.04;
        v[base + 3] = 0.04;
        let _ = OBS_DROP_SLOTS;
        v
    }

    #[test]
    fn scores_only_yolo_visible_meso() {
        let mut f = TrainingFitness::default();
        f.record_visible_drops(&[det(11, 100.0, 200.0, 130.0, 230.0)]);
        let g1 = f.try_score_pickup(DropKind::Meso, 115.0, 215.0, 3);
        assert!(g1 > 0.0);
        let g2 = f.try_score_pickup(DropKind::Meso, 500.0, 500.0, 3);
        assert_eq!(g2, 0.0);
    }

    #[test]
    fn vision_shaping_attack_when_enemy_in_obs() {
        let mut f = TrainingFitness::default();
        f.set_last_observation(&obs_with_enemy());
        f.try_score_action(Action::Attack);
        assert!(f.vision_shaping_score >= PTS_ATTACK_ALIGN);
        assert!(f.score >= PTS_ATTACK_ALIGN);
    }

    #[test]
    fn memory_shaping_scaled_by_weight() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig {
            memory_weight: 0.5,
        });
        f.record_mob_kill();
        assert!((f.score - 0.5 * PTS_MOB_KILL).abs() < 1e-3);
        assert_eq!(f.mob_kill_events, 1);
    }

    #[test]
    fn memory_shaping_off_when_weight_zero() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        f.record_mob_hit();
        assert_eq!(f.score, 0.0);
    }
}
