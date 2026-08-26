//! 视觉训练单步：每帧**一次** YOLO + OCR → 检测列表 + 自身玩家 + NEAT 观测（无磁盘 I/O）。

use std::path::Path;

use anyhow::Result;
use image::RgbImage;

use crate::player_name::{self, NamedPlayerHit};
use crate::yolo::{Detection, YoloDetector, YoloDevice};

use super::observation::VisionObservation;
use super::GameSim;
use super::{DEFAULT_PLAYER_NAME, WINDOW_H, WINDOW_W};

/// 过滤 YOLO 检测，仅保留 `conf >= min_conf`。
pub fn filter_detections(detections: Vec<Detection>, min_conf: f32) -> Vec<Detection> {
    detections
        .into_iter()
        .filter(|d| d.conf >= min_conf)
        .collect()
}

/// 视觉推理管线（YOLO + OCR + 观测编码），全程内存操作。
pub struct VisionPipeline {
    detector: YoloDetector,
    target_name: String,
    conf_thresh: f32,
}

impl VisionPipeline {
    pub fn load(model: &Path, device: YoloDevice, conf_thresh: f32) -> Result<Self> {
        let mut detector =
            YoloDetector::load_with_thresholds(model, device, conf_thresh, 0.7, 640)?;
        detector.set_thresholds(conf_thresh, 0.7);
        Ok(Self {
            detector,
            target_name: DEFAULT_PLAYER_NAME.to_string(),
            conf_thresh,
        })
    }

    pub fn with_target_name(mut self, name: impl Into<String>) -> Self {
        self.target_name = name.into();
        self
    }

    pub fn conf_thresh(&self) -> f32 {
        self.conf_thresh
    }

    /// 对一帧 RGB 做**单次**视觉感知（训练每逻辑帧调用一次）。
    ///
    /// 流程：`detect_rgb8`（一次 YOLO）→ 置信度过滤 → OCR 名牌（仅玩家框）
    /// → `VisionObservation::from_detections`（纯内存编码，不再调用模型）。
    pub fn perceive(&mut self, frame: &RgbImage) -> Result<VisionStep> {
        let w = frame.width();
        let h = frame.height();
        let raw = self.detector.detect_rgb8(w, h, frame.as_raw())?;
        let detections = filter_detections(raw, self.conf_thresh);
        let (self_player, _) = player_name::find_named_player_verbose(
            frame,
            &detections,
            &self.target_name,
            self.conf_thresh,
            false,
        )?;
        let observation = VisionObservation::from_detections(
            &detections,
            self_player.as_ref(),
            w,
            h,
        );
        Ok(VisionStep {
            detections,
            self_player,
            observation,
        })
    }
}

#[derive(Debug, Clone)]
pub struct VisionStep {
    pub detections: Vec<Detection>,
    pub self_player: Option<NamedPlayerHit>,
    pub observation: VisionObservation,
}

impl VisionStep {
    /// 将本帧 YOLO 可见掉落与观测写入 `GameSim`（训练计分 / shaping）。
    pub fn apply_fitness_hints(&self, sim: &mut GameSim) {
        sim.record_vision_loot(&self.detections);
        if sim.config.training {
            sim.fitness
                .set_last_observation(&self.observation.values);
        }
    }
}

/// 校验帧尺寸是否为训练标准分辨率。
pub fn assert_training_frame(frame: &RgbImage) -> Result<()> {
    if frame.width() != WINDOW_W as u32 || frame.height() != WINDOW_H as u32 {
        anyhow::bail!(
            "帧尺寸应为 {}x{}，实际 {}x{}",
            WINDOW_W,
            WINDOW_H,
            frame.width(),
            frame.height()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::NEAT_CONF_THRESH;
    use crate::yolo::CLASS_NAMES;

    #[test]
    fn filter_drops_low_conf() {
        let dets = vec![
            Detection {
                class_id: 0,
                label: CLASS_NAMES[0],
                conf: 0.5,
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            },
            Detection {
                class_id: 0,
                label: CLASS_NAMES[0],
                conf: 0.8,
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            },
        ];
        let out = filter_detections(dets, NEAT_CONF_THRESH);
        assert_eq!(out.len(), 1);
        assert!(out[0].conf >= NEAT_CONF_THRESH);
    }
}
