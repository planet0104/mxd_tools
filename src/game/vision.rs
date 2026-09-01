//! 视觉单步：YOLO+OCR → 观测向量。

use std::path::Path;

use anyhow::Result;
use image::RgbImage;

use crate::player_name::{self, NamedPlayerHit};
use crate::yolo::{Detection, YoloDetector, YoloDevice};

use super::config::VisionAnchorConfig;
use super::observation::VisionObservation;
use super::self_anchor::apply_anchor_jitter;
use super::{DEFAULT_PLAYER_NAME, WINDOW_H, WINDOW_W};

/// 训练 YOLO 模式：主线程随帧传入的 sim 快照（OCR 脚点抖动用 episode_seed）。
#[derive(Debug, Clone, Copy)]
pub struct SimVisionSnapshot {
    pub player_x: f32,
    pub player_y: f32,
    pub cam_x: f32,
    pub cam_y: f32,
    pub episode_seed: u64,
}

/// 过滤 YOLO 检测，仅保留 `conf >= min_conf`。
pub fn filter_detections(detections: Vec<Detection>, min_conf: f32) -> Vec<Detection> {
    detections
        .into_iter()
        .filter(|d| d.conf >= min_conf)
        .collect()
}

/// 视觉推理管线（YOLO + OCR + 观测编码）。
pub struct VisionPipeline {
    detector: YoloDetector,
    target_name: String,
    conf_thresh: f32,
    anchor: VisionAnchorConfig,
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
            anchor: VisionAnchorConfig::default(),
        })
    }

    pub fn with_anchor(mut self, anchor: VisionAnchorConfig) -> Self {
        self.anchor = anchor;
        self
    }

    pub fn anchor_config(&self) -> VisionAnchorConfig {
        self.anchor
    }

    pub fn with_target_name(mut self, name: impl Into<String>) -> Self {
        self.target_name = name.into();
        self
    }

    pub fn conf_thresh(&self) -> f32 {
        self.conf_thresh
    }

    /// 对一帧 RGB 做 YOLO+OCR 感知。
    pub fn perceive(&mut self, frame: &RgbImage) -> Result<VisionStep> {
        self.perceive_with_snapshot(frame, None)
    }

    pub fn perceive_with_snapshot(
        &mut self,
        frame: &RgbImage,
        sim_snapshot: Option<SimVisionSnapshot>,
    ) -> Result<VisionStep> {
        let detections = self.detect_frame(frame)?;
        self.build_agent_step(frame, &detections, &self.target_name, sim_snapshot)
    }

    pub fn detect_frame(&mut self, frame: &RgbImage) -> Result<Vec<Detection>> {
        let w = frame.width();
        let h = frame.height();
        let raw = self.detector.detect_rgb8(w, h, frame.as_raw())?;
        Ok(filter_detections(raw, self.conf_thresh))
    }

    pub fn build_agent_step(
        &self,
        frame: &RgbImage,
        detections: &[Detection],
        target_name: &str,
        sim_snapshot: Option<SimVisionSnapshot>,
    ) -> Result<VisionStep> {
        let w = frame.width();
        let h = frame.height();
        let self_player = self.resolve_self_player(frame, detections, target_name, sim_snapshot)?;
        let observation =
            VisionObservation::from_detections(detections, self_player.as_ref(), w, h);
        Ok(VisionStep {
            detections: detections.to_vec(),
            self_player,
            observation,
        })
    }

    fn resolve_self_player(
        &self,
        frame: &RgbImage,
        detections: &[Detection],
        target_name: &str,
        sim_snapshot: Option<SimVisionSnapshot>,
    ) -> Result<Option<NamedPlayerHit>> {
        let (mut self_player, _) = player_name::find_named_player_verbose(
            frame,
            detections,
            target_name,
            self.conf_thresh,
            false,
        )?;
        if self.anchor.uses_anchor_jitter() {
            if let (Some(hit), Some(snap)) = (&mut self_player, sim_snapshot) {
                apply_anchor_jitter(hit, snap.episode_seed, self.anchor.anchor_jitter_px);
            }
        }
        Ok(self_player)
    }
}

#[derive(Debug, Clone)]
pub struct VisionStep {
    pub detections: Vec<Detection>,
    pub self_player: Option<NamedPlayerHit>,
    pub observation: VisionObservation,
}

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
    use crate::game::VISION_CONF_THRESH;
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
        let out = filter_detections(dets, VISION_CONF_THRESH);
        assert_eq!(out.len(), 1);
        assert!(out[0].conf >= VISION_CONF_THRESH);
    }
}
