//! 视觉训练单步：每帧**一次** YOLO（+ 可选 OCR）→ 检测列表 + 自身玩家 + NEAT 观测（无磁盘 I/O）。

use std::path::Path;

use anyhow::Result;
use image::RgbImage;

use crate::player_name::{self, NamedPlayerHit};
use crate::yolo::{Detection, YoloDetector, YoloDevice};

use super::config::{VisionAnchorConfig, VisionAnchorMode};
use super::observation::VisionObservation;
use super::self_anchor::{episode_anchor_offset, find_self_player_by_sim};
use super::GameSim;
use super::{DEFAULT_PLAYER_NAME, WINDOW_H, WINDOW_W};

/// 训练 SimMatch 模式：主线程随帧传入的 sim 快照。
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

/// 视觉推理管线（YOLO + OCR + 观测编码），全程内存操作。
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

    /// 对一帧 RGB 做**单次**视觉感知（训练每逻辑帧调用一次）。
    ///
    /// 流程：`detect_rgb8`（一次 YOLO）→ 置信度过滤 → 自身锚点（OCR 或 SimMatch）
    /// → `VisionObservation::from_detections`（纯内存编码，不再调用模型）。
    pub fn perceive(&mut self, frame: &RgbImage) -> Result<VisionStep> {
        self.perceive_with_snapshot(frame, None)
    }

    /// 带 sim 快照的感知；SimMatch 模式下必须传入 `Some(snapshot)`。
    pub fn perceive_with_snapshot(
        &mut self,
        frame: &RgbImage,
        sim_snapshot: Option<SimVisionSnapshot>,
    ) -> Result<VisionStep> {
        let detections = self.detect_frame(frame)?;
        self.build_agent_step(frame, &detections, &self.target_name, sim_snapshot)
    }

    /// 一次 YOLO 推理 + 置信度过滤。
    pub fn detect_frame(&mut self, frame: &RgbImage) -> Result<Vec<Detection>> {
        let w = frame.width();
        let h = frame.height();
        let raw = self.detector.detect_rgb8(w, h, frame.as_raw())?;
        Ok(filter_detections(raw, self.conf_thresh))
    }

    /// 在已有检测框上定位自身锚点并编码观测（不再调用 YOLO）。
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
        let observation = VisionObservation::from_detections(
            detections,
            self_player.as_ref(),
            w,
            h,
        );
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
        match self.anchor.mode {
            VisionAnchorMode::Ocr => {
                let (self_player, _) = player_name::find_named_player_verbose(
                    frame,
                    detections,
                    target_name,
                    self.conf_thresh,
                    false,
                )?;
                Ok(self_player)
            }
            VisionAnchorMode::SimMatch => {
                let Some(snap) = sim_snapshot else {
                    anyhow::bail!("SimMatch 模式需要 SimVisionSnapshot");
                };
                let (ox, oy) = episode_anchor_offset(snap.episode_seed, self.anchor.sim_offset_px);
                Ok(find_self_player_by_sim(
                    detections,
                    snap.player_x,
                    snap.player_y,
                    snap.cam_x,
                    snap.cam_y,
                    self.conf_thresh,
                    ox,
                    oy,
                ))
            }
        }
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
