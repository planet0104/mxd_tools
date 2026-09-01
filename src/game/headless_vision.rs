//! Headless 离屏渲染 + YOLO/OCR 观测（与 `game_preview` 对齐）。
//!
//! - GL 渲染：主线程
//! - YOLO+OCR：独立线程，按真实 ONNX 耗时推理
//! - `schedule_capture_if_ready` + `poll_observation`：非阻塞，YOLO 慢时不刷队列

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use image::RgbImage;
use macroquad::prelude::*;

use crate::yolo::YoloDevice;

use super::config::VisionAnchorConfig;
use super::observation::OBS_DIM;
use super::sim::GameSim;
use super::view::{self, GameViewAssets};
use super::vision::{SimVisionSnapshot, VisionPipeline, VisionStep};
use super::vision_worker::VisionWorker;
use super::VISION_CONF_THRESH;

const INFER_TIMEOUT: Duration = Duration::from_secs(120);
/// 等待在途帧超过此逻辑 tick 数则允许再次 submit（防永久饿死）。
const MAX_PENDING_TICKS: u32 = 180;

/// 窗口预览：与 headless 相同的 YOLO 背压/轮询，capture 拆成 draw + flush（在 `next_frame` 后读像素）。
pub struct DeferredCaptureVision {
    worker: VisionWorker,
    pending_submit_tick: Option<u32>,
    pending_snapshot: Option<SimVisionSnapshot>,
    drawn: bool,
    /// 最近一次 YOLO 结果（供预览叠框）。
    last_detections: Vec<crate::yolo::Detection>,
    last_self_player: Option<crate::player_name::NamedPlayerHit>,
    last_step: Option<VisionStep>,
}

impl DeferredCaptureVision {
    pub fn spawn(pipeline: VisionPipeline) -> Self {
        Self {
            worker: VisionWorker::spawn(pipeline),
            pending_submit_tick: None,
            pending_snapshot: None,
            drawn: false,
            last_detections: Vec::new(),
            last_self_player: None,
            last_step: None,
        }
    }

    pub fn worker_dead(&self) -> bool {
        self.worker.is_dead()
    }

    pub fn clear_pending(&mut self) {
        self.pending_submit_tick = None;
        self.pending_snapshot = None;
        self.drawn = false;
        self.last_detections.clear();
        self.last_self_player = None;
        self.last_step = None;
    }

    pub fn last_detections(&self) -> &[crate::yolo::Detection] {
        &self.last_detections
    }

    pub fn last_self_player(&self) -> Option<&crate::player_name::NamedPlayerHit> {
        self.last_self_player.as_ref()
    }

    pub fn capture_pending(&self) -> bool {
        self.drawn
    }

    pub fn last_vision_step(&self) -> Option<&VisionStep> {
        self.last_step.as_ref()
    }

    pub fn poll_observation(&mut self, sim: &GameSim) -> Option<(u32, [f32; OBS_DIM])> {
        let result = self.worker.poll_result()?;
        self.last_detections = result.step.detections.clone();
        self.last_self_player = result.step.self_player.clone();
        self.last_step = Some(result.step.clone());
        self.pending_submit_tick = None;
        Some((result.tick, obs_from_step(sim, &result.step)))
    }

    pub fn schedule_draw_if_ready(
        &mut self,
        tick: u32,
        sim: &GameSim,
        assets: &GameViewAssets,
        rt: &RenderTarget,
    ) -> bool {
        if self.drawn {
            return false;
        }
        if let Some(since) = self.pending_submit_tick {
            if tick.saturating_sub(since) <= MAX_PENDING_TICKS {
                return false;
            }
            self.pending_submit_tick = None;
        }
        if !self.worker.can_accept_job() {
            return false;
        }
        view::draw_to_render_target(assets, sim, rt);
        self.pending_snapshot = Some(sim.vision_snapshot());
        self.pending_submit_tick = Some(tick);
        self.drawn = true;
        true
    }

    pub fn flush_submit(&mut self, rt: &RenderTarget) -> bool {
        if !self.drawn {
            return false;
        }
        let Some(tick) = self.pending_submit_tick else {
            self.drawn = false;
            return false;
        };
        let snapshot = self.pending_snapshot.take();
        let rgb = view::render_target_to_rgb(rt);
        if self.worker.try_submit(tick, rgb, snapshot) {
            self.drawn = false;
            true
        } else {
            self.pending_snapshot = snapshot;
            self.drawn = true;
            false
        }
    }

    /// 与 headless 一致：warmup + 渲染 + 立即 submit（探针模式用）。
    pub async fn schedule_capture_if_ready(
        &mut self,
        logic_tick: u32,
        sim: &GameSim,
        assets: &GameViewAssets,
        rt: &RenderTarget,
    ) -> Result<bool> {
        if let Some(since) = self.pending_submit_tick {
            if logic_tick.saturating_sub(since) <= MAX_PENDING_TICKS {
                return Ok(false);
            }
            self.pending_submit_tick = None;
        }
        if !self.worker.can_accept_job() {
            return Ok(false);
        }
        view::draw_to_render_target(assets, sim, rt);
        next_frame().await;
        view::draw_to_render_target(assets, sim, rt);
        next_frame().await;
        let rgb = view::render_target_to_rgb(rt);
        if self
            .worker
            .try_submit(logic_tick, rgb, Some(sim.vision_snapshot()))
        {
            self.pending_submit_tick = Some(logic_tick);
            self.drawn = false;
            self.pending_snapshot = None;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn observe_sim_blocking(
        &mut self,
        sim: &GameSim,
        assets: &GameViewAssets,
        rt: &RenderTarget,
    ) -> Result<[f32; OBS_DIM]> {
        view::draw_to_render_target(assets, sim, rt);
        next_frame().await;
        let rgb = view::render_target_to_rgb(rt);
        let snap = sim.vision_snapshot();
        let step = self
            .worker
            .infer_blocking(sim.state.tick as u32, rgb, Some(snap), INFER_TIMEOUT)
            .context("YOLO 感知")?;
        self.last_detections = step.detections.clone();
        self.last_self_player = step.self_player.clone();
        self.last_step = Some(step.clone());
        self.clear_pending();
        Ok(obs_from_step(sim, &step))
    }
}

pub fn default_yolo_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("onnx/yolo_nangang_e3000_best.onnx")
}

/// 离屏 GL 渲染 + 非阻塞 YOLO 管线（macroquad 已初始化后使用）。
pub struct HeadlessVisionEnv {
    assets: GameViewAssets,
    rt: RenderTarget,
    worker: VisionWorker,
    warmup_frames: usize,
    pending_submit_tick: Option<u32>,
}

impl HeadlessVisionEnv {
    pub async fn load(model: Option<&Path>) -> Result<Self> {
        let path = model
            .map(Path::to_path_buf)
            .unwrap_or_else(default_yolo_model_path);
        if !path.exists() {
            anyhow::bail!(
                "YOLO 模型不存在: {}（请放置模型或传入 --model）",
                path.display()
            );
        }
        let assets = view::load_view_assets()
            .await
            .map_err(|e| anyhow::anyhow!("加载游戏渲染资源: {e}"))?;
        let pipeline = VisionPipeline::load(&path, YoloDevice::Cpu, VISION_CONF_THRESH)
            .context("加载 YOLO")?
            .with_anchor(VisionAnchorConfig::ocr());
        let worker = VisionWorker::spawn(pipeline);
        Ok(Self {
            assets,
            rt: view::new_render_target(),
            worker,
            warmup_frames: 1,
            pending_submit_tick: None,
        })
    }

    pub fn worker_dead(&self) -> bool {
        self.worker.is_dead()
    }

    /// 上一帧仍在 YOLO 队列中则跳过渲染/submit，避免高并发下观测全丢。
    pub async fn schedule_capture_if_ready(
        &mut self,
        logic_tick: u32,
        sim: &GameSim,
    ) -> Result<bool> {
        if let Some(since) = self.pending_submit_tick {
            if logic_tick.saturating_sub(since) <= MAX_PENDING_TICKS {
                return Ok(false);
            }
            self.pending_submit_tick = None;
        }
        if !self.worker.can_accept_job() {
            return Ok(false);
        }
        let rgb = self.capture_rgb(sim).await;
        if self
            .worker
            .try_submit(logic_tick, rgb, Some(sim.vision_snapshot()))
        {
            self.pending_submit_tick = Some(logic_tick);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// 非阻塞取回最新推理观测（无新结果则 `None`）。
    pub fn poll_vision(&mut self, sim: &GameSim) -> Option<(u32, [f32; OBS_DIM], VisionStep)> {
        let result = self.worker.poll_result()?;
        self.pending_submit_tick = None;
        Some((
            result.tick,
            obs_from_step(sim, &result.step),
            result.step,
        ))
    }

    pub fn poll_observation(&mut self, sim: &GameSim) -> Option<(u32, [f32; OBS_DIM])> {
        self.poll_vision(sim).map(|(t, o, _)| (t, o))
    }

    /// 阻塞式单帧观测（仅诊断；探针/游戏请用 schedule + poll）。
    pub async fn observe_sim_blocking(&mut self, sim: &GameSim) -> Result<[f32; OBS_DIM]> {
        let (obs, _step) = self.observe_sim_blocking_with_step(sim).await?;
        Ok(obs)
    }

    pub async fn observe_sim_blocking_with_step(
        &mut self,
        sim: &GameSim,
    ) -> Result<([f32; OBS_DIM], VisionStep)> {
        let rgb = self.capture_rgb(sim).await;
        let snap = sim.vision_snapshot();
        let step = self
            .worker
            .infer_blocking(sim.state.tick as u32, rgb, Some(snap), INFER_TIMEOUT)
            .context("YOLO 感知")?;
        self.pending_submit_tick = None;
        Ok((obs_from_step(sim, &step), step))
    }

    async fn capture_rgb(&self, sim: &GameSim) -> RgbImage {
        for _ in 0..self.warmup_frames {
            view::draw_to_render_target(&self.assets, sim, &self.rt);
            next_frame().await;
        }
        view::draw_to_render_target(&self.assets, sim, &self.rt);
        next_frame().await;
        view::render_target_to_rgb(&self.rt)
    }
}

pub fn obs_from_step(_sim: &GameSim, step: &VisionStep) -> [f32; OBS_DIM] {
    let mut obs = [0.0_f32; OBS_DIM];
    let n = step.observation.values.len().min(OBS_DIM);
    obs[..n].copy_from_slice(&step.observation.values[..n]);
    obs
}
