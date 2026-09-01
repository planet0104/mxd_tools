//! YOLO+OCR 后台线程（主线程只负责 GL 渲染与游戏逻辑，推理按真实 ONNX 耗时在独立线程执行）。

use std::sync::atomic::{AtomicU32, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use image::RgbImage;

use super::vision::{SimVisionSnapshot, VisionPipeline, VisionStep};

/// 在途帧上限；满则不再 submit，避免高负载下刷掉全部观测。
const FRAME_QUEUE: usize = 4;

enum FrameMsg {
    Job {
        tick: u32,
        rgb: RgbImage,
        submitted_ns: u64,
        sim_snapshot: Option<SimVisionSnapshot>,
    },
    Shutdown,
}

/// 视觉线程完成的一帧推理结果（含真实耗时）。
#[derive(Debug, Clone)]
pub struct VisionJobResult {
    pub tick: u32,
    pub step: VisionStep,
    pub queue_wait_ms: f64,
    pub perceive_ms: f64,
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// 独立 YOLO 推理线程（`VisionPipeline` 独占于此线程）。
pub struct VisionWorker {
    frame_tx: std::sync::mpsc::SyncSender<FrameMsg>,
    result_rx: std::sync::mpsc::Receiver<VisionJobResult>,
    join: Option<JoinHandle<()>>,
    dead: bool,
    in_flight: AtomicU32,
}

impl VisionWorker {
    pub fn spawn(pipeline: VisionPipeline) -> Self {
        let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel(FRAME_QUEUE);
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let join = thread::Builder::new()
            .name("vision-yolo-worker".into())
            .spawn(move || vision_worker_loop(frame_rx, result_tx, pipeline))
            .expect("spawn vision-yolo-worker");

        Self {
            frame_tx,
            result_rx,
            join: Some(join),
            dead: false,
            in_flight: AtomicU32::new(0),
        }
    }

    pub fn is_dead(&self) -> bool {
        self.dead
    }

    pub fn in_flight_frames(&self) -> u32 {
        self.in_flight.load(Ordering::Relaxed)
    }

    pub fn can_accept_job(&self) -> bool {
        !self.dead && self.in_flight.load(Ordering::Relaxed) < FRAME_QUEUE as u32
    }

    /// 非阻塞提交一帧；队列满或在途过多时返回 false。
    pub fn try_submit(
        &self,
        tick: u32,
        rgb: RgbImage,
        sim_snapshot: Option<SimVisionSnapshot>,
    ) -> bool {
        if !self.can_accept_job() {
            return false;
        }
        match self.frame_tx.try_send(FrameMsg::Job {
            tick,
            rgb,
            submitted_ns: now_ns(),
            sim_snapshot,
        }) {
            Ok(()) => {
                self.in_flight.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(_) => false,
        }
    }

    /// 非阻塞取回已完成推理；游戏逻辑应在每逻辑 tick 调用，不阻塞等待 YOLO。
    pub fn poll_result(&mut self) -> Option<VisionJobResult> {
        if self.dead {
            return None;
        }
        match self.result_rx.try_recv() {
            Ok(result) => {
                self.in_flight.fetch_sub(1, Ordering::Relaxed);
                Some(result)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.dead = true;
                None
            }
        }
    }

    /// 阻塞等待指定 tick 的结果（仅测试/诊断用；正式游戏循环勿用）。
    pub fn infer_blocking(
        &mut self,
        tick: u32,
        rgb: RgbImage,
        sim_snapshot: Option<SimVisionSnapshot>,
        timeout: Duration,
    ) -> anyhow::Result<VisionStep> {
        if self.dead {
            anyhow::bail!("视觉线程已退出");
        }
        self.frame_tx
            .send(FrameMsg::Job {
                tick,
                rgb,
                submitted_ns: now_ns(),
                sim_snapshot,
            })
            .map_err(|_| anyhow::anyhow!("视觉线程已退出"))?;
        self.in_flight.fetch_add(1, Ordering::Relaxed);

        let deadline = Instant::now() + timeout;
        loop {
            if self.dead {
                anyhow::bail!("视觉线程已退出");
            }
            match self.result_rx.recv_timeout(Duration::from_millis(2)) {
                Ok(result) if result.tick == tick => return Ok(result.step),
                Ok(_) => {
                    self.in_flight.fetch_sub(1, Ordering::Relaxed);
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if Instant::now() >= deadline {
                        anyhow::bail!("等待 tick {tick} 视觉结果超时");
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    self.dead = true;
                    anyhow::bail!("视觉线程已退出");
                }
            }
        }
    }

    pub fn shutdown(&mut self) {
        let _ = self.frame_tx.send(FrameMsg::Shutdown);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl Drop for VisionWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn vision_worker_loop(
    frame_rx: std::sync::mpsc::Receiver<FrameMsg>,
    result_tx: std::sync::mpsc::Sender<VisionJobResult>,
    mut pipeline: VisionPipeline,
) {
    while let Ok(msg) = frame_rx.recv() {
        match msg {
            FrameMsg::Shutdown => break,
            FrameMsg::Job {
                tick,
                rgb,
                submitted_ns,
                sim_snapshot,
            } => {
                let queue_wait_ms = now_ns().saturating_sub(submitted_ns) as f64 / 1_000_000.0;
                let t0 = Instant::now();
                let step = match pipeline.perceive_with_snapshot(&rgb, sim_snapshot) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("视觉线程推理失败 tick={tick}: {e}");
                        break;
                    }
                };
                let perceive_ms = t0.elapsed().as_secs_f64() * 1000.0;
                if result_tx
                    .send(VisionJobResult {
                        tick,
                        step,
                        queue_wait_ms,
                        perceive_ms,
                    })
                    .is_err()
                {
                    break;
                }
            }
        }
    }
}
