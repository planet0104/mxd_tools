//! 游戏主线程与 YOLO+OCR+NEAT 决策线程解耦。
//!
//! 主线程：渲染、读 RGB、按 60Hz `sim.tick`。
//! 后台线程：`perceive` → NEAT 前向 → 回传 `Action` 与 `VisionStep`（非阻塞）。

use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use image::RgbImage;

use crate::game::action::Action;
use crate::game::{GameSim, VisionPipeline, VisionStep};
use crate::neat::{action_from_outputs, evaluate, Genome};

const FRAME_QUEUE: usize = 2;

enum FrameMsg {
    Job {
        tick: u32,
        rgb: RgbImage,
        submitted_ns: u64,
    },
    Shutdown,
}

#[derive(Debug, Clone, Default)]
pub struct VisionWorkerTiming {
    pub tick: u32,
    pub queue_wait_ms: f64,
    pub perceive_ms: f64,
    pub neat_ms: f64,
    pub worker_total_ms: f64,
}

struct VisionAgentResult {
    tick: u32,
    step: VisionStep,
    action: Action,
    timing: VisionWorkerTiming,
}

/// 后台视觉 + NEAT 控制器；`VisionPipeline` 仅在其工作线程内使用。
pub struct AgentController {
    frame_tx: std::sync::mpsc::SyncSender<FrameMsg>,
    result_rx: std::sync::mpsc::Receiver<VisionAgentResult>,
    genome: Arc<RwLock<Genome>>,
    join: Option<JoinHandle<()>>,
    last_action: Action,
    last_vision: Option<VisionStep>,
    /// 尚无视觉结果时为 `None`。
    last_applied_tick: Option<u32>,
    worker_dead: bool,
    /// `--profile`：最近一次 poll 收到的 worker 耗时。
    last_poll_timings: Vec<VisionWorkerTiming>,
}

impl AgentController {
    pub fn spawn(pipeline: VisionPipeline, genome: Genome) -> Self {
        let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel(FRAME_QUEUE);
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let genome = Arc::new(RwLock::new(genome));

        let genome_worker = Arc::clone(&genome);
        let join = thread::Builder::new()
            .name("vision-neat-agent".into())
            .spawn(move || {
                vision_agent_loop(frame_rx, result_tx, pipeline, genome_worker);
            })
            .expect("spawn vision-neat-agent");

        Self {
            frame_tx,
            result_rx,
            genome,
            join: Some(join),
            last_action: Action::Noop,
            last_vision: None,
            last_applied_tick: None,
            worker_dead: false,
            last_poll_timings: Vec::new(),
        }
    }

    pub fn set_genome(&self, genome: Genome) {
        if let Ok(mut g) = self.genome.write() {
            *g = genome;
        }
    }

    /// 非阻塞提交帧；队列满时丢弃（工人仍在处理上一帧）。
    pub fn try_submit_frame(&self, tick: u32, rgb: RgbImage) -> bool {
        if self.worker_dead {
            return false;
        }
        let submitted_ns = crate::trainer::profile::now_ns();
        self.frame_tx
            .try_send(FrameMsg::Job {
                tick,
                rgb,
                submitted_ns,
            })
            .is_ok()
    }

    pub fn action(&self) -> Action {
        self.last_action
    }

    pub fn vision(&self) -> Option<&VisionStep> {
        self.last_vision.as_ref()
    }

    pub fn last_applied_tick(&self) -> Option<u32> {
        self.last_applied_tick
    }

    pub fn worker_dead(&self) -> bool {
        self.worker_dead
    }

    /// 阻塞提交帧并等待该 tick 的视觉结果（训练 eval / capture 用，保证不丢帧）。
    pub fn submit_and_wait(
        &mut self,
        sim: &mut GameSim,
        tick: u32,
        rgb: RgbImage,
        timeout: Duration,
    ) -> anyhow::Result<()> {
        if self.worker_dead {
            anyhow::bail!("视觉线程已退出");
        }
        self.frame_tx
            .send(FrameMsg::Job {
                tick,
                rgb,
                submitted_ns: crate::trainer::profile::now_ns(),
            })
            .map_err(|_| anyhow::anyhow!("视觉线程已退出"))?;
        self.wait_for_tick(sim, tick, timeout)
    }

    /// 收取后台结果并写入 `sim` 计分提示。
    pub fn poll(&mut self, sim: &mut GameSim) {
        self.last_poll_timings.clear();
        loop {
            match self.result_rx.try_recv() {
                Ok(result) => {
                    self.last_poll_timings.push(result.timing.clone());
                    self.apply_result(sim, result);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.worker_dead = true;
                    break;
                }
            }
        }
    }

    pub fn take_poll_timings(&mut self) -> Vec<VisionWorkerTiming> {
        std::mem::take(&mut self.last_poll_timings)
    }

    /// 阻塞直到指定 tick 的视觉结果到达。
    pub fn wait_for_tick(
        &mut self,
        sim: &mut GameSim,
        tick: u32,
        timeout: Duration,
    ) -> anyhow::Result<()> {
        let deadline = Instant::now() + timeout;
        while self.last_applied_tick != Some(tick) {
            if self.worker_dead {
                anyhow::bail!("视觉线程已退出");
            }
            if Instant::now() >= deadline {
                anyhow::bail!(
                    "等待 tick {tick} 视觉结果超时（最近={:?}）",
                    self.last_applied_tick
                );
            }
            self.poll(sim);
            thread::sleep(Duration::from_millis(2));
        }
        Ok(())
    }

    fn apply_result(&mut self, sim: &mut GameSim, result: VisionAgentResult) {
        result.step.apply_fitness_hints(sim);
        self.last_action = result.action;
        self.last_vision = Some(result.step);
        self.last_applied_tick = Some(result.tick);
    }

    /// 新一局 eval 前重置视觉状态（worker 线程不重启）。
    pub fn reset_vision_state(&mut self) {
        self.last_action = Action::Noop;
        self.last_vision = None;
        self.last_applied_tick = None;
        self.last_poll_timings.clear();
    }

    /// 关闭视觉线程；队列满时阻塞直到 Shutdown 入队（不可 try_send）。
    pub fn shutdown(&mut self) {
        if self.join.is_none() {
            return;
        }
        let _ = self.frame_tx.send(FrameMsg::Shutdown);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl Drop for AgentController {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn vision_agent_loop(
    frame_rx: std::sync::mpsc::Receiver<FrameMsg>,
    result_tx: std::sync::mpsc::Sender<VisionAgentResult>,
    mut pipeline: VisionPipeline,
    genome: Arc<RwLock<Genome>>,
) {
    while let Ok(msg) = frame_rx.recv() {
        match msg {
            FrameMsg::Shutdown => break,
            FrameMsg::Job {
                tick,
                rgb,
                submitted_ns,
            } => {
                let worker_start = Instant::now();
                let queue_wait_ms = crate::trainer::profile::now_ns()
                    .saturating_sub(submitted_ns) as f64
                    / 1_000_000.0;
                let t0 = Instant::now();
                let step = match pipeline.perceive(&rgb) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("视觉线程推理失败 tick={tick}: {e}");
                        continue;
                    }
                };
                let perceive_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let t1 = Instant::now();
                let action = {
                    let g = genome.read().expect("genome lock");
                    let outputs = evaluate(&g, &step.observation.values);
                    action_from_outputs(&outputs)
                };
                let neat_ms = t1.elapsed().as_secs_f64() * 1000.0;
                let timing = VisionWorkerTiming {
                    tick,
                    queue_wait_ms,
                    perceive_ms,
                    neat_ms,
                    worker_total_ms: worker_start.elapsed().as_secs_f64() * 1000.0,
                };
                if result_tx
                    .send(VisionAgentResult {
                        tick,
                        step,
                        action,
                        timing,
                    })
                    .is_err()
                {
                    break;
                }
            }
        }
    }
}
