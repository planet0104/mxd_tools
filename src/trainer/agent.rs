//! 游戏主线程与 YOLO+OCR+NEAT 决策线程解耦（本地 CPU 视觉）。

use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use image::RgbImage;

use crate::game::InputFrame;
use crate::game::{GameSim, SimVisionSnapshot, VisionPipeline, VisionStep};
use crate::neat::{evaluate, input_from_outputs, Genome};

const FRAME_QUEUE: usize = 2;

enum FrameMsg {
    Job {
        tick: u32,
        rgb: RgbImage,
        submitted_ns: u64,
        sim_snapshot: Option<SimVisionSnapshot>,
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
    input: InputFrame,
    neat_outputs: Vec<f32>,
    timing: VisionWorkerTiming,
}

/// 视觉 + NEAT 控制器。
pub struct AgentController {
    frame_tx: std::sync::mpsc::SyncSender<FrameMsg>,
    result_rx: std::sync::mpsc::Receiver<VisionAgentResult>,
    genome: Arc<RwLock<Genome>>,
    join: Option<JoinHandle<()>>,
    last_input: InputFrame,
    last_vision: Option<VisionStep>,
    last_applied_tick: Option<u32>,
    last_neat_outputs: Option<Vec<f32>>,
    worker_dead: bool,
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
            .spawn(move || vision_agent_loop(frame_rx, result_tx, pipeline, genome_worker))
            .expect("spawn vision-neat-agent");

        Self {
            frame_tx,
            result_rx,
            genome,
            join: Some(join),
            last_input: InputFrame::default(),
            last_vision: None,
            last_applied_tick: None,
            last_neat_outputs: None,
            worker_dead: false,
            last_poll_timings: Vec::new(),
        }
    }

    pub fn set_genome(&self, genome: Genome) {
        if let Ok(mut g) = self.genome.write() {
            *g = genome;
        }
    }

    pub fn try_submit_frame(&self, tick: u32, rgb: RgbImage, sim_snapshot: Option<SimVisionSnapshot>) -> bool {
        if self.worker_dead {
            return false;
        }
        let submitted_ns = crate::trainer::profile::now_ns();
        self.frame_tx
            .try_send(FrameMsg::Job {
                tick,
                rgb,
                submitted_ns,
                sim_snapshot,
            })
            .is_ok()
    }

    pub fn input(&self) -> InputFrame {
        self.last_input
    }

    pub fn vision(&self) -> Option<&VisionStep> {
        self.last_vision.as_ref()
    }

    pub fn last_applied_tick(&self) -> Option<u32> {
        self.last_applied_tick
    }

    pub fn last_neat_outputs(&self) -> Option<&[f32]> {
        self.last_neat_outputs.as_deref()
    }

    pub fn worker_dead(&self) -> bool {
        self.worker_dead
    }

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
                sim_snapshot: Some(sim.vision_snapshot()),
            })
            .map_err(|_| anyhow::anyhow!("视觉线程已退出"))?;
        self.wait_for_tick(sim, tick, timeout)
    }

    pub fn poll(&mut self, sim: &mut GameSim) {
        self.last_poll_timings.clear();
        let batch: Vec<VisionAgentResult> = {
            let mut batch = Vec::new();
            loop {
                match self.result_rx.try_recv() {
                    Ok(result) => batch.push(result),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        self.worker_dead = true;
                        break;
                    }
                }
            }
            batch
        };
        for result in batch {
            self.last_poll_timings.push(result.timing.clone());
            self.apply_result(sim, result);
        }
    }

    pub fn take_poll_timings(&mut self) -> Vec<VisionWorkerTiming> {
        std::mem::take(&mut self.last_poll_timings)
    }

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
        self.last_input = result.input;
        self.last_vision = Some(result.step);
        self.last_neat_outputs = Some(result.neat_outputs);
        self.last_applied_tick = Some(result.tick);
    }

    pub fn reset_vision_state(&mut self) {
        self.last_input = InputFrame::default();
        self.last_vision = None;
        self.last_neat_outputs = None;
        self.last_applied_tick = None;
        self.last_poll_timings.clear();
    }

    pub fn shutdown(&mut self) {
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
                sim_snapshot,
            } => {
                let worker_start = Instant::now();
                let queue_wait_ms = crate::trainer::profile::now_ns()
                    .saturating_sub(submitted_ns) as f64
                    / 1_000_000.0;
                let t0 = Instant::now();
                let step = match pipeline.perceive_with_snapshot(&rgb, sim_snapshot) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("视觉线程推理失败 tick={tick}: {e}");
                        break;
                    }
                };
                let perceive_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let t1 = Instant::now();
                let (input, neat_outputs) = {
                    let g = genome.read().expect("genome lock");
                    let outputs = evaluate(&g, &step.observation.values);
                    (input_from_outputs(&outputs), outputs)
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
                        input,
                        neat_outputs,
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
