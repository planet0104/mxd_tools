//! 多槽并行 NEAT 评估：主线程 GL 离屏渲染，每槽独立 CPU YOLO 推理线程。

use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use macroquad::prelude::*;
use ::rand::rngs::StdRng;

use crate::game::config::VisionAnchorConfig;
use crate::game::fitness::FitnessShapingConfig;
use crate::game::headless_vision::obs_from_step;
use crate::game::map::GameMap;
use crate::game::view::{self, GameViewAssets};
use crate::game::vision::VisionPipeline;
use crate::game::vision_worker::VisionWorker;
use crate::game::GameSim;
use crate::yolo::YoloDevice;
use crate::game::VISION_CONF_THRESH;

use super::driver::NeatDriver;
use super::eval::EvalOutcome;
use super::genome::{rank_fitness, Genome};
use super::population::Population;
use super::snapshot::{save_best_if_improved, save_session_best};

const INFER_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_PENDING_TICKS: u32 = 180;
const STATUS_INTERVAL: Duration = Duration::from_secs(5);

pub struct ParallelTrainerConfig {
    pub workers: usize,
    pub total_spawns: u32,
    pub max_ticks: u32,
    pub vision_interval: u32,
    pub seed: u64,
    pub shaping: FitnessShapingConfig,
    pub checkpoint: Option<ParallelCheckpointConfig>,
}

pub struct ParallelCheckpointConfig {
    pub path: PathBuf,
    pub best_genome: PathBuf,
    pub session_best: PathBuf,
}

struct CheckpointState {
    last_saved_fitness: f32,
    last_session_fitness: f32,
}

pub struct TrainerShared {
    pub population: Population,
    pub rng: StdRng,
    pub spawns_completed: u32,
    pub next_spawn_id: u32,
}

impl TrainerShared {
    pub fn spawn_offspring(&mut self, spawn_index: u32) -> Genome {
        self.population.spawn_offspring(&mut self.rng, spawn_index)
    }
}

enum SlotTick {
    Idle,
    Booting,
    Running,
    Finished(EvalOutcome),
}

/// 单路并行评估槽（内含独立 CPU YOLO 线程）。
pub struct EvalSlot {
    worker: VisionWorker,
    rt: RenderTarget,
    sim: Option<GameSim>,
    driver: Option<NeatDriver>,
    spawn_id: u32,
    logic_tick: u32,
    peak_fitness: f32,
    pending_submit_tick: Option<u32>,
    bootstrap_phase: u8,
}

impl EvalSlot {
    fn new(worker: VisionWorker) -> Self {
        Self {
            worker,
            rt: view::new_render_target(),
            sim: None,
            driver: None,
            spawn_id: 0,
            logic_tick: 0,
            peak_fitness: 0.0,
            pending_submit_tick: None,
            bootstrap_phase: 0,
        }
    }

    fn active(&self) -> bool {
        self.sim.is_some()
    }

    fn start(&mut self, map: &GameMap, genome: Genome, spawn_id: u32, episode_seed: u64, shaping: FitnessShapingConfig) {
        let mut sim = GameSim::new_training(map.clone(), episode_seed);
        sim.fitness.configure_shaping(shaping);
        self.sim = Some(sim);
        self.driver = Some(NeatDriver::new(genome));
        self.spawn_id = spawn_id;
        self.logic_tick = 0;
        self.peak_fitness = 0.0;
        self.pending_submit_tick = None;
        self.bootstrap_phase = 0;
    }

    fn clear(&mut self) {
        self.sim = None;
        self.driver = None;
        self.logic_tick = 0;
        self.peak_fitness = 0.0;
        self.pending_submit_tick = None;
        self.bootstrap_phase = 0;
    }

    fn tick(
        &mut self,
        assets: &GameViewAssets,
        vision_interval: u32,
        max_ticks: u32,
    ) -> Result<SlotTick> {
        if self.sim.is_none() {
            return Ok(SlotTick::Idle);
        }
        if self.worker.is_dead() {
            anyhow::bail!("槽位 {} YOLO 线程已退出", self.spawn_id);
        }

        let sim = self.sim.as_mut().unwrap();
        let driver = self.driver.as_mut().unwrap();

        if let Some(result) = self.worker.poll_result() {
            self.pending_submit_tick = None;
            sim.record_vision_loot(&result.step.detections);
            let obs = obs_from_step(sim, &result.step);
            driver.apply_observation(sim, obs);
        }

        if self.bootstrap_phase < 2 {
            view::draw_to_render_target(assets, sim, &self.rt);
            self.bootstrap_phase += 1;
            if self.bootstrap_phase == 2 {
                let rgb = view::render_target_to_rgb(&self.rt);
                let snap = sim.vision_snapshot();
                let step = self
                    .worker
                    .infer_blocking(sim.state.tick as u32, rgb, Some(snap), INFER_TIMEOUT)
                    .context("槽位 bootstrap YOLO")?;
                sim.record_vision_loot(&step.detections);
                let obs = obs_from_step(sim, &step);
                driver.apply_observation(sim, obs);
            }
            return Ok(SlotTick::Booting);
        }

        if let Some(since) = self.pending_submit_tick {
            if self.logic_tick.saturating_sub(since) > MAX_PENDING_TICKS {
                self.pending_submit_tick = None;
            }
        }
        if self.pending_submit_tick.is_none()
            && self.worker.can_accept_job()
            && self.logic_tick % vision_interval == 0
        {
            view::draw_to_render_target(assets, sim, &self.rt);
            let rgb = view::render_target_to_rgb(&self.rt);
            if self
                .worker
                .try_submit(self.logic_tick, rgb, Some(sim.vision_snapshot()))
            {
                self.pending_submit_tick = Some(self.logic_tick);
            }
        }

        driver.tick_sim(sim);
        self.peak_fitness = self.peak_fitness.max(sim.fitness.score);
        self.logic_tick += 1;

        if sim.is_episode_over() || self.logic_tick >= max_ticks {
            sim.fitness.finalize_episode();
            let outcome = EvalOutcome {
                final_fitness: sim.fitness.score,
                peak_fitness: self.peak_fitness,
            };
            return Ok(SlotTick::Finished(outcome));
        }
        Ok(SlotTick::Running)
    }
}

fn spawn_cpu_worker(model: &Path) -> Result<VisionWorker> {
    let pipeline = VisionPipeline::load(model, YoloDevice::Cpu, VISION_CONF_THRESH)
        .with_context(|| format!("加载 YOLO {}", model.display()))?
        .with_anchor(VisionAnchorConfig::ocr());
    Ok(VisionWorker::spawn(pipeline))
}

pub async fn prepare_parallel_assets() -> Result<GameViewAssets> {
    view::load_view_assets()
        .await
        .map_err(|e| anyhow::anyhow!("加载游戏渲染资源: {e}"))
}

pub fn create_parallel_slots(workers: usize, model: &Path) -> Result<Vec<EvalSlot>> {
    (0..workers)
        .map(|i| {
            spawn_cpu_worker(model)
                .with_context(|| format!("创建并行 YOLO 线程 {i}"))
                .map(EvalSlot::new)
        })
        .collect()
}

fn save_training_checkpoint(
    cfg: &ParallelTrainerConfig,
    ckpt_state: &mut CheckpointState,
    shared: &TrainerShared,
) -> Result<()> {
    let Some(paths) = &cfg.checkpoint else {
        return Ok(());
    };
    let _ = save_best_if_improved(
        &paths.best_genome,
        &shared.population,
        cfg.seed,
        &mut ckpt_state.last_saved_fitness,
    )?;
    if rank_fitness(&shared.population.best_ever) > ckpt_state.last_session_fitness {
        save_session_best(
            &paths.session_best,
            &shared.population.best_ever,
            shared.population.generation,
            cfg.seed,
        )?;
        ckpt_state.last_session_fitness = rank_fitness(&shared.population.best_ever);
    }
    if let Some(parent) = paths.path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let ckpt = super::population::TrainingCheckpoint {
        population: shared.population.clone(),
        seed: cfg.seed,
        innovation: super::genome::export_innovation_state(),
        spawns_completed: shared.spawns_completed,
        next_spawn_id: shared.next_spawn_id,
        total_spawn_target: cfg.total_spawns,
    };
    let json = serde_json::to_string_pretty(&ckpt)?;
    let tmp = paths.path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &paths.path)?;
    Ok(())
}

pub async fn run_parallel_trainer(
    cfg: ParallelTrainerConfig,
    _model: &Path,
    map: &GameMap,
    assets: &GameViewAssets,
    shared: &Mutex<TrainerShared>,
    slots: &mut [EvalSlot],
) -> Result<()> {
    let mut ckpt_state = CheckpointState {
        last_saved_fitness: 0.0,
        last_session_fitness: 0.0,
    };
    let mut window_done: u32 = 0;
    let mut window_peak_sum: f32 = 0.0;
    let mut window_final_sum: f32 = 0.0;
    let mut window_peak_max: f32 = f32::NEG_INFINITY;
    let started = Instant::now();
    let mut last_status = Instant::now();
    let mut last_completed_for_rate = {
        let s = shared.lock().expect("trainer lock");
        s.spawns_completed
    };

    let initial = cfg.workers.min(cfg.total_spawns as usize);
    {
        let mut s = shared.lock().expect("trainer lock");
        for i in 0..initial {
            let spawn_id = s.next_spawn_id;
            s.next_spawn_id += 1;
            let genome = s.spawn_offspring(spawn_id);
            let episode_seed = cfg.seed.wrapping_add(spawn_id as u64);
            slots[i].start(map, genome, spawn_id, episode_seed, cfg.shaping);
        }
        eprintln!(
            "并行槽已启动: active={} / workers={} 目标局数={}",
            initial, cfg.workers, cfg.total_spawns
        );
    }

    loop {
        let done = {
            let s = shared.lock().expect("trainer lock");
            s.spawns_completed >= cfg.total_spawns
        };
        let any_active = slots.iter().any(|sl| sl.active());
        if done && !any_active {
            break;
        }

        for slot in slots.iter_mut() {
            if !slot.active() {
                continue;
            }
            match slot.tick(assets, cfg.vision_interval, cfg.max_ticks)? {
                SlotTick::Idle | SlotTick::Booting | SlotTick::Running => {}
                SlotTick::Finished(outcome) => {
                    window_done += 1;
                    window_peak_sum += outcome.peak_fitness;
                    window_final_sum += outcome.final_fitness;
                    window_peak_max = window_peak_max.max(outcome.peak_fitness);

                    let mut scored = slot.driver.as_ref().unwrap().genome().clone();
                    scored.fitness = outcome.final_fitness;
                    scored.peak_fitness = outcome.peak_fitness;
                    slot.clear();

                    let mut start_next = None;
                    {
                        let mut s = shared.lock().expect("trainer lock");
                        s.population.on_eval_complete(scored);
                        s.spawns_completed += 1;
                        let completed = s.spawns_completed;
                        s.population.bump_virtual_generation(completed);
                        save_training_checkpoint(&cfg, &mut ckpt_state, &s)?;
                        if s.spawns_completed < cfg.total_spawns {
                            let id = s.next_spawn_id;
                            s.next_spawn_id += 1;
                            let genome = s.spawn_offspring(id);
                            let episode_seed = cfg.seed.wrapping_add(id as u64);
                            start_next = Some((genome, id, episode_seed));
                        }
                    }
                    if let Some((genome, id, episode_seed)) = start_next {
                        slot.start(map, genome, id, episode_seed, cfg.shaping);
                    }
                }
            }
        }

        if last_status.elapsed() >= STATUS_INTERVAL {
            let (completed, best, gen, active) = {
                let s = shared.lock().expect("trainer lock");
                (
                    s.spawns_completed,
                    rank_fitness(&s.population.best_ever),
                    s.population.generation,
                    slots.iter().filter(|sl| sl.active()).count(),
                )
            };
            let elapsed = last_status.elapsed().as_secs_f32().max(0.001);
            let rate = (completed.saturating_sub(last_completed_for_rate)) as f32 / elapsed;
            let avg_peak = if window_done > 0 {
                window_peak_sum / window_done as f32
            } else {
                0.0
            };
            let avg_final = if window_done > 0 {
                window_final_sum / window_done as f32
            } else {
                0.0
            };
            let peak_max = if window_done > 0 {
                window_peak_max
            } else {
                0.0
            };
            eprintln!(
                "[{:>5.0}s] done={}/{} (+{}) rate={:.2}/s active={} gen={} best_peak={:.1} window: peak_avg={:.1} peak_max={:.1} final_avg={:.1}",
                started.elapsed().as_secs_f32(),
                completed,
                cfg.total_spawns,
                window_done,
                rate,
                active,
                gen,
                best,
                avg_peak,
                peak_max,
                avg_final,
            );
            window_done = 0;
            window_peak_sum = 0.0;
            window_final_sum = 0.0;
            window_peak_max = f32::NEG_INFINITY;
            last_status = Instant::now();
            last_completed_for_rate = completed;
        }

        next_frame().await;
    }
    Ok(())
}

pub fn trainer_progress(shared: &TrainerShared) -> (u32, f32) {
    (
        shared.spawns_completed,
        rank_fitness(&shared.population.best_ever),
    )
}
