//! 单基因组一局评估：主线程渲染+模拟，后台 YOLO+OCR+NEAT。

use anyhow::Result;
use image::RgbImage;
use macroquad::prelude::RenderTarget;
use std::time::Duration;

use crate::game::view::{self, GameViewAssets};
use crate::game::{
    action_to_input, load_default_map, FitnessShapingConfig, GameMap, GameSim, TrainingPaceConfig,
    VisionPipeline, VisionStep,
};
use crate::neat::Genome;
use crate::trainer::agent::AgentController;
use crate::trainer::progress::{maybe_emit_eval_heartbeat, EvalProgressConfig};
use crate::trainer::profile::{elapsed_ms, EvalProfileReport, TickProfile};
use crate::trainer::render::{capture_render_rgb, capture_render_rgb_fast, capture_render_rgb_timed};

const VISION_WAIT_TIMEOUT: Duration = Duration::from_secs(120);

pub struct TrainerEvalContext {
    pub assets: GameViewAssets,
    pub map: GameMap,
    pub rt: RenderTarget,
    agent: AgentController,
}

/// 从训练 eval 循环截取的验证帧。
pub struct TrainingCapture {
    pub tick: usize,
    pub rgb: RgbImage,
    pub step: VisionStep,
    pub fitness: f32,
}

impl TrainerEvalContext {
    pub async fn prepare(
        pipeline: VisionPipeline,
        rt: RenderTarget,
    ) -> Result<Self> {
        let assets = view::load_view_assets()
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
        let map = load_default_map()?;
        let agent = AgentController::spawn(
            pipeline,
            Genome {
                connections: vec![],
                fitness: 0.0,
                adjusted_fitness: 0.0,
            },
        );
        Ok(Self {
            assets,
            map,
            rt,
            agent,
        })
    }

    pub fn shutdown_agent(&mut self) {
        self.agent.shutdown();
    }
}

/// 评估一个基因组，返回 `sim.fitness.score`（**不使用** `ground_truth()`）。
pub async fn evaluate_genome(
    ctx: &mut TrainerEvalContext,
    genome: &Genome,
    episode_seed: u64,
    max_ticks: usize,
    pace: TrainingPaceConfig,
    shaping: FitnessShapingConfig,
    progress: Option<&EvalProgressConfig>,
) -> Result<f32> {
    pace.validate().map_err(|e| anyhow::anyhow!(e))?;
    ctx.agent.set_genome(genome.clone());
    ctx.agent.reset_vision_state();

    let mut sim = GameSim::new_training(ctx.map.clone(), episode_seed);
    sim.fitness.configure_shaping(shaping);
    let interval = pace.vision_interval_ticks as usize;
    let mut last_heartbeat = std::time::Instant::now();

    for tick in 0..max_ticks {
        maybe_emit_eval_heartbeat(&mut last_heartbeat, &sim, tick, max_ticks, progress);

        ctx.agent.poll(&mut sim);

        if tick % interval == 0 {
            let rgb = capture_render_rgb_fast(&ctx.assets, &sim, &ctx.rt).await;
            ctx.agent.try_submit_frame(tick as u32, rgb);
        }

        let action = ctx.agent.action();
        sim.tick_with_action(&action_to_input(action), Some(action));

        if sim.is_episode_over() {
            break;
        }
    }

    ctx.agent.poll(&mut sim);

    if let Some(cfg) = progress {
        let final_tick = sim.state.tick as usize;
        let mut status = crate::trainer::progress::EvalStatus::from_sim(
            &cfg.label,
            &sim,
            final_tick,
            max_ticks,
        );
        status.eval_done = true;
        if let Some(path) = &cfg.status_file {
            status.write_file(path);
        }
        if cfg.console {
            status.log_console();
        }
    }

    Ok(sim.fitness.score)
}

/// 单个体 eval 逐步耗时剖析（`neat_trainer --profile`）。
pub async fn evaluate_genome_profile(
    ctx: &mut TrainerEvalContext,
    genome: &Genome,
    episode_seed: u64,
    max_ticks: usize,
    pace: TrainingPaceConfig,
    shaping: FitnessShapingConfig,
) -> Result<EvalProfileReport> {
    use std::time::Instant;

    pace.validate().map_err(|e| anyhow::anyhow!(e))?;
    let wall0 = Instant::now();
    let setup0 = Instant::now();

    ctx.agent.set_genome(genome.clone());
    ctx.agent.reset_vision_state();

    let mut sim = GameSim::new_training(ctx.map.clone(), episode_seed);
    sim.fitness.configure_shaping(shaping);
    let interval = pace.vision_interval_ticks as usize;
    let setup_ms = elapsed_ms(setup0);

    let mut report = EvalProfileReport {
        setup_ms,
        ..Default::default()
    };

    let mut last_submitted_tick: Option<u32> = None;

    eprintln!(
        "profile: 开始单个体 eval  seed={episode_seed} max_ticks={max_ticks} pace={}",
        pace.vision_interval_ticks
    );

    for tick in 0..max_ticks {
        let loop0 = Instant::now();
        let mut tp = TickProfile {
            tick,
            vision_tick: tick % interval == 0,
            ..Default::default()
        };

        let poll0 = Instant::now();
        ctx.agent.poll(&mut sim);
        tp.poll_ms = elapsed_ms(poll0);
        tp.vision_results = ctx.agent.take_poll_timings();

        if tp.vision_tick {
            let (rgb, render_timing) =
                capture_render_rgb_timed(&ctx.assets, &sim, &ctx.rt).await;
            tp.render = Some(render_timing);
            let submit0 = Instant::now();
            let ok = ctx.agent.try_submit_frame(tick as u32, rgb);
            tp.submit_ok = Some(ok);
            tp.submit_ms = elapsed_ms(submit0);
            if ok {
                last_submitted_tick = Some(tick as u32);
            }
        }

        let sim0 = Instant::now();
        let action = ctx.agent.action();
        sim.tick_with_action(&action_to_input(action), Some(action));
        tp.sim_tick_ms = elapsed_ms(sim0);

        tp.loop_total_ms = elapsed_ms(loop0);
        report.ticks.push(tp);

        if sim.is_episode_over() {
            eprintln!("profile: tick {tick} GameOver，提前结束");
            break;
        }
    }

    report.eval_loop_ms = elapsed_ms(wall0) - setup_ms;

    let drain0 = Instant::now();
    let mut drained: Vec<crate::trainer::agent::VisionWorkerTiming> = Vec::new();
    if let Some(target) = last_submitted_tick {
        eprintln!("profile: 等待视觉线程完成 tick<={target} …");
        let deadline = Instant::now() + VISION_WAIT_TIMEOUT;
        while Instant::now() < deadline {
            ctx.agent.poll(&mut sim);
            drained.extend(ctx.agent.take_poll_timings());
            if ctx.agent.worker_dead() {
                eprintln!("profile: 警告 — 视觉线程已退出");
                break;
            }
            if ctx.agent.last_applied_tick() >= Some(target) {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        ctx.agent.poll(&mut sim);
        drained.extend(ctx.agent.take_poll_timings());
    }
    report.drain_ms = elapsed_ms(drain0);
    report.merge_worker_timings(&drained);

    let teardown0 = Instant::now();
    ctx.agent.poll(&mut sim);
    report.teardown_ms = elapsed_ms(teardown0);
    report.wall_total_ms = elapsed_ms(wall0);

    report.print_summary(pace.vision_interval_ticks, max_ticks);
    Ok(report)
}

/// 跑训练 eval 循环，在 `capture_at_tick` 的感知帧截取 RGB + YOLO/OCR 结果。
pub async fn capture_training_frame(
    ctx: &mut TrainerEvalContext,
    genome: &Genome,
    episode_seed: u64,
    capture_at_tick: usize,
    max_ticks: usize,
    pace: TrainingPaceConfig,
    shaping: FitnessShapingConfig,
) -> Result<TrainingCapture> {
    pace.validate().map_err(|e| anyhow::anyhow!(e))?;
    ctx.agent.set_genome(genome.clone());
    ctx.agent.reset_vision_state();

    let mut sim = GameSim::new_training(ctx.map.clone(), episode_seed);
    sim.fitness.configure_shaping(shaping);
    let interval = pace.vision_interval_ticks as usize;
    let mut captured: Option<TrainingCapture> = None;

    let limit = max_ticks.max(capture_at_tick + 1);
    let vision_steps = capture_at_tick / interval + 1;
    let mut vision_step = 0usize;

    eprintln!("模拟到 tick {capture_at_tick}（约 {vision_steps} 次 YOLO+OCR，请耐心等待）…");

    for tick in 0..limit {
        if tick % interval == 0 {
            vision_step += 1;
            eprintln!("  [{vision_step}/{vision_steps}] 渲染+感知 tick {tick}…");
            let rgb = capture_render_rgb(&ctx.assets, &sim, &ctx.rt).await;
            let rgb_for_save = if tick == capture_at_tick {
                Some(rgb.clone())
            } else {
                None
            };
            ctx.agent.submit_and_wait(
                &mut sim,
                tick as u32,
                rgb,
                VISION_WAIT_TIMEOUT,
            )?;
            eprintln!("  [{vision_step}/{vision_steps}] tick {tick} 感知完成");

            if let Some(rgb) = rgb_for_save {
                if let Some(step) = ctx.agent.vision().cloned() {
                    captured = Some(TrainingCapture {
                        tick: capture_at_tick,
                        rgb,
                        step,
                        fitness: sim.fitness.score,
                    });
                }
                break;
            }
        }

        let action = ctx.agent.action();
        sim.tick_with_action(&action_to_input(action), Some(action));

        if sim.is_episode_over() {
            eprintln!("  tick {tick} GameOver，提前结束");
            break;
        }
    }

    captured.ok_or_else(|| {
        anyhow::anyhow!(
            "capture_at_tick={capture_at_tick} 未触发感知（pace={}，最近视觉 tick={:?}，可能已 GameOver）",
            pace.vision_interval_ticks,
            ctx.agent.last_applied_tick()
        )
    })
}
