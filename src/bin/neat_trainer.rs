//! NEAT 训练 CLI（离屏渲染 + 真实 YOLO+OCR 视觉适应度）。
//!
//! 默认并行模式：`--workers` 路槽位同时演化，每槽 1 个 CPU YOLO 推理线程；
//! 主线程负责 GL 离屏渲染轮转。个体死亡后立即从 peak 排名前列补位。
//!
//! ```powershell
//! # OBS 扩维后旧基因组不兼容，请始终 --fresh（当前 NEAT_OBS_DIM=21）
//! cargo run --release --bin neat_trainer -- --fresh --generations 3000 --population 50 --workers 16 --elite-breed 8 --detect-hz 10 --fitness-shaping 0.5
//! cargo run --release --bin neat_trainer -- --sequential --generations 50 --population 10
//! ```

use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use macroquad::prelude::*;
use mxd_tools::game::fitness::FitnessShapingConfig;
use mxd_tools::game::{load_default_map, VisionPaceConfig};
use mxd_tools::headless_gl;
use mxd_tools::neat::{
    create_parallel_slots, evaluate_genome, export_innovation_state, prepare_parallel_assets,
    prepare_vision_env, rank_fitness, restore_innovation_state, restore_innovations_from_population,
    run_parallel_trainer, save_best_if_improved, save_session_best, ParallelCheckpointConfig,
    ParallelTrainerConfig, Population, PopulationConfig, TrainerShared, TrainingCheckpoint,
    DEFAULT_BEST_GENOME_FILE, DEFAULT_SESSION_BEST_FILE,
};

use ::rand::{rngs::StdRng, SeedableRng};

fn window_conf() -> Conf {
    headless_gl::headless_window_conf("neat_trainer")
}

struct Cli {
    generations: u32,
    population: usize,
    workers: usize,
    sequential: bool,
    elite_breed: usize,
    detect_hz: f32,
    max_ticks: u32,
    model: PathBuf,
    seed: u64,
    checkpoint: PathBuf,
    best_genome: PathBuf,
    session_best: PathBuf,
    fresh: bool,
    fitness_shaping: FitnessShapingConfig,
}

impl Cli {
    fn parse(args: &[String]) -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let population = arg_usize(args, "--population", 20);
        let sequential = args.iter().any(|a| a == "--sequential");
        Self {
            generations: arg_u32(args, "--generations", 3000),
            population,
            workers: if sequential {
                1
            } else {
                arg_usize(args, "--workers", 10.min(population))
            },
            sequential,
            elite_breed: arg_usize(args, "--elite-breed", (population / 3).max(4).min(population)),
            detect_hz: arg_f32(args, "--detect-hz", 10.0),
            max_ticks: arg_u32(args, "--max-ticks", 18_000),
            model: arg_path(args, "--model").unwrap_or_else(default_yolo_model_path),
            seed: arg_u64(args, "--seed", 42),
            checkpoint: arg_path(args, "--checkpoint")
                .unwrap_or_else(|| manifest.join("tmp/neat_checkpoint.json")),
            best_genome: arg_path(args, "--best-genome")
                .unwrap_or_else(|| manifest.join(DEFAULT_BEST_GENOME_FILE)),
            session_best: arg_path(args, "--session-best")
                .unwrap_or_else(|| manifest.join(DEFAULT_SESSION_BEST_FILE)),
            fresh: args.iter().any(|a| a == "--fresh"),
            fitness_shaping: FitnessShapingConfig {
                memory_weight: arg_f32(args, "--fitness-shaping", 0.5),
                hint_weight: arg_f32(args, "--hint-shaping", 0.35),
                vertical_rewards: true,
            },
        }
    }
}

fn default_yolo_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("onnx/yolo_nangang_e3000_best.onnx")
}

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1))
}

fn arg_u32(args: &[String], key: &str, default: u32) -> u32 {
    arg_value(args, key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    arg_value(args, key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    arg_value(args, key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn arg_f32(args: &[String], key: &str, default: f32) -> f32 {
    arg_value(args, key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn arg_path(args: &[String], key: &str) -> Option<PathBuf> {
    arg_value(args, key).map(PathBuf::from)
}

fn load_or_create_population(cli: &Cli) -> anyhow::Result<(Population, u32, u32)> {
    if !cli.fresh && cli.checkpoint.is_file() {
        let text = fs::read_to_string(&cli.checkpoint)?;
        let ckpt: TrainingCheckpoint = serde_json::from_str(&text)?;
        let mut pop = ckpt.population;
        if ckpt.innovation.entries.is_empty() {
            restore_innovations_from_population(&pop);
        } else {
            restore_innovation_state(&ckpt.innovation);
        }
        if pop.config.size != cli.population {
            let mut rng = StdRng::seed_from_u64(cli.seed);
            pop.resize_to(cli.population, &mut rng);
        }
        pop.config.elite_breed_count = cli.elite_breed.max(1).min(cli.population);
        let next_id = ckpt.next_spawn_id.max(ckpt.spawns_completed);
        return Ok((pop, ckpt.spawns_completed, next_id));
    }

    let mut rng = StdRng::seed_from_u64(cli.seed);
    let pop = Population::new(
        PopulationConfig {
            size: cli.population,
            elite_breed_count: cli.elite_breed.max(1).min(cli.population),
            ..PopulationConfig::default()
        },
        &mut rng,
    );
    Ok((pop, 0, 0))
}

fn save_checkpoint(
    path: &PathBuf,
    population: &Population,
    seed: u64,
    spawns_completed: u32,
    total_spawn_target: u32,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let ckpt = TrainingCheckpoint {
        population: population.clone(),
        seed,
        innovation: export_innovation_state(),
        spawns_completed,
        next_spawn_id: spawns_completed,
        total_spawn_target,
    };
    let json = serde_json::to_string_pretty(&ckpt)?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, json)?;
    fs::rename(tmp, path)?;
    Ok(())
}

#[macroquad::main(window_conf)]
async fn main() {
    if headless_gl::hide_gl_window() {
        // hidden GL placeholder
    }
    let args: Vec<String> = env::args().collect();
    let cli = Cli::parse(&args);

    if let Err(e) = run_trainer(&cli).await {
        eprintln!("训练失败: {e:#}");
        std::process::exit(1);
    }
}

async fn run_trainer(cli: &Cli) -> anyhow::Result<()> {
    let (population, spawns_completed, next_spawn_id) = load_or_create_population(cli)?;
    let total_spawns = cli.generations.max(1);
    let map = load_default_map()?;
    let pace = VisionPaceConfig::from_detect_hz(cli.detect_hz);
    let vision_interval = pace.vision_interval_ticks;

    if cli.sequential || cli.workers <= 1 {
        run_sequential_trainer(cli, population, spawns_completed, &map, vision_interval, total_spawns)
            .await
    } else {
        run_parallel_trainer_main(
            cli,
            population,
            spawns_completed,
            next_spawn_id,
            &map,
            vision_interval,
            total_spawns,
        )
        .await
    }
}

async fn run_parallel_trainer_main(
    cli: &Cli,
    population: Population,
    spawns_completed: u32,
    next_spawn_id: u32,
    map: &mxd_tools::game::map::GameMap,
    vision_interval: u32,
    total_spawns: u32,
) -> anyhow::Result<()> {
    eprintln!(
        "NEAT 并行持续进化: 目标局数={} 基因库={} 并行槽位={} YOLO线程={} elite={} detect_hz={} device=cpu",
        total_spawns,
        cli.population,
        cli.workers,
        cli.workers,
        cli.elite_breed,
        cli.detect_hz,
    );

    if cli.fresh && cli.checkpoint.is_file() {
        let _ = fs::remove_file(&cli.checkpoint);
    }

    let assets = prepare_parallel_assets().await?;
    let mut slots = create_parallel_slots(cli.workers, &cli.model)?;
    let shared = Mutex::new(TrainerShared {
        population,
        rng: StdRng::seed_from_u64(cli.seed.wrapping_add(spawns_completed as u64)),
        spawns_completed,
        next_spawn_id,
    });

    let parallel_cfg = ParallelTrainerConfig {
        workers: cli.workers,
        total_spawns,
        max_ticks: cli.max_ticks,
        vision_interval,
        seed: cli.seed,
        shaping: cli.fitness_shaping,
        checkpoint: Some(ParallelCheckpointConfig {
            path: cli.checkpoint.clone(),
            best_genome: cli.best_genome.clone(),
            session_best: cli.session_best.clone(),
        }),
    };

    run_parallel_trainer(
        parallel_cfg,
        &cli.model,
        map,
        &assets,
        &shared,
        &mut slots,
    )
    .await?;

    let s = shared.lock().expect("trainer lock");
    eprintln!(
        "训练完成: best_rank={:.2} gen={} completed={} → {}",
        rank_fitness(&s.population.best_ever),
        s.population.generation,
        s.spawns_completed,
        cli.best_genome.display()
    );
    Ok(())
}

async fn run_sequential_trainer(
    cli: &Cli,
    mut population: Population,
    mut spawns_completed: u32,
    map: &mxd_tools::game::map::GameMap,
    vision_interval: u32,
    total_spawns: u32,
) -> anyhow::Result<()> {
    eprintln!(
        "NEAT 顺序持续进化: 目标局数={} 基因库={} detect_hz={} device=cpu",
        total_spawns,
        cli.population,
        cli.detect_hz,
    );

    let mut vision = prepare_vision_env(Some(&cli.model)).await?;
    let mut rng = StdRng::seed_from_u64(cli.seed.wrapping_add(spawns_completed as u64));
    let mut last_saved_fitness = f32::NEG_INFINITY;
    let mut last_session_fitness = f32::NEG_INFINITY;
    let started = std::time::Instant::now();
    let mut last_status = std::time::Instant::now();
    let mut window_done: u32 = 0;
    let mut window_peak_sum: f32 = 0.0;
    let mut window_final_sum: f32 = 0.0;
    let mut window_peak_max: f32 = f32::NEG_INFINITY;
    let mut last_completed_for_rate = spawns_completed;

    if cli.fresh && cli.checkpoint.is_file() {
        let _ = fs::remove_file(&cli.checkpoint);
    }

    while spawns_completed < total_spawns {
        let spawn_id = spawns_completed;
        let genome = population.spawn_offspring(&mut rng, spawn_id);
        let episode_seed = cli.seed.wrapping_add(spawn_id as u64);

        let mut scored = genome;
        let shaping = cli.fitness_shaping.with_curriculum(population.generation);
        let outcome = evaluate_genome(
            &mut vision,
            map,
            &scored,
            episode_seed,
            cli.max_ticks,
            vision_interval,
            shaping,
        )
        .await?;
        scored.fitness = outcome.final_fitness;
        scored.peak_fitness = outcome.peak_fitness;
        window_done += 1;
        window_peak_sum += outcome.peak_fitness;
        window_final_sum += outcome.final_fitness;
        window_peak_max = window_peak_max.max(outcome.peak_fitness);

        population.on_eval_complete(scored);
        population.bump_virtual_generation(spawns_completed + 1);
        spawns_completed += 1;

        let _ = save_best_if_improved(
            &cli.best_genome,
            &population,
            cli.seed,
            &mut last_saved_fitness,
        );
        if rank_fitness(&population.best_ever) > last_session_fitness {
            save_session_best(
                &cli.session_best,
                &population.best_ever,
                population.generation,
                cli.seed,
            )?;
            last_session_fitness = rank_fitness(&population.best_ever);
        }
        save_checkpoint(
            &cli.checkpoint,
            &population,
            cli.seed,
            spawns_completed,
            total_spawns,
        )?;

        if last_status.elapsed() >= std::time::Duration::from_secs(5) {
            let elapsed = last_status.elapsed().as_secs_f32().max(0.001);
            let rate = (spawns_completed.saturating_sub(last_completed_for_rate)) as f32 / elapsed;
            let avg_peak = window_peak_sum / window_done.max(1) as f32;
            let avg_final = window_final_sum / window_done.max(1) as f32;
            eprintln!(
                "[{:>5.0}s] done={}/{} (+{}) rate={:.2}/s gen={} best_rank={:.1} window: peak_avg={:.1} peak_max={:.1} final_avg={:.1}",
                started.elapsed().as_secs_f32(),
                spawns_completed,
                total_spawns,
                window_done,
                rate,
                population.generation,
                rank_fitness(&population.best_ever),
                avg_peak,
                window_peak_max,
                avg_final,
            );
            window_done = 0;
            window_peak_sum = 0.0;
            window_final_sum = 0.0;
            window_peak_max = f32::NEG_INFINITY;
            last_status = std::time::Instant::now();
            last_completed_for_rate = spawns_completed;
        }
    }

    eprintln!(
        "训练完成: best_rank={:.2} gen={} → {}",
        rank_fitness(&population.best_ever),
        population.generation,
        cli.best_genome.display()
    );
    Ok(())
}
