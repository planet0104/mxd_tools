//! NEAT 训练 CLI（离屏渲染 + 本地 CPU YOLO+OCR 视觉适应度）。
//!
//! 默认 `--workers` = `--population`：每个基因组独立一局 eval（与真实单人游戏一致）。
//!
//! ```powershell
//! cargo run --release --bin neat_trainer -- `
//!   --generations 3000 --population 40 --pace 12 --max-ticks 18000
//!
//! # --generations = 总共出生的个体数；--population = 并行槽位与选择库规模
//!
//! # 可视化调试（建议 --workers 1）
//! cargo run --release --bin neat_trainer -- --visible --workers 1 --population 1
//! ```

use std::collections::{HashMap, VecDeque};
use std::env;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use macroquad::prelude::*;
use mxd_tools::game::view;
use mxd_tools::game::{FitnessShapingConfig, TrainingPaceConfig, VisionAnchorConfig, VisionPipeline, NEAT_CONF_THRESH};
use mxd_tools::headless_gl;
use mxd_tools::neat::{
    export_innovation_state, restore_innovation_state, restore_innovations_from_population,
    save_best_if_improved, save_session_best, BestGenomeSnapshot, Genome, Population,
    PopulationConfig, TrainingCheckpoint, DEFAULT_BEST_GENOME_FILE, DEFAULT_SESSION_BEST_FILE,
};
use mxd_tools::yolo::YoloDevice;
use ::rand::SeedableRng;
use mxd_tools::train_log;
use mxd_tools::trainer::{
    evaluate_genome, evaluate_genome_profile, log_steady_heartbeat, EvalProgressConfig,
    TrainerEvalContext, WorkerPool, WorkerPoolConfig, HEARTBEAT_INTERVAL,
};
use mxd_tools::trainer::worker_protocol::{
    read_json_line, write_json_line, WorkerJobRequest, WorkerJobResponse, WorkerQuit, WorkerReady,
};

fn window_conf() -> Conf {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--worker-daemon") {
        let id = arg_usize(&args, "--worker-id", 0);
        headless_gl::headless_window_conf(format!("neat_worker_{id}"))
    } else if args.iter().any(|a| a == "--visible") {
        headless_gl::visible_training_window_conf("NEAT 训练")
    } else {
        headless_gl::headless_window_conf("neat_trainer")
    }
}

struct Cli {
    generations: u32,
    population: usize,
    workers: Option<usize>,
    pace: TrainingPaceConfig,
    max_ticks: usize,
    model: PathBuf,
    seed: u64,
    checkpoint: PathBuf,
    best_genome: PathBuf,
    fresh: bool,
    worker_eval: bool,
    worker_daemon: bool,
    worker_id: usize,
    genome_file: Option<PathBuf>,
    status_file: Option<PathBuf>,
    worker_label: Option<String>,
    fitness_shaping: FitnessShapingConfig,
    reset_best: bool,
    profile: bool,
    profile_ticks: usize,
    visible: bool,
    no_ocr: bool,
    anchor_offset: f32,
}

impl Cli {
    fn parse(args: &[String]) -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let default_model = manifest.join("models/yolo_nangang_e3000_best.onnx");
        let pace_ticks = arg_u32(args, "--pace", 12);
        Self {
            generations: arg_u32(args, "--generations", 50) as u32,
            population: arg_usize(args, "--population", 50),
            workers: arg_value(args, "--workers").and_then(|s| s.parse().ok()),
            pace: TrainingPaceConfig {
                vision_interval_ticks: pace_ticks.max(1),
            },
            max_ticks: arg_usize(args, "--max-ticks", 18000),
            model: arg_path(args, "--model").unwrap_or(default_model),
            seed: arg_u64(args, "--seed", 42),
            checkpoint: arg_path(args, "--checkpoint")
                .unwrap_or_else(|| manifest.join("tmp/neat_checkpoint.json")),
            best_genome: arg_path(args, "--best-genome")
                .unwrap_or_else(|| manifest.join(DEFAULT_BEST_GENOME_FILE)),
            fresh: args.iter().any(|a| a == "--fresh"),
            worker_eval: args.iter().any(|a| a == "--worker-eval"),
            worker_daemon: args.iter().any(|a| a == "--worker-daemon"),
            worker_id: arg_usize(args, "--worker-id", 0),
            genome_file: arg_path(args, "--genome-file"),
            status_file: arg_path(args, "--status-file"),
            worker_label: arg_value(args, "--worker-label").cloned(),
            fitness_shaping: FitnessShapingConfig {
                memory_weight: arg_f32(args, "--fitness-shaping", 0.25),
            },
            reset_best: args.iter().any(|a| a == "--reset-best"),
            profile: args.iter().any(|a| a == "--profile"),
            profile_ticks: arg_usize(args, "--profile-ticks", 32),
            visible: args.iter().any(|a| a == "--visible"),
            no_ocr: args.iter().any(|a| a == "--no-ocr"),
            anchor_offset: arg_f32(args, "--anchor-offset", 10.0),
        }
    }

    fn vision_anchor(&self) -> VisionAnchorConfig {
        if self.no_ocr {
            VisionAnchorConfig::sim_match(self.anchor_offset)
        } else {
            VisionAnchorConfig::ocr()
        }
    }
}

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1))
}

fn arg_u32(args: &[String], key: &str, default: u32) -> u32 {
    arg_value(args, key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    arg_value(args, key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    arg_value(args, key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn arg_f32(args: &[String], key: &str, default: f32) -> f32 {
    arg_value(args, key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn arg_path(args: &[String], key: &str) -> Option<PathBuf> {
    arg_value(args, key).map(PathBuf::from)
}

fn resolve_parallel(cli: &Cli, pop_size: usize) -> usize {
    let pop_size = pop_size.max(1);
    match cli.workers {
        None => pop_size,
        Some(w) => {
            let w = w.max(1);
            if w != pop_size {
                train_log!(
                    "警告: --workers {w} != 种群 {pop_size}；建议二者相同以满并行评估"
                );
            }
            w
        }
    }
}

fn population_size(pop: &Population) -> usize {
    pop.genomes().count()
}

fn load_training_pipeline(model: &PathBuf, anchor: VisionAnchorConfig) -> anyhow::Result<VisionPipeline> {
    if !model.is_file() {
        anyhow::bail!("找不到 YOLO ONNX: {}", model.display());
    }
    VisionPipeline::load(model, YoloDevice::Cpu, NEAT_CONF_THRESH)
        .map(|p| p.with_anchor(anchor))
        .map_err(|e| anyhow::anyhow!("加载视觉管线失败: {e}"))
}

#[macroquad::main(window_conf)]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if headless_gl::should_hide_gl_window(&args) {
        if !headless_gl::hide_gl_window() {
            eprintln!("headless: 未能隐藏 GL 占位窗（训练仍可继续）");
        }
    }
    let cli = Cli::parse(&args);

    if cli.worker_daemon {
        if let Err(e) = run_worker_daemon(&cli).await {
            eprintln!("worker-daemon 失败: {e}");
            std::process::exit(1);
        }
        return;
    }

    if cli.worker_eval {
        if let Err(e) = run_worker_eval(&cli).await {
            eprintln!("worker-eval 失败: {e}");
            std::process::exit(1);
        }
        return;
    }

    if cli.profile {
        if let Err(e) = run_profile_eval(&cli).await {
            eprintln!("profile-eval 失败: {e}");
            std::process::exit(1);
        }
        return;
    }

    if let Err(e) = run_trainer(&cli).await {
        eprintln!("训练失败: {e}");
        std::process::exit(1);
    }
}

async fn run_worker_eval(cli: &Cli) -> anyhow::Result<()> {
    let genome_path = cli
        .genome_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--worker-eval 需要 --genome-file"))?;
    let text = fs::read_to_string(genome_path)?;
    let genome: Genome = serde_json::from_str(&text)?;

    let pipeline = load_training_pipeline(&cli.model, cli.vision_anchor())?;
    let rt = view::new_render_target();
    let mut ctx = TrainerEvalContext::prepare(pipeline, rt).await?;
    let episode_seed = cli.seed;
    let label = cli
        .worker_label
        .clone()
        .unwrap_or_else(|| format!("seed{episode_seed}"));
    let progress = EvalProgressConfig {
        label,
        status_file: cli.status_file.clone(),
        console: cli.status_file.is_none(),
    };
    let fitness = evaluate_genome(
        &mut ctx,
        &genome,
        episode_seed,
        cli.max_ticks,
        cli.pace,
        cli.fitness_shaping,
        Some(&progress),
        false,
    )
    .await?;
    ctx.shutdown_agent();
    use std::io::Write;
    println!("{fitness}");
    std::io::stdout().flush()?;
    Ok(())
}

async fn run_worker_daemon(cli: &Cli) -> anyhow::Result<()> {
    let worker_id = cli.worker_id;
    let pipeline = load_training_pipeline(&cli.model, cli.vision_anchor())?;
    let rt = view::new_render_target();
    let mut ctx = TrainerEvalContext::prepare(pipeline, rt).await?;

    let mut stdout = std::io::stdout();
    write_json_line(
        &mut stdout,
        &WorkerReady {
            ready: true,
            worker_id,
        },
    )?;

    let mut stdin = BufReader::new(std::io::stdin());
    loop {
        let line = match read_json_line(&mut stdin)? {
            Some(l) => l,
            None => break,
        };
        if let Ok(q) = serde_json::from_str::<WorkerQuit>(line.trim()) {
            if q.quit {
                break;
            }
        }
        let req: WorkerJobRequest = serde_json::from_str(line.trim())?;
        let progress = EvalProgressConfig {
            label: req.label.clone(),
            status_file: Some(PathBuf::from(&req.status_file)),
            console: false,
        };
        let resp = match evaluate_genome(
            &mut ctx,
            &req.genome,
            req.episode_seed,
            cli.max_ticks,
            cli.pace,
            cli.fitness_shaping,
            Some(&progress),
            false,
        )
        .await
        {
            Ok(fitness) => WorkerJobResponse {
                job_idx: req.job_idx,
                fitness,
                error: None,
            },
            Err(e) => WorkerJobResponse {
                job_idx: req.job_idx,
                fitness: 0.0,
                error: Some(e.to_string()),
            },
        };
        write_json_line(&mut stdout, &resp)?;
    }
    ctx.shutdown_agent();
    Ok(())
}

async fn run_profile_eval(cli: &Cli) -> anyhow::Result<()> {
    eprintln!(
        "NEAT profile 模式: ticks={} pace={} seed={}",
        cli.profile_ticks, cli.pace.vision_interval_ticks, cli.seed
    );

    let genome = load_profile_genome(cli)?;
    eprintln!("profile: 基因组 connections={}", genome.connections.len());

    let pipeline = load_training_pipeline(&cli.model, cli.vision_anchor())?;
    let rt = view::new_render_target();
    let mut ctx = TrainerEvalContext::prepare(pipeline, rt).await?;

    let _report = evaluate_genome_profile(
        &mut ctx,
        &genome,
        cli.seed,
        cli.profile_ticks,
        cli.pace,
        cli.fitness_shaping,
    )
    .await?;

    Ok(())
}

fn load_profile_genome(cli: &Cli) -> anyhow::Result<Genome> {
    if let Some(path) = &cli.genome_file {
        let text = fs::read_to_string(path)?;
        if let Ok(snap) = serde_json::from_str::<BestGenomeSnapshot>(&text) {
            return Ok(snap.genome);
        }
        return Ok(serde_json::from_str(&text)?);
    }
    if cli.best_genome.is_file() {
        if let Ok(text) = fs::read_to_string(&cli.best_genome) {
            if let Ok(snap) = serde_json::from_str::<BestGenomeSnapshot>(&text) {
                eprintln!("profile: 使用 {}", cli.best_genome.display());
                return Ok(snap.genome);
            }
        }
    }
    let mut rng = ::rand::rngs::StdRng::seed_from_u64(cli.seed);
    eprintln!("profile: 无基因组文件，使用 random_minimal");
    Ok(Genome::random_minimal(&mut rng))
}

async fn run_trainer(cli: &Cli) -> anyhow::Result<()> {
    let (mut population, mut progress) = load_or_create_population(cli)?;
    let parallel = resolve_parallel(cli, population.config.size);
    let total_spawns = cli.generations.max(1);

    train_log!(
        "NEAT 持续进化: 并行={} 目标出生={} 种群库={} pace={} max_ticks={} model={}",
        parallel,
        total_spawns,
        population.config.size,
        cli.pace.vision_interval_ticks,
        cli.max_ticks,
        cli.model.display()
    );
    if cli.workers.is_none() {
        train_log!("并行: 未指定 --workers，自动 workers=population={parallel}");
    }
    if cli.fresh {
        train_log!("模式: --fresh 从头训练（忽略已有检查点）");
        if cli.checkpoint.is_file() {
            fs::remove_file(&cli.checkpoint)?;
            train_log!("已删除旧检查点: {}", cli.checkpoint.display());
        }
        progress = SteadyProgress::fresh(total_spawns);
    } else if cli.checkpoint.is_file() {
        train_log!(
            "模式: 续训（已完成 {}/{} 出生）",
            progress.spawns_completed,
            total_spawns
        );
    } else {
        train_log!("模式: 新训练（无检查点，等同 --fresh）");
        progress = SteadyProgress::fresh(total_spawns);
    }
    train_log!(
        "说明: --generations={total_spawns} 表示共评估 {total_spawns} 个个体；任一 worker 结束即补种下一个"
    );
    train_log!(
        "适应度=拾取(主)+视觉shaping(上限100)+{:.2}×内存shaping−停滞惩罚",
        cli.fitness_shaping.memory_weight
    );
    if cli.no_ocr {
        train_log!(
            "视觉: --no-ocr SimMatch 锚点（跳过 OCR，偏移 ±{:.0}px）",
            cli.anchor_offset
        );
    } else {
        train_log!("视觉: 本地 CPU YOLO+OCR（单人 eval，相机跟随自身）");
    }
    if cli.visible {
        train_log!("窗口: --visible 可视化（建议 --workers 1，worker 子进程仍为 headless）");
        if parallel > 1 {
            train_log!(
                "警告: --visible 与 --workers {parallel} 并用时，主进程不跑 eval，画面不会出现；请用 --workers 1"
            );
        }
    } else {
        train_log!("窗口: headless（1×1 隐藏 GL 占位窗）");
    }

    let mut rng = ::rand::rngs::StdRng::seed_from_u64(cli.seed);
    let mut last_saved_best = init_best_tracking(cli, &mut population, progress.spawns_completed);
    if cli.reset_best {
        bootstrap_session_best(cli, &population)?;
        if population.best_ever.fitness > 0.0 {
            save_best_if_improved(
                &cli.best_genome,
                &population,
                cli.seed,
                &mut last_saved_best,
            )?;
        }
    }

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let session_best_path = manifest.join(DEFAULT_SESSION_BEST_FILE);

    let mut ctx = if parallel <= 1 {
        let pipeline = load_training_pipeline(&cli.model, cli.vision_anchor())?;
        let rt = view::new_render_target();
        Some(TrainerEvalContext::prepare(pipeline, rt).await?)
    } else {
        None
    };

    let mut pool = if parallel > 1 {
        train_log!("常驻 worker 池: {parallel} 进程（持续补种 eval）");
        Some(WorkerPool::spawn(&WorkerPoolConfig {
            workers: parallel,
            exe: env::current_exe()?,
            model: cli.model.clone(),
            max_ticks: cli.max_ticks,
            pace: cli.pace.vision_interval_ticks,
            fitness_shaping: cli.fitness_shaping.memory_weight,
            no_ocr: cli.no_ocr,
            anchor_offset: cli.anchor_offset,
        })?)
    } else {
        None
    };

    if let Some(parent) = cli.checkpoint.parent() {
        fs::create_dir_all(parent)?;
    }

    progress.total_spawn_target = total_spawns;
    let run_started = Instant::now();
    run_steady_state(
        cli,
        parallel,
        &mut population,
        &mut progress,
        ctx.as_mut(),
        pool.as_mut(),
        &mut rng,
        &mut last_saved_best,
        &session_best_path,
        run_started,
    )
    .await?;

    if let Some(pool) = pool {
        pool.shutdown()?;
    }
    if let Some(ctx) = ctx.as_mut() {
        ctx.shutdown_agent();
    }

    train_log!(
        "训练完成，出生 {}/{}，最佳适应度={:.2}，检查点 {}，最优基因组 {}",
        progress.spawns_completed,
        total_spawns,
        population.best_ever.fitness,
        cli.checkpoint.display(),
        cli.best_genome.display()
    );
    Ok(())
}

struct SteadyProgress {
    spawns_completed: u32,
    next_spawn_id: u32,
    total_spawn_target: u32,
}

impl SteadyProgress {
    fn fresh(total: u32) -> Self {
        Self {
            spawns_completed: 0,
            next_spawn_id: 0,
            total_spawn_target: total,
        }
    }
}

async fn run_steady_state(
    cli: &Cli,
    parallel: usize,
    population: &mut Population,
    progress: &mut SteadyProgress,
    ctx: Option<&mut TrainerEvalContext>,
    pool: Option<&mut WorkerPool>,
    rng: &mut ::rand::rngs::StdRng,
    last_saved_best: &mut f32,
    session_best_path: &PathBuf,
    run_started: Instant,
) -> anyhow::Result<()> {
    if progress.spawns_completed >= progress.total_spawn_target {
        train_log!("已达目标出生数，跳过训练");
        return Ok(());
    }

    let mut pending_initial: VecDeque<Genome> = if progress.spawns_completed == 0 {
        population.genomes().cloned().collect()
    } else {
        VecDeque::new()
    };

    let mut session_best = load_session_best_fitness(session_best_path).unwrap_or(0.0);
    if !session_best_path.is_file() {
        bootstrap_session_best_from_population(cli, population, session_best_path)?;
        session_best = load_session_best_fitness(session_best_path).unwrap_or(0.0);
    }
    let mut last_heartbeat = Instant::now();
    let mut last_checkpoint = Instant::now();

    if parallel <= 1 {
        let ctx = ctx.expect("parallel=1 需要 TrainerEvalContext");
        while progress.spawns_completed < progress.total_spawn_target {
            let spawn_id = progress.next_spawn_id;
            let genome = take_next_genome(population, rng, &mut pending_initial, spawn_id);
            progress.next_spawn_id += 1;

            let episode_seed = cli.seed.wrapping_add(spawn_id as u64);
            let label = format!("s{spawn_id}");
            train_log!(
                "评估 出生 {}/{} ({label})",
                progress.spawns_completed + 1,
                progress.total_spawn_target
            );
            let progress_cfg = EvalProgressConfig {
                label: label.clone(),
                status_file: None,
                console: true,
            };
            let score = evaluate_genome(
                ctx,
                &genome,
                episode_seed,
                cli.max_ticks,
                cli.pace,
                cli.fitness_shaping,
                Some(&progress_cfg),
                cli.visible,
            )
            .await?;

            let mut evaluated = genome;
            evaluated.fitness = score;
            if score > session_best {
                session_best = score;
            }
            maybe_record_session_best(
                cli,
                session_best_path,
                &evaluated,
                population.generation,
            )?;
            population.on_eval_complete(evaluated);
            progress.spawns_completed += 1;
            population.bump_virtual_generation(progress.spawns_completed);
            train_log!(
                "  完成 {label} 适应度={:.1} (累计 {}/{})",
                score,
                progress.spawns_completed,
                progress.total_spawn_target
            );
            save_best_if_improved(
                &cli.best_genome,
                population,
                cli.seed,
                last_saved_best,
            )?;
            if last_checkpoint.elapsed() >= Duration::from_secs(60) {
                save_checkpoint(cli, population, progress)?;
                last_checkpoint = Instant::now();
            }
        }
        save_checkpoint(cli, population, progress)?;
        return Ok(());
    }

    let pool = pool.expect("workers>1 需要 WorkerPool");
    let job_dir = pool.job_dir().to_path_buf();
    let mut inflight_genomes: HashMap<usize, Genome> = HashMap::new();

    while progress.spawns_completed < progress.total_spawn_target {
        while inflight_genomes.len() < parallel
            && progress.next_spawn_id < progress.total_spawn_target
        {
            let spawn_id = progress.next_spawn_id;
            let genome = take_next_genome(population, rng, &mut pending_initial, spawn_id);
            progress.next_spawn_id += 1;
            let episode_seed = cli.seed.wrapping_add(spawn_id as u64);
            let status_path = job_dir.join(format!("status_s{spawn_id}.txt"));
            let label = format!("s{spawn_id}");
            let spawn_id_usize = spawn_id as usize;
            if !pool.try_submit(spawn_id_usize, &genome, episode_seed, &label, &status_path)? {
                progress.next_spawn_id = spawn_id;
                break;
            }
            inflight_genomes.insert(spawn_id_usize, genome);
        }

        if last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
            last_heartbeat = Instant::now();
            log_steady_heartbeat(
                progress.spawns_completed,
                progress.total_spawn_target,
                &pool.in_flight_status(),
                run_started,
                session_best,
                population.best_ever.fitness,
            );
        }

        if let Some(result) = pool.try_recv_done() {
            let (spawn_id, score) = result?;
            let mut evaluated = inflight_genomes
                .remove(&spawn_id)
                .ok_or_else(|| anyhow::anyhow!("收到未知出生 s{spawn_id} 的结果"))?;
            evaluated.fitness = score;
            if score > session_best {
                session_best = score;
            }
            maybe_record_session_best(
                cli,
                session_best_path,
                &evaluated,
                population.generation,
            )?;
            population.on_eval_complete(evaluated);
            progress.spawns_completed += 1;
            population.bump_virtual_generation(progress.spawns_completed);
            train_log!(
                "  完成 s{spawn_id} 适应度={:.1} (累计 {}/{})",
                score,
                progress.spawns_completed,
                progress.total_spawn_target
            );
            save_best_if_improved(
                &cli.best_genome,
                population,
                cli.seed,
                last_saved_best,
            )?;
            if last_checkpoint.elapsed() >= Duration::from_secs(60) {
                save_checkpoint(cli, population, progress)?;
                last_checkpoint = Instant::now();
            }
        } else if inflight_genomes.is_empty()
            && progress.next_spawn_id >= progress.total_spawn_target
        {
            break;
        } else {
            thread::sleep(Duration::from_millis(50));
        }
    }

    save_checkpoint(cli, population, progress)?;
    Ok(())
}

fn take_next_genome(
    population: &mut Population,
    rng: &mut ::rand::rngs::StdRng,
    pending_initial: &mut VecDeque<Genome>,
    spawn_id: u32,
) -> Genome {
    if let Some(g) = pending_initial.pop_front() {
        g
    } else {
        population.spawn_offspring(rng, spawn_id)
    }
}

fn init_best_tracking(cli: &Cli, population: &mut Population, spawns_completed: u32) -> f32 {
    if cli.fresh || spawns_completed == 0 {
        population.best_ever.fitness = 0.0;
        population.best_ever.adjusted_fitness = 0.0;
        return 0.0;
    }

    if cli.reset_best {
        train_log!(
            "--reset-best: 忽略历史 neat_best_genome 分数门槛，本规则从种群最高已评估分重计"
        );
        population.best_ever.fitness = 0.0;
        population.best_ever.adjusted_fitness = 0.0;
        seed_best_from_population(population);
        train_log!(
            "本规则种群最高: fitness={:.2}",
            population.best_ever.fitness
        );
        return 0.0;
    }

    let mut last_saved = population.best_ever.fitness;
    if cli.best_genome.is_file() {
        if let Ok(snap) = BestGenomeSnapshot::load(&cli.best_genome) {
            last_saved = last_saved.max(snap.fitness);
            if snap.fitness > population.best_ever.fitness {
                let adj = snap.genome.adjusted_fitness;
                population.best_ever = snap.genome;
                population.best_ever.fitness = snap.fitness;
                population.best_ever.adjusted_fitness = adj;
            }
        }
    }
    train_log!(
        "续训历史最优: fitness={:.2}（检查点 + neat_best_genome 取高；本规则 session 见 {}）",
        population.best_ever.fitness,
        DEFAULT_SESSION_BEST_FILE
    );
    last_saved
}

fn seed_best_from_population(population: &mut Population) {
    if let Some(best) = population
        .genomes()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
    {
        if best.fitness > population.best_ever.fitness {
            population.best_ever = best.clone();
        }
    }
}

fn load_session_best_fitness(path: &PathBuf) -> anyhow::Result<f32> {
    Ok(BestGenomeSnapshot::load(path)?.fitness)
}

fn bootstrap_session_best(cli: &Cli, population: &Population) -> anyhow::Result<()> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_SESSION_BEST_FILE);
    bootstrap_session_best_from_population(cli, population, &path)
}

fn bootstrap_session_best_from_population(
    cli: &Cli,
    population: &Population,
    path: &PathBuf,
) -> anyhow::Result<()> {
    let Some(best) = population
        .genomes()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
    else {
        return Ok(());
    };
    if best.fitness <= 0.0 {
        return Ok(());
    }
    save_session_best(path, best, population.generation, cli.seed)
}

fn maybe_record_session_best(
    cli: &Cli,
    path: &PathBuf,
    genome: &Genome,
    generation: u32,
) -> anyhow::Result<()> {
    let prev = load_session_best_fitness(path).unwrap_or(0.0);
    if genome.fitness <= prev + 1e-3 {
        return Ok(());
    }
    save_session_best(path, genome, generation, cli.seed)
}

fn load_or_create_population(cli: &Cli) -> anyhow::Result<(Population, SteadyProgress)> {
    if cli.checkpoint.exists() && !cli.fresh {
        let text = fs::read_to_string(&cli.checkpoint)?;
        let ck: TrainingCheckpoint = serde_json::from_str(&text)?;
        train_log!("从检查点恢复: 代 {}", ck.population.generation);
        if !ck.innovation.entries.is_empty() || ck.innovation.next_id > 1 {
            restore_innovation_state(&ck.innovation);
        } else {
            restore_innovations_from_population(&ck.population);
        }
        let mut population = ck.population;
        if cli.population != population.config.size {
            train_log!(
                "种群规模调整: {} -> {}（保留精英与历史最优）",
                population.config.size,
                cli.population
            );
            let mut rng = ::rand::rngs::StdRng::seed_from_u64(
                cli.seed.wrapping_add(population.generation as u64),
            );
            population.resize_to(cli.population, &mut rng);
        }
        let total = if ck.total_spawn_target > 0 {
            ck.total_spawn_target
        } else {
            cli.generations.max(1)
        };
        let progress = SteadyProgress {
            spawns_completed: ck.spawns_completed,
            next_spawn_id: ck.next_spawn_id,
            total_spawn_target: total,
        };
        return Ok((population, progress));
    }
    if cli.fresh {
        train_log!("创建新种群: size={} seed={}", cli.population, cli.seed);
    }
    let mut rng = ::rand::rngs::StdRng::seed_from_u64(cli.seed);
    let population = Population::new(
        PopulationConfig {
            size: cli.population,
            ..Default::default()
        },
        &mut rng,
    );
    Ok((
        population,
        SteadyProgress::fresh(cli.generations.max(1)),
    ))
}

fn save_checkpoint(
    cli: &Cli,
    population: &Population,
    progress: &SteadyProgress,
) -> anyhow::Result<()> {
    let ck = TrainingCheckpoint {
        population: population.clone(),
        seed: cli.seed,
        innovation: export_innovation_state(),
        spawns_completed: progress.spawns_completed,
        next_spawn_id: progress.next_spawn_id,
        total_spawn_target: progress.total_spawn_target,
    };
    let json = serde_json::to_string_pretty(&ck)?;
    let tmp = cli.checkpoint.with_extension("json.tmp");
    fs::write(&tmp, &json)?;
    fs::rename(&tmp, &cli.checkpoint)?;
    Ok(())
}
