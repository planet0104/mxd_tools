//! NEAT 训练 CLI（离屏渲染 + YOLO+OCR 视觉适应度）。
//!
//! ```powershell
//! cargo run --release --bin neat_trainer -- --generations 2 --population 8 --max-ticks 300 --pace 4
//! cargo run --release --bin neat_trainer -- --workers 4 --generations 10 --population 20
//! ```

use std::env;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use macroquad::prelude::*;
use mxd_tools::game::view;
use mxd_tools::game::{FitnessShapingConfig, NEAT_CONF_THRESH, TrainingPaceConfig, VisionPipeline};
use mxd_tools::headless_gl;
use mxd_tools::yolo::YoloDevice;
use mxd_tools::neat::{
    export_innovation_state, restore_innovation_state, restore_innovations_from_population,
    save_best_if_improved, BestGenomeSnapshot, Genome, Population, PopulationConfig,
    TrainingCheckpoint, DEFAULT_BEST_GENOME_FILE,
};
use ::rand::SeedableRng;
use mxd_tools::train_log;
use mxd_tools::trainer::{
    evaluate_genome, evaluate_genome_profile, log_pool_heartbeat, EvalProgressConfig,
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
    } else {
        headless_gl::headless_window_conf("neat_trainer")
    }
}

struct Cli {
    generations: u32,
    population: usize,
    workers: usize,
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
    profile: bool,
    profile_ticks: usize,
}

impl Cli {
    fn parse(args: &[String]) -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let default_model = manifest.join("models/yolo_nangang_e3000_best.onnx");
        let pace_ticks = arg_u32(args, "--pace", 4);
        Self {
            generations: arg_u32(args, "--generations", 50) as u32,
            population: arg_usize(args, "--population", 50),
            workers: arg_usize(args, "--workers", 1).max(1),
            pace: TrainingPaceConfig {
                vision_interval_ticks: pace_ticks.max(1),
            },
            max_ticks: arg_usize(args, "--max-ticks", 3600),
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
                memory_weight: arg_f32(args, "--fitness-shaping", 0.2),
            },
            profile: args.iter().any(|a| a == "--profile"),
            profile_ticks: arg_usize(args, "--profile-ticks", 32),
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

#[macroquad::main(window_conf)]
async fn main() {
    if !headless_gl::hide_gl_window() {
        eprintln!("headless: 未能隐藏 GL 占位窗（训练仍可继续）");
    }
    let args: Vec<String> = env::args().collect();
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

    let rt = view::new_render_target();
    let pipeline = VisionPipeline::load(&cli.model, YoloDevice::Cpu, NEAT_CONF_THRESH)?;
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
    let rt = view::new_render_target();
    let pipeline = VisionPipeline::load(&cli.model, YoloDevice::Cpu, NEAT_CONF_THRESH)?;
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
    use ::rand::SeedableRng;

    eprintln!(
        "NEAT profile 模式: ticks={} pace={} seed={} model={}",
        cli.profile_ticks,
        cli.pace.vision_interval_ticks,
        cli.seed,
        cli.model.display()
    );

    let genome = load_profile_genome(cli)?;
    eprintln!("profile: 基因组 connections={}", genome.connections.len());

    let rt = view::new_render_target();
    let pipeline = VisionPipeline::load(&cli.model, YoloDevice::Cpu, NEAT_CONF_THRESH)?;
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
    train_log!(
        "NEAT 训练: pop={} gen={} workers={} pace={} max_ticks={} model={}",
        cli.population,
        cli.generations,
        cli.workers,
        cli.pace.vision_interval_ticks,
        cli.max_ticks,
        cli.model.display()
    );
    if cli.fresh {
        train_log!("模式: --fresh 从头训练（忽略已有检查点）");
        if cli.checkpoint.is_file() {
            fs::remove_file(&cli.checkpoint)?;
            train_log!("已删除旧检查点: {}", cli.checkpoint.display());
        }
    } else if cli.checkpoint.is_file() {
        train_log!("模式: 续训（从 {} 恢复）", cli.checkpoint.display());
    } else {
        train_log!("模式: 新训练（无检查点，等同 --fresh）");
    }
    train_log!(
        "适应度=拾取+视觉shaping+{:.2}×内存shaping（主目标仍为 YOLO 可见拾取）",
        cli.fitness_shaping.memory_weight
    );

    let mut population = load_or_create_population(cli)?;
    let mut rng = ::rand::rngs::StdRng::seed_from_u64(cli.seed);
    let mut last_saved_best = f32::NEG_INFINITY;

    let mut ctx = if cli.workers <= 1 {
        let rt = view::new_render_target();
        let pipeline = VisionPipeline::load(&cli.model, YoloDevice::Cpu, NEAT_CONF_THRESH)?;
        Some(TrainerEvalContext::prepare(pipeline, rt).await?)
    } else {
        None
    };

    let mut pool = if cli.workers > 1 {
        train_log!(
            "常驻 worker 池: {} 进程（各 1 隐藏 GL 窗，训练期间复用）",
            cli.workers
        );
        Some(WorkerPool::spawn(&WorkerPoolConfig {
            workers: cli.workers,
            exe: env::current_exe()?,
            model: cli.model.clone(),
            max_ticks: cli.max_ticks,
            pace: cli.pace.vision_interval_ticks,
            fitness_shaping: cli.fitness_shaping.memory_weight,
        })?)
    } else {
        None
    };

    if let Some(parent) = cli.checkpoint.parent() {
        fs::create_dir_all(parent)?;
    }
    let _ = save_best_if_improved(
        &cli.best_genome,
        &population,
        cli.seed,
        &mut last_saved_best,
    );

    for gen in 0..cli.generations {
        let t0 = Instant::now();
        train_log!(
            "—— 开始评估 代 {} (种群第 {} 代) ——",
            gen + 1,
            population.generation
        );
        evaluate_population(
            cli,
            &mut population,
            ctx.as_mut(),
            pool.as_mut(),
            &mut rng,
            gen,
            cli.generations,
            &mut last_saved_best,
        )
        .await?;
        population.evolve(&mut rng);

        let elapsed = t0.elapsed();
        train_log!(
            "代 {}/{} 最佳适应度={:.2} 用时 {:.1}s",
            gen + 1,
            cli.generations,
            population.best_ever.fitness,
            elapsed.as_secs_f64()
        );

        save_checkpoint(cli, &population)?;
        let _ = save_best_if_improved(
            &cli.best_genome,
            &population,
            cli.seed,
            &mut last_saved_best,
        )?;
    }

    if let Some(pool) = pool {
        pool.shutdown()?;
    }

    train_log!(
        "训练完成，最佳适应度={:.2}，检查点 {}，最优基因组 {}",
        population.best_ever.fitness,
        cli.checkpoint.display(),
        cli.best_genome.display()
    );
    Ok(())
}

async fn evaluate_population(
    cli: &Cli,
    population: &mut Population,
    ctx: Option<&mut TrainerEvalContext>,
    pool: Option<&mut WorkerPool>,
    rng: &mut ::rand::rngs::StdRng,
    generation: u32,
    generations_total: u32,
    last_saved_best: &mut f32,
) -> anyhow::Result<()> {
    if cli.workers <= 1 {
        let ctx = ctx.expect("workers=1 需要 TrainerEvalContext");
        let mut idx = 0usize;
        let mut best_threshold = population.best_ever.fitness;
        let mut best_candidate: Option<Genome> = None;
        let pop_gen = population.generation;
        for genome in population.genomes_mut() {
            let episode_seed =
                cli.seed.wrapping_add(generation as u64 * 100_000).wrapping_add(idx as u64);
            let label = format!("gen{pop_gen}#{idx}");
            train_log!("评估基因组 {}/{} ({label})", idx + 1, cli.population);
            let progress = EvalProgressConfig {
                label: label.clone(),
                status_file: None,
                console: true,
            };
            genome.fitness = evaluate_genome(
                ctx,
                genome,
                episode_seed,
                cli.max_ticks,
                cli.pace,
                cli.fitness_shaping,
                Some(&progress),
            )
            .await?;
            train_log!("  完成 {label} 适应度={:.1}", genome.fitness);
            if genome.fitness > best_threshold {
                best_threshold = genome.fitness;
                best_candidate = Some(genome.clone());
            }
            idx += 1;
        }
        if let Some(g) = best_candidate {
            population.best_ever = g;
            let snap = BestGenomeSnapshot::from_population(population, cli.seed);
            snap.save_atomic(&cli.best_genome)?;
            *last_saved_best = snap.fitness;
            train_log!(
                "最优基因组已更新: fitness={:.2} gen={} → {}",
                snap.fitness,
                snap.generation,
                cli.best_genome.display()
            );
        }
        return Ok(());
    }

    evaluate_population_worker_pool(
        cli,
        population,
        generation,
        generations_total,
        last_saved_best,
        pool.expect("workers>1 需要 WorkerPool"),
    )?;
    let _ = rng;
    Ok(())
}

fn evaluate_population_worker_pool(
    cli: &Cli,
    population: &mut Population,
    generation: u32,
    generations_total: u32,
    last_saved_best: &mut f32,
    pool: &mut WorkerPool,
) -> anyhow::Result<()> {
    let job_dir = pool.job_dir().to_path_buf();
    let genomes: Vec<Genome> = population.genomes().cloned().collect();
    let mut fitnesses = vec![0.0_f32; genomes.len()];
    let mut completed = 0usize;
    let mut best_done = 0.0_f32;
    let mut next_job = 0usize;
    let gen_started = Instant::now();
    let mut last_pool_heartbeat = Instant::now();
    let pop_gen = population.generation;
    let global_best = population.best_ever.fitness;

    while completed < genomes.len() {
        while next_job < genomes.len() {
            let idx = next_job;
            let episode_seed = cli
                .seed
                .wrapping_add(generation as u64 * 100_000)
                .wrapping_add(idx as u64);
            let status_path = job_dir.join(format!("status_{pop_gen}_{idx}.txt"));
            let label = format!("gen{pop_gen}#{idx}");
            if !pool.try_submit(
                idx,
                &genomes[idx],
                episode_seed,
                &label,
                &status_path,
            )? {
                break;
            }
            next_job += 1;
        }

        if last_pool_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
            last_pool_heartbeat = Instant::now();
            log_pool_heartbeat(
                pop_gen,
                generations_total,
                completed,
                genomes.len(),
                &pool.in_flight_status(),
                gen_started,
                best_done,
                global_best.max(best_done),
            );
        }

        if let Some(result) = pool.try_recv_done() {
            let (idx, fitness) = result?;
            fitnesses[idx] = fitness;
            if fitness > best_done {
                best_done = fitness;
            }
            completed += 1;
            train_log!(
                "  基因组 {}/{} (#{idx}) 适应度={:.1}",
                completed,
                genomes.len(),
                fitness
            );
        } else if pool.busy_count() == 0 && next_job >= genomes.len() {
            break;
        } else {
            thread::sleep(Duration::from_millis(50));
        }
    }

    let mut i = 0;
    for genome in population.genomes_mut() {
        genome.fitness = fitnesses[i];
        i += 1;
    }
    population.update_best();
    save_best_if_improved(
        &cli.best_genome,
        population,
        cli.seed,
        last_saved_best,
    )?;
    Ok(())
}

fn load_or_create_population(cli: &Cli) -> anyhow::Result<Population> {
    if cli.checkpoint.exists() && !cli.fresh {
        let text = fs::read_to_string(&cli.checkpoint)?;
        let ck: TrainingCheckpoint = serde_json::from_str(&text)?;
        train_log!("从检查点恢复: 代 {}", ck.population.generation);
        if !ck.innovation.entries.is_empty() || ck.innovation.next_id > 1 {
            restore_innovation_state(&ck.innovation);
        } else {
            restore_innovations_from_population(&ck.population);
        }
        return Ok(ck.population);
    }
    if cli.fresh {
        train_log!("创建新种群: size={} seed={}", cli.population, cli.seed);
    }
    let mut rng = ::rand::rngs::StdRng::seed_from_u64(cli.seed);
    Ok(Population::new(
        PopulationConfig {
            size: cli.population,
            ..Default::default()
        },
        &mut rng,
    ))
}

fn save_checkpoint(cli: &Cli, population: &Population) -> anyhow::Result<()> {
    let ck = TrainingCheckpoint {
        population: population.clone(),
        seed: cli.seed,
        innovation: export_innovation_state(),
    };
    let json = serde_json::to_string_pretty(&ck)?;
    let tmp = cli.checkpoint.with_extension("json.tmp");
    fs::write(&tmp, &json)?;
    fs::rename(&tmp, &cli.checkpoint)?;
    Ok(())
}
