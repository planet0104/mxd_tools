//! Headless YOLO bot 探针 CLI（与 integration test 共用逻辑）。
//!
//! ```powershell
//! cargo run --release --bin bot_yolo_probe -- --parallel
//! cargo run --release --bin bot_yolo_probe -- --parallel --jobs 4
//! cargo run --release --bin bot_yolo_probe -- --probe first_platform
//! ```

use std::env;
use std::path::PathBuf;

use macroquad::prelude::*;
use mxd_tools::game::{
    assert_yolo_probes_with, build_parallel_episode_jobs, default_parallel_episode_seeds,
    default_probe_seeds, run_parallel_probe_pool, run_parallel_probe_subprocess, run_yolo_probes,
    BotProbeConfig, HeadlessVisionEnv, YoloProbeSet, DEFAULT_PARALLEL_JOBS,
};
use mxd_tools::headless_gl;

fn window_conf() -> Conf {
    headless_gl::headless_window_conf("bot_yolo_probe")
}

fn arg_path(args: &[String], key: &str) -> Option<PathBuf> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
}

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_seeds(s: &str) -> Vec<u64> {
    s.split(',').filter_map(|p| p.trim().parse().ok()).collect()
}

fn parse_seeds_arg(args: &[String]) -> Vec<u64> {
    arg_path(args, "--seeds")
        .and_then(|p| p.to_str().map(parse_seeds))
        .filter(|v| !v.is_empty())
        .unwrap_or_else(default_probe_seeds)
}

fn parse_parallel_seeds_arg(args: &[String], jobs: usize) -> Vec<u64> {
    arg_path(args, "--parallel-seeds")
        .and_then(|p| p.to_str().map(parse_seeds))
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default_parallel_episode_seeds(jobs))
}

fn run_parallel_coordinator(args: &[String]) -> ! {
    let exe = env::current_exe().expect("current_exe");
    let model = arg_path(args, "--model");
    let jobs_n = arg_usize(args, "--jobs", DEFAULT_PARALLEL_JOBS);
    let episode_seeds = parse_parallel_seeds_arg(args, jobs_n);

    eprintln!("阶段1/2: first_platform + spawn（串行，避免 YOLO 冷启动争抢）");
    for (name, args_vec) in [
        (
            "first_platform",
            vec!["--probe".into(), "first_platform".into()],
        ),
        ("spawn", vec!["--probe".into(), "spawn".into()]),
    ] {
        if !run_parallel_probe_subprocess(&exe, model.as_deref(), name, &args_vec) {
            std::process::exit(1);
        }
    }

    let episode_jobs = build_parallel_episode_jobs(&episode_seeds);
    eprintln!(
        "阶段2/2: episodes×{} 并行 workers={jobs_n}",
        episode_seeds.len()
    );

    let report = run_parallel_probe_pool(&exe, model.as_deref(), episode_jobs, jobs_n);

    if !report.ok() {
        eprintln!(
            "episodes 并行失败: {}/{} 通过，失败: {:?}",
            report.passed, report.total, report.failed
        );
        std::process::exit(1);
    }
    eprintln!(
        "全部通过（阶段1×2 + 阶段2 {}/{}）",
        report.passed, report.total
    );
    std::process::exit(0);
}

#[macroquad::main(window_conf)]
async fn main() {
    let args: Vec<String> = env::args().collect();

    if args.iter().any(|a| a == "--parallel") {
        run_parallel_coordinator(&args);
    }

    let _ = headless_gl::hide_gl_window();
    let model = arg_path(&args, "--model");
    let probe = args
        .iter()
        .position(|a| a == "--probe")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| YoloProbeSet::parse(s))
        .unwrap_or(YoloProbeSet::All);
    let episode_seeds = parse_seeds_arg(&args);

    let mut vision = match HeadlessVisionEnv::load(model.as_deref()).await {
        Ok(v) => v,
        Err(e) => {
            eprintln!("初始化 YOLO 探针失败: {e}");
            eprintln!("未传 --model 时使用嵌入默认 ONNX");
            std::process::exit(2);
        }
    };

    let cfg = BotProbeConfig::default();
    eprintln!("YOLO bot 探针: probe={probe:?} seeds={episode_seeds:?} (YOLO 后台线程 + GL 主线程)");

    let summary = match run_yolo_probes(&mut vision, &cfg, probe, &episode_seeds).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("探针运行失败: {e:#}");
            std::process::exit(1);
        }
    };

    if probe == YoloProbeSet::FirstPlatform {
        let fp = &summary.first_platform;
        eprintln!(
            "first_platform 结果: kills={} end=({:.0},{:.0}) min_x_after={:.0} max_x_after={:.0} y_changed={} left={}",
            fp.kills,
            fp.end_x,
            fp.end_y,
            fp.min_x_after_kills,
            fp.max_x_after_kills,
            fp.y_band_changed,
            fp.left_first_platform(),
        );
    }

    if let Err(msg) = assert_yolo_probes_with(&summary, probe) {
        eprintln!("探针断言失败: {msg}");
        std::process::exit(1);
    }

    eprintln!("探针通过: {probe:?}");
}
