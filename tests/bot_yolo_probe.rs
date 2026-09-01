//! Headless YOLO bot 探针（`cargo test --release --test bot_yolo_probe`）。
//!
//! 阶段1串行 first_platform+spawn，阶段2并行 episodes（默认 4 workers）。

use std::env;
use std::path::PathBuf;

use macroquad::prelude::*;
use mxd_tools::game::{
    build_parallel_episode_jobs, default_parallel_episode_seeds, default_yolo_model_path,
    run_parallel_probe_pool, run_parallel_probe_subprocess, BotProbeConfig, HeadlessVisionEnv,
    YoloProbeSet, DEFAULT_PARALLEL_JOBS,
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

fn run_parallel_coordinator(args: &[String]) -> ! {
    let exe = env::current_exe().expect("current_exe");
    let model = arg_path(args, "--model");
    let jobs_n = arg_usize(args, "--jobs", DEFAULT_PARALLEL_JOBS);
    let episode_seeds = default_parallel_episode_seeds(jobs_n);

    for (name, probe_args) in [
        (
            "first_platform",
            vec!["--probe".into(), "first_platform".into()],
        ),
        ("spawn", vec!["--probe".into(), "spawn".into()]),
    ] {
        if !run_parallel_probe_subprocess(&exe, model.as_deref(), name, &probe_args) {
            std::process::exit(1);
        }
    }

    let report = run_parallel_probe_pool(
        &exe,
        model.as_deref(),
        build_parallel_episode_jobs(&episode_seeds),
        jobs_n,
    );

    if !report.ok() {
        std::process::exit(1);
    }
    std::process::exit(0);
}

#[macroquad::main(window_conf)]
async fn main() {
    let args: Vec<String> = env::args().collect();

    let single_probe = args
        .iter()
        .position(|a| a == "--probe")
        .and_then(|i| args.get(i + 1))
        .is_some();
    let want_parallel = args.iter().any(|a| a == "--parallel") || !single_probe;

    if want_parallel && !single_probe {
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

    let mut vision = match HeadlessVisionEnv::load(model.as_deref()).await {
        Ok(v) => v,
        Err(e) => {
            eprintln!("初始化 YOLO 探针失败: {e}");
            eprintln!("模型默认路径: {}", default_yolo_model_path().display());
            std::process::exit(2);
        }
    };

    let cfg = BotProbeConfig::default();
    let summary = match mxd_tools::game::run_yolo_probes(&mut vision, &cfg, probe, &[]).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("探针运行失败: {e:#}");
            std::process::exit(1);
        }
    };

    if let Err(msg) = mxd_tools::game::assert_yolo_probes_with(&summary, probe) {
        eprintln!("探针断言失败: {msg}");
        std::process::exit(1);
    }
}
