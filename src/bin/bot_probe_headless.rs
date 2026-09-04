//! Headless 规则 bot 多局探针（离屏 GL + 真实 YOLO/SelfTracker，与 game_preview 一致）。
//!
//! ```powershell
//! cargo run --release --bin bot_probe_headless
//! cargo run --release --bin bot_probe_headless -- --seeds 0,2 --ticks 7200
//! cargo test --release --test bot_yolo_probe
//! ```

use std::env;
use std::path::PathBuf;

use macroquad::prelude::*;
use mxd_tools::game::bot_harness::{
    default_probe_seeds, probe_duration_secs, run_probe_seeds, BotProbeConfig,
};
use mxd_tools::game::{HeadlessVisionEnv, VisionPaceConfig};
use mxd_tools::headless_gl;

fn window_conf() -> Conf {
    headless_gl::headless_window_conf("bot_probe_headless")
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn arg_path(args: &[String], key: &str) -> Option<PathBuf> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
}

fn parse_seeds(s: &str) -> Vec<u64> {
    s.split(',').filter_map(|p| p.trim().parse().ok()).collect()
}

#[macroquad::main(window_conf)]
async fn main() {
    let _ = headless_gl::hide_gl_window();
    let args: Vec<String> = env::args().collect();
    let seeds = arg_value(&args, "--seeds")
        .map(|s| parse_seeds(&s))
        .filter(|v| !v.is_empty())
        .unwrap_or_else(default_probe_seeds);

    let max_ticks = arg_value(&args, "--ticks")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_800u32);

    let detect_hz = arg_value(&args, "--detect-hz")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0_f32);

    let cfg = BotProbeConfig {
        max_ticks,
        vision_interval: VisionPaceConfig::from_detect_hz(detect_hz).vision_interval_ticks,
    };

    let model = arg_path(&args, "--model");
    let mut vision = match HeadlessVisionEnv::load(model.as_deref()).await {
        Ok(v) => v,
        Err(e) => {
            eprintln!("初始化 YOLO 探针失败: {e}");
            eprintln!("未传 --model 时使用嵌入默认 ONNX");
            std::process::exit(2);
        }
    };

    eprintln!(
        "bot_probe_headless (YOLO) seeds={:?} ticks={} (~{:.1}s/ep) detect_hz={}",
        seeds,
        cfg.max_ticks,
        probe_duration_secs(&cfg),
        detect_hz
    );

    match run_probe_seeds(&mut vision, &seeds, &cfg).await {
        Ok(reports) => {
            for r in &reports {
                let status = if r.passed() { "PASS" } else { "FAIL" };
                println!(
                    "{status} seed={} kills={} waves={} visited={} y_bands={} x_range={:.0} attacks={}",
                    r.seed,
                    r.kills,
                    r.waves_cleared,
                    r.visited_cells,
                    r.altitude_bands,
                    r.x_range,
                    r.attack_decisions,
                );
                for f in &r.failures {
                    eprintln!("  - {f}");
                }
            }
            let failed = reports.iter().filter(|r| !r.passed()).count();
            if failed > 2 {
                std::process::exit(1);
            }
            println!("ALL {} EPISODES PASSED (YOLO)", seeds.len());
        }
        Err(summary) => {
            eprintln!("{summary}");
            std::process::exit(1);
        }
    }
}
