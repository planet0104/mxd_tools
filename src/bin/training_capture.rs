//! 从训练 eval 路径截取一帧，保存原图 + YOLO/OCR 标注图供人工验证。
//!
//! ```powershell
//! cargo run --release --bin training_capture -- --seed 42 --capture-tick 800 --pace 4
//! ```

use std::env;
use std::fs;
use std::path::PathBuf;

use chrono::Local;
use image::{Rgb, RgbImage};
use macroquad::prelude::*;
use mxd_tools::game::view;
use mxd_tools::game::{FitnessShapingConfig, NEAT_CONF_THRESH, TrainingPaceConfig, VisionPipeline};
use mxd_tools::headless_gl;
use mxd_tools::neat::{BestGenomeSnapshot, Genome};
use mxd_tools::player_name::draw_named_player_hit;
use mxd_tools::trainer::{capture_training_frame, TrainerEvalContext};
use mxd_tools::yolo::{Detection, YoloDevice};
use ::rand::SeedableRng;

fn window_conf() -> Conf {
    headless_gl::headless_window_conf("training_capture")
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

fn arg_path(args: &[String], key: &str) -> Option<PathBuf> {
    arg_value(args, key).map(PathBuf::from)
}

fn draw_yolo_boxes(img: &mut RgbImage, dets: &[Detection], min_conf: f32) {
    for d in dets {
        if d.conf < min_conf {
            continue;
        }
        let color = class_color(d.class_id);
        let x1 = d.x1.round().max(0.0) as u32;
        let y1 = d.y1.round().max(0.0) as u32;
        let x2 = d.x2.round().min(img.width().saturating_sub(1) as f32) as u32;
        let y2 = d.y2.round().min(img.height().saturating_sub(1) as f32) as u32;
        draw_rect(img, x1, y1, x2, y2, color);
    }
}

fn class_color(id: usize) -> Rgb<u8> {
    const PALETTE: [[u8; 3]; 6] = [
        [255, 56, 56],
        [72, 249, 10],
        [0, 194, 255],
        [255, 178, 29],
        [207, 210, 49],
        [255, 112, 31],
    ];
    let c = PALETTE[id % PALETTE.len()];
    Rgb(c)
}

fn draw_rect(img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>) {
    if x2 <= x1 || y2 <= y1 {
        return;
    }
    for x in x1..=x2 {
        img.put_pixel(x, y1, color);
        img.put_pixel(x, y2, color);
    }
    for y in y1..=y2 {
        img.put_pixel(x1, y, color);
        img.put_pixel(x2, y, color);
    }
}

fn load_genome(args: &[String], seed: u64) -> anyhow::Result<Genome> {
    if let Some(path) = arg_path(args, "--genome-file") {
        let text = fs::read_to_string(&path)?;
        if let Ok(snap) = serde_json::from_str::<BestGenomeSnapshot>(&text) {
            return Ok(snap.genome);
        }
        return Ok(serde_json::from_str(&text)?);
    }
    let mut rng = ::rand::rngs::StdRng::seed_from_u64(seed);
    Ok(Genome::random_minimal(&mut rng))
}

#[macroquad::main(window_conf)]
async fn main() {
    let _ = headless_gl::hide_gl_window();
    let args: Vec<String> = env::args().collect();

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let default_model = manifest.join("models/yolo_nangang_e3000_best.onnx");
    let default_genome = manifest.join("tmp/neat_best_genome.json");

    let seed = arg_u64(&args, "--seed", 42);
    let capture_tick = arg_usize(&args, "--capture-tick", 800);
    let max_ticks = arg_usize(&args, "--max-ticks", capture_tick + 400);
    let pace = TrainingPaceConfig {
        vision_interval_ticks: arg_u32(&args, "--pace", 12).max(1),
    };
    let model = arg_path(&args, "--model").unwrap_or(default_model);
    let out_dir = arg_path(&args, "--out").unwrap_or_else(|| {
        let stamp = Local::now().format("%Y%m%d_%H%M%S");
        manifest.join(format!("tmp/training_capture/{stamp}"))
    });

    let genome = match load_genome(&args, seed) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("加载基因组失败: {e}");
            if default_genome.is_file() {
                eprintln!("尝试默认: {}", default_genome.display());
                let text = fs::read_to_string(&default_genome).unwrap_or_default();
                serde_json::from_str::<BestGenomeSnapshot>(&text)
                    .map(|s| s.genome)
                    .unwrap_or_else(|_| {
                        eprintln!("无法解析默认基因组，使用 random_minimal");
                        let mut rng = ::rand::rngs::StdRng::seed_from_u64(seed);
                        Genome::random_minimal(&mut rng)
                    })
            } else {
                let mut rng = ::rand::rngs::StdRng::seed_from_u64(seed);
                Genome::random_minimal(&mut rng)
            }
        }
    };

    eprintln!("训练截图验证（eval 同路径）");
    eprintln!("  model={}", model.display());
    eprintln!("  seed={seed} capture_tick={capture_tick} pace={} max_ticks={max_ticks}", pace.vision_interval_ticks);
    eprintln!("  输出目录: {}", out_dir.display());

    let rt = view::new_render_target();
    let pipeline = match VisionPipeline::load(&model, YoloDevice::Cpu, NEAT_CONF_THRESH) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("加载 VisionPipeline 失败: {e}");
            return;
        }
    };
    let mut ctx = match TrainerEvalContext::prepare(pipeline, rt).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("初始化训练上下文失败: {e}");
            return;
        }
    };

    let cap = match capture_training_frame(
        &mut ctx,
        &genome,
        seed,
        capture_tick,
        max_ticks,
        pace,
        FitnessShapingConfig::default(),
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("截取失败: {e}");
            return;
        }
    };

    fs::create_dir_all(&out_dir).ok();
    let raw_path = out_dir.join("frame_raw.png");
    let vis_path = out_dir.join("frame_yolo_ocr.jpg");
    if let Err(e) = cap.rgb.save(&raw_path) {
        eprintln!("保存原图失败: {e}");
        return;
    }

    let mut vis = cap.rgb.clone();
    draw_yolo_boxes(&mut vis, &cap.step.detections, NEAT_CONF_THRESH);
    if let Some(ref hit) = cap.step.self_player {
        draw_named_player_hit(&mut vis, hit);
    }
    if let Err(e) = vis.save(&vis_path) {
        eprintln!("保存标注图失败: {e}");
        return;
    }

    let player_count = cap
        .step
        .detections
        .iter()
        .filter(|d| d.label == "玩家" && d.conf >= NEAT_CONF_THRESH)
        .count();
    let mob_count = cap
        .step
        .detections
        .iter()
        .filter(|d| d.label == "怪物" && d.conf >= NEAT_CONF_THRESH)
        .count();
    let drop_count = cap
        .step
        .detections
        .iter()
        .filter(|d| (d.label == "金币" || d.label == "药水") && d.conf >= NEAT_CONF_THRESH)
        .count();

    println!("tick={} fitness={:.2}", cap.tick, cap.fitness);
    println!("YOLO(conf>={NEAT_CONF_THRESH}): 玩家={player_count} 怪物={mob_count} 掉落={drop_count} 总={}", cap.step.detections.len());
    match &cap.step.self_player {
        Some(hit) => println!(
            "OCR 自身: FOUND ({:.0},{:.0}) match={:.2} ocr=\"{}\"",
            hit.x, hit.y, hit.match_score, hit.ocr_text
        ),
        None => println!("OCR 自身: NOT FOUND"),
    }
    println!("原图: {}", raw_path.display());
    println!("标注: {}", vis_path.display());
}
