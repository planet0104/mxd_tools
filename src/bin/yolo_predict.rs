//! YOLO ONNX 推理 CLI（CPU / 可选 CUDA）。
//!
//! 用法：
//!   cargo run --release --bin yolo_predict -- \
//!     --model models/yolo_nangang_e1000.onnx \
//!     --source screen_caps/彩虹岛-南港西郊平原 \
//!     --device cpu

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use image::{Rgb, RgbImage};
use mxd_tools::yolo::{Detection, YoloDetector, YoloDevice};

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
}

fn arg_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

fn collect_images(source: &Path) -> Result<Vec<PathBuf>> {
    if source.is_file() {
        return Ok(vec![source.to_path_buf()]);
    }
    if !source.is_dir() {
        bail!("--source 不是文件或目录: {}", source.display());
    }
    let mut files: Vec<PathBuf> = fs::read_dir(source)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_ascii_lowercase())
                    .as_deref(),
                Some("png" | "jpg" | "jpeg" | "bmp" | "webp")
            )
        })
        .collect();
    files.sort();
    Ok(files)
}

fn draw_dets(img: &mut RgbImage, dets: &[Detection]) {
    for d in dets {
        let color = class_color(d.class_id);
        let x1 = d.x1.round().max(0.0) as u32;
        let y1 = d.y1.round().max(0.0) as u32;
        let x2 = d.x2.round().min(img.width().saturating_sub(1) as f32) as u32;
        let y2 = d.y2.round().min(img.height().saturating_sub(1) as f32) as u32;
        draw_rect(img, x1, y1, x2, y2, color);
    }
}

fn class_color(id: usize) -> Rgb<u8> {
    const PALETTE: [[u8; 3]; 12] = [
        [255, 56, 56],
        [255, 157, 151],
        [255, 112, 31],
        [255, 178, 29],
        [207, 210, 49],
        [72, 249, 10],
        [146, 204, 23],
        [61, 219, 134],
        [26, 147, 52],
        [0, 212, 187],
        [44, 153, 168],
        [0, 194, 255],
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

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") || args.is_empty() {
        eprintln!(
            "用法: yolo_predict --model <onnx> --source <图或目录> [--device cpu|cuda|cuda:0] \\\n\
             \t[--conf 0.25] [--iou 0.7] [--out <目录>] [--no-draw]"
        );
        std::process::exit(1);
    }

    let model = PathBuf::from(arg_value(&args, "--model").context("需要 --model")?);
    let source = PathBuf::from(arg_value(&args, "--source").context("需要 --source")?);
    let device = YoloDevice::parse(arg_value(&args, "--device").unwrap_or("cpu"));
    let conf: f32 = arg_value(&args, "--conf")
        .unwrap_or("0.25")
        .parse()
        .context("--conf")?;
    let iou: f32 = arg_value(&args, "--iou")
        .unwrap_or("0.7")
        .parse()
        .context("--iou")?;
    let out_dir = arg_value(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runs/detect/yolo_rust"));
    let no_draw = arg_flag(&args, "--no-draw");

    let mut det = YoloDetector::load_with_thresholds(&model, device, conf, iou, 640)?;
    eprintln!("device={}", det.device_label);

    fs::create_dir_all(&out_dir)?;
    let images = collect_images(&source)?;
    if images.is_empty() {
        bail!("未找到图片: {}", source.display());
    }

    for path in &images {
        let img = image::open(path)
            .with_context(|| format!("读图失败: {}", path.display()))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        let dets = det.detect_rgb8(w, h, img.as_raw())?;
        println!(
            "{} → {} boxes",
            path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            dets.len()
        );
        for d in &dets {
            println!(
                "  {} conf={:.3} xyxy=({:.0},{:.0},{:.0},{:.0})",
                d.label, d.conf, d.x1, d.y1, d.x2, d.y2
            );
        }
        if !no_draw {
            let mut vis = img;
            draw_dets(&mut vis, &dets);
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
            let out_path = out_dir.join(format!("{stem}_pred.jpg"));
            vis.save(&out_path)
                .with_context(|| format!("保存失败: {}", out_path.display()))?;
            println!("  saved {}", out_path.display());
        }
    }
    Ok(())
}
