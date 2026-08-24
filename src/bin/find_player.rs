//! 通过 YOLO 玩家框 + 名牌 OCR 定位指定玩家坐标。
//!
//! 用法：
//!   cargo run --release --bin find_player -- \
//!     --model models/yolo_nangang_e3000.onnx \
//!     --source "screen_caps/彩虹岛-南港西郊平原" \
//!     --name "光头强加强版"

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use chrono::Local;
use image::RgbImage;
use mxd_tools::player_name::{draw_named_player_hit, find_named_player_verbose};
use mxd_tools::yolo::{YoloDetector, YoloDevice};

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
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

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") || args.is_empty() {
        eprintln!(
            "用法: find_player --model <onnx> --source <图或目录> --name <玩家名> \\\n\
             \t[--device cpu|cuda:0] [--conf 0.25] [--player-conf 0.25] [--out <目录>]"
        );
        std::process::exit(1);
    }

    let model = PathBuf::from(arg_value(&args, "--model").context("需要 --model")?);
    let source = PathBuf::from(arg_value(&args, "--source").context("需要 --source")?);
    let target_name = arg_value(&args, "--name").context("需要 --name")?;
    let device = YoloDevice::parse(arg_value(&args, "--device").unwrap_or("cpu"));
    let conf: f32 = arg_value(&args, "--conf").unwrap_or("0.25").parse().context("--conf")?;
    let player_conf: f32 = arg_value(&args, "--player-conf")
        .unwrap_or("0.20")
        .parse()
        .context("--player-conf")?;
    let verbose = args.iter().any(|a| a == "--verbose");
    let out_dir = arg_value(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let stamp = Local::now().format("%Y%m%d_%H%M%S");
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tmp/find_player/{stamp}"))
        });

    let mut det = YoloDetector::load_with_thresholds(&model, device, conf, 0.7, 640)?;
    eprintln!("YOLO device={}", det.device_label);
    eprintln!("查找玩家: {target_name}");

    fs::create_dir_all(&out_dir)?;
    let images = collect_images(&source)?;
    if images.is_empty() {
        bail!("未找到图片: {}", source.display());
    }

    let mut ok = 0usize;
    for path in &images {
        let img = image::open(path)
            .with_context(|| format!("读图失败: {}", path.display()))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        let dets = det.detect_rgb8(w, h, img.as_raw())?;
        let player_count = dets.iter().filter(|d| d.label == "玩家").count();

        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        match find_named_player_verbose(&img, &dets, target_name, player_conf, verbose)? {
            (Some(hit), _) => {
                ok += 1;
                println!(
                    "{fname}: FOUND ({:.0},{:.0}) match={:.2}{} ocr=\"{}\" players={player_count} yolo_conf={:.2}",
                    hit.x,
                    hit.y,
                    hit.match_score,
                    if hit.partial { " partial" } else { "" },
                    hit.ocr_text,
                    hit.player_conf
                );

                let mut vis: RgbImage = img.clone();
                draw_named_player_hit(&mut vis, &hit);
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
                let out_path = out_dir.join(format!("{stem}_found.jpg"));
                vis.save(&out_path)
                    .with_context(|| format!("保存失败: {}", out_path.display()))?;
            }
            (None, attempts) => {
                println!("{fname}: NOT FOUND (players={player_count})");
                if verbose {
                    for (i, a) in attempts.iter().enumerate() {
                        println!(
                            "  [{i}] conf={:.2} roi={:?} ocr=\"{}\" match={:.2}",
                            a.player_conf, a.roi, a.ocr_text, a.match_score
                        );
                    }
                }
            }
        }
    }

    println!("\n===== 总结 =====");
    println!("成功: {ok}/{}", images.len());
    println!("输出: {}", out_dir.display());
    Ok(())
}
