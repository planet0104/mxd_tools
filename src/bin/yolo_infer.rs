//! YOLO ONNX 推理 CLI（带中文标签绘制）。
//!
//! 用法：
//!   cargo run --release --bin yolo_infer -- \
//!     --model models/yolo_nangang_e2000_best.onnx \
//!     --source "screen_caps/彩虹岛-南港西郊平原" \
//!     --out tmp/yolo_output/20260824_140530 \
//!     --device cpu

use std::collections::HashSet;
use std::env;
use std::fs::{self, FileTimes, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use anyhow::{bail, Context, Result};
use chrono::Local;
use image::{Rgb, RgbImage};
use mxd_tools::yolo::{Detection, YoloDetector, YoloDevice};
use rusttype::{point, Font, Scale};

// ============================================================
// 绘图
// ============================================================

/// 尝试加载字体，优先使用项目自带的字体
fn load_font() -> Font<'static> {
    // 先尝试系统字体（微软雅黑，保证中文支持）
    let font_paths = [
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\msyh.ttf",
        "C:\\Windows\\Fonts\\simhei.ttf",
    ];
    for path in &font_paths {
        if let Ok(data) = fs::read(path) {
            if let Some(font) = Font::try_from_vec(data) {
                eprintln!("字体: {}", path);
                return font;
            }
        }
    }
    // 回退到项目字体
    let project_font = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/fonts/VonwaonBitmap-16px.ttf");
    if let Ok(data) = fs::read(&project_font) {
        if let Some(font) = Font::try_from_vec(data) {
            eprintln!("字体: {}", project_font.display());
            return font;
        }
    }
    panic!("找不到任何可用字体");
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

fn draw_rect(img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>, thickness: u32) {
    if x2 <= x1 || y2 <= y1 {
        return;
    }
    let t = thickness;
    for d in 0..t {
        let x1s = x1.saturating_add(d).min(x2);
        let y1s = y1.saturating_add(d).min(y2);
        let x2s = x2.saturating_sub(d).max(x1s);
        let y2s = y2.saturating_sub(d).max(y1s);
        for x in x1s..=x2s {
            img.put_pixel(x, y1s, color);
            img.put_pixel(x, y2s, color);
        }
        for y in y1s..=y2s {
            img.put_pixel(x1s, y, color);
            img.put_pixel(x2s, y, color);
        }
    }
}

/// 在图片上绘制带半透明背景的文字标签
fn draw_label(
    img: &mut RgbImage,
    text: &str,
    x: i32,
    y: i32,
    text_color: Rgb<u8>,
    bg_color: Rgb<u8>,
    font: &Font,
    scale: f32,
) {
    let scale = Scale::uniform(scale);
    let v_metrics = font.v_metrics(scale);
    let offset = point(0.0, v_metrics.ascent);

    let glyphs: Vec<_> = font.layout(text, scale, offset).collect();
    if glyphs.is_empty() {
        return;
    }

    // 测量文字尺寸
    let text_w = (glyphs.last().unwrap().position().x
        + glyphs.last().unwrap().unpositioned().h_metrics().advance_width)
        .ceil() as i32;
    let text_h = (v_metrics.ascent - v_metrics.descent).ceil() as i32;
    let pad = 1i32;

    let bg_x1 = x.max(0) as u32;
    let bg_y1 = (y - text_h - pad).max(0) as u32;
    let bg_x2 = (x + text_w + pad * 2).max(0).min(img.width() as i32 - 1) as u32;
    let bg_y2 = (y - pad).max(0).min(img.height() as i32 - 1) as u32;

    if bg_x1 > bg_x2 || bg_y1 > bg_y2 {
        return;
    }

    // 半透明背景
    for py in bg_y1..=bg_y2 {
        for px in bg_x1..=bg_x2 {
            let orig = img.get_pixel(px, py);
            let r = (bg_color[0] as f32 * 0.6 + orig[0] as f32 * 0.4) as u8;
            let g = (bg_color[1] as f32 * 0.6 + orig[1] as f32 * 0.4) as u8;
            let b = (bg_color[2] as f32 * 0.6 + orig[2] as f32 * 0.4) as u8;
            img.put_pixel(px, py, Rgb([r, g, b]));
        }
    }

    // 绘制文字
    // rusttype 的 pixel_bounding_box 返回绝对屏幕坐标（含 position，y 向下）
    let base_x = bg_x1 as i32 + pad;
    let base_y = bg_y1 as i32 + pad;

    for g in &glyphs {
        if let Some(bb) = g.pixel_bounding_box() {
            g.draw(|gx, gy, v| {
                if v <= 0.0 {
                    return;
                }
                let px = (base_x + bb.min.x + gx as i32) as u32;
                let py = (base_y + bb.min.y + gy as i32) as u32;
                if px < img.width() && py < img.height() {
                    // rusttype 的 coverage v 已经是 0.0~1.0 的归一化值
                    let alpha = v;
                    let orig = img.get_pixel(px, py);
                    let r = (text_color[0] as f32 * alpha + orig[0] as f32 * (1.0 - alpha)) as u8;
                    let g = (text_color[1] as f32 * alpha + orig[1] as f32 * (1.0 - alpha)) as u8;
                    let b = (text_color[2] as f32 * alpha + orig[2] as f32 * (1.0 - alpha)) as u8;
                    img.put_pixel(px, py, Rgb([r, g, b]));
                }
            });
        }
    }
}

fn draw_dets(img: &mut RgbImage, dets: &[Detection], font: &Font, scale: f32) {
    let v_metrics = font.v_metrics(Scale::uniform(scale));
    let text_h = (v_metrics.ascent - v_metrics.descent).ceil() as i32;
    let pad = 1i32;

    for d in dets {
        let color = class_color(d.class_id);
        let x1 = d.x1.round().max(0.0) as u32;
        let y1 = d.y1.round().max(0.0) as u32;
        let x2 = d.x2.round().min(img.width().saturating_sub(1) as f32) as u32;
        let y2 = d.y2.round().min(img.height().saturating_sub(1) as f32) as u32;
        draw_rect(img, x1, y1, x2, y2, color, 2);

        let label = format!("{} {:.0}%", d.label, d.conf * 100.0);
        let label_x = x1 as i32;
        // 标签在框上方，如果上方空间不够就画在框内部
        let label_y_above = y1 as i32 - 4;
        let label_y = if label_y_above - text_h - pad >= 0 {
            label_y_above
        } else {
            // 画在框内部左上角
            y1 as i32 + 2
        };
        draw_label(img, &label, label_x, label_y, Rgb([255, 255, 255]), color, font, scale);
    }
}

// ============================================================
// CLI 工具
// ============================================================

/// 默认输出到带当前时间戳的子目录，避免覆盖旧结果且文件日期不会「卡」在首次创建时间。
fn default_out_dir() -> PathBuf {
    let stamp = Local::now().format("%Y%m%d_%H%M%S");
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tmp/yolo_output/{stamp}"))
}

fn save_output_image(path: &Path, img: &RgbImage) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if path.is_file() {
        fs::remove_file(path)?;
    }
    img.save(path)
        .with_context(|| format!("保存失败: {}", path.display()))?;
    touch_file_times(path)
}

fn touch_file_times(path: &Path) -> Result<()> {
    let now = SystemTime::now();
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .with_context(|| format!("打开已保存文件失败: {}", path.display()))?;
    #[cfg(windows)]
    use std::os::windows::fs::FileTimesExt;
    let mut times = FileTimes::new().set_accessed(now).set_modified(now);
    #[cfg(windows)]
    {
        times = times.set_created(now);
    }
    file.set_times(times)
        .with_context(|| format!("更新文件时间失败: {}", path.display()))
}

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
            "用法: yolo_infer --model <onnx> --source <图或目录> [--device cpu|cuda|cuda:0] \\\n\
             \t[--conf 0.25] [--iou 0.7] [--out <目录>] [--bench]\n\
             \t默认 --out 为 tmp/yolo_output/<当前时间戳>/，避免覆盖旧结果"
        );
        std::process::exit(1);
    }

    let bench = args.iter().any(|a| a == "--bench");

    let model = PathBuf::from(arg_value(&args, "--model").context("需要 --model")?);
    let source = PathBuf::from(arg_value(&args, "--source").context("需要 --source")?);
    let device = YoloDevice::parse(arg_value(&args, "--device").unwrap_or("cpu"));
    let conf: f32 = arg_value(&args, "--conf").unwrap_or("0.25").parse().context("--conf")?;
    let iou: f32 = arg_value(&args, "--iou").unwrap_or("0.7").parse().context("--iou")?;
    let out_dir = arg_value(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(default_out_dir);

    let mut det = YoloDetector::load_with_thresholds(&model, device, conf, iou, 640)?;
    eprintln!("device={}", det.device_label);

    // 加载字体
    let font = load_font();
    let font_scale = 14.0; // 标签文字大小

    if !bench {
        fs::create_dir_all(&out_dir)?;
        eprintln!("输出目录: {}", out_dir.display());
    }
    let images = collect_images(&source)?;
    if images.is_empty() {
        bail!("未找到图片: {}", source.display());
    }

    let mut total_dets = 0usize;
    let mut seen_classes: HashSet<String> = HashSet::new();
    let mut bench_ms: Vec<f64> = Vec::new();

    for path in &images {
        let img = image::open(path)
            .with_context(|| format!("读图失败: {}", path.display()))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        let t0 = Instant::now();
        let dets = det.detect_rgb8(w, h, img.as_raw())?;
        let infer_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_dets += dets.len();

        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        if bench {
            bench_ms.push(infer_ms);
            println!("BENCH {fname} {infer_ms:.2} ms boxes={}", dets.len());
            continue;
        }

        println!("{} → {} boxes", fname, dets.len());
        for d in &dets {
            seen_classes.insert(d.label.to_string());
            println!(
                "  {} conf={:.3} xyxy=({:.0},{:.0},{:.0},{:.0})",
                d.label, d.conf, d.x1, d.y1, d.x2, d.y2
            );
        }

        let mut vis = img;
        draw_dets(&mut vis, &dets, &font, font_scale);
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let out_path = out_dir.join(format!("{stem}_pred.jpg"));
        save_output_image(&out_path, &vis)?;
        println!("  saved {}", out_path.display());
    }

    if bench {
        let n = bench_ms.len();
        let sum: f64 = bench_ms.iter().sum();
        let avg = if n > 0 { sum / n as f64 } else { 0.0 };
        let min = bench_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let max = bench_ms.iter().copied().fold(0.0, f64::max);
        println!("\n===== BENCH =====");
        println!("图片数: {n}");
        println!("infer_ms_avg: {avg:.2}");
        println!("infer_ms_min: {min:.2}");
        println!("infer_ms_max: {max:.2}");
        return Ok(());
    }

    println!("\n===== 总结 =====");
    println!("图片数: {}", images.len());
    println!("总检测数: {}", total_dets);
    println!("出现类别: {}", seen_classes.into_iter().collect::<Vec<_>>().join(", "));
    println!("输出目录: {}", out_dir.display());

    Ok(())
}