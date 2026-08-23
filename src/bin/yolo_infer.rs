//! YOLO ONNX 推理 CLI（独立版，不依赖 mxd_tools/opencv）。
//!
//! 用法：
//!   cargo run --release --bin yolo_infer -- \
//!     --model yolo_nangang_e1000.onnx \
//!     --source "screen_caps/彩虹岛-南港西郊平原" \
//!     --out temp/yolo_output \
//!     --device cpu

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use image::{Rgb, RgbImage};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;
use rusttype::{point, Font, Scale};

// ============================================================
// 21 类标签（与 dataset/.../generated/yolo/data.yaml 对齐）
// ============================================================
const CLASS_NAMES: [&str; 21] = [
    "地板", "梯子", "绳子", "入口", "出口", "花蘑菇", "蓝蜗牛", "绿蜗牛",
    "红蜗牛", "树怪", "玩家", "金币", "药水", "武器", "装备", "材料",
    "小地图", "任务窗", "浮动按钮", "面板", "键盘",
];

fn class_name(id: usize) -> &'static str {
    CLASS_NAMES.get(id).copied().unwrap_or("未知")
}

// ============================================================
// 数据结构
// ============================================================
#[derive(Debug, Clone)]
struct Detection {
    class_id: usize,
    label: &'static str,
    conf: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[derive(Debug, Clone, Copy)]
struct LetterboxMeta {
    gain: f32,
    pad_x: f32,
    pad_y: f32,
    orig_w: u32,
    orig_h: u32,
}

#[derive(Debug, Clone, Copy)]
enum YoloDevice {
    Cpu,
    Cuda(u32),
}

impl YoloDevice {
    fn parse(s: &str) -> Self {
        let t = s.trim().to_ascii_lowercase();
        if t == "cpu" {
            return Self::Cpu;
        }
        if let Some(rest) = t.strip_prefix("cuda") {
            let id = rest.trim_start_matches(':').trim().parse::<u32>().unwrap_or(0);
            return Self::Cuda(id);
        }
        if t == "0" || t == "gpu" {
            return Self::Cuda(0);
        }
        Self::Cpu
    }
}

// ============================================================
// YOLO 检测器
// ============================================================
struct YoloDetector {
    session: Session,
    imgsz: u32,
    conf: f32,
    iou: f32,
    device_label: String,
}

impl YoloDetector {
    fn load(onnx: &Path, device: YoloDevice) -> Result<Self> {
        Self::load_with_thresholds(onnx, device, 0.25, 0.7, 640)
    }

    fn load_with_thresholds(
        onnx: &Path,
        device: YoloDevice,
        conf: f32,
        iou: f32,
        imgsz: u32,
    ) -> Result<Self> {
        if !onnx.is_file() {
            bail!("找不到 ONNX: {}", onnx.display());
        }

        let mut device_label = "cpu".to_string();
        let session = match device {
            YoloDevice::Cuda(id) => {
                #[cfg(feature = "cuda")]
                {
                    use ort::ep::CUDA;
                    let try_cuda = Session::builder()
                        .context("创建 ORT SessionBuilder 失败")?
                        .with_execution_providers([CUDA::default().with_device_id(id as i32).build()]);
                    match try_cuda {
                        Ok(b) => match b.commit_from_file(onnx) {
                            Ok(s) => {
                                device_label = format!("cuda:{id}");
                                eprintln!("YOLO: 使用 CUDA EP (device={id})");
                                s
                            }
                            Err(e) => {
                                eprintln!("YOLO: CUDA session 失败，回退 CPU: {e}");
                                device_label = "cpu(fallback)".to_string();
                                Session::builder()?.commit_from_file(onnx)?
                            }
                        },
                        Err(e) => {
                            eprintln!("YOLO: 注册 CUDA EP 失败，回退 CPU: {e}");
                            device_label = "cpu(fallback)".to_string();
                            Session::builder()?.commit_from_file(onnx)?
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = id;
                    eprintln!("YOLO: 未启用 cuda feature，使用 CPU");
                    device_label = "cpu(no-cuda-feature)".to_string();
                    Session::builder()?.commit_from_file(onnx)?
                }
            }
            YoloDevice::Cpu => Session::builder()?.commit_from_file(onnx)?,
        };

        Ok(Self {
            session,
            imgsz,
            conf,
            iou,
            device_label,
        })
    }

    fn detect_rgb8(&mut self, w: u32, h: u32, rgb: &[u8]) -> Result<Vec<Detection>> {
        if rgb.len() != (w as usize) * (h as usize) * 3 {
            bail!("RGB 缓冲长度不符: got {} expect {}", rgb.len(), w as usize * h as usize * 3);
        }
        let (input, meta) = letterbox_rgb(rgb, w, h, self.imgsz);
        let shape = input.shape().to_vec();
        let flat = input.into_raw_vec_and_offset().0;

        let outputs = {
            let input_tensor =
                TensorRef::from_array_view((shape, flat.as_slice())).context("构造输入张量失败")?;
            self.session
                .run(ort::inputs![input_tensor])
                .context("ORT 推理失败")?
        };

        let (_name, value) = outputs.iter().next().context("ORT 无输出")?;
        let (out_shape, out_data) = value
            .try_extract_tensor::<f32>()
            .context("解析输出张量失败")?;
        let arr = ndarray::ArrayD::from_shape_vec(
            out_shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
            out_data.to_vec(),
        )
        .context("输出 shape 与数据不匹配")?;

        Ok(decode_yolo_output(&arr, &meta, self.conf, self.iou))
    }
}

// ============================================================
// Letterbox 预处理
// ============================================================
fn letterbox_rgb(rgb: &[u8], w: u32, h: u32, imgsz: u32) -> (Array4<f32>, LetterboxMeta) {
    let wf = w as f32;
    let hf = h as f32;
    let size = imgsz as f32;
    let gain = (size / hf).min(size / wf);
    let nw = (wf * gain).round().max(1.0) as u32;
    let nh = (hf * gain).round().max(1.0) as u32;
    let dw = (size - nw as f32) * 0.5;
    let dh = (size - nh as f32) * 0.5;
    let pad_x = (dw - 0.1).round();
    let pad_y = (dh - 0.1).round();
    let left = pad_x.max(0.0) as u32;
    let top = pad_y.max(0.0) as u32;

    let resized = resize_rgb_bilinear(rgb, w, h, nw, nh);
    let mut canvas = vec![114u8; (imgsz as usize) * (imgsz as usize) * 3];
    for y in 0..nh {
        for x in 0..nw {
            let dx = left + x;
            let dy = top + y;
            if dx >= imgsz || dy >= imgsz {
                continue;
            }
            let si = ((y * nw + x) * 3) as usize;
            let di = ((dy * imgsz + dx) * 3) as usize;
            canvas[di] = resized[si];
            canvas[di + 1] = resized[si + 1];
            canvas[di + 2] = resized[si + 2];
        }
    }

    let mut arr = Array4::<f32>::zeros((1, 3, imgsz as usize, imgsz as usize));
    for y in 0..imgsz as usize {
        for x in 0..imgsz as usize {
            let i = (y * imgsz as usize + x) * 3;
            arr[[0, 0, y, x]] = canvas[i] as f32 / 255.0;
            arr[[0, 1, y, x]] = canvas[i + 1] as f32 / 255.0;
            arr[[0, 2, y, x]] = canvas[i + 2] as f32 / 255.0;
        }
    }

    let meta = LetterboxMeta {
        gain,
        pad_x,
        pad_y,
        orig_w: w,
        orig_h: h,
    };
    (arr, meta)
}

fn resize_rgb_bilinear(rgb: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    if w == nw && h == nh {
        return rgb.to_vec();
    }
    let mut out = vec![0u8; (nw as usize) * (nh as usize) * 3];
    let x_scale = w as f32 / nw as f32;
    let y_scale = h as f32 / nh as f32;
    for y in 0..nh {
        let fy = (y as f32 + 0.5) * y_scale - 0.5;
        let y0 = fy.floor().max(0.0) as i32;
        let y1 = (y0 + 1).min(h as i32 - 1);
        let wy = fy - y0 as f32;
        let y0 = y0.clamp(0, h as i32 - 1) as u32;
        let y1 = y1 as u32;
        for x in 0..nw {
            let fx = (x as f32 + 0.5) * x_scale - 0.5;
            let x0 = fx.floor().max(0.0) as i32;
            let x1 = (x0 + 1).min(w as i32 - 1);
            let wx = fx - x0 as f32;
            let x0 = x0.clamp(0, w as i32 - 1) as u32;
            let x1 = x1 as u32;
            let i00 = ((y0 * w + x0) * 3) as usize;
            let i01 = ((y0 * w + x1) * 3) as usize;
            let i10 = ((y1 * w + x0) * 3) as usize;
            let i11 = ((y1 * w + x1) * 3) as usize;
            let o = ((y * nw + x) * 3) as usize;
            for c in 0..3 {
                let v = (1.0 - wy) * ((1.0 - wx) * rgb[i00 + c] as f32 + wx * rgb[i01 + c] as f32)
                    + wy * ((1.0 - wx) * rgb[i10 + c] as f32 + wx * rgb[i11 + c] as f32);
                out[o + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

// ============================================================
// 后处理：解码 + NMS
// ============================================================
fn decode_yolo_output(
    output: &ndarray::ArrayD<f32>,
    meta: &LetterboxMeta,
    conf_thres: f32,
    iou_thres: f32,
) -> Vec<Detection> {
    let view = match output.ndim() {
        3 => output.index_axis(ndarray::Axis(0), 0),
        2 => output.view(),
        _ => return Vec::new(),
    };
    let shape = view.shape();
    if shape.len() != 2 {
        return Vec::new();
    }
    let channels = shape[0];
    let num = shape[1];
    if channels < 5 || num == 0 {
        return Vec::new();
    }
    let nc = channels - 4;

    let mut candidates: Vec<Detection> = Vec::new();
    for i in 0..num {
        let cx = view[[0, i]];
        let cy = view[[1, i]];
        let w = view[[2, i]];
        let h = view[[3, i]];

        let mut best_cls = 0usize;
        let mut best_score = view[[4, i]];
        for c in 1..nc {
            let s = view[[4 + c, i]];
            if s > best_score {
                best_score = s;
                best_cls = c;
            }
        }
        if best_score < conf_thres {
            continue;
        }

        let x1 = (cx - w * 0.5 - meta.pad_x) / meta.gain;
        let y1 = (cy - h * 0.5 - meta.pad_y) / meta.gain;
        let x2 = (cx + w * 0.5 - meta.pad_x) / meta.gain;
        let y2 = (cy + h * 0.5 - meta.pad_y) / meta.gain;

        let (x1, y1, x2, y2) = clip_xyxy(x1, y1, x2, y2, meta.orig_w, meta.orig_h);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        candidates.push(Detection {
            class_id: best_cls,
            label: class_name(best_cls),
            conf: best_score,
            x1,
            y1,
            x2,
            y2,
        });
    }

    nms(candidates, iou_thres)
}

fn clip_xyxy(x1: f32, y1: f32, x2: f32, y2: f32, w: u32, h: u32) -> (f32, f32, f32, f32) {
    let wf = w as f32;
    let hf = h as f32;
    (
        x1.clamp(0.0, wf),
        y1.clamp(0.0, hf),
        x2.clamp(0.0, wf),
        y2.clamp(0.0, hf),
    )
}

fn box_iou(a: &Detection, b: &Detection) -> f32 {
    let xx1 = a.x1.max(b.x1);
    let yy1 = a.y1.max(b.y1);
    let xx2 = a.x2.min(b.x2);
    let yy2 = a.y2.min(b.y2);
    let w = (xx2 - xx1).max(0.0);
    let h = (yy2 - yy1).max(0.0);
    let inter = w * h;
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let uni = area_a + area_b - inter;
    if uni <= 0.0 {
        0.0
    } else {
        inter / uni
    }
}

fn nms(mut dets: Vec<Detection>, iou_thres: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep = Vec::new();
    let mut suppressed = vec![false; dets.len()];
    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if suppressed[j] {
                continue;
            }
            if dets[i].class_id != dets[j].class_id {
                continue;
            }
            if box_iou(&dets[i], &dets[j]) > iou_thres {
                suppressed[j] = true;
            }
        }
    }
    keep
}

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
             \t[--conf 0.25] [--iou 0.7] [--out <目录>]"
        );
        std::process::exit(1);
    }

    let model = PathBuf::from(arg_value(&args, "--model").context("需要 --model")?);
    let source = PathBuf::from(arg_value(&args, "--source").context("需要 --source")?);
    let device = YoloDevice::parse(arg_value(&args, "--device").unwrap_or("cpu"));
    let conf: f32 = arg_value(&args, "--conf").unwrap_or("0.25").parse().context("--conf")?;
    let iou: f32 = arg_value(&args, "--iou").unwrap_or("0.7").parse().context("--iou")?;
    let out_dir = arg_value(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("temp/yolo_output"));

    let mut det = YoloDetector::load_with_thresholds(&model, device, conf, iou, 640)?;
    eprintln!("device={}", det.device_label);

    // 加载字体
    let font = load_font();
    let font_scale = 14.0; // 标签文字大小

    fs::create_dir_all(&out_dir)?;
    let images = collect_images(&source)?;
    if images.is_empty() {
        bail!("未找到图片: {}", source.display());
    }

    let mut total_dets = 0usize;
    let mut seen_classes: HashSet<String> = HashSet::new();

    for path in &images {
        let img = image::open(path)
            .with_context(|| format!("读图失败: {}", path.display()))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        let dets = det.detect_rgb8(w, h, img.as_raw())?;
        total_dets += dets.len();

        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
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
        vis.save(&out_path)
            .with_context(|| format!("保存失败: {}", out_path.display()))?;
        println!("  saved {}", out_path.display());
    }

    println!("\n===== 总结 =====");
    println!("图片数: {}", images.len());
    println!("总检测数: {}", total_dets);
    println!("出现类别: {}", seen_classes.into_iter().collect::<Vec<_>>().join(", "));
    println!("输出目录: {}", out_dir.display());

    Ok(())
}