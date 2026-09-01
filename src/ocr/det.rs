//! PP-OCRv5 文本检测（DB），基于 ONNX Runtime + OpenCV 后处理。

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Context, Result};
use image::RgbImage;
use opencv::core::{self, Mat, Point, Point2f, Scalar, Size, Vector};
use opencv::imgproc::{self, InterpolationFlags, CHAIN_APPROX_SIMPLE, RETR_LIST};
use opencv::prelude::*;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

const LIMIT_SIDE_LEN: u32 = 960;
const DET_THRESH: f32 = 0.3;
const BOX_THRESH: f32 = 0.55;
const UNCLIP_RATIO: f32 = 1.6;
const MIN_BOX_SIDE: f32 = 3.0;

#[derive(Debug, Clone)]
pub struct TextBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub score: f32,
}

struct DetMeta {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

struct PaddleDetEngine {
    session: Session,
    input_buf: Vec<f32>,
}

static DET_ENGINE: OnceLock<Result<Mutex<PaddleDetEngine>, String>> = OnceLock::new();

fn det_engine() -> Result<&'static Mutex<PaddleDetEngine>> {
    DET_ENGINE
        .get_or_init(|| {
            let onnx = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models/ocr/ch_PP-OCRv5_det_mobile.onnx");
            if !onnx.is_file() {
                return Err(format!(
                    "找不到 OCR det 模型: {}\n请下载 ch_PP-OCRv5_det_mobile.onnx 到 models/ocr/",
                    onnx.display()
                ));
            }
            let session = Session::builder()
                .map_err(|e| format!("创建 det SessionBuilder 失败: {e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| format!("设置 det 图优化失败: {e}"))?
                .with_intra_threads(4)
                .map_err(|e| format!("设置 det intra_threads 失败: {e}"))?
                .commit_from_file(&onnx)
                .map_err(|e| format!("加载 det ONNX 失败 ({}): {e}", onnx.display()))?;
            eprintln!("OCR: PP-OCRv5 det (ort CPU)");
            Ok(Mutex::new(PaddleDetEngine {
                session,
                input_buf: Vec::new(),
            }))
        })
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn resize_for_det(w: u32, h: u32) -> (u32, u32) {
    if w == 0 || h == 0 {
        return (32, 32);
    }
    let ratio = if w.max(h) > LIMIT_SIDE_LEN {
        LIMIT_SIDE_LEN as f32 / w.max(h) as f32
    } else {
        1.0
    };
    let mut rw = (w as f32 * ratio).round() as u32;
    let mut rh = (h as f32 * ratio).round() as u32;
    rw = ((rw + 31) / 32) * 32;
    rh = ((rh + 31) / 32) * 32;
    (rw.max(32), rh.max(32))
}

fn preprocess_det(img: &RgbImage, buf: &mut Vec<f32>) -> Result<DetMeta> {
    let (sw, sh) = img.dimensions();
    let (dw, dh) = resize_for_det(sw, sh);
    let src =
        Mat::new_rows_cols_with_bytes::<opencv::core::Vec3b>(sh as i32, sw as i32, img.as_raw())
            .map_err(|e| anyhow::anyhow!("det mat from rgb: {e}"))?;
    let mut dst = Mat::default();
    imgproc::resize(
        &src,
        &mut dst,
        Size::new(dw as i32, dh as i32),
        0.0,
        0.0,
        InterpolationFlags::INTER_LINEAR.into(),
    )
    .map_err(|e| anyhow::anyhow!("det resize: {e}"))?;

    let plane = (dh as usize) * (dw as usize);
    buf.resize(plane * 3, 0.0);
    let bytes = dst
        .data_bytes()
        .map_err(|e| anyhow::anyhow!("det data_bytes: {e}"))?;
    for y in 0..dh as usize {
        for x in 0..dw as usize {
            let i = (y * dw as usize + x) * 3;
            let b = bytes[i] as f32 / 255.0;
            let g = bytes[i + 1] as f32 / 255.0;
            let r = bytes[i + 2] as f32 / 255.0;
            let idx = y * dw as usize + x;
            buf[idx] = (r - 0.5) / 0.5;
            buf[plane + idx] = (g - 0.5) / 0.5;
            buf[plane * 2 + idx] = (b - 0.5) / 0.5;
        }
    }
    Ok(DetMeta {
        src_w: sw,
        src_h: sh,
        dst_w: dw,
        dst_h: dh,
    })
}

fn box_score_fast(prob: &Mat, pts: &[(f32, f32)]) -> Result<f32> {
    let xs = pts.iter().map(|p| p.0);
    let ys = pts.iter().map(|p| p.1);
    let xmin = xs.clone().fold(f32::INFINITY, f32::min).max(0.0) as i32;
    let xmax = xs
        .fold(f32::NEG_INFINITY, f32::max)
        .min(prob.cols() as f32 - 1.0) as i32;
    let ymin = ys.clone().fold(f32::INFINITY, f32::min).max(0.0) as i32;
    let ymax = ys
        .fold(f32::NEG_INFINITY, f32::max)
        .min(prob.rows() as f32 - 1.0) as i32;
    if xmax < xmin || ymax < ymin {
        return Ok(0.0);
    }
    let roi = prob
        .roi(opencv::core::Rect::new(
            xmin,
            ymin,
            xmax - xmin + 1,
            ymax - ymin + 1,
        ))
        .map_err(|e| anyhow::anyhow!("prob roi: {e}"))?;
    let mut mask = Mat::zeros(roi.rows(), roi.cols(), opencv::core::CV_8UC1)
        .map_err(|e| anyhow::anyhow!("mask zeros: {e}"))?
        .to_mat()
        .map_err(|e| anyhow::anyhow!("mask mat: {e}"))?;
    let mut poly = Vector::<Point>::new();
    for (x, y) in pts {
        poly.push(Point::new(
            (x - xmin as f32).round() as i32,
            (y - ymin as f32).round() as i32,
        ));
    }
    let pts_vec = Vector::<Vector<Point>>::from(vec![poly]);
    imgproc::fill_poly(
        &mut mask,
        &pts_vec,
        Scalar::all(255.0),
        imgproc::LINE_8,
        0,
        Point::new(0, 0),
    )
    .map_err(|e| anyhow::anyhow!("fill_poly: {e}"))?;
    let mean = core::mean(&roi, &mask).map_err(|e| anyhow::anyhow!("mean: {e}"))?;
    Ok(mean[0] as f32)
}

fn min_area_quad(contour: &Vector<Point>) -> Option<[(f32, f32); 4]> {
    if contour.len() < 3 {
        return None;
    }
    let rect = imgproc::min_area_rect(contour).ok()?;
    let mut pts = [Point2f::default(); 4];
    rect.points(&mut pts).ok()?;
    Some([
        (pts[0].x, pts[0].y),
        (pts[1].x, pts[1].y),
        (pts[2].x, pts[2].y),
        (pts[3].x, pts[3].y),
    ])
}

fn unclip_quad(pts: [(f32, f32); 4], ratio: f32) -> [(f32, f32); 4] {
    let cx = pts.iter().map(|p| p.0).sum::<f32>() / 4.0;
    let cy = pts.iter().map(|p| p.1).sum::<f32>() / 4.0;
    let scale = ratio.sqrt();
    pts.map(|(x, y)| (cx + (x - cx) * scale, cy + (y - cy) * scale))
}

fn quad_to_aabb(pts: &[(f32, f32); 4], meta: &DetMeta) -> Option<TextBox> {
    let scale_x = meta.src_w as f32 / meta.dst_w as f32;
    let scale_y = meta.src_h as f32 / meta.dst_h as f32;
    let xs = pts
        .iter()
        .map(|p| (p.0 * scale_x).clamp(0.0, meta.src_w as f32 - 1.0));
    let ys = pts
        .iter()
        .map(|p| (p.1 * scale_y).clamp(0.0, meta.src_h as f32 - 1.0));
    let x1 = xs.clone().fold(f32::INFINITY, f32::min);
    let x2 = xs.fold(f32::NEG_INFINITY, f32::max);
    let y1 = ys.clone().fold(f32::INFINITY, f32::min);
    let y2 = ys.fold(f32::NEG_INFINITY, f32::max);
    let w = (x2 - x1).round() as u32;
    let h = (y2 - y1).round() as u32;
    if w < 3 || h < 3 {
        return None;
    }
    Some(TextBox {
        x: x1.round() as u32,
        y: y1.round() as u32,
        w,
        h,
        score: 0.0,
    })
}

fn decode_db_boxes(prob: &[f32], rows: usize, cols: usize, meta: &DetMeta) -> Result<Vec<TextBox>> {
    let mut binary = vec![0u8; rows * cols];
    for (i, v) in prob.iter().enumerate() {
        if *v > DET_THRESH {
            binary[i] = 255;
        }
    }
    let mat = Mat::new_rows_cols_with_bytes::<u8>(rows as i32, cols as i32, &binary)
        .map_err(|e| anyhow::anyhow!("binary mat: {e}"))?;
    let prob_f32 = Mat::from_slice(prob)
        .map_err(|e| anyhow::anyhow!("prob from_slice: {e}"))?
        .reshape(1, rows as i32)
        .map_err(|e| anyhow::anyhow!("prob reshape: {e}"))?
        .try_clone()
        .map_err(|e| anyhow::anyhow!("prob clone: {e}"))?;

    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(
        &mat,
        &mut contours,
        RETR_LIST,
        CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )
    .map_err(|e| anyhow::anyhow!("find_contours: {e}"))?;

    let mut boxes = Vec::new();
    for i in 0..contours.len() {
        let contour = contours
            .get(i)
            .map_err(|e| anyhow::anyhow!("contour get: {e}"))?;
        let Some(mut quad) = min_area_quad(&contour) else {
            continue;
        };
        let side = {
            let d1 = ((quad[0].0 - quad[1].0).powi(2) + (quad[0].1 - quad[1].1).powi(2)).sqrt();
            let d2 = ((quad[0].0 - quad[3].0).powi(2) + (quad[0].1 - quad[3].1).powi(2)).sqrt();
            d1.min(d2)
        };
        if side < MIN_BOX_SIDE {
            continue;
        }
        let score = box_score_fast(&prob_f32, &quad)?;
        if score < BOX_THRESH {
            continue;
        }
        quad = unclip_quad(quad, UNCLIP_RATIO);
        let Some(mut tb) = quad_to_aabb(&quad, meta) else {
            continue;
        };
        tb.score = score;
        boxes.push(tb);
    }

    boxes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(boxes)
}

impl PaddleDetEngine {
    fn detect_inner(&mut self, img: &RgbImage) -> Result<Vec<TextBox>> {
        let meta = preprocess_det(img, &mut self.input_buf)?;
        if meta.dst_w == 0 || meta.dst_h == 0 {
            return Ok(Vec::new());
        }
        let shape = [1_i64, 3, meta.dst_h as i64, meta.dst_w as i64];
        let input = TensorRef::from_array_view((shape, self.input_buf.as_slice()))
            .context("构造 det 输入张量失败")?;
        let outputs = self
            .session
            .run(ort::inputs!["x" => input])
            .context("det 推理失败")?;
        let (_name, value) = outputs.iter().next().context("det 无输出")?;
        let (out_shape, out_data) = value
            .try_extract_tensor::<f32>()
            .context("解析 det 输出失败")?;
        if out_shape.len() != 4 || out_shape[1] != 1 {
            bail!("det 输出维度异常: {:?}", out_shape);
        }
        let rows = out_shape[2] as usize;
        let cols = out_shape[3] as usize;
        decode_db_boxes(out_data, rows, cols, &meta)
    }
}

/// 在图像中检测文本框（坐标相对 `img` 左上角）。
pub fn detect_text_boxes(img: &RgbImage) -> Result<Vec<TextBox>> {
    if img.width() < 8 || img.height() < 8 {
        return Ok(Vec::new());
    }
    let engine = det_engine()?;
    let mut guard = engine
        .lock()
        .map_err(|e| anyhow::anyhow!("det 引擎锁失败: {e}"))?;
    guard.detect_inner(img)
}
