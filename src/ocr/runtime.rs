//! 可配置 CPU/CUDA 的 OCR 运行时（det + rec）。

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use image::imageops::FilterType;
use image::RgbImage;
use opencv::core::{self, Mat, Point, Point2f, Scalar, Size, Vector};
use opencv::imgproc::{self, InterpolationFlags, CHAIN_APPROX_SIMPLE, RETR_LIST};
use opencv::prelude::*;
use ort::value::TensorRef;

use crate::ort_util::{build_session, OrtDevice};

pub use super::det::TextBox;

const LIMIT_SIDE_LEN: u32 = 960;
const DET_THRESH: f32 = 0.3;
const BOX_THRESH: f32 = 0.55;
const UNCLIP_RATIO: f32 = 1.6;
const MIN_BOX_SIDE: f32 = 3.0;
const INPUT_HEIGHT: u32 = 48;
const MIN_WIDTH: u32 = 32;
const MAX_WIDTH: u32 = 640;

struct DetMeta {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

pub struct OcrRuntime {
    det_session: ort::session::Session,
    rec_session: ort::session::Session,
    dict: Vec<String>,
    det_input: Vec<f32>,
    rec_input: Vec<f32>,
    pub device_label: String,
}

impl OcrRuntime {
    pub fn load(device: OrtDevice) -> Result<Self> {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/ocr");
        let det_onnx = base.join("ch_PP-OCRv5_det_mobile.onnx");
        let rec_onnx = base.join("ch_PP-OCRv4_rec_infer.onnx");
        let keys = base.join("ppocr_keys_v1.txt");
        if !det_onnx.is_file() {
            bail!("找不到 OCR det 模型: {}", det_onnx.display());
        }
        if !rec_onnx.is_file() {
            bail!("找不到 OCR rec 模型: {}", rec_onnx.display());
        }
        let dict = load_dict(&keys)?;
        let (det_session, det_label) = build_session(&det_onnx, device, 4)?;
        let (rec_session, rec_label) = build_session(&rec_onnx, device, 4)?;
        let device_label = format!("det={det_label} rec={rec_label}");
        eprintln!("OCR Runtime: {device_label}");
        Ok(Self {
            det_session,
            rec_session,
            dict,
            det_input: Vec::new(),
            rec_input: Vec::new(),
            device_label,
        })
    }

    pub fn detect_text_boxes(&mut self, img: &RgbImage) -> Result<Vec<TextBox>> {
        if img.width() < 8 || img.height() < 8 {
            return Ok(Vec::new());
        }
        let meta = preprocess_det(img, &mut self.det_input)?;
        if meta.dst_w == 0 || meta.dst_h == 0 {
            return Ok(Vec::new());
        }
        let shape = [1_i64, 3, meta.dst_h as i64, meta.dst_w as i64];
        let input = TensorRef::from_array_view((shape, self.det_input.as_slice()))
            .context("构造 det 输入张量失败")?;
        let outputs = self
            .det_session
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

    pub fn recognize_rgb_batch(&mut self, images: &[&RgbImage]) -> Result<Vec<String>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let mut widths = Vec::with_capacity(images.len());
        let mut planes = Vec::with_capacity(images.len());
        for img in images {
            let mut plane = Vec::new();
            let w = preprocess_rec(img, &mut plane);
            widths.push(w);
            planes.push(plane);
        }
        let max_w = widths.iter().copied().max().unwrap_or(0) as usize;
        if max_w == 0 {
            return Ok(vec![String::new(); images.len()]);
        }
        let n = images.len();
        let plane_size = INPUT_HEIGHT as usize * max_w;
        self.rec_input.resize(n * plane_size * 3, -1.0);
        for (i, plane) in planes.iter().enumerate() {
            let w = widths[i] as usize;
            if w == 0 {
                continue;
            }
            let src_plane = INPUT_HEIGHT as usize * w;
            for c in 0..3 {
                let src_off = c * src_plane;
                let dst_off = i * plane_size * 3 + c * plane_size;
                for y in 0..INPUT_HEIGHT as usize {
                    let src_row = src_off + y * w;
                    let dst_row = dst_off + y * max_w;
                    self.rec_input[dst_row..dst_row + w]
                        .copy_from_slice(&plane[src_row..src_row + w]);
                }
            }
        }
        let shape = [n as i64, 3, INPUT_HEIGHT as i64, max_w as i64];
        let input = TensorRef::from_array_view((shape, self.rec_input.as_slice()))
            .context("构造 rec 输入张量失败")?;
        let outputs = self
            .rec_session
            .run(ort::inputs!["x" => input])
            .context("rec 推理失败")?;
        let (_name, value) = outputs.iter().next().context("rec 无输出")?;
        let (out_shape, out_data) = value
            .try_extract_tensor::<f32>()
            .context("解析 rec 输出失败")?;
        if out_shape.len() != 3 {
            bail!("rec 输出维度异常: {:?}", out_shape);
        }
        let batch = out_shape[0] as usize;
        let seq_len = out_shape[1] as usize;
        let num_classes = out_shape[2] as usize;
        let mut texts = Vec::with_capacity(batch);
        for b in 0..batch {
            let row_len = seq_len * num_classes;
            let start = b * row_len;
            texts.push(ctc_greedy_decode_row(
                &out_data[start..start + row_len],
                seq_len,
                num_classes,
                &self.dict,
            ));
        }
        Ok(texts)
    }

    pub fn recognize_rgb(&mut self, img: &RgbImage) -> Result<String> {
        let mut v = self.recognize_rgb_batch(&[img])?;
        Ok(v.pop().unwrap_or_default())
    }
}

fn load_dict(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("读取 OCR 字典失败: {}", path.display()))?;
    let mut dict = vec![String::new()];
    for line in content.lines() {
        dict.push(line.to_string());
    }
    Ok(dict)
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
    let src = Mat::new_rows_cols_with_bytes::<opencv::core::Vec3b>(sh as i32, sw as i32, img.as_raw())
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

fn dst_width(src_w: u32, src_h: u32) -> u32 {
    if src_h == 0 {
        return MIN_WIDTH;
    }
    let ratio = src_w as f32 / src_h as f32;
    ((INPUT_HEIGHT as f32 * ratio).round() as u32).clamp(MIN_WIDTH, MAX_WIDTH)
}

fn preprocess_rec(img: &RgbImage, out: &mut Vec<f32>) -> u32 {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        out.clear();
        return 0;
    }
    let dw = dst_width(w, h);
    let resized = if w == dw && h == INPUT_HEIGHT {
        img.clone()
    } else {
        image::imageops::resize(img, dw, INPUT_HEIGHT, FilterType::Triangle)
    };
    let plane = (INPUT_HEIGHT as usize) * (dw as usize);
    out.resize(plane * 3, 0.0);
    for y in 0..INPUT_HEIGHT as usize {
        for x in 0..dw as usize {
            let [r, g, b] = resized.get_pixel(x as u32, y as u32).0;
            let idx = y * dw as usize + x;
            out[idx] = r as f32 / 255.0;
            out[plane + idx] = g as f32 / 255.0;
            out[plane * 2 + idx] = b as f32 / 255.0;
        }
    }
    for v in out.iter_mut() {
        *v = (*v - 0.5) / 0.5;
    }
    dw
}

fn ctc_greedy_decode_row(logits: &[f32], seq_len: usize, num_classes: usize, dict: &[String]) -> String {
    let mut prev: Option<usize> = None;
    let mut text = String::new();
    for t in 0..seq_len {
        let base = t * num_classes;
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let v = logits[base + c];
            if v > best_val {
                best_val = v;
                best_idx = c;
            }
        }
        if best_idx == 0 {
            prev = None;
            continue;
        }
        if Some(best_idx) == prev {
            continue;
        }
        if let Some(s) = dict.get(best_idx) {
            text.push_str(s);
        }
        prev = Some(best_idx);
    }
    text
}

fn box_score_fast(prob: &Mat, pts: &[(f32, f32)]) -> Result<f32> {
    let xs = pts.iter().map(|p| p.0);
    let ys = pts.iter().map(|p| p.1);
    let xmin = xs.clone().fold(f32::INFINITY, f32::min).max(0.0) as i32;
    let xmax = xs.fold(f32::NEG_INFINITY, f32::max).min(prob.cols() as f32 - 1.0) as i32;
    let ymin = ys.clone().fold(f32::INFINITY, f32::min).max(0.0) as i32;
    let ymax = ys.fold(f32::NEG_INFINITY, f32::max).min(prob.rows() as f32 - 1.0) as i32;
    if xmax < xmin || ymax < ymin {
        return Ok(0.0);
    }
    let roi = prob
        .roi(opencv::core::Rect::new(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
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
    imgproc::fill_poly(&mut mask, &pts_vec, Scalar::all(255.0), imgproc::LINE_8, 0, Point::new(0, 0))
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
    let xs = pts.iter().map(|p| (p.0 * scale_x).clamp(0.0, meta.src_w as f32 - 1.0));
    let ys = pts.iter().map(|p| (p.1 * scale_y).clamp(0.0, meta.src_h as f32 - 1.0));
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
        let contour = contours.get(i).map_err(|e| anyhow::anyhow!("contour get: {e}"))?;
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
    boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(boxes)
}
