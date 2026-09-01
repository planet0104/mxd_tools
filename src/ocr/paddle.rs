//! PP-OCRv4 识别模型（仅 rec），基于 ONNX Runtime。

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Context, Result};
use image::imageops::FilterType;
use image::RgbImage;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

const INPUT_HEIGHT: u32 = 48;
const MIN_WIDTH: u32 = 32;
const MAX_WIDTH: u32 = 640;

struct PaddleRecEngine {
    session: Session,
    dict: Vec<String>,
    input_buf: Vec<f32>,
}

static ENGINE: OnceLock<Result<Mutex<PaddleRecEngine>, String>> = OnceLock::new();

fn engine() -> Result<&'static Mutex<PaddleRecEngine>> {
    ENGINE
        .get_or_init(|| {
            let (onnx, keys) = model_paths().map_err(|e| e.to_string())?;
            let dict = load_dict(&keys).map_err(|e| e.to_string())?;
            let session = Session::builder()
                .map_err(|e| format!("创建 OCR SessionBuilder 失败: {e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| format!("设置 OCR 图优化失败: {e}"))?
                .with_intra_threads(4)
                .map_err(|e| format!("设置 OCR intra_threads 失败: {e}"))?
                .commit_from_file(&onnx)
                .map_err(|e| format!("加载 OCR ONNX 失败 ({}): {e}", onnx.display()))?;
            eprintln!("OCR: PP-OCRv4 rec (ort CPU)");
            Ok(Mutex::new(PaddleRecEngine {
                session,
                dict,
                input_buf: Vec::new(),
            }))
        })
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn model_paths() -> Result<(PathBuf, PathBuf)> {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/ocr");
    let onnx = base.join("ch_PP-OCRv4_rec_infer.onnx");
    let keys = base.join("ppocr_keys_v1.txt");
    if !onnx.is_file() {
        bail!(
            "找不到 OCR 模型: {}\n请下载 ch_PP-OCRv4_rec_infer.onnx 到 models/ocr/",
            onnx.display()
        );
    }
    if !keys.is_file() {
        bail!(
            "找不到 OCR 字典: {}\n请下载 ppocr_keys_v1.txt 到 models/ocr/",
            keys.display()
        );
    }
    Ok((onnx, keys))
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

fn dst_width(src_w: u32, src_h: u32) -> u32 {
    if src_h == 0 {
        return MIN_WIDTH;
    }
    let ratio = src_w as f32 / src_h as f32;
    let raw = (INPUT_HEIGHT as f32 * ratio).round() as u32;
    raw.clamp(MIN_WIDTH, MAX_WIDTH)
}

fn resize_rgb(img: &RgbImage, dst_w: u32, dst_h: u32) -> RgbImage {
    image::imageops::resize(img, dst_w, dst_h, FilterType::Triangle)
}

/// 单张 RGB 图预处理为 NCHW f32，写入 `out`，返回实际宽度。
fn preprocess_into(img: &RgbImage, out: &mut Vec<f32>) -> u32 {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        out.clear();
        return 0;
    }
    let dw = dst_width(w, h);
    let resized = if w == dw && h == INPUT_HEIGHT {
        img.clone()
    } else {
        resize_rgb(img, dw, INPUT_HEIGHT)
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

fn ctc_greedy_decode_row(
    logits: &[f32],
    seq_len: usize,
    num_classes: usize,
    dict: &[String],
) -> String {
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

impl PaddleRecEngine {
    fn recognize_batch_inner(&mut self, images: &[&RgbImage]) -> Result<Vec<String>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let mut widths = Vec::with_capacity(images.len());
        let mut planes = Vec::with_capacity(images.len());
        for img in images {
            let mut plane = Vec::new();
            let w = preprocess_into(img, &mut plane);
            if w == 0 {
                widths.push(0);
                planes.push(Vec::new());
            } else {
                widths.push(w);
                planes.push(plane);
            }
        }

        let max_w = widths.iter().copied().max().unwrap_or(0) as usize;
        if max_w == 0 {
            return Ok(vec![String::new(); images.len()]);
        }

        let n = images.len();
        let plane_size = INPUT_HEIGHT as usize * max_w;
        let batch_size = n * plane_size * 3;
        self.input_buf.resize(batch_size, -1.0);

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
                    self.input_buf[dst_row..dst_row + w]
                        .copy_from_slice(&plane[src_row..src_row + w]);
                }
            }
        }

        let shape = [n as i64, 3, INPUT_HEIGHT as i64, max_w as i64];
        let input = TensorRef::from_array_view((shape, self.input_buf.as_slice()))
            .context("构造 OCR 输入张量失败")?;
        let outputs = self
            .session
            .run(ort::inputs!["x" => input])
            .context("OCR 推理失败")?;
        let (_name, value) = outputs.iter().next().context("OCR 无输出")?;
        let (out_shape, out_data) = value
            .try_extract_tensor::<f32>()
            .context("解析 OCR 输出失败")?;

        if out_shape.len() != 3 {
            bail!("OCR 输出维度异常: {:?}", out_shape);
        }
        let batch = out_shape[0] as usize;
        let seq_len = out_shape[1] as usize;
        let num_classes = out_shape[2] as usize;

        let mut texts = Vec::with_capacity(batch);
        for b in 0..batch {
            let row_len = seq_len * num_classes;
            let start = b * row_len;
            let end = start + row_len;
            let row = &out_data[start..end];
            texts.push(ctc_greedy_decode_row(row, seq_len, num_classes, &self.dict));
        }
        Ok(texts)
    }
}

/// 对单张 RGB 图像做 OCR。
pub fn recognize_rgb(img: &RgbImage) -> Result<String> {
    let texts = recognize_rgb_batch(&[img])?;
    Ok(texts.into_iter().next().unwrap_or_default())
}

/// 批量 OCR，多张名牌 ROI 一次推理。
pub fn recognize_rgb_batch(imgs: &[&RgbImage]) -> Result<Vec<String>> {
    if imgs.is_empty() {
        return Ok(Vec::new());
    }
    let engine = engine()?;
    let mut guard = engine
        .lock()
        .map_err(|e| anyhow::anyhow!("OCR 引擎锁失败: {e}"))?;
    guard.recognize_batch_inner(imgs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dst_width_clamps() {
        assert_eq!(dst_width(10, 50), 32);
        assert_eq!(dst_width(100, 50), 96);
        assert_eq!(dst_width(2000, 50), 640);
    }
}
