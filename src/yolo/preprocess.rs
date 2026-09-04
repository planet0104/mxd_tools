//! YOLO letterbox 预处理（纯 Rust，`image` crate）。
//! 等比缩放 + 灰边 114，输出 NCHW float32 /255（RGB 通道序）。

use anyhow::{anyhow, bail, Result};
use image::imageops::{self, FilterType};
use image::{Rgb, RgbImage};

use crate::yolo::LetterboxMeta;

/// 可复用的中间缓冲，避免每帧分配大图。
pub struct LetterboxBuffers {
    /// 与 `RgbImage` 交替接管，减少每帧 `Vec` 分配。
    src_bytes: Vec<u8>,
    padded: RgbImage,
}

impl LetterboxBuffers {
    pub fn new() -> Self {
        Self {
            src_bytes: Vec::new(),
            padded: RgbImage::new(0, 0),
        }
    }
}

/// Letterbox + NCHW float32 归一化，写入复用缓冲 `out`。
pub fn letterbox_rgb_into(
    rgb: &[u8],
    w: u32,
    h: u32,
    imgsz: u32,
    bufs: &mut LetterboxBuffers,
    out: &mut Vec<f32>,
) -> Result<LetterboxMeta> {
    let expect = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(3))
        .unwrap_or(0);
    if rgb.len() != expect {
        bail!("RGB 长度不符: got {} expect {expect}", rgb.len());
    }
    if imgsz == 0 {
        bail!("imgsz 不能为 0");
    }

    let wf = w as f32;
    let hf = h as f32;
    let size = imgsz as f32;
    let gain = (size / hf).min(size / wf);
    let nw = (wf * gain).round().max(1.0) as u32;
    let nh = (hf * gain).round().max(1.0) as u32;
    let dw = size - nw as f32;
    let dh = size - nh as f32;
    let pad_x = (dw * 0.5 - 0.1).round();
    let pad_y = (dh * 0.5 - 0.1).round();
    let left = pad_x.max(0.0) as i64;
    let top = pad_y.max(0.0) as i64;

    bufs.src_bytes.clear();
    bufs.src_bytes.extend_from_slice(rgb);
    let src = RgbImage::from_raw(w, h, std::mem::take(&mut bufs.src_bytes))
        .ok_or_else(|| anyhow!("无法构造 RGB 图 {w}x{h}"))?;
    let resized = imageops::resize(&src, nw, nh, FilterType::Triangle);
    bufs.src_bytes = src.into_raw();

    if bufs.padded.width() != imgsz || bufs.padded.height() != imgsz {
        bufs.padded = RgbImage::from_pixel(imgsz, imgsz, Rgb([114, 114, 114]));
    } else {
        for p in bufs.padded.pixels_mut() {
            *p = Rgb([114, 114, 114]);
        }
    }
    imageops::replace(&mut bufs.padded, &resized, left, top);

    let plane = (imgsz as usize) * (imgsz as usize);
    out.resize(plane * 3, 0.0);
    let inv = 1.0 / 255.0;
    let bytes = bufs.padded.as_raw();
    for y in 0..imgsz as usize {
        let row = y * imgsz as usize * 3;
        for x in 0..imgsz as usize {
            let i = row + x * 3;
            let p = y * imgsz as usize + x;
            out[p] = bytes[i] as f32 * inv;
            out[plane + p] = bytes[i + 1] as f32 * inv;
            out[plane + plane + p] = bytes[i + 2] as f32 * inv;
        }
    }

    Ok(LetterboxMeta {
        gain,
        pad_x,
        pad_y,
        orig_w: w,
        orig_h: h,
    })
}
