use anyhow::Result;
use opencv::core::{self, Mat, Scalar, Size, Vec3b};
use opencv::imgproc::{self, InterpolationFlags};
use opencv::prelude::*;

use crate::yolo::LetterboxMeta;

pub struct LetterboxBuffers {
    bgr: Mat,
    resized: Mat,
    padded: Mat,
}

impl LetterboxBuffers {
    pub fn new() -> opencv::Result<Self> {
        Ok(Self {
            bgr: Mat::default(),
            resized: Mat::default(),
            padded: Mat::default(),
        })
    }
}

/// OpenCV letterbox + NCHW float32 归一化，写入复用缓冲 `out`。
pub fn letterbox_rgb_into(
    rgb: &[u8],
    w: u32,
    h: u32,
    imgsz: u32,
    bufs: &mut LetterboxBuffers,
    out: &mut Vec<f32>,
) -> Result<LetterboxMeta> {
    let wf = w as f32;
    let hf = h as f32;
    let size = imgsz as f32;
    let gain = (size / hf).min(size / wf);
    let nw = (wf * gain).round().max(1.0) as i32;
    let nh = (hf * gain).round().max(1.0) as i32;
    let dw = size - nw as f32;
    let dh = size - nh as f32;
    let pad_x = (dw * 0.5 - 0.1).round();
    let pad_y = (dh * 0.5 - 0.1).round();
    let left = pad_x.max(0.0) as i32;
    let top = pad_y.max(0.0) as i32;
    let right = ((dw - left as f32) + 0.1).round().max(0.0) as i32;
    let bottom = ((dh - top as f32) + 0.1).round().max(0.0) as i32;

    let src = Mat::new_rows_cols_with_bytes::<Vec3b>(h as i32, w as i32, rgb)
        .map_err(|e| anyhow::anyhow!("mat from rgb: {e}"))?;
    imgproc::cvt_color_def(&src, &mut bufs.bgr, imgproc::COLOR_RGB2BGR)
        .map_err(|e| anyhow::anyhow!("cvt_color: {e}"))?;
    imgproc::resize(
        &bufs.bgr,
        &mut bufs.resized,
        Size::new(nw, nh),
        0.0,
        0.0,
        InterpolationFlags::INTER_LINEAR.into(),
    )
    .map_err(|e| anyhow::anyhow!("resize: {e}"))?;
    core::copy_make_border(
        &bufs.resized,
        &mut bufs.padded,
        top,
        bottom,
        left,
        right,
        core::BORDER_CONSTANT,
        Scalar::new(114.0, 114.0, 114.0, 0.0),
    )
    .map_err(|e| anyhow::anyhow!("copy_make_border: {e}"))?;

    let plane = (imgsz as usize) * (imgsz as usize);
    out.resize(plane * 3, 0.0);
    let bytes = bufs
        .padded
        .data_bytes()
        .map_err(|e| anyhow::anyhow!("padded bytes: {e}"))?;
    let inv = 1.0 / 255.0;
    for y in 0..imgsz as usize {
        let row = y * imgsz as usize * 3;
        for x in 0..imgsz as usize {
            let i = row + x * 3;
            let b = bytes[i] as f32 * inv;
            let g = bytes[i + 1] as f32 * inv;
            let r = bytes[i + 2] as f32 * inv;
            let p = y * imgsz as usize + x;
            out[p] = r;
            out[plane + p] = g;
            out[plane + plane + p] = b;
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
