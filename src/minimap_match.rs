//! 小地图定位（OpenCV，对齐 scripts/locate_from_screencaps.py）。

use std::fs;
use std::path::{Path, PathBuf};

use image::{Rgb, RgbImage};
use opencv::calib3d;
use opencv::core::{self, KeyPoint, Mat, MatTraitConst, Point, Point2f, Scalar, Size, Vec3b, Vector};
use opencv::features2d::{self, DescriptorMatcherTrait, Feature2DTrait};
use opencv::imgproc::{self, InterpolationFlags, TemplateMatchModes};
use opencv::prelude::*;

use crate::image_util::{
    crop_rgb, draw_rect, mark_player_diamond, mark_player_diamond_small,
};
use crate::locate::{Align, LocateResult, VIEW_H, VIEW_W, VIEW_X, VIEW_Y};
use crate::paths::safe_filename;

const CLASSIC_VIEW: (u32, u32, u32, u32) = (VIEW_X, VIEW_Y, VIEW_W, VIEW_H);

#[derive(Debug, Clone)]
pub struct MinimapHit {
    pub mode: &'static str,
    pub score: f64,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub scale: f64,
    pub align: Option<Align>,
}

#[derive(Debug, Clone)]
pub struct CapsValidateSummary {
    pub total: usize,
    pub ok: usize,
    pub out_dir: PathBuf,
    pub lines: Vec<String>,
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![a];
    }
    (0..n)
        .map(|i| a + (b - a) * (i as f64) / ((n - 1) as f64))
        .collect()
}

/// image::RgbImage (RGB) → OpenCV Mat CV_8UC3 (BGR)。
fn rgb_to_bgr_mat(img: &RgbImage) -> opencv::Result<Mat> {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let mut mat = Mat::new_rows_cols_with_default(h, w, core::CV_8UC3, Scalar::all(0.0))?;
    for y in 0..img.height() {
        for x in 0..img.width() {
            let p = img.get_pixel(x, y).0;
            *mat.at_2d_mut::<Vec3b>(y as i32, x as i32)? = Vec3b::from([p[2], p[1], p[0]]);
        }
    }
    Ok(mat)
}

fn mat_to_gray(bgr: &Mat) -> opencv::Result<Mat> {
    let mut gray = Mat::default();
    imgproc::cvt_color_def(bgr, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    Ok(gray)
}

/// 玩家黄菱形：#FFFF88 心 + #FFFF00 边。取最密 7×7 簇，返回中心与底边。
fn find_yellow(rgb: &RgbImage) -> Option<(f64, f64, f64, usize)> {
    let mut pts: Vec<(i32, i32)> = Vec::new();
    for (x, y, p) in rgb.enumerate_pixels() {
        let [r, g, b] = p.0;
        let core = r >= 248 && g >= 248 && b >= 100 && b <= 155;
        let edge = r >= 248 && g >= 240 && b <= 45;
        let soft = r >= 240 && g >= 240 && b <= 165 && b >= 70;
        if core || edge || soft {
            pts.push((x as i32, y as i32));
        }
    }
    if pts.is_empty() {
        return None;
    }
    let mut best: Option<(usize, i32, i32)> = None;
    for &(x, y) in &pts {
        let cnt = pts
            .iter()
            .filter(|(px, py)| (px - x).abs() <= 2 && (py - y).abs() <= 2)
            .count();
        if best.is_none_or(|(c, _, _)| cnt > c) {
            best = Some((cnt, x, y));
        }
    }
    let (_, bx, by) = best.unwrap();
    let cluster: Vec<(i32, i32)> = pts
        .into_iter()
        .filter(|(px, py)| (px - bx).abs() <= 3 && (py - by).abs() <= 3)
        .collect();
    let n = cluster.len();
    let cx = cluster.iter().map(|(x, _)| *x as f64).sum::<f64>() / n as f64;
    let cy = cluster.iter().map(|(_, y)| *y as f64).sum::<f64>() / n as f64;
    let cy_bot = cluster.iter().map(|(_, y)| *y as f64).fold(f64::NEG_INFINITY, f64::max);
    Some((cx, cy, cy_bot, n))
}

fn is_sky_px(p: [u8; 3]) -> bool {
    (u16::from(p[0]) + u16::from(p[1]) + u16::from(p[2])) < 54
}

fn is_grass_px(p: [u8; 3]) -> bool {
    let (r, g, b) = (p[0] as i16, p[1] as i16, p[2] as i16);
    g >= r && g > 80 && b < g && (g - b) > 20
}

/// 黄点落在小地图「天空」时，向下吸附到最近草地平台；否则用菱形底边作脚底。
fn refine_player_on_canvas(canvas: &RgbImage, cx: f64, cy: f64, cy_bot: f64) -> (f64, f64) {
    let w = canvas.width() as i32;
    let h = canvas.height() as i32;
    let x = (cx.round() as i32).clamp(0, w - 1);
    let y = (cy.round() as i32).clamp(0, h - 1);
    let y0 = (cy_bot.round() as i32).clamp(0, h - 1);
    if !is_sky_px(canvas.get_pixel(x as u32, y as u32).0) {
        return (cx, cy_bot.clamp(0.0, (h - 1) as f64));
    }
    const MAX_DY: i32 = 14;
    for dy in 0..=MAX_DY {
        let yy = (y0 + dy).min(h - 1);
        let x0 = (x - 2).max(0);
        let x1 = (x + 3).min(w);
        let mut run = 0;
        for xx in x0..x1 {
            if is_grass_px(canvas.get_pixel(xx as u32, yy as u32).0) {
                run += 1;
            }
        }
        if run >= 2 {
            return (x as f64, yy as f64);
        }
    }
    (cx, cy_bot.clamp(0.0, (h - 1) as f64))
}

/// minimap 画布纵向内容带（非黑行）。
fn canvas_y_band(canvas: &RgbImage) -> (f64, f64) {
    let w = canvas.width() as f64;
    let h = canvas.height();
    let mut ys = Vec::new();
    for y in 0..h {
        let mut n = 0u32;
        for x in 0..canvas.width() {
            if !is_sky_px(canvas.get_pixel(x, y).0) {
                n += 1;
            }
        }
        if n as f64 / w > 0.08 {
            ys.push(y);
        }
    }
    if ys.is_empty() {
        (0.0, (h.saturating_sub(1)) as f64)
    } else {
        (*ys.first().unwrap() as f64, *ys.last().unwrap() as f64)
    }
}

/// 完整大地图纵向可玩内容带（排除天空蓝 / 近黑）。
fn full_y_band(full: &RgbImage) -> (f64, f64) {
    let w = full.width() as f64;
    let h = full.height();
    let mut ys = Vec::new();
    for y in 0..h {
        let mut n = 0u32;
        for x in 0..full.width() {
            let [r, g, b] = full.get_pixel(x, y).0;
            let ri = r as i16;
            let gi = g as i16;
            let bi = b as i16;
            let sky = bi > 140 && bi > ri + 15 && bi > gi + 10;
            let black = (u16::from(r) + u16::from(g) + u16::from(b)) < 40;
            if !sky && !black {
                n += 1;
            }
        }
        if n as f64 / w > 0.04 {
            ys.push(y);
        }
    }
    if ys.is_empty() {
        (0.0, (h.saturating_sub(1)) as f64)
    } else {
        (*ys.first().unwrap() as f64, *ys.last().unwrap() as f64)
    }
}

fn content_band_map(canvas: &RgbImage, full: &RgbImage, cx: f64, cy: f64) -> (f64, f64) {
    let fx = cx / canvas.width() as f64 * full.width() as f64;
    let (mt, mb) = canvas_y_band(canvas);
    let (ft, fb) = full_y_band(full);
    let span_m = (mb - mt).max(1.0);
    let fy = ft + (cy - mt) / span_m * (fb - ft);
    (
        fx.clamp(0.0, (full.width() - 1) as f64),
        fy.clamp(0.0, (full.height() - 1) as f64),
    )
}

fn rgb_to_gray_mat(img: &RgbImage) -> opencv::Result<Mat> {
    let bgr = rgb_to_bgr_mat(img)?;
    mat_to_gray(&bgr)
}

fn sil_gray_mat(img: &RgbImage, painted: bool) -> opencv::Result<Mat> {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let mut out = Mat::new_rows_cols_with_default(h, w, core::CV_8UC1, Scalar::all(0.0))?;
    for y in 0..img.height() {
        for x in 0..img.width() {
            let [r, g, b] = img.get_pixel(x, y).0;
            let on = if painted {
                let ri = r as i16;
                let gi = g as i16;
                let bi = b as i16;
                let sky = bi > 130 && bi > ri + 10 && bi > gi + 5;
                let black = (u16::from(r) + u16::from(g) + u16::from(b)) < 40;
                !sky && !black
            } else {
                !is_sky_px([r, g, b])
            };
            *out.at_2d_mut::<u8>(y as i32, x as i32)? = if on { 255 } else { 0 };
        }
    }
    Ok(out)
}

/// 大地图先缩到与小地图同宽，再 SIFT+FLANN；失败则剪影模板 / 内容带。
struct CanvasToFullAlign {
    mode: String,
    boost: i32,
    sx_back: f64,
    sy_back: f64,
    /// mini_boost → full_small_boost 的单应
    h: Option<Mat>,
    tmpl_xy: (i32, i32),
    full_w: u32,
    full_h: u32,
}

impl CanvasToFullAlign {
    fn build(canvas: &RgbImage, full: &RgbImage) -> Result<Self, String> {
        let boost = 3_i32;
        let cw = canvas.width() as i32;
        let ch = canvas.height() as i32;
        let fw = full.width() as i32;
        let fh = full.height() as i32;
        let fm_w = cw;
        let fm_h = ((fh as f64 * cw as f64 / fw as f64).round() as i32).max(1);
        let sx_back = fw as f64 / fm_w as f64;
        let sy_back = fh as f64 / fm_h as f64;

        let mut align = Self {
            mode: "content_band".into(),
            boost,
            sx_back,
            sy_back,
            h: None,
            tmpl_xy: (0, 0),
            full_w: full.width(),
            full_h: full.height(),
        };

        let full_s = {
            let full_bgr = rgb_to_bgr_mat(full).map_err(|e| e.to_string())?;
            let mut resized = Mat::default();
            imgproc::resize(
                &full_bgr,
                &mut resized,
                Size::new(fm_w, fm_h),
                0.0,
                0.0,
                InterpolationFlags::INTER_AREA.into(),
            )
            .map_err(|e| e.to_string())?;
            resized
        };
        let mini_g = rgb_to_gray_mat(canvas).map_err(|e| e.to_string())?;
        let mut full_s_g = Mat::default();
        imgproc::cvt_color_def(&full_s, &mut full_s_g, imgproc::COLOR_BGR2GRAY)
            .map_err(|e| e.to_string())?;

        let mut mini_b = Mat::default();
        let mut full_b = Mat::default();
        imgproc::resize(
            &mini_g,
            &mut mini_b,
            Size::new(fm_w * boost, ch * boost),
            0.0,
            0.0,
            InterpolationFlags::INTER_NEAREST.into(),
        )
        .map_err(|e| e.to_string())?;
        imgproc::resize(
            &full_s_g,
            &mut full_b,
            Size::new(fm_w * boost, fm_h * boost),
            0.0,
            0.0,
            InterpolationFlags::INTER_AREA.into(),
        )
        .map_err(|e| e.to_string())?;

        if let Ok((h, nin)) = sift_flann_homography(&mini_b, &full_b) {
            align.mode = format!("sift_flann(inliers={nin})");
            align.h = Some(h);
            return Ok(align);
        }

        // 剪影模板匹配（同尺度）
        let ms = sil_gray_mat(canvas, false).map_err(|e| e.to_string())?;
        let fs = {
            let mut rgb = image::RgbImage::new(fm_w as u32, fm_h as u32);
            for y in 0..fm_h {
                for x in 0..fm_w {
                    let p = full_s.at_2d::<Vec3b>(y, x).map_err(|e| e.to_string())?;
                    rgb.put_pixel(x as u32, y as u32, Rgb([p[2], p[1], p[0]]));
                }
            }
            sil_gray_mat(&rgb, true).map_err(|e| e.to_string())?
        };
        let mut result = Mat::default();
        imgproc::match_template(
            &fs,
            &ms,
            &mut result,
            TemplateMatchModes::TM_CCOEFF_NORMED.into(),
            &core::no_array(),
        )
        .map_err(|e| e.to_string())?;
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        let mut min_loc = Point::default();
        let mut max_loc = Point::default();
        core::min_max_loc(
            &result,
            Some(&mut min_val),
            Some(&mut max_val),
            Some(&mut min_loc),
            Some(&mut max_loc),
            &core::no_array(),
        )
        .map_err(|e| e.to_string())?;
        if max_val >= 0.12 {
            align.mode = format!("sil_template(score={max_val:.3})");
            align.tmpl_xy = (max_loc.x, max_loc.y);
        }
        Ok(align)
    }

    fn map_xy(&self, canvas: &RgbImage, full: &RgbImage, cx: f64, cy: f64) -> (f64, f64) {
        if let Some(ref h) = self.h {
            let b = self.boost as f64;
            let src = Vector::<Point2f>::from_iter([Point2f::new(
                (cx * b) as f32,
                (cy * b) as f32,
            )]);
            let mut dst = Vector::<Point2f>::new();
            if core::perspective_transform(&src, &mut dst, h).is_ok() {
                if let Some(p) = dst.get(0).ok() {
                    let fx = p.x as f64 / b * self.sx_back;
                    let fy = p.y as f64 / b * self.sy_back;
                    return (
                        fx.clamp(0.0, (self.full_w - 1) as f64),
                        fy.clamp(0.0, (self.full_h - 1) as f64),
                    );
                }
            }
        }
        if self.mode.starts_with("sil_template") {
            let (ox, oy) = self.tmpl_xy;
            let fx = (ox as f64 + cx) * self.sx_back;
            let fy = (oy as f64 + cy) * self.sy_back;
            return (
                fx.clamp(0.0, (self.full_w - 1) as f64),
                fy.clamp(0.0, (self.full_h - 1) as f64),
            );
        }
        content_band_map(canvas, full, cx, cy)
    }
}

fn sift_flann_homography(mini_b: &Mat, full_b: &Mat) -> Result<(Mat, i32), String> {
    let mut sift = features2d::SIFT::create(2000, 3, 0.02, 10.0, 1.6, false)
        .map_err(|e| e.to_string())?;
    let mut kp1 = Vector::<KeyPoint>::new();
    let mut kp2 = Vector::<KeyPoint>::new();
    let mut d1 = Mat::default();
    let mut d2 = Mat::default();
    sift.detect_and_compute(mini_b, &core::no_array(), &mut kp1, &mut d1, false)
        .map_err(|e| e.to_string())?;
    sift.detect_and_compute(full_b, &core::no_array(), &mut kp2, &mut d2, false)
        .map_err(|e| e.to_string())?;
    if kp1.len() < 8 || kp2.len() < 8 || d1.empty() || d2.empty() {
        return Err("SIFT 特征不足".into());
    }

    let mut matcher = features2d::FlannBasedMatcher::create()
        .map_err(|e| e.to_string())?;
    DescriptorMatcherTrait::add(&mut matcher, &d2).map_err(|e| e.to_string())?;
    DescriptorMatcherTrait::train(&mut matcher).map_err(|e| e.to_string())?;
    let mut knn = Vector::<Vector<core::DMatch>>::new();
    DescriptorMatcherTrait::knn_match_def(&mut matcher, &d1, &mut knn, 2)
        .map_err(|e| e.to_string())?;

    let mut src_pts = Vector::<Point2f>::new();
    let mut dst_pts = Vector::<Point2f>::new();
    for i in 0..knn.len() {
        let pair = knn.get(i).map_err(|e| e.to_string())?;
        if pair.len() < 2 {
            continue;
        }
        let m = pair.get(0).map_err(|e| e.to_string())?;
        let n = pair.get(1).map_err(|e| e.to_string())?;
        if m.distance < 0.8 * n.distance {
            let k1 = kp1.get(m.query_idx as usize).map_err(|e| e.to_string())?;
            let k2 = kp2.get(m.train_idx as usize).map_err(|e| e.to_string())?;
            src_pts.push(k1.pt());
            dst_pts.push(k2.pt());
        }
    }
    if src_pts.len() < 8 {
        return Err(format!("优质匹配不足: {}", src_pts.len()));
    }

    let mut mask = Mat::default();
    let h = calib3d::find_homography(
        &src_pts,
        &dst_pts,
        &mut mask,
        calib3d::RANSAC,
        4.0,
    )
    .map_err(|e| e.to_string())?;
    if h.empty() {
        return Err("findHomography 失败".into());
    }
    let mut inliers = 0;
    if !mask.empty() {
        for i in 0..mask.rows() {
            let v = *mask.at_2d::<u8>(i, 0).map_err(|e| e.to_string())?;
            if v != 0 {
                inliers += 1;
            }
        }
    }
    if inliers < 8 {
        return Err(format!("内点不足: {inliers}"));
    }
    Ok((h, inliers))
}

fn canvas_xy_to_full(
    align: &CanvasToFullAlign,
    canvas: &RgbImage,
    full: &RgbImage,
    cx: f64,
    cy: f64,
) -> (f64, f64) {
    align.map_xy(canvas, full, cx, cy)
}

fn zero_yellow_in_gray(bgr: &Mat, gray: &mut Mat) -> opencv::Result<()> {
    let h = bgr.rows();
    let w = bgr.cols();
    for y in 0..h {
        for x in 0..w {
            let p = bgr.at_2d::<Vec3b>(y, x)?;
            // BGR
            let (b, g, r) = (p[0], p[1], p[2]);
            if r >= 240 && g >= 240 && b <= 170 {
                *gray.at_2d_mut::<u8>(y, x)? = 0;
            }
        }
    }
    Ok(())
}

fn search_roi(shot: &RgbImage) -> (u32, u32) {
    let w = shot.width();
    let h = shot.height();
    let sw = ((w as f64 * 0.48).round() as u32).min(w).max(220.min(w));
    let sh = ((h as f64 * 0.58).round() as u32).min(h).max(220.min(h));
    (sw, sh)
}

fn match_scaled(
    search_g: &Mat,
    canvas_g: &Mat,
    mask0: &Mat,
    scale: f64,
) -> opencv::Result<Option<(f64, i32, i32, i32, i32)>> {
    let cw = canvas_g.cols();
    let ch = canvas_g.rows();
    let sw = search_g.cols();
    let sh = search_g.rows();
    let tw = (cw as f64 * scale).round() as i32;
    let th = (ch as f64 * scale).round() as i32;
    if tw < 40 || th < 24 || tw >= sw - 2 || th >= sh - 2 {
        return Ok(None);
    }
    let mut templ = Mat::default();
    let mut mask = Mat::default();
    imgproc::resize(
        canvas_g,
        &mut templ,
        Size::new(tw, th),
        0.0,
        0.0,
        InterpolationFlags::INTER_AREA.into(),
    )?;
    imgproc::resize(
        mask0,
        &mut mask,
        Size::new(tw, th),
        0.0,
        0.0,
        InterpolationFlags::INTER_NEAREST.into(),
    )?;
    let nz = core::count_non_zero(&mask)?;
    if nz < 80 {
        return Ok(None);
    }
    let mut result = Mat::default();
    imgproc::match_template(
        search_g,
        &templ,
        &mut result,
        TemplateMatchModes::TM_CCOEFF_NORMED.into(),
        &mask,
    )?;
    let mut min_val = 0.0;
    let mut max_val = 0.0;
    let mut min_loc = Point::default();
    let mut max_loc = Point::default();
    core::min_max_loc(
        &result,
        Some(&mut min_val),
        Some(&mut max_val),
        Some(&mut min_loc),
        Some(&mut max_loc),
        &core::no_array(),
    )?;
    Ok(Some((max_val, max_loc.x, max_loc.y, tw, th)))
}

/// 多尺度：完整 minimap 画布 → 截图左上角。
pub fn find_canvas_on_screen(shot: &RgbImage, canvas: &RgbImage) -> Result<Option<MinimapHit>, String> {
    let (sw, sh) = search_roi(shot);
    let search_rgb = crop_rgb(shot, 0, 0, sw, sh);
    let search_bgr = rgb_to_bgr_mat(&search_rgb).map_err(|e| e.to_string())?;
    let mut search_g = mat_to_gray(&search_bgr).map_err(|e| e.to_string())?;
    zero_yellow_in_gray(&search_bgr, &mut search_g).map_err(|e| e.to_string())?;

    let canvas_bgr = rgb_to_bgr_mat(canvas).map_err(|e| e.to_string())?;
    let canvas_g = mat_to_gray(&canvas_bgr).map_err(|e| e.to_string())?;
    let mut mask0 = Mat::default();
    core::compare(
        &canvas_g,
        &Scalar::all(12.0),
        &mut mask0,
        core::CMP_GT,
    )
    .map_err(|e| e.to_string())?;

    let mut scales = linspace(0.7, 2.8, 55);
    if !scales.iter().any(|s| (*s - 1.0).abs() < 1e-9) {
        scales.push(1.0);
    }

    let mut best: Option<MinimapHit> = None;
    for scale in scales {
        let hit = match_scaled(&search_g, &canvas_g, &mask0, scale).map_err(|e| e.to_string())?;
        let Some((score, x, y, tw, th)) = hit else {
            continue;
        };
        let cand = MinimapHit {
            mode: "canvas_on_screen",
            score,
            x: x as u32,
            y: y as u32,
            w: tw as u32,
            h: th as u32,
            scale,
            align: None,
        };
        if best.as_ref().is_none_or(|b| cand.score > b.score) {
            best = Some(cand);
        }
    }
    Ok(best)
}

fn view_to_canvas_align(view: &RgbImage, canvas: &RgbImage) -> Result<Option<Align>, String> {
    let view_bgr = rgb_to_bgr_mat(view).map_err(|e| e.to_string())?;
    let mut vg = mat_to_gray(&view_bgr).map_err(|e| e.to_string())?;
    zero_yellow_in_gray(&view_bgr, &mut vg).map_err(|e| e.to_string())?;
    let canvas_bgr = rgb_to_bgr_mat(canvas).map_err(|e| e.to_string())?;
    let cg = mat_to_gray(&canvas_bgr).map_err(|e| e.to_string())?;

    let vh = vg.rows();
    let vw = vg.cols();
    let ch = cg.rows();
    let cw = cg.cols();
    let mut best: Option<Align> = None;

    let mut consider = |mode: &'static str, score: f64, x: i32, y: i32, sx: f64, sy: f64| {
        let cand = Align {
            mode,
            score,
            loc: (x as u32, y as u32),
            scale: (sx, sy),
            view_origin: (0.0, 0.0),
        };
        if best.as_ref().is_none_or(|b| cand.score > b.score) {
            best = Some(cand);
        }
    };

    let run_match = |src: &Mat, tpl: &Mat| -> opencv::Result<(f64, i32, i32)> {
        let mut result = Mat::default();
        imgproc::match_template(
            src,
            tpl,
            &mut result,
            TemplateMatchModes::TM_CCOEFF_NORMED.into(),
            &core::no_array(),
        )?;
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        let mut min_loc = Point::default();
        let mut max_loc = Point::default();
        core::min_max_loc(
            &result,
            Some(&mut min_val),
            Some(&mut max_val),
            Some(&mut min_loc),
            Some(&mut max_loc),
            &core::no_array(),
        )?;
        Ok((max_val, max_loc.x, max_loc.y))
    };

    if vw <= cw && vh <= ch {
        if let Ok((s, x, y)) = run_match(&cg, &vg) {
            consider("view", s, x, y, 1.0, 1.0);
        }
    }
    let nw = ((vw as f64 * ch as f64 / vh.max(1) as f64).round() as i32).max(8);
    if nw <= cw {
        let mut scaled = Mat::default();
        imgproc::resize(
            &vg,
            &mut scaled,
            Size::new(nw, ch),
            0.0,
            0.0,
            InterpolationFlags::INTER_AREA.into(),
        )
        .map_err(|e| e.to_string())?;
        if let Ok((s, x, y)) = run_match(&cg, &scaled) {
            consider("hscroll", s, x, y, nw as f64 / vw as f64, ch as f64 / vh as f64);
        }
    }
    let nh = ((vh as f64 * cw as f64 / vw.max(1) as f64).round() as i32).max(8);
    if nh <= ch {
        let mut scaled = Mat::default();
        imgproc::resize(
            &vg,
            &mut scaled,
            Size::new(cw, nh),
            0.0,
            0.0,
            InterpolationFlags::INTER_AREA.into(),
        )
        .map_err(|e| e.to_string())?;
        if let Ok((s, x, y)) = run_match(&cg, &scaled) {
            consider("vscroll", s, x, y, cw as f64 / vw as f64, nh as f64 / vh as f64);
        }
    }
    if cw <= vw && ch <= vh {
        if let Ok((s, x, y)) = run_match(&vg, &cg) {
            consider("canvas_in_view", s, x, y, 1.0, 1.0);
        }
    }
    Ok(best)
}

pub fn find_view_fallback(shot: &RgbImage, canvas: &RgbImage) -> Result<Option<MinimapHit>, String> {
    let mut best: Option<MinimapHit> = None;
    let (w, h) = (shot.width(), shot.height());
    for ui_scale in linspace(0.75, 1.5, 16) {
        let vx = (CLASSIC_VIEW.0 as f64 * ui_scale).round() as u32;
        let vy = (CLASSIC_VIEW.1 as f64 * ui_scale).round() as u32;
        let vw = (CLASSIC_VIEW.2 as f64 * ui_scale).round() as u32;
        let vh = (CLASSIC_VIEW.3 as f64 * ui_scale).round() as u32;
        let mut y0 = 0u32;
        while y0 <= 46 {
            for x0 in 0..8u32 {
                let x = x0 + vx;
                let y = y0 + vy;
                if x + vw > w || y + vh > h {
                    continue;
                }
                let view = crop_rgb(shot, x, y, vw, vh);
                if find_yellow(&view).is_none() {
                    continue;
                }
                let Some(align) = view_to_canvas_align(&view, canvas)? else {
                    continue;
                };
                let cand = MinimapHit {
                    mode: "view_fallback",
                    score: align.score,
                    x,
                    y,
                    w: vw,
                    h: vh,
                    scale: ui_scale,
                    align: Some(align),
                };
                if best.as_ref().is_none_or(|b| cand.score > b.score) {
                    best = Some(cand);
                }
            }
            y0 += 2;
        }
    }
    Ok(best)
}

pub fn find_minimap_region(shot: &RgbImage, canvas: &RgbImage) -> Result<MinimapHit, String> {
    let canvas_hit = find_canvas_on_screen(shot, canvas)?;
    let use_fb = canvas_hit
        .as_ref()
        .map(|h| h.score < 0.42)
        .unwrap_or(true);
    if use_fb {
        if let Some(fb) = find_view_fallback(shot, canvas)? {
            if canvas_hit
                .as_ref()
                .map(|h| fb.score > h.score + 0.05)
                .unwrap_or(true)
            {
                return Ok(fb);
            }
        }
    }
    canvas_hit.ok_or_else(|| "未在截图左上角匹配到小地图".into())
}

fn player_on_canvas(lx: f64, ly: f64, align: &Align, cw: u32, ch: u32) -> (f64, f64) {
    let (ox, oy) = align.loc;
    let (sx, sy) = align.scale;
    let (cx, cy) = if align.mode == "canvas_in_view" {
        ((lx - ox as f64) / sx, (ly - oy as f64) / sy)
    } else {
        (ox as f64 + lx * sx, oy as f64 + ly * sy)
    };
    (
        cx.clamp(0.0, (cw - 1) as f64),
        cy.clamp(0.0, (ch - 1) as f64),
    )
}

pub fn locate_from_fullscreen_shot(
    shot: &RgbImage,
    canvas: &RgbImage,
    full: &RgbImage,
    street: &str,
    name: &str,
    map_id: Option<u64>,
) -> Result<(LocateResult, MinimapHit), String> {
    let map_align = CanvasToFullAlign::build(canvas, full)?;
    locate_from_fullscreen_shot_with_align(shot, canvas, full, &map_align, street, name, map_id)
}

fn locate_from_fullscreen_shot_with_align(
    shot: &RgbImage,
    canvas: &RgbImage,
    full: &RgbImage,
    map_align: &CanvasToFullAlign,
    street: &str,
    name: &str,
    map_id: Option<u64>,
) -> Result<(LocateResult, MinimapHit), String> {
    let hit = find_minimap_region(shot, canvas)?;
    let region = crop_rgb(shot, hit.x, hit.y, hit.w, hit.h);
    let (lx, ly, ly_bot, _) = find_yellow(&region).ok_or("小地图区域内未找到玩家黄点")?;

    let (cx, cy, cy_bot) = if hit.mode == "canvas_on_screen" {
        (lx / hit.scale, ly / hit.scale, ly_bot / hit.scale)
    } else {
        let align = hit.align.as_ref().ok_or("视口回退缺少对齐结果")?;
        let (cx, cy) = player_on_canvas(lx, ly, align, canvas.width(), canvas.height());
        let (_, cy_bot) = player_on_canvas(lx, ly_bot, align, canvas.width(), canvas.height());
        (cx, cy, cy_bot)
    };

    let (cx, cy) = refine_player_on_canvas(canvas, cx, cy, cy_bot);
    let cx = cx.clamp(0.0, (canvas.width() - 1) as f64);
    let cy = cy.clamp(0.0, (canvas.height() - 1) as f64);
    let (fx, fy) = canvas_xy_to_full(map_align, canvas, full, cx, cy);

    let align = hit.align.clone().unwrap_or(Align {
        mode: hit.mode,
        score: hit.score,
        loc: (hit.x, hit.y),
        scale: (hit.scale, hit.scale),
        view_origin: (0.0, 0.0),
    });

    Ok((
        LocateResult {
            street: street.to_string(),
            name: name.to_string(),
            map_id,
            shot_x: hit.x as f64 + lx,
            shot_y: hit.y as f64 + ly,
            view_x: lx,
            view_y: ly,
            canvas_x: cx,
            canvas_y: cy,
            full_x: fx,
            full_y: fy,
            canvas_w: canvas.width(),
            canvas_h: canvas.height(),
            full_w: full.width(),
            full_h: full.height(),
            align,
        },
        hit,
    ))
}

fn load_rgb(path: &Path) -> Result<RgbImage, String> {
    image::open(path)
        .map_err(|e| format!("打开 {} 失败: {e}", path.display()))
        .map(|i| i.to_rgb8())
}

fn list_pngs(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut shots: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("读目录 {} 失败: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .collect();
    shots.sort();
    shots.dedup();
    Ok(shots)
}

pub fn resolve_map_assets(root: &Path, map_query: &str, map_id: u64) -> Result<(PathBuf, PathBuf), String> {
    let assets = root.join("assets").join("maps").join(map_id.to_string());
    let named = root.join("maps").join(safe_filename(map_query.trim()));
    let mini = [
        assets.join(format!("map_{map_id}_minimap.png")),
        named.join(format!("map_{map_id}_minimap.png")),
    ]
    .into_iter()
    .find(|p| p.is_file())
    .ok_or_else(|| {
        format!("找不到 minimap（assets/maps/{map_id}/map_{map_id}_minimap.png）")
    })?;
    let full = [
        assets.join(format!("map_{map_id}_render_cn.png")),
        assets.join(format!("map_{map_id}_render.png")),
        named.join(format!("map_{map_id}_render_cn.png")),
        named.join(format!("map_{map_id}_render.png")),
    ]
    .into_iter()
    .find(|p| p.is_file())
    .ok_or_else(|| format!("找不到完整图（assets/maps/{map_id}/map_{map_id}_render*.png）"))?;
    Ok((mini, full))
}

pub fn resolve_caps_dir(root: &Path, map_query: &str) -> PathBuf {
    let q = map_query.trim();
    let direct = root.join("screen_caps").join(q);
    if direct.is_dir() {
        return direct;
    }
    root.join("screen_caps").join(safe_filename(q))
}

pub fn validate_screen_caps_dir(
    caps_dir: &Path,
    minimap_path: &Path,
    full_path: &Path,
    out_dir: &Path,
    map_id: Option<u64>,
) -> Result<CapsValidateSummary, String> {
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let canvas = load_rgb(minimap_path)?;
    let full = load_rgb(full_path)?;
    let map_align = CanvasToFullAlign::build(&canvas, &full)?;
    let shots = list_pngs(caps_dir)?;
    if shots.is_empty() {
        return Err(format!("目录内无 PNG: {}", caps_dir.display()));
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "截图 {} 张 | minimap {}x{} | full {}x{} | backend=OpenCV | map_align={}",
        shots.len(),
        canvas.width(),
        canvas.height(),
        full.width(),
        full.height(),
        map_align.mode
    ));
    let mut ok = 0usize;
    for (i, path) in shots.iter().enumerate() {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("shot");
        let shot = match load_rgb(path) {
            Ok(s) => s,
            Err(e) => {
                lines.push(format!("[{i}] FAIL {name}: {e}"));
                continue;
            }
        };
        match locate_from_fullscreen_shot_with_align(
            &shot, &canvas, &full, &map_align, "", "", map_id,
        ) {
            Ok((loc, hit)) => {
                ok += 1;
                let mut marked = full.clone();
                mark_player_diamond(&mut marked, loc.full_x, loc.full_y);
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("shot");
                let tag = safe_filename(stem);
                marked
                    .save(out_dir.join(format!("{i:02}_full_{tag}.png")))
                    .map_err(|e| e.to_string())?;

                let tw = 340u32.min(shot.width());
                let th = 340u32.min(shot.height());
                let mut tl = crop_rgb(&shot, 0, 0, tw, th);
                draw_rect(&mut tl, hit.x, hit.y, hit.w, hit.h, Rgb([255, 0, 0]), 2);
                mark_player_diamond_small(&mut tl, loc.shot_x, loc.shot_y);
                tl.save(out_dir.join(format!("{i:02}_tl_{tag}.png")))
                    .map_err(|e| e.to_string())?;

                lines.push(format!(
                    "[{i}] OK {name} score={:.3} mode={} full=({:.0},{:.0})",
                    hit.score, hit.mode, loc.full_x, loc.full_y
                ));
            }
            Err(e) => lines.push(format!("[{i}] FAIL {name}: {e}")),
        }
    }
    lines.push(format!("完成 {ok}/{} → {}", shots.len(), out_dir.display()));
    Ok(CapsValidateSummary {
        total: shots.len(),
        ok,
        out_dir: out_dir.to_path_buf(),
        lines,
    })
}
