//! 在 YOLO「玩家」框下方 OCR 名牌，定位指定玩家坐标。

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use opencv::core::{Mat, Size, Vec3b};
use opencv::imgproc::{self, InterpolationFlags};
use opencv::prelude::*;

use crate::image_util::{crop_rgb, mark_player_diamond};
use crate::ocr;
use crate::yolo::Detection;

/// 玩家脚点/中心坐标（像素，原图坐标系）。
#[derive(Debug, Clone)]
pub struct NamedPlayerHit {
    pub x: f32,
    pub y: f32,
    pub ocr_text: String,
    pub match_score: f32,
    /// OCR 未读全名（屏幕裁切或遮挡）时为 true。
    pub partial: bool,
    pub player_conf: f32,
    pub roi: (u32, u32, u32, u32),
}

const PLAYER_LABEL: &str = "玩家";

/// 在 YOLO 检测到的玩家框下方搜索名牌 OCR，返回与 `target_name` 最匹配的一个。
pub fn find_named_player(
    img: &RgbImage,
    detections: &[Detection],
    target_name: &str,
    min_player_conf: f32,
) -> Result<Option<NamedPlayerHit>> {
    find_named_player_verbose(img, detections, target_name, min_player_conf, false).map(|(hit, _)| hit)
}

pub fn find_named_player_verbose(
    img: &RgbImage,
    detections: &[Detection],
    target_name: &str,
    min_player_conf: f32,
    verbose: bool,
) -> Result<(Option<NamedPlayerHit>, Vec<PlayerOcrAttempt>)> {
    let mut best: Option<NamedPlayerHit> = None;
    let mut attempts = Vec::new();

    for det in detections {
        if det.label != PLAYER_LABEL || det.conf < min_player_conf {
            continue;
        }
        for (x, y, w, h) in name_search_rois(det, img.width(), img.height()) {
            if w < 8 || h < 8 {
                continue;
            }
            let roi = crop_rgb(img, x, y, w, h);
            if let Some((text, match_score)) =
                ocr_and_match_roi_variants(&roi, target_name).context("OCR 名牌区域失败")?
            {
                if verbose {
                    attempts.push(PlayerOcrAttempt {
                        player_conf: det.conf,
                        player_xyxy: (det.x1, det.y1, det.x2, det.y2),
                        roi: (x, y, w, h),
                        ocr_text: text.clone(),
                        match_score,
                    });
                }
                if match_score < 0.45 {
                    continue;
                }
                let hit = NamedPlayerHit {
                    x: (det.x1 + det.x2) * 0.5,
                    y: det.y2,
                    ocr_text: text.clone(),
                    match_score,
                    partial: is_partial_name(&text, target_name),
                    player_conf: det.conf,
                    roi: (x, y, w, h),
                };
                if best
                    .as_ref()
                    .map(|b| {
                        hit.match_score > b.match_score
                            || (hit.match_score == b.match_score && hit.player_conf > b.player_conf)
                    })
                    .unwrap_or(true)
                {
                    best = Some(hit);
                }
            } else if verbose {
                attempts.push(PlayerOcrAttempt {
                    player_conf: det.conf,
                    player_xyxy: (det.x1, det.y1, det.x2, det.y2),
                    roi: (x, y, w, h),
                    ocr_text: String::new(),
                    match_score: 0.0,
                });
            }
        }
    }

    if best.is_none() {
        let (fallback, fb_attempts) =
            scan_name_plates_fallback(img, target_name).context("全图名牌扫描失败")?;
        if verbose {
            attempts.extend(fb_attempts);
        }
        if fallback.is_some() {
            best = fallback;
        }
    }

    Ok((best, attempts))
}

#[derive(Debug, Clone)]
pub struct PlayerOcrAttempt {
    pub player_conf: f32,
    pub player_xyxy: (f32, f32, f32, f32),
    pub roi: (u32, u32, u32, u32),
    pub ocr_text: String,
    pub match_score: f32,
}

/// 在图像上标注找到的玩家位置（淡蓝菱形 + 名牌 ROI 框）。
pub fn draw_named_player_hit(img: &mut RgbImage, hit: &NamedPlayerHit) {
    mark_player_diamond(img, hit.x as f64, hit.y as f64);
    let (x, y, w, h) = hit.roi;
    crate::image_util::draw_rect(img, x, y, w, h, Rgb([255, 220, 0]), 2);
}

/// 名牌搜索区域：玩家框下方、水平居中扩展（多种纵向偏移）。
pub fn name_search_rois(det: &Detection, img_w: u32, img_h: u32) -> Vec<(u32, u32, u32, u32)> {
    let bw = (det.x2 - det.x1).max(1.0);
    let bh = (det.y2 - det.y1).max(1.0);
    let cx = (det.x1 + det.x2) * 0.5;

    let roi_w = (bw * 3.2).clamp(100.0, 360.0);
    let roi_h = (bh * 0.85).clamp(28.0, 80.0);

    let y_starts = [det.y2 - bh * 0.05, det.y2 + bh * 0.02, det.y2 + bh * 0.12];
    let mut out = Vec::new();
    for y1 in y_starts {
        let mut x1 = (cx - roi_w * 0.5).max(0.0);
        let mut x2 = (cx + roi_w * 0.5).min(img_w as f32);
        // 玩家靠近屏幕边缘时，名牌可能被裁切，ROI 扩展到画面边界
        if cx < img_w as f32 * 0.18 {
            x1 = 0.0;
        }
        if cx > img_w as f32 * 0.82 {
            x2 = img_w as f32;
        }
        let y2 = (y1 + roi_h).min(img_h as f32);
        let w = (x2 - x1).max(1.0) as u32;
        let h = (y2 - y1).max(1.0) as u32;
        if h >= 8 && w >= 8 {
            out.push((x1 as u32, y1.max(0.0) as u32, w, h));
        }
    }
    out
}

/// 兼容旧接口：取第一档 ROI。
pub fn name_search_roi(det: &Detection, img_w: u32, img_h: u32) -> (u32, u32, u32, u32) {
    name_search_rois(det, img_w, img_h)
        .into_iter()
        .next()
        .unwrap_or((0, 0, 1, 1))
}

/// 在 ROI 内收紧到高对比名牌横条（白底黑字 / 黑底白字）。
fn tighten_to_name_plate(roi: &RgbImage) -> RgbImage {
    let (w, h) = roi.dimensions();
    if w < 4 || h < 4 {
        return roi.clone();
    }

    let mut best_band: Option<(u32, u32, PlateMode)> = None;
    let mut best_score = 0.0f32;

    for mode in [PlateMode::Light, PlateMode::Dark] {
        for y in 0..h {
            let mut mode_px = 0u32;
            let mut edge = 0u32;
            let mut prev_l = luminance(roi.get_pixel(0, y).0);
            for x in 0..w {
                let l = luminance(roi.get_pixel(x, y).0);
                if mode.matches(l) {
                    mode_px += 1;
                }
                if x > 0 && (l as i32 - prev_l as i32).abs() > 28 {
                    edge += 1;
                }
                prev_l = l;
            }
            let fill = mode_px as f32 / w as f32;
            let edge_ratio = edge as f32 / w as f32;
            let score = match mode {
                PlateMode::Light => fill * 0.7 + edge_ratio * 0.3,
                PlateMode::Dark => (1.0 - fill) * 0.55 + edge_ratio * 0.45,
            };
            if score > best_score && fill > 0.12 && fill < 0.92 && edge_ratio > 0.035 {
                best_score = score;
                best_band = Some((y, y, mode));
            }
        }
    }

    let Some((mut y1, mut y2, mode)) = best_band else {
        return roi.clone();
    };

    let center_l = row_mean_luma(roi, (y1 + y2) / 2);
    while y1 > 0 && (row_mean_luma(roi, y1 - 1) - center_l).abs() < 45.0 {
        y1 -= 1;
    }
    while y2 + 1 < h && (row_mean_luma(roi, y2 + 1) - center_l).abs() < 45.0 {
        y2 += 1;
    }

    let pad_y = 2u32;
    y1 = y1.saturating_sub(pad_y);
    y2 = (y2 + pad_y).min(h - 1);
    let bh = y2 - y1 + 1;

    let (x1, x2) = horizontal_text_bounds(roi, y1, y2, mode);
    if x2 <= x1 || bh < 4 {
        return roi.clone();
    }
    crop_rgb(roi, x1, y1, x2 - x1 + 1, bh)
}

fn horizontal_text_bounds(roi: &RgbImage, y1: u32, y2: u32, mode: PlateMode) -> (u32, u32) {
    let w = roi.width();
    let is_text = |l: u8| match mode {
        PlateMode::Light => l < 120,
        PlateMode::Dark => l > 185,
    };

    let mut x1 = 0u32;
    let mut x2 = w.saturating_sub(1);
    'left: for xx in 0..w {
        for yy in y1..=y2 {
            if is_text(luminance(roi.get_pixel(xx, yy).0)) {
                x1 = xx.saturating_sub(2);
                break 'left;
            }
        }
    }
    'right: for xx in (0..w).rev() {
        for yy in y1..=y2 {
            if is_text(luminance(roi.get_pixel(xx, yy).0)) {
                x2 = (xx + 2).min(w - 1);
                break 'right;
            }
        }
    }
    (x1, x2)
}

fn roi_variants(roi: &RgbImage) -> Vec<RgbImage> {
    let mut out = vec![roi.clone()];
    let tight = tighten_to_name_plate(roi);
    if tight.width() >= 12
        && tight.height() >= 6
        && tight.width() * tight.height() < roi.width() * roi.height()
    {
        out.push(tight);
    }
    out
}

fn ocr_and_match_roi_variants(
    roi: &RgbImage,
    target_name: &str,
) -> Result<Option<(String, f32)>> {
    let mut best_text = String::new();
    let mut best_match = 0.0f32;
    for variant in roi_variants(roi) {
        let (text, _) = ocr_name_in_roi(&variant)?;
        let m = name_similarity(&text, target_name);
        if m > best_match {
            best_match = m;
            best_text = text;
        }
    }
    if best_match > 0.0 {
        Ok(Some((best_text, best_match)))
    } else {
        Ok(None)
    }
}

/// YOLO 漏检时：在游戏区域扫描名牌横条并 OCR。
fn scan_name_plates_fallback(
    img: &RgbImage,
    target_name: &str,
) -> Result<(Option<NamedPlayerHit>, Vec<PlayerOcrAttempt>)> {
    let (w, h) = img.dimensions();
    let y_top = 48u32;
    let y_bot = h.saturating_sub(130);
    let x_pad = (w as f32 * 0.04) as u32;
    let x1 = x_pad;
    let x2 = w.saturating_sub(x_pad);

    let mut attempts = Vec::new();
    let mut best: Option<NamedPlayerHit> = None;

    let mut y = y_top;
    while y + 10 < y_bot {
        if let Some((py1, py2, px1, px2)) = find_plate_at_row(img, y, x1, x2) {
            let pw = px2 - px1 + 1;
            let ph = py2 - py1 + 1;
            if pw >= 40 && pw <= 360 && ph >= 8 && ph <= 40 {
                let roi = crop_rgb(img, px1, py1, pw, ph);
                if let Some((text, match_score)) = ocr_and_match_roi_variants(&roi, target_name)? {
                    attempts.push(PlayerOcrAttempt {
                        player_conf: 0.0,
                        player_xyxy: (0.0, 0.0, 0.0, 0.0),
                        roi: (px1, py1, pw, ph),
                        ocr_text: text.clone(),
                        match_score,
                    });
                    if match_score >= 0.45 {
                        let hit = NamedPlayerHit {
                            x: px1 as f32 + pw as f32 * 0.5,
                            y: py1 as f32 - 4.0,
                            ocr_text: text.clone(),
                            match_score,
                            partial: is_partial_name(&text, target_name),
                            player_conf: 0.0,
                            roi: (px1, py1, pw, ph),
                        };
                        if best
                            .as_ref()
                            .map(|b| hit.match_score > b.match_score)
                            .unwrap_or(true)
                        {
                            best = Some(hit);
                        }
                    }
                }
            }
            y = py2 + 2;
        } else {
            y += 3;
        }
    }

    Ok((best, attempts))
}

fn find_plate_at_row(
    img: &RgbImage,
    y: u32,
    x1: u32,
    x2: u32,
) -> Option<(u32, u32, u32, u32)> {
    let score = row_plate_score(img, y, x1, x2);
    if score < 0.12 {
        return None;
    }

    let mut y1 = y;
    let mut y2 = y;
    while y1 > 0 && row_plate_score(img, y1 - 1, x1, x2) > 0.08 {
        y1 -= 1;
    }
    while y2 + 1 < img.height() && row_plate_score(img, y2 + 1, x1, x2) > 0.08 {
        y2 += 1;
    }

    let (mut px1, mut px2) = (x1, x2);
    let mid = (y1 + y2) / 2;
    'left: for xx in x1..=x2 {
        let l = luminance(img.get_pixel(xx, mid).0);
        if (l > 185) || (l < 120) {
            px1 = xx.saturating_sub(4);
            break 'left;
        }
    }
    'right: for xx in (x1..=x2).rev() {
        let l = luminance(img.get_pixel(xx, mid).0);
        if (l > 185) || (l < 120) {
            px2 = (xx + 4).min(x2);
            break 'right;
        }
    }

    if px2 <= px1 || y2 <= y1 {
        return None;
    }
    Some((y1, y2, px1, px2))
}

fn row_plate_score(img: &RgbImage, y: u32, x1: u32, x2: u32) -> f32 {
    if y >= img.height() || x2 <= x1 {
        return 0.0;
    }
    let mut dark = 0u32;
    let mut bright = 0u32;
    let mut edge = 0u32;
    let mut prev = luminance(img.get_pixel(x1, y).0);
    for x in x1..=x2 {
        let l = luminance(img.get_pixel(x, y).0);
        if (35..=130).contains(&l) {
            dark += 1;
        }
        if l >= 200 {
            bright += 1;
        }
        if (l as i32 - prev as i32).abs() > 35 {
            edge += 1;
        }
        prev = l;
    }
    let w = (x2 - x1 + 1) as f32;
    let dark_r = dark as f32 / w;
    let bright_r = bright as f32 / w;
    let edge_r = edge as f32 / w;

    let dark_plate = dark_r * 0.45 + bright_r * 1.2 + edge_r * 0.8;
    let light_plate = (bright_r + dark_r * 0.3) * 0.5 + edge_r;
    dark_plate.max(light_plate * 0.85)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlateMode {
    Light,
    Dark,
}

impl PlateMode {
    fn matches(self, l: u8) -> bool {
        match self {
            PlateMode::Light => l > 185,
            PlateMode::Dark => l < 95,
        }
    }
}

fn luminance(rgb: [u8; 3]) -> u8 {
    ((u16::from(rgb[0]) * 30 + u16::from(rgb[1]) * 59 + u16::from(rgb[2]) * 11) / 100) as u8
}

fn row_mean_luma(img: &RgbImage, y: u32) -> f32 {
    let w = img.width();
    let sum: u32 = (0..w)
        .map(|x| u32::from(luminance(img.get_pixel(x, y).0)))
        .sum();
    sum as f32 / w as f32
}

fn ocr_name_in_roi(roi: &RgbImage) -> Result<(String, f32)> {
    let mut best_text = String::new();
    let mut best_score = 0.0f32;

    for variant in preprocess_variants(roi)? {
        let text = ocr::recognize_rgb(&variant).unwrap_or_default();
        let score = ocr_quality_score(&text);
        if score > best_score {
            best_score = score;
            best_text = normalize_name(&text);
        }
    }

    Ok((best_text, best_score))
}

fn preprocess_variants(roi: &RgbImage) -> Result<Vec<RgbImage>> {
    let up5 = upscale_rgb(roi, 5.0)?;
    let stretched = contrast_stretch(&up5);
    let de_rope = remove_vertical_ropes(&up5);
    Ok(vec![
        up5.clone(),
        invert_rgb(&up5),
        stretched.clone(),
        invert_rgb(&stretched),
        de_rope.clone(),
        invert_rgb(&de_rope),
    ])
}

/// 去除名牌 ROI 内竖直绳子/遮挡线（强-加 之间常见）。
fn remove_vertical_ropes(img: &RgbImage) -> RgbImage {
    let (w, h) = img.dimensions();
    if w < 6 || h < 6 {
        return img.clone();
    }
    let mut col_score = vec![0u32; w as usize];
    for x in 1..w - 1 {
        let mut score = 0u32;
        for y in 1..h - 1 {
            let _c = luminance(img.get_pixel(x, y).0);
            let l = luminance(img.get_pixel(x - 1, y).0);
            let r = luminance(img.get_pixel(x + 1, y).0);
            let u = luminance(img.get_pixel(x, y - 1).0);
            let d = luminance(img.get_pixel(x, y + 1).0);
            let vx = (l as i32 - r as i32).abs();
            let vy = (u as i32 - d as i32).abs();
            if vx > vy + 18 && vx > 35 {
                score += 1;
            }
        }
        col_score[x as usize] = score;
    }
    let thresh = (h as f32 * 0.45) as u32;
    let mut out = img.clone();
    for x in 1..w - 1 {
        if col_score[x as usize] < thresh {
            continue;
        }
        for y in 0..h {
            let l = img.get_pixel(x.saturating_sub(1), y).0;
            let r = img.get_pixel((x + 1).min(w - 1), y).0;
            out.put_pixel(
                x,
                y,
                Rgb([
                    ((u16::from(l[0]) + u16::from(r[0])) / 2) as u8,
                    ((u16::from(l[1]) + u16::from(r[1])) / 2) as u8,
                    ((u16::from(l[2]) + u16::from(r[2])) / 2) as u8,
                ]),
            );
        }
    }
    out
}

fn upscale_rgb(img: &RgbImage, scale: f32) -> Result<RgbImage> {
    let (w, h) = img.dimensions();
    let nw = ((w as f32) * scale).round().max(1.0) as i32;
    let nh = ((h as f32) * scale).round().max(1.0) as i32;
    let src = Mat::new_rows_cols_with_bytes::<Vec3b>(h as i32, w as i32, img.as_raw())
        .map_err(|e| anyhow::anyhow!("mat from rgb: {e}"))?;
    let mut dst = Mat::default();
    imgproc::resize(
        &src,
        &mut dst,
        Size::new(nw, nh),
        0.0,
        0.0,
        InterpolationFlags::INTER_CUBIC.into(),
    )
    .map_err(|e| anyhow::anyhow!("resize: {e}"))?;
    let bytes = dst
        .data_bytes()
        .map_err(|e| anyhow::anyhow!("data_bytes: {e}"))?;
    let mut out = RgbImage::new(nw as u32, nh as u32);
    for y in 0..nh as u32 {
        for x in 0..nw as u32 {
            let i = ((y * nw as u32 + x) * 3) as usize;
            out.put_pixel(x, y, Rgb([bytes[i + 2], bytes[i + 1], bytes[i]]));
        }
    }
    Ok(out)
}

fn invert_rgb(img: &RgbImage) -> RgbImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        p.0 = [255 - p.0[0], 255 - p.0[1], 255 - p.0[2]];
    }
    out
}

fn contrast_stretch(img: &RgbImage) -> RgbImage {
    let mut min_v = 255u8;
    let mut max_v = 0u8;
    for p in img.pixels() {
        for c in p.0 {
            min_v = min_v.min(c);
            max_v = max_v.max(c);
        }
    }
    if max_v <= min_v + 8 {
        return img.clone();
    }
    let mut out = img.clone();
    for p in out.pixels_mut() {
        for c in p.0.iter_mut() {
            *c = ((*c as f32 - min_v as f32) / (max_v - min_v) as f32 * 255.0).round() as u8;
        }
    }
    out
}

fn mean_luminance(img: &RgbImage) -> f32 {
    if img.pixels().len() == 0 {
        return 128.0;
    }
    let sum: u64 = img
        .pixels()
        .map(|p| {
            let [r, g, b] = p.0;
            u64::from(r) * 30 + u64::from(g) * 59 + u64::from(b) * 11
        })
        .sum();
    (sum / (img.pixels().len() as u64 * 100)) as f32
}

fn normalize_name(s: &str) -> String {
    s.chars()
        .filter(|c| {
            *c == '_'
                || c.is_ascii_alphanumeric()
                || ('\u{4e00}'..='\u{9fff}').contains(c)
        })
        .collect()
}

fn ocr_quality_score(text: &str) -> f32 {
    let n = normalize_name(text).chars().count();
    if n == 0 {
        0.0
    } else {
        (n as f32).min(12.0) / 12.0
    }
}

/// 目标名与 OCR 结果的相似度 [0, 1]。
/// 支持：完整匹配、屏幕边缘截断（前缀/后缀）、绳子遮挡（有序子序列）。
pub fn name_similarity(ocr_text: &str, target: &str) -> f32 {
    let a = normalize_name(ocr_text);
    let b = normalize_name(target);
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a == b || a.contains(&b) || b.contains(&a) {
        return 1.0;
    }

    let a_len = a.chars().count();
    let b_len = b.chars().count();

    // 屏幕边缘截断：OCR 只读到前缀（如「光头强加」→「光头强加强版」）
    if b_len >= 4 && a_len >= 4 && b.starts_with(&a) {
        return 0.80 + 0.20 * (a_len as f32 / b_len as f32);
    }
    if a_len >= 4 && b_len >= 4 && a.starts_with(&b) {
        return 0.95;
    }

    // 绳子等于字符间插入噪声：按顺序子序列匹配
    let subseq = ordered_subsequence_score(&a, &b);
    if subseq >= 0.72 {
        return subseq;
    }

    // 至少 3 个连续字符命中（应对 OCR 噪声前缀后缀）
    if b_len >= 3 {
        let bc: Vec<char> = b.chars().collect();
        for win in (3..=bc.len()).rev() {
            for start in 0..=bc.len() - win {
                let sub: String = bc[start..start + win].iter().collect();
                if a.contains(&sub) {
                    return 0.72 + 0.28 * (win as f32 / b_len as f32);
                }
            }
        }
    }

    char_lcs_ratio(&a, &b)
}

/// `target` 字符以顺序出现在 `ocr` 中（允许中间被绳子/OCR 噪声隔开）。
fn ordered_subsequence_score(ocr: &str, target: &str) -> f32 {
    let oc: Vec<char> = ocr.chars().collect();
    let tc: Vec<char> = target.chars().collect();
    if oc.is_empty() || tc.is_empty() {
        return 0.0;
    }
    let mut ti = 0usize;
    for &c in &oc {
        if ti < tc.len() && c == tc[ti] {
            ti += 1;
        }
    }
    if ti < 4 {
        return 0.0;
    }
    let ratio = ti as f32 / tc.len() as f32;
    if ratio >= 0.99 {
        1.0
    } else if ratio >= 0.70 {
        0.78 + 0.22 * ratio
    } else {
        0.0
    }
}

fn is_partial_name(ocr_text: &str, target: &str) -> bool {
    let a = normalize_name(ocr_text);
    let b = normalize_name(target);
    if a.is_empty() || b.is_empty() {
        return false;
    }
    a != b && a.chars().count() < b.chars().count()
}

fn char_lcs_ratio(a: &str, b: &str) -> f32 {
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    if ac.is_empty() || bc.is_empty() {
        return 0.0;
    }
    let mut dp = vec![0u32; bc.len() + 1];
    for &ca in &ac {
        let mut prev = 0u32;
        for (j, &cb) in bc.iter().enumerate() {
            let tmp = dp[j + 1];
            dp[j + 1] = if ca == cb {
                prev + 1
            } else {
                dp[j + 1].max(dp[j])
            };
            prev = tmp;
        }
    }
    let lcs = *dp.last().unwrap_or(&0) as f32;
    lcs / a.len().max(b.len()) as f32
}

#[cfg(test)]
mod tests {
    use super::name_similarity;

    #[test]
    fn prefix_truncated_at_screen_edge() {
        let s = name_similarity("光头强加", "光头强加强版");
        assert!(s >= 0.85, "got {s}");
    }

    #[test]
    fn rope_splits_chars_subsequence() {
        let s = name_similarity("光头强加强版", "光头强加强版");
        assert!(s >= 0.99);
        // OCR 中间插入噪声字符
        let s2 = name_similarity("光头强X加强版", "光头强加强版");
        assert!(s2 >= 0.78, "got {s2}");
    }

    #[test]
    fn full_name_match() {
        assert!(name_similarity("光头强加强版", "光头强加强版") >= 0.99);
    }
}
