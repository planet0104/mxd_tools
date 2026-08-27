//! 在 YOLO「玩家」框下方 OCR 名牌，定位指定玩家坐标。

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};

use crate::image_util::{crop_rgb, mark_player_diamond};
use crate::ocr::{self, OcrRuntime};
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
const MATCH_STOP_SCORE: f32 = 0.92;

/// 归一化中心距离：0 为屏幕正中，1 为角落。
fn center_distance_norm(cx: f32, cy: f32, img_w: u32, img_h: u32) -> f32 {
    let hw = img_w as f32 * 0.5;
    let hh = img_h as f32 * 0.5;
    if hw == 0.0 || hh == 0.0 {
        return 1.0;
    }
    let dx = (cx - hw) / hw;
    let dy = (cy - hh) / hh;
    (dx * dx + dy * dy).sqrt().clamp(0.0, 1.0)
}

/// 综合匹配分：在 match_score 基础上，越靠近屏幕中心扣分越少。
fn player_score(match_score: f32, dist_norm: f32) -> f32 {
    const CENTER_PENALTY: f32 = 0.15;
    match_score - CENTER_PENALTY * dist_norm
}

/// 在 YOLO 检测到的玩家框下方搜索名牌 OCR，返回与 `target_name` 最匹配的一个。
pub fn find_named_player(
    img: &RgbImage,
    detections: &[Detection],
    target_name: &str,
    min_player_conf: f32,
) -> Result<Option<NamedPlayerHit>> {
    find_named_player_verbose(img, detections, target_name, min_player_conf, false).map(|(hit, _)| hit)
}

/// 使用指定 OCR 运行时（GPU batch 服务用）。
pub fn find_named_player_with_ocr(
    ocr: &mut OcrRuntime,
    img: &RgbImage,
    detections: &[Detection],
    target_name: &str,
    min_player_conf: f32,
) -> Result<Option<NamedPlayerHit>> {
    find_named_player_with_ocr_verbose(ocr, img, detections, target_name, min_player_conf, false)
        .map(|(hit, _)| hit)
}

pub fn find_named_player_with_ocr_verbose(
    ocr: &mut OcrRuntime,
    img: &RgbImage,
    detections: &[Detection],
    target_name: &str,
    min_player_conf: f32,
    verbose: bool,
) -> Result<(Option<NamedPlayerHit>, Vec<PlayerOcrAttempt>)> {
    let mut best: Option<(NamedPlayerHit, f32)> = None;
    let mut attempts = Vec::new();
    let img_w = img.width();
    let img_h = img.height();

    let mut players: Vec<&Detection> = detections
        .iter()
        .filter(|d| d.label == PLAYER_LABEL && d.conf >= min_player_conf)
        .collect();
    players.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap_or(std::cmp::Ordering::Equal));

    'players: for det in players {
        let (rx, ry, rw, rh) = name_search_region(det, img.width(), img.height());
        if rw < 8 || rh < 8 {
            continue;
        }
        let region = crop_rgb(img, rx, ry, rw, rh);
        let Some((text, match_score, (bx, by, bw, bh))) =
            ocr_and_match_region_det_runtime(ocr, &region, target_name)?
        else {
            if verbose {
                attempts.push(PlayerOcrAttempt {
                    player_conf: det.conf,
                    player_xyxy: (det.x1, det.y1, det.x2, det.y2),
                    roi: (rx, ry, rw, rh),
                    ocr_text: String::new(),
                    match_score: 0.0,
                });
            }
            continue;
        };
        if verbose {
            attempts.push(PlayerOcrAttempt {
                player_conf: det.conf,
                player_xyxy: (det.x1, det.y1, det.x2, det.y2),
                roi: (rx + bx, ry + by, bw, bh),
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
            roi: (rx + bx, ry + by, bw, bh),
        };
        let det_cx = (det.x1 + det.x2) * 0.5;
        let det_cy = (det.y1 + det.y2) * 0.5;
        let dist = center_distance_norm(det_cx, det_cy, img_w, img_h);
        let score = player_score(match_score, dist);
        if best
            .as_ref()
            .map(|(b, bdist)| {
                let b_score = player_score(b.match_score, *bdist);
                score > b_score
                    || (score == b_score
                        && (dist < *bdist || hit.match_score > b.match_score))
            })
            .unwrap_or(true)
        {
            best = Some((hit, dist));
        }
        if match_score >= MATCH_STOP_SCORE {
            break 'players;
        }
    }

    if best.is_none() {
        let (fallback, fb_attempts) =
            scan_name_plates_fallback_runtime(ocr, img, target_name)?;
        if verbose {
            attempts.extend(fb_attempts);
        }
        if let Some((hit, dist)) = fallback {
            best = Some((hit, dist));
        }
    }

    Ok((best.map(|(h, _)| h), attempts))
}

pub fn find_named_player_verbose(
    img: &RgbImage,
    detections: &[Detection],
    target_name: &str,
    min_player_conf: f32,
    verbose: bool,
) -> Result<(Option<NamedPlayerHit>, Vec<PlayerOcrAttempt>)> {
    let mut best: Option<(NamedPlayerHit, f32)> = None;
    let mut attempts = Vec::new();
    let img_w = img.width();
    let img_h = img.height();

    let mut players: Vec<&Detection> = detections
        .iter()
        .filter(|d| d.label == PLAYER_LABEL && d.conf >= min_player_conf)
        .collect();
    players.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap_or(std::cmp::Ordering::Equal));

    'players: for det in players {
        let (rx, ry, rw, rh) = name_search_region(det, img.width(), img.height());
        if rw < 8 || rh < 8 {
            continue;
        }
        let region = crop_rgb(img, rx, ry, rw, rh);
        let Some((text, match_score, (bx, by, bw, bh))) =
            ocr_and_match_region_det(&region, target_name).context("OCR 名牌区域失败")?
        else {
            if verbose {
                attempts.push(PlayerOcrAttempt {
                    player_conf: det.conf,
                    player_xyxy: (det.x1, det.y1, det.x2, det.y2),
                    roi: (rx, ry, rw, rh),
                    ocr_text: String::new(),
                    match_score: 0.0,
                });
            }
            continue;
        };
        if verbose {
            attempts.push(PlayerOcrAttempt {
                player_conf: det.conf,
                player_xyxy: (det.x1, det.y1, det.x2, det.y2),
                roi: (rx + bx, ry + by, bw, bh),
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
            roi: (rx + bx, ry + by, bw, bh),
        };
        let det_cx = (det.x1 + det.x2) * 0.5;
        let det_cy = (det.y1 + det.y2) * 0.5;
        let dist = center_distance_norm(det_cx, det_cy, img_w, img_h);
        let score = player_score(match_score, dist);
        if best
            .as_ref()
            .map(|(b, bdist)| {
                let b_score = player_score(b.match_score, *bdist);
                score > b_score
                    || (score == b_score
                        && (dist < *bdist || hit.match_score > b.match_score))
            })
            .unwrap_or(true)
        {
            best = Some((hit, dist));
        }
        if match_score >= MATCH_STOP_SCORE {
            break 'players;
        }
    }

    if best.is_none() {
        let (fallback, fb_attempts) =
            scan_name_plates_fallback(img, target_name).context("全图名牌扫描失败")?;
        if verbose {
            attempts.extend(fb_attempts);
        }
        if let Some((hit, dist)) = fallback {
            best = Some((hit, dist));
        }
    }

    Ok((best.map(|(h, _)| h), attempts))
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

/// 玩家框下方主搜索区（供 det 定位文本行）。
pub fn name_search_region(det: &Detection, img_w: u32, img_h: u32) -> (u32, u32, u32, u32) {
    let bw = (det.x2 - det.x1).max(1.0);
    let cx = (det.x1 + det.x2) * 0.5;

    // 名牌单行约 16px 高；ROI 收窄、贴近脚点，避免混入脚下草地
    let roi_w = (bw * 2.0).clamp(88.0, 168.0);
    let roi_h = 22.0_f32.clamp(18.0, 30.0);

    let y1 = det.y2 + 2.0;
    let mut x1 = (cx - roi_w * 0.5).max(0.0);
    let mut x2 = (cx + roi_w * 0.5).min(img_w as f32);
    if cx < img_w as f32 * 0.14 {
        x1 = 0.0;
        x2 = roi_w.min(img_w as f32);
    }
    if cx > img_w as f32 * 0.86 {
        x2 = img_w as f32;
        x1 = (x2 - roi_w).max(0.0);
    }
    let y2 = (y1 + roi_h).min(img_h as f32);
    let w = (x2 - x1).max(1.0) as u32;
    let h = (y2 - y1).max(1.0) as u32;
    if h >= 8 && w >= 8 {
        (x1 as u32, y1.max(0.0) as u32, w, h)
    } else {
        (0, 0, 1, 1)
    }
}

/// PP-OCRv5 det 定位文本行后做精简 rec 匹配。
fn ocr_and_match_region_det(
    region: &RgbImage,
    target_name: &str,
) -> Result<Option<(String, f32, (u32, u32, u32, u32))>> {
    let mut boxes = ocr::detect_text_boxes(region)?;
    if boxes.is_empty() {
        if let Some((text, score)) = ocr_and_match_roi_simple(region, target_name)? {
            if score > 0.0 {
                let (w, h) = region.dimensions();
                return Ok(Some((text, score, (0, 0, w, h))));
            }
        }
        return Ok(None);
    }

    boxes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut best: Option<(String, f32, (u32, u32, u32, u32))> = None;
    for tb in boxes {
        let crop = crop_rgb(region, tb.x, tb.y, tb.w, tb.h);
        let Some((text, score)) = ocr_and_match_roi_simple(&crop, target_name)? else {
            continue;
        };
        if score >= MATCH_STOP_SCORE {
            return Ok(Some((text, score, (tb.x, tb.y, tb.w, tb.h))));
        }
        if score > best.as_ref().map(|b| b.1).unwrap_or(0.0) {
            best = Some((text, score, (tb.x, tb.y, tb.w, tb.h)));
        }
    }
    Ok(best)
}

fn ocr_and_match_region_det_runtime(
    ocr: &mut OcrRuntime,
    region: &RgbImage,
    target_name: &str,
) -> Result<Option<(String, f32, (u32, u32, u32, u32))>> {
    let mut boxes = ocr.detect_text_boxes(region)?;
    if boxes.is_empty() {
        if let Some((text, score)) = ocr_and_match_roi_simple_runtime(ocr, region, target_name)? {
            if score > 0.0 {
                let (w, h) = region.dimensions();
                return Ok(Some((text, score, (0, 0, w, h))));
            }
        }
        return Ok(None);
    }
    boxes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut best: Option<(String, f32, (u32, u32, u32, u32))> = None;
    for tb in boxes {
        let crop = crop_rgb(region, tb.x, tb.y, tb.w, tb.h);
        let Some((text, score)) = ocr_and_match_roi_simple_runtime(ocr, &crop, target_name)? else {
            continue;
        };
        if score >= MATCH_STOP_SCORE {
            return Ok(Some((text, score, (tb.x, tb.y, tb.w, tb.h))));
        }
        if score > best.as_ref().map(|b| b.1).unwrap_or(0.0) {
            best = Some((text, score, (tb.x, tb.y, tb.w, tb.h)));
        }
    }
    Ok(best)
}

fn ocr_and_match_roi_simple_runtime(
    ocr: &mut OcrRuntime,
    roi: &RgbImage,
    target_name: &str,
) -> Result<Option<(String, f32)>> {
    let invert = invert_rgb(roi);
    let refs: [&RgbImage; 2] = [roi, &invert];
    let texts = ocr
        .recognize_rgb_batch(&refs)
        .unwrap_or_else(|_| vec![String::new(); 2]);
    let mut best_text = String::new();
    let mut best_match = 0.0f32;
    for text in texts {
        let normalized = normalize_name(&text);
        let m = name_similarity(&normalized, target_name);
        if m > best_match {
            best_match = m;
            best_text = normalized;
        }
    }
    if best_match > 0.0 {
        Ok(Some((best_text, best_match)))
    } else {
        Ok(None)
    }
}

fn scan_name_plates_fallback_runtime(
    ocr: &mut OcrRuntime,
    img: &RgbImage,
    target_name: &str,
) -> Result<(Option<(NamedPlayerHit, f32)>, Vec<PlayerOcrAttempt>)> {
    let (w, h) = img.dimensions();
    let mut attempts = Vec::new();
    let mut best: Option<(NamedPlayerHit, f32)> = None;

    let y_start = (h as f32 * 0.65) as u32;
    let y_end = h.saturating_sub(90);
    let x_start = w / 8;
    let x_end = w * 7 / 8;
    if y_end <= y_start + 20 || x_end <= x_start + 40 {
        return Ok((None, attempts));
    }

    let region = crop_rgb(img, x_start, y_start, x_end - x_start, y_end - y_start);
    if let Some((text, match_score, (bx, by, bw, bh))) =
        ocr_and_match_region_det_runtime(ocr, &region, target_name)?
    {
        attempts.push(PlayerOcrAttempt {
            player_conf: 0.0,
            player_xyxy: (0.0, 0.0, 0.0, 0.0),
            roi: (x_start + bx, y_start + by, bw, bh),
            ocr_text: text.clone(),
            match_score,
        });
        if match_score >= 0.45 {
            let px = x_start + bx;
            let py = y_start + by;
            let hit = NamedPlayerHit {
                x: px as f32 + bw as f32 * 0.5,
                y: py as f32 - 4.0,
                ocr_text: text,
                match_score,
                partial: is_partial_name(&attempts[0].ocr_text, target_name),
                player_conf: 0.0,
                roi: (px, py, bw, bh),
            };
            let plate_cx = px as f32 + bw as f32 * 0.5;
            let plate_cy = py as f32 + bh as f32 * 0.5;
            let dist = center_distance_norm(plate_cx, plate_cy, w, h);
            best = Some((hit, dist));
        }
    }

    Ok((best, attempts))
}

/// 对单个文本行 crop 做最多 2 次 rec（原图 + 反色）。
fn ocr_and_match_roi_simple(roi: &RgbImage, target_name: &str) -> Result<Option<(String, f32)>> {
    let invert = invert_rgb(roi);
    let refs: [&RgbImage; 2] = [roi, &invert];
    let texts = ocr::recognize_rgb_batch(&refs).unwrap_or_else(|_| vec![String::new(); 2]);
    let mut best_text = String::new();
    let mut best_match = 0.0f32;
    for text in texts {
        let normalized = normalize_name(&text);
        let m = name_similarity(&normalized, target_name);
        if m > best_match {
            best_match = m;
            best_text = normalized;
        }
    }
    if best_match > 0.0 {
        Ok(Some((best_text, best_match)))
    } else {
        Ok(None)
    }
}

/// YOLO 漏检时：对画面下部区域做 det + rec。
fn scan_name_plates_fallback(
    img: &RgbImage,
    target_name: &str,
) -> Result<(Option<(NamedPlayerHit, f32)>, Vec<PlayerOcrAttempt>)> {
    let (w, h) = img.dimensions();
    let mut attempts = Vec::new();
    let mut best: Option<(NamedPlayerHit, f32)> = None;

    let y_start = (h as f32 * 0.65) as u32;
    let y_end = h.saturating_sub(90);
    let x_start = w / 8;
    let x_end = w * 7 / 8;
    if y_end <= y_start + 20 || x_end <= x_start + 40 {
        return Ok((None, attempts));
    }

    let region = crop_rgb(img, x_start, y_start, x_end - x_start, y_end - y_start);
    if let Some((text, match_score, (bx, by, bw, bh))) =
        ocr_and_match_region_det(&region, target_name)?
    {
        attempts.push(PlayerOcrAttempt {
            player_conf: 0.0,
            player_xyxy: (0.0, 0.0, 0.0, 0.0),
            roi: (x_start + bx, y_start + by, bw, bh),
            ocr_text: text.clone(),
            match_score,
        });
        if match_score >= 0.45 {
            let px = x_start + bx;
            let py = y_start + by;
            let hit = NamedPlayerHit {
                x: px as f32 + bw as f32 * 0.5,
                y: py as f32 - 4.0,
                ocr_text: text,
                match_score,
                partial: is_partial_name(&attempts[0].ocr_text, target_name),
                player_conf: 0.0,
                roi: (px, py, bw, bh),
            };
            let plate_cx = px as f32 + bw as f32 * 0.5;
            let plate_cy = py as f32 + bh as f32 * 0.5;
            let dist = center_distance_norm(plate_cx, plate_cy, w, h);
            best = Some((hit, dist));
        }
    }

    Ok((best, attempts))
}

fn invert_rgb(img: &RgbImage) -> RgbImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        p.0 = [255 - p.0[0], 255 - p.0[1], 255 - p.0[2]];
    }
    out
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
        let s2 = name_similarity("光头强X加强版", "光头强加强版");
        assert!(s2 >= 0.78, "got {s2}");
    }

    #[test]
    fn full_name_match() {
        assert!(name_similarity("光头强加强版", "光头强加强版") >= 0.99);
    }

    #[test]
    fn training_npc_names_not_similar_to_self() {
        use crate::game::types::{DEFAULT_PLAYER_NAME, TRAINING_NPC_NAMES};
        for npc in TRAINING_NPC_NAMES {
            let s = name_similarity(npc, DEFAULT_PLAYER_NAME);
            assert!(
                s < 0.55,
                "装饰 NPC 名「{npc}」与主角相似度过高: {s:.2}"
            );
        }
    }

    #[test]
    #[ignore = "需要本地 screen_caps 与 OCR 模型"]
    fn fallback_finds_516_screenshot() {
        use super::find_named_player_verbose;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("screen_caps/彩虹岛-南港西郊平原/ScreenShot_2026-08-20_095154_516.png");
        if !path.is_file() {
            return;
        }
        let img = image::open(&path).unwrap().to_rgb8();
        let (hit, attempts) =
            find_named_player_verbose(&img, &[], "光头强加强版", 0.2, true).unwrap();
        eprintln!("attempts={} hit={hit:?}", attempts.len());
        assert!(hit.is_some(), "fallback should find target on 516");
    }
}
