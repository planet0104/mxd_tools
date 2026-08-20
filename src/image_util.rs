use image::{Rgb, RgbImage};

pub fn to_gray(img: &RgbImage) -> Vec<u8> {
    img.pixels()
        .map(|p| {
            let [r, g, b] = p.0;
            ((u16::from(r) * 30 + u16::from(g) * 59 + u16::from(b) * 11) / 100) as u8
        })
        .collect()
}

pub fn resize_gray(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let img = image::GrayImage::from_raw(sw, sh, src.to_vec()).expect("gray size");
    image::imageops::resize(&img, dw, dh, image::imageops::FilterType::Triangle).into_raw()
}

pub fn crop_gray(src: &[u8], sw: u32, x: u32, y: u32, w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h) as usize];
    for row in 0..h {
        let s = ((y + row) * sw + x) as usize;
        let d = (row * w) as usize;
        out[d..d + w as usize].copy_from_slice(&src[s..s + w as usize]);
    }
    out
}

/// OpenCV TM_CCOEFF_NORMED style match. Returns (score, x, y).
pub fn match_template_ccoeff_normed(
    src: &[u8],
    sw: u32,
    sh: u32,
    tpl: &[u8],
    tw: u32,
    th: u32,
) -> Option<(f64, u32, u32)> {
    if tw > sw || th > sh || tw * th < 400 {
        return None;
    }
    let tpl_len = (tw * th) as f64;
    let mut tpl_sum = 0.0;
    let mut tpl_sq = 0.0;
    for &v in tpl {
        let f = f64::from(v);
        tpl_sum += f;
        tpl_sq += f * f;
    }
    let tpl_mean = tpl_sum / tpl_len;
    let tpl_var = tpl_sq - tpl_sum * tpl_mean;
    if tpl_var <= 1e-6 {
        return None;
    }
    let tpl_std = tpl_var.sqrt();

    let mut best_score = f64::NEG_INFINITY;
    let mut best_xy = (0u32, 0u32);

    for y in 0..=(sh - th) {
        for x in 0..=(sw - tw) {
            let mut sum = 0.0;
            let mut sq = 0.0;
            let mut cross = 0.0;
            for j in 0..th {
                for i in 0..tw {
                    let s = f64::from(src[((y + j) * sw + (x + i)) as usize]);
                    let t = f64::from(tpl[(j * tw + i) as usize]);
                    sum += s;
                    sq += s * s;
                    cross += s * (t - tpl_mean);
                }
            }
            let src_var = sq - sum * sum / tpl_len;
            if src_var <= 1e-6 {
                continue;
            }
            let score = cross / (src_var.sqrt() * tpl_std);
            if score > best_score {
                best_score = score;
                best_xy = (x, y);
            }
        }
    }
    if best_score.is_finite() {
        Some((best_score, best_xy.0, best_xy.1))
    } else {
        None
    }
}

/// OpenCV TM_CCOEFF_NORMED；`mask` 非 0 的像素参与统计（对齐 cv2 mask）。
/// `stride>1` 时先粗搜，再在最优点邻域精修（纯 Rust 无 SIMD 时必需）。
pub fn match_template_ccoeff_normed_masked(
    src: &[u8],
    sw: u32,
    sh: u32,
    tpl: &[u8],
    mask: &[u8],
    tw: u32,
    th: u32,
) -> Option<(f64, u32, u32)> {
    match_template_ccoeff_normed_masked_stride(src, sw, sh, tpl, mask, tw, th, 1)
}

pub fn match_template_ccoeff_normed_masked_stride(
    src: &[u8],
    sw: u32,
    sh: u32,
    tpl: &[u8],
    mask: &[u8],
    tw: u32,
    th: u32,
    stride: u32,
) -> Option<(f64, u32, u32)> {
    if tw > sw || th > sh || tpl.len() != mask.len() || tpl.len() != (tw * th) as usize {
        return None;
    }
    let mut idxs = Vec::new();
    let mut tpl_sum = 0.0;
    let mut tpl_sq = 0.0;
    for (i, (&t, &m)) in tpl.iter().zip(mask.iter()).enumerate() {
        if m == 0 {
            continue;
        }
        let f = f64::from(t);
        idxs.push(i);
        tpl_sum += f;
        tpl_sq += f * f;
    }
    let n = idxs.len() as f64;
    if n < 80.0 {
        return None;
    }
    let tpl_mean = tpl_sum / n;
    let tpl_var = tpl_sq - tpl_sum * tpl_mean;
    if tpl_var <= 1e-6 {
        return None;
    }
    let tpl_std = tpl_var.sqrt();

    let score_at = |x: u32, y: u32| -> f64 {
        let mut sum = 0.0;
        let mut sq = 0.0;
        let mut cross = 0.0;
        for &i in &idxs {
            let j = (i as u32) / tw;
            let iix = (i as u32) % tw;
            let s = f64::from(src[((y + j) * sw + (x + iix)) as usize]);
            let t = f64::from(tpl[i]);
            sum += s;
            sq += s * s;
            cross += s * (t - tpl_mean);
        }
        let src_var = sq - sum * sum / n;
        if src_var <= 1e-6 {
            return f64::NEG_INFINITY;
        }
        cross / (src_var.sqrt() * tpl_std)
    };

    let stride = stride.max(1);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_xy = (0u32, 0u32);
    let mut y = 0u32;
    while y <= sh - th {
        let mut x = 0u32;
        while x <= sw - tw {
            let score = score_at(x, y);
            if score > best_score {
                best_score = score;
                best_xy = (x, y);
            }
            x += stride;
        }
        y += stride;
    }

    if stride > 1 && best_score.is_finite() {
        let (bx, by) = best_xy;
        let r = stride as i32;
        let x0 = bx as i32 - r;
        let y0 = by as i32 - r;
        let x1 = bx as i32 + r;
        let y1 = by as i32 + r;
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                if xx < 0 || yy < 0 {
                    continue;
                }
                let xu = xx as u32;
                let yu = yy as u32;
                if xu + tw > sw || yu + th > sh {
                    continue;
                }
                let score = score_at(xu, yu);
                if score > best_score {
                    best_score = score;
                    best_xy = (xu, yu);
                }
            }
        }
    }

    if best_score.is_finite() {
        Some((best_score, best_xy.0, best_xy.1))
    } else {
        None
    }
}

pub fn mask_yellow_in_place(img: &mut RgbImage) {
    for p in img.pixels_mut() {
        let [r, g, b] = p.0;
        if r >= 240 && g >= 240 && b <= 160 {
            *p = Rgb([0, 0, 0]);
        }
    }
}

pub fn find_player_yellow(view: &RgbImage) -> Option<(f64, f64)> {
    // 与 scripts/locate_from_screencaps.py 一致：先紧阈值 #FFFF88，再放宽
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (x, y, p) in view.enumerate_pixels() {
        let [r, g, b] = p.0;
        if r >= 248 && g >= 248 && b >= 100 && b <= 155 {
            xs.push(x as f64);
            ys.push(y as f64);
        }
    }
    if xs.len() < 2 {
        xs.clear();
        ys.clear();
        for (x, y, p) in view.enumerate_pixels() {
            let [r, g, b] = p.0;
            if r >= 240 && g >= 240 && b <= 165 && b >= 70 {
                xs.push(x as f64);
                ys.push(y as f64);
            }
        }
    }
    if xs.is_empty() {
        return None;
    }
    Some((
        xs.iter().sum::<f64>() / xs.len() as f64,
        ys.iter().sum::<f64>() / ys.len() as f64,
    ))
}

pub fn crop_rgb(img: &RgbImage, x: u32, y: u32, w: u32, h: u32) -> RgbImage {
    let mut out = RgbImage::new(w, h);
    for yy in 0..h {
        for xx in 0..w {
            let sx = x + xx;
            let sy = y + yy;
            if sx < img.width() && sy < img.height() {
                out.put_pixel(xx, yy, *img.get_pixel(sx, sy));
            }
        }
    }
    out
}

pub fn mark_cross(img: &mut RgbImage, fx: f64, fy: f64) {
    mark_player_diamond(img, fx, fy);
}

/// 淡蓝色大空心菱形，标出玩家位置（比十字更醒目）。
pub fn mark_player_diamond(img: &mut RgbImage, fx: f64, fy: f64) {
    let cx = fx.round() as i32;
    let cy = fy.round() as i32;
    let color = Rgb([80, 220, 255]); // 淡青蓝
    let w = img.width() as i32;
    let h = img.height() as i32;
    // 半对角线长度；线宽约 3px
    let half = ((w.min(h) as f64) * 0.028).round().clamp(22.0, 48.0) as i32;
    let thickness = 9i32;

    let put = |img: &mut RgbImage, x: i32, y: i32| {
        if x >= 0 && y >= 0 && x < w && y < h {
            img.put_pixel(x as u32, y as u32, color);
        }
    };

    // 菱形四边：上→右→下→左
    let verts = [
        (cx, cy - half),
        (cx + half, cy),
        (cx, cy + half),
        (cx - half, cy),
    ];
    for i in 0..4 {
        let (x0, y0) = verts[i];
        let (x1, y1) = verts[(i + 1) % 4];
        let steps = ((x1 - x0).abs().max((y1 - y0).abs())).max(1);
        for t in 0..=steps {
            let px = x0 + (x1 - x0) * t / steps;
            let py = y0 + (y1 - y0) * t / steps;
            for ox in -thickness..=thickness {
                for oy in -thickness..=thickness {
                    if ox * ox + oy * oy <= thickness * thickness {
                        put(img, px + ox, py + oy);
                    }
                }
            }
        }
    }
}

pub fn draw_rect(img: &mut RgbImage, x: u32, y: u32, w: u32, h: u32, color: Rgb<u8>, thickness: i32) {
    let iw = img.width() as i32;
    let ih = img.height() as i32;
    let x0 = x as i32;
    let y0 = y as i32;
    let x1 = (x + w.saturating_sub(1)) as i32;
    let y1 = (y + h.saturating_sub(1)) as i32;
    let t = thickness.max(1);
    let put = |img: &mut RgbImage, px: i32, py: i32| {
        if px >= 0 && py >= 0 && px < iw && py < ih {
            img.put_pixel(px as u32, py as u32, color);
        }
    };
    for d in 0..t {
        for px in x0..=x1 {
            put(img, px, y0 + d);
            put(img, px, y1 - d);
        }
        for py in y0..=y1 {
            put(img, x0 + d, py);
            put(img, x1 - d, py);
        }
    }
}

/// 小地图预览用较小空心菱形。
pub fn mark_player_diamond_small(img: &mut RgbImage, fx: f64, fy: f64) {
    let cx = fx.round() as i32;
    let cy = fy.round() as i32;
    let color = Rgb([80, 220, 255]);
    let w = img.width() as i32;
    let h = img.height() as i32;
    let half = 14i32;
    let thickness = 2i32;
    let put = |img: &mut RgbImage, x: i32, y: i32| {
        if x >= 0 && y >= 0 && x < w && y < h {
            img.put_pixel(x as u32, y as u32, color);
        }
    };
    let verts = [
        (cx, cy - half),
        (cx + half, cy),
        (cx, cy + half),
        (cx - half, cy),
    ];
    for i in 0..4 {
        let (x0, y0) = verts[i];
        let (x1, y1) = verts[(i + 1) % 4];
        let steps = ((x1 - x0).abs().max((y1 - y0).abs())).max(1);
        for t in 0..=steps {
            let px = x0 + (x1 - x0) * t / steps;
            let py = y0 + (y1 - y0) * t / steps;
            for ox in -thickness..=thickness {
                for oy in -thickness..=thickness {
                    if ox * ox + oy * oy <= thickness * thickness {
                        put(img, px + ox, py + oy);
                    }
                }
            }
        }
    }
}
