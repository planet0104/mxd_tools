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

pub fn mask_yellow_in_place(img: &mut RgbImage) {
    for p in img.pixels_mut() {
        let [r, g, b] = p.0;
        if r >= 240 && g >= 240 && b <= 160 {
            *p = Rgb([0, 0, 0]);
        }
    }
}

pub fn find_player_yellow(view: &RgbImage) -> Option<(f64, f64)> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (x, y, p) in view.enumerate_pixels() {
        let [r, g, b] = p.0;
        if r >= 250 && g >= 250 && (b as i16 - 136).abs() <= 20 {
            xs.push(x as f64);
            ys.push(y as f64);
        }
    }
    if xs.len() < 2 {
        xs.clear();
        ys.clear();
        for (x, y, p) in view.enumerate_pixels() {
            let [r, g, b] = p.0;
            if r >= 240 && g >= 240 && b <= 160 && b >= 70 {
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
    let x = fx.round() as i32;
    let y = fy.round() as i32;
    let color = Rgb([255, 32, 32]);
    let w = img.width() as i32;
    let h = img.height() as i32;
    for r in 14..=18 {
        for a in 0..360 {
            let rad = (a as f64).to_radians();
            let px = x + (rad.cos() * f64::from(r)).round() as i32;
            let py = y + (rad.sin() * f64::from(r)).round() as i32;
            if px >= 0 && py >= 0 && px < w && py < h {
                img.put_pixel(px as u32, py as u32, color);
            }
        }
    }
    for d in -28..=28 {
        let px = x + d;
        if px >= 0 && y >= 0 && px < w && y < h {
            img.put_pixel(px as u32, y as u32, color);
        }
        let py = y + d;
        if x >= 0 && py >= 0 && x < w && py < h {
            img.put_pixel(x as u32, py as u32, color);
        }
    }
}
