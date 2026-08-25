use image::{Rgb, RgbImage};

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

/// 淡蓝色大空心菱形，标出玩家位置（比十字更醒目）。
pub fn mark_player_diamond(img: &mut RgbImage, fx: f64, fy: f64) {
    let cx = fx.round() as i32;
    let cy = fy.round() as i32;
    let color = Rgb([80, 220, 255]);
    let w = img.width() as i32;
    let h = img.height() as i32;
    let half = ((w.min(h) as f64) * 0.028).round().clamp(22.0, 48.0) as i32;
    let thickness = 9i32;

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
