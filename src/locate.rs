use image::RgbImage;

use crate::image_util::{
    crop_rgb, find_player_yellow, mark_cross, mask_yellow_in_place, match_template_ccoeff_normed,
    resize_gray, to_gray,
};
use crate::map_api::{fetch_canvas, fetch_full_map, resolve_map_id, save_map};
use crate::ocr::read_map_names;
use crate::paths::{maps_dir, safe_filename};
use std::path::{Path, PathBuf};

pub const VIEW_X: u32 = 6;
pub const VIEW_Y: u32 = 72;
pub const VIEW_W: u32 = 210;
pub const VIEW_H: u32 = 134;

#[derive(Debug, Clone)]
pub struct Align {
    pub mode: &'static str,
    pub score: f64,
    pub loc: (u32, u32),
    pub scale: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct LocateResult {
    pub street: String,
    pub name: String,
    pub map_id: Option<u64>,
    /// 玩家在 222×222 截图中的中心坐标
    pub shot_x: f64,
    pub shot_y: f64,
    /// 相对小地图视口 (VIEW_*) 的坐标
    pub view_x: f64,
    pub view_y: f64,
    pub canvas_x: f64,
    pub canvas_y: f64,
    pub full_x: f64,
    pub full_y: f64,
    pub canvas_w: u32,
    pub canvas_h: u32,
    pub full_w: u32,
    pub full_h: u32,
    pub align: Align,
}

pub fn view_to_canvas(view: &RgbImage, canvas: &RgbImage) -> Result<Align, String> {
    let mut view_m = view.clone();
    mask_yellow_in_place(&mut view_m);
    let gray_v = to_gray(&view_m);
    let gray_c = to_gray(canvas);
    let vw = view_m.width();
    let vh = view_m.height();
    let cw = canvas.width();
    let ch = canvas.height();

    let mut best: Option<Align> = None;

    let th = vh.min(ch);
    let tw = vw.min(cw);
    let tpl: Vec<u8> = (0..th)
        .flat_map(|y| {
            let start = (y * vw) as usize;
            gray_v[start..start + tw as usize].to_vec()
        })
        .collect();
    if let Some((score, x, y)) = match_template_ccoeff_normed(&gray_c, cw, ch, &tpl, tw, th) {
        best = Some(Align {
            mode: "partial",
            score,
            loc: (x, y),
            scale: (1.0, 1.0),
        });
    }

    if cw <= vw && ch <= vh {
        if let Some((score, x, y)) =
            match_template_ccoeff_normed(&gray_v, vw, vh, &gray_c, cw, ch)
        {
            if best.as_ref().map(|b| score > b.score).unwrap_or(true) {
                best = Some(Align {
                    mode: "full",
                    score,
                    loc: (x, y),
                    scale: (1.0, 1.0),
                });
            }
        }
    }

    let scale = (vw as f64 / cw as f64).min(vh as f64 / ch as f64);
    let nw = ((cw as f64 * scale).round() as u32).max(8);
    let nh = ((ch as f64 * scale).round() as u32).max(8);
    let small = resize_gray(&gray_c, cw, ch, nw, nh);
    if let Some((score, x, y)) = match_template_ccoeff_normed(&gray_v, vw, vh, &small, nw, nh) {
        if best.as_ref().map(|b| score > b.score).unwrap_or(true) {
            best = Some(Align {
                mode: "full",
                score,
                loc: (x, y),
                scale: (scale, scale),
            });
        }
    }

    best.ok_or_else(|| "小地图与完整略缩图无法对齐".to_string())
}

pub fn player_on_canvas(px: f64, py: f64, align: &Align, cw: u32, ch: u32) -> (f64, f64) {
    let (ox, oy) = align.loc;
    let (sx, sy) = align.scale;
    let (cx, cy) = if align.mode == "partial" {
        (ox as f64 + px, oy as f64 + py)
    } else {
        ((px - ox as f64) / sx, (py - oy as f64) / sy)
    };
    (
        cx.clamp(0.0, (cw - 1) as f64),
        cy.clamp(0.0, (ch - 1) as f64),
    )
}

/// 使用本地小地图截图 + 官方画布 + 完整地图图，定位玩家（不依赖游戏进程）。
pub fn locate_from_images(
    shot: &RgbImage,
    canvas: &RgbImage,
    full: &RgbImage,
    street: &str,
    name: &str,
    map_id: Option<u64>,
) -> Result<LocateResult, String> {
    let view = crop_rgb(shot, VIEW_X, VIEW_Y, VIEW_W, VIEW_H);
    let (px, py) = find_player_yellow(&view).ok_or("未在小地图中找到玩家黄点")?;
    let align = view_to_canvas(&view, canvas)?;
    let (cx, cy) = player_on_canvas(px, py, &align, canvas.width(), canvas.height());
    let fx = cx / canvas.width() as f64 * full.width() as f64;
    let fy = cy / canvas.height() as f64 * full.height() as f64;

    Ok(LocateResult {
        street: street.to_string(),
        name: name.to_string(),
        map_id,
        shot_x: VIEW_X as f64 + px,
        shot_y: VIEW_Y as f64 + py,
        view_x: px,
        view_y: py,
        canvas_x: cx,
        canvas_y: cy,
        full_x: fx,
        full_y: fy,
        canvas_w: canvas.width(),
        canvas_h: canvas.height(),
        full_w: full.width(),
        full_h: full.height(),
        align,
    })
}

/// 自动截取游戏进程小地图 → OCR → 下载完整地图。
pub fn save_from_minimap(root: &Path) -> Result<(String, String, u64, PathBuf), String> {
    #[cfg(windows)]
    {
        let img = crate::capture::capture_minimap_image()?;
        let (street, name) = read_map_names(&img)?;
        let query = format!("{street}-{name}");
        let out_dir = maps_dir(root);
        let (map_id, path, _label) = save_map(&query, &out_dir)?;
        Ok((street, name, map_id, path))
    }
    #[cfg(not(windows))]
    {
        let _ = root;
        Err("仅支持 Windows".into())
    }
}

/// 自动截取游戏进程小地图 → OCR → 网络对齐 → 标注玩家位置。
pub fn locate_player(root: &Path) -> Result<String, String> {
    #[cfg(windows)]
    {
        let shot_im = crate::capture::capture_minimap_image()?;
        let (street, name) = read_map_names(&shot_im)?;
        let query = format!("{street}-{name}");
        let map_id = resolve_map_id(&query).ok_or_else(|| format!("找不到地图：{query}"))?;

        let canvas = fetch_canvas(map_id)?;
        let full = fetch_full_map(map_id)?;
        let result = locate_from_images(&shot_im, &canvas, &full, &street, &name, Some(map_id))?;

        let mut marked = full.clone();
        mark_cross(&mut marked, result.full_x, result.full_y);
        let out_dir = maps_dir(root);
        std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
        let marked_path = out_dir.join(format!(
            "player_{}-{}_{map_id}.png",
            safe_filename(&street),
            safe_filename(&name)
        ));
        marked.save(&marked_path).map_err(|e| e.to_string())?;

        Ok(format!(
            "一级地图 {street}\n二级地图 {name}\n地图ID {map_id}\n对齐方式 {}  分数 {:.3}  视口原点 {:?}\n玩家截图坐标 {:.1}, {:.1}\n画布坐标 {:.1}, {:.1}  ({}x{})\n完整地图坐标 {:.1}, {:.1}  ({}x{})\n已标注 {}",
            result.align.mode,
            result.align.score,
            result.align.loc,
            result.shot_x,
            result.shot_y,
            result.canvas_x,
            result.canvas_y,
            result.canvas_w,
            result.canvas_h,
            result.full_x,
            result.full_y,
            result.full_w,
            result.full_h,
            marked_path.display()
        ))
    }
    #[cfg(not(windows))]
    {
        let _ = root;
        Err("仅支持 Windows".into())
    }
}
