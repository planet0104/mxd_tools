//! macroquad 游戏画面绘制与离屏截图（窗口 / headless 共用）。

use std::collections::HashMap;
use std::path::PathBuf;

use image::RgbImage;
use macroquad::prelude::*;

use super::npc::NpcPlayerState;
use super::types::TRAINING_NPC_SPRITES;
use super::{
    assets_root, drop_texture_path, load_default_map, load_ui_layout, mob_sprite_dir,
    player_sprite_dir, player_sprite_dir_named, portal_sprite_dir, ui_texture_path, DropKind,
    GameModal, GameSim, MobAnim, MobState, PlayerAnim, PlayerState, Portal, UiLayoutFile, UiRect,
    ATTACK_DURATION, DEFAULT_PLAYER_NAME, LOGIC_DT, NAME_TAG_BG_ALPHA, NAME_TAG_FONT_SIZE,
    NAME_TAG_GAP_BELOW_FEET, NAME_TAG_PAD_X, NAME_TAG_PAD_Y, WINDOW_H, WINDOW_W, WORLD_VIEW_H,
};

pub struct AnimFrames {
    pub textures: Vec<Texture2D>,
    pub fps: f32,
}

/// 原版 Character.wz 常见 delay（ms）→ fps。
/// stand1/alert ≈ 500ms；walk1 ≈ 180ms。
const PLAYER_STAND_FPS: f32 = 1000.0 / 500.0;
const PLAYER_WALK_FPS: f32 = 1000.0 / 180.0;
const PLAYER_ALERT_FPS: f32 = 1000.0 / 500.0;
const PLAYER_ATTACK_FPS: f32 = 12.0;
const PLAYER_CLIMB_FPS: f32 = 8.0;

pub struct GameViewAssets {
    pub map_bg: Texture2D,
    pub ui: HashMap<String, Texture2D>,
    pub ui_layout: UiLayoutFile,
    pub player: HashMap<String, AnimFrames>,
    /// 装饰玩家精灵（键 = `assets/player/<名>/` 目录名）
    pub player_sets: HashMap<String, HashMap<String, AnimFrames>>,
    pub mobs: HashMap<u32, HashMap<String, AnimFrames>>,
    pub meso: Texture2D,
    pub potion: Texture2D,
    pub portal: AnimFrames,
    pub name_font: Font,
}

pub fn logical_camera() -> Camera2D {
    Camera2D {
        target: vec2(WINDOW_W * 0.5, WINDOW_H * 0.5),
        zoom: vec2(2.0 / WINDOW_W, 2.0 / WINDOW_H),
        ..Default::default()
    }
}

/// 将逻辑分辨率画面居中缩放到当前窗口（窗口模式预览/训练用）。
pub fn begin_logical_viewport() {
    let sw = screen_width();
    let sh = screen_height();
    let scale = f32::min(sw / WINDOW_W, sh / WINDOW_H);
    let vw = (WINDOW_W * scale).round();
    let vh = (WINDOW_H * scale).round();
    let ox = ((sw - vw) * 0.5).round() as i32;
    let oy_top = ((sh - vh) * 0.5).round() as i32;
    let oy = sh.round() as i32 - oy_top - vh as i32;

    let mut cam = logical_camera();
    cam.viewport = Some((ox, oy, vw as i32, vh as i32));
    set_camera(&cam);
}

pub fn new_render_target() -> RenderTarget {
    let rt = render_target(WINDOW_W as u32, WINDOW_H as u32);
    rt.texture.set_filter(FilterMode::Nearest);
    rt
}

/// 绘制一帧到离屏 RenderTarget（调用方需在 `next_frame()` 后读像素）。
pub fn draw_to_render_target(assets: &GameViewAssets, sim: &GameSim, rt: &RenderTarget) {
    let mut cam = logical_camera();
    cam.render_target = Some(rt.clone());
    set_camera(&cam);
    clear_background(Color::new(0.05, 0.05, 0.08, 1.0));
    draw_content(assets, sim);
    set_default_camera();
}

pub fn render_target_to_rgb(rt: &RenderTarget) -> RgbImage {
    let img = rt.texture.get_texture_data();
    let w = img.width as u32;
    let h = img.height as u32;
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            if i + 2 >= img.bytes.len() {
                continue;
            }
            out.put_pixel(
                x,
                y,
                image::Rgb([img.bytes[i], img.bytes[i + 1], img.bytes[i + 2]]),
            );
        }
    }
    out
}

/// 世界层 + UI（不含清屏与相机设置）。
pub fn draw_content(assets: &GameViewAssets, sim: &GameSim) {
    let cam_x = sim.state.cam_x;
    let cam_y = sim.state.cam_y;

    draw_rectangle(
        0.0,
        0.0,
        WINDOW_W,
        WORLD_VIEW_H,
        Color::new(0.05, 0.05, 0.08, 1.0),
    );
    draw_map(&assets.map_bg, cam_x, cam_y);

    for portal in &sim.map.portals {
        draw_portal(assets, portal, cam_x, cam_y, sim.state.tick);
    }

    for drop in &sim.state.drops {
        if !drop.alive {
            continue;
        }
        let sx = drop.x - cam_x;
        let sy = drop.y - cam_y + (drop.bob_t * 4.0).sin() * 3.0;
        let tex = match drop.kind {
            DropKind::Meso => &assets.meso,
            DropKind::RedPotion => &assets.potion,
        };
        let s = 24.0;
        draw_texture_ex(
            tex,
            sx - s * 0.5,
            sy - s,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(s, s)),
                ..Default::default()
            },
        );
    }

    draw_map_actors_y_sorted(assets, sim, cam_x, cam_y);

    draw_ui_shell(assets, sim);
}

/// 地图上的玩家、装饰 NPC、怪物按脚点 Y 排序绘制（怀旧版深度规则：Y 大者在前）。
enum MapActorDraw<'a> {
    Mob(&'a MobState),
    Player(&'a PlayerState),
    Npc(&'a NpcPlayerState),
}

impl MapActorDraw<'_> {
    fn foot_y(&self) -> f32 {
        match self {
            MapActorDraw::Mob(m) => m.y,
            MapActorDraw::Player(p) => p.y,
            MapActorDraw::Npc(n) => n.y,
        }
    }

    fn foot_x(&self) -> f32 {
        match self {
            MapActorDraw::Mob(m) => m.x,
            MapActorDraw::Player(p) => p.x,
            MapActorDraw::Npc(n) => n.x,
        }
    }
}

fn draw_map_actors_y_sorted(assets: &GameViewAssets, sim: &GameSim, cam_x: f32, cam_y: f32) {
    let mut actors: Vec<MapActorDraw<'_>> = Vec::new();
    for mob in &sim.state.mobs {
        if !mob.alive && mob.die_t <= 0.0 {
            continue;
        }
        actors.push(MapActorDraw::Mob(mob));
    }
    actors.push(MapActorDraw::Player(&sim.state.player));
    for npc in &sim.npc_players {
        actors.push(MapActorDraw::Npc(npc));
    }
    actors.sort_by(|a, b| {
        a.foot_y()
            .partial_cmp(&b.foot_y())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.foot_x()
                    .partial_cmp(&b.foot_x())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    for actor in actors {
        match actor {
            MapActorDraw::Mob(mob) => draw_mob(assets, mob, cam_x, cam_y),
            MapActorDraw::Player(p) => {
                draw_player(assets, p, cam_x, cam_y);
                draw_player_name_tag(
                    &assets.name_font,
                    DEFAULT_PLAYER_NAME,
                    p.x - cam_x,
                    p.y - cam_y,
                );
            }
            MapActorDraw::Npc(npc) => {
                draw_npc_player(assets, npc, cam_x, cam_y);
                draw_player_name_tag(&assets.name_font, &npc.name, npc.x - cam_x, npc.y - cam_y);
            }
        }
    }
}

pub async fn load_view_assets() -> Result<GameViewAssets, String> {
    let map_path = PathBuf::from(load_default_map().map_err(|e| e.to_string())?.image_path);
    let map_bg = load_texture(&map_path).await?;

    let ui_layout = load_ui_layout().map_err(|e| e.to_string())?;
    let mut ui = HashMap::new();
    let mut rects: Vec<&UiRect> = vec![
        &ui_layout.widgets.minimap,
        &ui_layout.widgets.panel,
        &ui_layout.widgets.keyboard,
    ];
    if let Some(r) = &ui_layout.widgets.quest {
        rects.push(r);
    }
    if let Some(r) = &ui_layout.widgets.float_buttons {
        rects.push(r);
    }
    for rect in rects {
        let path = ui_texture_path(&rect.file);
        if !path.is_file() {
            return Err(format!("缺少 UI 贴图: {}", path.display()));
        }
        ui.insert(rect.file.clone(), load_texture(&path).await?);
    }

    let player_dir = player_sprite_dir();
    let player = load_player_anims(&player_dir).await?;

    let mut player_sets = HashMap::new();
    for name in TRAINING_NPC_SPRITES {
        let dir = player_sprite_dir_named(name);
        if dir.is_dir() {
            if let Ok(anims) = load_player_anims(&dir).await {
                player_sets.insert(name.to_string(), anims);
            }
        }
    }

    let mut mobs = HashMap::new();
    for mob_id in [100101u32, 130101, 1210102, 130100] {
        let dir = mob_sprite_dir(mob_id);
        let mut anims = HashMap::new();
        anims.insert(
            "move".into(),
            load_anim_dir(&dir, "move", 4, 8.0)
                .await
                .unwrap_or_else(|_| AnimFrames {
                    textures: vec![],
                    fps: 8.0,
                }),
        );
        anims.insert(
            "hit1".into(),
            load_anim_dir(&dir, "hit1", 1, 1.0)
                .await
                .unwrap_or_else(|_| AnimFrames {
                    textures: vec![],
                    fps: 1.0,
                }),
        );
        anims.insert(
            "die1".into(),
            load_anim_dir(&dir, "die1", 3, 6.0)
                .await
                .unwrap_or_else(|_| AnimFrames {
                    textures: vec![],
                    fps: 6.0,
                }),
        );
        mobs.insert(mob_id, anims);
    }

    let meso = load_texture(&drop_texture_path(DropKind::Meso)).await?;
    let potion = load_texture(&drop_texture_path(DropKind::RedPotion)).await?;
    let portal_dir = portal_sprite_dir();
    let portal = load_anim_dir(&portal_dir, "pv", 8, 10.0)
        .await
        .unwrap_or_else(|_| AnimFrames {
            textures: vec![],
            fps: 10.0,
        });

    let name_font = load_cjk_font()?;

    Ok(GameViewAssets {
        map_bg,
        ui,
        ui_layout,
        player,
        player_sets,
        mobs,
        meso,
        potion,
        portal,
        name_font,
    })
}

async fn load_player_anims(dir: &PathBuf) -> Result<HashMap<String, AnimFrames>, String> {
    let mut player = HashMap::new();
    player.insert(
        "stand1".into(),
        load_anim_dir(dir, "stand1", 4, PLAYER_STAND_FPS).await?,
    );
    player.insert(
        "walk1".into(),
        load_anim_dir(dir, "walk1", 4, PLAYER_WALK_FPS).await?,
    );
    player.insert("jump".into(), load_anim_dir(dir, "jump", 1, 1.0).await?);
    player.insert(
        "alert".into(),
        load_anim_dir(dir, "alert", 3, PLAYER_ALERT_FPS).await?,
    );
    player.insert(
        "swingO1".into(),
        load_anim_dir(dir, "swingO1", 3, PLAYER_ATTACK_FPS)
            .await
            .unwrap_or_else(|_| AnimFrames {
                textures: vec![],
                fps: PLAYER_ATTACK_FPS,
            }),
    );
    player.insert(
        "ladder".into(),
        load_anim_dir(dir, "ladder", 4, PLAYER_CLIMB_FPS)
            .await
            .unwrap_or_else(|_| AnimFrames {
                textures: vec![],
                fps: PLAYER_CLIMB_FPS,
            }),
    );
    player.insert(
        "rope".into(),
        load_anim_dir(dir, "rope", 4, PLAYER_CLIMB_FPS)
            .await
            .unwrap_or_else(|_| AnimFrames {
                textures: vec![],
                fps: PLAYER_CLIMB_FPS,
            }),
    );
    Ok(player)
}

async fn load_texture(path: &PathBuf) -> Result<Texture2D, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path:?}: {e}"))?;
    let tex = Texture2D::from_file_with_format(&bytes, None);
    tex.set_filter(FilterMode::Nearest);
    Ok(tex)
}

/// IO 合成立绘各帧画布与内容裁切不一致；按不透明像素脚底中心对齐到统一画布，
/// 避免站立呼吸时整身左右抖出叠影。
fn pad_frames_foot_center(frames: Vec<image::RgbaImage>) -> Vec<image::RgbaImage> {
    if frames.is_empty() {
        return frames;
    }

    let metas: Vec<(image::RgbaImage, u32, u32, u32, u32)> = frames
        .into_iter()
        .map(|src| {
            let (bx, by, bw, bh) = opaque_bbox(&src).unwrap_or((0, 0, src.width(), src.height()));
            (src, bx, by, bw, bh)
        })
        .collect();

    let max_bw = metas.iter().map(|(_, _, _, bw, _)| *bw).max().unwrap_or(1);
    let max_bh = metas.iter().map(|(_, _, _, _, bh)| *bh).max().unwrap_or(1);
    // 左右各留 1px，避免贴边采样发糊
    let canvas_w = max_bw + 2;
    let canvas_h = max_bh + 1;
    let foot_x = (canvas_w / 2) as i64;
    let foot_y = (canvas_h - 1) as i64;

    metas
        .into_iter()
        .map(|(src, bx, by, bw, bh)| {
            let mut canvas = image::RgbaImage::new(canvas_w, canvas_h);
            let src_foot_x = bx as i64 + (bw as i64) / 2;
            let src_foot_y = by as i64 + bh as i64 - 1;
            let ox = foot_x - src_foot_x;
            let oy = foot_y - src_foot_y;
            image::imageops::overlay(&mut canvas, &src, ox, oy);
            canvas
        })
        .collect()
}

fn opaque_bbox(img: &image::RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = img.width();
    let mut min_y = img.height();
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut any = false;
    for (x, y, p) in img.enumerate_pixels() {
        if p.0[3] > 8 {
            any = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !any {
        return None;
    }
    Some((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

fn texture_from_rgba(img: image::RgbaImage) -> Texture2D {
    let w = img.width();
    let h = img.height();
    let tex = Texture2D::from_rgba8(w as u16, h as u16, img.as_raw());
    tex.set_filter(FilterMode::Nearest);
    tex
}

async fn load_anim_dir(
    dir: &PathBuf,
    prefix: &str,
    count: usize,
    fps: f32,
) -> Result<AnimFrames, String> {
    let mut raw: Vec<image::RgbaImage> = Vec::new();
    for i in 0..count {
        let name = format!("{prefix}_{i}.png");
        let path = dir.join(&name);
        if !path.is_file() {
            continue;
        }
        let bytes = std::fs::read(&path).map_err(|e| format!("{path:?}: {e}"))?;
        let img = image::load_from_memory(&bytes)
            .map_err(|e| format!("{path:?}: {e}"))?
            .to_rgba8();
        raw.push(img);
    }
    if raw.is_empty() {
        return Err(format!("{dir:?} 无 {prefix} 帧"));
    }
    let padded = pad_frames_foot_center(raw);
    let textures = padded.into_iter().map(texture_from_rgba).collect();
    Ok(AnimFrames { textures, fps })
}

fn load_cjk_font() -> Result<Font, String> {
    let candidates = [
        assets_root().join("fonts/VonwaonBitmap-16px.ttf"),
        PathBuf::from(r"C:\Windows\Fonts\msyh.ttc"),
        PathBuf::from(r"C:\Windows\Fonts\simhei.ttf"),
    ];
    for path in candidates {
        if !path.is_file() {
            continue;
        }
        let bytes = std::fs::read(&path).map_err(|e| format!("读字体 {}: {e}", path.display()))?;
        return load_ttf_font_from_bytes(&bytes)
            .map_err(|e| format!("加载字体 {}: {e}", path.display()));
    }
    Err("找不到中文字体（需要 assets/fonts/VonwaonBitmap-16px.ttf）".into())
}

fn draw_portal(assets: &GameViewAssets, portal: &Portal, cam_x: f32, cam_y: f32, tick: u64) {
    if assets.portal.textures.is_empty() {
        return;
    }
    let t = tick as f32 * LOGIC_DT;
    let fi = frame_index(t, assets.portal.fps, assets.portal.textures.len());
    let tex = &assets.portal.textures[fi];
    let sx = portal.x - cam_x;
    let sy = portal.y - cam_y;
    let w = tex.width();
    let h = tex.height();
    draw_texture_ex(
        tex,
        sx - w * 0.5,
        sy - h + 16.0,
        WHITE,
        DrawTextureParams {
            ..Default::default()
        },
    );
}

fn draw_map(tex: &Texture2D, cam_x: f32, cam_y: f32) {
    let src = Rect::new(cam_x, cam_y, WINDOW_W, WORLD_VIEW_H);
    draw_texture_ex(
        tex,
        0.0,
        0.0,
        WHITE,
        DrawTextureParams {
            source: Some(src),
            dest_size: Some(vec2(WINDOW_W, WORLD_VIEW_H)),
            ..Default::default()
        },
    );
}

fn frame_index(anim_t: f32, fps: f32, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    ((anim_t * fps) as usize) % len
}

fn draw_npc_player(assets: &GameViewAssets, npc: &NpcPlayerState, cam_x: f32, cam_y: f32) {
    let anims = assets
        .player_sets
        .get(&npc.sprite_dir)
        .or(Some(&assets.player));
    let Some(anims) = anims else { return };
    let face_right = npc.facing > 0.0;
    let (key, flip) = match npc.anim {
        PlayerAnim::Attack => ("swingO1", face_right),
        PlayerAnim::Walk => ("walk1", face_right),
        _ => ("stand1", face_right),
    };
    let anim = anims
        .get(key)
        .filter(|a| !a.textures.is_empty())
        .or_else(|| {
            if key == "swingO1" {
                anims.get("alert").filter(|a| !a.textures.is_empty())
            } else {
                None
            }
        })
        .or_else(|| anims.get("walk1"))
        .or_else(|| anims.get("stand1"));
    let Some(anim) = anim else { return };
    let anim_t = if npc.anim == PlayerAnim::Attack {
        ATTACK_DURATION - npc.attack_t
    } else {
        npc.anim_t
    };
    let fi = frame_index(anim_t.max(0.0), anim.fps, anim.textures.len());
    let tex = &anim.textures[fi];
    let sx = (npc.x - cam_x).round();
    let sy = (npc.y - cam_y).round();
    let w = tex.width();
    let h = tex.height();
    draw_texture_ex(
        tex,
        sx - (w * 0.5).floor(),
        sy - h + 8.0,
        WHITE,
        DrawTextureParams {
            flip_x: flip,
            ..Default::default()
        },
    );
}

fn draw_player(assets: &GameViewAssets, p: &PlayerState, cam_x: f32, cam_y: f32) {
    let face_right = p.facing > 0.0;
    let (key, flip) = match p.anim {
        PlayerAnim::Walk => ("walk1", face_right),
        PlayerAnim::Jump => ("jump", face_right),
        PlayerAnim::Attack => ("swingO1", face_right),
        PlayerAnim::Hurt => ("alert", face_right),
        PlayerAnim::Climb => {
            if p.climb_kind == "ladder" {
                ("ladder", false)
            } else {
                ("rope", false)
            }
        }
        PlayerAnim::Stand => ("stand1", face_right),
    };
    let anim = assets
        .player
        .get(key)
        .filter(|a| !a.textures.is_empty())
        .or_else(|| {
            if key == "swingO1" {
                assets
                    .player
                    .get("alert")
                    .filter(|a| !a.textures.is_empty())
            } else {
                None
            }
        })
        .or_else(|| assets.player.get("stand1"));
    let Some(anim) = anim else { return };
    let anim_t = if p.anim == PlayerAnim::Attack {
        ATTACK_DURATION - p.attack_t
    } else {
        p.anim_t
    };
    let fi = frame_index(anim_t, anim.fps, anim.textures.len());
    let tex = &anim.textures[fi];
    let sx = (p.x - cam_x).round();
    let sy = (p.y - cam_y).round();
    let w = tex.width();
    let h = tex.height();
    let alpha = if p.invuln_t > 0.0 && (p.invuln_t * 20.0) as i32 % 2 == 0 {
        0.45
    } else {
        1.0
    };
    draw_texture_ex(
        tex,
        sx - (w * 0.5).floor(),
        sy - h + 8.0,
        Color::new(1.0, 1.0, 1.0, alpha),
        DrawTextureParams {
            flip_x: flip,
            ..Default::default()
        },
    );
}

fn draw_player_name_tag(font: &Font, name: &str, foot_sx: f32, foot_sy: f32) {
    if name.is_empty() {
        return;
    }
    let dims = measure_text(name, Some(font), NAME_TAG_FONT_SIZE, 1.0);
    let box_w = dims.width + NAME_TAG_PAD_X * 2.0;
    let box_h = dims.height + NAME_TAG_PAD_Y * 2.0;
    let box_x = foot_sx - box_w * 0.5;
    let box_y = foot_sy + NAME_TAG_GAP_BELOW_FEET;
    draw_rectangle(
        box_x,
        box_y,
        box_w,
        box_h,
        Color::new(0.0, 0.0, 0.0, NAME_TAG_BG_ALPHA),
    );
    let text_x = foot_sx - dims.width * 0.5;
    let text_y = box_y + (box_h - dims.height) * 0.5 + dims.offset_y;
    draw_text_ex(
        name,
        text_x,
        text_y,
        TextParams {
            font: Some(font),
            font_size: NAME_TAG_FONT_SIZE,
            color: WHITE,
            ..Default::default()
        },
    );
}

fn draw_mob(assets: &GameViewAssets, mob: &MobState, cam_x: f32, cam_y: f32) {
    let anims = assets.mobs.get(&mob.mob_id);
    let Some(anims) = anims else { return };
    let key = match mob.anim {
        MobAnim::Move => "move",
        MobAnim::Hit => "hit1",
        MobAnim::Die => "die1",
    };
    let anim = anims.get(key).or_else(|| anims.get("move"));
    let Some(anim) = anim else { return };
    if anim.textures.is_empty() {
        return;
    }
    let fi = frame_index(mob.anim_t, anim.fps, anim.textures.len());
    let tex = &anim.textures[fi];
    let sx = mob.x - cam_x;
    let sy = mob.y - cam_y;
    let flip = mob.vx > 0.0;
    draw_texture_ex(
        tex,
        sx - tex.width() * 0.5,
        sy - tex.height() + 6.0,
        WHITE,
        DrawTextureParams {
            flip_x: flip,
            ..Default::default()
        },
    );
}

fn draw_ui_shell(assets: &GameViewAssets, sim: &GameSim) {
    let w = &assets.ui_layout.widgets;
    let mut rects: Vec<&UiRect> = vec![&w.minimap, &w.panel, &w.keyboard];
    if let Some(r) = &w.quest {
        rects.push(r);
    }
    if let Some(r) = &w.float_buttons {
        rects.push(r);
    }
    for rect in rects {
        if let Some(tex) = assets.ui.get(&rect.file) {
            draw_texture_ex(
                tex,
                rect.x,
                rect.y,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(rect.w, rect.h)),
                    ..Default::default()
                },
            );
        }
    }

    let hp = &assets.ui_layout.dynamic_overlay.hp_bar;
    draw_rectangle(hp.x, hp.y, hp.w, hp.h, Color::new(0.15, 0.15, 0.15, 0.9));
    let ratio = sim.state.player.hp as f32 / sim.state.player.max_hp as f32;
    draw_rectangle(
        hp.x,
        hp.y,
        hp.w * ratio.clamp(0.0, 1.0),
        hp.h,
        Color::new(0.85, 0.15, 0.15, 1.0),
    );

    let mp = &assets.ui_layout.dynamic_overlay.mp_bar;
    draw_rectangle(mp.x, mp.y, mp.w, mp.h, Color::new(0.15, 0.15, 0.2, 0.9));
    draw_rectangle(
        mp.x,
        mp.y,
        mp.w * 0.8,
        mp.h,
        Color::new(0.2, 0.35, 0.9, 1.0),
    );

    let text = format!(
        "HP {}/{}  药:{}  币:{}  击杀:{}  [I]背包 [1]喝药 [Z]拾取",
        sim.state.player.hp,
        sim.state.player.max_hp,
        sim.state.potions,
        sim.state.meso,
        sim.state.kills
    );
    draw_text(&text, 12.0, WORLD_VIEW_H - 8.0, 18.0, LIGHTGRAY);

    if let Some(hint) = &sim.state.portal_hint {
        draw_text(hint, 12.0, WORLD_VIEW_H - 30.0, 18.0, YELLOW);
    }

    if sim.state.modal == GameModal::Inventory {
        let inv = &assets.ui_layout.inventory_window;
        draw_rectangle(inv.x, inv.y, inv.w, inv.h, Color::new(0.1, 0.1, 0.15, 0.92));
        draw_rectangle_lines(inv.x, inv.y, inv.w, inv.h, 2.0, WHITE);
        draw_text("物品栏 (I 关闭)", inv.x + 12.0, inv.y + 24.0, 22.0, WHITE);
        draw_text(
            &format!("红色药水 x{}", sim.state.potions),
            inv.x + 20.0,
            inv.y + 60.0,
            20.0,
            PINK,
        );
        draw_texture_ex(
            &assets.potion,
            inv.x + 20.0,
            inv.y + 80.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(32.0, 32.0)),
                ..Default::default()
            },
        );
        draw_text(
            "点击或按 1 使用",
            inv.x + 60.0,
            inv.y + 100.0,
            18.0,
            LIGHTGRAY,
        );
    }

    if sim.state.modal == GameModal::GameOver {
        draw_rectangle(
            0.0,
            0.0,
            WINDOW_W,
            WINDOW_H,
            Color::new(0.0, 0.0, 0.0, 0.55),
        );
        draw_text(
            "Game Over — 按 R 重开",
            WINDOW_W * 0.5 - 120.0,
            WINDOW_H * 0.45,
            32.0,
            RED,
        );
    }
}

/// 在逻辑坐标系上绘制 YOLO 检测框（训练预览用）。
pub fn draw_yolo_overlay(detections: &[crate::yolo::Detection], min_conf: f32) {
    for d in detections {
        if d.conf < min_conf {
            continue;
        }
        let w = d.x2 - d.x1;
        let h = d.y2 - d.y1;
        draw_rectangle_lines(d.x1, d.y1, w, h, 2.0, Color::new(0.2, 1.0, 0.35, 0.9));
        let label = format!("{} {:.0}%", d.label, d.conf * 100.0);
        draw_text(
            &label,
            d.x1,
            d.y1 - 4.0,
            14.0,
            Color::new(0.2, 1.0, 0.35, 1.0),
        );
    }
}

/// 仅绘制地板类（class_id=0）检测框，便于核对离台/寻路是否被地板观测卡住。
pub fn draw_yolo_floor_overlay(detections: &[crate::yolo::Detection], min_conf: f32) {
    let floor_color = Color::new(0.15, 0.95, 1.0, 0.95);
    let mut n = 0u32;
    for d in detections {
        if d.class_id != 0 || d.conf < min_conf {
            continue;
        }
        n += 1;
        let w = (d.x2 - d.x1).max(1.0);
        let h = (d.y2 - d.y1).max(1.0);
        draw_rectangle_lines(d.x1, d.y1, w, h, 2.5, floor_color);
        let label = format!("地板 {:.0}%", d.conf * 100.0);
        draw_text(&label, d.x1, (d.y1 - 4.0).max(12.0), 14.0, floor_color);
    }
    draw_text(&format!("YOLO 地板框: {n}"), 12.0, 22.0, 18.0, floor_color);
}

/// 标记 OCR 匹配到的自身玩家脚点。
pub fn draw_self_player_marker(hit: &crate::player_name::NamedPlayerHit) {
    let s = 6.0;
    draw_rectangle(
        hit.x - s,
        hit.y - s,
        s * 2.0,
        s * 2.0,
        Color::new(1.0, 0.2, 0.2, 0.85),
    );
}
