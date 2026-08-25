//! 冒险岛怀旧风小游戏（窗口模式）。

use std::collections::HashMap;
use std::path::PathBuf;

use macroquad::prelude::*;
use mxd_tools::game::{
    self, DropKind, GameModal, GameSim, InputFrame, PlayerAnim, UiLayoutFile, ATTACK_DURATION,
    LOGIC_DT, WINDOW_H, WINDOW_W, WORLD_VIEW_H,
};
use mxd_tools::game::{MobAnim, MobState, PlayerState};

fn window_conf() -> Conf {
    Conf {
        window_title: "冒险岛小游戏".to_owned(),
        // 初始约为逻辑分辨率的 1/3，可拉伸；画面仍按 1368×768 逻辑坐标绘制
        window_width: (WINDOW_W / 3.0).round() as i32,
        window_height: (WINDOW_H / 3.0).round() as i32,
        window_resizable: true,
        high_dpi: true,
        ..Default::default()
    }
}

/// 将逻辑分辨率 1368×768 等比缩放到当前窗口，留黑边。
///
/// 注意：OpenGL viewport 的 y 从底部算起；`from_display_rect` 的负 zoom.y
/// 在部分环境下与 viewport 叠用会整屏倒置，这里改用正 zoom + 底部原点 viewport。
fn begin_logical_viewport() {
    let sw = screen_width();
    let sh = screen_height();
    let scale = f32::min(sw / WINDOW_W, sh / WINDOW_H);
    let vw = (WINDOW_W * scale).round();
    let vh = (WINDOW_H * scale).round();
    let ox = ((sw - vw) * 0.5).round() as i32;
    let oy_top = ((sh - vh) * 0.5).round() as i32;
    let oy = sh.round() as i32 - oy_top - vh as i32;

    set_camera(&Camera2D {
        target: vec2(WINDOW_W * 0.5, WINDOW_H * 0.5),
        zoom: vec2(2.0 / WINDOW_W, 2.0 / WINDOW_H),
        viewport: Some((ox, oy, vw as i32, vh as i32)),
        ..Default::default()
    });
}

fn end_logical_viewport() {
    set_default_camera();
}

struct AnimFrames {
    textures: Vec<Texture2D>,
    fps: f32,
}

struct GameAssets {
    map_bg: Texture2D,
    ui: HashMap<String, Texture2D>,
    ui_layout: UiLayoutFile,
    player: HashMap<String, AnimFrames>,
    mobs: HashMap<u32, HashMap<String, AnimFrames>>,
    meso: Texture2D,
    potion: Texture2D,
    portal: AnimFrames,
}

#[macroquad::main(window_conf)]
async fn main() {
    let assets = match load_assets().await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("加载资源失败: {e}");
            return;
        }
    };

    let map = match game::load_default_map() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("加载地图失败: {e}");
            return;
        }
    };

    let mut sim = GameSim::new(map, 42);
    let mut acc = 0.0f32;
    // is_key_pressed 只在渲染帧为真；逻辑步固定 60Hz 时可能丢边沿，需锁存
    let mut jump_latched = false;
    let mut attack_latched = false;

    loop {
        let mut input = poll_input();
        jump_latched |= input.jump;
        attack_latched |= input.attack;
        input.jump = jump_latched;
        input.attack = attack_latched;

        let dt = get_frame_time();
        acc += dt;
        while acc >= LOGIC_DT {
            sim.tick(&input);
            // 每个逻辑步最多消费一次边沿，避免同帧多 tick 重复触发
            if input.jump {
                jump_latched = false;
                input.jump = false;
            }
            if input.attack {
                attack_latched = false;
                input.attack = false;
            }
            acc -= LOGIC_DT;
        }

        draw_frame(&assets, &sim);
        next_frame().await;
    }
}

async fn load_texture(path: &PathBuf) -> Result<Texture2D, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path:?}: {e}"))?;
    let tex = Texture2D::from_file_with_format(&bytes, None);
    Ok(tex)
}

async fn load_anim_dir(dir: &PathBuf, prefix: &str, count: usize, fps: f32) -> Result<AnimFrames, String> {
    let mut textures = Vec::new();
    for i in 0..count {
        let name = format!("{prefix}_{i}.png");
        let path = dir.join(&name);
        if !path.is_file() {
            continue;
        }
        textures.push(load_texture(&path).await?);
    }
    if textures.is_empty() {
        return Err(format!("{dir:?} 无 {prefix} 帧"));
    }
    Ok(AnimFrames { textures, fps })
}

async fn load_assets() -> Result<GameAssets, String> {
    let map_path = PathBuf::from(game::load_default_map().map_err(|e| e.to_string())?.image_path);
    let map_bg = load_texture(&map_path).await?;

    let ui_layout = game::load_ui_layout().map_err(|e| e.to_string())?;
    let mut ui = HashMap::new();
    for (_k, rect) in [
        ("minimap", &ui_layout.widgets.minimap),
        ("quest", &ui_layout.widgets.quest),
        ("float_buttons", &ui_layout.widgets.float_buttons),
        ("panel", &ui_layout.widgets.panel),
        ("keyboard", &ui_layout.widgets.keyboard),
    ] {
        let path = game::ui_texture_path(&rect.file);
        ui.insert(rect.file.clone(), load_texture(&path).await?);
    }

    let player_dir = game::player_sprite_dir();
    let mut player = HashMap::new();
    player.insert(
        "stand1".into(),
        load_anim_dir(&player_dir, "stand1", 4, 6.0).await?,
    );
    player.insert(
        "walk1".into(),
        load_anim_dir(&player_dir, "walk1", 4, 8.0).await?,
    );
    player.insert(
        "jump".into(),
        load_anim_dir(&player_dir, "jump", 1, 1.0).await?,
    );
    player.insert(
        "alert".into(),
        load_anim_dir(&player_dir, "alert", 3, 10.0).await?,
    );
    player.insert(
        "swingO1".into(),
        load_anim_dir(&player_dir, "swingO1", 3, 12.0)
            .await
            .unwrap_or_else(|_| AnimFrames {
                textures: vec![],
                fps: 12.0,
            }),
    );
    player.insert(
        "ladder".into(),
        load_anim_dir(&player_dir, "ladder", 4, 8.0).await.unwrap_or_else(|_| AnimFrames {
            textures: vec![],
            fps: 8.0,
        }),
    );
    player.insert(
        "rope".into(),
        load_anim_dir(&player_dir, "rope", 4, 8.0).await.unwrap_or_else(|_| AnimFrames {
            textures: vec![],
            fps: 8.0,
        }),
    );

    let mut mobs = HashMap::new();
    for mob_id in [100101u32, 130101, 1210102, 130100] {
        let dir = game::mob_sprite_dir(mob_id);
        let mut anims = HashMap::new();
        anims.insert(
            "move".into(),
            load_anim_dir(&dir, "move", 4, 8.0).await.unwrap_or_else(|_| AnimFrames {
                textures: vec![],
                fps: 8.0,
            }),
        );
        anims.insert(
            "hit1".into(),
            load_anim_dir(&dir, "hit1", 1, 1.0).await.unwrap_or_else(|_| AnimFrames {
                textures: vec![],
                fps: 1.0,
            }),
        );
        anims.insert(
            "die1".into(),
            load_anim_dir(&dir, "die1", 3, 6.0).await.unwrap_or_else(|_| AnimFrames {
                textures: vec![],
                fps: 6.0,
            }),
        );
        mobs.insert(mob_id, anims);
    }

    let meso = load_texture(&game::drop_texture_path(DropKind::Meso)).await?;
    let potion = load_texture(&game::drop_texture_path(DropKind::RedPotion)).await?;
    let portal_dir = game::portal_sprite_dir();
    let portal = load_anim_dir(&portal_dir, "pv", 8, 10.0)
        .await
        .unwrap_or_else(|_| AnimFrames {
            textures: vec![],
            fps: 10.0,
        });

    Ok(GameAssets {
        map_bg,
        ui,
        ui_layout,
        player,
        mobs,
        meso,
        potion,
        portal,
    })
}

fn poll_input() -> InputFrame {
    InputFrame {
        left: is_key_down(KeyCode::Left) || is_key_down(KeyCode::A),
        right: is_key_down(KeyCode::Right) || is_key_down(KeyCode::D),
        jump: is_key_pressed(KeyCode::Space) || is_key_pressed(KeyCode::LeftAlt),
        attack: is_key_pressed(KeyCode::LeftControl) || is_key_pressed(KeyCode::J),
        up: is_key_down(KeyCode::Up) || is_key_down(KeyCode::W),
        down: is_key_down(KeyCode::Down) || is_key_down(KeyCode::S),
        pick_up: is_key_down(KeyCode::Z),
        use_potion: is_key_pressed(KeyCode::Key1),
        open_inventory: is_key_pressed(KeyCode::I),
        inventory_click: None,
        restart: is_key_pressed(KeyCode::R),
    }
}

fn draw_frame(assets: &GameAssets, sim: &GameSim) {
    clear_background(Color::new(0.05, 0.05, 0.08, 1.0));
    begin_logical_viewport();

    let cam_x = sim.state.cam_x;
    let cam_y = sim.state.cam_y;

    draw_rectangle(0.0, 0.0, WINDOW_W, WORLD_VIEW_H, Color::new(0.05, 0.05, 0.08, 1.0));
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

    for mob in &sim.state.mobs {
        if !mob.alive && mob.die_t <= 0.0 {
            continue;
        }
        draw_mob(assets, mob, cam_x, cam_y);
    }

    draw_player(assets, &sim.state.player, cam_x, cam_y);

    draw_ui_shell(assets, sim);
    end_logical_viewport();
}

fn draw_portal(
    assets: &GameAssets,
    portal: &mxd_tools::game::Portal,
    cam_x: f32,
    cam_y: f32,
    tick: u64,
) {
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

fn draw_player(assets: &GameAssets, p: &PlayerState, cam_x: f32, cam_y: f32) {
    // 资源立绘默认朝左；facing>0 向右时需要水平翻转
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
                assets.player.get("alert").filter(|a| !a.textures.is_empty())
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
    let sx = p.x - cam_x;
    let sy = p.y - cam_y;
    let w = tex.width();
    let h = tex.height();
    let alpha = if p.invuln_t > 0.0 && (p.invuln_t * 20.0) as i32 % 2 == 0 {
        0.45
    } else {
        1.0
    };
    draw_texture_ex(
        tex,
        sx - w * 0.5,
        sy - h + 8.0,
        Color::new(1.0, 1.0, 1.0, alpha),
        DrawTextureParams {
            flip_x: flip,
            ..Default::default()
        },
    );
}

fn draw_mob(assets: &GameAssets, mob: &MobState, cam_x: f32, cam_y: f32) {
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
    // 怪物贴图默认朝左；vx>0 向右走时翻转
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

fn draw_ui_shell(assets: &GameAssets, sim: &GameSim) {
    let w = &assets.ui_layout.widgets;
    for rect in [&w.minimap, &w.quest, &w.float_buttons, &w.panel, &w.keyboard] {
        if let Some(tex) = assets.ui.get(&rect.file) {
            draw_texture(tex, rect.x, rect.y, WHITE);
        }
    }

    let hp = &assets.ui_layout.dynamic_overlay.hp_bar;
    draw_rectangle(hp.x, hp.y, hp.w, hp.h, Color::new(0.15, 0.15, 0.15, 0.9));
    let ratio = sim.state.player.hp as f32 / sim.state.player.max_hp as f32;
    draw_rectangle(hp.x, hp.y, hp.w * ratio.clamp(0.0, 1.0), hp.h, Color::new(0.85, 0.15, 0.15, 1.0));

    let mp = &assets.ui_layout.dynamic_overlay.mp_bar;
    draw_rectangle(mp.x, mp.y, mp.w, mp.h, Color::new(0.15, 0.15, 0.2, 0.9));
    draw_rectangle(mp.x, mp.y, mp.w * 0.8, mp.h, Color::new(0.2, 0.35, 0.9, 1.0));

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
        draw_rectangle(
            inv.x,
            inv.y,
            inv.w,
            inv.h,
            Color::new(0.1, 0.1, 0.15, 0.92),
        );
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
        draw_text("点击或按 1 使用", inv.x + 60.0, inv.y + 100.0, 18.0, LIGHTGRAY);
    }

    if sim.state.modal == GameModal::GameOver {
        draw_rectangle(0.0, 0.0, WINDOW_W, WINDOW_H, Color::new(0.0, 0.0, 0.0, 0.55));
        draw_text(
            "Game Over — 按 R 重开",
            WINDOW_W * 0.5 - 120.0,
            WINDOW_H * 0.45,
            32.0,
            RED,
        );
    }
}
