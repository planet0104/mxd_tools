//! 冒险岛怀旧风小游戏（窗口模式）。
//!
//! 截图（离屏，供 YOLO / find_player）：
//!   cargo run --release --bin mini_game_headless -- --screenshot screen_caps/.../out.png
//!
//! 实时 YOLO+OCR 预览（手动操作）：
//!   cargo run --release --bin mini_game -- --vision-preview --model models/yolo_nangang_e3000_best.onnx
//!
//! NEAT 最优个体回放（训练时另开终端）：
//!   cargo run --release --bin neat_preview

use std::env;
use std::path::PathBuf;

use macroquad::prelude::*;
use mxd_tools::game::{
    self, GameSim, InputFrame, LOGIC_DT, NEAT_CONF_THRESH, VisionPipeline, VisionStep, WINDOW_H,
    WINDOW_W,
};
use mxd_tools::game::view;
use mxd_tools::yolo::YoloDevice;

fn window_conf() -> Conf {
    Conf {
        window_title: "冒险岛小游戏".to_owned(),
        window_width: (WINDOW_W / 3.0).round() as i32,
        window_height: (WINDOW_H / 3.0).round() as i32,
        window_resizable: true,
        high_dpi: true,
        ..Default::default()
    }
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn begin_logical_viewport() {
    let sw = screen_width();
    let sh = screen_height();
    let scale = f32::min(sw / WINDOW_W, sh / WINDOW_H);
    let vw = (WINDOW_W * scale).round();
    let vh = (WINDOW_H * scale).round();
    let ox = ((sw - vw) * 0.5).round() as i32;
    let oy_top = ((sh - vh) * 0.5).round() as i32;
    let oy = sh.round() as i32 - oy_top - vh as i32;

    let mut cam = view::logical_camera();
    cam.viewport = Some((ox, oy, vw as i32, vh as i32));
    set_camera(&cam);
}

#[macroquad::main(window_conf)]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let vision_preview = args.iter().any(|a| a == "--vision-preview");

    let assets = match view::load_view_assets().await {
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

    let mut vision: Option<VisionPipeline> = None;
    let mut last_vision: Option<VisionStep> = None;
    let rt = view::new_render_target();

    if vision_preview {
        let model = arg_value(&args, "--model")
            .unwrap_or_else(|| "models/yolo_nangang_e3000_best.onnx".to_string());
        match VisionPipeline::load(
            PathBuf::from(&model).as_path(),
            YoloDevice::Cpu,
            NEAT_CONF_THRESH,
        ) {
            Ok(p) => {
                eprintln!("视觉预览: YOLO conf>={NEAT_CONF_THRESH} model={model}");
                vision = Some(p);
            }
            Err(e) => eprintln!("加载 YOLO 失败，预览关闭: {e}"),
        }
    }

    let mut sim = GameSim::new(map, 42);
    let mut acc = 0.0f32;

    loop {
        let input = poll_input();
        acc += get_frame_time();
        while acc >= LOGIC_DT {
            sim.tick(&input);
            acc -= LOGIC_DT;
        }

        if let Some(pipeline) = vision.as_mut() {
            if sim.state.tick % 12 == 0 {
                view::draw_to_render_target(&assets, &sim, &rt);
                next_frame().await;
                let rgb = view::render_target_to_rgb(&rt);
                match pipeline.perceive(&rgb) {
                    Ok(step) => last_vision = Some(step),
                    Err(e) => eprintln!("视觉推理失败: {e}"),
                }
            }
        }

        clear_background(Color::new(0.05, 0.05, 0.08, 1.0));
        begin_logical_viewport();
        view::draw_content(&assets, &sim);
        if let Some(step) = last_vision.as_ref() {
            view::draw_yolo_overlay(&step.detections, NEAT_CONF_THRESH);
            if let Some(hit) = step.self_player.as_ref() {
                view::draw_self_player_marker(hit);
            }
        }
        set_default_camera();

        next_frame().await;
    }
}

fn poll_input() -> InputFrame {
    InputFrame {
        left: is_key_down(KeyCode::Left) || is_key_down(KeyCode::A),
        right: is_key_down(KeyCode::Right) || is_key_down(KeyCode::D),
        jump: is_key_down(KeyCode::Space) || is_key_down(KeyCode::LeftAlt),
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
