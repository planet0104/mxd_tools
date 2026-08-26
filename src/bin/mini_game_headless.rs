//! 无头模式：固定步进逻辑 + 离屏渲染截图（供 YOLO / find_player 测试）。
//!
//! ```powershell
//! cargo run --release --bin mini_game_headless -- --training --screenshot screen_caps/.../out.png
//! cargo run --release --bin mini_game_headless -- --seed 42 --ticks 600 --dump-every 60
//! ```

use std::env;
use std::path::{Path, PathBuf};

use macroquad::prelude::*;
use mxd_tools::game::{self, GameSim, InputFrame, LOGIC_DT, WINDOW_H, WINDOW_W};
use mxd_tools::game::view::{self, GameViewAssets};
use mxd_tools::headless_gl;

fn window_conf() -> Conf {
    headless_gl::headless_window_conf("mini_game_headless")
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[macroquad::main(window_conf)]
async fn main() {
    let _ = headless_gl::hide_gl_window();
    let args: Vec<String> = env::args().collect();
    let training = args.iter().any(|a| a == "--training");
    let seed = arg_u64(&args, "--seed", 42);
    let ticks = arg_u64(&args, "--ticks", 120) as usize;
    let dump_every = arg_u64(&args, "--dump-every", 0) as usize;
    let screenshot_out = arg_value(&args, "--screenshot");
    let warmup_frames = arg_u64(&args, "--warmup-frames", 3) as usize;

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

    let mut sim = if training {
        GameSim::new_training(map, seed)
    } else {
        GameSim::new(map, seed)
    };
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tmp/mini_game_headless");
    if dump_every > 0 {
        let _ = std::fs::create_dir_all(&out_dir);
    }

    let rt = view::new_render_target();

    for i in 0..ticks {
        sim.tick(&InputFrame::default());
        if dump_every > 0 && i % dump_every == 0 {
            let frame = capture_after_draw(&assets, &sim, &rt, warmup_frames).await;
            let path = out_dir.join(format!("frame_{i:05}.png"));
            if let Err(e) = frame.save(&path) {
                eprintln!("写帧失败: {e}");
            }
        }
    }

    if let Some(ref path) = screenshot_out {
        let frame = capture_after_draw(&assets, &sim, &rt, warmup_frames).await;
        if let Some(parent) = Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match frame.save(path) {
            Ok(()) => eprintln!("截图已保存: {path} ({WINDOW_W}x{WINDOW_H})"),
            Err(e) => eprintln!("截图失败: {e}"),
        }
    }

    let gt = sim.ground_truth();
    println!(
        "ticks={ticks} dt={LOGIC_DT} player=({:.0},{:.0}) hp={}/{} potions={} meso={} mobs_alive={}",
        gt.player_x,
        gt.player_y,
        gt.hp,
        gt.max_hp,
        gt.potions,
        gt.meso,
        gt.mob_count
    );
    if dump_every > 0 {
        println!("帧输出: {}", out_dir.display());
    }
}

async fn capture_after_draw(
    assets: &GameViewAssets,
    sim: &GameSim,
    rt: &RenderTarget,
    warmup_frames: usize,
) -> image::RgbImage {
    for _ in 0..warmup_frames {
        view::draw_to_render_target(assets, sim, rt);
        next_frame().await;
    }
    view::draw_to_render_target(assets, sim, rt);
    next_frame().await;
    view::render_target_to_rgb(rt)
}
