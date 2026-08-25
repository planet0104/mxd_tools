//! 无头模式：固定步进逻辑，可选导出帧（P3 基础）。

use std::env;
use std::path::PathBuf;

use image::RgbImage;
use mxd_tools::game::{GameSim, InputFrame, LOGIC_DT, WINDOW_H, WINDOW_W, WORLD_VIEW_H};

fn main() {
    let args: Vec<String> = env::args().collect();
    let seed = arg_u64(&args, "--seed", 42);
    let ticks = arg_u64(&args, "--ticks", 600) as usize;
    let dump_every = arg_u64(&args, "--dump-every", 0) as usize;

    let map = match mxd_tools::game::load_default_map() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("加载地图失败: {e}");
            std::process::exit(1);
        }
    };

    let mut sim = GameSim::new(map, seed);
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tmp/mini_game_headless");
    if dump_every > 0 {
        let _ = std::fs::create_dir_all(&out_dir);
    }

    for i in 0..ticks {
        let input = demo_input(i);
        sim.tick(&input);
        if dump_every > 0 && i % dump_every == 0 {
            let frame = render_placeholder(&sim);
            let path = out_dir.join(format!("frame_{i:05}.png"));
            if let Err(e) = frame.save(&path) {
                eprintln!("写帧失败: {e}");
            }
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

fn arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn demo_input(tick: usize) -> InputFrame {
    let phase = tick % 240;
    InputFrame {
        right: phase < 120,
        left: phase >= 120,
        jump: phase == 60 || phase == 180,
        attack: phase % 40 == 0,
        pick_up: phase % 50 == 0,
        ..Default::default()
    }
}

/// 占位渲染：深灰底 + 玩家位置标记（完整 macroquad 离屏渲染后续接入）。
fn render_placeholder(sim: &GameSim) -> RgbImage {
    let mut img = RgbImage::new(WINDOW_W as u32, WINDOW_H as u32);
    for p in img.pixels_mut() {
        *p = image::Rgb([30, 32, 40]);
    }
    let px = (sim.state.player.x - sim.state.cam_x).clamp(0.0, WINDOW_W - 4.0) as u32;
    let py = (sim.state.player.y - sim.state.cam_y).clamp(0.0, WORLD_VIEW_H - 4.0) as u32;
    for dy in 0..8 {
        for dx in 0..8 {
            if px + dx < WINDOW_W as u32 && py + dy < WINDOW_H as u32 {
                img.put_pixel(px + dx, py + dy, image::Rgb([80, 220, 255]));
            }
        }
    }
    img
}
