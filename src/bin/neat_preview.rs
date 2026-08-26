//! 可视化回放 NEAT 最优个体（与 `neat_trainer` 独立进程，可边训练边预览）。
//!
//! ```powershell
//! # 训练（另开终端）
//! cargo run --release --bin neat_trainer -- --fresh --generations 100 --population 50 --pace 4
//!
//! # 预览最优个体（热加载 tmp/neat_best_genome.json）
//! cargo run --release --bin neat_preview
//! cargo run --release --bin neat_preview -- --pace 4 --no-watch
//! cargo run --release --bin neat_preview -- --pace 1   # 全帧感知，CPU 上视觉线程仍慢但不卡模拟
//! ```

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

use macroquad::prelude::*;
use mxd_tools::game::view;
use mxd_tools::game::{
    self, action_to_input, Action, GameSim, LOGIC_DT, NEAT_CONF_THRESH, TrainingPaceConfig,
    VisionPipeline, WINDOW_H, WINDOW_W,
};
use mxd_tools::neat::{BestGenomeSnapshot, DEFAULT_BEST_GENOME_FILE};
use mxd_tools::trainer::AgentController;
use mxd_tools::yolo::YoloDevice;

fn window_conf() -> Conf {
    Conf {
        window_title: "NEAT 最优个体预览".to_owned(),
        window_width: (WINDOW_W / 3.0).round() as i32,
        window_height: (WINDOW_H / 3.0).round() as i32,
        window_resizable: true,
        high_dpi: true,
        ..Default::default()
    }
}

struct Cli {
    genome: PathBuf,
    model: PathBuf,
    episode_seed: u64,
    pace: TrainingPaceConfig,
    watch: bool,
}

impl Cli {
    fn parse(args: &[String]) -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let pace_ticks = arg_u32(args, "--pace", 4);
        Self {
            genome: arg_path(args, "--genome")
                .unwrap_or_else(|| manifest.join(DEFAULT_BEST_GENOME_FILE)),
            model: arg_path(args, "--model")
                .unwrap_or_else(|| manifest.join("models/yolo_nangang_e3000_best.onnx")),
            episode_seed: arg_u64(args, "--seed", 0),
            pace: TrainingPaceConfig {
                vision_interval_ticks: pace_ticks.max(1),
            },
            watch: !args.iter().any(|a| a == "--no-watch"),
        }
    }
}

struct PreviewState {
    snapshot: BestGenomeSnapshot,
    sim: GameSim,
    last_action: Action,
    logic_tick: u32,
    episode_seed: u64,
    restart_cooldown: f32,
}

impl PreviewState {
    fn new(snapshot: BestGenomeSnapshot, map: &game::GameMap, episode_seed: u64) -> Self {
        let sim = GameSim::new_training(map.clone(), episode_seed);
        Self {
            snapshot,
            sim,
            last_action: Action::Noop,
            logic_tick: 0,
            episode_seed,
            restart_cooldown: 0.0,
        }
    }

    fn reload(&mut self, snapshot: BestGenomeSnapshot, map: &game::GameMap) {
        let seed = if self.episode_seed == 0 {
            snapshot.training_seed
        } else {
            self.episode_seed
        };
        *self = Self::new(snapshot, map, seed);
    }

    fn restart_episode(&mut self, map: &game::GameMap) {
        let seed = self.episode_seed.wrapping_add(self.logic_tick as u64);
        self.sim = GameSim::new_training(map.clone(), seed);
        self.last_action = Action::Noop;
        self.logic_tick = 0;
        self.restart_cooldown = 0.0;
    }
}

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1))
}

fn arg_u32(args: &[String], key: &str, default: u32) -> u32 {
    arg_value(args, key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    arg_value(args, key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn arg_path(args: &[String], key: &str) -> Option<PathBuf> {
    arg_value(args, key).map(PathBuf::from)
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

fn file_mtime(path: &PathBuf) -> Option<SystemTime> {
    fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

#[macroquad::main(window_conf)]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let cli = Cli::parse(&args);

    let snapshot = match BestGenomeSnapshot::load(&cli.genome) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("加载最优基因组失败: {e}");
            eprintln!("请先运行 neat_trainer，或指定 --genome 路径");
            return;
        }
    };

    eprintln!(
        "预览: fitness={:.2} gen={} pace={} watch={} file={}",
        snapshot.fitness,
        snapshot.generation,
        cli.pace.vision_interval_ticks,
        cli.watch,
        cli.genome.display()
    );
    eprintln!("视觉+NEAT 在后台线程运行，游戏模拟与绘制保持 60Hz");

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

    let pipeline = match VisionPipeline::load(&cli.model, YoloDevice::Cpu, NEAT_CONF_THRESH) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("加载 YOLO 失败: {e}");
            return;
        }
    };

    let episode_seed = if cli.episode_seed == 0 {
        snapshot.training_seed
    } else {
        cli.episode_seed
    };

    let mut state = PreviewState::new(snapshot.clone(), &map, episode_seed);
    let mut agent = AgentController::spawn(pipeline, snapshot.genome);
    let rt = view::new_render_target();
    let mut acc = 0.0_f32;
    let mut file_mtime_cached = file_mtime(&cli.genome);
    let mut frame_counter = 0u32;
    let interval = cli.pace.vision_interval_ticks;
    let mut next_vision_at: u32 = 0;
    let mut pending_vision_capture = false;
    let mut pending_vision_tick: u32 = 0;

    loop {
        let dt = get_frame_time();
        acc += dt;

        if cli.watch && frame_counter % 30 == 0 {
            if let Some(mtime) = file_mtime(&cli.genome) {
                if file_mtime_cached.map(|t| t < mtime).unwrap_or(true) {
                    if let Ok(new_snap) = BestGenomeSnapshot::load(&cli.genome) {
                        if new_snap.fitness > state.snapshot.fitness
                            || new_snap.updated_at != state.snapshot.updated_at
                        {
                            eprintln!(
                                "热加载最优基因组: fitness={:.2} gen={}",
                                new_snap.fitness, new_snap.generation
                            );
                            agent.set_genome(new_snap.genome.clone());
                            state.reload(new_snap, &map);
                            next_vision_at = 0;
                            pending_vision_capture = false;
                        }
                    }
                    file_mtime_cached = Some(mtime);
                }
            }
        }
        frame_counter = frame_counter.wrapping_add(1);

        if is_key_pressed(KeyCode::R) {
            state.restart_episode(&map);
            next_vision_at = 0;
            pending_vision_capture = false;
        }

        agent.poll(&mut state.sim);

        if state.sim.is_episode_over() {
            state.restart_cooldown -= dt;
            if state.restart_cooldown <= 0.0 {
                state.restart_episode(&map);
                next_vision_at = 0;
                pending_vision_capture = false;
                state.restart_cooldown = 1.5;
            }
        } else {
            if state.logic_tick >= next_vision_at {
                view::draw_to_render_target(&assets, &state.sim, &rt);
                pending_vision_capture = true;
                pending_vision_tick = state.logic_tick;
                next_vision_at = state.logic_tick.saturating_add(interval);
            }

            while acc >= LOGIC_DT {
                let action = agent.action();
                state.last_action = action;
                state.sim.tick(&action_to_input(action));
                state.logic_tick = state.logic_tick.wrapping_add(1);
                acc -= LOGIC_DT;

                if state.sim.is_episode_over() {
                    state.restart_cooldown = 1.5;
                    break;
                }
            }
        }

        if agent.worker_dead() {
            eprintln!("视觉线程已退出，预览结束");
            break;
        }

        clear_background(Color::new(0.05, 0.05, 0.08, 1.0));
        begin_logical_viewport();
        view::draw_content(&assets, &state.sim);
        if let Some(step) = agent.vision() {
            view::draw_yolo_overlay(&step.detections, NEAT_CONF_THRESH);
            if let Some(hit) = step.self_player.as_ref() {
                view::draw_self_player_marker(hit);
            }
        }
        set_default_camera();
        draw_hud(&state, &cli, agent.action());

        next_frame().await;

        if pending_vision_capture {
            let rgb = view::render_target_to_rgb(&rt);
            agent.try_submit_frame(pending_vision_tick, rgb);
            pending_vision_capture = false;
        }
    }
}

fn draw_hud(state: &PreviewState, cli: &Cli, action: Action) {
    let over = state.sim.is_episode_over();
    let lines = [
        format!("NEAT 预览  fitness={:.1}", state.snapshot.fitness),
        format!(
            "代={}  tick={}  action={}  score={:.1}",
            state.snapshot.generation,
            state.logic_tick,
            action.label(),
            state.sim.fitness.score
        ),
        format!(
            "pace={}  watch={}  [R]重开  (视觉后台)",
            cli.pace.vision_interval_ticks,
            cli.watch
        ),
    ];
    for (i, line) in lines.iter().enumerate() {
        draw_text(line, 12.0, 22.0 + i as f32 * 20.0, 18.0, WHITE);
    }
    if over {
        draw_text(
            "Game Over — 即将重开",
            12.0,
            screen_height() - 28.0,
            20.0,
            Color::new(1.0, 0.35, 0.35, 1.0),
        );
    }
}
