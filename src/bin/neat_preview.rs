//! NEAT 最优个体预览（真实 YOLO+OCR + 离屏渲染）。
//!
//! ```powershell
//! cargo run --release --bin neat_trainer -- --generations 20 --population 5
//! cargo run --release --bin neat_preview
//! cargo run --release --bin neat_preview -- --genome tmp/neat_session_best.json
//! ```

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

use macroquad::prelude::*;
use mxd_tools::game::action::input_label;
use mxd_tools::game::view;
use mxd_tools::game::{
    self, DeferredCaptureVision, GameSim, LOGIC_DT, VisionAnchorConfig, VisionPaceConfig,
    VisionPipeline, WINDOW_H, WINDOW_W, VISION_CONF_THRESH,
};
use mxd_tools::neat::{
    BestGenomeSnapshot, NeatDriver, DEFAULT_BEST_GENOME_FILE, DEFAULT_SESSION_BEST_FILE,
};
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
    pace: VisionPaceConfig,
    watch: bool,
    quiet: bool,
}

impl Cli {
    fn parse(args: &[String]) -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            genome: arg_path(args, "--genome").unwrap_or_else(|| {
                let session = manifest.join(DEFAULT_SESSION_BEST_FILE);
                if session.is_file() {
                    session
                } else {
                    manifest.join(DEFAULT_BEST_GENOME_FILE)
                }
            }),
            model: arg_path(args, "--model")
                .unwrap_or_else(|| manifest.join("onnx/yolo_nangang_e3000_best.onnx")),
            episode_seed: arg_u64(args, "--seed", 0),
            pace: VisionPaceConfig::from_detect_hz(arg_f32(args, "--detect-hz", 10.0)),
            watch: !args.iter().any(|a| a == "--no-watch"),
            quiet: args.iter().any(|a| a == "--quiet"),
        }
    }
}

struct PreviewState {
    snapshot: BestGenomeSnapshot,
    sim: GameSim,
    logic_tick: u32,
    episode_seed: u64,
    restart_cooldown: f32,
}

impl PreviewState {
    fn new(snapshot: BestGenomeSnapshot, map: &game::GameMap, episode_seed: u64) -> Self {
        let sim = GameSim::new_preview(map.clone(), episode_seed);
        Self {
            snapshot,
            sim,
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
        self.sim = GameSim::new_preview(map.clone(), seed);
        self.logic_tick = 0;
        self.restart_cooldown = 0.0;
    }
}

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1))
}

fn arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    arg_value(args, key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn arg_f32(args: &[String], key: &str, default: f32) -> f32 {
    arg_value(args, key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn arg_path(args: &[String], key: &str) -> Option<PathBuf> {
    arg_value(args, key).map(PathBuf::from)
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
            eprintln!("加载基因组失败 ({}): {e:#}", cli.genome.display());
            return;
        }
    };

    eprintln!(
        "NEAT 预览: fitness={:.2} gen={} detect_hz={:.1} genome={}",
        snapshot.fitness,
        snapshot.generation,
        cli.pace.detect_hz(),
        cli.genome.display()
    );

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

    let pipeline = match VisionPipeline::load(&cli.model, YoloDevice::Cpu, VISION_CONF_THRESH)
        .map(|p| p.with_anchor(VisionAnchorConfig::ocr()))
    {
        Ok(p) => p,
        Err(e) => {
            eprintln!("加载 YOLO 失败: {e}");
            return;
        }
    };

    let mut vision = DeferredCaptureVision::spawn(pipeline);
    let rt = view::new_render_target();
    let interval = cli.pace.vision_interval_ticks;

    let seed = if cli.episode_seed == 0 {
        snapshot.training_seed
    } else {
        cli.episode_seed
    };
    let mut state = PreviewState::new(snapshot.clone(), &map, seed);
    let mut driver = NeatDriver::new(state.snapshot.genome.clone());
    if let Err(e) = driver
        .bootstrap_vision_preview(&mut vision, &mut state.sim, &assets, &rt)
        .await
    {
        eprintln!("首帧 YOLO 失败: {e:#}");
        return;
    }

    let mut last_mtime = file_mtime(&cli.genome);
    let mut last_logged_vision: Option<u32> = None;

    loop {
        if cli.watch {
            if let Some(mtime) = file_mtime(&cli.genome) {
                if last_mtime.map(|t| t < mtime).unwrap_or(true) {
                    if let Ok(new_snap) = BestGenomeSnapshot::load(&cli.genome) {
                        if new_snap.fitness > state.snapshot.fitness + 0.01 {
                            eprintln!(
                                "热加载基因组: fitness {:.2} → {:.2}",
                                state.snapshot.fitness, new_snap.fitness
                            );
                            state.reload(new_snap.clone(), &map);
                            driver.set_genome(new_snap.genome);
                            let _ = driver
                                .bootstrap_vision_preview(
                                    &mut vision,
                                    &mut state.sim,
                                    &assets,
                                    &rt,
                                )
                                .await;
                        }
                    }
                    last_mtime = Some(mtime);
                }
            }
        }

        if is_key_pressed(KeyCode::R) {
            state.restart_episode(&map);
            driver.set_genome(state.snapshot.genome.clone());
            let _ = driver
                .bootstrap_vision_preview(&mut vision, &mut state.sim, &assets, &rt)
                .await;
        }

        if state.sim.is_episode_over() {
            state.restart_cooldown -= get_frame_time();
            if state.restart_cooldown <= 0.0 {
                state.restart_episode(&map);
                driver.set_genome(state.snapshot.genome.clone());
                let _ = driver
                    .bootstrap_vision_preview(&mut vision, &mut state.sim, &assets, &rt)
                    .await;
                state.restart_cooldown = 1.0;
            }
        } else {
            let tick = state.logic_tick;
            let tick_start = Instant::now();

            if let Ok(Some((vtick, _obs))) = driver.logic_tick_preview(
                &mut vision,
                &mut state.sim,
                tick,
                interval,
                &assets,
                &rt,
            ) {
                if !cli.quiet && last_logged_vision != Some(vtick) {
                    eprintln!(
                        "tick={vtick} input={} fitness={:.1} meso={} hp={}/{}",
                        input_label(&driver.input()),
                        state.snapshot.fitness,
                        state.sim.state.meso,
                        state.sim.state.player.hp,
                        state.sim.state.player.max_hp,
                    );
                    last_logged_vision = Some(vtick);
                }
            }
            if tick % interval == 0 {
                vision.schedule_draw_if_ready(tick, &state.sim, &assets, &rt);
            }
            if vision.capture_pending() {
                let _ = vision.flush_submit(&rt);
            }

            driver.tick_sim(&mut state.sim);
            state.logic_tick += 1;

            if let Some(rest) = Duration::from_secs_f32(LOGIC_DT).checked_sub(tick_start.elapsed())
            {
                std::thread::sleep(rest);
            }
        }

        clear_background(Color::new(0.05, 0.05, 0.08, 1.0));
        view::begin_logical_viewport();
        view::draw_content(&assets, &state.sim);
        view::draw_yolo_overlay(vision.last_detections(), VISION_CONF_THRESH);
        if let Some(hit) = vision.last_self_player() {
            view::draw_self_player_marker(hit);
        }
        set_default_camera();

        next_frame().await;
        let _ = vision.flush_submit(&rt);
    }
}
