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
    self, DeferredCaptureVision, GameSim, GameSimConfig, LOGIC_DT, VisionAnchorConfig,
    VisionPaceConfig,
    VisionPipeline, WINDOW_H, WINDOW_W, VISION_CONF_THRESH, OBS_PROPRIO_START,
    obs_climb_grab_ready, obs_enemy_in_attack_range, obs_floor_ahead, obs_has_drop,
    obs_has_ladder_or_rope_signal, obs_has_nearby_platform_enemy, obs_has_platform_enemy,
    obs_has_same_level_enemy, obs_jump_target_ahead, obs_nearest_same_level_enemy_px,
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
    /// 额外卡住诊断（OCR 位移/blocked、同层怪、近战等）。
    diag: bool,
    /// 开砍怪状态机 + 怪物伤害；默认只看寻路。
    combat: bool,
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
            diag: args.iter().any(|a| a == "--diag") || !args.iter().any(|a| a == "--no-diag"),
            combat: args.iter().any(|a| a == "--combat"),
        }
    }
}

struct PreviewState {
    snapshot: BestGenomeSnapshot,
    sim: GameSim,
    logic_tick: u32,
    episode_seed: u64,
    restart_cooldown: f32,
    combat: bool,
}

impl PreviewState {
    fn new(
        snapshot: BestGenomeSnapshot,
        map: &game::GameMap,
        episode_seed: u64,
        combat: bool,
    ) -> Self {
        let sim = Self::new_sim(map, episode_seed, combat);
        Self {
            snapshot,
            sim,
            logic_tick: 0,
            episode_seed,
            restart_cooldown: 0.0,
            combat,
        }
    }

    fn new_sim(map: &game::GameMap, seed: u64, combat: bool) -> GameSim {
        let cfg = GameSimConfig::neat_preview().with_mob_damage(combat);
        GameSim::new_with_config(map.clone(), seed, cfg)
    }

    fn reload(&mut self, snapshot: BestGenomeSnapshot, map: &game::GameMap) {
        let seed = if self.episode_seed == 0 {
            snapshot.training_seed
        } else {
            self.episode_seed
        };
        *self = Self::new(snapshot, map, seed, self.combat);
    }

    fn restart_episode(&mut self, map: &game::GameMap) {
        let seed = self.episode_seed.wrapping_add(self.logic_tick as u64);
        self.sim = Self::new_sim(map, seed, self.combat);
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

fn attack_facing(input: &mxd_tools::game::InputFrame, player_facing: f32) -> f32 {
    if input.right && !input.left {
        1.0
    } else if input.left && !input.right {
        -1.0
    } else {
        player_facing.signum()
    }
}

#[derive(Default)]
struct DiagTracker {
    last_hint: String,
    empty_attack_streak: u32,
    wall_press_streak: u32,
    no_near_streak: u32,
    mono_left_streak: u32,
    mono_right_streak: u32,
    last_pos: Option<(f32, f32)>,
    still_streak: u32,
}

impl DiagTracker {
    fn note(
        &mut self,
        hint: &str,
        input: &mxd_tools::game::InputFrame,
        melee: bool,
        near_plat: bool,
        blk_l: bool,
        blk_r: bool,
        x: f32,
        y: f32,
        tick: u32,
    ) {
        if input.attack && !melee {
            self.empty_attack_streak = self.empty_attack_streak.saturating_add(1);
        } else {
            self.empty_attack_streak = 0;
        }
        let press_wall = (input.left && !input.right && blk_l)
            || (input.right && !input.left && blk_r);
        if press_wall {
            self.wall_press_streak = self.wall_press_streak.saturating_add(1);
        } else {
            self.wall_press_streak = 0;
        }
        if !near_plat {
            self.no_near_streak = self.no_near_streak.saturating_add(1);
        } else {
            self.no_near_streak = 0;
        }
        if input.left && !input.right {
            self.mono_left_streak = self.mono_left_streak.saturating_add(1);
            self.mono_right_streak = 0;
        } else if input.right && !input.left {
            self.mono_right_streak = self.mono_right_streak.saturating_add(1);
            self.mono_left_streak = 0;
        } else {
            self.mono_left_streak = 0;
            self.mono_right_streak = 0;
        }
        if let Some((px, py)) = self.last_pos {
            let moved = (x - px).abs() + (y - py).abs() > 8.0;
            self.still_streak = if moved {
                0
            } else {
                self.still_streak.saturating_add(1)
            };
        }
        self.last_pos = Some((x, y));

        if hint != self.last_hint {
            eprintln!("  EVENT tick={tick} hint {hint} (was {})", self.last_hint);
            self.last_hint = hint.to_string();
        }
        if self.wall_press_streak > 0 && self.wall_press_streak % 60 == 0 {
            eprintln!(
                "  EVENT tick={tick} wall_press_streak={} (OCR卡住硬顶)",
                self.wall_press_streak
            );
        }
        if self.empty_attack_streak > 0 && self.empty_attack_streak % 60 == 0 {
            eprintln!(
                "  EVENT tick={tick} empty_attack_streak={} (非近战仍砍)",
                self.empty_attack_streak
            );
        }
        if self.still_streak > 0 && self.still_streak % 120 == 0 {
            eprintln!(
                "  EVENT tick={tick} still_streak={} pos=({x:.0},{y:.0})",
                self.still_streak
            );
        }
    }
}

/// 预览卡住诊断：YOLO/OCR 观测 + 按键 + 训练适应度快照。
fn format_stuck_diag(
    obs: &[f32; mxd_tools::game::OBS_DIM],
    input: &mxd_tools::game::InputFrame,
    sim: &GameSim,
    tracker: &DiagTracker,
) -> (String, String, String, String) {
    let p = OBS_PROPRIO_START;
    let ocr_dx = obs.get(p).copied().unwrap_or(0.0) * WINDOW_W;
    let ocr_dy = obs.get(p + 1).copied().unwrap_or(0.0) * WINDOW_H;
    let blk_l = obs.get(p + 2).copied().unwrap_or(0.0) >= 0.5;
    let blk_r = obs.get(p + 3).copied().unwrap_or(0.0) >= 0.5;
    let facing = attack_facing(input, sim.state.player.facing);
    let same_lv = obs_has_same_level_enemy(obs);
    let plat_lv = obs_has_platform_enemy(obs);
    let near_plat = obs_has_nearby_platform_enemy(obs);
    let melee = obs_enemy_in_attack_range(obs, facing);
    let rope = obs_has_ladder_or_rope_signal(obs);
    let climb_ready = obs_climb_grab_ready(obs);
    let jump_ok = obs_jump_target_ahead(obs, facing, WINDOW_W, WINDOW_H);
    let floor_l = obs_floor_ahead(obs, -1.0);
    let floor_r = obs_floor_ahead(obs, 1.0);
    let drop = obs_has_drop(obs);
    let near = obs_nearest_same_level_enemy_px(obs, WINDOW_W, WINDOW_H);
    let near_s = match near {
        Some((dx, dy)) => format!("enemy≈({dx:.0},{dy:.0})"),
        None => "enemy=none".into(),
    };
    let moving = (input.left && !input.right) || (input.right && !input.left);
    let climbing = sim.state.player.climbing;
    let hint = if climbing {
        "HINT=climbing"
    } else if moving && (blk_l || blk_r) {
        "HINT=ocr_blocked_still_pressing"
    } else if !near_plat && plat_lv {
        "HINT=far_platform_enemy"
    } else if !plat_lv && same_lv {
        "HINT=other_platform_enemy"
    } else if input.attack && !melee {
        "HINT=empty_attack"
    } else if input.attack && !moving && melee {
        "HINT=stand_melee_ok"
    } else if !moving && !input.jump && !input.up {
        "HINT=no_horizontal"
    } else if input.jump && !moving {
        "HINT=jump_in_place"
    } else if input.up && !rope && !climb_ready && !climbing {
        "HINT=up_without_rope"
    } else {
        "HINT=ok"
    };
    let fit = sim.fitness.preview_diag();
    let line1 = format!(
        "pos=({:.0},{:.0}) face={} climb={} ocrΔ=({:.1},{:.1}) blkL={} blkR={} near={} plat={} wide={} melee={} {near_s} {hint}",
        sim.state.player.x,
        sim.state.player.y,
        if facing >= 0.0 { "R" } else { "L" },
        climbing as u8,
        ocr_dx,
        ocr_dy,
        blk_l as u8,
        blk_r as u8,
        near_plat as u8,
        plat_lv as u8,
        same_lv as u8,
        melee as u8,
    );
    let line2 = format!(
        "nav rope={} grab={} jumpOk={} floorL={} floorR={} drop={} streaks empty={} wall={} noNear={} monoL={} monoR={} still={}",
        rope as u8,
        climb_ready as u8,
        jump_ok as u8,
        floor_l as u8,
        floor_r as u8,
        drop as u8,
        tracker.empty_attack_streak,
        tracker.wall_press_streak,
        tracker.no_near_streak,
        tracker.mono_left_streak,
        tracker.mono_right_streak,
        tracker.still_streak,
    );
    let line3 = format!(
        "fit score={:.1} explore={:.0} stall={} cells={} yBands={} leftSpawn={} forfeit={} kills={} mesoEv={}",
        fit.score,
        fit.explore_score,
        fit.explore_stall_ticks,
        fit.cells,
        fit.y_bands,
        fit.left_spawn as u8,
        fit.idle_forfeit as u8,
        fit.kills,
        fit.meso_events,
    );
    (hint.to_string(), line1, line2, line3)
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
        "NEAT 预览: rank_fit={:.2} gen={} detect_hz={:.1} genome={} diag={} mode={}",
        snapshot.fitness,
        snapshot.generation,
        cli.pace.detect_hz(),
        cli.genome.display(),
        cli.diag,
        if cli.combat { "combat" } else { "nav" },
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
    let mut state = PreviewState::new(snapshot.clone(), &map, seed, cli.combat);
    let mut driver = NeatDriver::new(state.snapshot.genome.clone()).with_combat(cli.combat);
    if let Err(e) = driver
        .bootstrap_vision_preview(&mut vision, &mut state.sim, &assets, &rt)
        .await
    {
        eprintln!("首帧 YOLO 失败: {e:#}");
        return;
    }

    let mut last_mtime = file_mtime(&cli.genome);
    let mut last_logged_vision: Option<u32> = None;
    let mut diag_tracker = DiagTracker::default();
    let mut last_forfeit = false;

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
                            diag_tracker = DiagTracker::default();
                            last_forfeit = false;
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
            diag_tracker = DiagTracker::default();
            last_forfeit = false;
            let _ = driver
                .bootstrap_vision_preview(&mut vision, &mut state.sim, &assets, &rt)
                .await;
        }

        if state.sim.is_episode_over() {
            if cli.diag && !last_forfeit {
                let fit = state.sim.fitness.preview_diag();
                if fit.idle_forfeit {
                    eprintln!(
                        "EVENT episode_end FORFEIT(explore_stall) tick={} score={:.1} cells={} yBands={} hp={}",
                        state.logic_tick,
                        fit.score,
                        fit.cells,
                        fit.y_bands,
                        state.sim.state.player.hp,
                    );
                } else {
                    eprintln!(
                        "EVENT episode_end tick={} score={:.1} hp={} cells={} yBands={}",
                        state.logic_tick,
                        fit.score,
                        state.sim.state.player.hp,
                        fit.cells,
                        fit.y_bands,
                    );
                }
                last_forfeit = true;
            }
            state.restart_cooldown -= get_frame_time();
            if state.restart_cooldown <= 0.0 {
                state.restart_episode(&map);
                driver.set_genome(state.snapshot.genome.clone());
                diag_tracker = DiagTracker::default();
                last_forfeit = false;
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
                    let inp = driver.input();
                    let fit = state.sim.fitness.preview_diag();
                    eprintln!(
                        "tick={vtick} act={} input={} live_fit={:.1} rank_fit={:.1} meso={} hp={}/{}",
                        if driver.combat_active() {
                            "COMBAT"
                        } else {
                            driver.current_action().map(|a| a.name()).unwrap_or("-")
                        },
                        input_label(&inp),
                        fit.score,
                        state.snapshot.fitness,
                        state.sim.state.meso,
                        state.sim.state.player.hp,
                        state.sim.state.player.max_hp,
                    );
                    if cli.diag {
                        let obs = driver.last_obs();
                        let facing = attack_facing(&inp, state.sim.state.player.facing);
                        let p = OBS_PROPRIO_START;
                        let blk_l = obs.get(p + 2).copied().unwrap_or(0.0) >= 0.5;
                        let blk_r = obs.get(p + 3).copied().unwrap_or(0.0) >= 0.5;
                        let near_plat = obs_has_nearby_platform_enemy(obs);
                        let melee = obs_enemy_in_attack_range(obs, facing);
                        // 先用当前 tracker 算 hint，更新 streak，再输出含最新 streak 的三行。
                        let (hint, _, _, _) =
                            format_stuck_diag(obs, &inp, &state.sim, &diag_tracker);
                        diag_tracker.note(
                            &hint,
                            &inp,
                            melee,
                            near_plat,
                            blk_l,
                            blk_r,
                            state.sim.state.player.x,
                            state.sim.state.player.y,
                            vtick,
                        );
                        let (_, l1, l2, l3) =
                            format_stuck_diag(obs, &inp, &state.sim, &diag_tracker);
                        eprintln!("  diag {l1}");
                        eprintln!("  diag {l2}");
                        eprintln!("  diag {l3}");
                    }
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
