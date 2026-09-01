//! 规则 bot 自动玩预览（YOLO + OCR + 纯规则策略）。
//!
//! ```powershell
//! cargo run --release --bin game_preview
//! cargo run --release --bin game_preview -- --detect-hz 10
//! cargo run --release --bin game_preview -- --quiet
//! cargo run --release --bin game_preview -- --probe first_platform
//! cargo test --release --test game_preview_first_platform
//! ```
//!
//! 默认 `--detect-hz 10`（约每 6 逻辑帧一次感知，对齐 ~100ms 截图+推理）。
use std::collections::HashSet;
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use macroquad::prelude::*;
use mxd_tools::game::action::input_label;
use mxd_tools::game::view;
use mxd_tools::game::{
    self, default_yolo_model_path, evaluate_first_platform_report,
    format_first_platform_preview_done, DeferredCaptureVision, FirstPlatformTracker, GameSim,
    InputFrame, ProbeDriver, VisionAnchorConfig, VisionPaceConfig, VisionPipeline,
    FIRST_PLATFORM_PROBE_TICKS, LOGIC_DT, OBS_FLOOR_SLOTS, OBS_FLOOR_START, OBS_SLOT_DIM,
    VISION_CONF_THRESH, WINDOW_H, WINDOW_W,
};
use mxd_tools::yolo::YoloDevice;

fn window_conf() -> Conf {
    Conf {
        window_title: "规则 Bot 自动玩".to_owned(),
        window_width: (WINDOW_W / 3.0).round() as i32,
        window_height: (WINDOW_H / 3.0).round() as i32,
        window_resizable: true,
        high_dpi: true,
        ..Default::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreviewProbe {
    FirstPlatform,
}

impl PreviewProbe {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "first_platform" => Some(Self::FirstPlatform),
            _ => None,
        }
    }
}

struct Cli {
    model: PathBuf,
    episode_seed: u64,
    pace: VisionPaceConfig,
    quiet: bool,
    anchor_offset: f32,
    auto_ticks: u32,
    probe: Option<PreviewProbe>,
}

impl Cli {
    fn parse(args: &[String]) -> Self {
        if args.iter().any(|a| a == "--pace") {
            eprintln!(
                "错误: --pace 已废弃，请改用 --detect-hz（次/秒）。\n  \
                 对照: 原 --pace 12 ≈ --detect-hz 5；原 --pace 6 ≈ --detect-hz 10；原 --pace 1 ≈ --detect-hz 60"
            );
            std::process::exit(2);
        }
        let detect_hz = arg_f32(args, "--detect-hz", 10.0);
        let probe = args
            .iter()
            .position(|a| a == "--probe")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| PreviewProbe::parse(s));
        let auto_ticks = if probe == Some(PreviewProbe::FirstPlatform) {
            FIRST_PLATFORM_PROBE_TICKS
        } else {
            arg_u64(args, "--auto-ticks", 0) as u32
        };
        Self {
            model: arg_path(args, "--model").unwrap_or_else(default_yolo_model_path),
            episode_seed: arg_u64(args, "--seed", 0),
            pace: VisionPaceConfig::from_detect_hz(detect_hz),
            quiet: probe.is_some() || args.iter().any(|a| a == "--quiet"),
            anchor_offset: arg_f32(args, "--anchor-offset", 0.0),
            auto_ticks,
            probe,
        }
    }

    fn vision_anchor(&self) -> VisionAnchorConfig {
        if self.anchor_offset > 0.0 {
            VisionAnchorConfig::ocr_with_jitter(self.anchor_offset)
        } else {
            VisionAnchorConfig::ocr()
        }
    }
}

struct PreviewState {
    sim: GameSim,
    logic_tick: u32,
    episode_seed: u64,
    restart_cooldown: f32,
    last_logged_vision_tick: Option<u32>,
}

impl PreviewState {
    fn new(map: &game::GameMap, episode_seed: u64) -> Self {
        let sim = GameSim::new_preview(map.clone(), episode_seed);
        Self {
            sim,
            logic_tick: 0,
            episode_seed,
            restart_cooldown: 0.0,
            last_logged_vision_tick: None,
        }
    }

    fn restart_episode(&mut self, map: &game::GameMap) {
        let seed = self.episode_seed.wrapping_add(self.logic_tick as u64);
        self.sim = GameSim::new_preview(map.clone(), seed);
        self.logic_tick = 0;
        self.restart_cooldown = 0.0;
    }
}

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
}

fn arg_u64(args: &[String], key: &str, default: u64) -> u64 {
    arg_value(args, key)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_f32(args: &[String], key: &str, default: f32) -> f32 {
    arg_value(args, key)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_path(args: &[String], key: &str) -> Option<PathBuf> {
    arg_value(args, key).map(PathBuf::from)
}

struct AutoProbe {
    limit: u32,
    x_cells: HashSet<i32>,
    decision_n: u32,
    intend_move_n: u32,
    max_still_streak: u32,
    still_streak: u32,
    last_x_cell: Option<i32>,
    sample_n: u32,
}

impl AutoProbe {
    fn new(limit: u32) -> Self {
        Self {
            limit,
            x_cells: HashSet::new(),
            decision_n: 0,
            intend_move_n: 0,
            max_still_streak: 0,
            still_streak: 0,
            last_x_cell: None,
            sample_n: 0,
        }
    }

    fn on_tick(&mut self, x: f32) {
        let cell = (x / 48.0).floor() as i32;
        self.x_cells.insert(cell);
        self.sample_n += 1;
        if self.last_x_cell == Some(cell) {
            self.still_streak += 1;
        } else {
            self.still_streak = 1;
            self.last_x_cell = Some(cell);
        }
        self.max_still_streak = self.max_still_streak.max(self.still_streak);
    }

    fn on_decision(&mut self, intended: &InputFrame) {
        self.decision_n += 1;
        if intended.left || intended.right || intended.jump {
            self.intend_move_n += 1;
        }
    }

    fn finish(&self, sim: &GameSim) -> bool {
        let unique_x = self.x_cells.len();
        let move_ratio = if self.decision_n > 0 {
            self.intend_move_n as f32 / self.decision_n as f32
        } else {
            0.0
        };
        let stagnant = self.sample_n >= self.limit.saturating_mul(8) / 10
            && self.decision_n >= 20
            && (unique_x <= 4 || self.max_still_streak >= 900 || move_ratio < 0.1);
        eprintln!(
            "AUTO_PROBE ticks={} unique_x={} max_still={} decisions={} move_ratio={:.2} meso={} kills={} hp={}/{}",
            self.sample_n,
            unique_x,
            self.max_still_streak,
            self.decision_n,
            move_ratio,
            sim.state.meso,
            sim.state.kills,
            sim.state.player.hp,
            sim.state.player.max_hp,
        );
        if stagnant {
            eprintln!("AUTO_VERDICT=STAGNANT");
            println!(
                "PREVIEW_DONE verdict=STAGNANT meso={} kills={}",
                sim.state.meso, sim.state.kills
            );
        } else {
            eprintln!("AUTO_VERDICT=OK");
            println!(
                "PREVIEW_DONE verdict=OK meso={} kills={}",
                sim.state.meso, sim.state.kills
            );
        }
        let _ = std::io::Write::flush(&mut std::io::stdout());
        stagnant
    }
}

fn finish_first_platform_probe(tracker: FirstPlatformTracker, sim: &GameSim) -> ! {
    let report = tracker.finish(sim);
    eprintln!(
        "first_platform 结果: kills={} end=({:.0},{:.0}) min_x_after={:.0} max_x_after={:.0} y_changed={} left={}",
        report.kills,
        report.end_x,
        report.end_y,
        report.min_x_after_kills,
        report.max_x_after_kills,
        report.y_band_changed,
        report.left_first_platform(),
    );
    let eval = evaluate_first_platform_report(&report);
    let pass = eval.is_ok();
    println!("{}", format_first_platform_preview_done(&report, pass));
    let _ = std::io::Write::flush(&mut std::io::stdout());
    if let Err(msg) = eval {
        eprintln!("探针断言失败: {msg}");
        std::process::exit(1);
    }
    eprintln!("探针通过: FirstPlatform");
    std::process::exit(0);
}

async fn run_first_platform_probe_loop(
    cli: &Cli,
    _map: &game::GameMap,
    assets: &view::GameViewAssets,
    mut state: PreviewState,
    mut vision: DeferredCaptureVision,
    mut driver: ProbeDriver,
    rt: &RenderTarget,
    interval: u32,
    mut fp_tracker: FirstPlatformTracker,
) {
    for tick in 0..cli.auto_ticks {
        if state.sim.is_episode_over() {
            break;
        }
        if vision.worker_dead() {
            eprintln!("视觉线程已退出");
            finish_first_platform_probe(fp_tracker, &state.sim);
        }

        let mut just_decided: Option<u32> = None;
        if let Some((vtick, obs)) = vision.poll_observation(&state.sim) {
            driver.apply_observation(&mut state.sim, vtick, obs);
            if state.last_logged_vision_tick != Some(vtick) {
                just_decided = Some(vtick);
                state.last_logged_vision_tick = Some(vtick);
            }
        }
        if tick % interval == 0 {
            let _ = vision
                .schedule_capture_if_ready(tick, &state.sim, assets, rt)
                .await;
        }

        let tick_start = Instant::now();
        // 先 pace（内含 melee 刷新），再取意图，日志与 sim 一致。
        let input = driver.paced_input_for_sim(&state.sim, tick);
        let intended = driver.input();
        if let Some(vtick) = just_decided {
            let effective = state.sim.effective_bot_input(&input);
            if !cli.quiet {
                log_decision(
                    vtick,
                    &intended,
                    &effective,
                    &state.sim,
                    &driver,
                    vision.last_detections(),
                );
            }
        }
        state.sim.tick(&input);
        fp_tracker.on_tick(&state.sim);
        if let Some(rest) = Duration::from_secs_f32(LOGIC_DT).checked_sub(tick_start.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    finish_first_platform_probe(fp_tracker, &state.sim);
}

fn log_decision(
    tick: u32,
    intended: &InputFrame,
    effective: &InputFrame,
    sim: &GameSim,
    driver: &ProbeDriver,
    detections: &[mxd_tools::yolo::Detection],
) {
    use mxd_tools::game::{
        obs_enemy_in_attack_range, obs_farm_band_enemies, obs_floor_ahead_connected,
        obs_floor_drop_ahead, obs_has_same_level_enemy, obs_nearest_same_level_enemy_px,
        obs_step_up_dx, RuleBotCtx,
    };

    let p = &sim.state.player;
    let bot = driver.bot();
    let sense = driver.sense();
    let obs = driver.last_obs();
    let ctx = RuleBotCtx::from_vision(obs, sense);
    let (mob_dx, mob_dy, mob_dir) = ctx
        .engage
        .map(|e| (e.dx, e.dy, e.mob_dir))
        .unwrap_or((0.0, 0.0, 0.0));
    let alive = sim.state.mobs.iter().filter(|m| m.alive).count();
    let iw = WINDOW_W as f32;
    let ih = WINDOW_H as f32;
    let pr = Some(obs_floor_ahead_connected(obs, 1.0));
    let pl = Some(obs_floor_ahead_connected(obs, -1.0));
    let pdr = Some(obs_floor_drop_ahead(obs, 1.0));
    let pdl = Some(obs_floor_drop_ahead(obs, -1.0));
    let farm_local = obs_farm_band_enemies(obs, iw, 260.0);
    let farm_y_any = farm_local;
    let (visual_dx, visual_dy) = sense.visual_delta();
    let node = sense.location_node();
    let (net, path, span_x, span_y) = bot.progress_metrics();
    let _ = obs_nearest_same_level_enemy_px(obs, iw, ih);
    eprintln!(
        "BOT sense=yolo+ocr ctx=vision tick={} intent={} effective={} reason={} loop={} escape={} candidate={} failed_exits={} seek={} flips={} farm_local={} farm_y_any={} perch={} sim_kills={} alive={} sim_pos=({:.0},{:.0}) est_pos=({:.0},{:.0}) visual_delta=({:.1},{:.1}) visual_conf={} node=({},{},{}) progress=net:{:.0}/path:{:.0}/span:{:.0}x{:.0} engage_dx={:.0} dy={:.0} dir={:.0} walkR={:?} walkL={:?} dropR={:?} dropL={:?} step={:?} cliffR={} cliffL={}",
        tick,
        input_label(intended),
        input_label(effective),
        bot.last_reason,
        bot.loop_kind_name(),
        bot.escape_phase_name(),
        bot.escape_candidate_name(),
        bot.failed_exit_count(),
        bot.explore_seeking_vertical(),
        bot.dir_flip_streak_pub(),
        farm_local,
        farm_y_any,
        bot.perching,
        sim.state.kills,
        alive,
        p.x,
        p.y,
        sense.est_x,
        sense.est_y,
        visual_dx,
        visual_dy,
        sense.visual_confidence(),
        node.x,
        node.y,
        node.terrain,
        net,
        path,
        span_x,
        span_y,
        mob_dx,
        mob_dy,
        mob_dir,
        pr,
        pl,
        pdr,
        pdl,
        obs_step_up_dx(obs, iw, ih),
        pr == Some(false),
        pl == Some(false),
    );
    let noop = !intended.left
        && !intended.right
        && !intended.jump
        && !intended.attack
        && !intended.up
        && !intended.down
        && !effective.left
        && !effective.right
        && !effective.jump
        && !effective.attack
        && !effective.up
        && !effective.down;
    if !noop {
        return;
    }
    eprintln!(
        "  NOOP diag: facing={} strike={} footing={} farm_y={:.0}",
        if sense.facing >= 0.0 { "R" } else { "L" },
        obs_enemy_in_attack_range(obs, sense.facing),
        obs_has_same_level_enemy(obs),
        bot.farm_y,
    );
    // 最近地板检测框（像素，视口坐标）+ obs 归一化槽
    let mut floors: Vec<&mxd_tools::yolo::Detection> = detections
        .iter()
        .filter(|d| d.class_id == 0 && d.conf >= VISION_CONF_THRESH)
        .collect();
    floors.sort_by(|a, b| {
        let da = ((a.x1 + a.x2) * 0.5 - WINDOW_W * 0.5).abs()
            + ((a.y1 + a.y2) * 0.5 - WINDOW_H * 0.5).abs();
        let db = ((b.x1 + b.x2) * 0.5 - WINDOW_W * 0.5).abs()
            + ((b.y1 + b.y2) * 0.5 - WINDOW_H * 0.5).abs();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(f) = floors.first() {
        eprintln!(
            "  NOOP floor_yolo: n={} nearest conf={:.2} box=({:.0},{:.0})-({:.0},{:.0}) size={:.0}x{:.0}",
            floors.len(),
            f.conf,
            f.x1,
            f.y1,
            f.x2,
            f.y2,
            f.x2 - f.x1,
            f.y2 - f.y1,
        );
    } else {
        eprintln!("  NOOP floor_yolo: n=0 (no floor det above conf)");
    }
    let obs = driver.last_obs();
    let mut printed = 0u32;
    for i in 0..OBS_FLOOR_SLOTS {
        let base = OBS_FLOOR_START + i * OBS_SLOT_DIM;
        if base + 4 > obs.len() {
            break;
        }
        let (dx, dy, w, h) = (obs[base], obs[base + 1], obs[base + 2], obs[base + 3]);
        if w.abs() <= 1e-4 && h.abs() <= 1e-4 {
            continue;
        }
        eprintln!(
            "  NOOP floor_obs[{i}]: dx={dx:.3} dy={dy:.3} w={w:.3} h={h:.3} (px dx={:.0} dy={:.0} w={:.0} h={:.0})",
            dx * WINDOW_W,
            dy * WINDOW_H,
            w * WINDOW_W,
            h * WINDOW_H,
        );
        printed += 1;
        if printed >= 3 {
            break;
        }
    }
    if printed == 0 {
        eprintln!("  NOOP floor_obs: empty");
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let cli = Cli::parse(&args);

    eprintln!(
        "规则 Bot 预览 detect-hz={:.1} seed={} quiet={}",
        cli.pace.detect_hz(),
        cli.episode_seed,
        cli.quiet
    );
    if let Some(probe) = cli.probe {
        eprintln!(
            "自动探针: --probe={probe:?} ticks={} (~{:.0}s 墙钟)",
            cli.auto_ticks,
            cli.auto_ticks as f32 / 60.0
        );
    } else if cli.auto_ticks > 0 {
        eprintln!("自动探针: --auto-ticks={}", cli.auto_ticks);
    } else {
        eprintln!("按 R 重开本局；预览模式受击不死");
    }

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
        .map(|p| p.with_anchor(cli.vision_anchor()))
    {
        Ok(p) => p,
        Err(e) => {
            eprintln!("加载 YOLO 失败: {e}");
            return;
        }
    };

    let episode_seed = cli.episode_seed;
    let mut state = PreviewState::new(&map, episode_seed);
    let mut vision = DeferredCaptureVision::spawn(pipeline);
    let mut driver = if cli.probe.is_some() {
        ProbeDriver::new(episode_seed)
    } else {
        ProbeDriver::new_realtime(episode_seed)
    };
    let rt = view::new_render_target();
    let interval = cli.pace.vision_interval_ticks;
    let mut auto_probe =
        (cli.auto_ticks > 0 && cli.probe.is_none()).then(|| AutoProbe::new(cli.auto_ticks));
    let mut fp_tracker = cli
        .probe
        .map(|_| FirstPlatformTracker::new(state.sim.state.player.x, state.sim.state.player.y));

    if let Err(e) = driver
        .bootstrap_vision_preview(&mut vision, &mut state.sim, &assets, &rt)
        .await
    {
        eprintln!("首帧 YOLO 失败: {e}");
        return;
    }

    if cli.probe == Some(PreviewProbe::FirstPlatform) {
        let tracker = fp_tracker.expect("first_platform tracker");
        run_first_platform_probe_loop(
            &cli, &map, &assets, state, vision, driver, &rt, interval, tracker,
        )
        .await;
        return;
    }

    let mut acc = 0.0_f32;

    loop {
        let dt = get_frame_time();
        acc += dt;

        if auto_probe.is_none() && is_key_pressed(KeyCode::R) {
            state.restart_episode(&map);
            driver.reset(state.episode_seed);
            vision.clear_pending();
            state.last_logged_vision_tick = None;
        }

        if state.sim.is_episode_over() {
            if let Some(probe) = auto_probe.as_ref() {
                let stagnant = probe.finish(&state.sim);
                std::process::exit(if stagnant { 1 } else { 0 });
            }
            state.restart_cooldown -= dt;
            if state.restart_cooldown <= 0.0 {
                state.restart_episode(&map);
                driver.reset(state.episode_seed);
                vision.clear_pending();
                state.last_logged_vision_tick = None;
                state.restart_cooldown = 1.5;
            }
        } else {
            while acc >= LOGIC_DT {
                let mut just_decided: Option<u32> = None;
                if let Ok(Some((vtick, _obs))) = driver.logic_tick_preview(
                    &mut vision,
                    &mut state.sim,
                    state.logic_tick,
                    interval,
                    &assets,
                    &rt,
                ) {
                    if state.last_logged_vision_tick != Some(vtick) {
                        just_decided = Some(vtick);
                        state.last_logged_vision_tick = Some(vtick);
                    }
                }

                // 每逻辑帧只 pace 一次：决策日志不得再调 paced_input，否则攻击被 refractory 吃掉。
                let input = driver.paced_input_for_sim(&state.sim, state.logic_tick);
                let intended = driver.input();
                if let Some(vtick) = just_decided {
                    let effective = state.sim.effective_bot_input(&input);
                    if !cli.quiet {
                        log_decision(
                            vtick,
                            &intended,
                            &effective,
                            &state.sim,
                            &driver,
                            vision.last_detections(),
                        );
                    }
                    if let Some(probe) = auto_probe.as_mut() {
                        probe.on_decision(&input);
                    }
                }
                state.sim.tick(&input);
                if let Some(probe) = auto_probe.as_mut() {
                    probe.on_tick(state.sim.state.player.x);
                    if probe.sample_n >= probe.limit {
                        let stagnant = probe.finish(&state.sim);
                        std::process::exit(if stagnant { 1 } else { 0 });
                    }
                }
                state.logic_tick = state.logic_tick.wrapping_add(1);
                acc -= LOGIC_DT;

                if state.sim.is_episode_over() {
                    state.restart_cooldown = 1.5;
                    break;
                }
            }
        }

        if vision.worker_dead() {
            eprintln!("视觉线程已退出，预览结束");
            if let Some(probe) = auto_probe.as_ref() {
                let stagnant = probe.finish(&state.sim);
                std::process::exit(if stagnant { 1 } else { 0 });
            }
            break;
        }

        clear_background(Color::new(0.05, 0.05, 0.08, 1.0));
        view::begin_logical_viewport();
        view::draw_content(&assets, &state.sim);
        view::draw_yolo_floor_overlay(vision.last_detections(), VISION_CONF_THRESH);
        if let Some(hit) = vision.last_self_player() {
            view::draw_self_player_marker(hit);
        }
        set_default_camera();

        next_frame().await;

        let _ = vision.flush_submit(&rt);
    }
}
