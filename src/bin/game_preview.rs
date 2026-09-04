//! NavBot 自动玩预览（YOLO + SelfTracker + 地图图导航）。
//!
//! ```powershell
//! cargo run --release --bin game_preview
//! cargo run --release --bin game_preview -- --detect-hz 10
//! cargo run --release --bin game_preview -- --nav-log verbose
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
    self, evaluate_first_platform_report,
    format_first_platform_preview_done, DeferredCaptureVision, FirstPlatformTracker,
    GameSim, InputFrame, ProbeDriver, VisionAnchorConfig, VisionPaceConfig, VisionPipeline,
    FIRST_PLATFORM_PROBE_TICKS, LOGIC_DT, OBS_FLOOR_SLOTS, OBS_FLOOR_START, OBS_SLOT_DIM,
    VISION_CONF_THRESH, WINDOW_H, WINDOW_W,
};
use mxd_tools::yolo::YoloDevice;

fn window_conf() -> Conf {
    Conf {
        window_title: "NavBot 自动玩".to_owned(),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NavLogMode {
    /// 仅在状态变化、失败、step_up、noop 卡住时输出
    Event,
    /// 每次视觉决策帧输出一行摘要
    All,
    /// 每次输出时附带 YOLO/obs 详情
    Verbose,
}

impl NavLogMode {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "event" => Some(Self::Event),
            "all" => Some(Self::All),
            "verbose" => Some(Self::Verbose),
            _ => None,
        }
    }
}

struct Cli {
    /// `None` = 使用编译期嵌入的默认 ONNX。
    model: Option<PathBuf>,
    episode_seed: u64,
    pace: VisionPaceConfig,
    quiet: bool,
    anchor_offset: f32,
    auto_ticks: u32,
    probe: Option<PreviewProbe>,
    nav_log: NavLogMode,
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
        let nav_log = args
            .iter()
            .position(|a| a == "--nav-log")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| NavLogMode::parse(s))
            .unwrap_or(NavLogMode::Event);
        Self {
            model: arg_path(args, "--model"),
            episode_seed: arg_u64(args, "--seed", 0),
            pace: VisionPaceConfig::from_detect_hz(detect_hz),
            quiet: probe.is_some() || args.iter().any(|a| a == "--quiet"),
            anchor_offset: arg_f32(args, "--anchor-offset", 0.0),
            auto_ticks,
            probe,
            nav_log,
        }
    }

    fn vision_anchor(&self) -> VisionAnchorConfig {
        if self.anchor_offset > 0.0 {
            VisionAnchorConfig::with_jitter(self.anchor_offset)
        } else {
            VisionAnchorConfig::jitter()
        }
    }
}

struct NavLogState {
    last_reason: String,
    last_goal: String,
    last_visited: usize,
    last_nav_node: u32,
    repeat: u32,
}

impl NavLogState {
    fn new() -> Self {
        Self {
            last_reason: String::new(),
            last_goal: String::new(),
            last_visited: 0,
            last_nav_node: 0,
            repeat: 0,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

struct PreviewState {
    sim: GameSim,
    logic_tick: u32,
    episode_seed: u64,
    restart_cooldown: f32,
    last_logged_vision_tick: Option<u32>,
    nav_log: NavLogState,
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
            nav_log: NavLogState::new(),
        }
    }

    fn restart_episode(&mut self, map: &game::GameMap) {
        let seed = self.episode_seed.wrapping_add(self.logic_tick as u64);
        self.sim = GameSim::new_preview(map.clone(), seed);
        self.logic_tick = 0;
        self.restart_cooldown = 0.0;
        self.nav_log.reset();
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
        let input = driver.paced_input_for_sim(&mut state.sim, tick);
        let mut input = input;
        if vision.needs_probe() {
            input = vision.probe_input();
        }
        vision.note_commanded(&input);
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
                    cli.nav_log,
                    &mut state.nav_log,
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
    nav_log: NavLogMode,
    nav_state: &mut NavLogState,
) {
    log_nav_decision(
        tick,
        intended,
        effective,
        sim,
        driver,
        detections,
        nav_log,
        nav_state,
    );
}

fn log_nav_decision(
    tick: u32,
    intended: &InputFrame,
    effective: &InputFrame,
    sim: &GameSim,
    driver: &ProbeDriver,
    detections: &[mxd_tools::yolo::Detection],
    mode: NavLogMode,
    state: &mut NavLogState,
) {
    use mxd_tools::game::nav::ExecutorResult;

    let nav = driver.bot();
    let d = nav.diag();
    let _p = &sim.state.player;
    let visited = nav.visited_nodes();
    let total = nav.total_nodes();
    let goal_s = d.goal.label();
    let reason = nav.last_reason;
    let noop = input_is_noop(intended, effective);
    let intent_mismatch = input_label(intended) != input_label(effective);
    let node_mismatch = false;

    let event = state.last_reason != reason
        || state.last_goal != goal_s
        || state.last_visited != visited
        || state.last_nav_node != d.nav_node
        || d.exec != ExecutorResult::Running
        || d.escape_ticks > 0
        || intent_mismatch
        || noop
        || node_mismatch
        || reason.contains("step_up")
        || reason.contains("stalled")
        || reason.contains("escape");

    let throttle = matches!(reason, "step_up_wait" | "patrol" | "goto" | "goto_done")
        && !event
        && mode == NavLogMode::Event;

    let should_log = match mode {
        NavLogMode::All | NavLogMode::Verbose => true,
        NavLogMode::Event => {
            if throttle {
                state.repeat = state.repeat.saturating_add(1);
                state.repeat % 6 == 0
            } else if event {
                state.last_reason = reason.to_string();
                state.last_goal = goal_s.clone();
                state.last_visited = visited;
                state.last_nav_node = d.nav_node;
                state.repeat = 0;
                true
            } else {
                false
            }
        }
    };

    if !should_log {
        return;
    }

    let edge = match (d.pending_from, d.pending_kind, d.pending_to) {
        (Some(f), Some(k), Some(t)) => format!(" edge={f}-{}->{t}", k.label()),
        _ => String::new(),
    };

    eprintln!(
        "NAV tick={} intent={} eff={} exec={} reason={} goal={} visited={}/{} nav_node={} est_node={} est=({:.0},{:.0}) conf={} sub={}/{} fail={} esc={}{} farm={} alive={} kills={} meso={} ground_drops={} yolo_meso={} yolo_drop={}",
        tick,
        input_label(intended),
        input_label(effective),
        d.exec.label(),
        reason,
        goal_s,
        visited,
        total,
        d.nav_node,
        d.est_node,
        d.est_x,
        d.est_y,
        d.visual_conf,
        d.subgoal_ticks,
        if matches!(d.goal, mxd_tools::game::nav::SubGoal::StepUp { .. }) {
            nav.config.step_up_timeout_ticks
        } else {
            nav.config.subgoal_timeout_ticks
        },
        d.subgoal_failures,
        d.escape_ticks,
        edge,
        d.farm_local,
        sim.state.mobs.iter().filter(|m| m.alive).count(),
        sim.state.kills,
        sim.state.meso,
        sim.state.drops.iter().filter(|x| x.alive).count(),
        detections
            .iter()
            .filter(|x| x.label == "金币" && x.conf >= VISION_CONF_THRESH)
            .count(),
        detections
            .iter()
            .filter(|x| {
                matches!(x.label, "金币" | "药水" | "武器" | "装备" | "材料")
                    && x.conf >= VISION_CONF_THRESH
            })
            .count(),
    );

    let want_detail = mode == NavLogMode::Verbose
        || noop
        || d.exec == ExecutorResult::Failed
        || reason.contains("step_up")
        || node_mismatch
        || intent_mismatch;

    if !want_detail {
        return;
    }

    log_nav_detail(d, detections, driver.last_obs(), nav.config.step_up_stall);
}

fn log_nav_detail(
    d: &mxd_tools::game::nav::NavDiagSnapshot,
    detections: &[mxd_tools::yolo::Detection],
    obs: &[f32],
    step_stall_max: u32,
) {
    eprintln!(
        "  nav_detail: walkR={:?} walkL={:?} dropR={:?} dropL={:?} step_obs={:?} gnd_est={} jump_dir={:.0} jumped={} stall={}/{} cd={} esc_dir={:.0} blocked={} combat={}",
        d.walk_right,
        d.walk_left,
        d.drop_right,
        d.drop_left,
        d.obs_step_up,
        d.grounded_est,
        d.step_jump_dir,
        d.step_jumped,
        d.step_stall,
        step_stall_max,
        d.step_jump_cd,
        d.escape_dir,
        d.blocked_edges,
        d.combat_active,
    );
    log_yolo_floors(detections);
    log_obs_floors(obs);
}

fn input_is_noop(intended: &InputFrame, effective: &InputFrame) -> bool {
    !intended.left
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
        && !effective.down
}

fn log_yolo_floors(detections: &[mxd_tools::yolo::Detection]) {
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
            "  yolo_floor: n={} nearest conf={:.2} box=({:.0},{:.0})-({:.0},{:.0})",
            floors.len(),
            f.conf,
            f.x1,
            f.y1,
            f.x2,
            f.y2,
        );
    } else {
        eprintln!("  yolo_floor: n=0");
    }
}

fn log_obs_floors(obs: &[f32]) {
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
            "  obs_floor[{i}]: dx={:.0} dy={:.0} w={:.0} h={:.0}",
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
        eprintln!("  obs_floor: empty");
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let cli = Cli::parse(&args);

    eprintln!(
        "NavBot 预览 detect-hz={:.1} seed={} quiet={}",
        cli.pace.detect_hz(),
        cli.episode_seed,
        cli.quiet
    );
    if !cli.quiet {
        eprintln!(
            "Nav 日志: --nav-log={:?} (event=仅变化/step_up/卡住, all=每帧, verbose=含YOLO详情)",
            cli.nav_log
        );
    }
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

    let pipeline = match VisionPipeline::load_optional(
        cli.model.as_deref(),
        YoloDevice::Cpu,
        VISION_CONF_THRESH,
    )
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
        ProbeDriver::new_nav(&map, episode_seed)
    } else {
        ProbeDriver::new_realtime_with_map(&map, episode_seed)
    };
    driver.reset_with_sim(&state.sim, episode_seed);
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
            driver.reset_with_sim(&state.sim, state.episode_seed);
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
                driver.reset_with_sim(&state.sim, state.episode_seed);
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
                let mut input = driver.paced_input_for_sim(&mut state.sim, state.logic_tick);
                if vision.needs_probe() {
                    input = vision.probe_input();
                }
                vision.note_commanded(&input);
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
                            cli.nav_log,
                            &mut state.nav_log,
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
        view::draw_yolo_player_overlay(vision.last_detections(), VISION_CONF_THRESH);
        view::draw_self_track_hud(
            vision.tracker_mode().label(),
            vision.tracker().regime().label(),
        );
        if let Some(hit) = vision.last_self_player() {
            view::draw_self_player_box(hit, vision.tracker_mode().label());
        }
        set_default_camera();

        next_frame().await;

        let _ = vision.flush_submit(&rt);
    }
}
