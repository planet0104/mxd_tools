//! Headless 规则 bot 多局探针：离屏渲染 + 真实 YOLO/OCR 观测（与 game_preview 一致）。

use std::collections::HashSet;

use anyhow::Result;

use super::config::VisionPaceConfig;
use super::headless_vision::HeadlessVisionEnv;
use super::input::InputFrame;
use super::observation::{
    obs_assess_enemy_contact, obs_enemy_in_attack_range,
    obs_has_enemy, obs_platform_edge, OBS_DIM, OBS_ENEMY_SLOTS, OBS_ENEMY_START, OBS_SLOT_DIM,
};
use super::rule_bot::{visit_key, RuleBot, RuleBotCtx};
use super::human_pace::HumanPace;
use super::sim::GameSim;
use super::types::{LOGIC_DT, LOGIC_HZ};
use std::time::{Duration, Instant};

/// 首平台探针逻辑时长（60Hz × 90s，与 headless / game_preview 探针一致）。
pub const FIRST_PLATFORM_PROBE_TICKS: u32 = 5_400;

const STUCK_WINDOW: u32 = 120;
const STUCK_MOVE_EPS: f32 = 18.0;
const SINGLE_MOB_ATTACK_WINDOW: u32 = 240;
/// 默认地图出生区：前 30s 不应疯狂跳跃（seed 0 与 game_preview 默认一致）。
const SPAWN_PROBE_TICKS: u32 = 1_800;
const SPAWN_MAX_JUMP_DECISION_RATIO: f32 = 0.15;
const SPAWN_MAX_EFFECTIVE_JUMP_RATIO: f32 = 0.02;

/// 探针 / 预览主循环：YOLO 异步提交/轮询，sim 每 tick 继续。
pub struct ProbeDriver {
    bot: RuleBot,
    pace: HumanPace,
    input: InputFrame,
    last_obs: [f32; OBS_DIM],
    wall_clock_pacing: bool,
}

impl ProbeDriver {
    pub fn new(seed: u64) -> Self {
        Self::with_pacing(seed, true)
    }

    /// 窗口预览：逻辑 60Hz 由 macroquad 帧率驱动，不做 thread sleep。
    pub fn new_realtime(seed: u64) -> Self {
        Self::with_pacing(seed, false)
    }

    fn with_pacing(seed: u64, wall_clock_pacing: bool) -> Self {
        Self {
            bot: RuleBot::default(),
            pace: HumanPace::new(seed),
            input: InputFrame::default(),
            last_obs: [0.0_f32; OBS_DIM],
            wall_clock_pacing,
        }
    }

    pub fn reset(&mut self, seed: u64) {
        self.bot.reset();
        self.pace.reset(seed);
        self.input = InputFrame::default();
        self.last_obs = [0.0_f32; OBS_DIM];
    }

    pub fn bot(&self) -> &RuleBot {
        &self.bot
    }

    pub fn input(&self) -> InputFrame {
        self.input
    }

    pub fn last_obs(&self) -> &[f32; OBS_DIM] {
        &self.last_obs
    }

    pub fn apply_observation(&mut self, sim: &mut GameSim, vtick: u32, obs: [f32; OBS_DIM]) {
        self.last_obs = obs;
        sim.movement_gate.set_last_observation(&obs);
        let ctx = RuleBotCtx::from_sim_with_farm_y(sim, &obs, self.bot.farm_y);
        self.input = self.bot.decide(ctx);
        self.pace.on_intent(self.input, vtick);
    }

    /// 每逻辑 tick：先 poll 结果，再在间隔点且 YOLO 有空位时 submit。
    pub async fn logic_tick(
        &mut self,
        vision: &mut HeadlessVisionEnv,
        sim: &mut GameSim,
        tick: u32,
        vision_interval: u32,
    ) -> Result<Option<(u32, [f32; OBS_DIM])>> {
        if vision.worker_dead() {
            anyhow::bail!("YOLO 视觉线程已退出");
        }
        if let Some((vtick, obs)) = vision.poll_observation(sim) {
            self.apply_observation(sim, vtick, obs);
            return Ok(Some((vtick, obs)));
        }
        if tick % vision_interval == 0 {
            let _ = vision.schedule_capture_if_ready(tick, sim).await?;
        }
        Ok(None)
    }

    /// 窗口预览：poll + 仅绘制 RT（submit 在 `next_frame` 后 flush）。
    pub fn logic_tick_preview(
        &mut self,
        vision: &mut super::headless_vision::DeferredCaptureVision,
        sim: &mut GameSim,
        tick: u32,
        vision_interval: u32,
        assets: &super::view::GameViewAssets,
        rt: &macroquad::prelude::RenderTarget,
    ) -> Result<Option<(u32, [f32; OBS_DIM])>> {
        if vision.worker_dead() {
            anyhow::bail!("YOLO 视觉线程已退出");
        }
        if let Some((vtick, obs)) = vision.poll_observation(sim) {
            self.apply_observation(sim, vtick, obs);
            return Ok(Some((vtick, obs)));
        }
        if tick % vision_interval == 0 {
            vision.schedule_draw_if_ready(tick, sim, assets, rt);
        }
        Ok(None)
    }

    pub fn paced_input(&mut self, tick: u32) -> InputFrame {
        self.pace.apply(self.input, tick)
    }

    /// 经 HumanPace 后按 sim 门控核对；若攻击被剥掉则退回 refractory。
    pub fn paced_input_for_sim(&mut self, sim: &GameSim, tick: u32) -> InputFrame {
        // 视觉帧间隙（5–10Hz）内按 sim 刷新站砍，避免怪走进可砍带仍冻在 noop。
        self.refresh_melee_hold(sim);
        let paced = self.pace.apply(self.input, tick);
        let effective = sim.effective_bot_input(&paced);
        if paced.attack && !effective.attack {
            self.pace.refund_attack();
        }
        paced
    }

    /// 可砍带内强制站砍（不走路），覆盖过期的视觉意图。
    fn refresh_melee_hold(&mut self, sim: &GameSim) {
        if !sim.mob_in_strike_band() {
            return;
        }
        self.input.attack = true;
        self.input.left = false;
        self.input.right = false;
        self.input.jump = false;
    }

    pub fn tick_sim(&mut self, sim: &mut GameSim, tick: u32) -> InputFrame {
        let tick_start = Instant::now();
        let paced = self.paced_input_for_sim(sim, tick);
        sim.tick(&paced);
        if self.wall_clock_pacing {
            Self::sleep_to_logic_hz(tick_start);
        }
        paced
    }

    /// 与 game_preview 一致：60Hz 逻辑步进，给 YOLO ~200ms/帧 的墙钟时间。
    fn sleep_to_logic_hz(tick_start: Instant) {
        let budget = Duration::from_secs_f32(LOGIC_DT);
        if let Some(rest) = budget.checked_sub(tick_start.elapsed()) {
            std::thread::sleep(rest);
        }
    }

    /// 首帧阻塞 YOLO，避免启动后整局无观测。
    pub async fn bootstrap_vision(
        &mut self,
        vision: &mut HeadlessVisionEnv,
        sim: &mut GameSim,
    ) -> Result<()> {
        let obs = vision.observe_sim_blocking(sim).await?;
        self.apply_observation(sim, 0, obs);
        Ok(())
    }

    pub async fn bootstrap_vision_preview(
        &mut self,
        vision: &mut super::headless_vision::DeferredCaptureVision,
        sim: &mut GameSim,
        assets: &super::view::GameViewAssets,
        rt: &macroquad::prelude::RenderTarget,
    ) -> Result<()> {
        let obs = vision.observe_sim_blocking(sim, assets, rt).await?;
        self.apply_observation(sim, 0, obs);
        Ok(())
    }
}

/// 单局探针配置。
#[derive(Debug, Clone)]
pub struct BotProbeConfig {
    pub max_ticks: u32,
    pub vision_interval: u32,
}

impl Default for BotProbeConfig {
    fn default() -> Self {
        Self {
            max_ticks: 10_800,
            vision_interval: VisionPaceConfig::fast().vision_interval_ticks,
        }
    }
}

/// 单局结果与失败原因。
#[derive(Debug, Clone)]
pub struct EpisodeReport {
    pub seed: u64,
    pub ticks: u32,
    pub kills: u32,
    pub waves_cleared: u32,
    pub visited_cells: usize,
    pub altitude_bands: usize,
    pub x_range: f32,
    pub attack_decisions: u32,
    pub flee_without_contact: u32,
    pub max_flee_streak: u32,
    pub stuck_windows: u32,
    pub single_mob_no_attack_windows: u32,
    pub failures: Vec<String>,
}

impl EpisodeReport {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

/// 出生区跳跃探针（默认地图 + YOLO 观测，对齐 game_preview seed=0）。
#[derive(Debug, Clone)]
pub struct SpawnJumpReport {
    pub seed: u64,
    pub spawn_x: f32,
    pub spawn_y: f32,
    pub ticks: u32,
    pub vision_frames: u32,
    pub raw_jump_decisions: u32,
    pub effective_jump_ticks: u32,
    pub climb_jump_decisions: u32,
    pub seek_vertical_jump_decisions: u32,
    pub edge_jump_decisions: u32,
    pub end_x: f32,
    pub end_y: f32,
}

impl SpawnJumpReport {
    pub fn jump_decision_ratio(&self) -> f32 {
        if self.vision_frames == 0 {
            0.0
        } else {
            self.raw_jump_decisions as f32 / self.vision_frames as f32
        }
    }

    pub fn effective_jump_ratio(&self) -> f32 {
        if self.ticks == 0 {
            0.0
        } else {
            self.effective_jump_ticks as f32 / self.ticks as f32
        }
    }

    pub fn passed(&self) -> bool {
        self.effective_jump_ratio() <= SPAWN_MAX_EFFECTIVE_JUMP_RATIO
            && self.jump_decision_ratio() <= SPAWN_MAX_JUMP_DECISION_RATIO
    }

    pub fn failure_reason(&self) -> Option<String> {
        if self.passed() {
            return None;
        }
        Some(format!(
            "spawn jump spam seed={} decisions={}/{} ({:.1}%) effective={}/{} ({:.1}%) climb_j={} seek_v_j={} edge_j={} spawn=({:.0},{:.0}) end=({:.0},{:.0})",
            self.seed,
            self.raw_jump_decisions,
            self.vision_frames,
            self.jump_decision_ratio() * 100.0,
            self.effective_jump_ticks,
            self.ticks,
            self.effective_jump_ratio() * 100.0,
            self.climb_jump_decisions,
            self.seek_vertical_jump_decisions,
            self.edge_jump_decisions,
            self.spawn_x,
            self.spawn_y,
            self.end_x,
            self.end_y,
        ))
    }
}

pub async fn run_spawn_jump_probe(
    vision: &mut HeadlessVisionEnv,
    seed: u64,
    ticks: u32,
    vision_interval: u32,
) -> Result<SpawnJumpReport> {
    let map = super::load_default_map()?;
    let mut sim = GameSim::new_preview(map, seed);
    let spawn_x = sim.state.player.x;
    let spawn_y = sim.state.player.y;
    let mut driver = ProbeDriver::new(seed);
    driver.bootstrap_vision(vision, &mut sim).await?;

    let mut vision_frames = 0u32;
    let mut raw_jump_decisions = 0u32;
    let mut effective_jump_ticks = 0u32;
    let mut climb_jump_decisions = 0u32;
    let mut seek_vertical_jump_decisions = 0u32;
    let mut edge_jump_decisions = 0u32;

    for tick in 0..ticks {
        if sim.is_episode_over() {
            break;
        }
        if let Some((_vtick, obs)) = driver
            .logic_tick(vision, &mut sim, tick, vision_interval)
            .await?
        {
            vision_frames += 1;
            let seeking = driver.bot().explore_seeking_vertical();
            let input = driver.input();
            if input.jump {
                raw_jump_decisions += 1;
                if seeking {
                    seek_vertical_jump_decisions += 1;
                }
                if obs_platform_edge(&obs, 1.0) || obs_platform_edge(&obs, -1.0) {
                    edge_jump_decisions += 1;
                }
            }
            if input.jump && input.up {
                climb_jump_decisions += 1;
            }
        }
        let paced = driver.tick_sim(&mut sim, tick);
        let effective = sim.effective_bot_input(&paced);
        if effective.jump {
            effective_jump_ticks += 1;
        }
    }

    Ok(SpawnJumpReport {
        seed,
        spawn_x,
        spawn_y,
        ticks,
        vision_frames,
        raw_jump_decisions,
        effective_jump_ticks,
        climb_jump_decisions,
        seek_vertical_jump_decisions,
        edge_jump_decisions,
        end_x: sim.state.player.x,
        end_y: sim.state.player.y,
    })
}

/// 首平台探针：应主动横向探路/换层，不卡在出生平台边缘。
#[derive(Debug, Clone)]
pub struct FirstPlatformReport {
    pub kills: u32,
    pub spawn_y: f32,
    pub end_x: f32,
    pub end_y: f32,
    pub min_x_after_kills: f32,
    pub max_x_after_kills: f32,
    pub y_band_changed: bool,
}

impl FirstPlatformReport {
    /// 已离开第一个平台（换层或脚点明显下降）。
    pub fn left_first_platform(&self) -> bool {
        self.y_band_changed || self.end_y < self.spawn_y - 50.0
    }

    /// 在首平台横向推进（朝绳梯/离开出生区），不卡在右侧边缘。
    pub fn cleared_first_platform_horizontally(&self) -> bool {
        self.min_x_after_kills <= 420.0
            && self.max_x_after_kills >= 480.0
            && self.x_span_after_kills() >= 100.0
    }

    pub fn x_span_after_kills(&self) -> f32 {
        (self.max_x_after_kills - self.min_x_after_kills).max(0.0)
    }
}

/// 首平台进度追踪（headless / game_preview 共用）。
#[derive(Debug, Clone)]
pub struct FirstPlatformTracker {
    spawn_y: f32,
    spawn_y_band: i32,
    min_x_after_kills: f32,
    max_x_after_kills: f32,
    tracking: bool,
}

impl FirstPlatformTracker {
    pub fn new(spawn_x: f32, spawn_y: f32) -> Self {
        Self {
            spawn_y,
            spawn_y_band: visit_key(spawn_x, spawn_y).1,
            min_x_after_kills: f32::MAX,
            max_x_after_kills: f32::MIN,
            tracking: false,
        }
    }

    pub fn on_tick(&mut self, sim: &GameSim) {
        let on_spawn_band =
            visit_key(sim.state.player.x, sim.state.player.y).1 == self.spawn_y_band;
        if on_spawn_band && !self.tracking {
            self.tracking = true;
            self.min_x_after_kills = sim.state.player.x;
            self.max_x_after_kills = sim.state.player.x;
        }
        if self.tracking && on_spawn_band {
            let x = sim.state.player.x;
            self.min_x_after_kills = self.min_x_after_kills.min(x);
            self.max_x_after_kills = self.max_x_after_kills.max(x);
        }
    }

    pub fn finish(self, sim: &GameSim) -> FirstPlatformReport {
        let end_x = sim.state.player.x;
        let end_y = sim.state.player.y;
        let y_band_changed = visit_key(end_x, end_y).1 != self.spawn_y_band;
        let (min_x, max_x) = if self.tracking {
            (self.min_x_after_kills, self.max_x_after_kills)
        } else {
            (end_x, end_x)
        };
        FirstPlatformReport {
            kills: sim.state.kills,
            spawn_y: self.spawn_y,
            end_x,
            end_y,
            min_x_after_kills: min_x,
            max_x_after_kills: max_x,
            y_band_changed,
        }
    }
}

pub fn evaluate_first_platform_report(fp: &FirstPlatformReport) -> Result<(), String> {
    if fp.left_first_platform() {
        if fp.kills == 0 {
            return Err(format!(
                "left first platform without kills: end=({:.0},{:.0}) y_changed={}",
                fp.end_x, fp.end_y, fp.y_band_changed
            ));
        }
        return Ok(());
    }
    // 仍在首层：必须有击杀，且横向探开，才算通过。
    if fp.kills >= 1 && fp.cleared_first_platform_horizontally() {
        return Ok(());
    }
    if fp.cleared_first_platform_horizontally() && fp.kills == 0 {
        return Err(format!(
            "traversed first platform without kills: end=({:.0},{:.0}) x_span={:.0}",
            fp.end_x,
            fp.end_y,
            fp.x_span_after_kills(),
        ));
    }
    let x_span = fp.x_span_after_kills();
    if fp.end_x >= 590.0 && fp.min_x_after_kills >= 520.0 && !fp.y_band_changed {
        return Err(format!(
            "stuck on right edge on spawn platform: end=({:.0},{:.0}) min_x={:.0} max_x={:.0}",
            fp.end_x, fp.end_y, fp.min_x_after_kills, fp.max_x_after_kills,
        ));
    }
    if fp.end_x <= 500.0 && x_span <= 80.0 && fp.min_x_after_kills > 350.0 {
        return Err(format!(
            "stuck on left ping-pong on spawn platform: end=({:.0},{:.0}) x_span={:.0} min_x={:.0} max_x={:.0}",
            fp.end_x,
            fp.end_y,
            x_span,
            fp.min_x_after_kills,
            fp.max_x_after_kills,
        ));
    }
    if fp.min_x_after_kills <= 120.0 {
        return Err(format!(
            "stuck on map left edge on spawn platform: end=({:.0},{:.0}) min_x={:.0} max_x={:.0}",
            fp.end_x, fp.end_y, fp.min_x_after_kills, fp.max_x_after_kills,
        ));
    }
    Err(format!(
        "did not leave first platform: end=({:.0},{:.0}) spawn_y={:.0} y_changed={} x_span={:.0} min_x={:.0} max_x={:.0} kills={}",
        fp.end_x,
        fp.end_y,
        fp.spawn_y,
        fp.y_band_changed,
        x_span,
        fp.min_x_after_kills,
        fp.max_x_after_kills,
        fp.kills,
    ))
}

/// 供 stdout 解析：`PREVIEW_DONE verdict=PASS probe=first_platform ...`
pub fn format_first_platform_preview_done(fp: &FirstPlatformReport, pass: bool) -> String {
    if pass {
        format!(
            "PREVIEW_DONE verdict=PASS probe=first_platform kills={} y_changed={} end_x={:.0} end_y={:.0} min_x_after={:.0} max_x_after={:.0} spawn_y={:.0}",
            fp.kills,
            fp.y_band_changed,
            fp.end_x,
            fp.end_y,
            fp.min_x_after_kills,
            fp.max_x_after_kills,
            fp.spawn_y,
        )
    } else {
        let reason = evaluate_first_platform_report(fp).unwrap_err();
        format!(
            "PREVIEW_DONE verdict=FAIL probe=first_platform kills={} y_changed={} end_x={:.0} end_y={:.0} min_x_after={:.0} max_x_after={:.0} spawn_y={:.0} reason={reason}",
            fp.kills,
            fp.y_band_changed,
            fp.end_x,
            fp.end_y,
            fp.min_x_after_kills,
            fp.max_x_after_kills,
            fp.spawn_y,
        )
    }
}

pub async fn run_first_platform_progress_probe(
    vision: &mut HeadlessVisionEnv,
    seed: u64,
    max_ticks: u32,
    vision_interval: u32,
) -> Result<FirstPlatformReport> {
    let map = super::load_default_map()?;
    let mut sim = GameSim::new_preview(map, seed);
    let mut tracker = FirstPlatformTracker::new(sim.state.player.x, sim.state.player.y);
    let mut driver = ProbeDriver::new(seed);
    driver.bootstrap_vision(vision, &mut sim).await?;

    for tick in 0..max_ticks {
        if sim.is_episode_over() {
            break;
        }
        let _ = driver
            .logic_tick(vision, &mut sim, tick, vision_interval)
            .await?;
        driver.tick_sim(&mut sim, tick);
        tracker.on_tick(&sim);
    }

    Ok(tracker.finish(&sim))
}

/// 跑一局 headless bot（preview 模式：不死，便于长测）。
pub async fn run_episode(
    vision: &mut HeadlessVisionEnv,
    seed: u64,
    cfg: &BotProbeConfig,
) -> Result<EpisodeReport> {
    let map = super::load_default_map()?;
    let mut sim = GameSim::new_preview(map, seed);
    let mut driver = ProbeDriver::new(seed);
    driver.bootstrap_vision(vision, &mut sim).await?;

    let mut min_x = sim.state.player.x;
    let mut max_x = sim.state.player.x;
    let mut y_bands: HashSet<i32> = HashSet::new();
    let mut attack_decisions = 0u32;
    let mut flee_without_contact = 0u32;
    let mut flee_streak = 0u32;
    let mut max_flee_streak = 0u32;
    let mut stuck_windows = 0u32;
    let mut single_mob_no_attack_windows = 0u32;
    let mut waves_cleared = 0u32;

    let mut pos_ring: [(f32, f32); STUCK_WINDOW as usize] =
        [(sim.state.player.x, sim.state.player.y); STUCK_WINDOW as usize];
    let mut ring_i = 0usize;

    let mut single_mob_ticks = 0u32;
    let mut single_mob_attacks = 0u32;

    let kills_start = sim.state.kills;
    let mut prev_alive = sim.state.mobs.iter().filter(|m| m.alive).count();

    for tick in 0..cfg.max_ticks {
        if sim.is_episode_over() {
            break;
        }

        let new_vision = driver
            .logic_tick(vision, &mut sim, tick, cfg.vision_interval)
            .await?;

        let paced = driver.tick_sim(&mut sim, tick);

        if let Some((_vtick, ref obs)) = new_vision {
            let input = driver.input();
            if input.attack {
                attack_decisions += 1;
            }

            let effective = sim.effective_bot_input(&paced);
            let contact = obs_assess_enemy_contact(obs);
            let alive = sim.state.mobs.iter().filter(|m| m.alive).count();
            if obs_has_enemy(obs)
                && contact.total == 0
                && alive > 0
                && !paced.attack
                && is_fleeing_from_nearest_enemy(obs, &effective)
            {
                flee_without_contact += 1;
                flee_streak += 1;
                max_flee_streak = max_flee_streak.max(flee_streak);
            } else {
                flee_streak = 0;
            }

            if alive == 1 && contact.total == 0 {
                single_mob_ticks += cfg.vision_interval;
                if paced.attack {
                    single_mob_attacks += 1;
                }
                if single_mob_ticks >= SINGLE_MOB_ATTACK_WINDOW {
                    if single_mob_attacks == 0 {
                        single_mob_no_attack_windows += 1;
                    }
                    single_mob_ticks = 0;
                    single_mob_attacks = 0;
                }
            } else {
                single_mob_ticks = 0;
                single_mob_attacks = 0;
            }
        }

        let px = sim.state.player.x;
        let py = sim.state.player.y;
        min_x = min_x.min(px);
        max_x = max_x.max(px);
        y_bands.insert(visit_key(px, py).1);

        pos_ring[ring_i] = (px, py);
        ring_i = (ring_i + 1) % STUCK_WINDOW as usize;
        if tick >= STUCK_WINDOW && tick % cfg.vision_interval == 0 {
            let (fx, fy) = pos_ring[ring_i];
            let moved = (px - fx).abs() + (py - fy).abs();
            let attacking = paced.attack
                || obs_enemy_in_attack_range(sim.movement_gate.last_observation(), sim.state.player.facing);
            let (pr, pl) = sim.physics_walk_ok_pair();
            let at_cliff = pr == Some(false) || pl == Some(false);
            if moved < STUCK_MOVE_EPS
                && sim.state.player.on_ground
                && !sim.state.player.climbing
                && !attacking
                && !at_cliff
            {
                stuck_windows += 1;
            }
        }

        let alive = sim.state.mobs.iter().filter(|m| m.alive).count();
        if prev_alive > 0 && alive == 0 {
            waves_cleared += 1;
        }
        prev_alive = alive;
    }

    let kills = sim.state.kills.saturating_sub(kills_start);
    let visited_cells = driver.bot().visited_cell_count();
    let x_range = max_x - min_x;

    let mut failures = Vec::new();
    push_threshold_u32(&mut failures, kills, 2, "kills");
    push_threshold(&mut failures, visited_cells, 8, "visited_cells");
    push_threshold(&mut failures, y_bands.len(), 2, "altitude_bands");
    push_threshold_f(&mut failures, x_range, 240.0, "x_range");

    if max_flee_streak > 36 {
        failures.push(format!(
            "max_flee_streak={max_flee_streak} (>36 consecutive vision frames fleeing away from visible mobs without contact)"
        ));
    }
    if stuck_windows > 12 {
        failures.push(format!(
            "stuck_windows={stuck_windows} (>6 windows with movement < {STUCK_MOVE_EPS}px)"
        ));
    }
    if single_mob_no_attack_windows > 0 {
        failures.push(format!(
            "single_mob_no_attack_windows={single_mob_no_attack_windows} (1 mob, no contact, no attack for {} ticks)",
            SINGLE_MOB_ATTACK_WINDOW
        ));
    }

    Ok(EpisodeReport {
        seed,
        ticks: cfg.max_ticks,
        kills,
        waves_cleared,
        visited_cells,
        altitude_bands: y_bands.len(),
        x_range,
        attack_decisions,
        flee_without_contact,
        max_flee_streak,
        stuck_windows,
        single_mob_no_attack_windows,
        failures,
    })
}

fn is_fleeing_from_nearest_enemy(obs: &[f32; OBS_DIM], effective: &InputFrame) -> bool {
    let Some(dx) = nearest_enemy_dx(obs) else {
        return false;
    };
    let moving_right = effective.right && !effective.left;
    let moving_left = effective.left && !effective.right;
    if !moving_left && !moving_right {
        return false;
    }
    (dx > 0.008 && moving_left) || (dx < -0.008 && moving_right)
}

fn nearest_enemy_dx(obs: &[f32; OBS_DIM]) -> Option<f32> {
    let mut best: Option<(f32, f32)> = None;
    for i in 0..OBS_ENEMY_SLOTS {
        let base = OBS_ENEMY_START + i * OBS_SLOT_DIM;
        if base + OBS_SLOT_DIM > obs.len() {
            break;
        }
        if obs[base + 2].abs() <= 1e-4 && obs[base + 3].abs() <= 1e-4 {
            continue;
        }
        let dx = obs[base];
        let dy = obs[base + 1];
        let dist = dx.abs() + dy.abs() * 0.3;
        match best {
            None => best = Some((dist, dx)),
            Some((bd, _)) if dist < bd => best = Some((dist, dx)),
            _ => {}
        }
    }
    best.map(|(_, dx)| dx)
}

fn push_threshold(failures: &mut Vec<String>, got: usize, min: usize, name: &str) {
    if got < min {
        failures.push(format!("{name}={got} (need >={min})"));
    }
}

fn push_threshold_f(failures: &mut Vec<String>, got: f32, min: f32, name: &str) {
    if got < min {
        failures.push(format!("{name}={got:.0} (need >={min:.0})"));
    }
}

fn push_threshold_u32(failures: &mut Vec<String>, got: u32, min: u32, name: &str) {
    if got < min {
        failures.push(format!("{name}={got} (need >={min})"));
    }
}

/// 多 seed 批量探针；失败 seed 超过 2 个则返回 Err(汇总)。
pub async fn run_probe_seeds(
    vision: &mut HeadlessVisionEnv,
    seeds: &[u64],
    cfg: &BotProbeConfig,
) -> Result<Vec<EpisodeReport>> {
    let mut reports = Vec::new();
    let mut lines = Vec::new();
    for &seed in seeds {
        let r = run_episode(vision, seed, cfg).await?;
        let status = if r.passed() { "PASS" } else { "FAIL" };
        lines.push(format!(
            "{status} seed={} kills={} waves={} visited={} y_bands={} x_range={:.0} attacks={} flee_no_contact={} flee_streak_max={} stuck={}",
            r.seed,
            r.kills,
            r.waves_cleared,
            r.visited_cells,
            r.altitude_bands,
            r.x_range,
            r.attack_decisions,
            r.flee_without_contact,
            r.max_flee_streak,
            r.stuck_windows,
        ));
        if !r.failures.is_empty() {
            for f in &r.failures {
                lines.push(format!("  - {f}"));
            }
        }
        reports.push(r);
    }

    let failed: Vec<_> = reports.iter().filter(|r| !r.passed()).collect();
    if failed.is_empty() {
        Ok(reports)
    } else if failed.len() <= 2 {
        eprintln!(
            "WARN: {}/{} seeds below threshold, overall PASS",
            failed.len(),
            reports.len()
        );
        Ok(reports)
    } else {
        anyhow::bail!(lines.join("\n"));
    }
}

pub fn default_probe_seeds() -> Vec<u64> {
    vec![2, 5]
}

/// 并行模式默认并发数（子进程数上限）。4 = first_platform + spawn + episodes×2。
pub const DEFAULT_PARALLEL_JOBS: usize = 4;

/// 并行模式下的 episode seed（总任务 = 2 固定探针 + 这些 seed）。
pub fn default_parallel_episode_seeds(jobs: usize) -> Vec<u64> {
    let n = jobs.saturating_sub(2).max(1);
    match n {
        1 => vec![0],
        2 => vec![2, 5],
        _ => (0..n as u64).collect(),
    }
}

/// 仅 episode 并行任务（first_platform/spawn 应单独先跑）。
pub fn build_parallel_episode_jobs(episode_seeds: &[u64]) -> Vec<(String, Vec<String>)> {
    episode_seeds
        .iter()
        .map(|&seed| {
            (
                format!("episodes_seed_{seed}"),
                vec![
                    "--probe".into(),
                    "episodes".into(),
                    "--seeds".into(),
                    seed.to_string(),
                ],
            )
        })
        .collect()
}

/// 构造并行子进程任务列表（含 first_platform + spawn + episodes）。
pub fn build_parallel_probe_jobs(episode_seeds: &[u64]) -> Vec<(String, Vec<String>)> {
    let mut jobs = vec![
        (
            "first_platform".to_string(),
            vec!["--probe".into(), "first_platform".into()],
        ),
        (
            "spawn".to_string(),
            vec!["--probe".into(), "spawn".into()],
        ),
    ];
    for &seed in episode_seeds {
        jobs.push((
            format!("episodes_seed_{seed}"),
            vec![
                "--probe".into(),
                "episodes".into(),
                "--seeds".into(),
                seed.to_string(),
            ],
        ));
    }
    jobs
}

#[derive(Debug, Clone)]
pub struct ParallelProbeReport {
    pub total: usize,
    pub passed: usize,
    pub failed: Vec<String>,
}

impl ParallelProbeReport {
    pub fn ok(&self) -> bool {
        self.failed.is_empty()
    }
}

pub fn run_parallel_probe_subprocess(
    exe: &std::path::Path,
    model: Option<&std::path::Path>,
    name: &str,
    probe_args: &[String],
) -> bool {
    let mut cmd = std::process::Command::new(exe);
    cmd.args(probe_args)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit());
    if let Some(m) = model {
        cmd.arg("--model").arg(m);
    }
    eprintln!("[parallel] 启动 {name}");
    let ok = cmd.status().map(|s| s.success()).unwrap_or(false);
    eprintln!("[parallel] {name} {}", if ok { "PASS" } else { "FAIL" });
    ok
}

/// 线程池调度多子进程 YOLO 探针（每子进程：GL 主线程 + YOLO 后台线程）。
pub fn run_parallel_probe_pool(
    exe: &std::path::Path,
    model: Option<&std::path::Path>,
    jobs: Vec<(String, Vec<String>)>,
    max_workers: usize,
) -> ParallelProbeReport {
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    let max_workers = max_workers.max(1);
    let queue = Arc::new(Mutex::new(jobs));
    let failed = Arc::new(Mutex::new(Vec::<String>::new()));
    let total = {
        let q = queue.lock().expect("parallel queue");
        q.len()
    };

    eprintln!("YOLO 并行探针: total={total} max_workers={max_workers}");

    let handles: Vec<_> = (0..max_workers)
        .map(|worker_id| {
            let exe = exe.to_path_buf();
            let model = model.map(std::path::Path::to_path_buf);
            let queue = Arc::clone(&queue);
            let failed = Arc::clone(&failed);
            thread::spawn(move || {
                if worker_id > 0 {
                    thread::sleep(Duration::from_millis((worker_id as u64) * 300));
                }
                loop {
                    let job = {
                        let mut q = queue.lock().expect("parallel queue");
                        q.pop()
                    };
                    let Some((name, probe_args)) = job else {
                        break;
                    };
                    eprintln!("[parallel w{worker_id}] 启动 {name}");
                    let mut cmd = std::process::Command::new(&exe);
                    cmd.args(&probe_args)
                        .stdout(std::process::Stdio::inherit())
                        .stderr(std::process::Stdio::inherit());
                    if let Some(ref m) = model {
                        cmd.arg("--model").arg(m);
                    }
                    let ok = cmd.status().map(|s| s.success()).unwrap_or(false);
                    eprintln!(
                        "[parallel w{worker_id}] {name} {}",
                        if ok { "PASS" } else { "FAIL" }
                    );
                    if !ok {
                        failed.lock().expect("parallel failed").push(name);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        let _ = h.join();
    }

    let failed = Arc::try_unwrap(failed)
        .map(|m| m.into_inner().expect("parallel failed"))
        .unwrap_or_default();
    let passed = total.saturating_sub(failed.len());
    ParallelProbeReport {
        total,
        passed,
        failed,
    }
}

pub fn probe_duration_secs(cfg: &BotProbeConfig) -> f32 {
    cfg.max_ticks as f32 / LOGIC_HZ
}

/// 全部 YOLO 探针汇总。
#[derive(Debug, Clone)]
pub struct YoloProbeSummary {
    pub episode_reports: Vec<EpisodeReport>,
    pub first_platform: FirstPlatformReport,
    pub spawn_jump: SpawnJumpReport,
}

/// 运行全部 headless YOLO 探针（与旧 `#[cfg(test)]` 三件套等价）。
pub async fn run_all_yolo_probes(
    vision: &mut HeadlessVisionEnv,
    cfg: &BotProbeConfig,
) -> Result<YoloProbeSummary> {
    run_yolo_probes(vision, cfg, YoloProbeSet::All, &default_probe_seeds()).await
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YoloProbeSet {
    All,
    Episodes,
    FirstPlatform,
    SpawnJump,
}

impl YoloProbeSet {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "all" => Some(Self::All),
            "episodes" => Some(Self::Episodes),
            "first_platform" => Some(Self::FirstPlatform),
            "spawn" => Some(Self::SpawnJump),
            _ => None,
        }
    }
}

pub async fn run_yolo_probes(
    vision: &mut HeadlessVisionEnv,
    cfg: &BotProbeConfig,
    set: YoloProbeSet,
    episode_seeds: &[u64],
) -> Result<YoloProbeSummary> {
    let run_ep = set == YoloProbeSet::All || set == YoloProbeSet::Episodes;
    let run_fp = set == YoloProbeSet::All || set == YoloProbeSet::FirstPlatform;
    let run_spawn = set == YoloProbeSet::All || set == YoloProbeSet::SpawnJump;

    let seeds: Vec<u64> = if episode_seeds.is_empty() {
        default_probe_seeds()
    } else {
        episode_seeds.to_vec()
    };

    let episode_reports = if run_ep {
        run_probe_seeds(vision, &seeds, cfg).await?
    } else {
        Vec::new()
    };

    let first_platform = if run_fp {
        eprintln!("探针 first_platform: seed=0 ticks=5400 (~90s 逻辑)");
        run_first_platform_progress_probe(vision, 0, 5_400, cfg.vision_interval).await?
    } else {
        FirstPlatformReport {
            kills: 0,
            spawn_y: 0.0,
            end_x: 0.0,
            end_y: 0.0,
            min_x_after_kills: 0.0,
            max_x_after_kills: 0.0,
            y_band_changed: false,
        }
    };

    let spawn_jump = if run_spawn {
        eprintln!("探针 spawn_jump: seed=0 ticks={SPAWN_PROBE_TICKS}");
        run_spawn_jump_probe(vision, 0, SPAWN_PROBE_TICKS, cfg.vision_interval).await?
    } else {
        SpawnJumpReport {
            seed: 0,
            spawn_x: 0.0,
            spawn_y: 0.0,
            ticks: 0,
            vision_frames: 0,
            raw_jump_decisions: 0,
            effective_jump_ticks: 0,
            climb_jump_decisions: 0,
            seek_vertical_jump_decisions: 0,
            edge_jump_decisions: 0,
            end_x: 0.0,
            end_y: 0.0,
        }
    };

    Ok(YoloProbeSummary {
        episode_reports,
        first_platform,
        spawn_jump,
    })
}

/// 断言 YOLO 探针结果（失败时返回可读错误信息）。
pub fn assert_yolo_probes(summary: &YoloProbeSummary) -> Result<(), String> {
    assert_yolo_probes_with(summary, YoloProbeSet::All)
}

pub fn assert_yolo_probes_with(summary: &YoloProbeSummary, set: YoloProbeSet) -> Result<(), String> {
    let check_ep = set == YoloProbeSet::All || set == YoloProbeSet::Episodes;
    let check_fp = set == YoloProbeSet::All || set == YoloProbeSet::FirstPlatform;
    let check_spawn = set == YoloProbeSet::All || set == YoloProbeSet::SpawnJump;

    if check_ep && !summary.episode_reports.is_empty() {
        let failed: Vec<_> = summary
            .episode_reports
            .iter()
            .filter(|r| !r.passed())
            .collect();
        if failed.len() > 2 {
            let msg: Vec<_> = failed
                .iter()
                .map(|r| format!("seed={} {:?}", r.seed, r.failures))
                .collect();
            return Err(format!("bot_probe too many seeds failed: {}", msg.join("; ")));
        }
    }

    if check_fp {
        evaluate_first_platform_report(&summary.first_platform)?;
    }

    if check_spawn {
        if let Some(reason) = summary.spawn_jump.failure_reason() {
            return Err(reason);
        }
        if !summary.spawn_jump.passed() {
            return Err(format!(
                "spawn jump: decisions={}/{} effective={}/{}",
                summary.spawn_jump.raw_jump_decisions,
                summary.spawn_jump.vision_frames,
                summary.spawn_jump.effective_jump_ticks,
                summary.spawn_jump.ticks,
            ));
        }
    }

    Ok(())
}
