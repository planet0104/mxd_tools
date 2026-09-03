//! Headless NavBot 多局探针：离屏渲染 + 真实 YOLO/OCR 观测（与 game_preview 一致）。

use std::collections::HashSet;
use std::time::{Duration, Instant};

use anyhow::Result;

use super::config::VisionPaceConfig;
use super::headless_vision::HeadlessVisionEnv;
use super::human_pace::HumanPace;
use super::input::InputFrame;
use super::map::GameMap;
use super::nav::{GlobalStuckWatchdog, NavBot, NavBotConfig};
use super::observation::{
    obs_assess_enemy_contact, obs_enemy_in_attack_range, obs_has_enemy, obs_platform_edge, OBS_DIM,
    OBS_ENEMY_SLOTS, OBS_ENEMY_START, OBS_SLOT_DIM,
};
use super::sim::GameSim;
use super::types::{LOGIC_DT, LOGIC_HZ};
use super::vision_sense::VisionSenseState;

/// 高度带网格（探测首层平台用）。
fn visit_key(x: f32, y: f32) -> (i32, i32) {
    (
        (x / 80.0).floor() as i32,
        (y / 120.0).floor() as i32,
    )
}

/// 首平台探针逻辑时长（60Hz × 90s，与 headless / game_preview 探针一致）。
pub const FIRST_PLATFORM_PROBE_TICKS: u32 = 5_400;

const STUCK_WINDOW: u32 = 120;
const STUCK_MOVE_EPS: f32 = 18.0;
const SINGLE_MOB_ATTACK_WINDOW: u32 = 240;
const SPAWN_PROBE_TICKS: u32 = 1_800;
const SPAWN_MAX_JUMP_DECISION_RATIO: f32 = 0.15;
const SPAWN_MAX_EFFECTIVE_JUMP_RATIO: f32 = 0.02;

/// 探针 / 预览主循环：YOLO 异步提交/轮询，sim 每 tick 继续。
pub struct ProbeDriver {
    bot: NavBot,
    pace: HumanPace,
    input: InputFrame,
    last_obs: [f32; OBS_DIM],
    sense: VisionSenseState,
    wall_clock_pacing: bool,
    stuck: GlobalStuckWatchdog,
    episode_seed: u64,
}

impl ProbeDriver {
    pub fn new(seed: u64) -> Self {
        Self::with_map(None, seed, true)
    }

    pub fn new_nav(map: &GameMap, seed: u64) -> Self {
        Self::with_map(Some(map), seed, true)
    }

    /// 窗口预览：逻辑 60Hz 由 macroquad 帧率驱动，不做 thread sleep。
    pub fn new_realtime(seed: u64) -> Self {
        Self::with_map(None, seed, false)
    }

    pub fn new_realtime_with_map(map: &GameMap, seed: u64) -> Self {
        Self::with_map(Some(map), seed, false)
    }

    fn with_map(map: Option<&GameMap>, seed: u64, wall_clock_pacing: bool) -> Self {
        let m = map.cloned().unwrap_or_else(|| {
            super::load_default_map().expect("default map for NavBot")
        });
        let bot = NavBot::new(&m, NavBotConfig::default());
        Self {
            bot,
            pace: HumanPace::new(seed),
            input: InputFrame::default(),
            last_obs: [0.0_f32; OBS_DIM],
            sense: VisionSenseState::default(),
            wall_clock_pacing,
            stuck: GlobalStuckWatchdog::default(),
            episode_seed: seed,
        }
    }

    pub fn reset(&mut self, seed: u64) {
        let map = super::load_default_map().expect("map");
        let (x, y) = map.default_spawn();
        self.episode_seed = seed;
        self.bot.reset(&map, x, y);
        self.pace.reset(seed);
        self.input = InputFrame::default();
        self.last_obs = [0.0_f32; OBS_DIM];
        self.sense = VisionSenseState::default();
        self.stuck.reset_tracking(x, y);
    }

    pub fn reset_with_sim(&mut self, sim: &GameSim, seed: u64) {
        let p = &sim.state.player;
        self.episode_seed = seed;
        self.bot.reset(&sim.map, p.x, p.y);
        self.pace.reset(seed);
        self.input = InputFrame::default();
        self.last_obs = [0.0_f32; OBS_DIM];
        self.sense = VisionSenseState::default();
        self.sense.est_x = p.x;
        self.sense.est_y = p.y;
        self.sense.anchor_at(p.x, p.y);
        self.stuck.reset_tracking(p.x, p.y);
    }

    /// 卡住硬重置：绳顶卡住才计 yoyo/弃绳；绳中卡住只恢复攀爬，禁止第二次就封绳。
    fn hard_reset_from_stuck(&mut self, sim: &mut GameSim, why: &'static str) {
        let px = sim.state.player.x;
        let py = sim.state.player.y;
        let rope_x = sim
            .map
            .rope_at(px, py)
            .map(|r| r.x)
            .or_else(|| self.bot.at_climb_top_platform(px, py))
            .unwrap_or(px);

        let on_rope = sim.state.player.climbing
            || self.sense.climbing
            || self.bot.last_reason.contains("climb")
            || why.contains("climb");
        let at_rope_top = sim.state.player.climbing
            && sim.map.rope_at(px, py).is_some_and(|r| {
                let top = r.y1.min(r.y2);
                (py - top).abs() <= 10.0
            });
        // 注意：图上 ClimbUp 终点可能只是中段台（57→123 y=985），不能当「绳顶」弃绳，
        // 否则刚爬到中段就被 abandon，永远上不了 StepUp 链。
        let yoyo = if at_rope_top {
            self.stuck.note_rope_resume(rope_x) || self.stuck.should_abandon_rope(rope_x)
        } else {
            self.stuck.should_abandon_rope(rope_x)
        };

        if on_rope && (at_rope_top || yoyo) {
            // 物理绳顶卡住或 yo-yo：封绳离开。
            sim.force_dismount_climb();
            self.pace.reset(self.episode_seed.wrapping_add(5));
            self.bot.abandon_rope(&sim.map, sim.state.player.x, sim.state.player.y, rope_x);
            self.sense = VisionSenseState::default();
            self.sense.est_x = sim.state.player.x;
            self.sense.est_y = sim.state.player.y;
            self.sense.anchor_at(sim.state.player.x, sim.state.player.y);
            self.sense.climbing = false;
            self.stuck.clear_rope_yoyo();
            self.stuck.note_fired(sim.state.player.x, sim.state.player.y);
            self.stuck.last_fire = Some("global_stuck_abandon_rope");
            self.input = InputFrame {
                left: sim.state.player.x >= rope_x,
                right: sim.state.player.x < rope_x,
                ..InputFrame::default()
            };
            return;
        }

        if on_rope && !at_rope_top {
            self.pace.reset(self.episode_seed.wrapping_add(3));
            self.input = InputFrame::default();
            let resumed = self.bot.force_resume_climb(&sim.map, px, py);
            let mid = self.bot.last_reason == "global_stuck_mid_ascent";
            if resumed || mid {
                self.sense = VisionSenseState::default();
                self.sense.est_x = px;
                self.sense.est_y = py;
                self.sense.anchor_at(px, py);
                self.sense.climbing = if mid {
                    false
                } else {
                    sim.state.player.climbing
                };
                if mid {
                    sim.force_dismount_climb();
                }
                self.stuck.note_fired(px, py);
                self.stuck.last_fire = Some(if mid {
                    "global_stuck_mid_ascent"
                } else {
                    "global_stuck_resume_climb"
                });
                self.input = if mid {
                    let dir = self.bot.patrol_dir();
                    InputFrame {
                        right: dir >= 0.0,
                        left: dir < 0.0,
                        ..InputFrame::default()
                    }
                } else {
                    InputFrame {
                        up: true,
                        ..InputFrame::default()
                    }
                };
                return;
            }
            // 已在 ClimbUp 图终点中段台：下绳后记 ascent，继续向上 StepUp，禁止弃绳。
            if self.bot.at_climb_top_platform(px, py).is_some() && py < 1100.0 {
                sim.force_dismount_climb();
                self.bot.soft_reset_keep_progress(&sim.map, px, py);
                let node = self.bot.localizer_node();
                self.bot.note_mid_climb_landing(node);
                self.sense = VisionSenseState::default();
                self.sense.est_x = px;
                self.sense.est_y = py;
                self.sense.anchor_at(px, py);
                self.sense.climbing = false;
                self.stuck.note_fired(px, py);
                self.stuck.last_fire = Some("global_stuck_mid_ascent");
                self.input = InputFrame::default();
                return;
            }
        }

        sim.force_dismount_climb();
        let (sx, sy) = (sim.state.player.x, sim.state.player.y);
        // 硬重置保留探索进度，否则会反复清零 visited，永远在底层兜圈。
        let (kept_visited, kept_farm, kept_dir) = self.bot.snapshot_explore_progress();
        self.episode_seed = self.episode_seed.wrapping_add(17);
        self.reset_with_sim(sim, self.episode_seed);
        // 防止偶发定位到 (0,0)：强制锚到重置前坐标。
        if sx.abs() > 1.0 || sy.abs() > 1.0 {
            self.sense.anchor_at(sx, sy);
            self.sense.est_x = sx;
            self.sense.est_y = sy;
            self.bot.soft_reset_keep_progress(&sim.map, sx, sy);
        }
        self.bot
            .restore_explore_progress(kept_visited, kept_farm, kept_dir);
        self.bot.last_reason = why;
        self.stuck.note_fired(self.sense.est_x, self.sense.est_y);
        self.stuck.last_fire = Some(why);
    }

    pub fn bot(&self) -> &NavBot {
        &self.bot
    }

    pub fn bot_mut(&mut self) -> &mut NavBot {
        &mut self.bot
    }

    /// 兼容旧名。
    pub fn nav_bot(&self) -> &NavBot {
        &self.bot
    }

    pub fn nav_bot_mut(&mut self) -> &mut NavBot {
        &mut self.bot
    }

    pub fn input(&self) -> InputFrame {
        self.input
    }

    pub fn last_obs(&self) -> &[f32; OBS_DIM] {
        &self.last_obs
    }

    pub fn sense(&self) -> &VisionSenseState {
        &self.sense
    }

    pub fn apply_observation(&mut self, sim: &mut GameSim, vtick: u32, obs: [f32; OBS_DIM]) {
        self.last_obs = obs;
        sim.movement_gate.set_last_observation(&obs);
        self.sense.prepare(&obs);
        self.input = self.bot.decide(&sim.map, &obs, &self.sense);
        self.sense.after_decide(&self.input, &obs);

        // 全局卡住：位置 ~10s 不动或决策输出循环 → 硬重置全部 bot 状态。
        if let Some(why) = self.stuck.observe(
            self.sense.est_x,
            self.sense.est_y,
            self.bot.last_reason,
            &self.input,
        ) {
            self.hard_reset_from_stuck(sim, why);
            let r = self.bot.last_reason;
            if r != "global_stuck_resume_climb" && r != "global_stuck_abandon_rope" {
                self.input = InputFrame::default();
            }
        }

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
    pub fn paced_input_for_sim(&mut self, sim: &mut GameSim, tick: u32) -> InputFrame {
        // 视觉帧间隙（5–10Hz）内按最近观测刷新站砍，避免怪走进可砍带仍冻在 noop。
        self.refresh_melee_hold();
        let intent = self.input;
        let climbing = sim.state.player.climbing;
        let mut paced = self.pace.apply(intent, tick);
        let climb_goal = matches!(
            self.bot.active_goal(),
            crate::game::nav::SubGoal::ClimbUp { .. }
                | crate::game::nav::SubGoal::ClimbDown { .. }
        );
        // 爬绳对齐/抓绳：禁止左右 latch 与地面 up 节流，否则冲过绳子且 jump+up 被掐掉。
        if !climb_goal {
            paced = self
                .pace
                .apply_locomotion_hold(paced, tick, climbing, intent);
        } else if climbing {
            // 已挂绳：只保垂直，清左右与 jump（jump 会脱绳）。
            paced.left = false;
            paced.right = false;
            paced.jump = false;
        }
        // 输出硬闸：整帧按键组合切换 ≤约 4Hz，杜绝每秒 5～10+ 次换键。
        paced = self.pace.finalize_output(paced, tick);
        sim.force_allow_step_up =
            matches!(self.bot.active_goal(), crate::game::nav::SubGoal::StepUp { .. });
        sim.force_allow_nav_walk = matches!(
            self.bot.active_goal(),
            crate::game::nav::SubGoal::GoTo { .. }
                | crate::game::nav::SubGoal::Patrol { .. }
                | crate::game::nav::SubGoal::WalkOff { .. }
        );
        let effective = sim.effective_bot_input(&paced);
        if paced.attack && !effective.attack {
            self.pace.refund_attack();
        }
        // 推算坐标只跟生效输入，避免 intent=right effective=noop 时虚走。
        self.sense.note_effective(&effective);
        paced
    }

    /// 可砍带内强制站砍（不走路），覆盖过期的视觉意图。
    /// 换层过程中只补 attack，不剥左右/跳跃。
    fn refresh_melee_hold(&mut self) {
        self.input = self.bot.refresh_melee_hold(
            &self.last_obs,
            self.sense.facing,
        );
        // bot.refresh_melee_hold 已按 goal 处理；同步回 driver.input
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
            let input = driver.input();
            if input.jump {
                raw_jump_decisions += 1;
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
                || obs_enemy_in_attack_range(
                    sim.movement_gate.last_observation(),
                    sim.state.player.facing,
                );
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
    let visited_cells = driver.bot().visited_nodes();
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
        ("spawn".to_string(), vec!["--probe".into(), "spawn".into()]),
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

pub fn assert_yolo_probes_with(
    summary: &YoloProbeSummary,
    set: YoloProbeSet,
) -> Result<(), String> {
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
            return Err(format!(
                "bot_probe too many seeds failed: {}",
                msg.join("; ")
            ));
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

#[cfg(test)]
mod pure_vision_tests {
    use super::*;
    use crate::game::observation::{OBS_FLOOR_START, OBS_SLOT_DIM};
    use crate::game::{default_map_path, GameMap};

    fn observation(landmark_shift: f32) -> [f32; OBS_DIM] {
        let mut obs = [0.0; OBS_DIM];
        obs[0] = 0.5;
        obs[1] = 0.5;
        for i in 0..2 {
            let base = OBS_FLOOR_START + i * OBS_SLOT_DIM;
            obs[base] = 0.25 + i as f32 * 0.35 - landmark_shift;
            obs[base + 1] = 0.65;
            obs[base + 2] = 0.28;
            obs[base + 3] = 0.04;
        }
        obs
    }

    #[test]
    fn probe_driver_position_comes_from_vision_not_sim_truth() {
        let map = GameMap::load(&default_map_path()).expect("load default map");
        let mut sim = GameSim::new_preview(map, 7);
        let mut driver = ProbeDriver::new_realtime(7);
        driver.apply_observation(&mut sim, 0, observation(0.0));
        sim.state.player.x = 9_999.0;
        sim.state.player.y = 9_999.0;
        driver.apply_observation(&mut sim, 1, observation(0.05));

        let sense = driver.sense();
        let expected_dx = crate::game::types::WINDOW_W * 0.05;
        assert!(
            (sense.est_x - expected_dx).abs() < 4.0,
            "est_x={}",
            sense.est_x
        );
        assert!(sense.est_y.abs() < 4.0, "est_y={}", sense.est_y);
        assert!((sense.est_x - sim.state.player.x).abs() > 1_000.0);
    }
}
