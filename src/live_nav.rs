//! 外挂式 Live Nav：截取 mini_game 窗口 → YOLO + SelfTracker → NavBot → SendInput。
//! 与 GameSim / mini_game 源码无编译期耦合，仅按窗口标题/进程附着。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use mxd_tools::game::nav::{GlobalStuckWatchdog, NavBot, NavBotConfig, SubGoal};
use mxd_tools::game::{
    default_yolo_model_path, HumanPace, InputFrame, SelfTracker, VisionObservation, VisionPipeline,
    VisionSenseState, VisionWorker, WINDOW_H, WINDOW_W, LOGIC_HZ, OBS_DIM, VISION_CONF_THRESH,
};
use mxd_tools::game::{load_default_map, GameMap};
use mxd_tools::yolo::YoloDevice;

use crate::keyboard_input::HeldKeys;
use crate::live_nav_diag::NavDiagLogger;
use crate::win_capture::{
    capture_client_rgb, find_mini_game_window, focus_window, window_alive, GameWindow,
};

pub enum LiveNavEvent {
    Log(String),
    Status(String),
    Stopped { reason: String },
}

/// 无 GameSim 的 Nav 驱动：观测 → decide → HumanPace。
struct LiveNavDriver {
    map: GameMap,
    bot: NavBot,
    pace: HumanPace,
    input: InputFrame,
    last_obs: [f32; OBS_DIM],
    sense: VisionSenseState,
    stuck: GlobalStuckWatchdog,
    episode_seed: u64,
}

impl LiveNavDriver {
    fn new(map: GameMap, seed: u64) -> Self {
        let (sx, sy) = map.default_spawn();
        let bot = NavBot::new(&map, NavBotConfig::default());
        let mut sense = VisionSenseState::default();
        sense.anchor_at(sx, sy);
        sense.est_x = sx;
        sense.est_y = sy;
        let mut stuck = GlobalStuckWatchdog::default();
        stuck.reset_tracking(sx, sy);
        Self {
            map,
            bot,
            pace: HumanPace::new(seed),
            input: InputFrame::default(),
            last_obs: [0.0; OBS_DIM],
            sense,
            stuck,
            episode_seed: seed,
        }
    }

    fn apply_observation(&mut self, vtick: u32, obs: [f32; OBS_DIM]) {
        self.last_obs = obs;
        self.sense.prepare(&obs);
        self.input = self.bot.decide(&self.map, &obs, &self.sense);
        self.sense.after_decide(&self.input, &obs);

        if let Some(why) = self.stuck.observe(
            self.sense.est_x,
            self.sense.est_y,
            self.bot.last_reason,
            &self.input,
        ) {
            self.hard_reset_from_stuck(why);
            let r = self.bot.last_reason;
            if r != "global_stuck_resume_climb" && r != "global_stuck_abandon_rope" {
                self.input = InputFrame::default();
            }
        }
        self.pace.on_intent(self.input, vtick);
    }

    fn hard_reset_from_stuck(&mut self, why: &'static str) {
        let px = self.sense.est_x;
        let py = self.sense.est_y;
        let rope_x = self
            .map
            .rope_at(px, py)
            .map(|r| r.x)
            .or_else(|| self.bot.at_climb_top_platform(px, py))
            .unwrap_or(px);

        let on_rope = self.sense.climbing
            || self.bot.last_reason.contains("climb")
            || why.contains("climb");
        let at_rope_top = self.sense.climbing
            && self.map.rope_at(px, py).is_some_and(|r| {
                let top = r.y1.min(r.y2);
                (py - top).abs() <= 10.0
            });
        let yoyo = if at_rope_top {
            self.stuck.note_rope_resume(rope_x) || self.stuck.should_abandon_rope(rope_x)
        } else {
            self.stuck.should_abandon_rope(rope_x)
        };

        if on_rope && (at_rope_top || yoyo) {
            self.pace.reset(self.episode_seed.wrapping_add(5));
            self.bot.abandon_rope(&self.map, px, py, rope_x);
            self.sense = VisionSenseState::default();
            self.sense.anchor_at(px, py);
            self.sense.est_x = px;
            self.sense.est_y = py;
            self.sense.climbing = false;
            self.stuck.clear_rope_yoyo();
            self.stuck.note_fired(px, py);
            self.stuck.last_fire = Some("global_stuck_abandon_rope");
            self.input = InputFrame {
                left: px >= rope_x,
                right: px < rope_x,
                ..InputFrame::default()
            };
            return;
        }

        if on_rope && !at_rope_top {
            self.pace.reset(self.episode_seed.wrapping_add(3));
            self.input = InputFrame::default();
            let resumed = self.bot.force_resume_climb(&self.map, px, py);
            let mid = self.bot.last_reason == "global_stuck_mid_ascent";
            if resumed || mid {
                self.sense = VisionSenseState::default();
                self.sense.anchor_at(px, py);
                self.sense.est_x = px;
                self.sense.est_y = py;
                self.sense.climbing = !mid;
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
            if self.bot.at_climb_top_platform(px, py).is_some() && py < 1100.0 {
                self.bot.soft_reset_keep_progress(&self.map, px, py);
                let node = self.bot.localizer_node();
                self.bot.note_mid_climb_landing(node);
                self.sense = VisionSenseState::default();
                self.sense.anchor_at(px, py);
                self.sense.est_x = px;
                self.sense.est_y = py;
                self.sense.climbing = false;
                self.stuck.note_fired(px, py);
                self.stuck.last_fire = Some("global_stuck_mid_ascent");
                self.input = InputFrame::default();
                return;
            }
        }

        let (kept_visited, kept_farm, kept_dir) = self.bot.snapshot_explore_progress();
        self.episode_seed = self.episode_seed.wrapping_add(17);
        self.bot.soft_reset_keep_progress(&self.map, px, py);
        self.pace.reset(self.episode_seed);
        self.bot
            .restore_explore_progress(kept_visited, kept_farm, kept_dir);
        self.bot.last_reason = why;
        self.sense = VisionSenseState::default();
        self.sense.anchor_at(px, py);
        self.sense.est_x = px;
        self.sense.est_y = py;
        self.stuck.note_fired(px, py);
        self.stuck.last_fire = Some(why);
        self.input = InputFrame::default();
    }

    fn paced_input(&mut self, tick: u32) -> InputFrame {
        self.input = self
            .bot
            .refresh_melee_hold(&self.last_obs, self.sense.facing);
        let intent = self.input;
        let climbing = self.sense.climbing;
        let mut paced = self.pace.apply(intent, tick);
        let climb_goal = matches!(
            self.bot.active_goal(),
            SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. }
        );
        if !climb_goal {
            paced = self
                .pace
                .apply_locomotion_hold(paced, tick, climbing, intent);
        } else if climbing {
            paced.left = false;
            paced.right = false;
            paced.jump = false;
        }
        paced = self.pace.finalize_output(paced, tick);
        self.sense.note_effective(&paced);
        paced
    }

    fn status_line(&self) -> String {
        format!(
            "reason={} est=({:.0},{:.0}) climb={}",
            self.bot.last_reason,
            self.sense.est_x,
            self.sense.est_y,
            self.sense.climbing as u8,
        )
    }
}

fn obs_from_step(step: &mxd_tools::game::VisionStep) -> [f32; OBS_DIM] {
    let mut obs = [0.0_f32; OBS_DIM];
    let n = step.observation.values.len().min(OBS_DIM);
    obs[..n].copy_from_slice(&step.observation.values[..n]);
    obs
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// 动态感知：上限 `max_hz`，仅在 YOLO 空闲时提交；慢机自动降频，不排队堆叠。
struct AdaptiveVisionPace {
    max_hz: f32,
    min_interval_ticks: u32,
    last_submit_tick: Option<u32>,
    /// 指数滑动：最近感知耗时（ms），用于日志与间隔微调。
    ema_perceive_ms: f64,
    last_queue_ms: f64,
}

impl AdaptiveVisionPace {
    fn new(max_hz: f32) -> Self {
        let max_hz = max_hz.clamp(1.0, LOGIC_HZ);
        let min_interval_ticks = (LOGIC_HZ / max_hz).round().clamp(1.0, LOGIC_HZ) as u32;
        Self {
            max_hz,
            min_interval_ticks,
            last_submit_tick: None,
            ema_perceive_ms: 1000.0 / f64::from(max_hz),
            last_queue_ms: 0.0,
        }
    }

    fn on_result(&mut self, perceive_ms: f64, queue_ms: f64) {
        const ALPHA: f64 = 0.3;
        self.ema_perceive_ms = if self.ema_perceive_ms <= 0.0 {
            perceive_ms
        } else {
            ALPHA * perceive_ms + (1.0 - ALPHA) * self.ema_perceive_ms
        };
        self.last_queue_ms = queue_ms;
    }

    fn note_submit(&mut self, tick: u32) {
        self.last_submit_tick = Some(tick);
    }

    /// 在途必须为 0；且距上次提交至少 `min_interval`，并按 EMA 感知耗时再拉长间隔。
    fn should_capture(&self, tick: u32, in_flight: u32) -> bool {
        if in_flight > 0 {
            return false;
        }
        let Some(last) = self.last_submit_tick else {
            return true;
        };
        let elapsed = tick.saturating_sub(last);
        if elapsed < self.min_interval_ticks {
            return false;
        }
        // 感知慢时额外拉大间隔：约 perceive_ms 对应的逻辑帧数（+10% 余量）。
        let need_from_ema = ((self.ema_perceive_ms * 1.1 / 1000.0) * f64::from(LOGIC_HZ))
            .ceil()
            .clamp(1.0, f64::from(LOGIC_HZ)) as u32;
        let need = self.min_interval_ticks.max(need_from_ema);
        elapsed >= need
    }

    fn status_hint(&self) -> String {
        let ema_hz = if self.ema_perceive_ms > 1.0 {
            1000.0 / self.ema_perceive_ms
        } else {
            f64::from(self.max_hz)
        };
        let capped = ema_hz.min(f64::from(self.max_hz));
        format!(
            "vision≤{:.0}Hz ~{:.1}Hz(ema {:.0}ms q={:.0})",
            self.max_hz, capped, self.ema_perceive_ms, self.last_queue_ms
        )
    }
}

/// 逻辑环性能窗口：分段耗时 + 实际/理论帧率。
struct PerfWindow {
    started: Instant,
    frames: u32,
    sum_poll_ms: f64,
    sum_capture_ms: f64,
    sum_decide_ms: f64,
    sum_input_ms: f64,
    sum_work_ms: f64,
    sum_sleep_ms: f64,
    sum_frame_ms: f64,
    max_work_ms: f64,
    max_frame_ms: f64,
    capture_frames: u32,
    vision_results: u32,
    sum_yolo_perceive_ms: f64,
    sum_yolo_queue_ms: f64,
    max_yolo_perceive_ms: f64,
}

impl PerfWindow {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            frames: 0,
            sum_poll_ms: 0.0,
            sum_capture_ms: 0.0,
            sum_decide_ms: 0.0,
            sum_input_ms: 0.0,
            sum_work_ms: 0.0,
            sum_sleep_ms: 0.0,
            sum_frame_ms: 0.0,
            max_work_ms: 0.0,
            max_frame_ms: 0.0,
            capture_frames: 0,
            vision_results: 0,
            sum_yolo_perceive_ms: 0.0,
            sum_yolo_queue_ms: 0.0,
            max_yolo_perceive_ms: 0.0,
        }
    }

    fn note_yolo(&mut self, perceive_ms: f64, queue_ms: f64) {
        self.vision_results = self.vision_results.saturating_add(1);
        self.sum_yolo_perceive_ms += perceive_ms;
        self.sum_yolo_queue_ms += queue_ms;
        self.max_yolo_perceive_ms = self.max_yolo_perceive_ms.max(perceive_ms);
    }

    fn note_frame(
        &mut self,
        poll_ms: f64,
        capture_ms: f64,
        captured: bool,
        decide_ms: f64,
        input_ms: f64,
        work_ms: f64,
        sleep_ms: f64,
        frame_ms: f64,
    ) {
        self.frames = self.frames.saturating_add(1);
        self.sum_poll_ms += poll_ms;
        self.sum_capture_ms += capture_ms;
        if captured {
            self.capture_frames = self.capture_frames.saturating_add(1);
        }
        self.sum_decide_ms += decide_ms;
        self.sum_input_ms += input_ms;
        self.sum_work_ms += work_ms;
        self.sum_sleep_ms += sleep_ms;
        self.sum_frame_ms += frame_ms;
        self.max_work_ms = self.max_work_ms.max(work_ms);
        self.max_frame_ms = self.max_frame_ms.max(frame_ms);
    }

    fn avg(sum: f64, n: u32) -> f64 {
        if n == 0 {
            0.0
        } else {
            sum / f64::from(n)
        }
    }

    /// 返回 (status 短句, 详情日志)。
    fn flush_report(&mut self, target_hz: f32) -> Option<(String, String)> {
        if self.frames == 0 {
            return None;
        }
        let wall = self.started.elapsed().as_secs_f64().max(1e-6);
        let n = self.frames;
        let loop_fps = f64::from(n) / wall;
        let avg_work = Self::avg(self.sum_work_ms, n);
        let capacity_fps = if avg_work > 1e-6 {
            1000.0 / avg_work
        } else {
            f64::from(target_hz) * 10.0
        };
        let avg_frame = Self::avg(self.sum_frame_ms, n);
        let avg_poll = Self::avg(self.sum_poll_ms, n);
        let avg_capture = if self.capture_frames > 0 {
            Self::avg(self.sum_capture_ms, self.capture_frames)
        } else {
            0.0
        };
        let avg_decide = Self::avg(self.sum_decide_ms, n);
        let avg_input = Self::avg(self.sum_input_ms, n);
        let avg_sleep = Self::avg(self.sum_sleep_ms, n);
        let yolo_n = self.vision_results;
        let avg_yolo = Self::avg(self.sum_yolo_perceive_ms, yolo_n);
        let avg_queue = Self::avg(self.sum_yolo_queue_ms, yolo_n);
        let yolo_hz = f64::from(yolo_n) / wall;

        let status = format!(
            "loop={:.1}fps work={:.2}ms(cap≈{:.0}fps) yolo={:.1}Hz/{:.0}ms",
            loop_fps, avg_work, capacity_fps, yolo_hz, avg_yolo
        );
        let detail = format!(
            "性能[{:.1}s/{n}帧] 实际循环={loop_fps:.1}fps (目标{target_hz:.0}) | \
运算均={avg_work:.2}ms 峰={:.2}ms → 理论可持续≈{capacity_fps:.0}fps | \
整帧均={avg_frame:.2}ms 峰={:.2}ms | \
poll/决策={avg_poll:.2}ms 截图={avg_capture:.2}ms×{} 节奏={avg_decide:.2}ms 按键={avg_input:.2}ms 等待={avg_sleep:.2}ms | \
YOLO完成={yolo_n} 均感知={avg_yolo:.1}ms 峰={:.1}ms 排队={avg_queue:.1}ms 感知吞吐={yolo_hz:.1}/s",
            wall,
            self.max_work_ms,
            self.max_frame_ms,
            self.capture_frames,
            self.max_yolo_perceive_ms,
        );
        *self = Self::new();
        Some((status, detail))
    }
}

/// 后台寻路线程入口。
pub fn run_live_nav(stop: Arc<AtomicBool>, tx: Sender<LiveNavEvent>) {
    let send = |ev: LiveNavEvent| {
        let _ = tx.send(ev);
    };

    if let Err(e) = run_live_nav_inner(stop, &send) {
        send(LiveNavEvent::Stopped {
            reason: format!("异常退出：{e:#}"),
        });
    }
}

fn run_live_nav_inner(
    stop: Arc<AtomicBool>,
    send: &dyn Fn(LiveNavEvent),
) -> Result<()> {
    send(LiveNavEvent::Log(
        "正在加载地图与 YOLO 模型…".into(),
    ));
    let map = load_default_map().context("加载默认地图 50001")?;
    let model = default_yolo_model_path();
    let pipeline = VisionPipeline::load(&model, YoloDevice::Cpu, VISION_CONF_THRESH)
        .with_context(|| format!("加载 YOLO：{}", model.display()))?;
    let mut worker = VisionWorker::spawn(pipeline);
    let mut driver = LiveNavDriver::new(map, 42);
    let mut keys = HeldKeys::new();
    let mut tracker = SelfTracker::new();
    let mut last_commanded_dx = 0.0_f32;
    let mut pending_commanded_dx: Option<f32> = None;
    // 与 game_preview 默认 --detect-hz 10 对齐，便于预览日志对照真机。
    let mut vision = AdaptiveVisionPace::new(10.0);
    let target_w = WINDOW_W as u32;
    let target_h = WINDOW_H as u32;

    send(LiveNavEvent::Log(format!(
        "模型 {}；感知动态上限 {:.0} Hz（空闲才截图+推理，不堆叠）；逻辑目标 {} Hz；请保持「{}」在前台",
        model.display(),
        vision.max_hz,
        LOGIC_HZ,
        crate::win_capture::MINI_GAME_TITLE,
    )));
    send(LiveNavEvent::Log(
        "性能日志：每 2s 输出分段耗时与实际/理论帧率（理论=1000/运算均耗时）".into(),
    ));

    let mut hwnd_cache: Option<GameWindow> = None;
    let mut tick: u32 = 0;
    let mut last_status = Instant::now();
    let mut consecutive_capture_fail: u32 = 0;
    let mut perf = PerfWindow::new();
    let mut last_perf_status = String::new();
    let mut diag_log = NavDiagLogger::default();

    send(LiveNavEvent::Log(
        "诊断日志：跳台/边缘/有梯不爬；UTF-8 同时写入 tmp/nav_diag.log（比 Tee-Object 更干净）".into(),
    ));

    while !stop.load(Ordering::SeqCst) {
        let tick_start = Instant::now();
        let mut capture_ms = 0.0_f64;
        let mut captured = false;

        if worker.is_dead() {
            anyhow::bail!("YOLO 视觉线程已退出");
        }

        let t0 = Instant::now();
        while let Some(result) = worker.poll_result() {
            vision.on_result(result.perceive_ms, result.queue_wait_ms);
            perf.note_yolo(result.perceive_ms, result.queue_wait_ms);
            let cmd = pending_commanded_dx
                .take()
                .unwrap_or(last_commanded_dx);
            let hit = tracker.update(&result.step.detections, cmd, VISION_CONF_THRESH);
            let mut step = result.step;
            step.self_player = hit.clone();
            step.observation = VisionObservation::from_detections(
                &step.detections,
                hit.as_ref(),
                WINDOW_W as u32,
                WINDOW_H as u32,
            );
            let obs = obs_from_step(&step);
            driver.apply_observation(result.tick, obs);
            for line in diag_log.on_vision(
                result.tick,
                &step,
                &driver.last_obs,
                &driver.bot,
                &driver.sense,
                &driver.input,
            ) {
                send(LiveNavEvent::Log(line));
            }
        }
        let poll_ms = ms(t0.elapsed());

        if vision.should_capture(tick, worker.in_flight_frames()) {
            let win = match resolve_window(&mut hwnd_cache) {
                Ok(w) => w,
                Err(e) => {
                    consecutive_capture_fail = consecutive_capture_fail.saturating_add(1);
                    if consecutive_capture_fail == 1 || consecutive_capture_fail % 60 == 0 {
                        send(LiveNavEvent::Log(format!("等待游戏窗口：{e}")));
                    }
                    let work_ms = ms(tick_start.elapsed());
                    let sleep_ms = sleep_logic(tick_start);
                    let frame_ms = ms(tick_start.elapsed());
                    perf.note_frame(
                        poll_ms, 0.0, false, 0.0, 0.0, work_ms, sleep_ms, frame_ms,
                    );
                    maybe_emit_perf(
                        &mut perf,
                        &mut last_status,
                        &mut last_perf_status,
                        &driver,
                        &vision,
                        send,
                    );
                    tick = tick.wrapping_add(1);
                    continue;
                }
            };
            if consecutive_capture_fail > 0 {
                send(LiveNavEvent::Log(format!(
                    "已附着窗口：{} (hwnd={:#x})",
                    win.title, win.hwnd as u64
                )));
                consecutive_capture_fail = 0;
            }
            let tc = Instant::now();
            match capture_client_rgb(win.as_hwnd(), target_w, target_h) {
                Ok(rgb) => {
                    capture_ms = ms(tc.elapsed());
                    captured = true;
                    if worker.try_submit(tick, rgb, None) {
                        vision.note_submit(tick);
                        pending_commanded_dx = Some(last_commanded_dx);
                    }
                }
                Err(e) => {
                    capture_ms = ms(tc.elapsed());
                    consecutive_capture_fail = consecutive_capture_fail.saturating_add(1);
                    hwnd_cache = None;
                    if consecutive_capture_fail <= 3 || consecutive_capture_fail % 30 == 0 {
                        send(LiveNavEvent::Log(format!("截图失败：{e}")));
                    }
                }
            }
        }

        let td = Instant::now();
        let paced = if tracker.needs_probe() {
            tracker.probe_input()
        } else {
            driver.paced_input(tick)
        };
        last_commanded_dx = paced.horizontal();
        let decide_ms = ms(td.elapsed());

        let ti = Instant::now();
        if let Err(e) = keys.sync_frame(&paced) {
            send(LiveNavEvent::Log(format!("SendInput 失败：{e}")));
        }
        let input_ms = ms(ti.elapsed());

        let work_ms = ms(tick_start.elapsed());
        let sleep_ms = sleep_logic(tick_start);
        let frame_ms = ms(tick_start.elapsed());
        perf.note_frame(
            poll_ms,
            capture_ms,
            captured,
            decide_ms,
            input_ms,
            work_ms,
            sleep_ms,
            frame_ms,
        );
        maybe_emit_perf(
            &mut perf,
            &mut last_status,
            &mut last_perf_status,
            &driver,
            &vision,
            send,
        );

        tick = tick.wrapping_add(1);
    }

    if let Some((status, detail)) = perf.flush_report(LOGIC_HZ) {
        send(LiveNavEvent::Status(format!(
            "{} | {} | {}",
            driver.status_line(),
            vision.status_hint(),
            status
        )));
        send(LiveNavEvent::Log(detail));
    }

    let _ = keys.release_all();
    send(LiveNavEvent::Stopped {
        reason: "已停止寻路".into(),
    });
    Ok(())
}

fn maybe_emit_perf(
    perf: &mut PerfWindow,
    last_status: &mut Instant,
    last_perf_status: &mut String,
    driver: &LiveNavDriver,
    vision: &AdaptiveVisionPace,
    send: &dyn Fn(LiveNavEvent),
) {
    if last_status.elapsed() < Duration::from_secs(2) {
        return;
    }
    if let Some((status, detail)) = perf.flush_report(LOGIC_HZ) {
        *last_perf_status = status.clone();
        send(LiveNavEvent::Status(format!(
            "{} | {} | {}",
            driver.status_line(),
            vision.status_hint(),
            status
        )));
        send(LiveNavEvent::Log(format!(
            "{} | {}",
            vision.status_hint(),
            detail
        )));
    } else {
        send(LiveNavEvent::Status(format!(
            "{} | {} | {}",
            driver.status_line(),
            vision.status_hint(),
            last_perf_status
        )));
    }
    *last_status = Instant::now();
}

fn resolve_window(cache: &mut Option<GameWindow>) -> Result<GameWindow, String> {
    if let Some(w) = cache.as_ref() {
        if window_alive(w.as_hwnd()) {
            return Ok(w.clone());
        }
    }
    let w = find_mini_game_window().ok_or_else(|| {
        format!(
            "未找到「{}」窗口，请先单独运行：cargo run --bin mini_game",
            crate::win_capture::MINI_GAME_TITLE
        )
    })?;
    let _ = focus_window(w.as_hwnd());
    *cache = Some(w.clone());
    Ok(w)
}

fn sleep_logic(tick_start: Instant) -> f64 {
    let frame = Duration::from_secs_f64(1.0 / LOGIC_HZ as f64);
    let elapsed = tick_start.elapsed();
    if elapsed < frame {
        let wait = frame - elapsed;
        std::thread::sleep(wait);
        ms(wait)
    } else {
        0.0
    }
}
