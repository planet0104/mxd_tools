//! NEAT 个体驱动：纯 YOLO+OCR 观测 → OCR 脚点本体反馈 → 前向网络 → 动作宏 → 输入帧。
//!
//! 网络输入不含任何 sim 物理通道；自身位置/位移/卡住一律 OCR；动作宏的完成/失败判定同样只用
//! YOLO 槽位 + OCR 位移 + tick 超时，因此训练与真机部署行为一致（无 MovementGate）。

use anyhow::Result;

use crate::game::headless_vision::HeadlessVisionEnv;
use crate::game::input::InputFrame;
use crate::game::macro_action::{MacroAction, MacroRunner};
use crate::game::observation::{inject_proprioception, OBS_DIM};
use crate::game::sim::GameSim;
use crate::game::types::{WINDOW_H, WINDOW_W};
use crate::neat::genome::Genome;
use crate::player_name::NamedPlayerHit;
use super::network::{action_from_outputs, evaluate};
use super::obs_compact::compact_obs;

/// 10Hz 下正常走路约 12px/感知帧；站桩时 OCR 名牌抖动可达 ±10~14px（见 preview diag），
/// 单帧 |Δx| 阈值会被噪声打穿。改用「连续按方向时的多帧净位移」。
const STUCK_NET_FRAMES: u8 = 5;
/// 连续 STUCK_NET_FRAMES 帧内，朝意图方向的净位移小于此值 → blocked。
/// 正常走路 5 帧约 60px；站桩噪声大致抵消，净位移通常 <25px。
const STUCK_NET_PROGRESS_PX: f32 = 25.0;

pub struct NeatDriver {
    genome: Genome,
    runner: MacroRunner,
    input: InputFrame,
    last_obs: [f32; OBS_DIM],
    prev_ocr_x: f32,
    prev_ocr_y: f32,
    has_prev_ocr: bool,
    stuck_left_anchor_x: Option<f32>,
    stuck_left_frames: u8,
    stuck_left_latched: bool,
    stuck_right_anchor_x: Option<f32>,
    stuck_right_frames: u8,
    stuck_right_latched: bool,
}

impl NeatDriver {
    pub fn new(genome: Genome) -> Self {
        Self {
            genome,
            runner: MacroRunner::default(),
            input: InputFrame::default(),
            last_obs: [0.0_f32; OBS_DIM],
            prev_ocr_x: 0.0,
            prev_ocr_y: 0.0,
            has_prev_ocr: false,
            stuck_left_anchor_x: None,
            stuck_left_frames: 0,
            stuck_left_latched: false,
            stuck_right_anchor_x: None,
            stuck_right_frames: 0,
            stuck_right_latched: false,
        }
    }

    pub fn set_genome(&mut self, genome: Genome) {
        self.genome = genome;
        self.runner.reset();
        self.input = InputFrame::default();
        self.last_obs = [0.0_f32; OBS_DIM];
        self.prev_ocr_x = 0.0;
        self.prev_ocr_y = 0.0;
        self.has_prev_ocr = false;
        self.clear_stuck_trackers();
    }

    fn clear_stuck_trackers(&mut self) {
        self.stuck_left_anchor_x = None;
        self.stuck_left_frames = 0;
        self.stuck_left_latched = false;
        self.stuck_right_anchor_x = None;
        self.stuck_right_frames = 0;
        self.stuck_right_latched = false;
    }

    pub fn genome(&self) -> &Genome {
        &self.genome
    }

    pub fn input(&self) -> InputFrame {
        self.input
    }

    pub fn last_obs(&self) -> &[f32; OBS_DIM] {
        &self.last_obs
    }

    /// 多帧净位移卡住判定；`progress_px` 为朝意图方向的正进度。
    fn update_stuck_axis(
        want: bool,
        ocr_x: f32,
        progress_from_anchor: impl Fn(f32, f32) -> f32,
        anchor: &mut Option<f32>,
        frames: &mut u8,
        latched: &mut bool,
    ) -> bool {
        if !want {
            *anchor = None;
            *frames = 0;
            *latched = false;
            return false;
        }
        let a = match *anchor {
            Some(v) => v,
            None => {
                *anchor = Some(ocr_x);
                *frames = 1;
                return false;
            }
        };
        *frames = frames.saturating_add(1);
        if *frames < STUCK_NET_FRAMES {
            return *latched;
        }
        let progress = progress_from_anchor(a, ocr_x);
        if progress >= STUCK_NET_PROGRESS_PX {
            *latched = false;
        } else {
            *latched = true;
        }
        // 滚动新窗口，避免噪声长期积分漂移；latched 在窗口间保持。
        *anchor = Some(ocr_x);
        *frames = 1;
        *latched
    }

    /// `ocr_feet`：本帧 OCR 名牌脚点；`None` 表示丢检，不用 sim 坐标顶替。
    pub fn apply_observation(
        &mut self,
        sim: &mut GameSim,
        mut obs: [f32; OBS_DIM],
        ocr_feet: Option<(f32, f32)>,
    ) {
        let climbing = sim.state.player.climbing;
        let (last_dx, last_dy, blocked_left, blocked_right) = match ocr_feet {
            Some((px, py)) if self.has_prev_ocr => {
                let raw_dx = px - self.prev_ocr_x;
                let raw_dy = py - self.prev_ocr_y;
                let last_dx = (raw_dx / WINDOW_W).clamp(-1.0, 1.0);
                let last_dy = (raw_dy / WINDOW_H).clamp(-1.0, 1.0);
                // 爬绳时水平键不表示走路，OCR 水平抖动会误报 blocked。
                let (blocked_left, blocked_right) = if climbing {
                    self.clear_stuck_trackers();
                    (false, false)
                } else {
                    let want_left = self.input.left && !self.input.right;
                    let want_right = self.input.right && !self.input.left;
                    let bl = Self::update_stuck_axis(
                        want_left,
                        px,
                        |a, x| a - x,
                        &mut self.stuck_left_anchor_x,
                        &mut self.stuck_left_frames,
                        &mut self.stuck_left_latched,
                    );
                    let br = Self::update_stuck_axis(
                        want_right,
                        px,
                        |a, x| x - a,
                        &mut self.stuck_right_anchor_x,
                        &mut self.stuck_right_frames,
                        &mut self.stuck_right_latched,
                    );
                    (bl, br)
                };
                self.prev_ocr_x = px;
                self.prev_ocr_y = py;
                (last_dx, last_dy, blocked_left, blocked_right)
            }
            Some((px, py)) => {
                self.prev_ocr_x = px;
                self.prev_ocr_y = py;
                self.has_prev_ocr = true;
                self.clear_stuck_trackers();
                (0.0, 0.0, false, false)
            }
            None => (0.0, 0.0, false, false),
        };
        inject_proprioception(
            &mut obs,
            last_dx,
            last_dy,
            blocked_left,
            blocked_right,
            self.input.left,
            self.input.right,
            self.input.jump,
            self.input.attack,
        );

        self.last_obs = obs;
        if sim.config.training {
            sim.fitness.set_last_observation(&obs);
        }
        // 宏执行中不重新决策：一个意图必须跑完，否则跳台/爬绳会被每帧抖动撕碎。
        self.runner.observe(&obs);
        if self.runner.is_idle() {
            let compact = compact_obs(&obs, self.runner.last_failed());
            let outputs = evaluate(&self.genome, &compact);
            self.runner.begin(action_from_outputs(&outputs), &obs);
        }
    }

    pub fn current_action(&self) -> Option<MacroAction> {
        self.runner.last_action()
    }

    fn ocr_feet_from_hit(hit: Option<&NamedPlayerHit>) -> Option<(f32, f32)> {
        hit.map(|p| (p.x, p.y))
    }

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
        if let Some((vtick, obs, step)) = vision.poll_vision(sim) {
            sim.record_vision_loot(&step.detections);
            self.apply_observation(sim, obs, Self::ocr_feet_from_hit(step.self_player.as_ref()));
            return Ok(Some((vtick, self.last_obs)));
        }
        if tick % vision_interval == 0 {
            let _ = vision.schedule_capture_if_ready(tick, sim).await?;
        }
        Ok(None)
    }

    pub fn logic_tick_preview(
        &mut self,
        vision: &mut crate::game::headless_vision::DeferredCaptureVision,
        sim: &mut GameSim,
        tick: u32,
        vision_interval: u32,
        assets: &crate::game::view::GameViewAssets,
        rt: &macroquad::prelude::RenderTarget,
    ) -> Result<Option<(u32, [f32; OBS_DIM])>> {
        if vision.worker_dead() {
            anyhow::bail!("YOLO 视觉线程已退出");
        }
        if let Some((vtick, obs)) = vision.poll_observation(sim) {
            let ocr = Self::ocr_feet_from_hit(vision.last_self_player());
            if let Some(step) = vision.last_vision_step() {
                sim.record_vision_loot(&step.detections);
            }
            self.apply_observation(sim, obs, ocr);
            return Ok(Some((vtick, self.last_obs)));
        }
        if tick % vision_interval == 0 {
            vision.schedule_draw_if_ready(tick, sim, assets, rt);
        }
        Ok(None)
    }

    pub fn tick_sim(&mut self, sim: &mut GameSim) {
        let input = self.runner.next_frame();
        self.input = input;
        if sim.config.training {
            sim.fitness.try_score_input(&input, sim.state.tick);
        }
        sim.tick_with_action(&input);
    }

    pub async fn bootstrap_vision(
        &mut self,
        vision: &mut HeadlessVisionEnv,
        sim: &mut GameSim,
    ) -> Result<()> {
        let (obs, step) = vision.observe_sim_blocking_with_step(sim).await?;
        sim.record_vision_loot(&step.detections);
        self.apply_observation(sim, obs, Self::ocr_feet_from_hit(step.self_player.as_ref()));
        Ok(())
    }

    pub async fn bootstrap_vision_preview(
        &mut self,
        vision: &mut crate::game::headless_vision::DeferredCaptureVision,
        sim: &mut GameSim,
        assets: &crate::game::view::GameViewAssets,
        rt: &macroquad::prelude::RenderTarget,
    ) -> Result<()> {
        let obs = vision.observe_sim_blocking(sim, assets, rt).await?;
        let ocr = Self::ocr_feet_from_hit(vision.last_self_player());
        if let Some(step) = vision.last_vision_step() {
            sim.record_vision_loot(&step.detections);
        }
        self.apply_observation(sim, obs, ocr);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stuck_latches_when_net_progress_too_small() {
        let mut anchor = None;
        let mut frames = 0u8;
        let mut latched = false;
        // 5 帧几乎不动
        for x in [100.0, 102.0, 99.0, 101.0, 100.5] {
            let b = NeatDriver::update_stuck_axis(
                true,
                x,
                |a, cur| a - cur,
                &mut anchor,
                &mut frames,
                &mut latched,
            );
            let _ = b;
        }
        assert!(latched, "net progress ~0 should latch blocked");
        // 继续噪声仍保持
        let b = NeatDriver::update_stuck_axis(
            true,
            103.0,
            |a, cur| a - cur,
            &mut anchor,
            &mut frames,
            &mut latched,
        );
        assert!(b);
    }

    #[test]
    fn stuck_clears_when_net_progress_enough() {
        let mut anchor = None;
        let mut frames = 0u8;
        let mut latched = false;
        // 先卡住
        for x in [200.0_f32, 200.0, 201.0, 199.0, 200.0] {
            NeatDriver::update_stuck_axis(
                true,
                x,
                |a, cur| a - cur,
                &mut anchor,
                &mut frames,
                &mut latched,
            );
        }
        assert!(latched);
        // 向左走出一大段
        for x in [200.0_f32, 180.0, 160.0, 140.0, 120.0] {
            NeatDriver::update_stuck_axis(
                true,
                x,
                |a, cur| a - cur,
                &mut anchor,
                &mut frames,
                &mut latched,
            );
        }
        assert!(!latched, "large leftward net progress should clear");
    }
}
