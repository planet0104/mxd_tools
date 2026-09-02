//! NEAT 个体驱动：纯 YOLO+OCR 观测 → 地标里程计本体反馈 → 前向网络 → 动作宏 → 输入帧。
//!
//! 网络输入不含任何 sim 物理通道；自身位移由相邻两帧静态地标反推；动作宏的完成/失败判定同样只用
//! YOLO 槽位 + 该位移 + tick 超时，因此训练与真机部署行为一致（无 MovementGate）。
//!
//! 网络只负责寻路。砍怪由 `CombatFsm` 在本台出现怪时主动接管输入（可选开启，训练时关闭），
//! 交接只发生在宏边界：走路可随时被打断，跳台/爬绳必须跑完。

use anyhow::Result;

use crate::game::combat_fsm::CombatFsm;
use crate::game::explore_memory::ExploreMemory;
use crate::game::headless_vision::HeadlessVisionEnv;
use crate::game::input::InputFrame;
use crate::game::macro_action::{MacroAction, MacroRunner};
use crate::game::observation::{inject_proprioception, OBS_DIM};
use crate::game::odometry::estimate_world_delta_px;
use crate::game::sim::GameSim;
use crate::game::types::{WINDOW_H, WINDOW_W};
use crate::neat::genome::Genome;
use crate::player_name::NamedPlayerHit;
use super::network::{action_from_outputs, evaluate};
use super::obs_compact::compact_obs;

pub struct NeatDriver {
    genome: Genome,
    runner: MacroRunner,
    combat: Option<CombatFsm>,
    input: InputFrame,
    last_obs: [f32; OBS_DIM],
    /// 上一帧 OCR 是否命中：命中才能与本帧做地标匹配。
    has_prev_ocr: bool,
    explore: ExploreMemory,
}

impl NeatDriver {
    pub fn new(genome: Genome) -> Self {
        Self {
            genome,
            runner: MacroRunner::default(),
            combat: None,
            input: InputFrame::default(),
            last_obs: [0.0_f32; OBS_DIM],
            has_prev_ocr: false,
            explore: ExploreMemory::default(),
        }
    }

    /// 开启砍怪状态机接管（预览/部署用；训练寻路时保持关闭）。
    pub fn with_combat(mut self, on: bool) -> Self {
        self.combat = on.then(CombatFsm::default);
        self
    }

    pub fn set_genome(&mut self, genome: Genome) {
        self.genome = genome;
        self.runner.reset();
        if let Some(fsm) = self.combat.as_mut() {
            fsm.reset();
        }
        self.input = InputFrame::default();
        self.last_obs = [0.0_f32; OBS_DIM];
        self.has_prev_ocr = false;
        self.explore.reset();
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

    pub fn combat_active(&self) -> bool {
        self.combat.as_ref().map_or(false, |f| f.is_active())
    }

    pub fn current_action(&self) -> Option<MacroAction> {
        self.runner.last_action()
    }

    /// `ocr_feet`：本帧 OCR 名牌脚点；`None` 表示丢检——此时 obs 里的相对坐标以屏幕中心为原点，
    /// 与上一帧不可比，本帧位移记 0。位移本身由相邻两帧静态地标反推（见 `odometry`）。
    pub fn apply_observation(
        &mut self,
        sim: &mut GameSim,
        mut obs: [f32; OBS_DIM],
        ocr_feet: Option<(f32, f32)>,
    ) {
        let ocr_ok = ocr_feet.is_some();
        let delta_px = if ocr_ok && self.has_prev_ocr {
            estimate_world_delta_px(&self.last_obs, &obs)
        } else {
            None
        };
        let (last_dx, last_dy) = match delta_px {
            Some((dx, dy)) => (
                (dx / WINDOW_W).clamp(-1.0, 1.0),
                (dy / WINDOW_H).clamp(-1.0, 1.0),
            ),
            None => (0.0, 0.0),
        };
        self.has_prev_ocr = ocr_ok;
        let world_truth = sim.config.training.then(|| {
            (sim.state.player.x, sim.state.player.y)
        });
        let explore_hints = self.explore.tick(&obs, delta_px, ocr_ok, world_truth);
        inject_proprioception(
            &mut obs,
            last_dx,
            last_dy,
            self.runner.blocked_left(),
            self.runner.blocked_right(),
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

        if let Some(fsm) = self.combat.as_mut() {
            fsm.observe(&obs);
            if fsm.is_active() {
                if self.runner.interruptible() {
                    self.runner.cancel();
                }
                return;
            }
        }

        if self.runner.is_idle() {
            let compact = compact_obs(&obs, self.runner.last_failed(), &explore_hints);
            let outputs = evaluate(&self.genome, &compact);
            let seek = self.explore.seek_vertical();
            let allowed = self.runner.allowed(&obs, seek);
            let action = action_from_outputs(&outputs, &allowed);
            if sim.config.training {
                sim.fitness.score_nav_decision(action, &explore_hints);
            }
            self.runner.begin(action, &obs);
        }
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
        let input = match self.combat.as_mut() {
            Some(fsm) if fsm.is_active() && self.runner.is_idle() => fsm.next_frame(),
            _ => self.runner.next_frame(),
        };
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
