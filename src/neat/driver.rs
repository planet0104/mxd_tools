//! NEAT 个体驱动：YOLO 观测 → 前向网络 → 输入帧。

use anyhow::Result;

use crate::game::headless_vision::HeadlessVisionEnv;
use crate::game::input::InputFrame;
use crate::game::observation::OBS_DIM;
use crate::game::sim::GameSim;
use crate::neat::genome::Genome;
use super::network::{evaluate, input_from_outputs};

pub struct NeatDriver {
    genome: Genome,
    input: InputFrame,
    last_obs: [f32; OBS_DIM],
}

impl NeatDriver {
    pub fn new(genome: Genome) -> Self {
        Self {
            genome,
            input: InputFrame::default(),
            last_obs: [0.0_f32; OBS_DIM],
        }
    }

    pub fn set_genome(&mut self, genome: Genome) {
        self.genome = genome;
        self.input = InputFrame::default();
        self.last_obs = [0.0_f32; OBS_DIM];
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

    pub fn apply_observation(&mut self, sim: &mut GameSim, obs: [f32; OBS_DIM]) {
        self.last_obs = obs;
        sim.movement_gate.set_last_observation(&obs);
        if sim.config.training {
            sim.fitness.set_last_observation(&obs);
        }
        let outputs = evaluate(&self.genome, &obs);
        self.input = input_from_outputs(&outputs);
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
            self.apply_observation(sim, obs);
            return Ok(Some((vtick, obs)));
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
            if let Some(step) = vision.last_vision_step() {
                sim.record_vision_loot(&step.detections);
            }
            self.apply_observation(sim, obs);
            return Ok(Some((vtick, obs)));
        }
        if tick % vision_interval == 0 {
            vision.schedule_draw_if_ready(tick, sim, assets, rt);
        }
        Ok(None)
    }

    pub fn tick_sim(&mut self, sim: &mut GameSim) {
        let input = self.input;
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
        self.apply_observation(sim, obs);
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
        if let Some(step) = vision.last_vision_step() {
            sim.record_vision_loot(&step.detections);
        }
        self.apply_observation(sim, obs);
        Ok(())
    }
}
