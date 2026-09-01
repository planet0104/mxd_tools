//! 单基因组一局评估：离屏渲染 + 真实 YOLO+OCR + NEAT 决策。

use anyhow::Result;

use crate::game::fitness::FitnessShapingConfig;
use crate::game::headless_vision::HeadlessVisionEnv;
use crate::game::sim::GameSim;
use crate::game::{load_default_map, GameMap};

use super::driver::NeatDriver;
use super::genome::Genome;

#[derive(Debug, Clone, Copy)]
pub struct EvalOutcome {
    /// 局末得分（含终局惩罚）。
    pub final_fitness: f32,
    /// 对局过程中历史最高得分（选亲排名用，不受后期惩罚拉低）。
    pub peak_fitness: f32,
}

pub async fn evaluate_genome(
    vision: &mut HeadlessVisionEnv,
    map: &GameMap,
    genome: &Genome,
    episode_seed: u64,
    max_ticks: u32,
    vision_interval: u32,
    shaping: FitnessShapingConfig,
) -> Result<EvalOutcome> {
    let mut sim = GameSim::new_training(map.clone(), episode_seed);
    sim.fitness.configure_shaping(shaping);

    let mut driver = NeatDriver::new(genome.clone());
    driver.bootstrap_vision(vision, &mut sim).await?;

    let mut peak_fitness = 0.0_f32;
    for tick in 0..max_ticks {
        if sim.is_episode_over() {
            break;
        }
        let _ = driver
            .logic_tick(vision, &mut sim, tick, vision_interval)
            .await?;
        driver.tick_sim(&mut sim);
        if sim.fitness.allows_peak_update() {
            peak_fitness = peak_fitness.max(sim.fitness.score);
        }
        if sim.is_episode_over() {
            break;
        }
    }

    sim.fitness.finalize_episode();
    Ok(EvalOutcome {
        final_fitness: sim.fitness.score,
        peak_fitness,
    })
}

pub async fn prepare_vision_env(model: Option<&std::path::Path>) -> Result<HeadlessVisionEnv> {
    HeadlessVisionEnv::load(model).await
}

pub async fn load_training_map() -> Result<GameMap> {
    load_default_map()
}
