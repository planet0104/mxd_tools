//! NEAT 进化算法（独立于具体游戏环境）。

mod driver;
mod eval;
mod genome;
mod network;
pub mod obs_compact;
mod parallel;
mod population;
mod snapshot;

pub use driver::NeatDriver;
pub use eval::{evaluate_genome, prepare_vision_env, EvalOutcome};
pub use parallel::{
    create_parallel_slots, prepare_parallel_assets, run_parallel_trainer, trainer_progress,
    ParallelCheckpointConfig, ParallelTrainerConfig, TrainerShared,
};
pub use genome::{
    crossover, export_innovation_state, mutate, rank_fitness, restore_innovation_state,
    restore_innovations_from_population, Compatibility, ConnectionGene, Genome, InnovationState,
};
pub use network::{action_from_outputs, evaluate};
pub use obs_compact::{compact_obs, NEAT_OBS_DIM};
pub use population::{Population, PopulationConfig, TrainingCheckpoint};
pub use snapshot::{
    save_best_if_improved, save_session_best, BestGenomeSnapshot, DEFAULT_BEST_GENOME_FILE,
    DEFAULT_SESSION_BEST_FILE,
};

use crate::game::macro_action::MACRO_ACTION_COUNT;

/// 网络输入：压缩后的事实摘要，而非 102 维原始 YOLO 槽位。
pub const INPUT_SIZE: usize = NEAT_OBS_DIM;
/// 输出：互斥动作宏（走左/走右/砍/左跳台/右跳台/攀爬），取最大一路执行。
pub const OUTPUT_SIZE: usize = MACRO_ACTION_COUNT;
pub const OUTPUT_NODE_START: usize = INPUT_SIZE;
pub const HIDDEN_NODE_START: usize = INPUT_SIZE + OUTPUT_SIZE;
