//! NEAT 进化算法（独立于具体游戏环境）。

mod driver;
mod eval;
mod genome;
mod network;
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
pub use network::{evaluate, input_from_outputs, OUTPUT_THRESHOLD};
pub use population::{Population, PopulationConfig, TrainingCheckpoint};
pub use snapshot::{
    save_best_if_improved, save_session_best, BestGenomeSnapshot, DEFAULT_BEST_GENOME_FILE,
    DEFAULT_SESSION_BEST_FILE,
};

use crate::game::OBS_DIM;

pub const INPUT_SIZE: usize = OBS_DIM;
pub const OUTPUT_SIZE: usize = 8;
pub const OUTPUT_NODE_START: usize = INPUT_SIZE;
pub const HIDDEN_NODE_START: usize = INPUT_SIZE + OUTPUT_SIZE;
