//! NEAT 训练循环（单 GL 线程评估 + 可选子进程 worker）。

mod agent;
mod eval;
pub mod log;
mod profile;
mod progress;
mod render;
mod worker_pool;
pub mod worker_protocol;

pub use agent::{AgentController, VisionWorkerTiming};
pub use eval::{
    capture_training_frame, evaluate_genome, evaluate_genome_profile, TrainerEvalContext,
    TrainingCapture,
};
pub use profile::{EvalProfileReport, RenderStepTiming, TickProfile};
pub use progress::{
    log_pool_heartbeat, log_steady_heartbeat, EvalProgressConfig, EvalStatus, HEARTBEAT_INTERVAL,
};
pub use log::{log_line, ts};
pub use render::{
    capture_render_rgb, capture_render_rgb_fast, capture_render_rgb_headless,
    capture_render_rgb_headless_timed, capture_render_rgb_timed, present_training_frame,
};
pub use worker_pool::{WorkerPool, WorkerPoolConfig};
