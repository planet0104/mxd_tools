//! 地图拓扑导航 bot：MapGraph 规划 + YOLO/OCR 执行 + 砍怪/拾取中断。

mod bot_driver;
mod combat_fsm;
mod executor;
mod interrupt;
mod localizer;
mod map_graph;
mod navigator;
mod pickup;
mod progress;
mod stuck_watchdog;
mod types;

pub use bot_driver::NavBot;
pub use combat_fsm::CombatFsm;
pub use executor::NavCtx;
pub use map_graph::MapGraph;
pub use stuck_watchdog::{GlobalStuckWatchdog, GLOBAL_STUCK_SECS, ROPE_BLOCK_TICKS};
pub use types::{ExecutorResult, NavBotConfig, NavDiagSnapshot, PlatformNodeId, SubGoal};

