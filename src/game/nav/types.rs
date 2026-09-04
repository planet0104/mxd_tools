use std::collections::{HashMap, HashSet};

use super::super::input::InputFrame;

pub type PlatformNodeId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Walk,
    Fall,
    ClimbUp,
    ClimbDown,
    StepUp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SubGoal {
    GoTo { x: f32 },
    WalkOff { side: Side },
    ClimbUp { rope_x: f32 },
    ClimbDown { rope_x: f32 },
    StepUp { target_x: f32 },
    Patrol { dir: f32 },
    Idle,
}

impl Default for SubGoal {
    fn default() -> Self {
        SubGoal::Idle
    }
}

impl SubGoal {
    /// 换层/越障过程中禁止战斗抢走左右与跳跃。
    pub fn is_transit(self) -> bool {
        matches!(
            self,
            Self::StepUp { .. }
                | Self::ClimbUp { .. }
                | Self::ClimbDown { .. }
                | Self::WalkOff { .. }
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LocState {
    pub world_x: f32,
    pub world_y: f32,
    pub confidence: u8,
    pub node_id: PlatformNodeId,
    pub on_ground: bool,
    pub climbing: bool,
}

#[derive(Debug, Clone)]
pub struct PlatformNode {
    pub id: PlatformNodeId,
    pub x_min: f32,
    pub x_max: f32,
    pub y: f32,
    pub layer: i32,
    pub group: i32,
    pub prev: u32,
    pub next: u32,
}

impl PlatformNode {
    pub fn width(&self) -> f32 {
        self.x_max - self.x_min
    }

    pub fn is_patrol_worthy(&self) -> bool {
        self.width() >= PATROL_MIN_PLATFORM_W
    }
}

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub kind: EdgeKind,
    pub from: PlatformNodeId,
    pub to: PlatformNodeId,
    pub rope_x: Option<f32>,
    pub target_x: f32,
    pub cost: u32,
}

#[derive(Debug, Clone, Default)]
pub struct ExploreState {
    pub visited: HashSet<PlatformNodeId>,
    pub patrol_dir: f32,
    pub blocked_edges: HashMap<(PlatformNodeId, EdgeKind, PlatformNodeId), u32>,
    pub active_subgoal: SubGoal,
    pub subgoal_ticks: u32,
    pub subgoal_failures: u32,
    pub pending_edge: Option<(PlatformNodeId, EdgeKind, PlatformNodeId)>,
    pub escape_ticks: u32,
    pub escape_dir: f32,
    /// 到未访问节点的整条边路径（按序执行，避免每步重算导致 ping-pong）
    pub explore_path: Vec<usize>,
    /// 覆盖全图平台的随机固定巡逻边序列
    pub patrol_route: Vec<usize>,
    pub patrol_cursor: usize,
    /// 上一跳 Walk 完成端点，用于禁止立刻原路返回
    pub last_walk_hop: Option<(PlatformNodeId, PlatformNodeId)>,
    /// 刚完成的 StepUp/ClimbUp（from→to），禁止立刻掉/跳回起点。
    pub last_ascent_hop: Option<(PlatformNodeId, PlatformNodeId)>,
    /// 上台粘滞剩余 tick：期间禁止从落点 Fall。
    pub ascent_hold_ticks: u32,
    /// 刚 StepUp/ClimbUp 落到的台：宽台先沿 patrol_dir 扫到边缘。
    pub sweep_after_ascent: Option<PlatformNodeId>,
    /// 上台后优先向前开荒（与 patrol_dir 同向），避免窄台落点立刻找左侧最近未访问。
    pub prefer_forward_explore: bool,
    /// 爬绳失败后的侧向重试：按半精灵宽多档站位轮换试跳。
    pub climb_retry: Option<ClimbRetry>,
}

#[derive(Debug, Clone, Copy)]
pub struct ClimbRetry {
    pub from: PlatformNodeId,
    pub to: PlatformNodeId,
    pub kind: EdgeKind,
    pub rope_x: f32,
    /// `CLIMB_PROBE_OFFSETS` 下标：绳心→左右半身→左右一身…
    pub offset_idx: u8,
    pub attempts: u32,
}

#[derive(Debug, Clone)]
pub struct NavBotConfig {
    pub farm_mode: bool,
    pub chase_max_dx: f32,
    pub pickup_chase_dx: f32,
    pub pickup_chase_max_ticks: u32,
    pub pickup_near_only_on_farm: bool,
    pub subgoal_timeout_ticks: u32,
    pub edge_block_ticks: u32,
    pub goto_tolerance_px: f32,
    pub climb_align_px: f32,
    pub vision_min_conf: u8,
    /// 起跳冷却，按 60Hz 逻辑帧换算成墙钟（18 ≈ 300ms）；与感知频率无关。
    pub step_up_jump_cooldown: u32,
    pub step_up_stall: u32,
    pub step_up_timeout_ticks: u32,
    pub patrol_seed: u64,
    /// 爬绳失败封边时长（短于普通边，便于侧移后重试）
    pub climb_block_ticks: u32,
    pub climb_retry_max: u32,
}

impl Default for NavBotConfig {
    fn default() -> Self {
        Self {
            farm_mode: true,
            chase_max_dx: 200.0,
            pickup_chase_dx: 200.0,
            pickup_chase_max_ticks: 72,
            pickup_near_only_on_farm: true,
            subgoal_timeout_ticks: 150,
            edge_block_ticks: 600,
            goto_tolerance_px: 16.0,
            climb_align_px: 18.0,
            vision_min_conf: 3,
            step_up_jump_cooldown: 18,
            step_up_stall: 5,
            step_up_timeout_ticks: 72,
            patrol_seed: 42,
            climb_block_ticks: 240,
            climb_retry_max: 7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutorResult {
    #[default]
    Running,
    Done,
    Failed,
}

impl ExecutorResult {
    pub fn label(self) -> &'static str {
        match self {
            Self::Running => "run",
            Self::Done => "done",
            Self::Failed => "fail",
        }
    }
}

impl EdgeKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Walk => "walk",
            Self::Fall => "fall",
            Self::ClimbUp => "climb_up",
            Self::ClimbDown => "climb_down",
            Self::StepUp => "step_up",
        }
    }
}

impl SubGoal {
    pub fn label(self) -> String {
        match self {
            Self::GoTo { x } => format!("goto({x:.0})"),
            Self::WalkOff { side } => format!(
                "walk_off({})",
                match side {
                    Side::Left => "L",
                    Side::Right => "R",
                }
            ),
            Self::ClimbUp { rope_x } => format!("climb_up({rope_x:.0})"),
            Self::ClimbDown { rope_x } => format!("climb_down({rope_x:.0})"),
            Self::StepUp { target_x } => format!("step_up({target_x:.0})"),
            Self::Patrol { dir } => format!("patrol({:.0})", dir.signum()),
            Self::Idle => "idle".to_string(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct NavDiagSnapshot {
    pub goal: SubGoal,
    pub exec: ExecutorResult,
    pub nav_node: PlatformNodeId,
    pub est_node: PlatformNodeId,
    pub nav_x: f32,
    pub nav_y: f32,
    pub est_x: f32,
    pub est_y: f32,
    pub pending_from: Option<PlatformNodeId>,
    pub pending_kind: Option<EdgeKind>,
    pub pending_to: Option<PlatformNodeId>,
    pub subgoal_ticks: u32,
    pub subgoal_failures: u32,
    pub escape_ticks: u32,
    pub escape_dir: f32,
    pub step_stall: u32,
    pub step_jump_dir: f32,
    pub step_jumped: bool,
    pub step_jump_cd: u32,
    pub walk_right: Option<bool>,
    pub walk_left: Option<bool>,
    pub drop_right: Option<bool>,
    pub drop_left: Option<bool>,
    pub obs_step_up: Option<f32>,
    pub grounded_est: bool,
    pub visual_conf: u8,
    pub blocked_edges: usize,
    pub combat_active: bool,
    pub farm_local: bool,
}

#[derive(Debug, Clone, Default)]
pub struct PickupState {
    pub drop_memory: u32,
    pub chase_ticks: u32,
}

#[derive(Debug, Clone, Default)]
pub struct ProgressState {
    pub last_x: f32,
    pub last_y: f32,
    pub last_node: PlatformNodeId,
    pub stagnant_ticks: u32,
    pub last_visited_count: usize,
    pub global_stall_ticks: u32,
}

pub const DROP_MEMORY_TICKS: u32 = 72;
pub const SAME_PLATFORM_DY_PX: f32 = 55.0;
/// 宽度低于此值的平台仅作落脚，不参与固定巡逻
pub const PATROL_MIN_PLATFORM_W: f32 = crate::game::types::PLAYER_SPAWN_MIN_PLATFORM_W;
pub const FARM_LOCAL_DX: f32 = 260.0;

pub fn merge_frames(base: InputFrame, overlay: InputFrame) -> InputFrame {
    let mut out = base;
    if overlay.left {
        out.left = true;
        out.right = false;
    }
    if overlay.right {
        out.right = true;
        out.left = false;
    }
    if overlay.jump {
        out.jump = true;
    }
    if overlay.attack {
        out.attack = true;
    }
    if overlay.up {
        out.up = true;
    }
    if overlay.down {
        out.down = true;
    }
    if overlay.pick_up {
        out.pick_up = true;
    }
    out
}
