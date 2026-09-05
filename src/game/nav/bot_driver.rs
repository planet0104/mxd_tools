use super::super::input::InputFrame;
use super::super::map::GameMap;
use super::super::observation::{
    obs_drop_in_pickup_range, obs_floor_underfoot, obs_step_up_dx, OBS_DIM,
};
use super::super::vision_sense::VisionSenseState;
use super::super::{WINDOW_H, WINDOW_W};
use super::executor::{MotionExecutor, NavCtx};
use super::interrupt::{CombatAdapter, InterruptArbiter};
use super::localizer::Localizer;
use super::map_graph::MapGraph;
use super::navigator::Navigator;
use super::pickup::PickupController;
use super::progress::ProgressMonitor;
use super::survival::SurvivalFsm;
use super::survival::SurvivalMode;
use super::types::{
    EdgeKind, ExecutorResult, NavBotConfig, NavDiagSnapshot, PlatformNodeId, Side, SubGoal,
};

fn holds_subgoal(goal: SubGoal) -> bool {
    !matches!(goal, SubGoal::Patrol { .. } | SubGoal::Idle)
}

fn is_elevated_node(graph: &MapGraph, node: u32) -> bool {
    graph.get(node).is_some_and(|n| n.y < 1150.0)
}

pub struct NavBot {
    pub config: NavBotConfig,
    pub graph: MapGraph,
    localizer: Localizer,
    navigator: Navigator,
    executor: MotionExecutor,
    pickup: PickupController,
    combat: CombatAdapter,
    progress: ProgressMonitor,
    survival: SurvivalFsm,
    /// 0~1：YOLO 血条或 sim 真值，供保命决策。
    pub hp_ratio: f32,
    pub last_reason: &'static str,
    pub last_diag: NavDiagSnapshot,
    nav_intent: InputFrame,
    prev_node: u32,
    spawn_y: f32,
    /// 连续若干决策帧农怪带为空 → 才标记 farm_cleared（防 YOLO 闪断）
    farm_clear_streak: u32,
    farm_mob_streak: u32,
}

impl NavBot {
    pub fn new(map: &GameMap, config: NavBotConfig) -> Self {
        let graph = MapGraph::build(map);
        let patrol_seed = config.patrol_seed;
        let (sx, sy) = map.default_spawn();
        let spawn_node = graph
            .node_at(map, sx, sy)
            .or_else(|| {
                map.platforms
                    .first()
                    .map(|p| p.id)
                    .filter(|id| graph.nodes.contains_key(id))
            })
            .unwrap_or(0);
        let mut localizer = Localizer::default();
        localizer.reset(sx, sy, spawn_node);
        let mut progress = ProgressMonitor::default();
        progress.reset(sx, sy, spawn_node);
        let navigator = Navigator::new(spawn_node, &graph, patrol_seed);
        Self {
            config,
            graph,
            navigator,
            executor: MotionExecutor::default(),
            pickup: PickupController::default(),
            combat: CombatAdapter::default(),
            progress,
            survival: SurvivalFsm::default(),
            hp_ratio: 1.0,
            last_reason: "init",
            last_diag: NavDiagSnapshot::default(),
            nav_intent: InputFrame::default(),
            prev_node: spawn_node,
            spawn_y: sy,
            farm_clear_streak: 0,
            farm_mob_streak: 0,
            localizer,
        }
    }

    pub fn set_hp_ratio(&mut self, ratio: f32) {
        self.hp_ratio = ratio.clamp(0.0, 1.0);
    }

    pub fn survival_mode(&self) -> SurvivalMode {
        self.survival.mode()
    }

    pub fn reset(&mut self, map: &GameMap, spawn_x: f32, spawn_y: f32) {
        let spawn_node = self.graph.node_at(map, spawn_x, spawn_y).unwrap_or(0);
        self.localizer.reset(spawn_x, spawn_y, spawn_node);
        self.navigator.reset(spawn_node, &self.graph);
        self.executor.reset();
        self.pickup.reset();
        self.combat.reset();
        self.survival.reset();
        self.hp_ratio = 1.0;
        self.progress.reset(spawn_x, spawn_y, spawn_node);
        self.nav_intent = InputFrame::default();
        self.prev_node = spawn_node;
        self.spawn_y = spawn_y;
        self.farm_clear_streak = 0;
        self.farm_mob_streak = 0;
        self.last_reason = "reset";
        self.last_diag = NavDiagSnapshot::default();
    }

    /// 卡住软重置：保留已访问节点与坐标，清掉卡住的攀爬/逃逸/pending。
    /// 半空时保留 ascent / ClimbDown 封锁，避免 resume 后立刻爬下去（日志 122→climb_down）。
    pub fn soft_reset_keep_progress(&mut self, map: &GameMap, x: f32, y: f32) {
        let elevated = y < 1100.0;
        let kept_ascent = self.navigator.explore.last_ascent_hop;
        let kept_hold = self.navigator.explore.ascent_hold_ticks;
        let kept_sweep = self.navigator.explore.sweep_after_ascent;
        let kept_forward = self.navigator.explore.prefer_forward_explore;
        let kept_down_blocks: Vec<_> = if elevated {
            self.navigator
                .explore
                .blocked_edges
                .iter()
                .filter(|((_, kind, _), _)| *kind == EdgeKind::ClimbDown)
                .map(|(k, v)| (*k, *v))
                .collect()
        } else {
            Vec::new()
        };

        let node = self
            .graph
            .node_at(map, x, y)
            .unwrap_or(self.localizer.state.node_id);
        self.localizer.reset(x, y, node);
        self.executor.reset();
        self.pickup.reset();
        self.combat.reset();
        self.survival.reset();
        self.progress.reset(x, y, node);
        self.nav_intent = InputFrame::default();
        self.prev_node = node;
        self.navigator.explore.active_subgoal = SubGoal::Idle;
        self.navigator.explore.pending_edge = None;
        self.navigator.explore.escape_ticks = 0;
        self.navigator.explore.climb_retry = None;
        self.navigator.explore.explore_path.clear();
        self.navigator.explore.subgoal_ticks = 0;
        self.navigator.explore.subgoal_failures = 0;
        self.navigator.explore.last_walk_hop = None;
        if elevated {
            self.navigator.explore.last_ascent_hop = kept_ascent;
            self.navigator.explore.ascent_hold_ticks = kept_hold.max(180);
            self.navigator.explore.sweep_after_ascent = kept_sweep;
            self.navigator.explore.prefer_forward_explore = kept_forward || kept_ascent.is_some();
            // 只清 ClimbUp，保留/恢复 ClimbDown 封锁。
            self.navigator.clear_climb_blocks();
            for (k, v) in kept_down_blocks {
                self.navigator.explore.blocked_edges.insert(k, v);
            }
        } else {
            self.navigator.explore.last_ascent_hop = None;
            self.navigator.explore.ascent_hold_ticks = 0;
            self.navigator.explore.sweep_after_ascent = None;
            self.navigator.explore.prefer_forward_explore = false;
            self.navigator
                .explore
                .blocked_edges
                .retain(|(_, kind, _), _| {
                    !matches!(kind, EdgeKind::ClimbUp | EdgeKind::ClimbDown)
                });
        }
        self.last_reason = "soft_reset";
    }

    pub fn localizer_node(&self) -> u32 {
        self.localizer.state.node_id
    }

    pub fn patrol_dir(&self) -> f32 {
        self.navigator.explore.patrol_dir
    }

    /// 爬到图上 ClimbUp 中段终点：记 ascent，下一步优先向上 StepUp。
    pub fn note_mid_climb_landing(&mut self, node: u32) {
        self.navigator
            .note_arrived_climb_top(&self.graph, node, None);
        self.navigator.explore.active_subgoal = SubGoal::Idle;
        self.last_reason = "mid_climb_landing";
    }

    /// 在当前位置重新挂上最近的爬绳边（默认向上）。成功返回 true。
    pub fn force_resume_climb(&mut self, map: &GameMap, x: f32, y: f32) -> bool {
        self.soft_reset_keep_progress(map, x, y);
        let node = self.localizer.state.node_id;
        // 已升高到某条 ClimbUp 终点高度：一律改中段向上，禁止再挂绳（尤其禁止 ClimbDown）。
        if y < 1100.0 {
            if let Some(landing) = self.nearest_climb_up_landing(x, y) {
                self.localizer.state.node_id = landing;
                self.prev_node = landing;
                self.note_mid_climb_landing(landing);
                // 先前 StepUp 失败会封边；中段卡死时清掉落点向上封锁再试。
                self.navigator.explore.blocked_edges.retain(|&(from, kind, _), _| {
                    !(from == landing && kind == EdgeKind::StepUp)
                });
                if let Some(goal) =
                    self.navigator.plan_continue_ascent(&self.graph, landing)
                {
                    self.navigator.set_subgoal(goal);
                    self.executor.reset();
                    self.last_reason = "global_stuck_mid_ascent";
                    return true;
                }
                let dir = self.navigator.explore.patrol_dir.signum();
                self.navigator.set_subgoal(SubGoal::Patrol {
                    dir: if dir == 0.0 { 1.0 } else { dir },
                });
                self.executor.reset();
                self.last_reason = "global_stuck_mid_ascent";
                return true;
            }
        }
        let Some(edge) = self.pick_resume_climb_edge(node, x, y).cloned() else {
            return false;
        };
        if edge.kind == EdgeKind::ClimbDown {
            return false;
        }
        let goal = self.navigator.commit_edge(&self.graph, &edge);
        self.navigator.set_subgoal(goal);
        self.executor.reset();
        self.last_reason = "global_stuck_resume_climb";
        true
    }

    /// 当前高度是否已到某条 ClimbUp 落点（允许 x 偏离，视觉定位常漂到旁台）。
    fn nearest_climb_up_landing(&self, x: f32, y: f32) -> Option<u32> {
        let mut best: Option<(f32, u32)> = None;
        for e in &self.graph.edges {
            if e.kind != EdgeKind::ClimbUp {
                continue;
            }
            let Some(dest) = self.graph.get(e.to) else {
                continue;
            };
            if dest.y >= 1100.0 {
                continue;
            }
            if (y - dest.y).abs() > 56.0 {
                continue;
            }
            let mid = (dest.x_min + dest.x_max) * 0.5;
            let d = (y - dest.y).abs() * 2.0 + (x - mid).abs() * 0.15;
            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, e.to));
            }
        }
        best.map(|(_, id)| id)
    }

    /// 弃绳：封掉该绳附近爬绳边，软重置后强制侧移离开。
    /// 半空（已爬升）不封上爬边，只短逃逸后再爬，避免十几分钟锁死在右侧。
    pub fn abandon_rope(&mut self, map: &GameMap, x: f32, y: f32, rope_x: f32) {
        self.soft_reset_keep_progress(map, x, y);
        let elevated = y < 1100.0;
        if !elevated {
            for e in &self.graph.edges {
                if !matches!(e.kind, EdgeKind::ClimbUp | EdgeKind::ClimbDown) {
                    continue;
                }
                let rx = e.rope_x.unwrap_or(e.target_x);
                if (rx - rope_x).abs() <= 40.0 {
                    self.navigator.explore.blocked_edges.insert(
                        (e.from, e.kind, e.to),
                        super::stuck_watchdog::ROPE_BLOCK_TICKS,
                    );
                }
            }
        } else {
            self.navigator.clear_climb_blocks();
        }
        let dir = if x >= rope_x { -1.0 } else { 1.0 };
        self.navigator
            .begin_escape(dir, if elevated { 18 } else { 48 });
        self.navigator.explore.patrol_dir = dir;
        if elevated {
            self.navigator.explore.prefer_forward_explore = false;
        }
        self.last_reason = "global_stuck_abandon_rope";
    }

    /// 是否站在某条 Climb 边的上端平台附近。
    pub fn at_climb_top_platform(&self, x: f32, y: f32) -> Option<f32> {
        let mut best: Option<(f32, f32)> = None;
        for e in &self.graph.edges {
            if e.kind != EdgeKind::ClimbUp {
                continue;
            }
            let rope = e.rope_x.unwrap_or(e.target_x);
            if (rope - x).abs() > 72.0 {
                continue;
            }
            let Some(dest) = self.graph.get(e.to) else {
                continue;
            };
            if (y - dest.y).abs() > 48.0 {
                continue;
            }
            if x < dest.x_min - 24.0 || x > dest.x_max + 24.0 {
                continue;
            }
            let d = (y - dest.y).abs() + (rope - x).abs() * 0.25;
            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, rope));
            }
        }
        best.map(|(_, rope)| rope)
    }

    fn pick_resume_climb_edge(
        &self,
        node: u32,
        x: f32,
        y: f32,
    ) -> Option<&super::types::GraphEdge> {
        let mut best: Option<(f32, &super::types::GraphEdge)> = None;
        for e in &self.graph.edges {
            // 恢复攀爬只选上爬；下爬会把中段直接送回底层（日志 resume→climb_down）。
            if e.kind != EdgeKind::ClimbUp {
                continue;
            }
            let rope = e.rope_x.unwrap_or(e.target_x);
            let near_rope = (rope - x).abs() <= 72.0;
            let from_here = e.from == node;
            if !near_rope && !from_here {
                continue;
            }
            let from_y = self.graph.get(e.from).map(|n| n.y).unwrap_or(y);
            let to_y = self.graph.get(e.to).map(|n| n.y).unwrap_or(y);
            // 已接近/超过终点高度：这条上爬已完成，不要再挂。
            if y <= to_y + 36.0 {
                continue;
            }
            let dist_from = (y - from_y).abs();
            let dist_to = (y - to_y).abs();
            let dir_penalty = if dist_to + 20.0 < dist_from {
                40.0
            } else {
                0.0
            };
            let score = (rope - x).abs() + if from_here { 0.0 } else { 24.0 } + dir_penalty;
            if best.map(|(s, _)| score < s).unwrap_or(true) {
                best = Some((score, e));
            }
        }
        best.map(|(_, e)| e)
    }

    fn note_farm_clear(&mut self, band_empty: bool) -> bool {
        const CLEAR_FRAMES: u32 = 12;
        const MOB_RESET_FRAMES: u32 = 4;
        if self.navigator.farm_cleared() {
            return true;
        }
        if band_empty {
            self.farm_clear_streak = self.farm_clear_streak.saturating_add(1);
            self.farm_mob_streak = 0;
        } else {
            self.farm_mob_streak = self.farm_mob_streak.saturating_add(1);
            // YOLO 闪一下有怪不立刻清零；连续多帧才打断清空计数。
            if self.farm_mob_streak >= MOB_RESET_FRAMES {
                self.farm_clear_streak = 0;
            }
        }
        if self.farm_clear_streak >= CLEAR_FRAMES {
            self.navigator.mark_farm_cleared();
            return true;
        }
        false
    }

    pub fn diag(&self) -> &NavDiagSnapshot {
        &self.last_diag
    }

    /// 当前节点可出边摘要（诊断：是否有 Climb / StepUp）。
    pub fn exit_summary(&self, node: PlatformNodeId) -> String {
        let mut parts = Vec::new();
        for e in &self.graph.edges {
            if e.from != node {
                continue;
            }
            let blocked = self
                .navigator
                .explore
                .blocked_edges
                .contains_key(&(e.from, e.kind, e.to));
            parts.push(format!(
                "{}->{}@{}{}{}",
                e.kind.label(),
                e.to,
                e.target_x as i32,
                e.rope_x
                    .map(|x| format!(" rope={x:.0}"))
                    .unwrap_or_default(),
                if blocked { "!" } else { "" }
            ));
        }
        if parts.is_empty() {
            "none".into()
        } else {
            parts.join(", ")
        }
    }

    /// 是否贴近图节点左右缘（约 28px，对齐 step_up at_ledge）。
    pub fn at_node_ledge(&self, node: PlatformNodeId, x: f32) -> bool {
        self.graph.get(node).is_some_and(|n| {
            x >= n.x_max - 28.0 || x <= n.x_min + 28.0
        })
    }

    pub fn visited_nodes(&self) -> usize {
        self.navigator.visited_count()
    }

    /// 硬重置用：导出已访问集合、farm 标记与巡逻方向。
    pub fn snapshot_explore_progress(&self) -> (std::collections::HashSet<u32>, bool, f32) {
        (
            self.navigator.explore.visited.clone(),
            self.navigator.farm_cleared(),
            self.navigator.explore.patrol_dir,
        )
    }

    /// 硬重置后写回探索进度。
    pub fn restore_explore_progress(
        &mut self,
        visited: std::collections::HashSet<u32>,
        farm_cleared: bool,
        patrol_dir: f32,
    ) {
        self.navigator.explore.visited = visited;
        if patrol_dir.abs() > 0.1 {
            self.navigator.explore.patrol_dir = patrol_dir.signum();
        }
        if farm_cleared {
            self.navigator.mark_farm_cleared();
        }
    }

    pub fn total_nodes(&self) -> usize {
        self.graph.node_count()
    }

    /// 纯 YOLO+SelfTracker：仅 obs 与视觉里程计，不读游戏内存。
    pub fn decide(
        &mut self,
        map: &GameMap,
        obs: &[f32; OBS_DIM],
        sense: &VisionSenseState,
    ) -> InputFrame {
        let under = obs_floor_underfoot(obs);
        // 脚下有地板时不以攀爬论：否则 on_ground 永假、执行器走 climb_orphan 只按上不跳。
        let climbing = sense.climbing && !under;
        let on_ground = under;

        let visual_conf = sense.visual_confidence();
        let low_confidence = visual_conf < self.config.vision_min_conf;

        self.localizer.tick(
            map,
            &self.graph,
            sense.est_x,
            sense.est_y,
            visual_conf,
            on_ground,
            climbing,
            self.config.vision_min_conf,
        );

        // 台缝上 stand_at 会在 from/to 间抖；Walk 走廊内粘住 from/to。
        if on_ground {
            if let Some((from, kind, to)) = self.navigator.explore.pending_edge {
                if kind == EdgeKind::Walk {
                    let x = self.localizer.state.world_x;
                    let y = self.localizer.state.world_y;
                    let stick = [to, from].into_iter().find(|&id| {
                        self.graph.get(id).is_some_and(|d| {
                            x >= d.x_min - 8.0
                                && x <= d.x_max + 8.0
                                && (y - d.y).abs() < 100.0
                        })
                    });
                    if let Some(id) = stick {
                        self.localizer.state.node_id = id;
                    }
                }
            } else if let Some((_, to)) = self.navigator.explore.last_walk_hop {
                if let Some(dest) = self.graph.get(to) {
                    let x = self.localizer.state.world_x;
                    let y = self.localizer.state.world_y;
                    if x >= dest.x_min - 8.0
                        && x <= dest.x_max + 8.0
                        && (y - dest.y).abs() < 100.0
                    {
                        self.localizer.state.node_id = to;
                    }
                }
            }
        }

        let loc = self.localizer.state;
        let nav_node = loc.node_id;
        let nav_x = loc.world_x;
        let nav_y = loc.world_y;

        self.survival.observe(obs, self.hp_ratio, nav_y);

        if nav_node != self.prev_node {
            self.navigator
                .on_node_changed(&self.graph, self.prev_node, nav_node);
            self.prev_node = nav_node;
        }

        // Walk / StepUp / Climb：pending 的 from 对不上当前节点时尽快放弃或改挂本台同目的边。
        // 爬绳半空/已爬升：禁止因节点抖动 fail（日志里上到一半改 step_up(1574)）。
        // StepUp 起跳后/已升高：禁止 fail+reset。
        if let Some((from, kind, to)) = self.navigator.explore.pending_edge {
            let climb_kind = matches!(kind, EdgeKind::ClimbUp | EdgeKind::ClimbDown);
            let step_committed = kind == EdgeKind::StepUp
                && self.executor.step_up_committed(on_ground, nav_y);
            let climb_committed = climb_kind
                && (climbing || !on_ground || nav_y < 1160.0);
            if holds_subgoal(self.navigator.explore.active_subgoal)
                && nav_node != from
                && nav_node != to
                && on_ground
                && !climbing
                && !step_committed
                && !climb_committed
            {
                let retarget = matches!(kind, EdgeKind::StepUp | EdgeKind::ClimbUp | EdgeKind::ClimbDown)
                    .then(|| {
                        self.graph.edges.iter().find(|e| {
                            e.from == nav_node && e.to == to && e.kind == kind
                        })
                    })
                    .flatten();
                if let Some(edge) = retarget {
                    self.navigator.explore.pending_edge =
                        Some((edge.from, edge.kind, edge.to));
                    self.navigator.explore.active_subgoal =
                        self.graph.edge_to_subgoal(edge);
                    // StepUp 同向改挂：保留起跳方向，禁止 reset 后反向 approach。
                    if kind != EdgeKind::StepUp {
                        self.executor.reset();
                    }
                } else if matches!(
                    kind,
                    EdgeKind::Walk
                        | EdgeKind::StepUp
                        | EdgeKind::ClimbUp
                        | EdgeKind::ClimbDown
                ) {
                    self.navigator.on_subgoal_failed(
                        &self.graph,
                        self.config.edge_block_ticks,
                        !low_confidence,
                    );
                    self.executor.reset();
                }
            } else if step_committed && nav_node != from && nav_node != to && on_ground {
                // 已升高落在旁台：若存在同目的台阶边则静默改挂 from，不 reset。
                if let Some(edge) = self.graph.edges.iter().find(|e| {
                    e.from == nav_node && e.to == to && e.kind == EdgeKind::StepUp
                }) {
                    self.navigator.explore.pending_edge =
                        Some((edge.from, edge.kind, edge.to));
                    self.navigator.explore.active_subgoal =
                        self.graph.edge_to_subgoal(edge);
                }
            }
        }

        self.pickup.tick_memory(obs);
        if self.survival.suppress_combat() {
            // 撤离时绝不砍怪：补刀会站桩挨打，且方向键冲突导致上不了台。
            self.combat.reset();
        } else {
            self.combat.observe(obs);
        }

        let mut ctx = NavCtx::from_vision(
            obs,
            sense.facing,
            on_ground,
            climbing,
            loc.world_x,
            loc.world_y,
            loc.node_id,
        );
        let farm_cleared = self.note_farm_clear(!ctx.farm_band_mobs);
        let planned = self.navigator.plan(
            &self.graph,
            nav_node,
            nav_x,
            farm_cleared,
            self.config.goto_tolerance_px,
        );
        let mut goal = if holds_subgoal(self.navigator.explore.active_subgoal) {
            self.navigator.explore.active_subgoal
        } else {
            planned
        };
        // 保命覆盖在 plan 之后：低血必须换台，禁止被 Idle/假 step_up 吃掉。
        if self.survival.force_seek_safe_platform() {
            let ticks = self.navigator.explore.subgoal_ticks;
            let step_stuck = matches!(goal, SubGoal::StepUp { .. })
                && (matches!(
                    self.last_reason,
                    "step_up_unreachable"
                        | "step_up_stalled"
                        | "step_up_fell"
                        | "step_up_wait"
                ) || (self.last_reason == "step_up_approach" && ticks > 20)
                    || ticks > 40);
            if step_stuck {
                // 卡在「右侧略高台」来回接近：封边并改爬绳/别的路。
                if let Some(key) = self.navigator.explore.pending_edge.take() {
                    self.navigator
                        .explore
                        .blocked_edges
                        .insert(key, 240);
                }
                self.navigator.explore.active_subgoal = SubGoal::Idle;
                self.navigator.explore.subgoal_ticks = 0;
                self.executor.reset();
                goal = SubGoal::Idle;
            }
            let keep_escape = matches!(
                goal,
                SubGoal::ClimbUp { .. }
                    | SubGoal::ClimbDown { .. }
                    | SubGoal::StepUp { .. }
                    | SubGoal::GoTo { .. }
                    | SubGoal::WalkOff { .. }
            ) && self.navigator.explore.pending_edge.is_some()
                && !step_stuck
                && self.last_reason != "step_up_done"
                && self.last_reason != "idle"
                && self.last_reason != "post_clear_patrol"
                && self.last_reason != "step_up_unreachable";
            if !keep_escape {
                let esc = self.plan_heal_escape(nav_node, nav_x);
                goal = esc;
                self.navigator.set_subgoal(esc);
                self.last_reason = "survive_flee_safe";
            }
        } else if self.survival.prefer_idle_heal(obs, nav_y) {
            goal = SubGoal::Idle;
            self.navigator.set_subgoal(SubGoal::Idle);
            self.last_reason = "survive_heal_wait";
        } else if matches!(goal, SubGoal::Idle) {
            // 杀完怪后禁止干站：强制巡逻。
            let dir = if self.navigator.explore.patrol_dir.abs() > 0.1 {
                self.navigator.explore.patrol_dir.signum()
            } else {
                1.0
            };
            goal = SubGoal::Patrol { dir };
            self.navigator.set_subgoal(goal);
            self.last_reason = "post_clear_patrol";
        }
        // 台阶已起跳/已升高：强制保持 StepUp（保命逃离中若未起跳则可改挂）。
        if self.executor.step_up_committed(on_ground, nav_y)
            && !(self.survival.force_seek_safe_platform() && on_ground && !climbing)
        {
            if let SubGoal::StepUp { .. } = self.executor.active_goal() {
                goal = self.executor.active_goal();
                self.navigator.explore.active_subgoal = goal;
            } else if let SubGoal::StepUp { .. } = self.navigator.explore.active_subgoal {
                goal = self.navigator.explore.active_subgoal;
            }
        }
        self.navigator.set_subgoal(goal);

        if let Some((_, kind, to)) = self.navigator.explore.pending_edge {
            let at_target = loc.node_id == to;
            let step_up = matches!(goal, SubGoal::StepUp { .. });
            let climb = matches!(
                goal,
                SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. }
            );
            // 爬绳完成只由 executor 判定（必须到顶/底台）；禁止中途旁台假完成。
            if at_target && holds_subgoal(goal) && !step_up && !climb && on_ground {
                self.navigator.on_subgoal_done(&self.graph, nav_node);
                self.executor.reset();
                goal = SubGoal::Idle;
            }
            let _ = kind;
        }

        // 必须在 plan/set_subgoal 之后刷新：否则 GoTo 看不到 pending_target，
        // 会在 |dx|<=tol 时假完成（死胡同台边缘空转 idle/goto_done）。
        ctx.pending_target = self
            .navigator
            .explore
            .pending_edge
            .map(|(_, _, to)| to);

        // 挂绳处理：
        // - 已在绳顶平台：记 ascent；非爬绳目标则侧移下绳（禁止 orphan 只按上）。
        // - 绳中且非爬绳目标：强制 ClimbUp，避免 Idle 时左右无效卡绳。
        if climbing {
            let at_top = self.at_climb_top_platform(nav_x, nav_y).is_some();
            if at_top {
                self.navigator
                    .note_arrived_climb_top(&self.graph, nav_node, None);
                if !matches!(
                    goal,
                    SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. }
                ) {
                    let dir = if self.navigator.explore.patrol_dir >= 0.0 {
                        1.0
                    } else {
                        -1.0
                    };
                    goal = SubGoal::Patrol { dir };
                    self.navigator.set_subgoal(goal);
                    ctx.pending_target = None;
                }
            } else if !matches!(
                goal,
                SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. }
            ) {
                self.navigator.explore.escape_ticks = 0;
                goal = SubGoal::ClimbUp { rope_x: nav_x };
                self.navigator.set_subgoal(goal);
                ctx.pending_target = None;
            }
        }

        let (mut nav_out, exec_result, reason) =
            self.executor
                .step(&self.config, &self.graph, &ctx, goal);
        self.last_reason = reason;
        // 撤离时逼近高台/绳：接近阶段也起跳，避免只走路永远进不了起跳窗。
        if self.survival.force_seek_safe_platform()
            && matches!(
                goal,
                SubGoal::StepUp { .. } | SubGoal::ClimbUp { .. }
            )
            && matches!(
                reason,
                "step_up_approach"
                    | "step_up_wait"
                    | "climb_align"
                    | "climb_up_nudge"
                    | "climb_up_wait"
                    | "climb_up"
            )
        {
            nav_out.jump = true;
        }

        let block_edge = !low_confidence;

        match exec_result {
            ExecutorResult::Done => {
                // Idle 每帧 Done：安全回血才站桩，否则立刻重规划。
                if matches!(goal, SubGoal::Idle) {
                    if !self.survival.prefer_idle_heal(obs, nav_y) {
                        let dir = if self.navigator.explore.patrol_dir.abs() > 0.1 {
                            self.navigator.explore.patrol_dir.signum()
                        } else {
                            1.0
                        };
                        self.navigator
                            .set_subgoal(SubGoal::Patrol { dir });
                        self.last_reason = "idle_replan_patrol";
                    }
                } else if self.navigator.explore.pending_edge.is_some() {
                    // Walk 完成时若视觉节点滞后，把定位掰到目的节点，否则下一拍仍从 from 重规划。
                    self.snap_walk_dest_if_needed(nav_node, nav_x);
                    let done_node = self.localizer.state.node_id;
                    let was_walk = self
                        .navigator
                        .explore
                        .pending_edge
                        .is_some_and(|(_, k, _)| k == EdgeKind::Walk);
                    self.navigator.on_subgoal_done(&self.graph, done_node);
                    self.executor.reset();
                    // 底层接缝 Walk 秒完成（47↔58）：立刻改爬绳，禁止每帧 goto_done 空转。
                    if was_walk
                        && self
                            .graph
                            .get(done_node)
                            .is_some_and(|n| n.y >= 1180.0)
                        && (self.navigator.explore.visited.len() >= 8
                            || self.navigator.bottom_visited_count(&self.graph) >= 4)
                    {
                        self.navigator.clear_climb_blocks();
                        if let Some(climb) = self.navigator.plan_path_to_climb_pub(
                            &self.graph,
                            done_node,
                            nav_x,
                            self.config.goto_tolerance_px,
                        ) {
                            self.navigator.set_subgoal(climb);
                        }
                    }
                } else {
                    // 无 pending（孤儿爬绳 / climb_retry 侧向 goto）：清 active 以便重规划。
                    if matches!(goal, SubGoal::ClimbUp { .. }) && nav_y < 1100.0 {
                        self.navigator
                            .note_arrived_climb_top(&self.graph, nav_node, None);
                    }
                    self.navigator.explore.active_subgoal = SubGoal::Idle;
                    self.navigator.explore.subgoal_ticks = 0;
                    self.executor.reset();
                }
            }
            ExecutorResult::Failed => {
                // 高台禁止跳崖 / 台阶掉下：立刻改向上，禁止 Idle 后回底空转。
                if reason == "patrol_elevated_no_drop"
                    || reason == "step_up_fell"
                    || reason == "climb_finish_fell"
                    || reason == "climb_finish_missed"
                {
                    if reason == "step_up_fell"
                        || reason == "climb_finish_fell"
                        || reason == "climb_finish_missed"
                    {
                        self.navigator.on_subgoal_failed(
                            &self.graph,
                            self.config.edge_block_ticks,
                            block_edge,
                        );
                    } else {
                        self.navigator.explore.active_subgoal = SubGoal::Idle;
                        self.navigator.explore.pending_edge = None;
                        self.navigator.explore.explore_path.clear();
                    }
                    self.navigator.explore.ascent_hold_ticks =
                        self.navigator.explore.ascent_hold_ticks.max(180);
                    self.navigator.explore.prefer_forward_explore = true;
                    if is_elevated_node(&self.graph, nav_node) {
                        self.navigator
                            .note_arrived_climb_top(&self.graph, nav_node, None);
                    }
                    if let Some(up) =
                        self.navigator.plan_continue_ascent(&self.graph, nav_node)
                    {
                        self.navigator.set_subgoal(up);
                    } else if is_elevated_node(&self.graph, nav_node) {
                        let dir = self.navigator.explore.patrol_dir.signum();
                        self.navigator.set_subgoal(SubGoal::Patrol {
                            dir: if dir == 0.0 { 1.0 } else { dir },
                        });
                    } else if let Some(climb) = self.navigator.plan_path_to_climb_pub(
                        &self.graph,
                        nav_node,
                        nav_x,
                        self.config.goto_tolerance_px,
                    ) {
                        // 掉回底层：立刻改爬绳上楼，禁止 patrol/goto 在右侧空转。
                        self.navigator.set_subgoal(climb);
                    }
                    self.executor.reset();
                } else if matches!(goal, SubGoal::Patrol { .. }) {
                    // 巡逻撞边：若该侧有 Fall，优先下跳而不是立刻翻向走回。
                    let fall_side = match goal {
                        SubGoal::Patrol { dir } if dir > 0.0 => Some(Side::Right),
                        SubGoal::Patrol { dir } if dir < 0.0 => Some(Side::Left),
                        _ => None,
                    };
                    let fall_edge = fall_side.and_then(|side| {
                        let from_y = self.graph.get(nav_node).map(|n| n.y).unwrap_or(0.0);
                        self.graph.edges.iter().find(|e| {
                            e.from == nav_node
                                && e.kind == EdgeKind::Fall
                                && !self
                                    .navigator
                                    .explore
                                    .blocked_edges
                                    .contains_key(&(e.from, e.kind, e.to))
                                && match side {
                                    Side::Right => e.target_x
                                        >= self
                                            .graph
                                            .get(nav_node)
                                            .map(|n| n.x_max - 16.0)
                                            .unwrap_or(e.target_x),
                                    Side::Left => e.target_x
                                        <= self
                                            .graph
                                            .get(nav_node)
                                            .map(|n| n.x_min + 16.0)
                                            .unwrap_or(e.target_x),
                                }
                                // 中/上层撞边：不要 Fall 回底层，改翻向或爬绳。
                                && self.graph.get(e.to).is_some_and(|dest| {
                                    !(from_y < 1180.0 && dest.y >= 1180.0)
                                })
                        })
                    });
                    if let Some(edge) = fall_edge {
                        self.navigator.explore.explore_path.clear();
                        self.navigator.explore.last_walk_hop = None;
                        self.navigator.explore.pending_edge =
                            Some((edge.from, edge.kind, edge.to));
                        self.navigator
                            .set_subgoal(self.graph.edge_to_subgoal(edge));
                        self.executor.reset();
                    } else {
                        self.navigator.explore.patrol_dir *= -1.0;
                        self.navigator.explore.last_walk_hop = None;
                        self.navigator.clear_blocked_from(nav_node);
                        self.navigator.explore.explore_path.clear();
                        self.navigator.explore.active_subgoal = SubGoal::Idle;
                        self.executor.reset();
                    }
                } else if matches!(
                    goal,
                    SubGoal::StepUp { .. } | SubGoal::ClimbDown { .. } | SubGoal::ClimbUp { .. }
                ) {
                    let pending = self.navigator.explore.pending_edge;
                    let is_climb = matches!(
                        goal,
                        SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. }
                    );
                    if is_climb {
                        let rope_x = match goal {
                            SubGoal::ClimbUp { rope_x } | SubGoal::ClimbDown { rope_x } => rope_x,
                            _ => nav_x,
                        };
                        let (from, kind, to) = pending.unwrap_or((
                            nav_node,
                            if matches!(goal, SubGoal::ClimbUp { .. }) {
                                EdgeKind::ClimbUp
                            } else {
                                EdgeKind::ClimbDown
                            },
                            nav_node,
                        ));
                        let retrying = self.navigator.begin_climb_retry(
                            from,
                            to,
                            kind,
                            rope_x,
                            nav_x,
                            self.config.climb_retry_max,
                        );
                        if retrying {
                            // 未达上限：不封边，侧移后重抓。
                            self.navigator.explore.pending_edge = None;
                            self.navigator.explore.active_subgoal = SubGoal::Idle;
                            self.navigator.explore.subgoal_ticks = 0;
                            self.navigator.explore.explore_path.clear();
                        } else {
                            // 侧移重试耗尽：强制封边（低置信度也不放行，否则同一绳死循环）。
                            self.navigator.on_subgoal_failed(
                                &self.graph,
                                self.config.climb_block_ticks,
                                true,
                            );
                            self.navigator.begin_escape(
                                if matches!(goal, SubGoal::ClimbUp { .. }) {
                                    -1.0
                                } else {
                                    1.0
                                },
                                12,
                            );
                        }
                    } else {
                        self.navigator.on_subgoal_failed(
                            &self.graph,
                            self.config.edge_block_ticks,
                            block_edge,
                        );
                        // StepUp 失败：顺带封掉同节点其它台阶，避免 224/314/302 轮流空跳。
                        if matches!(goal, SubGoal::StepUp { .. }) && block_edge {
                            if let Some((from, _, _)) = pending {
                                for e in &self.graph.edges {
                                    if e.from == from && e.kind == EdgeKind::StepUp {
                                        self.navigator.explore.blocked_edges.insert(
                                            (e.from, e.kind, e.to),
                                            self.config.edge_block_ticks,
                                        );
                                    }
                                }
                            }
                        }
                        // 失败后沿起跳方向逃逸，不要立刻 -dir 折返（跳上台又往回）。
                        let esc = {
                            let d = self.executor.step_jump_dir();
                            if d != 0.0 {
                                d
                            } else {
                                self.navigator.explore.patrol_dir
                            }
                        };
                        self.navigator.begin_escape(esc, 12);
                    }
                    self.executor.reset();
                } else if matches!(goal, SubGoal::GoTo { .. } | SubGoal::WalkOff { .. }) {
                    // climb_retry 侧移站位走不动：换下一档，禁止 force_unstuck 清封边后重挂同一绳。
                    if let Some(retry) = self.navigator.explore.climb_retry {
                        let retrying = self.navigator.begin_climb_retry(
                            retry.from,
                            retry.to,
                            retry.kind,
                            retry.rope_x,
                            nav_x,
                            self.config.climb_retry_max,
                        );
                        if retrying {
                            self.navigator.explore.active_subgoal = SubGoal::Idle;
                            self.navigator.explore.subgoal_ticks = 0;
                            self.navigator.explore.explore_path.clear();
                            self.navigator.explore.pending_edge = None;
                        } else {
                            let hold = self.config.climb_block_ticks.max(240);
                            self.navigator
                                .explore
                                .blocked_edges
                                .insert((retry.from, retry.kind, retry.to), hold);
                            self.navigator.explore.climb_retry = None;
                            self.navigator.explore.active_subgoal = SubGoal::Idle;
                            self.navigator.explore.subgoal_ticks = 0;
                            self.navigator.explore.explore_path.clear();
                            self.navigator.explore.pending_edge = None;
                            let esc = if retry.kind == EdgeKind::ClimbUp {
                                -1.0
                            } else {
                                1.0
                            };
                            self.navigator.begin_escape(
                                super::navigator::Navigator::clamp_escape_dir(
                                    &self.graph,
                                    nav_node,
                                    nav_x,
                                    esc,
                                ),
                                24,
                            );
                        }
                        self.executor.reset();
                    } else {
                        // 右墙/物理走不动时 YOLO 仍报可走 → goto_stalled。必须立刻封边，
                        // 低置信度也不能跳过（否则会无限重试同一条 Walk）。
                        let force_block = matches!(
                            reason,
                            "goto_stalled" | "goto_blocked" | "goto_edge_stuck"
                        );
                        self.navigator.on_subgoal_failed(
                            &self.graph,
                            self.config.edge_block_ticks,
                            block_edge || force_block,
                        );
                        let esc_dir = match goal {
                            SubGoal::GoTo { x } => {
                                if nav_x < x {
                                    -1.0
                                } else {
                                    1.0
                                }
                            }
                            SubGoal::WalkOff { side } => {
                                if side == Side::Right {
                                    -1.0
                                } else {
                                    1.0
                                }
                            }
                            _ => -self.navigator.explore.patrol_dir,
                        };
                        let esc_dir = super::navigator::Navigator::clamp_escape_dir(
                            &self.graph,
                            nav_node,
                            nav_x,
                            esc_dir,
                        );
                        self.navigator.explore.last_walk_hop = None;
                        // 底层 Walk 卡死：直接 force_unstuck 优先爬绳，禁止顶墙 escape 空转。
                        let bottom = self
                            .graph
                            .get(nav_node)
                            .is_some_and(|n| n.y >= 1180.0);
                        if bottom || force_block {
                            self.navigator.force_unstuck(
                                &self.graph,
                                nav_node,
                                nav_x,
                                self.config.goto_tolerance_px,
                                esc_dir,
                            );
                        } else {
                            self.navigator.begin_escape(esc_dir, 24);
                        }
                        self.executor.reset();
                    }
                } else {
                    self.navigator.on_subgoal_failed(
                        &self.graph,
                        self.config.edge_block_ticks,
                        block_edge,
                    );
                    self.executor.reset();
                }
                self.navigator.explore.subgoal_failures =
                    self.navigator.explore.subgoal_failures.saturating_add(1);
            }
            ExecutorResult::Running => {
                self.navigator.on_subgoal_tick();
                let step_up = matches!(goal, SubGoal::StepUp { .. });
                let timed_out = if step_up {
                    self.navigator.explore.subgoal_ticks
                        > self.config.step_up_timeout_ticks
                } else {
                    self.navigator.explore.subgoal_ticks > self.config.subgoal_timeout_ticks
                };
                if timed_out {
                    if step_up {
                        self.navigator.on_subgoal_failed(
                            &self.graph,
                            self.config.edge_block_ticks,
                            block_edge,
                        );
                        self.navigator.begin_escape(self.executor.step_escape_dir(), 12);
                    } else if matches!(goal, SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. }) {
                        let rope_x = match goal {
                            SubGoal::ClimbUp { rope_x } | SubGoal::ClimbDown { rope_x } => rope_x,
                            _ => nav_x,
                        };
                        let pending = self.navigator.explore.pending_edge;
                        let (from, kind, to) = pending.unwrap_or((
                            nav_node,
                            if matches!(goal, SubGoal::ClimbUp { .. }) {
                                EdgeKind::ClimbUp
                            } else {
                                EdgeKind::ClimbDown
                            },
                            nav_node,
                        ));
                        let retrying = self.navigator.begin_climb_retry(
                            from,
                            to,
                            kind,
                            rope_x,
                            nav_x,
                            self.config.climb_retry_max,
                        );
                        if retrying {
                            self.navigator.explore.pending_edge = None;
                            self.navigator.explore.active_subgoal = SubGoal::Idle;
                            self.navigator.explore.subgoal_ticks = 0;
                            self.navigator.explore.explore_path.clear();
                        } else {
                            // 超时且侧移重试耗尽：强制封边。
                            self.navigator.on_subgoal_failed(
                                &self.graph,
                                self.config.climb_block_ticks,
                                true,
                            );
                            self.navigator.begin_escape(
                                if matches!(goal, SubGoal::ClimbUp { .. }) {
                                    -1.0
                                } else {
                                    1.0
                                },
                                12,
                            );
                        }
                        self.executor.reset();
                    } else if let SubGoal::GoTo { x } = goal {
                        // 已在目标附近却因定位/latch 抖到超时：当完成，勿封边。
                        let near = (nav_x - x).abs()
                            <= self.config.goto_tolerance_px.max(20.0) + 28.0;
                        let on_dest = self
                            .navigator
                            .explore
                            .pending_edge
                            .and_then(|(_, _, to)| self.graph.get(to).map(|d| (to, d)))
                            .is_some_and(|(_, d)| {
                                nav_x >= d.x_min - 8.0
                                    && nav_x <= d.x_max + 8.0
                                    && (nav_y - d.y).abs() < 120.0
                            });
                        if near || on_dest {
                            self.snap_walk_dest_if_needed(nav_node, nav_x);
                            let done_node = self.localizer.state.node_id;
                            self.navigator.on_subgoal_done(&self.graph, done_node);
                        } else {
                            self.navigator.on_subgoal_failed(
                                &self.graph,
                                self.config.edge_block_ticks,
                                block_edge,
                            );
                        }
                    } else {
                        self.navigator.on_subgoal_failed(
                            &self.graph,
                            self.config.edge_block_ticks,
                            block_edge,
                        );
                    }
                    self.executor.reset();
                }
            }
        }

        let combat_active = if self.survival.suppress_combat() {
            false
        } else {
            self.combat.is_active()
        };
        // 脚边掉落：战斗中也按拾取键，不改位移（由 arbiter 叠 pick_up）。
        let pickup_near = if obs_drop_in_pickup_range(obs) {
            self.pickup.try_pickup(
                &ctx,
                &self.graph,
                &self.config,
                true,
                goal,
                ctx.farm_band_mobs,
            )
        } else {
            None
        };
        // 追拾：金币可追；farm 近只清怪时仍允许追金币（pickup 内部区分）。
        let pickup_chase = if pickup_near.is_none() && !matches!(
            goal,
            SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. }
        ) {
            self.pickup.try_pickup(
                &ctx,
                &self.graph,
                &self.config,
                false,
                goal,
                ctx.farm_band_mobs,
            )
        } else {
            None
        };

        let combat_frame = self.combat.intent_frame();

        if self.progress.tick(
            nav_x,
            nav_y,
            nav_node,
            self.navigator.visited_count(),
            // Patrol/边失败已在上面处理翻向；这里不要再用 Failed 触发二次翻向。
            false,
        ) {
            // 封住当前卡死的边；优先改挂爬绳/离开片段，避免底层 escape patrol 死循环。
            let esc = if matches!(
                self.navigator.explore.active_subgoal,
                SubGoal::ClimbUp { .. }
            ) {
                -1.0
            } else if matches!(
                self.navigator.explore.active_subgoal,
                SubGoal::ClimbDown { .. }
            ) {
                1.0
            } else {
                -self.navigator.explore.patrol_dir
            };
            let esc = super::navigator::Navigator::clamp_escape_dir(
                &self.graph,
                nav_node,
                nav_x,
                esc,
            );
            self.navigator.force_unstuck(
                &self.graph,
                nav_node,
                nav_x,
                self.config.goto_tolerance_px,
                esc,
            );
            self.navigator.explore.patrol_dir *= -1.0;
            self.executor.reset();
            self.combat.reset();
        }

        let force_transit = self.is_bottom_right_climb_approach(goal, nav_node, nav_x)
            || self.survival.force_climb_escape();
        self.nav_intent = InterruptArbiter::merge(
            nav_out,
            pickup_near,
            pickup_chase,
            combat_frame,
            combat_active,
            goal,
            force_transit,
        );

        let (step_stall, step_jump_dir, step_jumped, step_jump_cd) = self.executor.step_diag();
        let pending = self.navigator.explore.pending_edge;
        let iw = WINDOW_W as f32;
        let ih = WINDOW_H as f32;
        self.last_diag = NavDiagSnapshot {
            goal: self.navigator.explore.active_subgoal,
            exec: exec_result,
            nav_node,
            est_node: loc.node_id,
            nav_x,
            nav_y,
            est_x: loc.world_x,
            est_y: loc.world_y,
            pending_from: pending.map(|(f, _, _)| f),
            pending_kind: pending.map(|(_, k, _)| k),
            pending_to: pending.map(|(_, _, t)| t),
            subgoal_ticks: self.navigator.explore.subgoal_ticks,
            subgoal_failures: self.navigator.explore.subgoal_failures,
            escape_ticks: self.navigator.explore.escape_ticks,
            escape_dir: self.navigator.explore.escape_dir,
            step_stall,
            step_jump_dir,
            step_jumped,
            step_jump_cd,
            walk_right: ctx.walk_right_ok,
            walk_left: ctx.walk_left_ok,
            drop_right: ctx.drop_ahead_right,
            drop_left: ctx.drop_ahead_left,
            obs_step_up: obs_step_up_dx(obs, iw, ih),
            grounded_est: on_ground,
            visual_conf: visual_conf,
            blocked_edges: self.navigator.explore.blocked_edges.len(),
            combat_active,
            farm_local: ctx.farm_band_mobs,
        };

        self.nav_intent
    }

    /// 低血撤离：爬绳 → 寻路上楼 → 上台阶 → 离台 → 巡逻。永不返回 Idle。
    fn plan_heal_escape(&mut self, nav_node: u32, nav_x: f32) -> SubGoal {
        self.navigator.clear_climb_blocks();
        let tol = self.config.goto_tolerance_px;
        if let Some(edge) = self.graph.edges.iter().find(|e| {
            e.from == nav_node
                && e.kind == EdgeKind::ClimbUp
                && !self
                    .navigator
                    .explore
                    .blocked_edges
                    .contains_key(&(e.from, e.kind, e.to))
        }) {
            return self.navigator.commit_edge(&self.graph, edge);
        }
        if let Some(g) = self
            .navigator
            .plan_path_to_climb_pub(&self.graph, nav_node, nav_x, tol)
        {
            return g;
        }
        if let Some(g) = self.navigator.plan_continue_ascent(&self.graph, nav_node) {
            return g;
        }
        if let Some(g) = self.navigator.plan_leave_segment(&self.graph, nav_node) {
            return g;
        }
        let dir = if self.navigator.explore.patrol_dir.abs() > 0.1 {
            self.navigator.explore.patrol_dir.signum()
        } else {
            1.0
        };
        SubGoal::Patrol { dir }
    }

    fn is_bottom_right_climb_approach(
        &self,
        goal: SubGoal,
        nav_node: u32,
        nav_x: f32,
    ) -> bool {
        let Some(n) = self.graph.get(nav_node) else {
            return false;
        };
        if n.y < 1180.0 || nav_x < n.x_max - 48.0 {
            return false;
        }
        matches!(goal, SubGoal::GoTo { x } if x < nav_x - 4.0)
            && self
                .navigator
                .explore
                .pending_edge
                .is_some_and(|(_, k, _)| k == EdgeKind::Walk)
    }

    fn snap_walk_dest_if_needed(&mut self, nav_node: u32, nav_x: f32) {
        let Some((_, kind, to)) = self.navigator.explore.pending_edge else {
            return;
        };
        if kind != EdgeKind::Walk || nav_node == to {
            return;
        }
        let Some(dest) = self.graph.get(to) else {
            return;
        };
        if nav_x >= dest.x_min - 8.0 && nav_x <= dest.x_max + 8.0 {
            self.localizer.state.node_id = to;
            self.prev_node = to;
        }
    }

    pub fn refresh_melee_hold(&mut self, obs: &[f32; OBS_DIM], facing: f32) -> InputFrame {
        // 撤离/回血：禁止叠攻击，专心跑路。
        if self.survival.suppress_combat() {
            return self.nav_intent;
        }
        let goal = self.navigator.explore.active_subgoal;
        let nav = self.nav_intent;
        let nav_node = self.localizer.state.node_id;
        let nav_x = self.localizer.state.world_x;
        let climb_escape = self.is_bottom_right_climb_approach(goal, nav_node, nav_x);

        if goal.is_transit() || climb_escape {
            // 换层 / 最右底层去爬绳：补 attack+朝怪；非出刀帧保留导航左右。
            self.nav_intent = InterruptArbiter::refresh_melee_hold(obs, facing, nav, goal);
            if climb_escape && !self.nav_intent.attack {
                let mut out = self.nav_intent;
                out.left = nav.left || out.left;
                out.right = false;
                if nav.left {
                    out.left = true;
                }
                self.nav_intent = out;
            }
            return self.nav_intent;
        }

        // 可砍带：站砍并朝怪（方向由 refresh_melee_hold 填）。
        let melee = InterruptArbiter::refresh_melee_hold(obs, facing, nav, goal);
        if melee.attack && !self.combat.is_active() {
            self.nav_intent = melee;
            return self.nav_intent;
        }

        if self.combat.is_active() {
            let combat = self.combat.intent_frame();
            if combat.left || combat.right {
                // Approach，或 Strike 出刀瞬间带朝向。
                let mut out = combat;
                out.jump = false;
                self.nav_intent = out;
                return self.nav_intent;
            }
            // Strike CD / Hold：无方向时用 melee 的朝怪脉冲补刀。
            let mut out = combat;
            out.left = false;
            out.right = false;
            out.jump = false;
            if !out.attack && melee.attack {
                out.attack = true;
                out.left = melee.left;
                out.right = melee.right;
            }
            self.nav_intent = out;
            return self.nav_intent;
        }

        self.nav_intent = melee;
        self.nav_intent
    }

    pub fn active_goal(&self) -> SubGoal {
        self.navigator.explore.active_subgoal
    }

    pub fn nav_intent(&self) -> InputFrame {
        self.nav_intent
    }
}
