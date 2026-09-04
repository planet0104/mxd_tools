use super::map_graph::MapGraph;
use super::types::{EdgeKind, ExploreState, GraphEdge, PlatformNodeId, SubGoal};

fn subgoal_committed(goal: SubGoal) -> bool {
    matches!(
        goal,
        SubGoal::GoTo { .. }
            | SubGoal::WalkOff { .. }
            | SubGoal::ClimbUp { .. }
            | SubGoal::ClimbDown { .. }
            | SubGoal::StepUp { .. }
            | SubGoal::Patrol { .. }
    )
}

/// Patrol 淇濇寔涓婇檺锛堥€昏緫甯р増瑙嗚鍐崇瓥娆℃暟绱鍦?subgoal_ticks锛?
const PATROL_HOLD_TICKS: u32 = 90;
/// 抓绳探测站位（相对绳 x，约半精灵~一身宽左右轮换）。
const CLIMB_PROBE_OFFSETS: [f32; 7] = [0.0, -28.0, 28.0, -48.0, 48.0, -68.0, 68.0];

#[derive(Debug, Clone)]
pub struct Navigator {
    pub explore: ExploreState,
    spawn_node: PlatformNodeId,
    farm_node: PlatformNodeId,
    farm_cleared: bool,
    patrol_seed: u64,
}

impl Navigator {
    pub fn new(spawn_node: PlatformNodeId, graph: &MapGraph, seed: u64) -> Self {
        let patrol_route = graph.build_patrol_route(spawn_node, seed);
        // 初始左右对半；之后撞边/脱困才翻向，并沿该方向粘住。
        let patrol_dir = if seed & 1 == 0 { 1.0 } else { -1.0 };
        Self {
            explore: ExploreState {
                patrol_dir,
                patrol_route,
                active_subgoal: SubGoal::Idle,
                ..Default::default()
            },
            spawn_node,
            farm_node: spawn_node,
            farm_cleared: false,
            patrol_seed: seed,
        }
    }

    pub fn reset(&mut self, spawn_node: PlatformNodeId, graph: &MapGraph) {
        *self = Self::new(spawn_node, graph, self.patrol_seed);
    }

    pub fn mark_farm_cleared(&mut self) {
        self.farm_cleared = true;
    }

    pub fn visited_count(&self) -> usize {
        self.explore.visited.len()
    }

    pub fn tick_blocked_edges(&mut self) {
        self.explore
            .blocked_edges
            .retain(|_, ticks| {
                *ticks = ticks.saturating_sub(1);
                *ticks > 0
            });
        if self.explore.ascent_hold_ticks > 0 {
            self.explore.ascent_hold_ticks = self.explore.ascent_hold_ticks.saturating_sub(1);
            if self.explore.ascent_hold_ticks == 0 {
                self.explore.last_ascent_hop = None;
            }
        }
    }

    pub fn begin_escape(&mut self, dir: f32, ticks: u32) {
        self.explore.escape_ticks = ticks;
        self.explore.escape_dir = dir.signum();
        self.explore.active_subgoal = SubGoal::Idle;
        self.explore.subgoal_ticks = 0;
        self.explore.subgoal_failures = 0;
        self.explore.explore_path.clear();
        self.explore.sweep_after_ascent = None;
        // 保留 last_ascent / hold：逃逸不应立刻允许掉回刚跳上的起点。
        self.explore.prefer_forward_explore = false;
    }

    /// 逃逸方向贴墙则翻向，避免最右台顶墙空转几十分钟。
    pub fn clamp_escape_dir(
        graph: &MapGraph,
        node_id: PlatformNodeId,
        world_x: f32,
        dir: f32,
    ) -> f32 {
        let mut d = dir.signum();
        if d == 0.0 {
            d = 1.0;
        }
        let Some(n) = graph.get(node_id) else {
            return d;
        };
        if d > 0.0 && world_x >= n.x_max - 28.0 {
            return -1.0;
        }
        if d < 0.0 && world_x <= n.x_min + 28.0 {
            return 1.0;
        }
        d
    }

    fn at_escape_wall(graph: &MapGraph, node_id: PlatformNodeId, world_x: f32, dir: f32) -> bool {
        let Some(n) = graph.get(node_id) else {
            return false;
        };
        if dir > 0.0 {
            world_x >= n.x_max - 28.0
        } else if dir < 0.0 {
            world_x <= n.x_min + 28.0
        } else {
            false
        }
    }

    /// 抓绳失败：换下一档横向站位再试跳（半精灵宽扫描）。
    pub fn begin_climb_retry(
        &mut self,
        from: PlatformNodeId,
        to: PlatformNodeId,
        kind: EdgeKind,
        rope_x: f32,
        _nav_x: f32,
        max_attempts: u32,
    ) -> bool {
        let prev = self.explore.climb_retry.filter(|r| {
            r.from == from && r.to == to && r.kind == kind
        });
        let attempts = prev.map(|r| r.attempts.saturating_add(1)).unwrap_or(1);
        // 首次失败从下标 1 起（0=绳心已在首轮 Climb 试过）。
        let offset_idx = prev
            .map(|r| r.offset_idx.saturating_add(1))
            .unwrap_or(1);
        if attempts > max_attempts || (offset_idx as usize) >= CLIMB_PROBE_OFFSETS.len() {
            self.explore.climb_retry = None;
            return false;
        }
        self.explore.climb_retry = Some(super::types::ClimbRetry {
            from,
            to,
            kind,
            rope_x,
            offset_idx,
            attempts,
        });
        self.explore.active_subgoal = SubGoal::Idle;
        self.explore.subgoal_ticks = 0;
        self.explore.explore_path.clear();
        self.explore.pending_edge = None;
        true
    }

    fn plan_climb_retry(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
        world_x: f32,
        goto_tol: f32,
    ) -> Option<SubGoal> {
        let retry = *self.explore.climb_retry.as_ref()?;
        let near_from = node_id == retry.from
            || Self::same_platform_segment(graph, node_id, retry.from)
            || graph.get(retry.from).is_some_and(|n| {
                world_x >= n.x_min - 40.0
                    && world_x <= n.x_max + 40.0
                    && graph
                        .get(node_id)
                        .is_some_and(|c| (c.y - n.y).abs() < 80.0)
            });
        if !near_from {
            return None;
        }
        let off = CLIMB_PROBE_OFFSETS
            .get(retry.offset_idx as usize)
            .copied()
            .unwrap_or(0.0);
        let aim = retry.rope_x + off;
        let tol = goto_tol.max(20.0);
        if (world_x - aim).abs() > tol {
            // 先走到探测站位，不设 pending（避免当成 Walk 完成）。
            return Some(SubGoal::GoTo { x: aim });
        }
        self.explore
            .blocked_edges
            .remove(&(retry.from, retry.kind, retry.to));
        let edge = graph.edges.iter().find(|e| {
            e.from == retry.from && e.to == retry.to && e.kind == retry.kind
        })?;
        // 用探测站位作为抓绳 aim，在该 x 起跳。
        let mut goal = self.commit_edge(graph, edge);
        match &mut goal {
            SubGoal::ClimbUp { rope_x } | SubGoal::ClimbDown { rope_x } => {
                *rope_x = aim;
            }
            _ => {}
        }
        Some(goal)
    }

    pub fn plan(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
        world_x: f32,
        farm_cleared: bool,
        goto_tol: f32,
    ) -> SubGoal {
        self.tick_blocked_edges();

        if self.explore.escape_ticks > 0 {
            // 已顶在逃逸方向的台缘：立刻结束逃逸并重规划（底层则去爬绳），禁止顶墙空转。
            if Self::at_escape_wall(graph, node_id, world_x, self.explore.escape_dir) {
                self.explore.escape_ticks = 0;
                self.explore.prefer_forward_explore = false;
                if Self::is_bottom_band(graph, node_id) {
                    if let Some(goal) = self.plan_path_to_climb(graph, node_id, world_x, goto_tol)
                    {
                        return goal;
                    }
                }
            } else {
                self.explore.escape_ticks -= 1;
                return SubGoal::Patrol {
                    dir: self.explore.escape_dir,
                };
            }
        }

        if farm_cleared {
            self.farm_cleared = true;
        }

        // 掉回底层：清掉「继续向上」粘滞，否则 patrol 卡住时不改爬绳。
        if Self::is_bottom_band(graph, node_id) {
            self.explore.prefer_forward_explore = false;
            self.explore.sweep_after_ascent = None;
            self.explore.ascent_hold_ticks = 0;
        }

        if let Some(goal) = self.plan_climb_retry(graph, node_id, world_x, goto_tol) {
            return goal;
        }

        // 刚跳上/爬上的台：先继续向上 StepUp，再扫台；禁止中段落地后原地空转/跳下。
        if !Self::is_bottom_band(graph, node_id)
            && (self.explore.ascent_hold_ticks > 0 || self.explore.prefer_forward_explore)
        {
            if let Some(goal) = self.plan_continue_ascent(graph, node_id) {
                return goal;
            }
        }

        // 刚跳上/爬上的台：先沿开荒方向扫到边缘，禁止立刻规划反向 goto。
        if let Some(goal) = self.plan_ascent_sweep(graph, node_id, world_x) {
            return goal;
        }

        // 巡逻卡住时也尝试离开同台接缝，不只 climb/cross。
        if let SubGoal::Patrol { dir } = self.explore.active_subgoal {
            if self.explore.subgoal_ticks < PATROL_HOLD_TICKS {
                // 底层已开过足够台：打断长巡逻，改爬绳（日志右侧 patrol(-1) 空转）。
                if Self::is_bottom_band(graph, node_id)
                    && self.explore.subgoal_ticks > 24
                    && (self.bottom_visited_count(graph) >= 5 || self.explore.visited.len() >= 10)
                {
                    self.clear_climb_blocks();
                    if let Some(goal) =
                        self.plan_path_to_climb(graph, node_id, world_x, goto_tol)
                    {
                        return goal;
                    }
                }
                let walks_done = self.local_walk_neighbors_visited(graph, node_id);
                let stuck_patrol = self.explore.subgoal_failures >= 3;
                let holding_ascent = !Self::is_bottom_band(graph, node_id)
                    && (self.explore.prefer_forward_explore
                        || self.explore.sweep_after_ascent.is_some()
                        || self.explore
                            .last_ascent_hop
                            .is_some_and(|(_, to)| to == node_id));
                if walks_done || stuck_patrol {
                    // 粘滞向上：先 StepUp 继续升层，禁止先 path_to_climb 绕回底绳。
                    if holding_ascent {
                        if let Some(goal) = self.plan_continue_ascent(graph, node_id) {
                            return goal;
                        }
                        if let Some(goal) = self.plan_forward_walk(graph, node_id) {
                            return goal;
                        }
                    } else {
                        if let Some(goal) =
                            self.plan_path_to_climb(graph, node_id, world_x, goto_tol)
                        {
                            return goal;
                        }
                        if let Some(goal) = self.plan_leave_segment(graph, node_id) {
                            return goal;
                        }
                        if let Some(goal) =
                            self.plan_cross_platform(graph, node_id, world_x, goto_tol)
                        {
                            return goal;
                        }
                    }
                    if walks_done && !holding_ascent {
                        if let Some(goal) = self.plan_after_clearing_local_blocks(
                            graph, node_id, world_x, goto_tol,
                        ) {
                            return goal;
                        }
                    }
                }
                return SubGoal::Patrol { dir };
            }
            self.explore.active_subgoal = SubGoal::Idle;
        } else if subgoal_committed(self.explore.active_subgoal) {
            // 底层同层 Walk/GoTo 耗太久：向右开荒放宽；向左/原地才打断改爬绳。
            if Self::is_bottom_band(graph, node_id)
                && self
                    .explore
                    .pending_edge
                    .is_some_and(|(_, k, _)| k == EdgeKind::Walk)
            {
                let force_climb = match self.explore.active_subgoal {
                    SubGoal::GoTo { x } => {
                        let limit = if x > world_x + 24.0 { 150 } else { 40 };
                        self.explore.subgoal_ticks > limit
                    }
                    _ => false,
                };
                if force_climb {
                    self.clear_climb_blocks();
                    self.explore.explore_path.clear();
                    self.explore.pending_edge = None;
                    self.explore.last_walk_hop = None;
                    if let Some(goal) = self.plan_path_to_climb(graph, node_id, world_x, goto_tol)
                    {
                        return goal;
                    }
                }
            }
            return self.explore.active_subgoal;
        }

        let Some(node) = graph.get(node_id) else {
            return self.plan_patrol_route(graph, node_id);
        };

        let at_left = world_x <= node.x_min + 20.0;
        let at_right = world_x >= node.x_max - 20.0;

        if !self.explore.visited.contains(&node_id) {
            if self.explore.patrol_dir > 0.0 {
                if at_right {
                    self.explore.visited.insert(node_id);
                } else {
                    return SubGoal::Patrol { dir: 1.0 };
                }
            } else if at_left {
                self.explore.visited.insert(node_id);
            } else {
                return SubGoal::Patrol { dir: -1.0 };
            }
        }

        if !self.explore.visited.contains(&node_id) {
            return SubGoal::Patrol {
                dir: self.explore.patrol_dir,
            };
        }

        // 底层（y≥1180）：短暂同层开荒后优先爬绳上楼；勿等扫完所有底台。
        if Self::is_bottom_band(graph, node_id) {
            let at_right = world_x >= node.x_max - 28.0;
            let at_left = world_x <= node.x_min + 28.0;
            let bottom_vis = self.bottom_visited_count(graph);
            // 贴边 / 已看过若干底台 / 农怪结束 → 立刻找绳，避免十几分钟底层空转。
            let prefer_climb = at_right
                || at_left
                || bottom_vis >= 5
                || self.explore.visited.len() >= 10
                || self.farm_cleared;
            if prefer_climb {
                self.explore.prefer_forward_explore = false;
                self.clear_climb_blocks();
                if let Some(goal) = self.plan_path_to_climb(graph, node_id, world_x, goto_tol) {
                    return goal;
                }
                if let Some(goal) =
                    self.plan_path_to_unvisited_upward(graph, node_id, world_x, goto_tol)
                {
                    return goal;
                }
            }
            if let Some(goal) =
                self.plan_path_to_unvisited_bottom(graph, node_id, world_x, goto_tol)
            {
                return goal;
            }
            self.clear_climb_blocks();
            if let Some(goal) = self.plan_path_to_climb(graph, node_id, world_x, goto_tol) {
                return goal;
            }
            if let Some(goal) = self.plan_path_to_unvisited_upward(graph, node_id, world_x, goto_tol)
            {
                return goal;
            }
        }

        // 中层：优先爬绳/再往上；禁止同层 step_up 霸占（日志 step_up(1574) 空耗）。
        if Self::is_mid_band(graph, node_id) {
            self.clear_climb_blocks();
            if let Some(goal) = self.plan_path_to_climb(graph, node_id, world_x, goto_tol) {
                return goal;
            }
            if let Some(goal) = self.plan_path_to_unvisited_upward(graph, node_id, world_x, goto_tol)
            {
                return goal;
            }
            // 中层已无上爬路径时才允许同层开荒；仍避免 Fall 回底层。
            if let Some(goal) = self.plan_forward_walk(graph, node_id) {
                return goal;
            }
        }

        // 农怪清完后：同层未访问 → 离开片段 → 爬绳 → 跨台。
        if self.farm_cleared {
            if let Some(goal) = self.plan_path_to_unvisited(graph, node_id, world_x, goto_tol) {
                return goal;
            }
            if let Some(goal) = self.plan_leave_segment(graph, node_id) {
                return goal;
            }
            if let Some(goal) = self.plan_path_to_climb(graph, node_id, world_x, goto_tol) {
                return goal;
            }
            if let Some(goal) = self.plan_cross_platform(graph, node_id, world_x, goto_tol) {
                return goal;
            }
        }

        if let Some(goal) = self.plan_path_to_unvisited(graph, node_id, world_x, goto_tol) {
            return goal;
        }

        if let Some(goal) = self.plan_path_to_climb(graph, node_id, world_x, goto_tol) {
            return goal;
        }

        if let Some(goal) = self.plan_cross_platform(graph, node_id, world_x, goto_tol) {
            return goal;
        }

        // 鍑鸿竟鍏ㄨ blocked 鏃朵細钀藉埌鍘熷湴 patrol 姝婚攣锛堝 15锛歐alk/StepUp 閮藉皝锛夈€?
        // 娓呮帀浠庡綋鍓嶈妭鐐瑰嚭鍙戠殑灏佺鍚庡啀璇曚竴娆°€?
        if let Some(goal) =
            self.plan_after_clearing_local_blocks(graph, node_id, world_x, goto_tol)
        {
            return goal;
        }

        self.plan_patrol_route(graph, node_id)
    }

    pub fn clear_blocked_from(&mut self, node_id: PlatformNodeId) -> bool {
        let before = self.explore.blocked_edges.len();
        self.explore
            .blocked_edges
            .retain(|(from, _, _), _| *from != node_id);
        self.explore.blocked_edges.len() < before
    }

    /// 久困脱困：封当前 pending、清路径；优先改挂爬绳路径，否则侧移离开。
    pub fn force_unstuck(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
        world_x: f32,
        goto_tol: f32,
        escape_dir: f32,
    ) {
        // 已在逃逸中：不要每帧重置 escape / 封边，否则永远减不完、也规划不了爬绳。
        // 但若正顶着逃逸方向的台缘，清掉逃逸并继续脱困（否则最右台永远 return）。
        if self.explore.escape_ticks > 0 {
            if Self::at_escape_wall(graph, node_id, world_x, self.explore.escape_dir) {
                self.explore.escape_ticks = 0;
            } else {
                return;
            }
        }
        self.clear_climb_blocks();
        if let Some(key) = self.explore.pending_edge.take() {
            // 脱困时不要把 Climb 边也封死，否则底层永远上不去。
            // 通向爬绳的唯一 Walk 也不要封：否则 47→58→绳 被封后只会顶墙/原地战斗。
            let block_ok = match key.1 {
                EdgeKind::ClimbUp | EdgeKind::ClimbDown => false,
                EdgeKind::Walk => {
                    let mut probe = self.explore.blocked_edges.clone();
                    probe.insert(key, 1);
                    graph
                        .path_to_nearest_kind(node_id, EdgeKind::ClimbUp, &probe)
                        .is_some()
                }
                _ => true,
            };
            if block_ok {
                self.explore.blocked_edges.insert(key, 300);
            }
        }
        self.explore.explore_path.clear();
        self.explore.climb_retry = None;
        self.explore.last_walk_hop = None;
        self.explore.sweep_after_ascent = None;
        self.explore.prefer_forward_explore = false;
        self.explore.last_ascent_hop = None;
        self.explore.ascent_hold_ticks = 0;
        self.explore.active_subgoal = SubGoal::Idle;
        self.explore.subgoal_ticks = 0;
        if let Some(goal) = self.plan_path_to_climb(graph, node_id, world_x, goto_tol) {
            self.set_subgoal(goal);
            return;
        }
        if let Some(goal) = self.plan_leave_segment(graph, node_id) {
            self.set_subgoal(goal);
            return;
        }
        self.begin_escape(
            Self::clamp_escape_dir(graph, node_id, world_x, escape_dir),
            36,
        );
    }

    fn plan_after_clearing_local_blocks(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
        world_x: f32,
        goto_tol: f32,
    ) -> Option<SubGoal> {
        if !self.clear_blocked_from(node_id) {
            return None;
        }
        self.explore.explore_path.clear();
        if let Some(goal) = self.plan_path_to_unvisited(graph, node_id, world_x, goto_tol) {
            return Some(goal);
        }
        if let Some(goal) = self.plan_cross_platform(graph, node_id, world_x, goto_tol) {
            return Some(goal);
        }
        self.plan_path_to_climb(graph, node_id, world_x, goto_tol)
    }

    pub fn commit_edge(&mut self, graph: &MapGraph, edge: &GraphEdge) -> SubGoal {
        self.explore.pending_edge = Some((edge.from, edge.kind, edge.to));
        if self.explore.prefer_forward_explore
            && (edge.kind == EdgeKind::ClimbUp || self.edge_goes_forward(graph, edge.from, edge.to))
        {
            self.explore.prefer_forward_explore = false;
        }
        // 通向爬绳台的 Walk：对准绳子 x，不要走到台缘（57 右缘 1511，绳在 1477）。
        if edge.kind == EdgeKind::Walk {
            if let Some(rope) = Self::climb_rope_on_dest(graph, edge.to) {
                return SubGoal::GoTo { x: rope };
            }
        }
        graph.edge_to_subgoal(edge)
    }

    fn climb_rope_on_dest(graph: &MapGraph, dest: PlatformNodeId) -> Option<f32> {
        let mut best: Option<(f32, f32)> = None;
        for e in &graph.edges {
            if e.from != dest || e.kind != EdgeKind::ClimbUp {
                continue;
            }
            let rope = e.rope_x.unwrap_or(e.target_x);
            let Some(n) = graph.get(dest) else {
                continue;
            };
            if rope < n.x_min - 8.0 || rope > n.x_max + 8.0 {
                continue;
            }
            let mid = (n.x_min + n.x_max) * 0.5;
            let d = (rope - mid).abs();
            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, rope));
            }
        }
        best.map(|(_, r)| r)
    }

    pub(crate) fn is_bottom_band(graph: &MapGraph, node_id: PlatformNodeId) -> bool {
        graph.get(node_id).is_some_and(|n| n.y >= 1180.0)
    }

    /// 中层：已离开底带但未到高层（日志里 step_up 空转带）。
    fn is_mid_band(graph: &MapGraph, node_id: PlatformNodeId) -> bool {
        graph
            .get(node_id)
            .is_some_and(|n| n.y >= 900.0 && n.y < 1180.0)
    }

    pub(crate) fn bottom_visited_count(&self, graph: &MapGraph) -> usize {
        self.explore
            .visited
            .iter()
            .filter(|&&id| Self::is_bottom_band(graph, id))
            .count()
    }

    pub(crate) fn clear_climb_blocks(&mut self) {
        // 只清 ClimbUp 封边以便重试上楼；保留 ClimbDown 封锁，避免刚爬上又原路下去。
        self.explore
            .blocked_edges
            .retain(|(_, kind, _), _| *kind != EdgeKind::ClimbUp);
    }

    /// 到达爬绳上端平台：记 ascent、扫台、封下爬，禁止立刻 yo-yo。
    pub fn note_arrived_climb_top(
        &mut self,
        graph: &MapGraph,
        top_node: PlatformNodeId,
        from_hint: Option<PlatformNodeId>,
    ) {
        let from = from_hint
            .or_else(|| {
                self.explore
                    .pending_edge
                    .filter(|(_, k, t)| *k == EdgeKind::ClimbUp && *t == top_node)
                    .map(|(f, _, _)| f)
            })
            .or_else(|| {
                graph.edges.iter().find_map(|e| {
                    (e.to == top_node && e.kind == EdgeKind::ClimbUp).then_some(e.from)
                })
            })
            .unwrap_or(top_node);
        self.explore.sweep_after_ascent = Some(top_node);
        self.explore.prefer_forward_explore = true;
        self.explore.last_ascent_hop = Some((from, top_node));
        self.explore.ascent_hold_ticks = 480;
        self.explore.last_walk_hop = None;
        self.explore.explore_path.clear();
        self.explore.climb_retry = None;
        // 必须朝「继续向上」的水平方向，不能保留底层贴边翻向的 -1（日志 mid 后 patrol(-1) 掉下）。
        self.explore.patrol_dir = Self::ascent_continue_dir(graph, top_node).unwrap_or(1.0);
        // 封所有从顶台下去的 ClimbDown（不限 from），打断绳顶上下空转。
        for e in &graph.edges {
            if e.from == top_node && e.kind == EdgeKind::ClimbDown {
                self.explore
                    .blocked_edges
                    .insert((e.from, e.kind, e.to), 600);
            }
            if e.to == top_node && e.kind == EdgeKind::ClimbUp {
                // 短暂不让底层立刻再爬同一根绳（若已掉下去）。
                self.explore
                    .blocked_edges
                    .entry((e.from, e.kind, e.to))
                    .or_insert(120);
            }
        }
        self.explore.visited.insert(from);
    }

    /// 中段落点继续向上的水平方向（有 StepUp/ClimbUp 则朝落点，否则 +1）。
    fn ascent_continue_dir(graph: &MapGraph, node_id: PlatformNodeId) -> Option<f32> {
        let from_mid = graph.get(node_id).map(|n| (n.x_min + n.x_max) * 0.5)?;
        let mut best: Option<(f32, f32)> = None;
        for e in &graph.edges {
            if e.from != node_id || !matches!(e.kind, EdgeKind::StepUp | EdgeKind::ClimbUp) {
                continue;
            }
            let to_y = graph.get(e.to).map(|n| n.y).unwrap_or(9999.0);
            let from_y = graph.get(node_id).map(|n| n.y).unwrap_or(0.0);
            if to_y >= from_y - 20.0 {
                continue;
            }
            let to_mid = graph
                .get(e.to)
                .map(|n| (n.x_min + n.x_max) * 0.5)
                .unwrap_or(e.target_x);
            let height = from_y - to_y;
            if best.map(|(h, _)| height > h).unwrap_or(true) {
                best = Some((height, to_mid - from_mid));
            }
        }
        best.map(|(_, dx)| if dx >= 0.0 { 1.0 } else { -1.0 })
    }

    fn is_reverse_walk(&self, edge: &GraphEdge) -> bool {
        edge.kind == EdgeKind::Walk
            && self
                .explore
                .last_walk_hop
                .is_some_and(|(a, b)| a == edge.to && b == edge.from)
    }

    /// 刚上台后又掉/跳回起点（预览 89→Fall→96→StepUp→89）。
    fn is_reverse_ascent(&self, edge: &GraphEdge) -> bool {
        let Some((from, to)) = self.explore.last_ascent_hop else {
            return false;
        };
        if edge.from != to {
            return false;
        }
        if edge.to == from {
            return matches!(
                edge.kind,
                EdgeKind::Fall | EdgeKind::StepUp | EdgeKind::ClimbDown | EdgeKind::Walk
            );
        }
        // Fall 在粘滞期内一律禁止，先沿台走完。
        if edge.kind == EdgeKind::Fall && self.explore.ascent_hold_ticks > 0 {
            return true;
        }
        false
    }

    /// 刚走过的同一条 Walk，禁止立刻再 commit（台缝上目标已在容差内会导致 goto_done 空转）。
    fn is_repeat_walk(&self, edge: &GraphEdge) -> bool {
        edge.kind == EdgeKind::Walk
            && self
                .explore
                .last_walk_hop
                .is_some_and(|(a, b)| a == edge.from && b == edge.to)
    }

    /// 同物理台多节点且两端已访问的 Walk（接缝对抽候选）。
    fn is_same_segment_visited_walk(&self, graph: &MapGraph, edge: &GraphEdge) -> bool {
        edge.kind == EdgeKind::Walk
            && self.explore.visited.contains(&edge.from)
            && self.explore.visited.contains(&edge.to)
            && Self::same_platform_segment(graph, edge.from, edge.to)
    }

    /// 上台后强制扫台：未到巡逻方向边缘前只 Patrol，避免最近未访问在左侧时立刻折返。
    fn plan_ascent_sweep(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
        world_x: f32,
    ) -> Option<SubGoal> {
        let sweep = self.explore.sweep_after_ascent?;
        if node_id != sweep {
            // 同层接缝挪到邻节点：粘住扫台，禁止清掉后立刻反向 goto。
            if Self::same_platform_segment(graph, sweep, node_id)
                || (self.explore.ascent_hold_ticks > 0
                    && self.edge_goes_forward(graph, sweep, node_id))
            {
                self.explore.visited.insert(sweep);
                self.explore.sweep_after_ascent = Some(node_id);
            } else {
                self.explore.visited.insert(sweep);
                self.explore.sweep_after_ascent = None;
                return None;
            }
        }
        let sweep = self.explore.sweep_after_ascent?;
        let node = graph.get(sweep)?;
        // 窄台也至少沿开荒方向走几步；完全跳过会立刻 leave Fall 回起点。
        let dir = if self.explore.patrol_dir >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let edge_margin = if node.width() < 56.0 { 8.0 } else { 20.0 };
        let at_end = if dir > 0.0 {
            world_x >= node.x_max - edge_margin
        } else {
            world_x <= node.x_min + edge_margin
        };
        if at_end {
            self.explore.visited.insert(sweep);
            self.explore.sweep_after_ascent = None;
            // 贴边：先向上 StepUp，再同向 Walk；禁止结束扫台后 Fall。
            if let Some(goal) = self.plan_continue_ascent(graph, sweep) {
                return Some(goal);
            }
            self.plan_forward_walk(graph, sweep)
        } else {
            Some(SubGoal::Patrol { dir })
        }
    }

    /// 中段爬绳落点继续向上（123→StepUp→117→133），这是右侧上楼主链。
    pub(crate) fn plan_continue_ascent(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
    ) -> Option<SubGoal> {
        let from_y = graph.get(node_id)?.y;
        let mut best: Option<(usize, f32)> = None;
        for (idx, e) in graph.edges.iter().enumerate() {
            if e.from != node_id {
                continue;
            }
            if !matches!(e.kind, EdgeKind::StepUp | EdgeKind::ClimbUp) {
                continue;
            }
            if self
                .explore
                .blocked_edges
                .contains_key(&(e.from, e.kind, e.to))
            {
                continue;
            }
            if self.is_reverse_ascent(e) {
                continue;
            }
            let to_y = graph.get(e.to).map(|n| n.y).unwrap_or(from_y);
            if to_y >= from_y - 20.0 {
                continue;
            }
            // 越高越好（y 更小），同高则偏巡逻方向。
            let height = from_y - to_y;
            let forward_bonus = if self.edge_goes_forward(graph, e.from, e.to) {
                30.0
            } else {
                0.0
            };
            let score = height + forward_bonus;
            if best.map(|(_, s)| score > s).unwrap_or(true) {
                best = Some((idx, score));
            }
        }
        best.map(|(idx, _)| self.commit_edge(graph, &graph.edges[idx]))
    }

    /// 沿 patrol_dir 的同层 Walk（可已访问），用于上台后继续穿过接缝。
    fn plan_forward_walk(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
    ) -> Option<SubGoal> {
        let mut best: Option<(usize, f32)> = None;
        for (idx, e) in graph.edges.iter().enumerate() {
            if e.from != node_id || e.kind != EdgeKind::Walk {
                continue;
            }
            if self
                .explore
                .blocked_edges
                .contains_key(&(e.from, e.kind, e.to))
            {
                continue;
            }
            if self.is_reverse_walk(e) || self.is_reverse_ascent(e) {
                continue;
            }
            if !self.edge_goes_forward(graph, e.from, e.to) {
                continue;
            }
            let to_mid = graph
                .get(e.to)
                .map(|n| (n.x_min + n.x_max) * 0.5)
                .unwrap_or(e.target_x);
            let score = if self.explore.patrol_dir >= 0.0 {
                to_mid
            } else {
                -to_mid
            };
            if best.map(|(_, s)| score > s).unwrap_or(true) {
                best = Some((idx, score));
            }
        }
        best.map(|(idx, _)| self.commit_edge(graph, &graph.edges[idx]))
    }

    fn edge_goes_forward(&self, graph: &MapGraph, from: PlatformNodeId, to: PlatformNodeId) -> bool {
        let from_mid = graph
            .get(from)
            .map(|n| (n.x_min + n.x_max) * 0.5)
            .unwrap_or(0.0);
        let to_mid = graph
            .get(to)
            .map(|n| (n.x_min + n.x_max) * 0.5)
            .unwrap_or(0.0);
        if self.explore.patrol_dir >= 0.0 {
            to_mid + 24.0 >= from_mid
        } else {
            to_mid - 24.0 <= from_mid
        }
    }

    /// 离开当前同台片段：优先 Climb/Walk/Fall；StepUp 最后且不对已访问/窄台空耗。
    pub(crate) fn plan_leave_segment(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
    ) -> Option<SubGoal> {
        // 保留 last_walk_hop，避免上台后 leave 立刻丢掉折返保护。
        let kind_order = [
            EdgeKind::ClimbUp,
            EdgeKind::Walk,
            EdgeKind::Fall,
            EdgeKind::ClimbDown,
            EdgeKind::StepUp,
        ];
        for prefer_unvisited in [true, false] {
            for forward_only in [self.explore.prefer_forward_explore, false] {
                if !forward_only && self.explore.prefer_forward_explore {
                    // 已在 forward_only=true 试过；false 再试全向。
                }
                let mut best_idx: Option<usize> = None;
                let mut best_key: Option<(u32, u32, u32)> = None;
                for (ki, kind) in kind_order.iter().enumerate() {
                    for (idx, e) in graph.edges.iter().enumerate() {
                        if e.from != node_id || e.kind != *kind {
                            continue;
                        }
                        // 中/上层绝对禁止主动 ClimbDown（右侧绳 57→123 后 yo-yo 主因）。
                        if e.kind == EdgeKind::ClimbDown {
                            if graph.get(e.from).is_some_and(|n| n.y < 1100.0) {
                                continue;
                            }
                        }
                        if self
                            .explore
                            .blocked_edges
                            .contains_key(&(e.from, e.kind, e.to))
                        {
                            continue;
                        }
                        if self.is_reverse_ascent(e) || self.is_reverse_walk(e) {
                            continue;
                        }
                        // 上层/粘滞期：禁止主动掉回底层（日志 130 次回底）。
                        if e.kind == EdgeKind::Fall {
                            if let (Some(from_n), Some(to_n)) =
                                (graph.get(e.from), graph.get(e.to))
                            {
                                let drop_to_bottom =
                                    to_n.y >= 1180.0 && from_n.y < 1180.0;
                                let big_drop = to_n.y > from_n.y + 80.0;
                                if drop_to_bottom
                                    || (big_drop
                                        && (self.explore.ascent_hold_ticks > 0
                                            || self.explore.prefer_forward_explore))
                                {
                                    continue;
                                }
                            }
                        }
                        if forward_only
                            && *kind != EdgeKind::ClimbUp
                            && !self.edge_goes_forward(graph, e.from, e.to)
                        {
                            continue;
                        }
                        // 粘滞向前/刚爬上：禁止下落与下爬，避免绳顶 yo-yo。
                        if (self.explore.prefer_forward_explore
                            || self.explore.ascent_hold_ticks > 0
                            || self.explore.sweep_after_ascent.is_some())
                            && matches!(*kind, EdgeKind::Fall | EdgeKind::ClimbDown)
                        {
                            continue;
                        }
                        if e.kind == EdgeKind::Walk
                            && Self::same_platform_segment(graph, e.from, e.to)
                            && self.explore.visited.contains(&e.to)
                            && !self.explore.prefer_forward_explore
                        {
                            continue;
                        }
                        // 已访问的台阶/窄台：离开片段时跳过，否则出生点 46↔15↔19 永远对抽。
                        if e.kind == EdgeKind::StepUp {
                            if self.explore.visited.contains(&e.to) {
                                continue;
                            }
                            if graph.get(e.to).is_some_and(|d| d.x_max - d.x_min < 48.0)
                                && !prefer_unvisited
                            {
                                continue;
                            }
                        }
                        let to_unvisited = !self.explore.visited.contains(&e.to);
                        if prefer_unvisited && !to_unvisited {
                            continue;
                        }
                        let to_x = graph
                            .get(e.to)
                            .map(|n| (n.x_min + n.x_max) * 0.5)
                            .unwrap_or(e.target_x);
                        let from_mid = graph
                            .get(node_id)
                            .map(|n| (n.x_min + n.x_max) * 0.5)
                            .unwrap_or(to_x);
                        let forward = if self.explore.patrol_dir >= 0.0 {
                            to_x + 24.0 >= from_mid
                        } else {
                            to_x - 24.0 <= from_mid
                        };
                        // 先同 kind，再同向（粘住上次方向），最后用稳定 idx；禁止 -to_x 写死偏右。
                        let dir_pen = if forward { 0u32 } else { 1u32 };
                        let key = (ki as u32, dir_pen, idx as u32);
                        if best_key.map(|bk| key < bk).unwrap_or(true) {
                            best_key = Some(key);
                            best_idx = Some(idx);
                        }
                    }
                }
                if let Some(idx) = best_idx {
                    self.explore.explore_path.clear();
                    return Some(self.commit_edge(graph, &graph.edges[idx]));
                }
                if !self.explore.prefer_forward_explore {
                    break;
                }
            }
        }
        None
    }

    fn walk_hop_satisfied(
        graph: &MapGraph,
        edge: &GraphEdge,
        world_x: f32,
        _goto_tol: f32,
    ) -> bool {
        if edge.kind != EdgeKind::Walk {
            return false;
        }
        // 蹇呴』宸插湪鐩殑鍙?x 鍐咃紱涓嶈兘浠呴潬 target_x 瀹瑰樊锛堥偦鍙拌竟缂樹細璇烦杩囷級銆?
        graph.get(edge.to).is_some_and(|d| {
            world_x >= d.x_min - 8.0 && world_x <= d.x_max + 8.0
        })
    }

    fn consume_satisfied_walk(
        &mut self,
        graph: &MapGraph,
        edge: &GraphEdge,
    ) {
        self.explore.visited.insert(edge.from);
        self.explore.visited.insert(edge.to);
        if edge.kind == EdgeKind::Walk {
            self.explore.last_walk_hop = Some((edge.from, edge.to));
        }
    }

    fn local_walk_neighbors_visited(&self, graph: &MapGraph, node_id: PlatformNodeId) -> bool {
        let mut saw_walk = false;
        for e in &graph.edges {
            if e.from != node_id || e.kind != EdgeKind::Walk {
                continue;
            }
            if self
                .explore
                .blocked_edges
                .contains_key(&(e.from, e.kind, e.to))
            {
                continue;
            }
            saw_walk = true;
            if !self.explore.visited.contains(&e.to) {
                return false;
            }
        }
        let _ = saw_walk;
        true
    }

    fn trim_explore_path_front(&mut self, graph: &MapGraph, node_id: PlatformNodeId) {
        while let Some(&idx) = self.explore.explore_path.first() {
            let e = &graph.edges[idx];
            if e.from == node_id {
                break;
            }
            if e.to == node_id {
                self.explore.explore_path.remove(0);
                continue;
            }
            self.explore.explore_path.clear();
            break;
        }
    }

    fn next_from_explore_path(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
        world_x: f32,
        goto_tol: f32,
    ) -> Option<SubGoal> {
        self.trim_explore_path_front(graph, node_id);
        let mut cur = node_id;
        while let Some(&idx) = self.explore.explore_path.first() {
            let edge = graph.edges[idx].clone();
            if edge.from != cur {
                self.explore.explore_path.clear();
                return None;
            }
            if self.is_reverse_walk(&edge) || self.is_repeat_walk(&edge) || self.is_reverse_ascent(&edge) {
                self.explore.explore_path.remove(0);
                continue;
            }
            // 同台已访问接缝：若能直接离开片段则离开，否则允许穿过（如 47→58→爬绳）。
            if self.is_same_segment_visited_walk(graph, &edge) {
                if let Some(goal) = self.plan_leave_segment(graph, node_id) {
                    return Some(goal);
                }
            }
            if Self::walk_hop_satisfied(graph, &edge, world_x, goto_tol) {
                // x 已落入邻台，但视觉节点尚未跟上时，不要虚推进，否则会直接 commit 下一段 StepUp。
                if node_id != edge.to && node_id != edge.from {
                    return Some(self.commit_edge(graph, &edge));
                }
                self.explore.explore_path.remove(0);
                self.consume_satisfied_walk(graph, &edge);
                cur = edge.to;
                continue;
            }
            // 换层边必须人已在 from；虚 walk 推进后视觉未到则清路径重规划。
            if edge.kind != EdgeKind::Walk && edge.from != node_id {
                self.explore.explore_path.clear();
                return None;
            }
            return Some(self.commit_edge(graph, &edge));
        }
        None
    }

    fn plan_path_to_unvisited(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
        world_x: f32,
        goto_tol: f32,
    ) -> Option<SubGoal> {
        if let Some(goal) = self.next_from_explore_path(graph, node_id, world_x, goto_tol) {
            return Some(goal);
        }

        // 同层 walk 邻居都已访问时，优先向上爬绳/台阶，避免 Fall 把人留在底层对抽。
        if self.local_walk_neighbors_visited(graph, node_id) {
            if let Some(goal) = self.plan_path_to_climb(graph, node_id, world_x, goto_tol) {
                return Some(goal);
            }
            if let Some(goal) = self.plan_leave_segment(graph, node_id) {
                return Some(goal);
            }
            if let Some(goal) = self.plan_cross_platform(graph, node_id, world_x, goto_tol) {
                return Some(goal);
            }
            // 死胡同（如 67 只有折返 Walk）：允许折返，否则会原地卡死。
            self.explore.last_walk_hop = None;
        }

        let mut path = if self.explore.prefer_forward_explore {
            let mut blocked = self.explore.blocked_edges.clone();
            for e in &graph.edges {
                if e.from == node_id
                    && e.kind != EdgeKind::ClimbUp
                    && !self.edge_goes_forward(graph, e.from, e.to)
                {
                    blocked.entry((e.from, e.kind, e.to)).or_insert(60);
                }
            }
            graph
                .path_to_nearest_unvisited(
                    node_id,
                    &self.explore.visited,
                    &blocked,
                    self.explore.patrol_dir,
                )
                .or_else(|| {
                    graph.path_to_nearest_unvisited(
                        node_id,
                        &self.explore.visited,
                        &self.explore.blocked_edges,
                        self.explore.patrol_dir,
                    )
                })?
        } else {
            graph.path_to_nearest_unvisited(
                node_id,
                &self.explore.visited,
                &self.explore.blocked_edges,
                self.explore.patrol_dir,
            )?
        };

        if let Some(&idx) = path.first() {
            let e = &graph.edges[idx];
            if self.is_reverse_walk(e) || self.is_repeat_walk(e) || self.is_reverse_ascent(e) {
                if let Some(goal) = self.plan_path_to_climb(graph, node_id, world_x, goto_tol) {
                    return Some(goal);
                }
                if let Some(goal) = self.plan_leave_segment(graph, node_id) {
                    return Some(goal);
                }
                if let Some(goal) = self.plan_cross_platform(graph, node_id, world_x, goto_tol) {
                    return Some(goal);
                }
                self.explore
                    .blocked_edges
                    .insert((e.from, e.kind, e.to), 180);
                path = graph.path_to_nearest_unvisited(
                    node_id,
                    &self.explore.visited,
                    &self.explore.blocked_edges,
                    self.explore.patrol_dir,
                )?;
            } else if self.is_same_segment_visited_walk(graph, e) {
                if let Some(goal) = self.plan_leave_segment(graph, node_id) {
                    return Some(goal);
                }
            }
        }

        self.explore.explore_path = path;
        self.next_from_explore_path(graph, node_id, world_x, goto_tol)
    }

    fn plan_path_to_climb(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
        world_x: f32,
        goto_tol: f32,
    ) -> Option<SubGoal> {
        // 上爬脱困允许折返刚走过的 Walk（否则 58→47 后 last_walk_hop 会挡住唯一出路）。
        self.explore.last_walk_hop = None;
        let path = graph.path_to_nearest_kind(
            node_id,
            EdgeKind::ClimbUp,
            &self.explore.blocked_edges,
        )?;
        self.explore.explore_path = path;
        self.next_from_explore_path(graph, node_id, world_x, goto_tol)
    }

    pub(crate) fn plan_path_to_climb_pub(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
        world_x: f32,
        goto_tol: f32,
    ) -> Option<SubGoal> {
        self.plan_path_to_climb(graph, node_id, world_x, goto_tol)
    }

    /// 只寻找明显高于当前台的未访问节点（上高层），找不到则 None。
    fn plan_path_to_unvisited_upward(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
        world_x: f32,
        goto_tol: f32,
    ) -> Option<SubGoal> {
        let from_y = graph.get(node_id)?.y;
        let path = graph.path_to_nearest_unvisited_with(
            node_id,
            &self.explore.visited,
            &self.explore.blocked_edges,
            |to_y| to_y < from_y - 40.0,
            self.explore.patrol_dir,
        )?;
        self.explore.last_walk_hop = None;
        self.explore.explore_path = path;
        self.next_from_explore_path(graph, node_id, world_x, goto_tol)
    }

    /// 底层未访问台（含向右同层），避免一出生就爬绳、右侧底层永远空着。
    fn plan_path_to_unvisited_bottom(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
        world_x: f32,
        goto_tol: f32,
    ) -> Option<SubGoal> {
        let path = graph.path_to_nearest_unvisited_with(
            node_id,
            &self.explore.visited,
            &self.explore.blocked_edges,
            |to_y| to_y >= 1180.0,
            self.explore.patrol_dir,
        )?;
        // 折返 Walk 时仍允许：沿当前巡逻方向开荒优先于卡死。
        self.explore.last_walk_hop = None;
        self.explore.explore_path = path;
        self.next_from_explore_path(graph, node_id, world_x, goto_tol)
    }

    fn plan_cross_platform(
        &mut self,
        graph: &MapGraph,
        node_id: PlatformNodeId,
        _world_x: f32,
        _goto_tol: f32,
    ) -> Option<SubGoal> {
        if let Some(edge) =
            graph.prefer_cross_platform_edge(
                node_id,
                &self.explore.blocked_edges,
                self.explore.patrol_dir,
            )
        {
            // 不要主动 Fall 回底层。
            if edge.kind == EdgeKind::Fall {
                if let (Some(a), Some(b)) = (graph.get(edge.from), graph.get(edge.to)) {
                    if b.y >= 1180.0 && a.y < 1180.0 {
                        return None;
                    }
                }
            }
            self.explore.explore_path.clear();
            return Some(self.commit_edge(graph, edge));
        }
        None
    }

    fn plan_patrol_route(&mut self, graph: &MapGraph, node_id: PlatformNodeId) -> SubGoal {
        let n = self.explore.patrol_route.len();
        if n == 0 {
            return SubGoal::Patrol {
                dir: self.explore.patrol_dir,
            };
        }

        // 浼樺厛锛氫粠褰撳墠鑺傜偣鍑哄彂銆侀€氬悜鏈闂殑杈癸紙璺宠繃鍒氳蛋瀹岀殑寰€杩?Walk锛夈€?
        for prefer_unvisited in [true, false] {
            for i in 0..n {
                let cur = (self.explore.patrol_cursor + i) % n;
                let idx = self.explore.patrol_route[cur];
                let edge = &graph.edges[idx];
                if edge.from != node_id {
                    continue;
                }
                if self.is_reverse_walk(edge) || self.is_repeat_walk(edge) || self.is_reverse_ascent(edge) {
                    continue;
                }
                if prefer_unvisited && self.explore.visited.contains(&edge.to) {
                    continue;
                }
                self.explore.patrol_cursor = cur;
                self.explore.explore_path.clear();
                return self.commit_edge(graph, edge);
            }
        }

        // 无可用巡逻边：硬走任意未封 Walk（允许折返），避免死胡同原地 patrol。
        self.explore.last_walk_hop = None;
        for e in &graph.edges {
            if e.from != node_id || e.kind != EdgeKind::Walk {
                continue;
            }
            if self
                .explore
                .blocked_edges
                .contains_key(&(e.from, e.kind, e.to))
            {
                continue;
            }
            self.explore.explore_path.clear();
            return self.commit_edge(graph, e);
        }

        SubGoal::Patrol {
            dir: self.explore.patrol_dir,
        }
    }

    pub fn set_subgoal(&mut self, goal: SubGoal) {
        if self.explore.active_subgoal != goal {
            self.explore.active_subgoal = goal;
            self.explore.subgoal_ticks = 0;
        }
    }

    pub fn on_subgoal_tick(&mut self) {
        self.explore.subgoal_ticks = self.explore.subgoal_ticks.saturating_add(1);
    }

    pub fn on_subgoal_done(&mut self, graph: &MapGraph, node_id: PlatformNodeId) {
        let completed = self.explore.pending_edge.take();
        let climb_done = matches!(
            completed.as_ref().map(|(_, k, _)| *k),
            Some(EdgeKind::ClimbUp | EdgeKind::ClimbDown)
        );
        let ascent_done = matches!(
            completed.as_ref().map(|(_, k, _)| *k),
            Some(EdgeKind::StepUp | EdgeKind::ClimbUp)
        );
        if let Some((from, kind, to)) = completed {
            self.explore.visited.insert(from);
            // 水平位移后粘住该方向：下次离台/寻路优先同向，避免 50/50 每步重掷。
            if let (Some(a), Some(b)) = (graph.get(from), graph.get(to)) {
                let from_mid = (a.x_min + a.x_max) * 0.5;
                let to_mid = (b.x_min + b.x_max) * 0.5;
                let dx = to_mid - from_mid;
                if dx.abs() > 28.0 {
                    self.explore.patrol_dir = dx.signum();
                }
            }
            // ClimbUp 落点：水平方向改看继续向上的 StepUp，不能沿用底层贴边翻向。
            if kind == EdgeKind::ClimbUp {
                if let Some(dir) = Self::ascent_continue_dir(graph, to) {
                    self.explore.patrol_dir = dir;
                }
            }
            if ascent_done {
                // 落点先扫台/向前开荒，不立刻标 visited，否则下一拍会找反向最近未访问而折返。
                self.explore.sweep_after_ascent = Some(to);
                self.explore.prefer_forward_explore = true;
                self.explore.last_ascent_hop = Some((from, to));
                self.explore.ascent_hold_ticks = 360;
                self.explore.last_walk_hop = None;
                self.explore.explore_path.clear();
                // 短时封回起点的 StepUp/Fall，打断 96↔89 空转。
                self.explore
                    .blocked_edges
                    .insert((to, EdgeKind::StepUp, from), 400);
                self.explore
                    .blocked_edges
                    .insert((to, EdgeKind::Fall, from), 400);
                for e in &graph.edges {
                    if e.from == to && e.kind == EdgeKind::Fall && e.to != from {
                        // 其它下落也暂封：先沿台走完再允许掉。
                        self.explore
                            .blocked_edges
                            .entry((e.from, e.kind, e.to))
                            .or_insert(180);
                    }
                }
            } else {
                self.explore.visited.insert(to);
                if kind == EdgeKind::Walk {
                    self.explore.last_walk_hop = Some((from, to));
                } else {
                    self.explore.last_walk_hop = None;
                }
            }
            // Fall/走下：短时封反向，打断「爬梯→左掉→再右掉回坑」空转。
            if kind == EdgeKind::Fall {
                self.explore
                    .blocked_edges
                    .insert((to, EdgeKind::Fall, from), 220);
            }

            if let Some(&idx) = self.explore.explore_path.first() {
                let e = &graph.edges[idx];
                if e.from == from && e.kind == kind && e.to == to {
                    self.explore.explore_path.remove(0);
                }
            }

            let n = self.explore.patrol_route.len();
            if n > 0 {
                let cur = self.explore.patrol_cursor % n;
                let idx = self.explore.patrol_route[cur];
                let e = &graph.edges[idx];
                if e.from == from && e.kind == kind && e.to == to {
                    self.explore.patrol_cursor = (cur + 1) % n;
                }
            }
        }
        // 上台扫台中：落点先不标 visited。
        if self.explore.sweep_after_ascent != Some(node_id) {
            self.explore.visited.insert(node_id);
        }
        self.explore.subgoal_ticks = 0;
        self.explore.subgoal_failures = 0;
        self.explore.active_subgoal = SubGoal::Idle;
        if climb_done {
            self.explore.climb_retry = None;
            // 刚爬完立刻原路折返会上下空转；封反向爬绳边更久。
            if let Some((from, kind, to)) = completed {
                let rev = match kind {
                    EdgeKind::ClimbUp => EdgeKind::ClimbDown,
                    EdgeKind::ClimbDown => EdgeKind::ClimbUp,
                    _ => kind,
                };
                let hold = if kind == EdgeKind::ClimbUp { 600 } else { 240 };
                self.explore
                    .blocked_edges
                    .insert((to, rev, from), hold);
                if kind == EdgeKind::ClimbUp {
                    // 顶台其它下爬边一并封。
                    for e in &graph.edges {
                        if e.from == to && e.kind == EdgeKind::ClimbDown && e.to != from {
                            self.explore
                                .blocked_edges
                                .entry((e.from, e.kind, e.to))
                                .or_insert(400);
                        }
                    }
                }
            }
        }
        let _ = graph;
    }

    pub fn on_subgoal_failed(
        &mut self,
        graph: &MapGraph,
        block_ticks: u32,
        block_edge: bool,
    ) {
        self.explore.subgoal_failures = self.explore.subgoal_failures.saturating_add(1);
        self.explore.subgoal_ticks = 0;
        self.explore.active_subgoal = SubGoal::Idle;
        self.explore.explore_path.clear();
        if block_edge {
            if let Some(key) = self.explore.pending_edge.take() {
                self.explore.blocked_edges.insert(key, block_ticks);
                let n = self.explore.patrol_route.len();
                if n > 0 {
                    self.explore.patrol_cursor = (self.explore.patrol_cursor + 1) % n;
                }
            }
        } else {
            self.explore.pending_edge = None;
        }
        let _ = graph;
    }

    pub fn farm_cleared(&self) -> bool {
        self.farm_cleared
    }

    pub fn on_node_changed(
        &mut self,
        graph: &MapGraph,
        prev: PlatformNodeId,
        next: PlatformNodeId,
    ) {
        if prev != next && prev != 0 {
            self.explore.visited.insert(prev);
            if !Self::same_platform_segment(graph, prev, next) {
                self.explore.subgoal_ticks = 0;
                self.explore.subgoal_failures = 0;
            }
        }
    }

    /// 鍚屼竴鐗╃悊骞冲彴琚媶鎴愬涓?nav 鑺傜偣鏃讹紙38鈫?6锛夛紝鍒囨崲涓嶅簲閲嶇疆 subgoal 璁℃椂銆?
    fn same_platform_segment(
        graph: &MapGraph,
        a: PlatformNodeId,
        b: PlatformNodeId,
    ) -> bool {
        let Some(na) = graph.get(a) else {
            return false;
        };
        let Some(nb) = graph.get(b) else {
            return false;
        };
        if (na.y - nb.y).abs() > 4.0 {
            return false;
        }
        let overlap = na.x_max.min(nb.x_max) - na.x_min.max(nb.x_min);
        overlap > -12.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::load_default_map;

    #[test]
    fn mid_band_leave_never_climb_down() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let mut nav = Navigator::new(123, &graph, 1);
        nav.explore.visited.insert(123);
        nav.explore.visited.insert(57);
        // 无 ascent 粘滞时也不允许从 123 ClimbDown。
        let leave = nav.plan_leave_segment(&graph, 123);
        if let Some((_, kind, to)) = nav.explore.pending_edge {
            assert_ne!(
                kind,
                EdgeKind::ClimbDown,
                "123 must not leave via ClimbDown ->{to}"
            );
        }
        if let Some(g) = leave {
            assert!(
                !matches!(g, SubGoal::ClimbDown { .. }),
                "got {}",
                g.label()
            );
        }
    }

    #[test]
    fn after_mid_climb_prefers_step_up_not_fall() {
        // ClimbUp 57→123 只到 y=985；必须接着 StepUp→117，不能 Fall/ClimbDown 回底。
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let has = graph
            .edges
            .iter()
            .any(|e| e.from == 57 && e.to == 123 && e.kind == EdgeKind::ClimbUp);
        if !has {
            return;
        }
        let mut nav = Navigator::new(57, &graph, 2);
        nav.explore.visited.insert(57);
        nav.explore.pending_edge = Some((57, EdgeKind::ClimbUp, 123));
        nav.on_subgoal_done(&graph, 123);
        assert!(nav.explore.ascent_hold_ticks > 0);

        let mid = graph
            .get(123)
            .map(|n| (n.x_min + n.x_max) * 0.5)
            .unwrap_or(1497.0);
        let g = nav.plan(&graph, 123, mid, true, 16.0);
        if let Some((from, kind, to)) = nav.explore.pending_edge {
            assert!(
                !(kind == EdgeKind::ClimbDown || kind == EdgeKind::Fall),
                "must not drop after mid climb, got {:?} {}->{}",
                kind,
                from,
                to
            );
            let to_y = graph.get(to).map(|n| n.y).unwrap_or(9999.0);
            assert!(
                to_y < 985.0 || kind == EdgeKind::Walk || kind == EdgeKind::StepUp,
                "should continue upward, got {:?} ->{} y={:.0} goal={}",
                kind,
                to,
                to_y,
                g.label()
            );
        } else {
            assert!(
                matches!(g, SubGoal::Patrol { .. } | SubGoal::StepUp { .. }),
                "expected ascent continue, got {}",
                g.label()
            );
        }
        // 明确：continue_ascent 应能选出 StepUp→117
        let mut nav2 = Navigator::new(123, &graph, 2);
        nav2.explore.ascent_hold_ticks = 100;
        nav2.explore.prefer_forward_explore = true;
        let up = nav2.plan_continue_ascent(&graph, 123);
        assert!(
            up.is_some(),
            "123 must have upward StepUp while ascending"
        );
        if let Some((_, kind, to)) = nav2.explore.pending_edge {
            assert_eq!(kind, EdgeKind::StepUp);
            assert_eq!(to, 117);
        }
    }

    #[test]
    fn after_climb_up_does_not_immediately_climb_down() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        // 出生点上方绳：20 ↔ ClimbUp ↔ 105（日志 yo-yo 对）。
        let edge = graph.edges.iter().find(|e| {
            e.kind == EdgeKind::ClimbUp
                && ((e.from == 20 && e.to == 105) || (e.from == 105) || (e.to == 105))
        });
        let Some(edge) = edge else {
            return;
        };
        let (from, to) = (edge.from, edge.to);
        let mut nav = Navigator::new(from, &graph, 2);
        nav.explore.visited.insert(from);
        nav.explore.pending_edge = Some((from, EdgeKind::ClimbUp, to));
        nav.on_subgoal_done(&graph, to);
        assert!(nav.explore.prefer_forward_explore);
        assert!(nav.explore.ascent_hold_ticks > 0);
        assert_eq!(nav.explore.sweep_after_ascent, Some(to));

        let mid = graph
            .get(to)
            .map(|n| (n.x_min + n.x_max) * 0.5)
            .unwrap_or(500.0);
        for _ in 0..8 {
            let g = nav.plan(&graph, to, mid, true, 16.0);
            if let Some((f, kind, t)) = nav.explore.pending_edge {
                assert!(
                    !(f == to && kind == EdgeKind::ClimbDown),
                    "must not climb back down after ascent, got {:?} {}->{} goal={}",
                    kind,
                    f,
                    t,
                    g.label()
                );
            }
            assert!(
                !matches!(g, SubGoal::ClimbDown { .. }),
                "must not plan ClimbDown right after climb-up, got {}",
                g.label()
            );
        }
    }

    #[test]
    fn note_arrived_climb_top_blocks_climb_down() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let top = graph
            .edges
            .iter()
            .find(|e| e.kind == EdgeKind::ClimbUp && e.to == 105)
            .map(|e| e.to)
            .unwrap_or(105);
        let mut nav = Navigator::new(20, &graph, 1);
        nav.note_arrived_climb_top(&graph, top, Some(20));
        assert!(nav.explore.ascent_hold_ticks > 0);
        assert!(nav
            .explore
            .blocked_edges
            .keys()
            .any(|(f, k, _)| *f == top && *k == EdgeKind::ClimbDown));
        let leave = nav.plan_leave_segment(&graph, top);
        if let Some(g) = leave {
            assert!(
                !matches!(g, SubGoal::ClimbDown { .. }),
                "leave_segment must not ClimbDown during ascent hold, got {}",
                g.label()
            );
        }
    }

    #[test]
    fn after_step_up_prefers_forward_not_immediate_left_goto() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let Some(n24) = graph.get(24) else {
            return;
        };
        let has_edge = graph
            .edges
            .iter()
            .any(|e| e.from == 42 && e.to == 24 && e.kind == EdgeKind::StepUp);
        if !has_edge {
            return;
        }

        let mut nav = Navigator::new(42, &graph, 1);
        nav.explore.patrol_dir = 1.0;
        nav.explore.visited.insert(42);
        nav.explore.pending_edge = Some((42, EdgeKind::StepUp, 24));
        nav.on_subgoal_done(&graph, 24);

        assert!(nav.explore.prefer_forward_explore);
        let mid = (n24.x_min + n24.x_max) * 0.5;
        let g = nav.plan(&graph, 24, mid, false, 16.0);
        // 不得一上台就 goto 到落点左侧（预览里的 goto(611)）。
        if let SubGoal::GoTo { x } = g {
            assert!(
                x + 8.0 >= n24.x_min,
                "must not plan left of landing platform, got goto({x:.0}) plat={:.0}..{:.0}",
                n24.x_min,
                n24.x_max
            );
        }
        assert!(
            !matches!(g, SubGoal::GoTo { x } if x < n24.x_min - 8.0),
            "got {}",
            g.label()
        );
    }

    #[test]
    fn explore_path_cache_survives_walk_hop_without_replan_flip() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        if !(graph.nodes.contains_key(&60) && graph.nodes.contains_key(&62)) {
            return;
        }

        let mut nav = Navigator::new(60, &graph, 99);
        nav.explore.visited.insert(60);
        nav.explore.visited.insert(62);

        let blocked = std::collections::HashMap::new();
        let Some(full_path) =
            graph.path_to_nearest_unvisited(60, &nav.explore.visited, &blocked, 1.0)
        else {
            return;
        };

        nav.explore.explore_path = full_path;
        let g0 = nav.plan(&graph, 60, 980.0, false, 16.0).label();
        assert!(
            g0.starts_with("goto"),
            "first hop from 60 should be goto, got {g0}"
        );
        let first_edge = nav.explore.pending_edge;

        nav.on_subgoal_done(&graph, first_edge.map(|(_, _, t)| t).unwrap_or(62));

        let _g1 = nav.plan(&graph, 62, 965.0, false, 16.0).label();
        if let (Some((from, _, to)), Some((f0, _, t0))) = (nav.explore.pending_edge, first_edge) {
            assert!(
                !(from == 62 && to == 60 && f0 == 60 && t0 == 62),
                "should not ping-pong 60<->62 on replan"
            );
        }
    }

    #[test]
    fn climb_retry_repositions_then_recommits() {
        let map = load_default_map().expect("map");
        let g = MapGraph::build(&map);
        // 57-climb_up->123 rope≈1477；首次失败从 offset[-28] 起探。
        let mut nav = Navigator::new(41, &g, 0);
        assert!(nav.begin_climb_retry(57, 123, EdgeKind::ClimbUp, 1477.0, 1485.0, 7));
        let g0 = nav.plan(&g, 57, 1485.0, true, 16.0);
        assert!(
            g0.label().starts_with("goto"),
            "should walk to probe stand, got {}",
            g0.label()
        );
        nav.explore.active_subgoal = SubGoal::Idle;
        let g1 = nav.plan(&g, 57, 1449.0, true, 16.0);
        assert!(
            matches!(g1, SubGoal::ClimbUp { rope_x } if (rope_x - 1449.0).abs() < 1.0),
            "at probe stand should climb with probe aim, got {}",
            g1.label()
        );
    }

    #[test]
    fn node_56_blocked_walk_right_plans_climb_left() {
        // 复现：56→58 右走顶墙 stalled 后应封边，改走左侧去 ClimbUp。
        let map = load_default_map().expect("map");
        let g = MapGraph::build(&map);
        let mut nav = Navigator::new(56, &g, 0);
        for id in g.nodes.keys() {
            nav.explore.visited.insert(*id);
        }
        nav.explore.visited.remove(&123);
        nav.explore
            .blocked_edges
            .insert((56, EdgeKind::Walk, 58), 600);
        nav.explore.last_walk_hop = None;
        nav.mark_farm_cleared();
        nav.explore.active_subgoal = SubGoal::Idle;
        let goal = nav.plan(&g, 56, 1716.0, true, 24.0);
        assert!(
            !matches!(goal, SubGoal::Patrol { .. }),
            "should leave 56 after right walk blocked, got {}",
            goal.label()
        );
        let pending = nav.explore.pending_edge;
        assert!(
            pending.is_some_and(|(f, k, t)| f == 56 && k == EdgeKind::Walk && t != 58),
            "must not retry Walk→58, pending={pending:?} goal={}",
            goal.label()
        );
    }

    #[test]
    fn node_47_reverse_walk_still_plans_climb_escape() {
        // 复现：58→47 后 last_walk_hop 禁止折返，唯一出路 Walk→58→…→ClimbUp 被挡，只能 patrol。
        let map = load_default_map().expect("map");
        let g = MapGraph::build(&map);
        assert!(
            g.edges
                .iter()
                .any(|e| e.from == 47 && e.kind == EdgeKind::Walk && e.to == 58),
            "expected Walk 47→58"
        );
        let mut nav = Navigator::new(47, &g, 0);
        for id in g.nodes.keys() {
            nav.explore.visited.insert(*id);
        }
        nav.explore.visited.remove(&123);
        nav.explore.last_walk_hop = Some((58, 47));
        nav.explore.subgoal_failures = 5;
        nav.mark_farm_cleared();
        nav.explore.active_subgoal = SubGoal::Idle;
        let idle = nav.plan(&g, 47, 1920.0, true, 24.0);
        assert!(
            idle.label().starts_with("goto") || idle.label().starts_with("climb"),
            "idle must leave via walk/climb, got {}",
            idle.label()
        );
        assert_eq!(
            nav.explore.pending_edge.map(|(f, k, t)| (f, k, t)),
            Some((47, EdgeKind::Walk, 58)),
            "first hop should be Walk 47→58 toward climb"
        );

        nav.explore.last_walk_hop = Some((58, 47));
        nav.explore.pending_edge = None;
        nav.explore.explore_path.clear();
        nav.explore.active_subgoal = SubGoal::Patrol { dir: -1.0 };
        nav.explore.subgoal_ticks = 10;
        let hold = nav.plan(&g, 47, 1920.0, true, 24.0);
        assert!(
            !matches!(hold, SubGoal::Patrol { .. }),
            "stuck patrol hold must break to climb path, got {}",
            hold.label()
        );
    }

    #[test]
    fn patrol_route_is_deterministic_for_seed() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let (sx, sy) = map.default_spawn();
        let start = graph.node_at(&map, sx, sy).expect("spawn");
        let a = graph.build_patrol_route(start, 7);
        let b = graph.build_patrol_route(start, 7);
        assert_eq!(a, b);
        let c = graph.build_patrol_route(start, 8);
        assert_ne!(a, c);
    }

    #[test]
    fn node_15_with_stepup_blocked_still_plans_exit() {
        let map = load_default_map().expect("map");
        let g = MapGraph::build(&map);
        let mut nav = Navigator::new(41, &g, 0);
        for id in [15u32, 12, 16, 19, 50, 46, 38, 40, 41, 42, 39, 21] {
            nav.explore.visited.insert(id);
        }
        nav.explore
            .blocked_edges
            .insert((15, EdgeKind::StepUp, 19), 600);
        nav.explore
            .blocked_edges
            .insert((38, EdgeKind::StepUp, 16), 600);
        nav.explore
            .blocked_edges
            .insert((40, EdgeKind::StepUp, 16), 600);
        nav.mark_farm_cleared();
        nav.explore.active_subgoal = SubGoal::Idle;

        let goal = nav.plan(&g, 15, 230.0, true, 24.0);
        eprintln!("plan from 15 => {}", goal.label());
        assert!(
            !matches!(goal, SubGoal::Patrol { .. }),
            "should leave via walk/step_up, got {}",
            goal.label()
        );
    }

    #[test]
    fn node_15_dead_end_when_all_exits_blocked_returns_patrol_today() {
        let map = load_default_map().expect("map");
        let g = MapGraph::build(&map);
        let mut nav = Navigator::new(41, &g, 0);
        for id in [15u32, 12, 16, 19, 50, 46, 38, 40, 41, 42, 39, 21] {
            nav.explore.visited.insert(id);
        }
        nav.explore
            .blocked_edges
            .insert((15, EdgeKind::StepUp, 19), 600);
        nav.explore
            .blocked_edges
            .insert((15, EdgeKind::Walk, 12), 120);
        nav.mark_farm_cleared();
        nav.explore.active_subgoal = SubGoal::Idle;
        let goal = nav.plan(&g, 15, 230.0, true, 24.0);
        eprintln!("dead-end plan => {}", goal.label());
        assert!(
            !matches!(goal, SubGoal::Patrol { .. }),
            "should clear local blocks and exit, got {}",
            goal.label()
        );
    }

    #[test]
    fn node_98_99_segment_prefers_leave_not_churn() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        if !(graph.nodes.contains_key(&98) && graph.nodes.contains_key(&99)) {
            return;
        }
        let mut nav = Navigator::new(98, &graph, 1);
        for id in [98u32, 99, 100, 102, 105] {
            if graph.nodes.contains_key(&id) {
                nav.explore.visited.insert(id);
            }
        }
        nav.mark_farm_cleared();
        nav.explore.active_subgoal = SubGoal::Idle;
        nav.explore.last_walk_hop = Some((98, 99));
        // 接缝处 x≈707
        let goal = nav.plan(&graph, 98, 705.0, true, 16.0);
        if let Some((from, kind, to)) = nav.explore.pending_edge {
            assert!(
                !(kind == EdgeKind::Walk
                    && ((from == 98 && to == 99) || (from == 99 && to == 98))),
                "must leave segment, not churn 98↔99, got {} pending={from}-{kind:?}->{to}",
                goal.label()
            );
        }
        assert!(
            !matches!(goal, SubGoal::Idle),
            "should plan an exit from 98, got {}",
            goal.label()
        );
    }

    #[test]
    fn node_67_dead_end_walks_back_not_idle() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        if !(graph.nodes.contains_key(&67) && graph.nodes.contains_key(&66)) {
            return;
        }
        let mut nav = Navigator::new(49, &graph, 1);
        for id in [49u32, 66, 67] {
            nav.explore.visited.insert(id);
        }
        nav.explore.last_walk_hop = Some((66, 67));
        nav.mark_farm_cleared();
        nav.explore.active_subgoal = SubGoal::Idle;
        // 67 右缘：与 66←67 的 target 很近，旧 bug 会在无 pending_target 时假完成。
        let x = graph.get(67).map(|n| n.x_max - 4.0).unwrap_or(1370.0);
        let goal = nav.plan(&graph, 67, x, true, 16.0);
        assert!(
            !matches!(goal, SubGoal::Idle),
            "dead-end 67 must plan exit, got {}",
            goal.label()
        );
        let pending = nav.explore.pending_edge;
        assert!(
            pending.is_some_and(|(f, _k, t)| f == 67 && t == 66)
                || matches!(
                    goal,
                    SubGoal::ClimbUp { .. }
                        | SubGoal::GoTo { .. }
                        | SubGoal::WalkOff { .. }
                ),
            "expected leave 67 (walk/climb/walk-off), got {} pending={pending:?}",
            goal.label()
        );
    }

    #[test]
    fn satisfied_or_repeat_walk_does_not_spin_goto_done() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        // 98鈫?02 鍙扮紳锛歺=617 宸插湪 102 鍙崇紭锛?8鈫?02 鐨?target鈮?11銆?
        if !(graph.nodes.contains_key(&98) && graph.nodes.contains_key(&102)) {
            return;
        }
        let mut nav = Navigator::new(98, &graph, 1);
        nav.explore.visited.insert(98);
        nav.mark_farm_cleared();

        let g0 = nav.plan(&graph, 98, 617.0, true, 16.0);
        // 涓嶅簲鍙嶅 commit 绔嬪埢瀹屾垚鐨?98鈫?02锛涘簲璺宠繃鎴栨崲杈广€?
        if let Some((from, kind, to)) = nav.explore.pending_edge {
            assert!(
                !(from == 98 && to == 102 && kind == EdgeKind::Walk),
                "must not commit already-satisfied 98-walk->102 at x=617, goal={}",
                g0.label()
            );
        }
        // 鍐?plan 涓€娆′篃涓嶅簲绌鸿浆鍚屼竴 hop
        nav.explore.active_subgoal = SubGoal::Idle;
        nav.explore.pending_edge = None;
        nav.explore.last_walk_hop = Some((98, 102));
        nav.explore.visited.insert(102);
        let _g1 = nav.plan(&graph, 98, 617.0, true, 16.0);
        if let Some((from, kind, to)) = nav.explore.pending_edge {
            assert!(
                !(from == 98 && to == 102 && kind == EdgeKind::Walk),
                "repeat walk 98->102 must be blocked"
            );
        }
    }

    #[test]
    fn walk_onto_climb_node_aims_rope_x() {
        let map = load_default_map().expect("map");
        let g = MapGraph::build(&map);
        let mut nav = Navigator::new(48, &g, 1);
        nav.explore.visited.insert(48);
        nav.mark_farm_cleared();
        nav.explore.active_subgoal = SubGoal::Idle;
        let goal = nav.plan(&g, 48, 1550.0, true, 24.0);
        eprintln!("from 48 => {} pending={:?}", goal.label(), nav.explore.pending_edge);
        // 48 可直接 Climb? 无。应走/爬向 57 绳。
        match goal {
            SubGoal::ClimbUp { rope_x } => {
                assert!((rope_x - 1477.0).abs() < 1.0);
            }
            SubGoal::GoTo { x } => {
                let to = nav.explore.pending_edge.map(|(_, _, t)| t);
                if to == Some(57) {
                    assert!(
                        (x - 1477.0).abs() < 30.0,
                        "48->57 must aim rope, got {x}"
                    );
                }
            }
            other => panic!("expected climb approach, got {}", other.label()),
        }
    }

    #[test]
    fn bottom_right_edge_47_prefers_climb_over_wall_escape() {
        let map = load_default_map().expect("map");
        let g = MapGraph::build(&map);
        let Some(n) = g.get(47) else {
            return;
        };
        let mut nav = Navigator::new(47, &g, 1);
        for (id, node) in &g.nodes {
            if node.y >= 1180.0 {
                nav.explore.visited.insert(*id);
            }
        }
        nav.mark_farm_cleared();
        // 模拟 goto_stalled 后错误右逃并顶墙。
        nav.begin_escape(1.0, 24);
        nav.explore.escape_dir = 1.0;
        let at_right = n.x_max - 8.0;
        let goal = nav.plan(&g, 47, at_right, true, 24.0);
        eprintln!(
            "right-edge 47 => {} esc={} pending={:?}",
            goal.label(),
            nav.explore.escape_ticks,
            nav.explore.pending_edge
        );
        assert_eq!(nav.explore.escape_ticks, 0, "must clear wall escape");
        assert!(
            matches!(
                goal,
                SubGoal::ClimbUp { .. } | SubGoal::GoTo { .. } | SubGoal::StepUp { .. }
            ),
            "rightmost bottom must seek climb, got {}",
            goal.label()
        );
        assert!(
            !matches!(goal, SubGoal::Patrol { dir } if dir > 0.0),
            "must not keep patrolling into right wall, got {}",
            goal.label()
        );
    }

    #[test]
    fn bottom_node_47_plans_climb_not_patrol() {
        let map = load_default_map().expect("map");
        let g = MapGraph::build(&map);
        if g.get(47).is_none() || g.get(57).is_none() {
            return;
        }
        let mut nav = Navigator::new(47, &g, 1);
        // 底层台都逛过 → 才应去爬绳上楼
        for (id, n) in &g.nodes {
            if n.y >= 1180.0 {
                nav.explore.visited.insert(*id);
            }
        }
        nav.mark_farm_cleared();
        nav.explore.active_subgoal = SubGoal::Idle;
        let n = g.get(47).unwrap();
        let goal = nav.plan(&g, 47, n.x_min + 20.0, true, 24.0);
        eprintln!("from 47 => {} pending={:?}", goal.label(), nav.explore.pending_edge);
        assert!(
            matches!(
                goal,
                SubGoal::ClimbUp { .. } | SubGoal::GoTo { .. }
            ),
            "bottom 47 must seek climb/approach rope, got {}",
            goal.label()
        );
        assert!(
            !matches!(goal, SubGoal::Patrol { .. } | SubGoal::WalkOff { .. }),
            "must not patrol/fall on bottom when climb exists, got {}",
            goal.label()
        );
        if let (SubGoal::GoTo { x }, Some((_, _, to))) = (goal, nav.explore.pending_edge) {
            if to == 57 {
                assert!(
                    (x - 1477.0).abs() < 30.0,
                    "walk onto climb node 57 must aim rope 1477, got {x}"
                );
            }
        }
    }

    #[test]
    fn bottom_cluster_prefers_climb_when_walks_visited() {
        let map = load_default_map().expect("map");
        let g = MapGraph::build(&map);
        // 日志末段常卡在 47/52/56/58 底层簇。
        let cluster = [47u32, 52, 56, 58];
        if !cluster.iter().all(|id| g.nodes.contains_key(id)) {
            return;
        }
        let mut nav = Navigator::new(47, &g, 1);
        for (id, n) in &g.nodes {
            if n.y >= 1180.0 {
                nav.explore.visited.insert(*id);
            }
        }
        // 把同层 walk 邻居也标已访问，迫使换层。
        for e in &g.edges {
            if cluster.contains(&e.from) && e.kind == EdgeKind::Walk {
                nav.explore.visited.insert(e.to);
            }
        }
        nav.mark_farm_cleared();
        nav.explore.active_subgoal = SubGoal::Idle;
        let n = g.get(47).unwrap();
        let goal = nav.plan(&g, 47, (n.x_min + n.x_max) * 0.5, true, 24.0);
        eprintln!("bottom cluster plan => {}", goal.label());
        assert!(
            matches!(
                goal,
                SubGoal::ClimbUp { .. }
                    | SubGoal::GoTo { .. }
                    | SubGoal::StepUp { .. }
            ),
            "should seek climb/approach, not idle/patrol/fall, got {}",
            goal.label()
        );
        assert!(
            !matches!(goal, SubGoal::WalkOff { .. } | SubGoal::Patrol { .. }),
            "must not fall/patrol when climb available, got {}",
            goal.label()
        );
    }

    #[test]
    fn reverse_walk_hop_prefers_cross_platform() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let (sx, sy) = map.default_spawn();
        let start = graph.node_at(&map, sx, sy).expect("spawn");
        let mut nav = Navigator::new(start, &graph, 42);

        let walk_to = graph
            .edges
            .iter()
            .find(|e| e.from == start && e.kind == EdgeKind::Walk)
            .map(|e| e.to);
        let Some(nbr) = walk_to else {
            return;
        };

        nav.explore.visited.insert(start);
        nav.explore.visited.insert(nbr);
        for e in &graph.edges {
            if e.from == nbr && e.kind == EdgeKind::Walk {
                nav.explore.visited.insert(e.to);
            }
        }
        nav.explore.last_walk_hop = Some((start, nbr));

        let goal = nav.plan(
            &graph,
            nbr,
            graph.get(nbr).map(|n| n.x_min + 8.0).unwrap_or(0.0),
            true,
            16.0,
        );
        if let Some((from, kind, to)) = nav.explore.pending_edge {
            assert!(
                !(kind == EdgeKind::Walk && from == nbr && to == start),
                "must not reverse walk {nbr}->{start}, got goal={}",
                goal.label()
            );
        }
    }

    #[test]
    fn leave_segment_follows_patrol_dir_not_rightmost() {
        // 复现：walk_off(L) 落后立刻因 -to_x 选右落回坑。
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let node = 67u32;
        let falls: Vec<_> = graph
            .edges
            .iter()
            .filter(|e| e.from == node && e.kind == EdgeKind::Fall)
            .collect();
        if falls.len() < 2 {
            return;
        }
        let leftish = falls
            .iter()
            .min_by(|a, b| {
                graph
                    .get(a.to)
                    .map(|n| (n.x_min + n.x_max) * 0.5)
                    .unwrap_or(0.0)
                    .partial_cmp(
                        &graph
                            .get(b.to)
                            .map(|n| (n.x_min + n.x_max) * 0.5)
                            .unwrap_or(0.0),
                    )
                    .unwrap()
            })
            .map(|e| e.to);
        let rightish = falls
            .iter()
            .max_by(|a, b| {
                graph
                    .get(a.to)
                    .map(|n| (n.x_min + n.x_max) * 0.5)
                    .unwrap_or(0.0)
                    .partial_cmp(
                        &graph
                            .get(b.to)
                            .map(|n| (n.x_min + n.x_max) * 0.5)
                            .unwrap_or(0.0),
                    )
                    .unwrap()
            })
            .map(|e| e.to);
        let (Some(left_to), Some(right_to)) = (leftish, rightish) else {
            return;
        };
        if left_to == right_to {
            return;
        }

        let mut nav = Navigator::new(node, &graph, 1);
        nav.explore.visited.insert(node);
        nav.explore.patrol_dir = -1.0;
        nav.explore.prefer_forward_explore = false;
        let goal = nav
            .plan_leave_segment(&graph, node)
            .expect("leave edge");
        let pending = nav.explore.pending_edge.expect("pending");
        assert_eq!(pending.1, EdgeKind::Fall);
        assert_eq!(
            pending.2, left_to,
            "patrol_dir=-1 should pick leftish fall {left_to}, not right {right_to}; goal={}",
            goal.label()
        );
    }

    #[test]
    fn after_step_up_does_not_fall_back_to_origin() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        // 96 → StepUp → 89 是日志里的空转对。
        let has = graph
            .edges
            .iter()
            .any(|e| e.from == 96 && e.to == 89 && e.kind == EdgeKind::StepUp);
        if !has {
            return;
        }
        let mut nav = Navigator::new(96, &graph, 2);
        nav.explore.visited.insert(96);
        nav.explore.pending_edge = Some((96, EdgeKind::StepUp, 89));
        nav.on_subgoal_done(&graph, 89);
        assert!(nav.explore.prefer_forward_explore);
        assert_eq!(nav.explore.last_ascent_hop, Some((96, 89)));
        assert!(nav.explore.ascent_hold_ticks > 0);

        // 不应立刻规划回 96 的 Fall/StepUp。
        for _ in 0..5 {
            let mid = graph.get(89).map(|n| (n.x_min + n.x_max) * 0.5).unwrap_or(1337.0);
            let g = nav.plan(&graph, 89, mid, true, 16.0);
            if let Some((from, kind, to)) = nav.explore.pending_edge {
                assert!(
                    !(from == 89 && to == 96),
                    "must not reverse to origin 96, got {:?} {} {}",
                    kind,
                    from,
                    to
                );
                assert!(
                    !matches!(kind, EdgeKind::Fall) || to != 96,
                    "must not fall to 96, goal={}",
                    g.label()
                );
            }
            if matches!(g, SubGoal::WalkOff { .. }) {
                panic!("must not walk_off back after step_up, got {}", g.label());
            }
            // 推进：若已挂 Walk 则视为成功离开折返陷阱。
            if matches!(g, SubGoal::GoTo { .. }) || nav.explore.pending_edge.is_some_and(|(_, k, _)| k == EdgeKind::Walk) {
                return;
            }
            nav.set_subgoal(g);
            nav.on_subgoal_tick();
        }
    }

    #[test]
    fn bottom_prefers_climb_after_modest_explore() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let (sx, sy) = map.default_spawn();
        let start = graph.node_at(&map, sx, sy).expect("spawn");
        assert!(
            Navigator::is_bottom_band(&graph, start) || sy >= 1180.0,
            "spawn should be bottom-ish"
        );

        let mut nav = Navigator::new(start, &graph, 42);
        // 模拟已逛过若干底台。
        for id in graph.nodes.keys().copied() {
            if Navigator::is_bottom_band(&graph, id) {
                nav.explore.visited.insert(id);
                if nav.bottom_visited_count(&graph) >= 5 {
                    break;
                }
            }
        }
        nav.explore.visited.insert(start);
        let x = graph
            .get(start)
            .map(|n| (n.x_min + n.x_max) * 0.5)
            .unwrap_or(sx);
        let goal = nav.plan(&graph, start, x, false, 16.0);
        assert!(
            matches!(
                goal,
                SubGoal::ClimbUp { .. } | SubGoal::GoTo { .. } | SubGoal::StepUp { .. }
            ),
            "after exploring bottom should seek climb/approach up, got {}",
            goal.label()
        );
        // 不应还在纯底台 patrol 开荒。
        assert!(
            !matches!(goal, SubGoal::Patrol { .. }),
            "must not keep patrolling bottom, got {}",
            goal.label()
        );
    }

    #[test]
    fn seed_picks_patrol_dir_fifty_fifty() {
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let start = graph
            .node_at(&map, map.default_spawn().0, map.default_spawn().1)
            .unwrap_or(1);
        let a = Navigator::new(start, &graph, 40).explore.patrol_dir;
        let b = Navigator::new(start, &graph, 41).explore.patrol_dir;
        assert!(
            (a > 0.0) != (b > 0.0),
            "even/odd seed should flip initial dir, got {a} and {b}"
        );
    }
}

