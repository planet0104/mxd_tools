use super::super::input::InputFrame;
use super::super::map::ClimbHint;
use super::super::observation::{
    obs_climb_hint, obs_floor_ahead, obs_floor_ahead_connected, obs_floor_drop_ahead,
    obs_has_floor_signal, obs_same_level_gap_ahead, obs_step_up_dx,
};
use super::super::types::{WINDOW_H, WINDOW_W};
use super::map_graph::MapGraph;
use super::types::{EdgeKind, ExecutorResult, NavBotConfig, Side, SubGoal};

/// 对齐封存 rule_bot：距台阶超过该值且前方可走才走近，否则起跳。
const STEP_UP_JUMP_DX: f32 = 48.0;
/// 对齐封存 rule_bot：水平对准阈值。
const STEP_UP_ALIGN_DX: f32 = 14.0;
/// 抓绳横向容差：未对准只走不跳。
const CLIMB_GRAB_TOL_PX: f32 = 8.0;

pub struct NavCtx<'a> {
    pub obs: &'a [f32],
    pub facing: f32,
    pub on_ground: bool,
    pub climbing: bool,
    pub world_x: f32,
    pub world_y: f32,
    pub node_id: u32,
    pub walk_right_ok: Option<bool>,
    pub walk_left_ok: Option<bool>,
    pub drop_ahead_right: Option<bool>,
    pub drop_ahead_left: Option<bool>,
    pub climb: Option<ClimbHint>,
    pub step_up_dx: Option<f32>,
    pub farm_band_mobs: bool,
    pub pending_target: Option<u32>,
}

impl NavCtx<'_> {
    pub fn grounded(&self) -> bool {
        self.on_ground
    }

    pub fn effective_climbing(&self) -> bool {
        self.climbing
    }

    pub fn step_up_hint(&self) -> Option<f32> {
        self.step_up_dx
    }

    pub fn physics_x(&self) -> f32 {
        self.world_x
    }

    pub fn physics_y(&self) -> f32 {
        self.world_y
    }

    pub fn nav_node_id(&self) -> u32 {
        self.node_id
    }
}

impl<'a> NavCtx<'a> {
    pub fn from_vision(
        obs: &'a [f32],
        facing: f32,
        on_ground: bool,
        climbing: bool,
        est_x: f32,
        est_y: f32,
        node_id: u32,
    ) -> Self {
        let iw = WINDOW_W as f32;
        let ih = WINDOW_H as f32;
        Self {
            obs,
            facing,
            on_ground,
            climbing,
            world_x: est_x,
            world_y: est_y,
            node_id,
            walk_right_ok: Some(obs_floor_ahead_connected(obs, 1.0)),
            walk_left_ok: Some(obs_floor_ahead_connected(obs, -1.0)),
            drop_ahead_right: Some(obs_floor_drop_ahead(obs, 1.0)),
            drop_ahead_left: Some(obs_floor_drop_ahead(obs, -1.0)),
            climb: obs_climb_hint(obs, iw, ih),
            step_up_dx: obs_step_up_dx(obs, iw, ih),
            farm_band_mobs: super::super::observation::obs_farm_band_enemies(obs, iw, 260.0),
            pending_target: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MotionExecutor {
    active: SubGoal,
    climb_align_ticks: u32,
    was_climbing: bool,
    climb_stall: u32,
    /// 开始爬绳时的脚点 y，用于判断是否真正升高/降低到顶底。
    climb_origin_y: f32,
    /// 判定到顶/底后继续顶垂直键的剩余 decide 帧。
    climb_finish_hold: u32,
    step_origin_y: f32,
    step_jumped: bool,
    step_jump_dir: f32,
    step_jump_cd: u32,
    step_stall: u32,
    step_approach_stall: u32,
    step_approach_x: f32,
    goto_stall: u32,
    goto_stall_x: f32,
    goto_dir: f32,
    goto_best_dx: f32,
    patrol_stall: u32,
    patrol_stall_x: f32,
}

/// 到顶/底后继续按垂直键的时长（decide/视觉帧数）。
/// 预览常为 ~5–10Hz 检测，8 帧约 0.8–1.6s；勿用 60（在 5Hz 下会拖到 ~12s 且易被节点抖动掐断）。
const CLIMB_FINISH_HOLD_TICKS: u32 = 8;

impl MotionExecutor {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn step_escape_dir(&self) -> f32 {
        if self.step_jump_dir != 0.0 {
            -self.step_jump_dir
        } else {
            1.0
        }
    }

    pub fn step_diag(&self) -> (u32, f32, bool, u32) {
        (
            self.step_stall,
            self.step_jump_dir,
            self.step_jumped,
            self.step_jump_cd,
        )
    }

    /// 起跳后粘滞：空中/冷却中/已升高时禁止改挂反向台阶或清目标（日志里跳起立刻往回）。
    pub fn step_up_committed(&self, grounded: bool, physics_y: f32) -> bool {
        matches!(self.active, SubGoal::StepUp { .. })
            && (self.step_jumped
                || self.step_jump_cd > 0
                || !grounded
                || (self.step_origin_y - physics_y) > 8.0)
    }

    pub fn active_goal(&self) -> SubGoal {
        self.active
    }

    pub fn step_jump_dir(&self) -> f32 {
        self.step_jump_dir
    }

    pub fn step(
        &mut self,
        config: &NavBotConfig,
        graph: &MapGraph,
        ctx: &NavCtx<'_>,
        goal: SubGoal,
    ) -> (InputFrame, ExecutorResult, &'static str) {
        if goal != self.active {
            let step_committed = self.step_up_committed(ctx.grounded(), ctx.physics_y());
            if step_committed {
                // 起跳后只允许同向换落点；取消/反向目标一律忽略，避免空中掉头。
                if let SubGoal::StepUp { target_x } = goal {
                    let new_dir = (target_x - ctx.physics_x()).signum();
                    if self.step_jump_dir == 0.0
                        || new_dir == 0.0
                        || new_dir == self.step_jump_dir
                    {
                        self.active = goal;
                    }
                }
            } else {
                self.active = goal;
                self.climb_align_ticks = 0;
                self.was_climbing = false;
                self.climb_stall = 0;
                self.climb_finish_hold = 0;
                self.goto_stall = 0;
                self.goto_dir = 0.0;
                self.goto_best_dx = 0.0;
                self.goto_stall_x = ctx.physics_x();
                self.patrol_stall = 0;
                self.patrol_stall_x = ctx.physics_x();
                if matches!(
                    goal,
                    SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. }
                ) {
                    self.climb_origin_y = ctx.physics_y();
                }
                if let SubGoal::StepUp { target_x } = goal {
                    self.step_origin_y = ctx.physics_y();
                    self.step_jumped = false;
                    self.step_stall = 0;
                    self.step_jump_cd = 0;
                    self.step_approach_stall = 0;
                    self.step_approach_x = ctx.physics_x();
                    let dx = target_x - ctx.physics_x();
                    self.step_jump_dir = dx.signum();
                    if self.step_jump_dir == 0.0 {
                        self.step_jump_dir = if ctx.facing >= 0.0 { 1.0 } else { -1.0 };
                    }
                }
            }
        }

        let mut out = InputFrame::default();
        let reason = match goal {
            SubGoal::Idle => {
                return (out, ExecutorResult::Done, "idle");
            }
            SubGoal::Patrol { dir } => self.patrol(ctx, graph, dir, &mut out),
            SubGoal::GoTo { x } => self.go_to(ctx, config, graph, x, &mut out),
            SubGoal::WalkOff { side } => Self::walk_off(ctx, graph, side, &mut out),
            SubGoal::ClimbUp { rope_x } => self.climb_up(ctx, config, graph, rope_x, &mut out),
            SubGoal::ClimbDown { rope_x } => self.climb_down(ctx, config, graph, rope_x, &mut out),
            SubGoal::StepUp { target_x } => self.step_up(ctx, config, graph, target_x, &mut out),
        };
        (out, reason.0, reason.1)
    }

    fn patrol(
        &mut self,
        ctx: &NavCtx<'_>,
        graph: &MapGraph,
        dir: f32,
        out: &mut InputFrame,
    ) -> (ExecutorResult, &'static str) {
        let d = dir.signum();
        let px = ctx.physics_x();
        // 已离开底层：禁止 patrol 主动跳崖回底（日志爬到 117 后 patrol_drop）。
        let elevated = ctx.physics_y() < 1150.0;
        if let Some(node) = graph.get(ctx.nav_node_id()) {
            if d > 0.0 && px >= node.x_max - 16.0 && ctx.walk_right_ok == Some(false) {
                if drop_ahead(ctx, d) {
                    if elevated {
                        self.patrol_stall = 0;
                        return (ExecutorResult::Failed, "patrol_elevated_no_drop");
                    }
                    set_move(out, d);
                    out.jump = true;
                    self.patrol_stall = 0;
                    return (ExecutorResult::Running, "patrol_drop");
                }
                if try_gap_hop(ctx, d, out) {
                    self.patrol_stall = 0;
                    return (ExecutorResult::Running, "patrol_gap_hop");
                }
                self.patrol_stall = 0;
                return (ExecutorResult::Failed, "patrol_right_wall");
            }
            if d < 0.0 && px <= node.x_min + 16.0 && ctx.walk_left_ok == Some(false) {
                if drop_ahead(ctx, d) {
                    if elevated {
                        self.patrol_stall = 0;
                        return (ExecutorResult::Failed, "patrol_elevated_no_drop");
                    }
                    set_move(out, d);
                    out.jump = true;
                    self.patrol_stall = 0;
                    return (ExecutorResult::Running, "patrol_drop");
                }
                if try_gap_hop(ctx, d, out) {
                    self.patrol_stall = 0;
                    return (ExecutorResult::Running, "patrol_gap_hop");
                }
                self.patrol_stall = 0;
                return (ExecutorResult::Failed, "patrol_left_wall");
            }
            // 窄台 / YOLO 仍报可走：位置几乎不动也判撞墙，避免永远 patrol 顶边。
            if d > 0.0 && px >= node.x_max - 16.0 {
                self.patrol_stall = self.patrol_stall.saturating_add(1);
                if self.patrol_stall >= 18 {
                    self.patrol_stall = 0;
                    return (ExecutorResult::Failed, "patrol_right_edge");
                }
            } else if d < 0.0 && px <= node.x_min + 16.0 {
                self.patrol_stall = self.patrol_stall.saturating_add(1);
                if self.patrol_stall >= 18 {
                    self.patrol_stall = 0;
                    return (ExecutorResult::Failed, "patrol_left_edge");
                }
            } else if (px - self.patrol_stall_x).abs() > 6.0 {
                self.patrol_stall = 0;
                self.patrol_stall_x = px;
            } else {
                self.patrol_stall = self.patrol_stall.saturating_add(1);
                if self.patrol_stall >= 36 {
                    self.patrol_stall = 0;
                    return (ExecutorResult::Failed, "patrol_stalled");
                }
            }
        }
        if can_walk(ctx, d) {
            set_move(out, d);
            // 可下落但连通门控挡着：跳下，避免台边左右蹭。
            if drop_ahead(ctx, d) && walk_blocked(ctx, d) {
                if elevated {
                    return (ExecutorResult::Failed, "patrol_elevated_no_drop");
                }
                out.jump = true;
                return (ExecutorResult::Running, "patrol_drop");
            }
            return (ExecutorResult::Running, "patrol");
        }
        // 撞边且可下落：优先走/跳下，不要立刻翻向来回蹭。
        if !elevated && try_leave_edge(ctx, d, out) {
            return (ExecutorResult::Running, "patrol_drop");
        }
        if try_gap_hop(ctx, d, out) {
            return (ExecutorResult::Running, "patrol_gap_hop");
        }
        if can_walk(ctx, -d) {
            set_move(out, -d);
            return (ExecutorResult::Running, "patrol_flip");
        }
        if !elevated && try_leave_edge(ctx, -d, out) {
            return (ExecutorResult::Running, "patrol_edge");
        }
        if try_gap_hop(ctx, -d, out) {
            return (ExecutorResult::Running, "patrol_gap_hop");
        }
        (ExecutorResult::Failed, "patrol_blocked")
    }

    fn go_to(
        &mut self,
        ctx: &NavCtx<'_>,
        config: &NavBotConfig,
        graph: &MapGraph,
        x: f32,
        out: &mut InputFrame,
    ) -> (ExecutorResult, &'static str) {
        let px = ctx.physics_x();
        let py = ctx.physics_y();
        let dx = x - px;
        let tol = config.goto_tolerance_px.max(20.0);

        if let Some(target_node) = ctx.pending_target {
            if ctx.nav_node_id() == target_node {
                self.goto_dir = 0.0;
                return (ExecutorResult::Done, "goto_done");
            }
            if let Some(dest) = graph.get(target_node) {
                // 必须真正落在目的台 x 范围内；不能仅凭 |dx|<=tol（会在邻台边缘假完成）。
                if px >= dest.x_min - 8.0
                    && px <= dest.x_max + 8.0
                    && (py - dest.y).abs() < 100.0
                {
                    self.goto_dir = 0.0;
                    return (ExecutorResult::Done, "goto_done");
                }
            }
        } else if dx.abs() <= tol {
            self.goto_dir = 0.0;
            return (ExecutorResult::Done, "goto_done");
        }

        let mut dir = dx.signum();
        if dir == 0.0 {
            if let Some(node) = graph.get(ctx.nav_node_id()) {
                dir = if (x - node.x_min).abs() <= (x - node.x_max).abs() {
                    -1.0
                } else {
                    1.0
                };
            } else {
                return (ExecutorResult::Failed, "goto_blocked");
            }
        }

        // 方向粘滞：未明显越过目标前不左右翻，避免 locomotion latch 对抽。
        if self.goto_dir == 0.0 {
            self.goto_dir = dir;
        } else if dx.signum() == self.goto_dir {
            // 仍朝目标
        } else if dx.abs() <= tol + 12.0 && ctx.pending_target.is_none() {
            self.goto_dir = 0.0;
            return (ExecutorResult::Done, "goto_done");
        } else if dx.abs() <= tol + 12.0 {
            // 有 pending：过冲也须落在目的台，否则继续粘滞方向走完。
            if let Some(to) = ctx.pending_target {
                if let Some(dest) = graph.get(to) {
                    if px >= dest.x_min - 8.0 && px <= dest.x_max + 8.0 {
                        self.goto_dir = 0.0;
                        return (ExecutorResult::Done, "goto_done");
                    }
                }
            }
        } else {
            self.goto_dir = dir;
        }
        let dir = self.goto_dir;

        if let Some(node) = graph.get(ctx.nav_node_id()) {
            let at_edge = if dir > 0.0 {
                px >= node.x_max - 12.0
            } else {
                px <= node.x_min + 12.0
            };
            if at_edge && !can_walk(ctx, dir) {
                if try_leave_edge(ctx, dir, out) {
                    return (ExecutorResult::Running, "goto_edge");
                }
                if try_gap_hop(ctx, dir, out) {
                    return (ExecutorResult::Running, "goto_gap_hop");
                }
                self.goto_dir = 0.0;
                return (ExecutorResult::Failed, "goto_edge_stuck");
            }
        }

        // 用「朝目标的最大进展」判停滞，避免 OCR x 抖动反复清零 stall。
        let progress = dx.abs();
        if self.goto_best_dx <= 0.0 || progress < self.goto_best_dx - 4.0 {
            self.goto_best_dx = progress;
            self.goto_stall = 0;
        } else if ctx.grounded() {
            self.goto_stall = self.goto_stall.saturating_add(1);
            // 意图方向 YOLO 报可走但坐标不动：更快失败，避免顶墙几十秒。
            let stall_limit = if can_walk(ctx, dir) { 12 } else { 8 };
            if self.goto_stall >= stall_limit {
                self.goto_stall = 0;
                self.goto_dir = 0.0;
                self.goto_best_dx = 0.0;
                // 台边走不动但前方可落：强制跳下。
                if drop_ahead(ctx, dir) {
                    set_move(out, dir);
                    out.jump = true;
                    return (ExecutorResult::Running, "goto_drop_jump");
                }
                // 同层缝隙：YOLO 看到对面台但不衔接 → 跳过去。
                if try_gap_hop(ctx, dir, out) {
                    return (ExecutorResult::Running, "goto_gap_hop");
                }
                if try_leave_edge(ctx, dir, out) {
                    return (ExecutorResult::Running, "goto_edge");
                }
                return (ExecutorResult::Failed, "goto_stalled");
            }
        }

        if can_walk(ctx, dir) {
            set_move(out, dir);
            if drop_ahead(ctx, dir) && walk_blocked(ctx, dir) {
                out.jump = true;
            }
            (ExecutorResult::Running, "goto")
        } else if try_leave_edge(ctx, dir, out) {
            (ExecutorResult::Running, "goto_edge")
        } else if try_gap_hop(ctx, dir, out) {
            (ExecutorResult::Running, "goto_gap_hop")
        } else {
            self.goto_dir = 0.0;
            (ExecutorResult::Failed, "goto_blocked")
        }
    }

    fn walk_off(
        ctx: &NavCtx<'_>,
        graph: &MapGraph,
        side: Side,
        out: &mut InputFrame,
    ) -> (ExecutorResult, &'static str) {
        let dir = if side == Side::Left { -1.0 } else { 1.0 };
        let Some(node) = graph.get(ctx.nav_node_id()) else {
            return (ExecutorResult::Failed, "walkoff_no_node");
        };
        let px = ctx.physics_x();
        let at_edge = if dir > 0.0 {
            px >= node.x_max - 24.0
        } else {
            px <= node.x_min + 24.0
        };
        let blocked = walk_blocked(ctx, dir);
        if at_edge && (drop_ahead(ctx, dir) || blocked) {
            set_move(out, dir);
            // 台边走不动：跳下去（下方有接台时 walk 会被当成 Blocked）。
            if blocked || !can_walk(ctx, dir) {
                out.jump = true;
            }
            return (ExecutorResult::Running, "walkoff_fall");
        }
        if can_walk(ctx, dir) {
            set_move(out, dir);
            (ExecutorResult::Running, "walkoff_approach")
        } else if drop_ahead(ctx, dir) || (blocked && at_edge) {
            set_move(out, dir);
            out.jump = true;
            (ExecutorResult::Running, "walkoff_drop")
        } else {
            (ExecutorResult::Failed, "walkoff_blocked")
        }
    }

    fn climb_up(
        &mut self,
        ctx: &NavCtx<'_>,
        config: &NavBotConfig,
        graph: &MapGraph,
        rope_x: f32,
        out: &mut InputFrame,
    ) -> (ExecutorResult, &'static str) {
        if ctx.effective_climbing() {
            self.was_climbing = true;
            // 已开始收尾则坚持倒计时：nav_node 同高抖到旁台时 near_dest 会短暂变 false，
            // 若此处要求 near_dest，会停表并永远卡在 climb_up_active（日志 105→101）。
            if Self::climb_near_dest(ctx, graph, true, self.climb_origin_y)
                || self.climb_finish_hold > 0
            {
                return self.climb_finish_tick(ctx, graph, true, out);
            }
            // 孤儿挂绳：无 pending 时，若已到绳顶高度则侧移下绳，禁止只按上卡死。
            if ctx.pending_target.is_none() {
                let at_top = graph.edges.iter().any(|e| {
                    e.kind == EdgeKind::ClimbUp
                        && graph.get(e.to).is_some_and(|d| {
                            (ctx.physics_y() - d.y).abs() <= 56.0
                                && (e.rope_x.unwrap_or(e.target_x) - ctx.physics_x()).abs()
                                    <= 72.0
                        })
                });
                if at_top {
                    let dir = if ctx.facing >= 0.0 { 1.0 } else { -1.0 };
                    set_move(out, dir);
                    // 侧移下绳，不再按 up。
                    return (ExecutorResult::Running, "climb_orphan_dismount");
                }
                out.up = true;
                return (ExecutorResult::Running, "climb_orphan_up");
            }
            out.up = true;
            return (ExecutorResult::Running, "climb_up_active");
        }

        if ctx.pending_target.is_none() && self.was_climbing {
            // 孤儿挂绳已落地。
            self.was_climbing = false;
            self.climb_stall = 0;
            self.climb_finish_hold = 0;
            return (ExecutorResult::Done, "climb_orphan_landed");
        }

        let near_top = Self::climb_near_dest(ctx, graph, true, self.climb_origin_y);
        if near_top || self.climb_finish_hold > 0 {
            return self.climb_finish_tick(ctx, graph, true, out);
        }

        let dx = rope_x - ctx.physics_x();
        let align_tol = config.climb_align_px;
        let toward_blocked = if dx > 0.0 {
            ctx.walk_right_ok == Some(false)
        } else if dx < 0.0 {
            ctx.walk_left_ok == Some(false)
        } else {
            false
        };
        let at_node_edge = graph.get(ctx.nav_node_id()).is_some_and(|n| {
            if dx > 0.0 {
                ctx.physics_x() >= n.x_max - 14.0
            } else if dx < 0.0 {
                ctx.physics_x() <= n.x_min + 14.0
            } else {
                false
            }
        });
        // 卡住时允许进入微调区，但绝不在 >grab_tol 时起跳。
        let in_approach = dx.abs() <= align_tol.max(40.0)
            && (dx.abs() <= align_tol || toward_blocked || at_node_edge || self.climb_align_ticks > 18);

        if !in_approach {
            let improved = self.goto_best_dx == 0.0 || dx.abs() + 4.0 < self.goto_best_dx;
            if improved {
                self.goto_best_dx = dx.abs();
                self.climb_align_ticks = 0;
            } else {
                self.climb_align_ticks = self.climb_align_ticks.saturating_add(1);
            }
            if self.climb_align_ticks > 36 {
                return (ExecutorResult::Failed, "climb_align_timeout");
            }
            set_move(out, dx.signum());
            // 横移被门控吃掉或靠近绳子时起跳。
            if ctx.grounded()
                && (toward_blocked
                    || self.climb_align_ticks > 6
                    || dx.abs() <= align_tol.max(40.0))
            {
                out.jump = true;
                out.up = true;
            }
            return (ExecutorResult::Running, "climb_align");
        }
        if dx.abs() <= align_tol {
            self.climb_align_ticks = 0;
            self.goto_best_dx = 0.0;
        }

        out.up = true;
        // 未进抓绳窗口：横移+起跳蹭绳（绳在台内差 10~40px 时只走会 noop）。
        if dx.abs() > CLIMB_GRAB_TOL_PX {
            set_move(out, dx.signum());
            if ctx.grounded() {
                self.climb_stall = self.climb_stall.saturating_add(1);
                out.jump = true;
                if self.climb_stall > 48 {
                    return (ExecutorResult::Failed, "climb_up_stalled");
                }
                return (ExecutorResult::Running, "climb_up_nudge");
            }
            // 空中未对准：只保留 up+横移，不要 jump（悬空梯抓到后会被 jump_off）。
            return (ExecutorResult::Running, "climb_up_air");
        }

        // 已对准绳心：空中只按 up；地面才 jump（梯子底高于站立面）。
        if !ctx.grounded() {
            self.was_climbing = true;
            self.climb_stall = 0;
            return (ExecutorResult::Running, "climb_up_air");
        }

        out.jump = true;
        self.climb_stall = self.climb_stall.saturating_add(1);
        if self.climb_stall > 36 {
            return (ExecutorResult::Failed, "climb_up_stalled");
        }
        if self.was_climbing {
            return (ExecutorResult::Running, "climb_up_regrab");
        }
        if ctx.climb.is_some() {
            (ExecutorResult::Running, "climb_up")
        } else {
            (ExecutorResult::Running, "climb_up_wait")
        }
    }

    fn climb_down(
        &mut self,
        ctx: &NavCtx<'_>,
        config: &NavBotConfig,
        graph: &MapGraph,
        rope_x: f32,
        out: &mut InputFrame,
    ) -> (ExecutorResult, &'static str) {
        if ctx.effective_climbing() {
            self.was_climbing = true;
            self.climb_stall = 0;
            if Self::climb_near_dest(ctx, graph, false, self.climb_origin_y)
                || self.climb_finish_hold > 0
            {
                return self.climb_finish_tick(ctx, graph, false, out);
            }
            out.down = true;
            return (ExecutorResult::Running, "climb_down_active");
        }

        let near_bot = Self::climb_near_dest(ctx, graph, false, self.climb_origin_y);
        if near_bot || self.climb_finish_hold > 0 {
            return self.climb_finish_tick(ctx, graph, false, out);
        }

        let dx = rope_x - ctx.physics_x();
        let align_tol = config.climb_align_px;
        let toward_blocked = if dx > 0.0 {
            ctx.walk_right_ok == Some(false)
        } else if dx < 0.0 {
            ctx.walk_left_ok == Some(false)
        } else {
            false
        };
        let at_node_edge = graph.get(ctx.nav_node_id()).is_some_and(|n| {
            if dx > 0.0 {
                ctx.physics_x() >= n.x_max - 14.0
            } else if dx < 0.0 {
                ctx.physics_x() <= n.x_min + 14.0
            } else {
                false
            }
        });
        let in_approach = dx.abs() <= align_tol.max(40.0)
            && (dx.abs() <= align_tol || toward_blocked || at_node_edge || self.climb_align_ticks > 18);

        if !in_approach {
            let improved = self.goto_best_dx == 0.0 || dx.abs() + 4.0 < self.goto_best_dx;
            if improved {
                self.goto_best_dx = dx.abs();
                self.climb_align_ticks = 0;
            } else {
                self.climb_align_ticks = self.climb_align_ticks.saturating_add(1);
            }
            if self.climb_align_ticks > 36 {
                return (ExecutorResult::Failed, "climb_align_timeout");
            }
            set_move(out, dx.signum());
            return (ExecutorResult::Running, "climb_down_align");
        }
        if dx.abs() <= align_tol {
            self.climb_align_ticks = 0;
            self.goto_best_dx = 0.0;
        }

        out.down = true;
        if dx.abs() > CLIMB_GRAB_TOL_PX {
            set_move(out, dx.signum());
            if ctx.grounded() {
                self.climb_stall = self.climb_stall.saturating_add(1);
                if self.climb_stall > 48 {
                    return (ExecutorResult::Failed, "climb_down_stalled");
                }
                return (ExecutorResult::Running, "climb_down_nudge");
            }
            return (ExecutorResult::Running, "climb_down_air");
        }

        if !ctx.grounded() {
            self.was_climbing = true;
            return (ExecutorResult::Running, "climb_down_air");
        }
        self.climb_stall = self.climb_stall.saturating_add(1);
        if self.climb_stall > 36 {
            return (ExecutorResult::Failed, "climb_down_stalled");
        }
        if self.was_climbing {
            return (ExecutorResult::Running, "climb_down_regrab");
        }
        (ExecutorResult::Running, "climb_down_grab")
    }

    /// 到顶/底后持续按垂直键，倒计时结束即 Done（climbing 粘性下也必须能结束）。
    fn climb_finish_tick(
        &mut self,
        ctx: &NavCtx<'_>,
        graph: &MapGraph,
        going_up: bool,
        out: &mut InputFrame,
    ) -> (ExecutorResult, &'static str) {
        // 收尾中掉下中段窄台：立刻失败，禁止倒计时结束后在底旁台假 Done（日志 123→48）。
        if going_up {
            if let Some(t) = ctx.pending_target {
                if let Some(dest) = graph.get(t) {
                    if ctx.physics_y() > dest.y + 90.0 {
                        self.climb_finish_hold = 0;
                        self.was_climbing = false;
                        self.climb_stall = 0;
                        return (ExecutorResult::Failed, "climb_finish_fell");
                    }
                }
            }
        }

        if self.climb_finish_hold == 0 {
            self.climb_finish_hold = CLIMB_FINISH_HOLD_TICKS;
        }
        if going_up {
            out.up = true;
            // 朝落点台面中心靠拢，禁止固定 +1 把人从窄台 123 推下去。
            if let Some(t) = ctx.pending_target {
                if let Some(dest) = graph.get(t) {
                    let mid = (dest.x_min + dest.x_max) * 0.5;
                    let px = ctx.physics_x();
                    let dx = mid - px;
                    let toward_edge = if dx > 0.0 {
                        px >= dest.x_max - 12.0
                    } else if dx < 0.0 {
                        px <= dest.x_min + 12.0
                    } else {
                        true
                    };
                    if dx.abs() > 8.0 && !toward_edge {
                        set_move(out, dx.signum());
                    }
                }
            }
        } else {
            out.down = true;
        }
        self.climb_finish_hold = self.climb_finish_hold.saturating_sub(1);
        if self.climb_finish_hold == 0 {
            // 倒计时结束仍须在落点高度附近，否则不算完成。
            if going_up && !Self::climb_near_dest(ctx, graph, true, self.climb_origin_y) {
                self.was_climbing = false;
                self.climb_stall = 0;
                return (ExecutorResult::Failed, "climb_finish_missed");
            }
            self.was_climbing = false;
            self.climb_stall = 0;
            return (
                ExecutorResult::Done,
                if going_up {
                    "climb_up_done"
                } else {
                    "climb_down_done"
                },
            );
        }
        (
            ExecutorResult::Running,
            if going_up {
                "climb_up_finish_hold"
            } else {
                "climb_down_finish_hold"
            },
        )
    }

    /// 是否已到/接近爬绳目的台（允许一定 OCR y 偏差；须有足够垂直位移）。
    fn climb_near_dest(
        ctx: &NavCtx<'_>,
        graph: &MapGraph,
        going_up: bool,
        origin_y: f32,
    ) -> bool {
        let Some(to) = ctx.pending_target else {
            return false;
        };
        let Some(dest) = graph.get(to) else {
            return false;
        };
        // 放宽 y 容差：OCR 常比台面高/低几十像素。
        if (ctx.physics_y() - dest.y).abs() > 80.0 {
            return false;
        }
        // 允许同高旁台误判（绳顶 105↔101），否则 finish_hold 永远到不了 Done。
        if ctx.nav_node_id() != to {
            let same_band = graph
                .get(ctx.nav_node_id())
                .is_some_and(|n| (n.y - dest.y).abs() <= 40.0);
            if !same_band {
                return false;
            }
        }
        let climbed = if going_up {
            origin_y - ctx.physics_y()
        } else {
            ctx.physics_y() - origin_y
        };
        climbed > 28.0
    }

    fn step_up(
        &mut self,
        ctx: &NavCtx<'_>,
        config: &NavBotConfig,
        graph: &MapGraph,
        target_x: f32,
        out: &mut InputFrame,
    ) -> (ExecutorResult, &'static str) {
        let py = ctx.physics_y();
        let px = ctx.physics_x();
        let dy_up = self.step_origin_y - py;

        // 起跳后掉回更低台/底层：立刻失败，禁止在底台对中段目标空等（日志 step_up_wait@52）。
        if ctx.grounded()
            && self.step_origin_y > 0.0
            && (py > self.step_origin_y + 70.0 || (self.step_origin_y < 1120.0 && py >= 1180.0))
        {
            self.step_jumped = false;
            self.step_jump_dir = 0.0;
            self.step_stall = 0;
            return (ExecutorResult::Failed, "step_up_fell");
        }
        // 人已在底层，目标却是中高台：不必等起跳记录，直接失败改爬绳。
        if ctx.grounded() && py >= 1180.0 {
            if let Some(t) = ctx.pending_target {
                if graph.get(t).is_some_and(|d| d.y < 1120.0) {
                    self.step_jumped = false;
                    self.step_jump_dir = 0.0;
                    self.step_stall = 0;
                    return (ExecutorResult::Failed, "step_up_fell");
                }
            }
        }

        // 已在落点节点，或坐标已落入落点台面：完成。
        if let Some(t) = ctx.pending_target {
            if let Some(dest) = graph.get(t) {
                let on_dest_node = ctx.nav_node_id() == t;
                let on_dest_xy = px >= dest.x_min - 24.0
                    && px <= dest.x_max + 24.0
                    && (py - dest.y).abs() <= 48.0;
                if ctx.grounded() && (on_dest_node || on_dest_xy) && dy_up > 8.0 {
                    self.step_jumped = false;
                    self.step_jump_dir = 0.0;
                    self.step_stall = 0;
                    return (ExecutorResult::Done, "step_up_done");
                }
            }
        }

        // 以图目标为准；仅同向且更近的 YOLO 台阶可微调，避免反向提示把人拽下台。
        let graph_dx = target_x - px;
        let dx = match ctx.step_up_hint() {
            Some(v)
                if v.signum() == graph_dx.signum()
                    && v.abs() < graph_dx.abs()
                    && graph_dx.abs() > 1.0 =>
            {
                v
            }
            _ => graph_dx,
        };

        // 起跳后 / 冷却中 / 空中：锁定起跳方向。过冲 target_x 时 graph_dx 会变号，
        // 若跟着变向就会「跳起来往回走」永远上不了台。
        let in_flight = self.step_jumped || self.step_jump_cd > 0 || !ctx.grounded();
        if !in_flight {
            if dx.abs() > 1.0 {
                self.step_jump_dir = dx.signum();
            } else if self.step_jump_dir == 0.0 {
                self.step_jump_dir = if ctx.facing >= 0.0 { 1.0 } else { -1.0 };
            }
        } else if self.step_jump_dir == 0.0 {
            self.step_jump_dir = if dx.abs() > 1.0 {
                dx.signum()
            } else if ctx.facing >= 0.0 {
                1.0
            } else {
                -1.0
            };
        }
        let dir = self.step_jump_dir;

        if self.step_jumped && ctx.grounded() {
            let at_node = ctx
                .pending_target
                .map(|t| ctx.nav_node_id() == t)
                .unwrap_or(false);
            if at_node && dy_up > 8.0 {
                self.step_jumped = false;
                self.step_jump_dir = 0.0;
                self.step_stall = 0;
                return (ExecutorResult::Done, "step_up_done");
            }
            self.step_jumped = false;
            if dy_up > 8.0 {
                // 升高了但节点未更新：立刻再跳对准，不计入失败；保持原方向。
                self.step_stall = 0;
            } else {
                self.step_stall = self.step_stall.saturating_add(1);
                if self.step_stall >= config.step_up_stall {
                    return (ExecutorResult::Failed, "step_up_stalled");
                }
            }
        }

        if self.step_jump_cd > 0 {
            self.step_jump_cd -= 1;
            if !ctx.grounded() {
                set_move(out, dir);
                self.step_jumped = true;
                return (ExecutorResult::Running, "step_up_air");
            }
            // 落地冷却：只沿锁定方向、且尚未到达/越过落点时短移；禁止反向回走。
            let toward_target = (dir > 0.0 && px < target_x - STEP_UP_ALIGN_DX)
                || (dir < 0.0 && px > target_x + STEP_UP_ALIGN_DX);
            if toward_target && step_up_can_approach(ctx, dir) {
                set_move(out, dir);
                return (ExecutorResult::Running, "step_up_approach");
            }
            return (ExecutorResult::Running, "step_up_wait");
        }

        if ctx.grounded() {
            if (px - self.step_approach_x).abs() < 2.0 {
                self.step_approach_stall = self.step_approach_stall.saturating_add(1);
            } else {
                self.step_approach_stall = 0;
                self.step_approach_x = px;
            }
        }
        let stuck_now = self.step_approach_stall >= 3;

        if !ctx.grounded() {
            set_move(out, dir);
            self.step_jumped = true;
            return (ExecutorResult::Running, "step_up_air");
        }

        // 起跳窗：靠近目标，或已到本台朝向落点的边缘（落点 target_x 常在邻台深处）。
        let at_ledge = graph.get(ctx.nav_node_id()).is_some_and(|n| {
            if dir > 0.0 {
                px >= n.x_max - 28.0
            } else {
                px <= n.x_min + 28.0
            }
        });
        // 距离用图目标；方向已锁定，避免过冲后 |dx| 变大又去 approach 反向。
        let aim_dx = graph_dx;
        let dest_higher = ctx.pending_target.and_then(|t| graph.get(t)).is_some_and(|d| {
            d.y + 20.0 < graph.get(ctx.nav_node_id()).map(|n| n.y).unwrap_or(d.y)
        });
        // 中段上台：贴边即跳（间隙常 >48px），避免 step_up_wait 空磨后掉下。
        let in_jump_range = aim_dx.abs() <= STEP_UP_JUMP_DX * 1.25
            || (at_ledge && aim_dx.abs() <= 160.0)
            || (at_ledge && dest_higher && aim_dx.abs() <= 220.0);
        let floor_ok = !obs_has_floor_signal(ctx.obs) || obs_floor_ahead(ctx.obs, dir);
        let can_approach = step_up_can_approach(ctx, dir);

        // 远处且还能走近、也未贴边：继续接近（仅锁定方向）。
        let need_closer = (dir > 0.0 && px < target_x - STEP_UP_JUMP_DX)
            || (dir < 0.0 && px > target_x + STEP_UP_JUMP_DX);
        if need_closer && !in_jump_range && can_approach && floor_ok && !stuck_now && !at_ledge {
            set_move(out, dir);
            return (ExecutorResult::Running, "step_up_approach");
        }
        // 未到起跳条件却走不动/贴边空磨：失败换边，避免小高台无限顶边。
        if !in_jump_range {
            if stuck_now || at_ledge || !can_approach {
                self.step_stall = self.step_stall.saturating_add(1);
                if self.step_stall >= config.step_up_stall {
                    return (ExecutorResult::Failed, "step_up_unreachable");
                }
                if can_approach && !at_ledge && need_closer {
                    set_move(out, dir);
                    return (ExecutorResult::Running, "step_up_approach");
                }
                return (ExecutorResult::Running, "step_up_wait");
            }
            if need_closer {
                set_move(out, dir);
                return (ExecutorResult::Running, "step_up_approach");
            }
            // 已过冲但未贴边：站桩等进入起跳窗/贴边再跳，禁止回头。
            return (ExecutorResult::Running, "step_up_wait");
        }

        out.jump = true;
        set_move(out, dir);
        self.step_jumped = true;
        self.step_jump_cd = config.step_up_jump_cooldown;
        self.step_approach_stall = 0;
        (ExecutorResult::Running, "step_up_jump")
    }
}

/// 台阶接近：物理可走，且有地板信号时前方必须有地板（与 rule_bot can_walk_dir 一致）。
fn step_up_can_approach(ctx: &NavCtx<'_>, dir: f32) -> bool {
    if !can_walk(ctx, dir) {
        return false;
    }
    if ctx.grounded()
        && !ctx.climbing
        && obs_has_floor_signal(ctx.obs)
        && !obs_floor_ahead(ctx.obs, dir)
    {
        return false;
    }
    true
}

fn set_move(out: &mut InputFrame, dir: f32) {
    if dir > 0.0 {
        out.right = true;
    } else if dir < 0.0 {
        out.left = true;
    }
}

fn walk_blocked(ctx: &NavCtx<'_>, dir: f32) -> bool {
    if dir > 0.0 {
        ctx.walk_right_ok == Some(false)
    } else if dir < 0.0 {
        ctx.walk_left_ok == Some(false)
    } else {
        false
    }
}

fn can_walk(ctx: &NavCtx<'_>, dir: f32) -> bool {
    if dir > 0.0 {
        if ctx.walk_right_ok == Some(false) && !drop_ahead(ctx, dir) {
            return false;
        }
        ctx.walk_right_ok.unwrap_or(true) || drop_ahead(ctx, dir)
    } else if dir < 0.0 {
        if ctx.walk_left_ok == Some(false) && !drop_ahead(ctx, dir) {
            return false;
        }
        ctx.walk_left_ok.unwrap_or(true) || drop_ahead(ctx, dir)
    } else {
        false
    }
}

fn drop_ahead(ctx: &NavCtx<'_>, dir: f32) -> bool {
    if dir > 0.0 {
        ctx.drop_ahead_right == Some(true)
    } else {
        ctx.drop_ahead_left == Some(true)
    }
}

fn try_leave_edge(ctx: &NavCtx<'_>, dir: f32, out: &mut InputFrame) -> bool {
    if drop_ahead(ctx, dir) {
        set_move(out, dir);
        // 走不下去时跳下坠落（门控挡、或仅靠 drop 才“可走”）。
        if walk_blocked(ctx, dir) || !ctx_walk_ok(ctx, dir) {
            out.jump = true;
        }
        return true;
    }
    if ctx.step_up_hint().is_some() {
        out.jump = true;
        if dir > 0.0 {
            out.right = true;
        } else {
            out.left = true;
        }
        return true;
    }
    false
}

/// 同层缝隙 hop：YOLO 看到近同高对面台但不衔接。
fn try_gap_hop(ctx: &NavCtx<'_>, dir: f32, out: &mut InputFrame) -> bool {
    if !ctx.grounded() || ctx.climbing {
        return false;
    }
    if !obs_same_level_gap_ahead(ctx.obs, dir, WINDOW_W, WINDOW_H) {
        return false;
    }
    set_move(out, dir);
    out.jump = true;
    true
}

fn ctx_walk_ok(ctx: &NavCtx<'_>, dir: f32) -> bool {
    if dir > 0.0 {
        ctx.walk_right_ok.unwrap_or(true)
    } else if dir < 0.0 {
        ctx.walk_left_ok.unwrap_or(true)
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::load_default_map;
    use crate::game::observation::{OBS_DIM, OBS_FLOOR_START};
    use crate::game::nav::types::SubGoal;

    fn floor_underfoot(obs: &mut [f32]) {
        obs[OBS_FLOOR_START] = 0.0;
        obs[OBS_FLOOR_START + 1] = 0.02;
        obs[OBS_FLOOR_START + 2] = 0.20;
        obs[OBS_FLOOR_START + 3] = 0.05;
    }

    #[test]
    fn climb_align_times_out_when_x_does_not_improve() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_underfoot(&mut obs);
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();
        // 离绳 26px（>18 align），位置不变 → 应超时而非永久 align。
        let mut last = ExecutorResult::Running;
        let mut reason = "init";
        for _ in 0..50 {
            let mut ctx = NavCtx::from_vision(&obs, 1.0, true, false, 1260.0, 1105.0, 49);
            ctx.pending_target = Some(92);
            let (_, r, why) = exec.step(
                &config,
                &graph,
                &ctx,
                SubGoal::ClimbUp { rope_x: 1286.0 },
            );
            last = r;
            reason = why;
            if matches!(last, ExecutorResult::Failed) {
                break;
            }
        }
        assert!(
            !matches!(reason, "climb_align"),
            "climb_align must not persist 50 frames without progress, last={reason}"
        );
    }

    #[test]
    fn climb_up_done_only_at_top_platform_after_ascent() {
        // 半高旁台（非 pending to）落地不得完成。
        let mut obs = [0.0_f32; OBS_DIM];
        floor_underfoot(&mut obs);
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();

        let mut ctx = NavCtx::from_vision(&obs, 1.0, true, false, 1477.0, 1225.0, 57);
        ctx.pending_target = Some(123);
        let (_, _, reason0) = exec.step(
            &config,
            &graph,
            &ctx,
            SubGoal::ClimbUp { rope_x: 1477.0 },
        );
        assert!(
            reason0.starts_with("climb"),
            "start climb, got {reason0}"
        );

        // 蹭到半高旁台 67：应 regrab/wait，不能 Done。
        let mut mid = NavCtx::from_vision(&obs, 1.0, true, false, 1369.0, 1178.0, 67);
        mid.pending_target = Some(123);
        exec.was_climbing = true;
        let (out_mid, result_mid, reason_mid) = exec.step(
            &config,
            &graph,
            &mid,
            SubGoal::ClimbUp { rope_x: 1477.0 },
        );
        assert!(
            !matches!(result_mid, ExecutorResult::Done),
            "mid platform must not finish climb, reason={reason_mid}"
        );
        assert!(
            out_mid.up || out_mid.jump || out_mid.right || out_mid.left,
            "should keep climbing/aligning, out={out_mid:?} reason={reason_mid}"
        );

        // 真正到顶台 123：先 finish_hold 持续按 up，约 1s 后再 Done。
        let top_y = graph.get(123).map(|n| n.y).unwrap_or(900.0);
        let mut top = NavCtx::from_vision(&obs, 1.0, true, false, 1477.0, top_y, 123);
        top.pending_target = Some(123);
        let (out_hold, result_hold, reason_hold) = exec.step(
            &config,
            &graph,
            &top,
            SubGoal::ClimbUp { rope_x: 1477.0 },
        );
        assert!(
            matches!(result_hold, ExecutorResult::Running),
            "first top frame should hold up, got {reason_hold}"
        );
        assert_eq!(reason_hold, "climb_up_finish_hold");
        assert!(out_hold.up && !out_hold.jump, "finish hold: up, no jump");

        // 收尾中掉下窄台：必须 Failed，禁止在旁台假 Done。
        let mut fell = NavCtx::from_vision(&obs, 1.0, true, false, 1525.0, 1130.0, 48);
        fell.pending_target = Some(123);
        exec.climb_finish_hold = 4;
        exec.was_climbing = true;
        let (_, fell_r, fell_reason) = exec.step(
            &config,
            &graph,
            &fell,
            SubGoal::ClimbUp { rope_x: 1477.0 },
        );
        assert!(
            matches!(fell_r, ExecutorResult::Failed),
            "fell during finish must fail, got {fell_reason}"
        );
        assert_eq!(fell_reason, "climb_finish_fell");

        // 重新到顶再收尾完成。
        exec.climb_finish_hold = 0;
        exec.was_climbing = true;
        exec.climb_origin_y = 1225.0;
        let (out_hold, result_hold, reason_hold) = exec.step(
            &config,
            &graph,
            &top,
            SubGoal::ClimbUp { rope_x: 1477.0 },
        );
        assert_eq!(reason_hold, "climb_up_finish_hold");
        assert!(out_hold.up && !out_hold.jump);

        let mut reason_top = reason_hold;
        let mut result_top = result_hold;
        for _ in 0..CLIMB_FINISH_HOLD_TICKS {
            let (_, r, reason) = exec.step(
                &config,
                &graph,
                &top,
                SubGoal::ClimbUp { rope_x: 1477.0 },
            );
            result_top = r;
            reason_top = reason;
            if matches!(result_top, ExecutorResult::Done) {
                break;
            }
        }
        assert!(
            matches!(result_top, ExecutorResult::Done),
            "after finish hold must done, reason={reason_top}"
        );
        assert_eq!(reason_top, "climb_up_done");

        // climbing 粘在 true 时到顶也必须能结束（复现无限 finish_hold）。
        let mut exec2 = MotionExecutor::default();
        let mut bot = NavCtx::from_vision(&obs, 1.0, true, false, 1477.0, 1225.0, 57);
        bot.pending_target = Some(123);
        let _ = exec2.step(
            &config,
            &graph,
            &bot,
            SubGoal::ClimbUp { rope_x: 1477.0 },
        );
        let mut sticky = NavCtx::from_vision(&obs, 1.0, false, true, 1477.0, top_y, 123);
        sticky.pending_target = Some(123);
        let mut done_sticky = false;
        for _ in 0..=CLIMB_FINISH_HOLD_TICKS {
            let (_, r, _) = exec2.step(
                &config,
                &graph,
                &sticky,
                SubGoal::ClimbUp { rope_x: 1477.0 },
            );
            if matches!(r, ExecutorResult::Done) {
                done_sticky = true;
                break;
            }
        }
        assert!(
            done_sticky,
            "sticky climbing at top must finish within hold window"
        );
    }

    #[test]
    fn climb_finish_hold_survives_same_band_node_flicker() {
        // 复现 preview：绳顶 finish_hold 时 nav 105→101，倒计时被掐成 climb_up_active。
        let mut obs = [0.0_f32; OBS_DIM];
        floor_underfoot(&mut obs);
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();

        let bot_y = graph.get(57).map(|n| n.y).unwrap_or(1225.0);
        let mut bot = NavCtx::from_vision(&obs, 1.0, true, false, 1477.0, bot_y, 57);
        bot.pending_target = Some(123);
        let _ = exec.step(
            &config,
            &graph,
            &bot,
            SubGoal::ClimbUp { rope_x: 1477.0 },
        );

        let top_y = graph.get(123).map(|n| n.y).unwrap_or(900.0);
        let flicker = graph
            .nodes
            .values()
            .find(|n| n.id != 123 && (n.y - top_y).abs() <= 40.0)
            .map(|n| n.id)
            .expect("same-band neighbor");

        let mut top = NavCtx::from_vision(&obs, 1.0, false, true, 1477.0, top_y, 123);
        top.pending_target = Some(123);
        let (_, r0, reason0) = exec.step(
            &config,
            &graph,
            &top,
            SubGoal::ClimbUp { rope_x: 1477.0 },
        );
        assert!(matches!(r0, ExecutorResult::Running));
        assert_eq!(reason0, "climb_up_finish_hold");

        let mut flicked = NavCtx::from_vision(&obs, 1.0, false, true, 1470.0, top_y, flicker);
        flicked.pending_target = Some(123);
        let mut done = false;
        for _ in 0..=CLIMB_FINISH_HOLD_TICKS + 2 {
            let (_, r, reason) = exec.step(
                &config,
                &graph,
                &flicked,
                SubGoal::ClimbUp { rope_x: 1477.0 },
            );
            assert!(
                reason == "climb_up_finish_hold" || reason == "climb_up_done",
                "must keep finish_hold across flicker, got {reason}"
            );
            if matches!(r, ExecutorResult::Done) {
                done = true;
                break;
            }
        }
        assert!(done, "finish_hold must complete after same-band flicker");
    }

    #[test]
    fn step_up_jumps_when_near_target_no_floor_ahead() {
        // 对齐 rule_bot seek_step_up_jumps_when_near_ledge：dx≈30 < 48 → 起跳。
        let mut obs = [0.0_f32; OBS_DIM];
        floor_underfoot(&mut obs);
        let mut ctx = NavCtx::from_vision(&obs, 1.0, true, false, 735.0, 1225.0, 43);
        ctx.walk_right_ok = Some(true);
        ctx.walk_left_ok = Some(true);
        ctx.drop_ahead_right = Some(false);
        ctx.drop_ahead_left = Some(false);
        ctx.pending_target = Some(54);

        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();
        let (out, result, reason) = exec.step(
            &config,
            &graph,
            &ctx,
            SubGoal::StepUp { target_x: 764.0 },
        );
        assert!(
            matches!(result, ExecutorResult::Running),
            "result={result:?}"
        );
        assert!(
            out.jump && out.right,
            "near step target must jump+right, reason={reason} out={out:?}"
        );
        assert_eq!(reason, "step_up_jump");
    }

    #[test]
    fn step_up_keeps_jump_dir_when_airborne_overshoots_target() {
        // 复现 preview：左跳过冲 target 后 graph_dx 变号，不得空中改右走。
        let mut obs = [0.0_f32; OBS_DIM];
        floor_underfoot(&mut obs);
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();

        let mut ctx = NavCtx::from_vision(&obs, -1.0, true, false, 652.0, 1233.0, 42);
        ctx.walk_right_ok = Some(true);
        ctx.walk_left_ok = Some(true);
        ctx.drop_ahead_right = Some(false);
        ctx.drop_ahead_left = Some(false);
        ctx.pending_target = Some(24);
        let (out, _, reason) = exec.step(
            &config,
            &graph,
            &ctx,
            SubGoal::StepUp { target_x: 650.0 },
        );
        assert_eq!(reason, "step_up_jump");
        assert!(out.jump && out.left && !out.right, "out={out:?}");

        // 空中过冲到 target 左侧：graph_dx>0，旧逻辑会右走。
        let mut air = NavCtx::from_vision(&obs, -1.0, false, false, 643.0, 1219.0, 42);
        air.walk_right_ok = Some(true);
        air.walk_left_ok = Some(true);
        air.pending_target = Some(24);
        let (out_air, _, reason_air) = exec.step(
            &config,
            &graph,
            &air,
            SubGoal::StepUp { target_x: 650.0 },
        );
        assert!(
            matches!(reason_air, "step_up_air" | "step_up_wait" | "step_up_approach"),
            "reason={reason_air}"
        );
        assert!(
            out_air.left && !out_air.right,
            "airborne must keep left, reason={reason_air} out={out_air:?}"
        );
        assert!(!out_air.jump || out_air.left);
    }

    #[test]
    fn step_up_ignores_reverse_goal_while_airborne() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_underfoot(&mut obs);
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();

        let mut ctx = NavCtx::from_vision(&obs, 1.0, true, false, 735.0, 1225.0, 43);
        ctx.walk_right_ok = Some(true);
        ctx.walk_left_ok = Some(true);
        ctx.drop_ahead_right = Some(false);
        ctx.drop_ahead_left = Some(false);
        ctx.pending_target = Some(54);
        let (out, _, reason) = exec.step(
            &config,
            &graph,
            &ctx,
            SubGoal::StepUp { target_x: 764.0 },
        );
        assert_eq!(reason, "step_up_jump");
        assert!(out.right && out.jump, "out={out:?}");
        assert!(exec.step_up_committed(false, 1180.0));

        // 空中被改成反向台阶：必须忽略，继续右飞。
        let mut air = NavCtx::from_vision(&obs, 1.0, false, false, 750.0, 1180.0, 43);
        air.walk_right_ok = Some(true);
        air.walk_left_ok = Some(true);
        air.pending_target = Some(24);
        let (out_air, _, reason_air) = exec.step(
            &config,
            &graph,
            &air,
            SubGoal::StepUp { target_x: 650.0 },
        );
        assert!(
            matches!(reason_air, "step_up_air" | "step_up_wait"),
            "reason={reason_air}"
        );
        assert!(
            out_air.right && !out_air.left,
            "must ignore reverse StepUp mid-air, reason={reason_air} out={out_air:?}"
        );
        assert!(matches!(
            exec.active_goal(),
            SubGoal::StepUp { target_x } if (target_x - 764.0).abs() < 1.0
        ));
    }

    #[test]
    fn step_up_walks_when_far_and_floor_ahead() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_underfoot(&mut obs);
        obs[OBS_FLOOR_START + 4] = 0.08;
        obs[OBS_FLOOR_START + 5] = 0.02;
        obs[OBS_FLOOR_START + 6] = 0.18;
        obs[OBS_FLOOR_START + 7] = 0.05;

        let mut ctx = NavCtx::from_vision(&obs, 1.0, true, false, 700.0, 1225.0, 43);
        ctx.walk_right_ok = Some(true);
        ctx.drop_ahead_right = Some(false);

        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();
        let (out, _, reason) = exec.step(
            &config,
            &graph,
            &ctx,
            SubGoal::StepUp { target_x: 780.0 },
        );
        assert!(
            !out.jump && out.right,
            "far with floor ahead should walk, reason={reason} out={out:?}"
        );
        assert_eq!(reason, "step_up_approach");
    }

    #[test]
    fn step_up_jumps_at_ledge_even_if_target_far() {
        // 105 右缘 → 98：target_x=623 在邻台深处，|dx|≈100 > 起跳窗，但贴边应起跳。
        let mut obs = [0.0_f32; OBS_DIM];
        floor_underfoot(&mut obs);
        let mut ctx = NavCtx::from_vision(&obs, 1.0, true, false, 519.0, 865.0, 105);
        ctx.walk_right_ok = Some(true);
        ctx.walk_left_ok = Some(true);
        ctx.drop_ahead_right = Some(false);
        ctx.pending_target = Some(98);

        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        assert!(graph.get(105).is_some_and(|n| 519.0 >= n.x_max - 28.0));
        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();
        let (out, _, reason) = exec.step(
            &config,
            &graph,
            &ctx,
            SubGoal::StepUp { target_x: 623.0 },
        );
        assert!(
            out.jump && out.right,
            "ledge with far target must jump, reason={reason} out={out:?}"
        );
        assert_eq!(reason, "step_up_jump");
    }

    #[test]
    fn walk_off_jumps_when_drop_ahead_but_walk_blocked() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_underfoot(&mut obs);
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let Some(n) = graph.get(135) else {
            return;
        };
        let mut ctx = NavCtx::from_vision(&obs, -1.0, true, false, n.x_min + 4.0, n.y, 135);
        ctx.walk_left_ok = Some(false);
        ctx.walk_right_ok = Some(true);
        ctx.drop_ahead_left = Some(true);
        ctx.drop_ahead_right = Some(false);

        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();
        let (out, _, reason) = exec.step(
            &config,
            &graph,
            &ctx,
            SubGoal::WalkOff {
                side: crate::game::nav::types::Side::Left,
            },
        );
        assert!(
            out.left && out.jump,
            "blocked ledge drop must jump left, reason={reason} out={out:?}"
        );
    }

    #[test]
    fn patrol_jumps_off_when_drop_blocked_by_gate() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_underfoot(&mut obs);
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        // 底层才允许 patrol_drop；用底台 y，避免 elevated_no_drop。
        let Some(n) = graph.get(57).or_else(|| graph.get(41)) else {
            return;
        };
        let mut ctx = NavCtx::from_vision(&obs, -1.0, true, false, n.x_min + 8.0, n.y.max(1200.0), n.id);
        ctx.walk_left_ok = Some(false);
        ctx.walk_right_ok = Some(false);
        ctx.drop_ahead_left = Some(true);
        ctx.drop_ahead_right = Some(false);

        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();
        let (out, _, reason) = exec.step(
            &config,
            &graph,
            &ctx,
            SubGoal::Patrol { dir: -1.0 },
        );
        assert!(
            out.left && out.jump,
            "patrol at blocked drop must jump, reason={reason} out={out:?}"
        );
        assert_eq!(reason, "patrol_drop");
    }

    #[test]
    fn patrol_elevated_refuses_drop() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_underfoot(&mut obs);
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let Some(n) = graph.get(123) else {
            return;
        };
        let mut ctx = NavCtx::from_vision(&obs, 1.0, true, false, n.x_max - 8.0, n.y, 123);
        ctx.walk_left_ok = Some(false);
        ctx.walk_right_ok = Some(false);
        ctx.drop_ahead_right = Some(true);

        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();
        let (out, result, reason) = exec.step(
            &config,
            &graph,
            &ctx,
            SubGoal::Patrol { dir: 1.0 },
        );
        assert!(
            matches!(result, ExecutorResult::Failed),
            "elevated patrol must not jump down, reason={reason} out={out:?}"
        );
        assert_eq!(reason, "patrol_elevated_no_drop");
        assert!(!out.jump);
    }

    #[test]
    fn step_up_fails_when_already_on_bottom() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_underfoot(&mut obs);
        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();
        let mut ctx = NavCtx::from_vision(&obs, 1.0, true, false, 1670.0, 1234.0, 52);
        ctx.pending_target = Some(117);
        let (_, result, reason) = exec.step(
            &config,
            &graph,
            &ctx,
            SubGoal::StepUp { target_x: 1574.0 },
        );
        assert!(
            matches!(result, ExecutorResult::Failed),
            "bottom must not wait on elevated step_up, reason={reason}"
        );
        assert_eq!(reason, "step_up_fell");
    }

    #[test]
    fn goto_gap_hops_when_same_level_platforms_disconnected() {
        let mut obs = [0.0_f32; OBS_DIM];
        // 脚下窄台 + 右侧缝后同层台（与 observation 单测同构）。
        obs[OBS_FLOOR_START] = 0.0;
        obs[OBS_FLOOR_START + 1] = 0.01;
        obs[OBS_FLOOR_START + 2] = 12.0 / 1368.0;
        obs[OBS_FLOOR_START + 3] = 0.02;
        let under_half = 6.0 / 1368.0;
        let gap = 50.0 / 1368.0;
        let opp_half = 40.0 / 1368.0;
        obs[OBS_FLOOR_START + 4] = under_half + gap + opp_half;
        obs[OBS_FLOOR_START + 5] = 0.01;
        obs[OBS_FLOOR_START + 6] = 80.0 / 1368.0;
        obs[OBS_FLOOR_START + 7] = 0.02;

        let mut ctx = NavCtx::from_vision(&obs, 1.0, true, false, 400.0, 1105.0, 19);
        assert_eq!(ctx.walk_right_ok, Some(false));
        ctx.drop_ahead_right = Some(false);
        ctx.drop_ahead_left = Some(false);

        let map = load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let config = NavBotConfig::default();
        let mut exec = MotionExecutor::default();
        let (out, result, reason) = exec.step(
            &config,
            &graph,
            &ctx,
            SubGoal::GoTo { x: 520.0 },
        );
        assert!(
            matches!(result, ExecutorResult::Running),
            "result={result:?} reason={reason}"
        );
        assert!(
            out.jump && out.right,
            "disconnected same-level must hop, reason={reason} out={out:?}"
        );
        assert_eq!(reason, "goto_gap_hop");
    }
}
