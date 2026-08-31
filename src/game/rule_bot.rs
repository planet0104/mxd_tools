//! 纯规则自动玩：YOLO 观测 → 输入帧（含脱困、短时记忆、访问网格探索）。

use std::collections::HashSet;

use super::input::InputFrame;
use super::map::{ClimbDir, ClimbHint};
use super::observation::{
    obs_assess_enemy_contact, obs_drop_in_pickup_range, obs_enemy_in_attack_range,
    obs_floor_ahead, obs_has_drop, obs_has_enemy, obs_has_floor_signal,
    obs_has_ladder_or_rope_signal, EnemyContactAssessment, OBS_DIM, OBS_DROP_SLOTS,
    OBS_DROP_START, OBS_ENEMY_SLOTS, OBS_ENEMY_START, OBS_SLOT_DIM,
};

const MEMORY_TICKS: u32 = 72;
const EXPLORE_ROPE_BOOST: u32 = 36;
const STUCK_MOVE_EPS: f32 = 2.5;
const STUCK_TICKS: u32 = 15;
const STUCK_REVERSE_TICKS: u32 = 36;
const CLIMB_ALIGN_OBS: f32 = 0.015;
/// 绳梯水平对齐（地图像素）。
const CLIMB_ALIGN_PX: f32 = 12.0;
const HP_POTION_RATIO: f32 = 0.55;
/// 仅在血量危急且贴身时才撤退（优先还手）。
const HP_RETREAT_RATIO: f32 = 0.35;
/// 单侧贴身怪 ≥ 此数 → 撤退换区。
const CONTACT_RETREAT_MIN: u32 = 2;
/// 被围：左右至少各有 1 只贴身怪。
const SURROUNDED_SIDE_MIN: u32 = 1;
/// 贴身危险：小于此距离优先避险（touch 半径约 28）。
const TOUCH_AVOID_DX: f32 = 36.0;
/// 站砍上限：与挥砍前伸约 90 对齐，进距即砍，防止贴脸才出手。
const STRIKE_HOLD_MAX: f32 = 90.0;
/// 农怪「本段」水平半径：超过则视为已清本段，允许离台换层。
const FARM_LOCAL_DX: f32 = 260.0;
/// 同台高度差：超过则视为其他平台，不站等/不禁追（否则永远跳不上去）。
const SAME_PLATFORM_DY: f32 = 28.0;
/// 近距站等：怪在此距离内才站桩等折返；更远则巡逻接近，禁止永久 noop。
const STAND_WAIT_MAX: f32 = 220.0;
/// 禁追：仅在此距离内禁止朝怪走（远处允许巡逻接近）。
const NO_CHASE_DX: f32 = 220.0;
/// 高台上落回农怪：怪水平距离大于此值才视为安全窗口。
const DROP_SAFE_DX: f32 = 95.0;
/// 访问网格：水平 80px、垂直 120px（与旧探索计分一致）。
const X_CELL_PX: f32 = 80.0;
const ALTITUDE_BAND_PX: f32 = 120.0;
/// 决策次数（约 5Hz 视觉帧）；同一高度层停留过久 → 强制换层探路。
const BAND_STAGNATION_DECISIONS: u32 = 30;
/// 连续决策未踏入新格子 → 强制换层探路。
const NO_NEW_CELL_DECISIONS: u32 = 18;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StuckPhase {
    Normal,
    Reverse,
}

/// 允许触发跳跃的场景（换层 / 抓绳梯 / 必要越崖），不含平地巡逻蹭边。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JumpPurpose {
    /// 探索换层：SeekVertical 模式下的平台边/悬崖。
    PlatformChange,
    /// 战斗追怪、撤退脱困、卡死恢复：物理悬崖且确有必要越崖。
    CliffCrossing,
    ClimbEntry,
    /// 跳上一层紧邻台阶。
    StepUp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExploreMode {
    Normal,
    /// 当前层/区域已逛够，优先找绳梯或平台边跳离。
    SeekVertical,
}

/// 规则 bot 状态（跨帧记忆）。
#[derive(Debug, Clone)]
pub struct RuleBot {
    patrol_dir: f32,
    stuck_ticks: u32,
    last_x: f32,
    last_y: f32,
    rope_memory: u32,
    drop_memory: u32,
    stuck_phase: StuckPhase,
    stuck_phase_ticks: u32,
    climb_attempt_ticks: u32,
    initialized: bool,
    visited: HashSet<(i32, i32)>,
    last_y_band: i32,
    ticks_on_band: u32,
    ticks_without_new_cell: u32,
    explore_mode: ExploreMode,
    explore_mode_ticks: u32,
    spawn_y: f32,
    spawn_y_band: i32,
    /// 当前要清怪的农怪脚点高度（避险跳上高台后仍记住）。
    pub farm_y: f32,
    /// 是否在高台避险、等待落回农怪。
    pub perching: bool,
    /// 高台等待 tick，超时强制找落点。
    perch_ticks: u32,
    /// 刚落回农怪层：短暂禁止再跳上台，先站砍/让位。
    land_cooldown: u32,
    /// 最近一次决策原因（预览日志）。
    pub last_reason: &'static str,
    /// 水平方向抖动计数：左右来回则强制换层跳。
    last_move_dir: f32,
    dir_flip_streak: u32,
}

impl Default for RuleBot {
    fn default() -> Self {
        Self {
            patrol_dir: 1.0,
            stuck_ticks: 0,
            last_x: 0.0,
            last_y: 0.0,
            rope_memory: 0,
            drop_memory: 0,
            stuck_phase: StuckPhase::Normal,
            stuck_phase_ticks: 0,
            climb_attempt_ticks: 0,
            initialized: false,
            visited: HashSet::new(),
            last_y_band: 0,
            ticks_on_band: 0,
            ticks_without_new_cell: 0,
            explore_mode: ExploreMode::Normal,
            explore_mode_ticks: 0,
            spawn_y: 0.0,
            spawn_y_band: 0,
            farm_y: 0.0,
            perching: false,
            perch_ticks: 0,
            land_cooldown: 0,
            last_reason: "init",
            last_move_dir: 0.0,
            dir_flip_streak: 0,
        }
    }
}

pub fn visit_key(x: f32, y: f32) -> (i32, i32) {
    (
        (x / X_CELL_PX).floor() as i32,
        (y / ALTITUDE_BAND_PX).floor() as i32,
    )
}

/// 单帧决策上下文。
#[derive(Debug, Clone, Copy)]
pub struct RuleBotCtx<'a> {
    pub obs: &'a [f32; OBS_DIM],
    pub facing: f32,
    pub on_ground: bool,
    pub climbing: bool,
    pub hp: i32,
    pub max_hp: i32,
    pub potions: u32,
    pub player_x: f32,
    pub player_y: f32,
    pub kills: u32,
    pub physics_right_ok: Option<bool>,
    pub physics_left_ok: Option<bool>,
    pub physics_drop_right: Option<bool>,
    pub physics_drop_left: Option<bool>,
    pub sim_mob_in_melee: bool,
    pub sim_mob_on_attackable_footing: bool,
    pub engage: Option<super::EngageHint>,
    /// 含脚下低一层怪的宽接战信息（高台避险用）。
    pub engage_wide: Option<super::EngageHint>,
    /// 紧邻当前层的绳/梯（物理），远处上层绳不算。
    pub climb: Option<ClimbHint>,
    /// 可跳上一层台阶的相对 dx。
    pub step_up_dx: Option<f32>,
    /// 农怪高度带是否还有活怪（清层/禁止提前离场）。
    pub farm_band_mobs: bool,
}

impl<'a> RuleBotCtx<'a> {
    pub fn from_sim(sim: &super::GameSim, obs: &'a [f32; OBS_DIM]) -> Self {
        Self::from_sim_with_farm_y(sim, obs, sim.state.player.y)
    }

    pub fn from_sim_with_farm_y(
        sim: &super::GameSim,
        obs: &'a [f32; OBS_DIM],
        farm_y: f32,
    ) -> Self {
        let (pr, pl) = sim.physics_walk_ok_pair();
        let (pdr, pdl) = sim.physics_drop_ok_pair();
        let p = &sim.state.player;
        Self {
            obs,
            facing: p.facing,
            on_ground: p.on_ground,
            climbing: p.climbing,
            hp: p.hp,
            max_hp: p.max_hp,
            potions: sim.state.potions,
            player_x: p.x,
            player_y: p.y,
            kills: sim.state.kills,
            physics_right_ok: pr,
            physics_left_ok: pl,
            physics_drop_right: pdr,
            physics_drop_left: pdl,
            sim_mob_in_melee: sim.mob_in_strike_band(),
            sim_mob_on_attackable_footing: sim.mob_on_attackable_footing(),
            engage: sim.nearest_engage_hint(),
            engage_wide: sim.nearest_engage_hint_wide(),
            climb: sim.nearest_adjacent_climb(),
            step_up_dx: sim.nearest_step_up_dx(),
            farm_band_mobs: farm_y > 0.0
                && sim.mobs_near_xy(farm_y, 55.0, p.x, FARM_LOCAL_DX),
        }
    }
}

impl RuleBot {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn decide(&mut self, ctx: RuleBotCtx<'_>) -> InputFrame {
        self.tick_memory(ctx.obs);
        self.tick_stuck(ctx);

        if !self.initialized {
            self.last_x = ctx.player_x;
            self.last_y = ctx.player_y;
            self.last_y_band = visit_key(ctx.player_x, ctx.player_y).1;
            self.spawn_y = ctx.player_y;
            self.spawn_y_band = self.last_y_band;
            self.farm_y = ctx.player_y;
            self.perching = false;
            self.perch_ticks = 0;
            self.land_cooldown = 0;
            self.initialized = true;
            self.last_reason = "init";
            self.last_move_dir = 0.0;
            self.dir_flip_streak = 0;
        }
        self.tick_exploration(ctx.player_x, ctx.player_y);
        self.update_perch_state(ctx);
        if self.land_cooldown > 0 {
            self.land_cooldown -= 1;
        }

        let farm_mobs = ctx.farm_band_mobs || self.farm_layer_has_mobs(ctx);
        // 仅挥砍距离内才算「必须留下来打」；远处同层怪不得取消换层。
        let melee_hold = ctx.sim_mob_in_melee
            || ctx
                .engage
                .filter(|e| e.dy.abs() <= SAME_PLATFORM_DY && e.dx.abs() <= STRIKE_HOLD_MAX)
                .is_some();
        let on_first = !self.left_first_platform_layer(ctx);

        // 已在换层：只因贴身接战才退回 Normal；远处怪不得撕掉 SeekVertical（否则左右抖）。
        if on_first && melee_hold && farm_mobs && self.explore_mode == ExploreMode::SeekVertical {
            self.explore_mode = ExploreMode::Normal;
            self.explore_mode_ticks = 0;
            self.last_reason = "seek_cancel_melee";
        } else if !on_first
            && obs_has_enemy(ctx.obs)
            && !ctx.sim_mob_on_attackable_footing
            && !self.perching
            && !farm_mobs
        {
            self.explore_mode = ExploreMode::SeekVertical;
            self.explore_mode_ticks = 0;
            if self.rope_memory < EXPLORE_ROPE_BOOST {
                self.rope_memory = EXPLORE_ROPE_BOOST;
            }
        } else if on_first
            && !melee_hold
            && !ctx.sim_mob_in_melee
            && self.explore_mode == ExploreMode::Normal
            && (ctx.kills > 0 || self.ticks_on_band >= 8)
            && (!farm_mobs || self.dir_flip_streak >= 3)
        {
            // 首台本段半径内已无怪（或左右抖）→ 换层。
            self.explore_mode = ExploreMode::SeekVertical;
            self.explore_mode_ticks = 0;
            if self.rope_memory < EXPLORE_ROPE_BOOST {
                self.rope_memory = EXPLORE_ROPE_BOOST;
            }
        } else if !on_first
            && !ctx.sim_mob_on_attackable_footing
            && !self.perching
            && !farm_mobs
            && self.explore_mode == ExploreMode::Normal
            && (self.ticks_on_band >= BAND_STAGNATION_DECISIONS
                || self.ticks_without_new_cell >= NO_NEW_CELL_DECISIONS)
        {
            self.explore_mode = ExploreMode::SeekVertical;
            self.explore_mode_ticks = 0;
        }

        if let Some(frame) = self.try_stuck_recovery(ctx) {
            let mut frame = frame;
            self.strip_chase_toward_mob(ctx, &mut frame);
            self.last_reason = "stuck_recovery";
            self.note_move_dir(&frame);
            return frame;
        }

        let mut out = InputFrame::default();

        // 优先级：喝药 > 贴身还手/正面接战 > 被围才撤 > 探路。
        // SeekVertical 换层中：只还手（melee），不站等/躲避，否则起跳方向会被撕掉。
        if self.try_potion(ctx, &mut out) {
            self.last_reason = "potion";
            self.note_move_dir(&out);
            return out;
        }
        let seeking_now = self.explore_mode == ExploreMode::SeekVertical;
        let want_combat = if seeking_now {
            ctx.sim_mob_in_melee
        } else {
            !self.must_flee(ctx)
        };
        if want_combat && self.try_combat(ctx, &mut out) {
            // 战斗中禁止 ensure_locomotion 补走路，否则站砍会被补成追怪。
            if !seeking_now {
                self.strip_chase_toward_mob(ctx, &mut out);
            }
            self.last_reason = if seeking_now {
                "seek_melee"
            } else {
                "combat"
            };
            self.note_move_dir(&out);
            return out;
        }
        if !seeking_now && self.try_flee(ctx, &mut out) {
            self.ensure_locomotion(ctx, &mut out);
            self.strip_chase_toward_mob(ctx, &mut out);
            self.last_reason = "flee";
            self.note_move_dir(&out);
            return out;
        }

        // 左右抖动：贴边强制换层跳，打断巡逻/拾取互撕。
        if self.dir_flip_streak >= 4 && ctx.on_ground && !melee_hold {
            let prefer = if self.last_move_dir != 0.0 {
                self.last_move_dir
            } else {
                self.patrol_dir.signum()
            };
            if self.try_leave_edge(ctx, prefer, &mut out)
                || self.try_leave_edge(ctx, -prefer, &mut out)
            {
                self.explore_mode = ExploreMode::SeekVertical;
                self.explore_mode_ticks = 0;
                self.dir_flip_streak = 0;
                self.last_reason = "anti_osc_leave";
                self.note_move_dir(&out);
                return out;
            }
        }

        let seeking = seeking_now;
        // 换层探索中禁止「避险落回」占用决策。
        if seeking && self.perching {
            self.perching = false;
            self.perch_ticks = 0;
        }
        // 已离首台：不再因旧层 farm_band 卡住换层。
        // 首台 Seek：本段仍有怪则先巡逻；Normal 含 perch。
        let farm_uncleared = if seeking && self.left_first_platform_layer(ctx) {
            false
        } else if seeking {
            ctx.farm_band_mobs
        } else {
            ctx.farm_band_mobs || self.perching
        };
        if seeking {
            // 腾空：保持换层起跳方向，禁止中途改成往回走。
            if !ctx.on_ground {
                let dir = if self.patrol_dir >= 0.0 { 1.0 } else { -1.0 };
                set_move_dir(&mut out, dir, false);
                out.jump = false;
                self.last_reason = "seek_airborne";
                self.note_move_dir(&out);
                return out;
            }
            if farm_uncleared {
                if self.try_pickup(ctx, &mut out, true) {
                    self.ensure_locomotion(ctx, &mut out);
                    self.strip_chase_toward_mob(ctx, &mut out);
                    self.last_reason = "seek_farm_pickup";
                    self.note_move_dir(&out);
                    return out;
                }
                self.try_patrol(ctx, &mut out);
                self.ensure_locomotion(ctx, &mut out);
                self.strip_chase_toward_mob(ctx, &mut out);
                self.last_reason = "seek_farm_patrol";
                self.note_move_dir(&out);
                return out;
            }
            if self.try_pickup(ctx, &mut out, true) {
                self.ensure_locomotion(ctx, &mut out);
                self.strip_chase_toward_mob(ctx, &mut out);
                self.last_reason = "seek_pickup_near";
                self.note_move_dir(&out);
                return out;
            }
            if self.try_climb(ctx, &mut out) {
                self.resolve_blocked_horizontal(ctx, &mut out);
                self.strip_chase_toward_mob(ctx, &mut out);
                self.last_reason = "seek_climb";
                self.note_move_dir(&out);
                return out;
            }
            if self.try_step_up(ctx, &mut out) {
                self.resolve_blocked_horizontal(ctx, &mut out);
                self.strip_chase_toward_mob(ctx, &mut out);
                // strip 后若只剩空走，改走换层边，禁止朝怪 step_up。
                if !out.left && !out.right && !out.jump && !out.up {
                    let _ = self.try_edge_jump(ctx, &mut out);
                }
                self.last_reason = "seek_step_up";
                self.note_move_dir(&out);
                return out;
            }
            // 本段已清：换层跳优先，远处怪不得挡住。
            if !melee_hold && self.try_edge_jump(ctx, &mut out) {
                self.last_reason = "seek_edge_jump";
                self.note_move_dir(&out);
                return out;
            }
            self.try_seek_vertical_walk(ctx, &mut out);
            self.ensure_locomotion(ctx, &mut out);
            self.strip_chase_toward_mob(ctx, &mut out);
            self.last_reason = "seek_walk";
            self.note_move_dir(&out);
            return out;
        }

        // 首台有击杀后：只近距捡，禁止远距拾取转向左右撕。
        let pickup_near_only = on_first && ctx.kills > 0;
        if self.try_pickup(ctx, &mut out, pickup_near_only) {
            self.ensure_locomotion(ctx, &mut out);
            self.strip_chase_toward_mob(ctx, &mut out);
            self.last_reason = if pickup_near_only {
                "normal_pickup_near"
            } else {
                "normal_pickup"
            };
            self.note_move_dir(&out);
            return out;
        }
        if !farm_uncleared && self.try_climb(ctx, &mut out) {
            self.resolve_blocked_horizontal(ctx, &mut out);
            self.last_reason = "normal_climb";
            self.note_move_dir(&out);
            return out;
        }
        if !farm_uncleared && self.try_step_up(ctx, &mut out) {
            self.resolve_blocked_horizontal(ctx, &mut out);
            self.last_reason = "normal_step_up";
            self.note_move_dir(&out);
            return out;
        }
        self.try_patrol(ctx, &mut out);
        self.ensure_locomotion(ctx, &mut out);
        self.strip_chase_toward_mob(ctx, &mut out);
        self.last_reason = "normal_patrol";
        self.note_move_dir(&out);
        out
    }

    fn note_move_dir(&mut self, out: &InputFrame) {
        let d = if out.right && !out.left {
            1.0
        } else if out.left && !out.right {
            -1.0
        } else {
            0.0
        };
        if d != 0.0 {
            if self.last_move_dir != 0.0 && d != self.last_move_dir {
                self.dir_flip_streak = self.dir_flip_streak.saturating_add(1);
            } else if d == self.last_move_dir {
                self.dir_flip_streak = 0;
            }
            self.last_move_dir = d;
        }
    }

    /// 同台近距有怪时禁止朝怪方向走。腾空/跳跃/攀爬时绝不禁走（否则换层跳会被撕掉）。
    fn strip_chase_toward_mob(&self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) {
        if !ctx.on_ground || out.jump || out.up || out.down {
            return;
        }
        let Some(e) = ctx.engage.filter(|e| e.dy.abs() <= SAME_PLATFORM_DY) else {
            return;
        };
        let on_first = !self.left_first_platform_layer(ctx);
        let farm_hold = ctx.farm_band_mobs || self.farm_layer_has_mobs(ctx);
        // 首台农怪未清：整段禁追（怪走过身后也不跟），避免左右抖。
        if e.dx.abs() > NO_CHASE_DX && !(on_first && farm_hold) {
            return;
        }
        let toward = e.toward_mob();
        if toward > 0.0 {
            out.right = false;
        } else if toward < 0.0 {
            out.left = false;
        }
    }

    /// 决策若为空按键则强制给一个位移，避免原地 noop 抖动发呆。
    fn ensure_locomotion(&mut self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) {
        // 已有方向但会被门控成 noop：改跳或掉头，禁止 right→noop 死循环。
        self.resolve_blocked_horizontal(ctx, out);

        // 站定挥砍 / 高台避险 / 近距等怪：禁止补走路。
        // 农怪带仍有怪但都在远处时必须能走动接近，不能 noop 站死。
        if !out.left && !out.right && !out.jump && !out.up && !out.down {
            if out.attack || self.perching {
                return;
            }
            let near_mob = ctx.sim_mob_in_melee
                || ctx
                    .engage
                    .filter(|e| e.dy.abs() <= 40.0 && e.dx.abs() <= STAND_WAIT_MAX)
                    .is_some();
            if near_mob && ctx.farm_band_mobs {
                return;
            }
        }
        if out.left || out.right || out.jump || out.up || out.down {
            return;
        }
        if !ctx.on_ground || ctx.climbing {
            return;
        }
        let dir = if self.patrol_dir >= 0.0 { 1.0 } else { -1.0 };
        if Self::can_walk_dir(ctx, dir) {
            set_move_dir(out, dir, false);
            return;
        }
        if Self::can_walk_dir(ctx, -dir) {
            self.patrol_dir = -dir;
            set_move_dir(out, -dir, false);
            return;
        }
        // 两侧都走不通：优先换层跳，绝不输出会被滤掉的方向键。
        if self.try_leave_edge(ctx, dir, out) || self.try_leave_edge(ctx, -dir, out) {
            return;
        }
    }

    /// 与 movement_gate.walk_allowed 对齐：物理悬崖或 YOLO 前方无地板 → 不可走。
    fn can_walk_dir(ctx: RuleBotCtx<'_>, dir: f32) -> bool {
        if dir == 0.0 {
            return false;
        }
        if Self::at_cliff(ctx, dir) {
            return false;
        }
        if ctx.on_ground
            && !ctx.climbing
            && obs_has_floor_signal(ctx.obs)
            && !obs_floor_ahead(ctx.obs, dir)
        {
            return false;
        }
        true
    }

    /// 当前水平意图若走不通：改成起跳离场或反向，避免 intent=right effective=noop。
    fn resolve_blocked_horizontal(&mut self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) {
        if out.jump || out.up || out.down || !ctx.on_ground || ctx.climbing {
            return;
        }
        let dir = if out.right && !out.left {
            1.0
        } else if out.left && !out.right {
            -1.0
        } else {
            return;
        };
        if Self::can_walk_dir(ctx, dir) {
            return;
        }
        out.left = false;
        out.right = false;
        // SeekVertical：贴边优先换层跳，避免左右来回抖。
        if self.explore_mode == ExploreMode::SeekVertical {
            if self.try_leave_edge(ctx, dir, out) {
                return;
            }
            if self.try_leave_edge(ctx, -dir, out) {
                return;
            }
        } else if self.try_leave_edge(ctx, dir, out) {
            return;
        }
        if Self::can_walk_dir(ctx, -dir) {
            self.patrol_dir = -dir;
            set_move_dir(out, -dir, false);
            return;
        }
        let _ = self.try_leave_edge(ctx, -dir, out);
    }

    /// 贴边换层：台阶跳 / 悬崖跳 / 可落点走出。
    fn try_leave_edge(
        &mut self,
        ctx: RuleBotCtx<'_>,
        dir: f32,
        out: &mut InputFrame,
    ) -> bool {
        if ctx.step_up_dx.is_some()
            && self.try_edge_jump_dir(ctx, dir, JumpPurpose::StepUp, out)
        {
            self.patrol_dir = dir;
            return true;
        }
        let purpose = if self.explore_mode == ExploreMode::SeekVertical
            || self.perching
            || !ctx.farm_band_mobs
        {
            JumpPurpose::PlatformChange
        } else {
            JumpPurpose::CliffCrossing
        };
        let phys_edge =
            Self::at_cliff(ctx, dir) || Self::physics_drop_ahead(ctx, dir);
        // YOLO 边沿单独不足以起跳；仅当反向也走不通时才用边沿跳脱身。
        let yolo_dead_end = obs_has_floor_signal(ctx.obs)
            && !obs_floor_ahead(ctx.obs, dir)
            && !Self::can_walk_dir(ctx, -dir);
        if phys_edge || yolo_dead_end {
            if self.try_edge_jump_dir(ctx, dir, purpose, out) {
                self.patrol_dir = dir;
                return true;
            }
        }
        false
    }

    fn tick_exploration(&mut self, x: f32, y: f32) {
        let key = visit_key(x, y);
        let y_band = key.1;
        if self.visited.insert(key) {
            self.ticks_without_new_cell = 0;
        } else {
            self.ticks_without_new_cell = self.ticks_without_new_cell.saturating_add(1);
        }

        if y_band != self.last_y_band {
            self.last_y_band = y_band;
            self.ticks_on_band = 0;
            // 登上新高度层：提交新农怪脚点，禁止「高台避险」把人拽回旧层。
            if self.farm_y <= 0.0 || (y - self.farm_y).abs() > 35.0 {
                self.farm_y = y;
            }
            self.perching = false;
            self.perch_ticks = 0;
            self.land_cooldown = 0;
            self.dir_flip_streak = 0;
            self.explore_mode = ExploreMode::Normal;
            self.explore_mode_ticks = 0;
        } else {
            self.ticks_on_band = self.ticks_on_band.saturating_add(1);
        }

        if self.explore_mode == ExploreMode::Normal
            && (self.ticks_on_band >= BAND_STAGNATION_DECISIONS
                || self.ticks_without_new_cell >= NO_NEW_CELL_DECISIONS)
        {
            self.explore_mode = ExploreMode::SeekVertical;
            self.explore_mode_ticks = 0;
        }

        if self.explore_mode == ExploreMode::SeekVertical {
            self.explore_mode_ticks = self.explore_mode_ticks.saturating_add(1);
            if self.rope_memory < EXPLORE_ROPE_BOOST {
                self.rope_memory = EXPLORE_ROPE_BOOST;
            }
        }
    }

    pub fn visited_cell_count(&self) -> usize {
        self.visited.len()
    }

    pub fn explore_seeking_vertical(&self) -> bool {
        self.explore_mode == ExploreMode::SeekVertical
    }

    pub fn dir_flip_streak_pub(&self) -> u32 {
        self.dir_flip_streak
    }

    fn tick_memory(&mut self, obs: &[f32; OBS_DIM]) {
        if obs_has_ladder_or_rope_signal(obs) {
            self.rope_memory = MEMORY_TICKS;
        } else if self.rope_memory > 0 {
            self.rope_memory -= 1;
        }
        if obs_has_drop(obs) {
            self.drop_memory = MEMORY_TICKS;
        } else if self.drop_memory > 0 {
            self.drop_memory -= 1;
        }
    }

    fn exiting_spawn_platform(&self, ctx: RuleBotCtx<'_>) -> bool {
        self.explore_mode == ExploreMode::SeekVertical && !self.left_first_platform_layer(ctx)
    }

    fn tick_stuck(&mut self, ctx: RuleBotCtx<'_>) {
        let x = ctx.player_x;
        let y = ctx.player_y;
        let exiting_spawn = self.exiting_spawn_platform(ctx);
        let moved = (x - self.last_x).abs() + (y - self.last_y).abs();
        if moved < STUCK_MOVE_EPS {
            self.stuck_ticks = self.stuck_ticks.saturating_add(1);
        } else {
            self.stuck_ticks = 0;
            if self.stuck_phase == StuckPhase::Normal {
                self.stuck_phase_ticks = 0;
            }
        }
        self.last_x = x;
        self.last_y = y;

        if self.stuck_phase != StuckPhase::Normal {
            self.stuck_phase_ticks = self.stuck_phase_ticks.saturating_add(1);
            if self.stuck_phase_ticks >= STUCK_REVERSE_TICKS {
                self.stuck_phase = StuckPhase::Normal;
                self.stuck_phase_ticks = 0;
                self.stuck_ticks = 0;
            }
        } else if self.stuck_ticks >= STUCK_TICKS {
            if exiting_spawn && ctx.player_x > 480.0 {
                self.stuck_ticks = 0;
                self.patrol_dir = -1.0;
            } else {
                self.stuck_phase = StuckPhase::Reverse;
                self.stuck_phase_ticks = 0;
                self.patrol_dir = -self.patrol_dir;
                self.stuck_ticks = 0;
            }
        }
    }

    fn try_stuck_recovery(&mut self, ctx: RuleBotCtx<'_>) -> Option<InputFrame> {
        if self.stuck_phase == StuckPhase::Normal {
            return None;
        }
        if self.exiting_spawn_platform(ctx) {
            return None;
        }
        let mut out = InputFrame::default();
        if self.try_climb(ctx, &mut out) {
            return Some(out);
        }
        let dir = self.patrol_dir.signum();
        if obs_has_enemy(ctx.obs) {
            steer_toward_slot(
                ctx.obs,
                OBS_ENEMY_START,
                OBS_ENEMY_SLOTS,
                &mut out,
                ctx.facing,
            );
        } else if dir > 0.0 {
            out.right = true;
        } else {
            out.left = true;
        }
        Some(out)
    }

    fn should_allow_jump(&self, ctx: RuleBotCtx<'_>, facing: f32, purpose: JumpPurpose) -> bool {
        if ctx.climbing {
            return purpose == JumpPurpose::ClimbEntry;
        }
        if !ctx.on_ground {
            return false;
        }
        match purpose {
            JumpPurpose::ClimbEntry => ctx.climb.is_some() || ctx.climbing,
            JumpPurpose::CliffCrossing => Self::at_cliff(ctx, facing),
            JumpPurpose::PlatformChange => {
                if self.explore_mode != ExploreMode::SeekVertical
                    && !self.perching
                    && ctx.farm_band_mobs
                {
                    return false;
                }
                // 物理悬崖/可落点；YOLO 边沿仅在死胡同（反向也不可走）时由 try_leave_edge 触发。
                Self::at_cliff(ctx, facing)
                    || Self::physics_drop_ahead(ctx, facing)
                    || (obs_has_floor_signal(ctx.obs)
                        && !obs_floor_ahead(ctx.obs, facing)
                        && !Self::can_walk_dir(ctx, -facing))
            }
            JumpPurpose::StepUp => ctx.step_up_dx.is_some() && ctx.on_ground,
        }
    }

    fn physics_drop_ahead(ctx: RuleBotCtx<'_>, dir: f32) -> bool {
        if dir > 0.0 {
            ctx.physics_drop_right == Some(true)
        } else if dir < 0.0 {
            ctx.physics_drop_left == Some(true)
        } else {
            false
        }
    }

    fn left_first_platform_layer(&self, ctx: RuleBotCtx<'_>) -> bool {
        visit_key(ctx.player_x, ctx.player_y).1 != self.spawn_y_band
            || ctx.player_y < self.spawn_y - 50.0
    }

    fn try_potion(&self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) -> bool {
        if ctx.potions == 0 || ctx.max_hp <= 0 {
            return false;
        }
        let ratio = ctx.hp as f32 / ctx.max_hp as f32;
        if ratio <= HP_POTION_RATIO {
            out.use_potion = true;
            return true;
        }
        false
    }

    fn try_pickup(&self, ctx: RuleBotCtx<'_>, out: &mut InputFrame, near_only: bool) -> bool {
        if obs_has_enemy(ctx.obs) && !near_only {
            return false;
        }
        let near = obs_drop_in_pickup_range(ctx.obs);
        if near_only {
            if !near {
                return false;
            }
            out.pick_up = true;
            return true;
        }
        let sees = obs_has_drop(ctx.obs) || self.drop_memory > 0;
        if !sees {
            return false;
        }
        out.pick_up = near || obs_has_drop(ctx.obs);
        steer_toward_slot(
            ctx.obs,
            OBS_DROP_START,
            OBS_DROP_SLOTS,
            out,
            ctx.facing,
        );
        true
    }

    fn at_cliff(ctx: RuleBotCtx<'_>, dir: f32) -> bool {
        if dir > 0.0 {
            ctx.physics_right_ok == Some(false)
        } else if dir < 0.0 {
            ctx.physics_left_ok == Some(false)
        } else {
            false
        }
    }

    fn try_combat(&mut self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) -> bool {
        // 腾空中绝不纠偏/落回，否则换层跳会被撕成往回走。
        if !ctx.on_ground {
            return false;
        }

        // 换层探索中：不做高台避险落回（那是农怪贴脸才用的）。
        let seeking = self.explore_mode == ExploreMode::SeekVertical;
        if !seeking && (self.perching || (self.farm_y > 0.0 && ctx.player_y < self.farm_y - 40.0))
        {
            self.perching = true;
            if self.should_drop_to_farm(ctx) {
                return self.try_drop_to_farm(ctx, out);
            }
            out.left = false;
            out.right = false;
            return true;
        }

        // 只认本台怪；邻台/其他高度当不存在。
        if !Self::same_platform_threat(ctx) {
            return false;
        }

        let engage = ctx.engage.filter(|e| e.dy.abs() <= SAME_PLATFORM_DY);

        let Some(engage) = engage else {
            if ctx.sim_mob_in_melee {
                out.left = false;
                out.right = false;
                out.attack = true;
                return true;
            }
            return false;
        };

        // SeekVertical：非 melee 不占用决策。
        if seeking && !ctx.sim_mob_in_melee {
            return false;
        }

        if ctx.sim_mob_in_melee {
            out.left = false;
            out.right = false;
            out.attack = true;
            return true;
        }

        let toward = engage.toward_mob();
        let dist = engage.dx.abs();
        let walking_away = engage.player_behind();
        let in_swing = dist <= STRIKE_HOLD_MAX;

        if in_swing {
            out.left = false;
            out.right = false;
            out.attack = true;
            if dist < TOUCH_AVOID_DX && self.land_cooldown == 0 && !seeking {
                if self.try_step_up(ctx, out) {
                    self.perching = true;
                    if !ctx.sim_mob_in_melee {
                        out.attack = false;
                    }
                    return true;
                }
                if toward != 0.0 && !Self::at_cliff(ctx, -toward) {
                    set_move_dir(out, -toward, false);
                    if !ctx.sim_mob_in_melee {
                        out.attack = false;
                    }
                }
            }
            return true;
        }

        if dist <= STAND_WAIT_MAX && (walking_away || engage.mob_approaching() || toward != 0.0)
        {
            out.left = false;
            out.right = false;
            return true;
        }

        false
    }

    /// 本台是否有可接战威胁（|dy|≤同台阈值）。邻台怪一律忽略。
    fn same_platform_threat(ctx: RuleBotCtx<'_>) -> bool {
        if ctx.sim_mob_in_melee {
            return true;
        }
        if ctx
            .engage
            .filter(|e| e.dy.abs() <= SAME_PLATFORM_DY)
            .is_some()
        {
            return true;
        }
        ctx.farm_band_mobs
    }

    fn update_perch_state(&mut self, ctx: RuleBotCtx<'_>) {
        // 换层探索中登上更高台：直接认新层，禁止进入避险态。
        if self.explore_mode == ExploreMode::SeekVertical
            && self.farm_y > 0.0
            && ctx.player_y < self.farm_y - 40.0
        {
            if ctx.on_ground {
                self.farm_y = ctx.player_y;
                self.perching = false;
                self.perch_ticks = 0;
                self.land_cooldown = 0;
            }
            return;
        }
        if self.farm_y > 0.0 && ctx.player_y < self.farm_y - 40.0 {
            // 仅农怪贴脸避险才进 perch；无本台怪时不要无高度差误触发。
            if ctx.farm_band_mobs || self.perching {
                self.perching = true;
                self.perch_ticks = self.perch_ticks.saturating_add(1);
            }
        } else if self.farm_y > 0.0 && ctx.player_y >= self.farm_y - 20.0 {
            if self.perching {
                self.land_cooldown = 50;
            }
            if self.perching && !ctx.farm_band_mobs {
                self.farm_y = ctx.player_y;
            }
            self.perching = false;
            self.perch_ticks = 0;
        }
    }

    fn farm_layer_has_mobs(&self, ctx: RuleBotCtx<'_>) -> bool {
        if ctx.farm_band_mobs {
            return true;
        }
        if let Some(e) = ctx.engage_wide {
            let mob_y = ctx.player_y + e.dy;
            if (mob_y - self.farm_y).abs() <= 55.0 && e.dx.abs() <= FARM_LOCAL_DX {
                return true;
            }
        }
        false
    }

    fn should_drop_to_farm(&self, ctx: RuleBotCtx<'_>) -> bool {
        if !ctx.farm_band_mobs && !self.farm_layer_has_mobs(ctx) {
            return true;
        }
        if self.perch_ticks >= 18 {
            return true;
        }
        let Some(e) = ctx.engage_wide else {
            return true;
        };
        let dist = e.dx.abs();
        if dist >= DROP_SAFE_DX {
            return true;
        }
        if e.player_behind() && dist > TOUCH_AVOID_DX + 8.0 {
            return true;
        }
        e.mob_approaching() && dist > STRIKE_HOLD_MAX
    }

    fn try_drop_to_farm(&mut self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) -> bool {
        let prefer = ctx
            .engage_wide
            .map(|e| {
                if e.dx.abs() > 1.0 {
                    -e.dx.signum()
                } else {
                    self.patrol_dir.signum()
                }
            })
            .unwrap_or_else(|| self.patrol_dir.signum());

        for dir in [prefer, -prefer] {
            if Self::physics_drop_ahead(ctx, dir) || Self::at_cliff(ctx, dir) {
                set_move_dir(out, dir, false);
                self.patrol_dir = dir;
                return true;
            }
            if Self::can_walk_dir(ctx, dir) {
                set_move_dir(out, dir, false);
                self.patrol_dir = dir;
                return true;
            }
        }
        set_move_dir(out, prefer, false);
        self.patrol_dir = prefer;
        true
    }

    /// 迎面站砍（保留给单测/回退路径）。
    #[allow(dead_code)]
    fn stand_and_strike_approaching(
        &self,
        ctx: RuleBotCtx<'_>,
        out: &mut InputFrame,
        dx: f32,
        toward: f32,
        dist: f32,
    ) {
        let need_face = (dx > 2.0 && ctx.facing <= 0.0) || (dx < -2.0 && ctx.facing >= 0.0);

        if dist > STRIKE_HOLD_MAX {
            out.left = false;
            out.right = false;
            return;
        }

        if dist < TOUCH_AVOID_DX {
            // 仍贴脸且未能上高台：只转身砍，尽量少平移。
            if need_face {
                set_move_dir(out, toward, false);
            } else {
                out.left = false;
                out.right = false;
            }
            out.attack = true;
            return;
        }

        if need_face {
            set_move_dir(out, toward, false);
        } else {
            out.left = false;
            out.right = false;
        }
        out.attack = true;
    }

    /// 转向怪并在可挥砍时攻击（无 sim 接战信息时的 YOLO 回退）。
    fn face_and_maybe_attack(
        &self,
        ctx: RuleBotCtx<'_>,
        out: &mut InputFrame,
        dx: f32,
        allow_attack: bool,
    ) {
        // 归一化 dx：换算到像素量级的相对远近仅作方向。
        let toward = if dx > CLIMB_ALIGN_OBS {
            1.0
        } else if dx < -CLIMB_ALIGN_OBS {
            -1.0
        } else {
            0.0
        };
        let need_turn = (toward > 0.0 && ctx.facing <= 0.0) || (toward < 0.0 && ctx.facing >= 0.0);
        if need_turn {
            set_move_dir(out, toward, false);
        } else if toward != 0.0 && ctx.sim_mob_in_melee {
            // 已在框内：站定
            out.left = false;
            out.right = false;
        } else if toward != 0.0 {
            set_move_dir(out, toward, false);
        }
        if !allow_attack {
            return;
        }
        let face = if out.right {
            1.0
        } else if out.left {
            -1.0
        } else if toward != 0.0 {
            toward
        } else {
            ctx.facing
        };
        if (obs_enemy_in_attack_range(ctx.obs, face)
            || obs_enemy_in_attack_range(ctx.obs, ctx.facing)
            || obs_assess_enemy_contact(ctx.obs).total > 0
            || ctx.sim_mob_in_melee)
            && (ctx.sim_mob_in_melee || obs_assess_enemy_contact(ctx.obs).total > 0)
        {
            out.attack = true;
        }
    }

    fn must_flee(&self, ctx: RuleBotCtx<'_>) -> bool {
        let contact = obs_assess_enemy_contact(ctx.obs);
        if contact.total == 0 {
            return false;
        }
        if is_surrounded(&contact) {
            return true;
        }
        let crowded = contact.left.max(contact.right) >= CONTACT_RETREAT_MIN;
        let hp_r = ctx.hp as f32 / ctx.max_hp.max(1) as f32;
        crowded && hp_r < HP_RETREAT_RATIO
    }

    fn should_retreat(&self, ctx: RuleBotCtx<'_>, contact: &EnemyContactAssessment) -> bool {
        if contact.total == 0 {
            return false;
        }
        if is_surrounded(contact) {
            return true;
        }
        let crowded = contact.left.max(contact.right);
        if crowded >= CONTACT_RETREAT_MIN {
            return true;
        }
        let hp_r = ctx.hp as f32 / ctx.max_hp.max(1) as f32;
        contact.total >= 1 && hp_r < HP_RETREAT_RATIO
    }

    fn try_flee(&mut self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) -> bool {
        if !obs_has_enemy(ctx.obs) {
            return false;
        }
        let contact = obs_assess_enemy_contact(ctx.obs);
        if contact.total == 0 {
            return false;
        }

        if is_surrounded(&contact) {
            return self.try_flee_surrounded(ctx, &contact, out);
        }
        if !self.should_retreat(ctx, &contact) {
            return false;
        }
        self.try_flee_crowded_side(ctx, &contact, out)
    }

    fn try_flee_surrounded(
        &mut self,
        ctx: RuleBotCtx<'_>,
        contact: &EnemyContactAssessment,
        out: &mut InputFrame,
    ) -> bool {
        if ctx.climbing || ctx.climb.is_some() {
            self.rope_memory = self.rope_memory.max(EXPLORE_ROPE_BOOST);
            return self.try_climb(ctx, out);
        }

        let escape = self.best_escape_dir(ctx, contact);
        if let Some(dir) = escape {
            self.patrol_dir = dir;
            if ctx.on_ground && !ctx.climbing {
                if Self::at_cliff(ctx, dir) {
                    return self.try_edge_jump_dir(ctx, dir, JumpPurpose::CliffCrossing, out);
                }
            }
            set_move_dir(out, dir, false);
            return true;
        }

        if ctx.on_ground && !ctx.climbing {
            for dir in [1.0_f32, -1.0] {
                if self.try_edge_jump_dir(ctx, dir, JumpPurpose::CliffCrossing, out) {
                    self.patrol_dir = dir;
                    return true;
                }
            }
            set_move_dir(out, self.patrol_dir.signum(), false);
            return true;
        }
        false
    }

    fn try_flee_crowded_side(
        &mut self,
        ctx: RuleBotCtx<'_>,
        contact: &EnemyContactAssessment,
        out: &mut InputFrame,
    ) -> bool {
        let Some(dir) = flee_dir_from_contact(contact) else {
            return self.try_flee_surrounded(ctx, contact, out);
        };
        self.patrol_dir = dir;

        if ctx.on_ground && !ctx.climbing {
            if Self::at_cliff(ctx, dir) {
                return self.try_edge_jump_dir(ctx, dir, JumpPurpose::CliffCrossing, out);
            }
            set_move_dir(out, dir, false);
            return true;
        }

        if ctx.climbing {
            out.down = true;
            return true;
        }
        false
    }

    fn best_escape_dir(&self, ctx: RuleBotCtx<'_>, contact: &EnemyContactAssessment) -> Option<f32> {
        let (cx, cy) = visit_key(ctx.player_x, ctx.player_y);
        let mut candidates: [(f32, i32); 2] = [(1.0, 0), (-1.0, 0)];
        for (dir, score) in &mut candidates {
            *score = 0;
            if *dir > 0.0 {
                *score -= (contact.right as i32) * 10;
                if !self.visited.contains(&(cx + 1, cy)) {
                    *score += 3;
                }
                if ctx.physics_right_ok == Some(false) {
                    *score -= 5;
                }
            } else {
                *score -= (contact.left as i32) * 10;
                if !self.visited.contains(&(cx - 1, cy)) {
                    *score += 3;
                }
                if ctx.physics_left_ok == Some(false) {
                    *score -= 5;
                }
            }
        }
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        if candidates[0].1 > candidates[1].1 {
            Some(candidates[0].0)
        } else if candidates[0].1 == candidates[1].1 && candidates[0].1 > i32::MIN / 2 {
            Some(candidates[0].0)
        } else {
            None
        }
    }

    fn try_climb(&mut self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) -> bool {
        // 农怪层未清 / 高台避险中：禁止新抓绳离场；已在绳上则继续直到落地。
        if !ctx.climbing && (ctx.farm_band_mobs || self.perching) {
            return false;
        }
        if ctx.climbing {
            out.up = true;
            return true;
        }

        // 只用紧邻当前层的绳/梯；YOLO 远处上层绳不驱动走位。
        let Some(climb) = ctx.climb else {
            self.climb_attempt_ticks = 0;
            return false;
        };

        self.climb_attempt_ticks = self.climb_attempt_ticks.saturating_add(1);
        if climb.dx.abs() > CLIMB_ALIGN_PX {
            set_move_dir(out, climb.dx.signum(), false);
            return true;
        }

        match climb.dir {
            ClimbDir::Up => {
                out.up = true;
                if ctx.on_ground {
                    out.jump = true;
                }
            }
            ClimbDir::Down => {
                out.down = true;
            }
        }
        true
    }

    /// 走向并跳上紧邻更高台阶（一层跳跃可达）。
    fn try_step_up(&mut self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) -> bool {
        let Some(dx) = ctx.step_up_dx else {
            return false;
        };
        let dir = if dx.abs() > 1.0 {
            dx.signum()
        } else {
            ctx.facing.signum()
        };
        if dx.abs() > 14.0 {
            if Self::can_walk_dir(ctx, dir) {
                set_move_dir(out, dir, false);
                self.patrol_dir = dir;
                return true;
            }
            // 朝台阶方向走不通：贴边起跳登上更高台。
            if self.try_leave_edge(ctx, dir, out) {
                return true;
            }
            return false;
        }
        // 已对准台阶：起跳。
        if !self.should_allow_jump(ctx, dir, JumpPurpose::StepUp) {
            return false;
        }
        out.jump = true;
        if dx.abs() > 2.0 {
            set_move_dir(out, dx.signum(), false);
        }
        true
    }

    fn try_edge_jump(&self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) -> bool {
        let facing = if self.patrol_dir >= 0.0 { 1.0 } else { -1.0 };
        self.try_edge_jump_dir(ctx, facing, JumpPurpose::PlatformChange, out)
    }

    fn try_edge_jump_dir(
        &self,
        ctx: RuleBotCtx<'_>,
        facing: f32,
        purpose: JumpPurpose,
        out: &mut InputFrame,
    ) -> bool {
        if !self.should_allow_jump(ctx, facing, purpose) {
            return false;
        }
        out.jump = true;
        if facing > 0.0 {
            out.right = true;
        } else {
            out.left = true;
        }
        true
    }

    fn prefer_patrol_dir(&self, ctx: RuleBotCtx<'_>) -> f32 {
        let (cx, cy) = visit_key(ctx.player_x, ctx.player_y);
        let left_key = (cx - 1, cy);
        let right_key = (cx + 1, cy);
        let left_new = !self.visited.contains(&left_key);
        let right_new = !self.visited.contains(&right_key);
        let can_left = Self::can_walk_dir(ctx, -1.0);
        let can_right = Self::can_walk_dir(ctx, 1.0);

        if can_right && right_new && !(can_left && left_new) {
            return 1.0;
        }
        if can_left && left_new && !(can_right && right_new) {
            return -1.0;
        }
        if can_right && right_new {
            return 1.0;
        }
        if can_left && left_new {
            return -1.0;
        }
        self.patrol_dir.signum()
    }

    fn try_patrol(&mut self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) {
        self.patrol_dir = self.prefer_patrol_dir(ctx);
        let dir = self.patrol_dir.signum();
        if !Self::can_walk_dir(ctx, dir) {
            // 先掉头，不要在普通巡逻时贴崖乱跳。
            if Self::can_walk_dir(ctx, -dir) {
                self.patrol_dir = -dir;
            } else if self.try_leave_edge(ctx, dir, out) || self.try_leave_edge(ctx, -dir, out)
            {
                return;
            } else {
                return;
            }
        }
        let dir = self.patrol_dir.signum();
        if Self::can_walk_dir(ctx, dir) {
            set_move_dir(out, dir, false);
        }
    }

    /// SeekVertical：优先紧邻绳梯 / 台阶；否则找可落边，不再追远处上层绳。
    fn try_seek_vertical_walk(&mut self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) {
        if let Some(climb) = ctx.climb {
            if climb.dx.abs() > CLIMB_ALIGN_PX {
                let dir = climb.dx.signum();
                if Self::can_walk_dir(ctx, dir) {
                    set_move_dir(out, dir, false);
                    self.patrol_dir = dir;
                } else if self.try_leave_edge(ctx, dir, out) {
                    return;
                } else if Self::can_walk_dir(ctx, -dir) {
                    self.patrol_dir = -dir;
                    set_move_dir(out, -dir, false);
                }
            } else {
                match climb.dir {
                    ClimbDir::Up => {
                        out.up = true;
                        if ctx.on_ground {
                            out.jump = true;
                        }
                    }
                    ClimbDir::Down => out.down = true,
                }
            }
            return;
        }

        if self.try_step_up(ctx, out) {
            return;
        }

        // 无紧邻绳梯/台阶：走向可落下缘换层；悬崖侧优先跳/反向。
        for dir in [self.patrol_dir.signum(), -self.patrol_dir.signum()] {
            if Self::physics_drop_ahead(ctx, dir) && Self::can_walk_dir(ctx, dir) {
                set_move_dir(out, dir, false);
                self.patrol_dir = dir;
                return;
            }
            if self.try_leave_edge(ctx, dir, out) {
                return;
            }
        }

        for dir in [-1.0_f32, 1.0] {
            if Self::can_walk_dir(ctx, dir) {
                self.patrol_dir = dir;
                set_move_dir(out, dir, false);
                return;
            }
        }

        // 双侧都堵：仍尝试换层跳，禁止输出会被门控成 noop 的方向键。
        let dir = self.patrol_dir.signum();
        if self.try_leave_edge(ctx, dir, out) || self.try_leave_edge(ctx, -dir, out) {
            return;
        }
        if Self::can_walk_dir(ctx, -dir) {
            self.patrol_dir = -dir;
            set_move_dir(out, -dir, false);
        }
    }
}

fn is_surrounded(c: &EnemyContactAssessment) -> bool {
    c.left >= SURROUNDED_SIDE_MIN
        && c.right >= SURROUNDED_SIDE_MIN
        && c.total >= 2
}

fn flee_dir_from_contact(c: &EnemyContactAssessment) -> Option<f32> {
    if c.left > c.right {
        Some(1.0)
    } else if c.right > c.left {
        Some(-1.0)
    } else {
        None
    }
}

fn set_move_dir(out: &mut InputFrame, dir: f32, jump: bool) {
    if dir > 0.0 {
        out.right = true;
    } else if dir < 0.0 {
        out.left = true;
    }
    if jump {
        out.jump = true;
    }
}

fn read_slot(values: &[f32], base: usize) -> Option<(f32, f32)> {
    if base + OBS_SLOT_DIM > values.len() {
        return None;
    }
    if values[base + 2].abs() <= 1e-4 && values[base + 3].abs() <= 1e-4 {
        return None;
    }
    Some((values[base], values[base + 1]))
}

fn nearest_same_level_enemy_dx(obs: &[f32; OBS_DIM]) -> Option<f32> {
    let mut best: Option<(f32, f32)> = None;
    for i in 0..OBS_ENEMY_SLOTS {
        let base = OBS_ENEMY_START + i * OBS_SLOT_DIM;
        let Some((dx, dy)) = read_slot(obs, base) else {
            continue;
        };
        if dy.abs() > super::observation::ENEMY_SAME_LEVEL_DY {
            continue;
        }
        let dist = dx.abs() + dy.abs() * 0.3;
        match best {
            None => best = Some((dist, dx)),
            Some((bd, _)) if dist < bd => best = Some((dist, dx)),
            _ => {}
        }
    }
    best.map(|(_, dx)| dx)
}

fn steer_toward_same_level_enemy(
    obs: &[f32; OBS_DIM],
    out: &mut InputFrame,
    facing: f32,
) {
    let mut best: Option<(f32, f32)> = None;
    for i in 0..OBS_ENEMY_SLOTS {
        let base = OBS_ENEMY_START + i * OBS_SLOT_DIM;
        let Some((dx, dy)) = read_slot(obs, base) else {
            continue;
        };
        if dy.abs() > super::observation::ENEMY_SAME_LEVEL_DY {
            continue;
        }
        let dist = dx.abs() + dy.abs() * 0.3;
        match best {
            None => best = Some((dist, dx)),
            Some((bd, _)) if dist < bd => best = Some((dist, dx)),
            _ => {}
        }
    }
    let Some((_, dx)) = best else {
        if facing >= 0.0 {
            out.right = true;
        } else {
            out.left = true;
        }
        return;
    };
    if dx.abs() <= CLIMB_ALIGN_OBS {
        if facing >= 0.0 {
            out.right = true;
        } else {
            out.left = true;
        }
    } else if dx > 0.0 {
        out.right = true;
    } else {
        out.left = true;
    }
}

fn steer_toward_slot(
    obs: &[f32; OBS_DIM],
    slot_start: usize,
    slot_count: usize,
    out: &mut InputFrame,
    facing: f32,
) {
    let mut best: Option<(f32, f32)> = None;
    for i in 0..slot_count {
        let base = slot_start + i * OBS_SLOT_DIM;
        let Some((dx, dy)) = read_slot(obs, base) else {
            continue;
        };
        let dist = dx.abs() + dy.abs() * 0.3;
        match best {
            None => best = Some((dist, dx)),
            Some((bd, _)) if dist < bd => best = Some((dist, dx)),
            _ => {}
        }
    }
    let Some((_, dx)) = best else {
        if facing >= 0.0 {
            out.right = true;
        } else {
            out.left = true;
        }
        return;
    };
    if dx.abs() <= CLIMB_ALIGN_OBS {
        if facing >= 0.0 {
            out.right = true;
        } else {
            out.left = true;
        }
    } else if dx > 0.0 {
        out.right = true;
    } else {
        out.left = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::observation::{inject_physics_walk_flags, OBS_FLOOR_START};

    #[test]
    fn patrol_moves_right_by_default() {
        let mut bot = RuleBot::default();
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 0.0,
            player_y: 0.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.right);
        assert!(!inp.left);
    }

    #[test]
    fn low_hp_uses_potion() {
        let mut bot = RuleBot::default();
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 20,
            max_hp: 100,
            potions: 3,
            player_x: 0.0,
            player_y: 0.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.use_potion);
    }

    #[test]
    fn patrol_avoids_cliff_by_turning_instead_of_jumping() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_FLOOR_START + 2] = 0.08;
        obs[OBS_FLOOR_START + 3] = 0.04;
        inject_physics_walk_flags(&mut obs, Some(false), Some(true));
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 100.0,
            player_y: 100.0,
            kills: 0,
            physics_right_ok: Some(false),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(!inp.jump, "normal patrol should turn away from cliff, not jump");
        assert!(inp.left);
        assert!(!inp.right);
    }

    #[test]
    fn patrol_prefers_unvisited_right() {
        let mut bot = RuleBot::default();
        bot.visited.insert(visit_key(0.0, 0.0));
        bot.visited.insert((1, 0));
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 80.0,
            player_y: 0.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.right);
        assert!(!inp.left);
    }

    #[test]
    fn band_stagnation_triggers_seek_vertical() {
        let mut bot = RuleBot::default();
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 50.0,
            player_y: 200.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        for _ in 0..=NO_NEW_CELL_DECISIONS {
            let _ = bot.decide(ctx);
        }
        assert!(bot.explore_seeking_vertical());
    }

    #[test]
    fn combat_stops_at_cliff_without_jumping() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        inject_physics_walk_flags(&mut obs, Some(false), Some(true));
        obs[OBS_ENEMY_START] = 0.12;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 430.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(false),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(!inp.jump, "chasing mob beyond cliff should not spam jump");
    }

    #[test]
    fn cleared_farm_at_yolo_right_edge_jumps_or_turns_not_noop_right() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1225.0;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = 1.0;
        let mut obs = [0.0_f32; OBS_DIM];
        // 有地板信号但右侧无地板（与门控 walk_allowed 一致）→ 不可再输出 right。
        obs[OBS_FLOOR_START] = -0.05;
        obs[OBS_FLOOR_START + 1] = 0.02;
        obs[OBS_FLOOR_START + 2] = 0.2;
        obs[OBS_FLOOR_START + 3] = 0.04;
        inject_physics_walk_flags(&mut obs, Some(true), Some(true));
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 720.0,
            player_y: 1225.0,
            kills: 3,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: Some(false),
            physics_drop_left: Some(false),
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: Some(-200.0),
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(
            !(inp.right && !inp.jump && !inp.left),
            "cleared farm: must not emit bare right into YOLO edge (becomes noop)"
        );
        assert!(
            inp.jump || inp.left || inp.up,
            "should jump/turn/climb toward leave, got {:?}",
            inp
        );
    }

    #[test]
    fn seek_vertical_jumps_at_cliff() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.last_y_band = visit_key(430.0, 1225.0).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_FLOOR_START + 2] = 8.0 / 1368.0;
        obs[OBS_FLOOR_START + 3] = 0.04;
        inject_physics_walk_flags(&mut obs, Some(false), Some(true));
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 430.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(false),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.jump, "seek vertical should jump on physics cliff");
        assert!(inp.right);
    }

    #[test]
    fn seek_vertical_does_not_jump_on_yolo_edge_alone() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.last_y_band = visit_key(430.0, 1225.0).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = 1.0;
        let mut obs = [0.0_f32; OBS_DIM];
        // 右侧 YOLO 无前方地板，但左侧有 → 应掉头，不因右侧边沿误跳。
        obs[OBS_FLOOR_START] = -0.08;
        obs[OBS_FLOOR_START + 1] = 0.02;
        obs[OBS_FLOOR_START + 2] = 0.25;
        obs[OBS_FLOOR_START + 3] = 0.04;
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 430.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(!inp.jump, "YOLO right-edge alone must turn, not jump spam");
        assert!(inp.left, "should reverse toward remaining floor");
    }

    #[test]
    fn retreat_when_touching_enemies_on_right_and_low_hp() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        set_touching_enemy(&mut obs, 0, 0.018, 0.0);
        set_touching_enemy(&mut obs, 1, 0.022, 0.01);
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 65,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.attack || inp.right,
            "two touching mobs at mid HP should still fight back, not only flee"
        );
    }

    #[test]
    fn retreat_when_many_touching_on_one_side_even_at_full_hp() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        set_touching_enemy(&mut obs, 0, 0.015, 0.0);
        set_touching_enemy(&mut obs, 1, 0.020, 0.0);
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.attack || inp.right,
            "full HP should fight crowded contact instead of fleeing"
        );
        assert!(!inp.jump);
    }

    #[test]
    fn seek_vertical_jump_not_reversed_by_same_level_mob() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1225.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(700.0, 1225.0).1;
        bot.last_y_band = bot.spawn_y_band;
        bot.explore_mode = ExploreMode::SeekVertical;
        let obs = [0.0_f32; OBS_DIM];
        // 右侧有同台怪：换层起跳时不得被禁追撕成 left。
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 700.0,
            player_y: 1225.0,
            kills: 3,
            physics_right_ok: Some(false),
            physics_left_ok: Some(true),
            physics_drop_right: Some(true),
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 100.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: Some(30.0),
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(
            bot.explore_seeking_vertical(),
            "must stay SeekVertical"
        );
        assert!(
            !inp.left || inp.jump || inp.right || inp.up,
            "must not solely walk left away from ledge: {inp:?}"
        );
        assert!(
            inp.jump || inp.right || inp.up,
            "should keep climbing/jumping toward next platform, got left-only dodge"
        );
    }

    #[test]
    fn local_farm_clear_seeks_vertical_despite_far_same_level_mob() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1225.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(416.0, 1225.0).1;
        bot.last_y_band = bot.spawn_y_band;
        bot.explore_mode = ExploreMode::Normal;
        let obs = [0.0_f32; OBS_DIM];
        // 本段已清（farm_band=false），但同层远处还有怪（footing 仍 true）。
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 416.0,
            player_y: 1225.0,
            kills: 3,
            physics_right_ok: Some(false),
            physics_left_ok: Some(true),
            physics_drop_right: Some(true),
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 900.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: Some(40.0),
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(
            bot.explore_seeking_vertical(),
            "far same-level mob must not block SeekVertical after local clear"
        );
        assert!(
            inp.jump || inp.right || inp.left || inp.up,
            "must try to leave first platform"
        );
    }

    #[test]
    fn far_farm_mob_allows_patrol_not_permanent_noop() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1225.0;
        let obs = [0.0_f32; OBS_DIM];
        // 同层远处怪 ~1000px：应巡逻接近，不能站死 noop。
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 416.0,
            player_y: 1225.0,
            kills: 3,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 1080.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: true,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.left || inp.right || inp.jump,
            "far farm mob must allow locomotion, got noop"
        );
        assert!(!inp.attack);
    }

    #[test]
    fn distant_mobs_on_left_still_chases_instead_of_fleeing() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        set_enemy_slot(&mut obs, 0, -0.08, 0.0);
        set_enemy_slot(&mut obs, 1, -0.10, 0.02);
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: -110.0,
                mob_dir: -1.0,
            }),
        engage_wide: None,
        climb: None,
        step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(!inp.left && !inp.right, "distant walking-away mobs: stand wait, no chase");
        assert!(!inp.jump);
    }

    #[test]
    fn single_touching_enemy_at_full_hp_still_attacks_in_range() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        set_touching_enemy(&mut obs, 0, 0.018, 0.0);
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.attack, "one touching mob at full hp should attack, not flee");
    }

    #[test]
    fn surrounded_jumps_when_cliffs_both_sides() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        set_touching_enemy(&mut obs, 0, -0.018, 0.0);
        set_touching_enemy(&mut obs, 1, 0.018, 0.0);
        inject_physics_walk_flags(&mut obs, Some(false), Some(false));
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 80,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(false),
            physics_left_ok: Some(false),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.jump, "surrounded with no walk escape should jump");
    }

    #[test]
    fn no_jump_when_chasing_enemy_on_flat() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        set_enemy_slot(&mut obs, 0, 0.08, 0.0);
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 90.0,
                mob_dir: -1.0,
            }),
        engage_wide: None,
        climb: None,
        step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(!inp.jump, "front approach on flat should not jump");
        assert!(
            inp.attack,
            "within strike band (~90): start chopping before touch"
        );
        assert!(
            !inp.left && !inp.right,
            "must stand chop, not chase"
        );
    }

    #[test]
    fn combat_attacks_left_side_before_touch() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: -70.0,
                mob_dir: 1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.attack, "left approaching mob in strike band must chop before touch");
        assert!(!inp.left && !inp.right);
    }

    #[test]
    fn cleared_first_farm_seeks_vertical_immediately() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1225.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(416.0, 1225.0).1;
        bot.last_y_band = bot.spawn_y_band;
        bot.explore_mode = ExploreMode::Normal;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 416.0,
            player_y: 1225.0,
            kills: 3,
            physics_right_ok: Some(false),
            physics_left_ok: Some(true),
            physics_drop_right: Some(true),
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: Some(20.0),
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(
            bot.explore_seeking_vertical(),
            "cleared farm must SeekVertical immediately"
        );
        assert!(
            inp.jump || inp.right || inp.left || inp.up,
            "cleared farm must not stand noop; got label move/jump"
        );
        assert!(!inp.attack);
    }

    #[test]
    fn seek_edge_jump_not_blocked_by_far_same_layer_mob() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1225.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(520.0, 1225.0).1;
        bot.last_y_band = bot.spawn_y_band;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = 1.0;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 520.0,
            player_y: 1225.0,
            kills: 2,
            physics_right_ok: Some(false),
            physics_left_ok: Some(true),
            physics_drop_right: Some(true),
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: -180.0,
                mob_dir: 1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(
            bot.explore_seeking_vertical(),
            "far same-layer mob must not cancel SeekVertical"
        );
        assert!(
            inp.jump,
            "cleared melee must edge-jump despite far engage; reason={}",
            bot.last_reason
        );
    }

    #[test]
    fn dir_flip_streak_forces_leave_jump() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1225.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(400.0, 1225.0).1;
        bot.last_y_band = bot.spawn_y_band;
        bot.explore_mode = ExploreMode::Normal;
        bot.last_move_dir = 1.0;
        bot.dir_flip_streak = 4;
        bot.patrol_dir = 1.0;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 2,
            physics_right_ok: Some(false),
            physics_left_ok: Some(false),
            physics_drop_right: Some(true),
            physics_drop_left: Some(true),
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 150.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: true,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.jump || bot.explore_seeking_vertical(),
            "flip streak must force leave; reason={} jump={} seek={}",
            bot.last_reason,
            inp.jump,
            bot.explore_seeking_vertical()
        );
    }

    #[test]
    fn no_jump_when_fleeing_on_flat() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        set_touching_enemy(&mut obs, 0, 0.015, 0.0);
        set_touching_enemy(&mut obs, 1, 0.020, 0.0);
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 25,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.left, "critical HP + crowded should flee");
        assert!(!inp.jump, "flee on flat ground should not jump");
    }

    #[test]
    fn no_jump_on_flat_patrol() {
        let mut bot = RuleBot::default();
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 100.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.right);
        assert!(!inp.jump);
    }

    #[test]
    fn ignores_enemy_on_lower_platform() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        set_enemy_slot(&mut obs, 0, 0.05, 0.45);
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 700.0,
            player_y: 805.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(!inp.attack, "should not attack mob on much lower platform");
        assert!(
            inp.jump || inp.left || inp.right || inp.up || inp.down,
            "should seek vertical exit instead of air-attacking: {:?}",
            inp
        );
    }

    fn set_enemy_slot(obs: &mut [f32; OBS_DIM], idx: usize, dx: f32, dy: f32) {
        let base = OBS_ENEMY_START + idx * OBS_SLOT_DIM;
        obs[base] = dx;
        obs[base + 1] = dy;
        obs[base + 2] = 0.05;
        obs[base + 3] = 0.07;
    }

    fn set_touching_enemy(obs: &mut [f32; OBS_DIM], idx: usize, dx: f32, dy: f32) {
        set_enemy_slot(obs, idx, dx, dy);
    }

    #[test]
    fn seek_vertical_does_not_emit_empty_noop_when_blocked() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = bot.spawn_y_band;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = 1.0;
        let mut obs = [0.0_f32; OBS_DIM];
        // 有地板信号但前方无地板 → can_walk_dir 两边都 false
        obs[OBS_FLOOR_START + 2] = 0.3;
        obs[OBS_FLOOR_START + 3] = 0.04;
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 500.0,
            player_y: 1225.0,
            kills: 1,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.left || inp.right || inp.jump,
            "must not freeze as noop when walk checks fail"
        );
    }

    #[test]
    fn seek_vertical_prefers_combat_over_jump_when_attackable() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(430.0, 1225.0).1;
        bot.last_y_band = bot.spawn_y_band;
        bot.explore_mode = ExploreMode::SeekVertical;
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_ENEMY_START] = 0.08;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 520.0,
            player_y: 1225.0,
            kills: 1,
            physics_right_ok: Some(false),
            physics_left_ok: Some(true),
            physics_drop_right: Some(true),
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(!inp.jump, "should fight instead of cliff-jumping over attackable mobs");
        assert!(inp.attack || inp.right || inp.left);
    }

    #[test]
    fn seek_landing_on_higher_platform_does_not_drop_back() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1225.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(735.0, 1225.0).1;
        bot.last_y_band = bot.spawn_y_band;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = 1.0;
        let obs = [0.0_f32; OBS_DIM];
        // 刚跳上二台落地：高度带变化后应认新 farm_y，禁止左走落回。
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 741.0,
            player_y: 1155.0,
            kills: 3,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: Some(true),
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: Some(crate::game::EngageHint {
                dy: 70.0,
                dx: 120.0,
                mob_dir: -1.0,
            }),
            engage_wide: Some(crate::game::EngageHint {
                dy: 70.0,
                dx: 120.0,
                mob_dir: -1.0,
            }),
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(
            !bot.perching,
            "must not enter perch after intentional platform change"
        );
        assert!(
            (bot.farm_y - 1155.0).abs() < 1.0,
            "farm_y should commit to new platform, got {}",
            bot.farm_y
        );
        assert_ne!(
            bot.last_reason, "seek_perch",
            "must not drop-back via seek_perch"
        );
    }

    #[test]
    fn airborne_seek_ignores_off_platform_combat_steer() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1225.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(735.0, 1225.0).1;
        bot.last_y_band = bot.spawn_y_band;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = 1.0;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: false,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 741.0,
            player_y: 1171.0,
            kills: 3,
            physics_right_ok: None,
            physics_left_ok: None,
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: -6.0,
                dx: 117.0,
                mob_dir: 1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(
            !inp.left,
            "airborne must not steer left toward/away off-path mobs; reason={}",
            bot.last_reason
        );
    }

    #[test]
    fn combat_takes_step_up_refuge_when_too_close() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1225.0;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 24.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: Some(-80.0),
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.left, "should walk toward left ledge for refuge");
        assert!(!inp.attack, "do not swing while escaping to perch");
        assert!(bot.perching || inp.left);
    }

    #[test]
    fn combat_waits_when_mob_walking_away_no_chase() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        let obs = [0.0_f32; OBS_DIM];
        // 怪在右侧并向右走：玩家在背后 → 站桩等折返，禁止追赶/跳跃超车。
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 120.0,
                mob_dir: 1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(!inp.jump, "must not leap to overtake");
        assert!(!inp.left && !inp.right, "must stand and wait for turnaround");
        assert!(!inp.attack, "too far while mob walks away");
    }

    #[test]
    fn combat_attacks_when_touching_and_in_melee() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_ENEMY_START] = 0.02;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 28.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.attack, "in melee hitbox: chop instead of only kiting");
        assert!(!inp.right);
    }

    #[test]
    fn combat_kites_when_touching_but_not_in_melee() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 28.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.left, "touching but not in hitbox → create space");
        assert!(!inp.attack, "do not spam attack that gate will noop");
    }

    #[test]
    fn combat_stands_and_attacks_in_hold_band() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_ENEMY_START] = 0.03;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
        // 怪在右侧、向左走：玩家在正面，距离在站砍带。
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 50.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.attack, "hold band should continuously attack");
        assert!(
            !inp.left && !inp.right,
            "must stand still, not chase left/right"
        );
    }

    #[test]
    fn seek_vertical_walks_toward_step_up() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.last_y_band = visit_key(416.0, 1225.0).1;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = bot.last_y_band;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 416.0,
            player_y: 1225.0,
            kills: 3,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: Some(-120.0),
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.left, "should walk toward left step-up ledge");
        assert!(!inp.jump, "not yet aligned; no blind jump");
    }

    #[test]
    fn combat_faces_and_attacks_when_in_front() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_ENEMY_START] = 0.03;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
        // 怪在右侧但向左走：玩家在正面
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: -1.0,
            on_ground: true,
            climbing: false,
            hp: 80,
            max_hp: 100,
            potions: 0,
            player_x: 500.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 40.0,
                mob_dir: -1.0,
            }),
        engage_wide: None,
        climb: None,
        step_up_dx: None,
        farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.attack, "front engage should attack");
        assert!(
            !inp.left && !inp.right,
            "in swing range: stand-chop only; facing is applied by sim on attack"
        );
        assert!(!inp.jump);
    }

    #[test]
    fn combat_does_not_chase_when_mob_on_left_while_facing_right() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: -40.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
        };
        let inp = bot.decide(ctx);
        assert!(inp.attack);
        assert!(
            !inp.left && !inp.right,
            "must not follow mob to the left with direction keys"
        );
    }

    #[test]
    fn reset_clears_visit_memory() {
        let mut bot = RuleBot::default();
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 10.0,
            player_y: 10.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: true,
            sim_mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
        farm_band_mobs: false,
        };
        let _ = bot.decide(ctx);
        assert!(bot.visited_cell_count() >= 1);
        bot.reset();
        assert_eq!(bot.visited_cell_count(), 0);
        assert!(!bot.explore_seeking_vertical());
    }

    #[test]
    fn oracle_farm_first_platform_gets_kills() {
        use crate::game::load_default_map;
        use crate::game::observation_from_sim;
        use crate::game::GameSim;

        let map = load_default_map().expect("map");
        let mut sim = GameSim::new_preview(map, 0);
        let mut bot = RuleBot::default();
        let mut attacks = 0u32;
        let mut perch_ticks = 0u32;
        for _ in 0..3_600 {
            if sim.is_episode_over() {
                break;
            }
            let obs = observation_from_sim(&sim);
            let ctx = RuleBotCtx::from_sim_with_farm_y(&sim, &obs, bot.farm_y);
            if bot.perching {
                perch_ticks += 1;
            }
            let inp = bot.decide(ctx);
            if inp.attack {
                attacks += 1;
            }
            sim.tick(&inp);
        }
        eprintln!(
            "oracle farm: kills={} attacks={} perch_ticks={} end=({:.0},{:.0}) farm_y={:.0} mobs={}",
            sim.state.kills,
            attacks,
            perch_ticks,
            sim.state.player.x,
            sim.state.player.y,
            bot.farm_y,
            sim.state.mobs.iter().filter(|m| m.alive).count()
        );
        assert!(
            sim.state.kills >= 1,
            "oracle should get at least 1 kill on first platform"
        );
    }

    #[test]
    fn oracle_chops_approaching_mob_before_many_touches() {
        use crate::game::load_default_map;
        use crate::game::observation_from_sim;
        use crate::game::GameSim;

        let map = load_default_map().expect("map");
        let mut sim = GameSim::new_preview(map, 0);
        let px = sim.state.player.x;
        let py = sim.state.player.y;
        let Some(m) = sim.state.mobs.iter_mut().find(|m| m.alive) else {
            panic!("preview should spawn mobs");
        };
        m.x = px + 100.0;
        m.y = py;
        m.hp = 45;
        m.max_hp = 45;
        m.vx = -70.0;
        m.walk_x1 = px - 200.0;
        m.walk_x2 = px + 200.0;
        let keep_id = m.mob_id;
        for m in sim.state.mobs.iter_mut() {
            if m.mob_id != keep_id {
                m.alive = false;
            }
        }

        let mut bot = RuleBot::default();
        let mut attacks_before_touch = 0u32;
        let mut saw_attack_in_band = false;
        for _ in 0..600 {
            if let Some(m) = sim.state.mobs.iter_mut().find(|m| m.alive) {
                m.vx = -70.0;
                m.y = py;
            }
            let obs = observation_from_sim(&sim);
            let ctx = RuleBotCtx::from_sim_with_farm_y(&sim, &obs, bot.farm_y);
            let mob_dx = sim
                .state
                .mobs
                .iter()
                .find(|m| m.alive)
                .map(|m| m.x - sim.state.player.x);
            let inp = bot.decide(ctx);
            if let Some(dx) = mob_dx {
                if dx.abs() <= 90.0 && dx.abs() > 28.0 && inp.attack {
                    saw_attack_in_band = true;
                }
            }
            if sim.state.touch_hits == 0 && inp.attack {
                attacks_before_touch += 1;
            }
            sim.tick(&inp);
            if !sim.state.mobs.iter().any(|m| m.alive) {
                break;
            }
        }
        eprintln!(
            "oracle pre-touch chop: attacks_before_touch={} saw_in_band={} touches={} kills={} alive={}",
            attacks_before_touch,
            saw_attack_in_band,
            sim.state.touch_hits,
            sim.state.kills,
            sim.state.mobs.iter().filter(|m| m.alive).count()
        );
        assert!(
            saw_attack_in_band || attacks_before_touch > 0,
            "must start attacking while mob is in strike band before/without relying on touch"
        );
        assert!(
            sim.state.kills >= 1 || sim.state.touch_hits <= 2,
            "should kill or at most take a couple touches: kills={} touches={}",
            sim.state.kills,
            sim.state.touch_hits
        );
    }

    #[test]
    fn oracle_leaves_first_platform_after_farm_cleared() {
        use crate::game::load_default_map;
        use crate::game::observation_from_sim;
        use crate::game::GameSim;

        let map = load_default_map().expect("map");
        let mut sim = GameSim::new_preview(map, 0);
        let spawn_y = sim.state.player.y;
        // 清掉首层所有怪，验证会 SeekVertical / 换高度，而不是 noop 站死。
        for m in sim.state.mobs.iter_mut() {
            m.alive = false;
            m.die_t = 0.0;
        }
        sim.state.mobs.clear();
        sim.state.kills = 5;

        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = spawn_y;
        bot.spawn_y = spawn_y;
        bot.spawn_y_band = visit_key(sim.state.player.x, spawn_y).1;
        bot.last_y_band = bot.spawn_y_band;
        bot.explore_mode = ExploreMode::Normal;

        let mut moved = false;
        let mut sought = false;
        for _ in 0..600 {
            let obs = observation_from_sim(&sim);
            let ctx = RuleBotCtx::from_sim_with_farm_y(&sim, &obs, bot.farm_y);
            let inp = bot.decide(ctx);
            if bot.explore_seeking_vertical() {
                sought = true;
            }
            if inp.jump || inp.up || inp.left || inp.right {
                moved = true;
            }
            sim.tick(&inp);
            if (sim.state.player.y - spawn_y).abs() > 40.0 {
                break;
            }
        }
        eprintln!(
            "oracle leave farm: sought={} moved={} end=({:.0},{:.0}) spawn_y={:.0} mobs={}",
            sought,
            moved,
            sim.state.player.x,
            sim.state.player.y,
            spawn_y,
            sim.state.mobs.len()
        );
        assert!(sought, "cleared farm must enter SeekVertical");
        assert!(moved, "cleared farm must not stand noop");
    }

    #[test]
    fn oracle_does_not_chase_mob_crossing_to_the_left() {
        use crate::game::load_default_map;
        use crate::game::observation_from_sim;
        use crate::game::GameSim;

        let map = load_default_map().expect("map");
        let mut sim = GameSim::new_preview(map, 0);
        let px = sim.state.player.x;
        let py = sim.state.player.y;
        // 复用已刷怪：放到玩家右侧并强制向左走。
        let Some(m) = sim.state.mobs.iter_mut().find(|m| m.alive) else {
            panic!("preview should spawn mobs");
        };
        m.x = px + 120.0;
        m.y = py;
        m.hp = 200;
        m.max_hp = 200;
        m.vx = -80.0;
        m.walk_x1 = px - 220.0;
        m.walk_x2 = px + 220.0;
        let keep_id = m.mob_id;
        for m in sim.state.mobs.iter_mut() {
            if m.mob_id != keep_id {
                m.alive = false;
            }
        }

        let mut bot = RuleBot::default();
        let mut chase_left_when_mob_left = 0u32;
        let mut decisions_mob_on_left = 0u32;
        for _ in 0..900 {
            if let Some(m) = sim.state.mobs.iter_mut().find(|m| m.alive) {
                m.vx = -80.0;
                m.y = py;
            }
            let obs = observation_from_sim(&sim);
            let ctx = RuleBotCtx::from_sim_with_farm_y(&sim, &obs, bot.farm_y);
            let mob_dx = sim
                .state
                .mobs
                .iter()
                .find(|m| m.alive)
                .map(|m| m.x - sim.state.player.x)
                .unwrap_or(0.0);
            let inp = bot.decide(ctx);
            if mob_dx < -8.0 {
                decisions_mob_on_left += 1;
                if inp.left && !inp.right {
                    chase_left_when_mob_left += 1;
                }
            }
            sim.tick(&inp);
        }

        eprintln!(
            "repro left-chase: chase_left={}/{} touch_hits={} kills={} end_hp={}",
            chase_left_when_mob_left,
            decisions_mob_on_left,
            sim.state.touch_hits,
            sim.state.kills,
            sim.state.player.hp
        );
        assert!(
            decisions_mob_on_left > 30,
            "scenario should observe mob on the left"
        );
        assert!(
            chase_left_when_mob_left * 5 < decisions_mob_on_left,
            "bot must not chase left after mob: chase={chase_left_when_mob_left} left_obs={decisions_mob_on_left}"
        );
        assert!(
            sim.state.touch_hits <= 3,
            "touch hits too high while mob crosses left: {}",
            sim.state.touch_hits
        );
    }

    #[test]
    fn strip_chase_blocks_left_toward_mob_on_left() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1225.0;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            hp: 100,
            max_hp: 100,
            potions: 0,
            player_x: 400.0,
            player_y: 1225.0,
            kills: 0,
            physics_right_ok: Some(true),
            physics_left_ok: Some(true),
            physics_drop_right: None,
            physics_drop_left: None,
            sim_mob_in_melee: false,
            sim_mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: -90.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: true,
        };
        let inp = bot.decide(ctx);
        assert!(
            !inp.left,
            "patrol/seek must not walk left toward distant mob on the left"
        );
    }
}
