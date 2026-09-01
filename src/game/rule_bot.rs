//! 纯规则自动玩：仅 YOLO 检测 + OCR 自身定位 → 观测向量 → 输入帧。
//! 决策不得读取 GameSim 真值；`RuleBotCtx::from_vision` 是唯一生产路径。

use std::collections::HashSet;

use super::input::InputFrame;
use super::map::{ClimbDir, ClimbHint};
use super::observation::{
    obs_assess_enemy_contact, obs_climb_grab_ready, obs_climb_hint, obs_drop_in_pickup_range,
    obs_enemy_in_attack_range, obs_enemy_in_attack_range_platform, obs_farm_band_enemies,
    obs_floor_ahead, obs_floor_ahead_connected, obs_floor_drop_ahead, obs_floor_underfoot,
    obs_has_drop, obs_has_enemy, obs_has_floor_signal, obs_has_ladder_or_rope_signal,
    obs_nearest_enemy_wide_px, obs_nearest_same_level_enemy_px,
    obs_step_up_dx, obs_vertical_nav_allowed, EnemyContactAssessment, OBS_DIM, OBS_DROP_SLOTS,
    OBS_DROP_START, OBS_ENEMY_SLOTS, OBS_ENEMY_START, OBS_SLOT_DIM,
};
use super::sim::EngageHint;
use super::types::{WINDOW_H, WINDOW_W};
use super::visual_progress::{
    location_node, LocationNode, LoopKind, VisualMotionEstimator, VisualProgressMonitor,
};

const MEMORY_TICKS: u32 = 72;
const EXPLORE_ROPE_BOOST: u32 = 36;
const CLIMB_ALIGN_OBS: f32 = 0.015;
/// 绳梯水平对齐（地图像素）。
const CLIMB_ALIGN_PX: f32 = 12.0;
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
/// YOLO 农怪带闪断粘性（决策帧）：避免误判已清而过早 SeekVertical。
const FARM_BAND_STICKY: u32 = 24;
/// 台阶水平距小于此值时直接起跳（避免 vision 可走、sim 门控剥掉走路）。
const STEP_UP_JUMP_DX: f32 = 48.0;
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
/// Seek 局部循环：N 帧无新格子则打破（小于 BAND_STAGNATION 以提前脱困）。
const SEEK_LOOP_BREAK_CELLS: u32 = 10;
/// 刚爬过的绳/梯：禁止立刻反向再走（平台+梯子闭环主因）。
const CLIMB_TRANSIT_BAN_TICKS: u32 = 60;
const CLIMB_BAN_X_SLACK: f32 = 72.0;
const ESCAPE_PROBE_TICKS: u32 = 6;
const ESCAPE_SEEK_TICKS: u32 = 10;
const ESCAPE_FORCE_TICKS: u32 = 8;
const ESCAPE_ALT_TICKS: u32 = 10;
/// 视觉置信度低于此值时不写入访问网格（防 est_pos 跳变伪造新格子）。
const VISIT_MIN_CONF: u8 = 3;
/// 方向抖动 + 无新格子 → 强制脱困。
const FLIP_ESCAPE_FLIPS: u32 = 3;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EscapePhase {
    Normal,
    ProbeHorizontal,
    SeekExit,
    ForceVertical,
    AlternateExit,
}

/// 规则 bot 状态（跨帧记忆）。
#[derive(Debug, Clone)]
pub struct RuleBot {
    patrol_dir: f32,
    last_x: f32,
    last_y: f32,
    rope_memory: u32,
    drop_memory: u32,
    climb_attempt_ticks: u32,
    initialized: bool,
    visited: HashSet<(i32, i32)>,
    /// 已踏足的高度带（y_band）；Seek 优先走向未访问带。
    visited_y_bands: HashSet<i32>,
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
    /// 农怪带观测粘性：YOLO 漏检时仍视为本段有怪。
    farm_band_sticky: u32,
    /// 连续攀爬按 up 的决策帧；绳顶无脚下时靠 step_up 脱身。
    climb_up_stall: u32,
    /// 最近换层所用绳/梯的世界 x；在 ban_ticks 内禁止再抓/再落这根。
    climb_ban_x: f32,
    climb_ban_ticks: u32,
    /// 上一帧是否在绳/梯上；用于检测攀爬结束并写入 ban。
    was_climbing: bool,
    climb_start_x: f32,
    climb_start_y: f32,
    progress: VisualProgressMonitor,
    escape_phase: EscapePhase,
    escape_ticks: u32,
    escape_attempt: u8,
    escape_dir: f32,
    escape_anchor_x: f32,
    escape_anchor_y: f32,
    failed_exits: u8,
    last_commanded_locomotion: bool,
}

impl Default for RuleBot {
    fn default() -> Self {
        Self {
            patrol_dir: 1.0,
            last_x: 0.0,
            last_y: 0.0,
            rope_memory: 0,
            drop_memory: 0,
            climb_attempt_ticks: 0,
            initialized: false,
            visited: HashSet::new(),
            visited_y_bands: HashSet::new(),
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
            farm_band_sticky: 0,
            climb_up_stall: 0,
            climb_ban_x: 0.0,
            climb_ban_ticks: 0,
            was_climbing: false,
            climb_start_x: 0.0,
            climb_start_y: 0.0,
            progress: VisualProgressMonitor::default(),
            escape_phase: EscapePhase::Normal,
            escape_ticks: 0,
            escape_attempt: 0,
            escape_dir: 1.0,
            escape_anchor_x: 0.0,
            escape_anchor_y: 0.0,
            failed_exits: 0,
            last_commanded_locomotion: false,
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
    pub player_x: f32,
    pub player_y: f32,
    pub walk_right_ok: Option<bool>,
    pub walk_left_ok: Option<bool>,
    pub drop_ahead_right: Option<bool>,
    pub drop_ahead_left: Option<bool>,
    pub mob_in_melee: bool,
    pub mob_on_attackable_footing: bool,
    pub engage: Option<super::EngageHint>,
    /// 含脚下低一层怪的宽接战信息（高台避险用）。
    pub engage_wide: Option<super::EngageHint>,
    /// 紧邻当前层的绳/梯（YOLO 槽位推断）。
    pub climb: Option<ClimbHint>,
    /// 可跳上一层台阶的相对 dx。
    pub step_up_dx: Option<f32>,
    /// 农怪高度带是否还有活怪（清层/禁止提前离场）。
    pub farm_band_mobs: bool,
    /// 本帧视觉里程计匹配置信度（地标数）。
    pub visual_confidence: u8,
}

impl<'a> RuleBotCtx<'a> {
    #[cfg(test)]
    pub fn from_sim(sim: &super::GameSim, obs: &'a [f32; OBS_DIM]) -> Self {
        Self::from_sim_with_farm_y(sim, obs, sim.state.player.y)
    }

    #[cfg(test)]
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
            player_x: p.x,
            player_y: p.y,
            walk_right_ok: pr,
            walk_left_ok: pl,
            drop_ahead_right: pdr,
            drop_ahead_left: pdl,
            mob_in_melee: sim.mob_in_strike_band(),
            mob_on_attackable_footing: sim.mob_on_attackable_footing(),
            engage: sim.nearest_engage_hint(),
            engage_wide: sim.nearest_engage_hint_wide(),
            climb: sim.nearest_adjacent_climb(),
            step_up_dx: sim.nearest_step_up_dx(),
            farm_band_mobs: farm_y > 0.0 && sim.mobs_near_xy(farm_y, 55.0, p.x, FARM_LOCAL_DX),
            visual_confidence: 11,
        }
    }

    /// 纯视觉决策：不读 GameSim 真值。粘性状态由 `VisionSenseState` 维护。
    pub fn from_vision(obs: &'a [f32; OBS_DIM], sense: &VisionSenseState) -> Self {
        let iw = WINDOW_W as f32;
        let ih = WINDOW_H as f32;
        let right_ok = obs_floor_ahead_connected(obs, 1.0);
        let left_ok = obs_floor_ahead_connected(obs, -1.0);
        let engage = obs_nearest_same_level_enemy_px(obs, iw, ih).map(|(dx, dy)| EngageHint {
            dx,
            dy,
            mob_dir: if dx.abs() > 1.0 {
                -dx.signum()
            } else {
                sense.facing.signum()
            },
        });
        let engage = engage.filter(|e| e.dy.abs() <= SAME_PLATFORM_DY);
        let engage_wide =
            obs_nearest_enemy_wide_px(obs, iw, ih, 140.0).map(|(dx, dy)| EngageHint {
                dx,
                dy,
                mob_dir: if dx.abs() > 1.0 {
                    -dx.signum()
                } else {
                    sense.facing.signum()
                },
            });
        let in_melee = obs_enemy_in_attack_range_platform(obs, sense.facing)
            || engage.filter(|e| e.dx.abs() <= STRIKE_HOLD_MAX).is_some();
        let under = obs_floor_underfoot(obs);
        // 脚下有台则本帧视为已落地（即使攀爬粘性尚未清），否则绳顶小台会永远 !on_ground 死按 up。
        let climbing = sense.climbing && !under;
        let on_ground = under && !climbing;
        Self {
            obs,
            facing: sense.facing,
            on_ground,
            climbing,
            player_x: sense.est_x,
            player_y: sense.est_y,
            walk_right_ok: Some(right_ok),
            walk_left_ok: Some(left_ok),
            drop_ahead_right: Some(obs_floor_drop_ahead(obs, 1.0)),
            drop_ahead_left: Some(obs_floor_drop_ahead(obs, -1.0)),
            mob_in_melee: in_melee,
            mob_on_attackable_footing: engage.is_some(),
            engage,
            engage_wide,
            climb: obs_climb_hint(obs, iw, ih),
            step_up_dx: obs_step_up_dx(obs, iw, ih),
            farm_band_mobs: obs_farm_band_enemies(obs, iw, FARM_LOCAL_DX),
            visual_confidence: sense.visual_confidence(),
        }
    }
}

/// 纯视觉闭环的粘性感知（朝向、攀爬、推算坐标）。
#[derive(Debug, Clone)]
pub struct VisionSenseState {
    pub facing: f32,
    pub climbing: bool,
    pub est_x: f32,
    pub est_y: f32,
    initialized: bool,
    /// 连续「脚下有地板且远离绳」决策帧；满额才清攀爬粘性。
    climb_ground_release: u32,
    motion: VisualMotionEstimator,
}

impl Default for VisionSenseState {
    fn default() -> Self {
        Self {
            facing: 1.0,
            climbing: false,
            est_x: 0.0,
            est_y: 0.0,
            initialized: false,
            climb_ground_release: 0,
            motion: VisualMotionEstimator::default(),
        }
    }
}

impl VisionSenseState {
    /// 每帧观测到达时：更新攀爬粘性与视觉里程计。
    pub fn prepare(&mut self, obs: &[f32; OBS_DIM]) {
        self.motion.update(obs);
        self.est_x = self.motion.x;
        self.est_y = self.motion.y;
        self.initialized = true;

        let near_climb = obs_vertical_nav_allowed(obs, false);
        let under = obs_floor_underfoot(obs);
        if self.climbing {
            // 绳顶小台：绳仍在视野（near_climb）也要下绳；纯误检脚下则要求可走/台阶。
            let can_land = under
                && (!near_climb
                    || obs_step_up_dx(obs, WINDOW_W as f32, WINDOW_H as f32).is_some()
                    || obs_floor_ahead_connected(obs, 1.0)
                    || obs_floor_ahead_connected(obs, -1.0));
            if can_land {
                self.climb_ground_release = self.climb_ground_release.saturating_add(1);
                if self.climb_ground_release >= 2 {
                    self.climbing = false;
                    self.climb_ground_release = 0;
                }
            } else {
                self.climb_ground_release = 0;
            }
        } else if near_climb && !under {
            // 腾空且近绳：视为已抓绳，避免 seek_airborne 只左右晃。
            self.climbing = true;
            self.climb_ground_release = 0;
        }
    }

    /// 决策后：更新朝向与攀爬粘性（位移见 `note_effective`，避免门控剥掉的意图虚走）。
    pub fn after_decide(&mut self, out: &InputFrame, obs: &[f32; OBS_DIM]) {
        if out.right {
            self.facing = 1.0;
        } else if out.left {
            self.facing = -1.0;
        }
        let near = obs_vertical_nav_allowed(obs, false);
        let under = obs_floor_underfoot(obs);
        // 脚下有台时不要因 up 重锁攀爬（绳顶小台旁绳仍可见）。
        if (out.up || out.down || (out.jump && out.up)) && near && !under {
            self.climbing = true;
            self.climb_ground_release = 0;
        }
        // 攀爬粘性只在 prepare 里按多帧落地清，避免绳上单帧假地板误清后左右空走。
    }

    /// 记录实际生效输入的朝向；位移仅由下一视觉帧估计。
    pub fn note_effective(&mut self, effective: &InputFrame) {
        if effective.right {
            self.facing = 1.0;
        } else if effective.left {
            self.facing = -1.0;
        }
        // 水平位移不再虚推：撞墙时 effective 仍可能是 left，虚推会让 stuck 永远检测不到。
    }

    pub fn visual_delta(&self) -> (f32, f32) {
        (self.motion.dx, self.motion.dy)
    }

    pub fn visual_confidence(&self) -> u8 {
        self.motion.confidence
    }

    pub fn location_node(&self) -> LocationNode {
        self.motion.node()
    }
}

impl RuleBot {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn decide(&mut self, ctx: RuleBotCtx<'_>) -> InputFrame {
        self.tick_memory(ctx.obs);
        let prev_x = self.last_x;
        let prev_y = self.last_y;
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
            self.climb_ban_ticks = 0;
            self.initialized = true;
            self.last_reason = "init";
            self.last_move_dir = 0.0;
            self.dir_flip_streak = 0;
        }
        if self.climb_ban_ticks > 0 {
            self.climb_ban_ticks = self.climb_ban_ticks.saturating_sub(1);
        }
        self.note_climb_transit(ctx);
        self.tick_exploration(
            ctx.player_x,
            ctx.player_y,
            prev_x,
            prev_y,
            ctx.climbing,
            ctx.visual_confidence,
        );
        self.update_perch_state(ctx);
        if self.land_cooldown > 0 {
            self.land_cooldown -= 1;
        }

        if ctx.farm_band_mobs {
            self.farm_band_sticky = FARM_BAND_STICKY;
        } else if self.farm_band_sticky > 0 {
            self.farm_band_sticky = self.farm_band_sticky.saturating_sub(1);
        }
        let farm_mobs =
            ctx.farm_band_mobs || self.farm_band_sticky > 0 || self.farm_layer_has_mobs(ctx);
        // 仅挥砍距离内才算「必须留下来打」；远处同层怪不得取消换层。
        let melee_hold = ctx.mob_in_melee
            || ctx
                .engage
                .filter(|e| e.dy.abs() <= SAME_PLATFORM_DY && e.dx.abs() <= STRIKE_HOLD_MAX)
                .is_some();
        let on_first = !self.left_first_platform_layer(ctx);

        let farm_cleared = !farm_mobs;
        let ready_to_seek = farm_cleared
            || self.ticks_on_band >= 8
            || self.dir_flip_streak >= FLIP_ESCAPE_FLIPS;
        let flip_or_clear = farm_cleared || self.dir_flip_streak >= 3;

        // 首台本段仍有怪：禁止 SeekVertical 贴边换层（YOLO 闪断也会被 sticky 挡住）。
        if on_first && farm_mobs && self.explore_mode == ExploreMode::SeekVertical {
            self.explore_mode = ExploreMode::Normal;
            self.explore_mode_ticks = 0;
            if melee_hold {
                self.last_reason = "seek_cancel_melee";
            }
        } else if !on_first
            && obs_has_enemy(ctx.obs)
            && !ctx.mob_on_attackable_footing
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
            && !ctx.mob_in_melee
            && self.explore_mode == ExploreMode::Normal
            && ready_to_seek
            && flip_or_clear
        {
            // 首台本段半径内已无怪（或左右抖）→ 换层。
            self.explore_mode = ExploreMode::SeekVertical;
            self.explore_mode_ticks = 0;
            if self.rope_memory < EXPLORE_ROPE_BOOST {
                self.rope_memory = EXPLORE_ROPE_BOOST;
            }
        } else if !on_first
            && !ctx.mob_on_attackable_footing
            && !self.perching
            && !farm_mobs
            && self.explore_mode == ExploreMode::Normal
            && (self.ticks_on_band >= BAND_STAGNATION_DECISIONS
                || self.ticks_without_new_cell >= NO_NEW_CELL_DECISIONS)
        {
            self.explore_mode = ExploreMode::SeekVertical;
            self.explore_mode_ticks = 0;
        }

        let node = location_node(ctx.player_x, ctx.player_y, ctx.obs);
        self.progress.update(
            ctx.player_x,
            ctx.player_y,
            node,
            self.explore_mode == ExploreMode::SeekVertical,
            self.last_commanded_locomotion,
        );

        let mut out = InputFrame::default();

        // SeekVertical 换层中：只还手（melee），不站等/躲避，否则起跳方向会被撕掉。
        let seeking_now = self.explore_mode == ExploreMode::SeekVertical;
        let want_combat = if seeking_now {
            ctx.mob_in_melee
        } else {
            !self.must_flee(ctx)
        };
        if want_combat && self.try_combat(ctx, &mut out) {
            // 战斗中禁止 ensure_locomotion 补走路，否则站砍会被补成追怪。
            if !seeking_now {
                self.strip_chase_toward_mob(ctx, &mut out);
            }
            self.last_reason = if seeking_now { "seek_melee" } else { "combat" };
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

        if !melee_hold {
            self.update_escape_state(ctx);
            if let Some(frame) = self.try_escape(ctx) {
                self.last_reason = self.escape_reason();
                self.note_move_dir(&frame);
                return frame;
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
            ctx.farm_band_mobs || self.farm_band_sticky > 0
        } else {
            ctx.farm_band_mobs || self.farm_band_sticky > 0 || self.perching
        };
        if seeking {
            // 腾空：若已在绳/梯或近绳，继续攀爬；否则保持换层起跳水平方向。
            if !ctx.on_ground {
                // 绳顶小台：长时间 up 不动且有台阶 → 改登台，禁止死按 up。
                // 直接强制起跳：角色已挂在绳顶（up 无位移），try_step_up 若仍选择
                // 「走路接近」会在绳上左右空摆，必须 jump 登台/跳离绳。
                // 已清农怪台下方：不要 jump 回同层，继续爬/落。
                if ctx.step_up_dx.is_some()
                    && self.climb_up_stall >= 8
                    && !self.step_up_returns_to_cleared_farm(ctx)
                {
                    if let Some(dx) = ctx.step_up_dx {
                        let jdir = if dx.abs() > 1.0 {
                            dx.signum()
                        } else if self.patrol_dir >= 0.0 {
                            1.0
                        } else {
                            -1.0
                        };
                        set_move_dir(&mut out, jdir, true);
                        self.climb_up_stall = 0;
                        self.last_reason = "seek_climb_top";
                        self.note_move_dir(&out);
                        return out;
                    }
                }
                if ctx.climbing || ctx.climb.is_some() || obs_vertical_nav_allowed(ctx.obs, false) {
                    if self.try_climb(ctx, &mut out) {
                        if out.up {
                            self.climb_up_stall = self.climb_up_stall.saturating_add(1);
                        } else {
                            self.climb_up_stall = 0;
                        }
                        self.last_reason = "seek_climb_air";
                        self.note_move_dir(&out);
                        return out;
                    }
                    // 粘性攀爬 / 已抓绳：禁止只左右空摆挂死在绳上。
                    if ctx.climbing || obs_climb_grab_ready(ctx.obs) {
                        if self.climb_x_banned(ctx.player_x) {
                            out.jump = true;
                            out.up = false;
                            out.left = false;
                            out.right = false;
                            self.last_reason = "seek_climb_ban_leave";
                            self.note_move_dir(&out);
                            return out;
                        }
                        out.up = true;
                        out.jump = false;
                        self.climb_up_stall = self.climb_up_stall.saturating_add(1);
                        self.last_reason = "seek_climb_hold";
                        self.note_move_dir(&out);
                        return out;
                    }
                }
                self.climb_up_stall = 0;
                let mut dir = if self.patrol_dir >= 0.0 { 1.0 } else { -1.0 };
                // 视觉误判腾空且贴墙：stuck 相位或物理不可走则反向。
                if self.escape_phase != EscapePhase::Normal || !Self::physics_walk_ok(ctx, dir) {
                    if Self::physics_walk_ok(ctx, -dir) || self.escape_phase != EscapePhase::Normal
                    {
                        dir = -dir;
                        self.patrol_dir = dir;
                    }
                }
                set_move_dir(&mut out, dir, false);
                out.jump = false;
                self.last_reason = "seek_airborne";
                self.note_move_dir(&out);
                return out;
            }
            self.climb_up_stall = 0;
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
            // 绳中禁止 step_up 扯离；绳顶（脚下有台或长时间 up 后）允许登台。
            // try_climb 水平对准超时后会放弃，再走台阶（避免 step=-100 时右对齐卡死）。
            // 已清农怪台：禁止从略低处 step_up 跳回同层（日志 865→805 拉回主因）。
            let allow_step =
                (!ctx.climbing || obs_floor_underfoot(ctx.obs) || self.climb_up_stall >= 8)
                    && !self.step_up_returns_to_cleared_farm(ctx)
                    && self.dir_flip_streak < FLIP_ESCAPE_FLIPS;
            if allow_step && self.try_step_up(ctx, &mut out) {
                self.resolve_blocked_horizontal(ctx, &mut out);
                self.strip_chase_toward_mob(ctx, &mut out);
                if !out.left && !out.right && !out.jump && !out.up {
                    let _ = self.try_edge_jump(ctx, &mut out);
                }
                self.climb_up_stall = 0;
                self.last_reason = "seek_step_up";
                self.note_move_dir(&out);
                return out;
            }
            // 本段已清：可落缘先走下；真虚空边再换层跳。远处怪不得挡住。
            if !melee_hold {
                if self.try_drop_walk(ctx, &mut out) {
                    self.last_reason = "seek_drop";
                    self.note_move_dir(&out);
                    return out;
                }
                if self.try_edge_jump(ctx, &mut out) {
                    self.last_reason = "seek_edge_jump";
                    self.note_move_dir(&out);
                    return out;
                }
            }
            self.try_seek_vertical_walk(ctx, &mut out);
            self.ensure_locomotion(ctx, &mut out);
            self.strip_chase_toward_mob(ctx, &mut out);
            self.last_reason = "seek_walk";
            self.note_move_dir(&out);
            return out;
        }

        // 首台有击杀后：只近距捡，禁止远距拾取转向左右撕。
        let pickup_near_only = on_first && !farm_mobs;
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
        self.last_commanded_locomotion = out.left || out.right || out.jump || out.up || out.down;
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

    fn effective_loop_kind(&self) -> LoopKind {
        let base = self.progress.kind();
        if base != LoopKind::None {
            return base;
        }
        if self.explore_mode != ExploreMode::SeekVertical {
            return LoopKind::None;
        }
        if self.dir_flip_streak >= FLIP_ESCAPE_FLIPS
            && self.ticks_without_new_cell >= SEEK_LOOP_BREAK_CELLS
        {
            return LoopKind::PingPong;
        }
        if self.ticks_on_band >= BAND_STAGNATION_DECISIONS / 2
            && self.dir_flip_streak >= 2
            && self.ticks_without_new_cell >= SEEK_LOOP_BREAK_CELLS / 2
        {
            return LoopKind::SeekStagnant;
        }
        LoopKind::None
    }

    fn update_escape_state(&mut self, ctx: RuleBotCtx<'_>) {
        if self.escape_phase != EscapePhase::Normal {
            let changed_height = (ctx.player_y - self.escape_anchor_y).abs() >= 45.0;
            let probe_moved = self.escape_phase == EscapePhase::ProbeHorizontal
                && (ctx.player_x - self.escape_anchor_x).abs() >= 96.0;
            if changed_height || probe_moved {
                self.escape_phase = EscapePhase::Normal;
                self.escape_ticks = 0;
                self.escape_attempt = 0;
                self.failed_exits = 0;
                self.progress.clear_loop();
                return;
            }
        }

        if self.escape_phase == EscapePhase::Normal {
            self.escape_phase = match self.effective_loop_kind() {
                LoopKind::None => return,
                LoopKind::Blocked => EscapePhase::ProbeHorizontal,
                LoopKind::PingPong | LoopKind::SeekStagnant => EscapePhase::SeekExit,
            };
            self.escape_ticks = 0;
            self.escape_attempt = 0;
            self.escape_anchor_x = ctx.player_x;
            self.escape_anchor_y = ctx.player_y;
            self.escape_dir = if self.last_move_dir != 0.0 {
                -self.last_move_dir
            } else {
                self.prefer_explore_dir(ctx)
            };
            self.explore_mode = ExploreMode::SeekVertical;
        }

        self.escape_ticks = self.escape_ticks.saturating_add(1);
        let limit = match self.escape_phase {
            EscapePhase::Normal => 0,
            EscapePhase::ProbeHorizontal => ESCAPE_PROBE_TICKS,
            EscapePhase::SeekExit => ESCAPE_SEEK_TICKS,
            EscapePhase::ForceVertical => ESCAPE_FORCE_TICKS,
            EscapePhase::AlternateExit => ESCAPE_ALT_TICKS,
        };
        if self.escape_ticks < limit {
            return;
        }

        self.escape_ticks = 0;
        self.escape_phase = match self.escape_phase {
            EscapePhase::Normal => EscapePhase::Normal,
            EscapePhase::ProbeHorizontal => EscapePhase::SeekExit,
            EscapePhase::SeekExit => {
                self.failed_exits |= 1;
                EscapePhase::ForceVertical
            }
            EscapePhase::ForceVertical => {
                self.failed_exits |= 2;
                self.escape_attempt = self.escape_attempt.saturating_add(1);
                self.escape_dir = -self.escape_dir;
                EscapePhase::AlternateExit
            }
            EscapePhase::AlternateExit => {
                self.failed_exits |= 1 << (2 + (self.escape_attempt.min(5)));
                self.escape_attempt = self.escape_attempt.saturating_add(1);
                self.escape_dir = -self.escape_dir;
                EscapePhase::ForceVertical
            }
        };
    }

    fn try_escape(&mut self, ctx: RuleBotCtx<'_>) -> Option<InputFrame> {
        if self.escape_phase == EscapePhase::Normal {
            return None;
        }
        let mut out = InputFrame::default();
        let dir = self.escape_dir.signum();
        match self.escape_phase {
            EscapePhase::Normal => return None,
            EscapePhase::ProbeHorizontal => {
                if Self::can_walk_dir(ctx, dir) {
                    set_move_dir(&mut out, dir, false);
                } else {
                    self.escape_ticks = ESCAPE_PROBE_TICKS;
                    out.jump = true;
                    set_move_dir(&mut out, -dir, false);
                }
            }
            EscapePhase::SeekExit => {
                if self.try_drop_walk(ctx, &mut out)
                    || (!self.step_up_returns_to_cleared_farm(ctx)
                        && self.try_step_up(ctx, &mut out))
                    || self.try_climb(ctx, &mut out)
                {
                    return Some(out);
                }
                if Self::can_walk_dir(ctx, dir) {
                    set_move_dir(&mut out, dir, false);
                } else {
                    self.escape_ticks = ESCAPE_SEEK_TICKS;
                    out.jump = true;
                    set_move_dir(&mut out, dir, false);
                }
            }
            EscapePhase::ForceVertical => {
                if !self.try_forced_platform_leave(ctx, &mut out) {
                    out.jump = true;
                    set_move_dir(&mut out, dir, false);
                }
            }
            EscapePhase::AlternateExit => match self.escape_attempt % 3 {
                0 => {
                    out.jump = true;
                    set_move_dir(&mut out, dir, false);
                }
                1 => {
                    out.down = true;
                    out.jump = true;
                    set_move_dir(&mut out, dir, false);
                }
                _ => {
                    out.up = true;
                    out.jump = true;
                    set_move_dir(&mut out, dir, false);
                }
            },
        }
        Some(out)
    }

    fn escape_reason(&self) -> &'static str {
        match self.escape_phase {
            EscapePhase::Normal => "escape_normal",
            EscapePhase::ProbeHorizontal => "escape_probe",
            EscapePhase::SeekExit => "escape_seek_exit",
            EscapePhase::ForceVertical => "escape_force_vertical",
            EscapePhase::AlternateExit => "escape_alternate",
        }
    }

    pub fn loop_kind_name(&self) -> &'static str {
        match self.effective_loop_kind() {
            LoopKind::None => "none",
            LoopKind::Blocked => "blocked",
            LoopKind::PingPong => "pingpong",
            LoopKind::SeekStagnant => "seek_stagnant",
        }
    }

    pub fn escape_phase_name(&self) -> &'static str {
        self.escape_reason()
    }

    pub fn escape_candidate_name(&self) -> &'static str {
        match (
            self.escape_phase,
            self.escape_attempt % 3,
            self.escape_dir >= 0.0,
        ) {
            (EscapePhase::Normal, _, _) => "none",
            (EscapePhase::ProbeHorizontal, _, true) => "probe_right",
            (EscapePhase::ProbeHorizontal, _, false) => "probe_left",
            (EscapePhase::SeekExit, _, true) => "visible_exit_or_right",
            (EscapePhase::SeekExit, _, false) => "visible_exit_or_left",
            (EscapePhase::ForceVertical, _, true) => "jump_right",
            (EscapePhase::ForceVertical, _, false) => "jump_left",
            (EscapePhase::AlternateExit, 0, true) => "alternate_jump_right",
            (EscapePhase::AlternateExit, 0, false) => "alternate_jump_left",
            (EscapePhase::AlternateExit, 1, true) => "drop_right",
            (EscapePhase::AlternateExit, 1, false) => "drop_left",
            (EscapePhase::AlternateExit, _, true) => "climb_right",
            (EscapePhase::AlternateExit, _, false) => "climb_left",
        }
    }

    pub fn progress_metrics(&self) -> (f32, f32, f32, f32) {
        (
            self.progress.net_distance,
            self.progress.path_distance,
            self.progress.span_x,
            self.progress.span_y,
        )
    }

    pub fn failed_exit_count(&self) -> u32 {
        self.failed_exits.count_ones()
    }

    /// 同台近距有怪时禁止朝怪方向走。腾空/跳跃/攀爬时绝不禁走（否则换层跳会被撕掉）。
    fn strip_chase_toward_mob(&self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) {
        // 已离首台换层探索：不要禁走，否则贴崖+怪在可走侧会 noop 卡死。
        if self.explore_mode == ExploreMode::SeekVertical && self.left_first_platform_layer(ctx) {
            return;
        }
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
            let near_mob = ctx.mob_in_melee
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

    /// 与 movement_gate.walk_allowed 对齐：物理悬崖挡死；可落下缘允许走下去。
    fn can_walk_dir(ctx: RuleBotCtx<'_>, dir: f32) -> bool {
        if dir == 0.0 {
            return false;
        }
        if !Self::physics_walk_ok(ctx, dir) {
            return false;
        }
        // 下方有平台的落缘：主动走下去换层，不要当成虚空。
        if Self::physics_drop_ahead(ctx, dir) {
            return true;
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

    fn physics_walk_ok(ctx: RuleBotCtx<'_>, dir: f32) -> bool {
        if dir > 0.0 {
            ctx.walk_right_ok.unwrap_or(true)
        } else if dir < 0.0 {
            ctx.walk_left_ok.unwrap_or(true)
        } else {
            false
        }
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

    /// 贴边换层：可落先走下 → 台阶跳 / 悬崖跳。
    fn try_leave_edge(&mut self, ctx: RuleBotCtx<'_>, dir: f32, out: &mut InputFrame) -> bool {
        // 右/左下方有平台：走下去，禁止在可落缘瞎跳（永远够不着右上高台）。
        if Self::physics_drop_ahead(ctx, dir) {
            set_move_dir(out, dir, false);
            self.patrol_dir = dir;
            return true;
        }
        if ctx.step_up_dx.is_some() && self.try_edge_jump_dir(ctx, dir, JumpPurpose::StepUp, out) {
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
        let phys_edge = Self::at_cliff(ctx, dir) || Self::physics_drop_ahead(ctx, dir);
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

    fn tick_exploration(
        &mut self,
        x: f32,
        y: f32,
        prev_x: f32,
        prev_y: f32,
        climbing: bool,
        visual_confidence: u8,
    ) {
        let key = visit_key(x, y);
        let y_band = key.1;
        let position_trustworthy = visual_confidence >= VISIT_MIN_CONF
            || ((x - prev_x).abs() + (y - prev_y).abs()) < 36.0;
        self.visited_y_bands.insert(y_band);
        if position_trustworthy {
            if self.visited.insert(key) {
                self.ticks_without_new_cell = 0;
            } else {
                self.ticks_without_new_cell = self.ticks_without_new_cell.saturating_add(1);
            }
        } else {
            self.ticks_without_new_cell = self.ticks_without_new_cell.saturating_add(1);
        }

        if y_band != self.last_y_band {
            if !position_trustworthy {
                self.ticks_on_band = self.ticks_on_band.saturating_add(1);
                return;
            }
            // 非攀爬换层（seek_drop 自由落体）：禁用途经绳/梯，防立刻爬回。
            // 攀爬中途跨 band 不写 ban，否则会在半绳触发 jump 离绳。
            if !climbing && (y - prev_y).abs() >= 40.0 {
                self.climb_ban_x = if (x - prev_x).abs() < 1.0 {
                    x
                } else {
                    (x + prev_x) * 0.5
                };
                self.climb_ban_ticks = CLIMB_TRANSIT_BAN_TICKS;
            }
            self.last_y_band = y_band;
            self.ticks_on_band = 0;
            // 换高度带 = 已离开旧层局部网格，清局部循环计数（梯顶立刻 loop_break 主因）。
            self.ticks_without_new_cell = 0;
            // 登上新高度层：提交新农怪脚点，禁止「高台避险」把人拽回旧层。
            if self.farm_y <= 0.0 || (y - self.farm_y).abs() > 35.0 {
                self.farm_y = y;
            }
            self.perching = false;
            self.perch_ticks = 0;
            self.land_cooldown = 0;
            self.dir_flip_streak = 0;
            // SeekVertical 换层中跨 y_band（爬绳）不要退回 Normal，否则绳上会丢攀爬意图。
            if self.explore_mode != ExploreMode::SeekVertical {
                self.explore_mode = ExploreMode::Normal;
                self.explore_mode_ticks = 0;
            }
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

    fn note_climb_transit(&mut self, ctx: RuleBotCtx<'_>) {
        if ctx.climbing && !self.was_climbing {
            self.climb_start_x = ctx.player_x;
            self.climb_start_y = ctx.player_y;
        } else if !ctx.climbing && self.was_climbing {
            if (ctx.player_y - self.climb_start_y).abs() >= 40.0 {
                self.climb_ban_x = (self.climb_start_x + ctx.player_x) * 0.5;
                self.climb_ban_ticks = CLIMB_TRANSIT_BAN_TICKS;
            }
        }
        self.was_climbing = ctx.climbing;
    }

    fn climb_x_banned(&self, world_x: f32) -> bool {
        self.climb_ban_ticks > 0 && (world_x - self.climb_ban_x).abs() <= CLIMB_BAN_X_SLACK
    }

    fn climb_hint_banned(&self, ctx: RuleBotCtx<'_>, climb: ClimbHint) -> bool {
        self.climb_x_banned(ctx.player_x + climb.dx)
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
            ctx.drop_ahead_right == Some(true)
        } else if dir < 0.0 {
            ctx.drop_ahead_left == Some(true)
        } else {
            false
        }
    }

    fn left_first_platform_layer(&self, ctx: RuleBotCtx<'_>) -> bool {
        visit_key(ctx.player_x, ctx.player_y).1 != self.spawn_y_band
            || ctx.player_y < self.spawn_y - 50.0
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
        steer_toward_slot(ctx.obs, OBS_DROP_START, OBS_DROP_SLOTS, out, ctx.facing);
        true
    }

    fn at_cliff(ctx: RuleBotCtx<'_>, dir: f32) -> bool {
        if dir > 0.0 {
            ctx.walk_right_ok == Some(false)
        } else if dir < 0.0 {
            ctx.walk_left_ok == Some(false)
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
        if !seeking && (self.perching || (self.farm_y > 0.0 && ctx.player_y < self.farm_y - 40.0)) {
            self.perching = true;
            if self.should_drop_to_farm(ctx) {
                return self.try_drop_to_farm(ctx, out);
            }
            out.left = false;
            out.right = false;
            return true;
        }

        // 贴身接触盒重叠：必砍（YOLO 接战槽闪断时仍能出手）。
        let contact = obs_assess_enemy_contact(ctx.obs);
        if contact.total > 0 {
            let toward = if contact.right > contact.left {
                1.0
            } else if contact.left > contact.right {
                -1.0
            } else {
                ctx.facing.signum()
            };
            let need_face =
                (toward > 0.0 && ctx.facing <= 0.0) || (toward < 0.0 && ctx.facing >= 0.0);
            // 已在挥砍带：站砍；仅未进距时先转身。
            if need_face && !ctx.mob_in_melee {
                set_move_dir(out, toward, false);
            } else {
                out.left = false;
                out.right = false;
                out.attack = true;
            }
            return true;
        }

        // 只认本台怪；邻台/其他高度当不存在。
        if !Self::same_platform_threat(ctx) {
            return false;
        }

        let engage = ctx.engage.filter(|e| e.dy.abs() <= SAME_PLATFORM_DY);

        let Some(engage) = engage else {
            if ctx.mob_in_melee {
                out.left = false;
                out.right = false;
                out.attack = true;
                return true;
            }
            return false;
        };

        // SeekVertical：非 melee 不占用决策。
        if seeking && !ctx.mob_in_melee {
            return false;
        }

        if ctx.mob_in_melee {
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
                    if !ctx.mob_in_melee {
                        out.attack = false;
                    }
                    return true;
                }
                if toward != 0.0 && !Self::at_cliff(ctx, -toward) {
                    set_move_dir(out, -toward, false);
                    if !ctx.mob_in_melee {
                        out.attack = false;
                    }
                }
            }
            return true;
        }

        // 中距：背离则站等折返；仅需转身则转身；其余让出给巡逻，禁止 noop 站死。
        if dist <= STAND_WAIT_MAX && (walking_away || engage.mob_approaching() || toward != 0.0) {
            let need_face =
                (toward > 0.0 && ctx.facing <= 0.0) || (toward < 0.0 && ctx.facing >= 0.0);
            if need_face {
                set_move_dir(out, toward, false);
                return true;
            }
            if walking_away {
                out.left = false;
                out.right = false;
                return true;
            }
            return false;
        }

        false
    }

    /// 本台是否有可接战威胁（|dy|≤同台阈值）。邻台怪一律忽略。
    fn same_platform_threat(ctx: RuleBotCtx<'_>) -> bool {
        if ctx.mob_in_melee {
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
        } else if toward != 0.0 && ctx.mob_in_melee {
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
            || ctx.mob_in_melee)
            && (ctx.mob_in_melee || obs_assess_enemy_contact(ctx.obs).total > 0)
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
        contact.left.max(contact.right) >= CONTACT_RETREAT_MIN
    }

    fn should_retreat(&self, _ctx: RuleBotCtx<'_>, contact: &EnemyContactAssessment) -> bool {
        if contact.total == 0 {
            return false;
        }
        if is_surrounded(contact) {
            return true;
        }
        contact.left.max(contact.right) >= CONTACT_RETREAT_MIN
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

    fn best_escape_dir(
        &self,
        ctx: RuleBotCtx<'_>,
        contact: &EnemyContactAssessment,
    ) -> Option<f32> {
        let (cx, cy) = visit_key(ctx.player_x, ctx.player_y);
        let mut candidates: [(f32, i32); 2] = [(1.0, 0), (-1.0, 0)];
        for (dir, score) in &mut candidates {
            *score = 0;
            if *dir > 0.0 {
                *score -= (contact.right as i32) * 10;
                if !self.visited.contains(&(cx + 1, cy)) {
                    *score += 3;
                }
                if ctx.walk_right_ok == Some(false) {
                    *score -= 5;
                }
            } else {
                *score -= (contact.left as i32) * 10;
                if !self.visited.contains(&(cx - 1, cy)) {
                    *score += 3;
                }
                if ctx.walk_left_ok == Some(false) {
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
            // 刚用这根梯子换过层：禁止继续上爬回到对面，跳离绳/梯。
            if self.climb_x_banned(ctx.player_x) {
                out.jump = true;
                out.up = false;
                out.down = false;
                out.left = false;
                out.right = false;
                return true;
            }
            // 已在绳上：优先继续上；到顶由 sim 落平台。
            out.up = true;
            out.jump = false;
            out.left = false;
            out.right = false;
            return true;
        }

        // 只用紧邻当前层的绳/梯；YOLO 远处上层绳不算。
        let Some(climb) = ctx.climb else {
            self.climb_attempt_ticks = 0;
            return false;
        };

        // 刚爬过的同一根：禁止立刻再抓（上下来回主因）。
        if self.climb_hint_banned(ctx, climb) {
            self.climb_attempt_ticks = 0;
            return false;
        }

        // 腾空挂绳：禁止水平对准空走（日志里梯子左右抖死主因），直接爬。
        if !ctx.on_ground {
            out.left = false;
            out.right = false;
            out.jump = false;
            match climb.dir {
                ClimbDir::Up => out.up = true,
                ClimbDir::Down => out.down = true,
            }
            return true;
        }

        // 站在绳顶平台时会一直收到 Down：若本层还能走/跳，禁止立刻下爬（日志里卡死主因）。
        if climb.dir == ClimbDir::Down && ctx.on_ground {
            let can_explore = Self::can_walk_dir(ctx, 1.0)
                || Self::can_walk_dir(ctx, -1.0)
                || ctx.step_up_dx.is_some();
            if can_explore || self.ticks_on_band < BAND_STAGNATION_DECISIONS {
                self.climb_attempt_ticks = 0;
                return false;
            }
        }
        // 已对齐绳且脚下有更高台：优先登台，不要反复 jump+up 抓同一根绳。
        // （未对齐时仍要走路接近绳，不能被 step_up 抢走。）
        if climb.dir == ClimbDir::Up
            && ctx.on_ground
            && climb.dx.abs() <= CLIMB_ALIGN_PX
            && ctx.step_up_dx.is_some()
        {
            self.climb_attempt_ticks = 0;
            return false;
        }

        if climb.dx.abs() > CLIMB_ALIGN_PX {
            // 水平对准久无进展会卡在绳旁（effective=right 但 sim 不动）：放弃抓绳改台阶/走路。
            self.climb_attempt_ticks = self.climb_attempt_ticks.saturating_add(1);
            if self.climb_attempt_ticks > 10 {
                self.climb_attempt_ticks = 0;
                return false;
            }
            set_move_dir(out, climb.dx.signum(), false);
            return true;
        }

        // 已对准仍卡住：放弃本轮抓绳，改走路。
        self.climb_attempt_ticks = self.climb_attempt_ticks.saturating_add(1);
        if self.climb_attempt_ticks > 20 {
            self.climb_attempt_ticks = 0;
            return false;
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

    /// Seek 已清农怪后：step_up 跳回已清层/同高度带 → 禁止（865→805 闭环主因）。
    fn step_up_returns_to_cleared_farm(&self, ctx: RuleBotCtx<'_>) -> bool {
        if self.explore_mode != ExploreMode::SeekVertical {
            return false;
        }
        if ctx.farm_band_mobs || self.farm_band_sticky > 0 {
            return false;
        }
        if self.farm_y <= 0.0 || ctx.step_up_dx.is_none() {
            return false;
        }
        let farm_band = visit_key(0.0, self.farm_y).1;
        let here_band = visit_key(ctx.player_x, ctx.player_y).1;
        // 人在 farm 下方（y 更大）：step_up 多半跳回已清上层。
        if ctx.player_y > self.farm_y + 25.0 {
            return true;
        }
        // 与 farm 同高度带且局部已扫够：禁止 lateral step_up 空转。
        if here_band == farm_band && self.ticks_without_new_cell >= SEEK_LOOP_BREAK_CELLS {
            return true;
        }
        let below = ctx.player_y - self.farm_y;
        below > 18.0 && below < 110.0
    }

    /// Seek 换层：优先未访问格子/可落缘；都走过则伪随机换向打破闭环。
    fn prefer_explore_dir(&self, ctx: RuleBotCtx<'_>) -> f32 {
        let (cx, cy) = visit_key(ctx.player_x, ctx.player_y);
        let mut best_dir = self.patrol_dir.signum();
        let mut best_score = i32::MIN;
        for dir in [1.0_f32, -1.0] {
            let key = if dir > 0.0 {
                (cx + 1, cy)
            } else {
                (cx - 1, cy)
            };
            let mut score = 0;
            if !self.visited.contains(&key) {
                score += 12;
            }
            if Self::can_walk_dir(ctx, dir) {
                score += 6;
            }
            if Self::physics_drop_ahead(ctx, dir) {
                let drop_band = visit_key(ctx.player_x + dir * 80.0, ctx.player_y + 60.0).1;
                if !self.visited_y_bands.contains(&drop_band) {
                    score += 8;
                } else {
                    score += 2;
                }
            }
            if let Some(climb) = ctx.climb {
                if dir.signum() == climb.dx.signum() && self.climb_hint_banned(ctx, climb) {
                    score -= 8;
                }
            }
            if score > best_score {
                best_score = score;
                best_dir = dir;
            }
        }
        if best_score <= 0 {
            let flip = ((ctx.player_x as i32 / 80) ^ (self.explore_mode_ticks as i32)) & 1;
            best_dir = if flip == 0 { 1.0 } else { -1.0 };
        }
        best_dir
    }

    /// 视觉无悬崖时仍强制跳离当前层（Seek 扫台过久兜底）。
    fn try_forced_platform_leave(&mut self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) -> bool {
        if !ctx.on_ground || ctx.climbing {
            return false;
        }

        // 梯子口/梯顶：禁止 jump+up 空转（日志 1286↔676、1161↔556 卡死主因）。
        if let Some(climb) = ctx.climb {
            if self.climb_hint_banned(ctx, climb) {
                let away = -climb.dx.signum();
                if Self::can_walk_dir(ctx, away) {
                    set_move_dir(out, away, false);
                    self.patrol_dir = away;
                    return true;
                }
                let away_ban = (ctx.player_x - self.climb_ban_x).signum();
                if away_ban.abs() > 0.0 && Self::can_walk_dir(ctx, away_ban) {
                    set_move_dir(out, away_ban, false);
                    self.patrol_dir = away_ban;
                    return true;
                }
            } else if climb.dir == ClimbDir::Down {
                let dir = self.prefer_explore_dir(ctx);
                set_move_dir(out, dir, false);
                self.patrol_dir = dir;
                return true;
            } else if climb.dir == ClimbDir::Up {
                if climb.dx.abs() <= CLIMB_ALIGN_PX {
                    out.up = true;
                    out.jump = true;
                    return true;
                }
                set_move_dir(out, climb.dx.signum(), false);
                self.patrol_dir = climb.dx.signum();
                return true;
            }
        }

        let dir = self.prefer_explore_dir(ctx);
        self.patrol_dir = dir;
        if self.try_leave_edge(ctx, dir, out) || self.try_leave_edge(ctx, -dir, out) {
            return true;
        }
        if obs_vertical_nav_allowed(ctx.obs, false) {
            out.up = true;
        }
        out.jump = true;
        set_move_dir(out, dir, false);
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
            // 远距且前方视觉可走：走近。近距或门控会剥掉走路时直接起跳。
            // 无地板 YOLO 时（oracle/单测）以 physics 为准，不误强制跳。
            let floor_ok = !obs_has_floor_signal(ctx.obs) || obs_floor_ahead(ctx.obs, dir);
            // 卡死恢复中：视觉可走但物理不动（挂在绳上 / 被台阶侧墙挡住），
            // 禁止只走路空转 —— 必须起跳登台或跳离绳。
            let stuck_now = self.escape_phase != EscapePhase::Normal;
            if !stuck_now && Self::can_walk_dir(ctx, dir) && floor_ok && dx.abs() > STEP_UP_JUMP_DX
            {
                set_move_dir(out, dir, false);
                self.patrol_dir = dir;
                return true;
            }
            if self.should_allow_jump(ctx, dir, JumpPurpose::StepUp) || stuck_now {
                out.jump = true;
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
        // 可落缘不要跳：跳出去够不着右上高台，只会反复空跳。
        if Self::physics_drop_ahead(ctx, facing) {
            return false;
        }
        self.try_edge_jump_dir(ctx, facing, JumpPurpose::PlatformChange, out)
    }

    /// 走向可落下缘（下方有更低平台），不跳。
    fn try_drop_walk(&mut self, ctx: RuleBotCtx<'_>, out: &mut InputFrame) -> bool {
        for dir in [self.patrol_dir.signum(), -self.patrol_dir.signum()] {
            if !Self::physics_drop_ahead(ctx, dir) {
                continue;
            }
            // 落缘朝向刚爬过的梯子 → 会掉回对面层，跳过。
            if self.climb_ban_ticks > 0 {
                let toward_ban = (self.climb_ban_x - ctx.player_x).signum() == dir
                    || (self.climb_ban_x - ctx.player_x).abs() <= CLIMB_BAN_X_SLACK;
                if toward_ban && (self.climb_ban_x - ctx.player_x).abs() < 140.0 {
                    continue;
                }
            }
            set_move_dir(out, dir, false);
            self.patrol_dir = dir;
            return true;
        }
        false
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
            } else if self.try_leave_edge(ctx, dir, out) || self.try_leave_edge(ctx, -dir, out) {
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
        if ctx.climbing {
            if self.climb_x_banned(ctx.player_x) {
                out.jump = true;
                out.up = false;
                out.left = false;
                out.right = false;
                return;
            }
            out.up = true;
            out.left = false;
            out.right = false;
            out.jump = false;
            return;
        }
        if let Some(climb) = ctx.climb {
            let banned = self.climb_hint_banned(ctx, climb);
            // 地面上的 Down 提示交给 try_climb 统一过滤；禁用绳/梯则走开。
            let take = !banned
                && match climb.dir {
                    ClimbDir::Down if ctx.on_ground => false,
                    _ => true,
                };
            if take {
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
        }

        if self.try_step_up(ctx, out) && self.dir_flip_streak < FLIP_ESCAPE_FLIPS {
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

        for dir in [self.patrol_dir.signum(), -self.patrol_dir.signum()] {
            if Self::can_walk_dir(ctx, dir) {
                self.patrol_dir = dir;
                set_move_dir(out, dir, false);
                return;
            }
        }

        // 双侧都堵：优先未探索方向，再换层跳。
        let dir = self.prefer_explore_dir(ctx);
        self.patrol_dir = dir;
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
    c.left >= SURROUNDED_SIDE_MIN && c.right >= SURROUNDED_SIDE_MIN && c.total >= 2
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

fn steer_toward_same_level_enemy(obs: &[f32; OBS_DIM], out: &mut InputFrame, facing: f32) {
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
    use crate::game::observation::OBS_FLOOR_START;

    #[test]
    fn patrol_moves_right_by_default() {
        let mut bot = RuleBot::default();
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 0.0,
            player_y: 0.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(inp.right);
        assert!(!inp.left);
    }

    #[test]
    fn vision_never_uses_potion_without_hp_detection() {
        let mut bot = RuleBot::default();
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 0.0,
            player_y: 0.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            !inp.use_potion,
            "无 YOLO 血量/药水感知时不应触发喝药"
        );
    }

    #[test]
    fn patrol_avoids_cliff_by_turning_instead_of_jumping() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_FLOOR_START + 2] = 0.08;
        obs[OBS_FLOOR_START + 3] = 0.04;
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 100.0,
            player_y: 100.0,
            walk_right_ok: Some(false),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: true,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            !inp.jump,
            "normal patrol should turn away from cliff, not jump"
        );
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
            player_x: 80.0,
            player_y: 0.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 50.0,
            player_y: 200.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
        obs[OBS_ENEMY_START] = 0.12;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 430.0,
            player_y: 1225.0,
            walk_right_ok: Some(false),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 720.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: Some(-200.0),
            farm_band_mobs: false,
            visual_confidence: 11,
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
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 430.0,
            player_y: 1225.0,
            walk_right_ok: Some(false),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
        // 地板框只覆盖左侧走廊，避免宽框同时命中左右 ahead。
        obs[OBS_FLOOR_START] = -0.12;
        obs[OBS_FLOOR_START + 1] = 0.02;
        obs[OBS_FLOOR_START + 2] = 0.12;
        obs[OBS_FLOOR_START + 3] = 0.04;
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 430.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(!inp.jump, "YOLO right-edge alone must turn, not jump spam");
        assert!(inp.left, "should reverse toward remaining floor");
    }

    #[test]
    fn crowded_touching_on_one_side_retreats() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        set_touching_enemy(&mut obs, 0, 0.018, 0.0);
        set_touching_enemy(&mut obs, 1, 0.022, 0.01);
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.left && !inp.attack,
            "同侧两只贴身怪应撤退，不应站砍"
        );
    }

    #[test]
    fn crowded_touching_on_one_side_retreats_even_without_hp() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        set_touching_enemy(&mut obs, 0, 0.015, 0.0);
        set_touching_enemy(&mut obs, 1, 0.020, 0.0);
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.left && !inp.attack,
            "同侧多只贴身怪应撤退（纯视觉接触判定，不依赖血量）"
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
            player_x: 700.0,
            player_y: 1225.0,
            walk_right_ok: Some(false),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 100.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: Some(30.0),
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(bot.explore_seeking_vertical(), "must stay SeekVertical");
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
            player_x: 416.0,
            player_y: 1225.0,
            walk_right_ok: Some(false),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(true),
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 900.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: Some(40.0),
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 416.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 1080.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: true,
            visual_confidence: 11,
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: -110.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: true,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            !inp.left && !inp.right,
            "distant walking-away mobs: stand wait, no chase"
        );
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.attack,
            "one touching mob at full hp should attack, not flee"
        );
    }

    #[test]
    fn surrounded_jumps_when_cliffs_both_sides() {
        let mut bot = RuleBot::default();
        let mut obs = [0.0_f32; OBS_DIM];
        set_touching_enemy(&mut obs, 0, -0.018, 0.0);
        set_touching_enemy(&mut obs, 1, 0.018, 0.0);
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(false),
            walk_left_ok: Some(false),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 90.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(!inp.jump, "front approach on flat should not jump");
        assert!(
            inp.attack,
            "within strike band (~90): start chopping before touch"
        );
        assert!(!inp.left && !inp.right, "must stand chop, not chase");
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: -70.0,
                mob_dir: 1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.attack,
            "left approaching mob in strike band must chop before touch"
        );
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
            player_x: 416.0,
            player_y: 1225.0,
            walk_right_ok: Some(false),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(true),
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: Some(20.0),
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 520.0,
            player_y: 1225.0,
            walk_right_ok: Some(false),
            walk_left_ok: Some(true),
            // 真虚空/隔空跳：无落下平台。可落缘应走下去而不是跳。
            drop_ahead_right: Some(false),
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: -180.0,
                mob_dir: 1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(false),
            walk_left_ok: Some(false),
            drop_ahead_right: Some(true),
            drop_ahead_left: Some(true),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 150.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: true,
            visual_confidence: 11,
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 100.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 700.0,
            player_y: 805.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 500.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 520.0,
            player_y: 1225.0,
            walk_right_ok: Some(false),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(true),
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            !inp.jump,
            "should fight instead of cliff-jumping over attackable mobs"
        );
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
            player_x: 741.0,
            player_y: 1155.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: Some(true),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
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
            visual_confidence: 11,
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
            player_x: 741.0,
            player_y: 1171.0,
            walk_right_ok: None,
            walk_left_ok: None,
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: -6.0,
                dx: 117.0,
                mob_dir: 1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 24.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: Some(-80.0),
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 120.0,
                mob_dir: 1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(!inp.jump, "must not leap to overtake");
        assert!(
            !inp.left && !inp.right,
            "must stand and wait for turnaround"
        );
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 28.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 28.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 50.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 416.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: Some(-120.0),
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(inp.left, "should walk toward left step-up ledge");
        assert!(!inp.jump, "not yet aligned; no blind jump");
    }

    #[test]
    fn rope_top_ground_ignores_down_climb_and_walks() {
        // 复现日志：爬绳到 (1020,865) 后 climb=Down，一直按 down 不动。
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1105.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(400.0, 1225.0).1;
        bot.last_y_band = visit_key(1020.0, 865.0).1;
        bot.ticks_on_band = 5;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = 1.0;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 1020.0,
            player_y: 865.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: Some(ClimbHint {
                dx: 0.0,
                dir: ClimbDir::Down,
            }),
            step_up_dx: Some(-100.0),
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            !inp.down,
            "must not hold down at rope top; reason={}",
            bot.last_reason
        );
        assert!(
            inp.left || inp.right || inp.jump || inp.up,
            "should explore horizontally / step-up instead of standing; reason={}",
            bot.last_reason
        );
    }

    #[test]
    fn seek_prefers_drop_walk_over_edge_jump() {
        // 次高台右崖：下方有平台可落，禁止对着右上方空跳。
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 805.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(400.0, 1225.0).1;
        bot.last_y_band = visit_key(902.0, 805.0).1;
        bot.ticks_on_band = 20;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = 1.0;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 902.0,
            player_y: 805.0,
            walk_right_ok: Some(false),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(true),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.right && !inp.jump,
            "must walk off drop ledge, not jump; reason={} right={} jump={}",
            bot.last_reason,
            inp.right,
            inp.jump
        );
        assert_eq!(bot.last_reason, "seek_drop");
    }

    #[test]
    fn seek_walks_to_up_climb_instead_of_step_back() {
        // 落到 y=865 后应走向右上绳，而不是 step_up 跳回左边 805。
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 865.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(400.0, 1225.0).1;
        bot.last_y_band = visit_key(961.0, 865.0).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = -1.0;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: -1.0,
            on_ground: true,
            climbing: false,
            player_x: 961.0,
            player_y: 865.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: Some(ClimbHint {
                dx: 131.0,
                dir: ClimbDir::Up,
            }),
            step_up_dx: Some(-41.0),
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.right && !inp.left && !inp.jump,
            "must walk toward Up climb, not step_up left; reason={}",
            bot.last_reason
        );
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
            player_x: 500.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: 40.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: -40.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            player_x: 10.0,
            player_y: 10.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: true,
            mob_on_attackable_footing: true,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
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
            sim.movement_gate.set_last_observation(&obs);
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
            sim.movement_gate.set_last_observation(&obs);
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
            sim.movement_gate.set_last_observation(&obs);
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
            sim.movement_gate.set_last_observation(&obs);
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
            decisions_mob_on_left > 30 || sim.state.kills >= 1,
            "scenario should observe mob on the left (or kill it before it crosses)"
        );
        if decisions_mob_on_left > 0 {
            assert!(
                chase_left_when_mob_left * 5 < decisions_mob_on_left,
                "bot must not chase left after mob: chase={chase_left_when_mob_left} left_obs={decisions_mob_on_left}"
            );
        }
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
            player_x: 400.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: true,
            engage: Some(crate::game::EngageHint {
                dy: 0.0,
                dx: -90.0,
                mob_dir: -1.0,
            }),
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: true,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            !inp.left,
            "patrol/seek must not walk left toward distant mob on the left"
        );
    }

    #[test]
    fn from_vision_reads_enemy_cliff_and_climb_without_sim() {
        use crate::game::map::ClimbDir;
        use crate::game::observation::{OBS_ENEMY_START, OBS_FLOOR_START, OBS_ROPE_START};

        let mut obs = [0.0_f32; OBS_DIM];
        // 脚下地板（略偏左，右侧不覆盖行走前方带 → 右崖）
        obs[OBS_FLOOR_START] = -0.02;
        obs[OBS_FLOOR_START + 1] = 0.02;
        obs[OBS_FLOOR_START + 2] = 0.05;
        obs[OBS_FLOOR_START + 3] = 0.04;
        // 右侧同层怪
        obs[OBS_ENEMY_START] = 0.05;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.04;
        obs[OBS_ENEMY_START + 3] = 0.05;
        // 右崖：无前方同层地板；下方有地板框
        obs[OBS_FLOOR_START + 4] = 0.08;
        obs[OBS_FLOOR_START + 5] = 0.12;
        obs[OBS_FLOOR_START + 6] = 0.15;
        obs[OBS_FLOOR_START + 7] = 0.04;
        // 近处上爬绳：底在脚点附近上方
        obs[OBS_ROPE_START] = 0.02;
        obs[OBS_ROPE_START + 1] = -0.06;
        obs[OBS_ROPE_START + 2] = 0.02;
        obs[OBS_ROPE_START + 3] = 0.18;

        let sense = VisionSenseState::default();
        let ctx = RuleBotCtx::from_vision(&obs, &sense);
        assert!(ctx.on_ground);
        assert!(ctx.mob_in_melee || ctx.engage.is_some());
        assert_eq!(ctx.walk_right_ok, Some(false));
        assert_eq!(ctx.drop_ahead_right, Some(true));
        assert!(ctx.farm_band_mobs);
        let climb = ctx.climb.expect("rope up climb");
        assert!(matches!(climb.dir, ClimbDir::Up));
    }

    #[test]
    fn seek_on_rope_airborne_keeps_climbing_up() {
        use crate::game::observation::OBS_ROPE_START;

        let mut bot = RuleBot::default();
        bot.initialized = true;
        // 与当前高度同 band，避免 tick_exploration 把 SeekVertical 清回 Normal。
        let (px, py) = (488.0_f32, 1045.0);
        bot.farm_y = py;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(px, py).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = -1.0;

        let mut obs = [0.0_f32; OBS_DIM];
        // 近绳（腾空挂绳）：无脚下地板
        obs[OBS_ROPE_START] = 0.005;
        obs[OBS_ROPE_START + 1] = -0.02;
        obs[OBS_ROPE_START + 2] = 0.02;
        obs[OBS_ROPE_START + 3] = 0.20;

        let ctx = RuleBotCtx {
            obs: &obs,
            facing: -1.0,
            on_ground: false,
            climbing: true,
            player_x: px,
            player_y: py,
            walk_right_ok: None,
            walk_left_ok: None,
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: Some(ClimbHint {
                dx: 4.0,
                dir: ClimbDir::Up,
            }),
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.up,
            "hanging on rope during SeekVertical must keep climbing up, got reason={}",
            bot.last_reason
        );
        assert!(
            bot.last_reason == "seek_climb_air" || bot.last_reason == "seek_climb_hold",
            "unexpected reason {}",
            bot.last_reason
        );
    }

    #[test]
    fn seek_airborne_near_ladder_presses_up_not_align_walk() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        let (px, py) = (1477.0_f32, 1028.0);
        bot.farm_y = py;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(px, py).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = -1.0;

        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: -1.0,
            on_ground: false,
            climbing: false,
            player_x: px,
            player_y: py,
            walk_right_ok: None,
            walk_left_ok: None,
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: Some(ClimbHint {
                dx: -39.0,
                dir: ClimbDir::Up,
            }),
            step_up_dx: Some(97.0),
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.up,
            "airborne near ladder must climb up, not align-walk; reason={} inp={:?}",
            bot.last_reason, inp
        );
        assert!(
            !inp.left && !inp.right,
            "must not horizontal-align while airborne on ladder"
        );
    }

    #[test]
    fn contact_overlap_attacks_without_engage() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 1225.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = bot.spawn_y_band;
        // 贴身重叠：无 engage 也应砍（复现「贴怪走过不砍」）。
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_ENEMY_START] = 0.01;
        obs[OBS_ENEMY_START + 1] = 0.0;
        obs[OBS_ENEMY_START + 2] = 0.04;
        obs[OBS_ENEMY_START + 3] = 0.08;
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 500.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: true,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.attack || inp.right || inp.left,
            "contact must face or attack, got reason={}",
            bot.last_reason
        );
        assert_ne!(bot.last_reason, "normal_patrol");
    }

    #[test]
    fn seek_step_up_jumps_when_near_ledge() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.farm_y = 1225.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(735.0, 1225.0).1;
        bot.last_y_band = bot.spawn_y_band;
        let mut obs = [0.0_f32; OBS_DIM];
        // 脚下有地板信号，前方无衔接 → 不可空走，应跳上台阶。
        use crate::game::observation::OBS_FLOOR_START;
        obs[OBS_FLOOR_START] = 0.0;
        obs[OBS_FLOOR_START + 1] = 0.02;
        obs[OBS_FLOOR_START + 2] = 0.20;
        obs[OBS_FLOOR_START + 3] = 0.05;
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 735.0,
            player_y: 1225.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: Some(31.0),
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert!(
            inp.jump,
            "near step-up with no floor ahead must jump, got reason={} inp={:?}",
            bot.last_reason, inp
        );
    }

    #[test]
    fn rope_top_underfoot_clears_climb_and_steps() {
        use crate::game::observation::{OBS_FLOOR_START, OBS_ROPE_START};

        // 绳顶小台：脚下有台 + 绳仍近 + 更高一层台阶（复现 endless seek_climb_air up）。
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_FLOOR_START] = 0.0;
        obs[OBS_FLOOR_START + 1] = 0.02;
        obs[OBS_FLOOR_START + 2] = 0.18;
        obs[OBS_FLOOR_START + 3] = 0.05;
        // 更高一层台（step_up 约 60px，须 < MAX_UP=80）
        obs[OBS_FLOOR_START + 4] = 0.07;
        obs[OBS_FLOOR_START + 5] = -0.06;
        obs[OBS_FLOOR_START + 6] = 0.20;
        obs[OBS_FLOOR_START + 7] = 0.05;
        obs[OBS_ROPE_START] = 0.005;
        obs[OBS_ROPE_START + 1] = 0.05;
        obs[OBS_ROPE_START + 2] = 0.02;
        obs[OBS_ROPE_START + 3] = 0.25;

        let mut sense = VisionSenseState::default();
        sense.climbing = true;
        sense.facing = -1.0;
        sense.est_x = 1477.0;
        sense.est_y = 985.0;

        // 即使粘性尚未清，本帧 underfoot → 视为落地
        let ctx0 = RuleBotCtx::from_vision(&obs, &sense);
        assert!(ctx0.on_ground, "underfoot must count as on_ground");
        assert!(!ctx0.climbing, "underfoot must clear effective climbing");
        assert!(ctx0.step_up_dx.is_some(), "rope-top should see step_up");

        sense.prepare(&obs);
        sense.prepare(&obs);
        assert!(
            !sense.climbing,
            "prepare must land-off after underfoot+step"
        );

        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.farm_y = 985.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(1477.0, 985.0).1;
        bot.patrol_dir = -1.0;
        bot.climb_up_stall = 10;

        let ctx = RuleBotCtx::from_vision(&obs, &sense);
        let inp = bot.decide(ctx);
        assert!(
            !inp.up || inp.jump || inp.left || inp.right,
            "must leave rope-top (not endless up); reason={} inp={:?}",
            bot.last_reason,
            inp
        );
        assert_ne!(bot.last_reason, "seek_climb_air");
        assert_ne!(bot.last_reason, "seek_climb_hold");
    }

    #[test]
    fn climb_up_stall_steps_off_when_no_underfoot() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        let (px, py) = (1477.0_f32, 985.0);
        bot.farm_y = py;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(px, py).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = -1.0;
        bot.climb_up_stall = 10;

        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: -1.0,
            on_ground: false,
            climbing: true,
            player_x: px,
            player_y: py,
            walk_right_ok: None,
            walk_left_ok: None,
            drop_ahead_right: None,
            drop_ahead_left: None,
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: Some(ClimbHint {
                dx: 0.0,
                dir: ClimbDir::Up,
            }),
            step_up_dx: Some(92.0),
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert_eq!(bot.last_reason, "seek_climb_top");
        assert!(
            inp.jump || inp.left || inp.right,
            "stall with step_up must leave climb; inp={:?}",
            inp
        );
        assert!(!inp.up || inp.jump);
    }

    #[test]
    fn frozen_position_escalates_unified_escape() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 805.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(621.0, 805.0).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.patrol_dir = -1.0;
        bot.last_x = 621.0;
        bot.last_y = 805.0;

        let obs = [0.0_f32; OBS_DIM];
        let ctx_at = |x: f32| RuleBotCtx {
            obs: &obs,
            facing: -1.0,
            on_ground: true,
            climbing: false,
            player_x: x,
            player_y: 805.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };

        for _ in 0..24 {
            let _ = bot.decide(ctx_at(621.0));
        }
        let inp = bot.decide(ctx_at(621.0));
        assert!(
            bot.escape_phase != EscapePhase::Normal,
            "frozen position must activate unified escape"
        );
        assert!(
            bot.last_reason.starts_with("escape_"),
            "reason={} inp={:?}",
            bot.last_reason,
            inp
        );
    }

    #[test]
    fn seek_rejects_step_up_back_to_cleared_farm() {
        // 复现 865→805：已清台后掉到略低处，step_up 拉回同层。
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 805.0;
        bot.farm_band_sticky = 0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(956.0, 865.0).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.explore_mode_ticks = 5;
        bot.ticks_on_band = 5;
        bot.patrol_dir = 1.0;

        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: -1.0,
            on_ground: true,
            climbing: false,
            player_x: 956.0,
            player_y: 865.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: Some(-38.0),
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert_ne!(
            bot.last_reason, "seek_step_up",
            "must not jump back to cleared farm; reason={} inp={:?}",
            bot.last_reason, inp
        );
    }

    #[test]
    fn seek_force_leaves_after_long_band_sweep() {
        // 视觉无悬崖时扫台过久仍要跳离（复现 cliff/drop 全 false 的 805 ping-pong）。
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 805.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(800.0, 805.0).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.explore_mode_ticks = BAND_STAGNATION_DECISIONS;
        bot.ticks_on_band = BAND_STAGNATION_DECISIONS;
        bot.patrol_dir = 1.0;

        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 800.0,
            player_y: 805.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let mut saw_vertical_escape = false;
        for _ in 0..40 {
            let inp = bot.decide(ctx);
            saw_vertical_escape |= inp.jump || inp.up || inp.down;
        }
        assert!(
            saw_vertical_escape,
            "stagnant Seek must escalate beyond horizontal walking; phase={}",
            bot.escape_phase_name()
        );
        assert_ne!(bot.escape_phase, EscapePhase::Normal);
    }

    #[test]
    fn climb_transit_bans_same_ladder_reclimb() {
        // 复现 685↔565：爬上后立刻 seek_drop / 再 seek_climb 同一根梯子。
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 565.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(1161.0, 565.0).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.explore_mode_ticks = 5;
        bot.ticks_on_band = 5;
        bot.patrol_dir = -1.0;
        bot.last_x = 1161.0;
        bot.last_y = 565.0;
        // 模拟刚沿 x≈1161 爬上来结束。
        bot.climb_ban_x = 1161.0;
        bot.climb_ban_ticks = CLIMB_TRANSIT_BAN_TICKS;
        bot.was_climbing = false;

        let obs = [0.0_f32; OBS_DIM];
        let ctx_drop = RuleBotCtx {
            obs: &obs,
            facing: -1.0,
            on_ground: true,
            climbing: false,
            player_x: 1113.0,
            player_y: 565.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(false),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(true),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: Some(ClimbHint {
                dx: 48.0,
                dir: ClimbDir::Down,
            }),
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx_drop);
        assert_ne!(
            bot.last_reason, "seek_drop",
            "must not drop back onto banned ladder; reason={} inp={:?}",
            bot.last_reason, inp
        );
        assert_ne!(bot.last_reason, "seek_climb");

        let ctx_reclimb = RuleBotCtx {
            player_x: 1092.0,
            player_y: 685.0,
            facing: 1.0,
            walk_left_ok: Some(true),
            drop_ahead_left: Some(false),
            climb: Some(ClimbHint {
                dx: 70.0,
                dir: ClimbDir::Up,
            }),
            ..ctx_drop
        };
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.last_y_band = visit_key(1092.0, 685.0).1;
        bot.climb_ban_x = 1161.0;
        bot.climb_ban_ticks = CLIMB_TRANSIT_BAN_TICKS;
        let inp2 = bot.decide(ctx_reclimb);
        assert_ne!(
            bot.last_reason, "seek_climb",
            "must not re-climb banned ladder; reason={} inp={:?}",
            bot.last_reason, inp2
        );
    }

    #[test]
    fn seek_loop_break_at_banned_ladder_walks_away() {
        // 复现 1286,685：刚下梯 ban 未消，loop_break 不应 jump+up 空转。
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 685.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(1286.0, 685.0).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.ticks_without_new_cell = SEEK_LOOP_BREAK_CELLS;
        bot.ticks_on_band = BAND_STAGNATION_DECISIONS;
        bot.patrol_dir = -1.0;
        bot.climb_ban_x = 1161.0;
        bot.climb_ban_ticks = CLIMB_TRANSIT_BAN_TICKS;
        bot.last_x = 1286.0;
        bot.last_y = 685.0;
        bot.visited.insert(visit_key(1286.0, 685.0));
        bot.escape_phase = EscapePhase::SeekExit;
        bot.escape_dir = 1.0;
        bot.escape_anchor_x = 1286.0;
        bot.escape_anchor_y = 685.0;

        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: -1.0,
            on_ground: true,
            climbing: false,
            player_x: 1286.0,
            player_y: 685.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: Some(ClimbHint {
                dx: -125.0,
                dir: ClimbDir::Up,
            }),
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert_eq!(bot.last_reason, "escape_seek_exit");
        assert!(
            inp.right,
            "must walk away from banned ladder; inp={:?}",
            inp
        );
        assert!(
            !inp.jump,
            "must not jump+up at banned ladder mouth; inp={:?}",
            inp
        );
        assert!(
            !inp.up,
            "must not press up at banned ladder mouth; inp={:?}",
            inp
        );
    }

    #[test]
    fn seek_loop_break_at_ladder_top_walks_not_jumps() {
        // 复现 1161,565 梯顶：Down 提示 + loop_break 只应水平走开。
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 565.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(1161.0, 565.0).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.ticks_without_new_cell = SEEK_LOOP_BREAK_CELLS;
        bot.ticks_on_band = BAND_STAGNATION_DECISIONS;
        bot.patrol_dir = 1.0;
        bot.escape_phase = EscapePhase::SeekExit;
        bot.escape_dir = 1.0;
        bot.escape_anchor_x = 1161.0;
        bot.escape_anchor_y = 565.0;

        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 1161.0,
            player_y: 565.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: Some(ClimbHint {
                dx: 48.0,
                dir: ClimbDir::Down,
            }),
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };
        let inp = bot.decide(ctx);
        assert_eq!(bot.last_reason, "escape_seek_exit");
        assert!(
            inp.left || inp.right,
            "must walk off ladder top; inp={:?}",
            inp
        );
        assert!(!inp.jump, "must not jump spam at ladder top; inp={:?}", inp);
        assert!(!inp.up, "must not press up at ladder top; inp={:?}", inp);
    }

    #[test]
    fn stuck_on_rope_top_prefers_step_up_not_climb_align() {
        // 复现 (1020,904)：绳顶 + step≈-100 + climb 右对齐 → stuck_recovery 只 right 永不跳。
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 904.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(1020.0, 904.0).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.escape_phase = EscapePhase::SeekExit;
        bot.escape_anchor_x = 1020.0;
        bot.escape_anchor_y = 904.0;
        bot.escape_dir = 1.0;
        bot.last_x = 1020.0;
        bot.last_y = 904.0;
        bot.patrol_dir = 1.0;

        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 1020.0,
            player_y: 904.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: Some(ClimbHint {
                dx: 100.0,
                dir: ClimbDir::Up,
            }),
            step_up_dx: Some(-100.0),
            farm_band_mobs: false,
            visual_confidence: 11,
        };

        let inp = bot.decide(ctx);
        assert_eq!(bot.last_reason, "escape_seek_exit");
        assert!(
            inp.left || inp.jump,
            "must step/jump toward ledge, not climb-align right: {:?}",
            inp
        );
        assert!(
            !inp.right || inp.jump,
            "must not only walk right: {:?}",
            inp
        );
    }

    #[test]
    fn failed_force_exit_advances_without_immediate_retry() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 805.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(800.0, 805.0).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.escape_phase = EscapePhase::ForceVertical;
        bot.escape_ticks = ESCAPE_FORCE_TICKS - 1;
        bot.escape_anchor_x = 800.0;
        bot.escape_anchor_y = 805.0;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 800.0,
            player_y: 805.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };

        let first = bot.decide(ctx);
        assert_eq!(bot.escape_phase, EscapePhase::AlternateExit);
        assert_eq!(bot.last_reason, "escape_alternate");
        assert!(first.jump);
        assert!(bot.failed_exit_count() >= 1);

        let _ = bot.decide(ctx);
        assert_eq!(
            bot.escape_phase,
            EscapePhase::AlternateExit,
            "failed ForceVertical must not restart immediately"
        );
    }

    #[test]
    fn visual_height_change_clears_escape_history() {
        let mut bot = RuleBot::default();
        bot.initialized = true;
        bot.farm_y = 805.0;
        bot.spawn_y = 1225.0;
        bot.spawn_y_band = visit_key(500.0, 1225.0).1;
        bot.last_y_band = visit_key(800.0, 805.0).1;
        bot.explore_mode = ExploreMode::SeekVertical;
        bot.escape_phase = EscapePhase::AlternateExit;
        bot.escape_anchor_x = 800.0;
        bot.escape_anchor_y = 805.0;
        bot.failed_exits = 7;
        let obs = [0.0_f32; OBS_DIM];
        let ctx = RuleBotCtx {
            obs: &obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            player_x: 800.0,
            player_y: 735.0,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: Some(false),
            drop_ahead_left: Some(false),
            mob_in_melee: false,
            mob_on_attackable_footing: false,
            engage: None,
            engage_wide: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            visual_confidence: 11,
        };

        let _ = bot.decide(ctx);
        assert_eq!(bot.escape_phase, EscapePhase::Normal);
        assert_eq!(bot.failed_exit_count(), 0);
    }
}
