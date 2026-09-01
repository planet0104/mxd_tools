//! NEAT 训练计分：正分为主。
//!
//! 动作宏化之后，「左右同时按 / up+down 互抵 / 原地蹦跳 / 单向狂按」这类无效组合已在
//! 结构上不可能出现，旧版为对抗乱按键堆的十几项重罚全部删除。现在的分数构成是
//! 「打怪 + 捡钱 + 换平台 + 活着」的正向累加，只保留四项小额惩罚（空砍、挨打、顶墙、挂机），
//! 保证摆烂个体分数趋近 0 而不是 -500，让选择压力落在真实产出差异上。
//!
//! 数据源：拾取必须与 YOLO 可见框对齐；顶墙/位移一律 OCR 脚点。击杀/命中来自 sim 记忆
//! （仅用于计分，不进网络输入）。

use std::collections::HashSet;

use super::observation::{
    obs_enemy_in_attack_range, obs_floor_ahead, obs_floor_drop_ahead, obs_has_nearby_platform_enemy,
    obs_has_platform_in_direction, obs_nearest_same_level_enemy_px, obs_platform_edge, OBS_DIM,
    OBS_PROPRIO_START,
};
use super::types::{DropKind, WINDOW_H, WINDOW_W};
use crate::yolo::Detection;

const MESO_LABEL: &str = "金币";
const POTION_LABEL: &str = "药水";

/// 主产出。
const PTS_MESO_UNIT: f32 = 15.0;
const PTS_POTION_PICKUP: f32 = 12.0;
const PTS_MOB_HIT: f32 = 3.0;
const PTS_MOB_KILL: f32 = 25.0;
/// 击杀后及时把掉落捡走。
const PTS_KILL_LOOT_CHAIN: f32 = 25.0;
const KILL_LOOT_CHAIN_TICKS: u64 = 180;

/// 探索：新网格、新高度带、离开出生平台、换到更多平台。
const PTS_NEW_CELL: f32 = 1.0;
const NEW_CELL_CAP: f32 = 30.0;
const PTS_NEW_Y_BAND: f32 = 8.0;
const PTS_LEAVE_SPAWN_BAND: f32 = 55.0;
/// 离开出生点至少水平移动这么多，才算 leave_spawn（防原地换带刷分）。
const LEAVE_SPAWN_MIN_DX_PX: f32 = 96.0;
const PTS_EXTRA_PLATFORM: f32 = 40.0;
const PLATFORM_CHANGE_CAP: f32 = 200.0;
const PTS_MULTI_BAND: f32 = 35.0;
const MULTI_BAND_BONUS_CAP: f32 = 105.0;

/// 存活基线：无条件给，保证「什么都没干」也是小正分而不是大负分。
const PTS_SURVIVAL_PER_SEC: f32 = 1.5;
const SURVIVAL_BONUS_CAP: f32 = 60.0;

/// 空砍：按「一刀」计，不按 tick 计（动作宏一次只出一刀）。
const PENALTY_EMPTY_SWING: f32 = 3.0;
const EMPTY_SWING_PENALTY_CAP: f32 = 120.0;
/// 被怪碰到。
const PENALTY_TOUCH_HIT: f32 = 8.0;
const TOUCH_HIT_PENALTY_CAP: f32 = 120.0;
/// OCR 判定「按了方向却没动」：小额软罚 + 长时间硬顶才认输。
const PENALTY_BLOCKED_WALK: f32 = 0.5;
const BLOCKED_WALK_PENALTY_CAP: f32 = 60.0;
const WALL_PUSH_FORFEIT_TICKS: u32 = 90;

/// 挂机：位置长时间不变。
pub const STAGNATION_TICKS: u32 = 150;
const STAGNATION_MOVE_THRESHOLD: f32 = 48.0;
const STAGNATION_PENALTY: f32 = 20.0;
const STAGNATION_PENALTY_CAP: f32 = 120.0;
pub const IDLE_FORFEIT_GRACE_TICKS: u32 = 300;

/// 本台连续无怪多久算「已清层」。必须足够长，防止个体走远几步就「清台」刷探索分。
const CLEAR_BAND_NO_ENEMY_TICKS: u32 = 300;

// ── 行为引导：教网络「先清台 → 再走到边缘 → 跳向有平台的方向」 ──

/// 清台一次性奖励：当前平台近处怪全灭后触发。
const PTS_PLATFORM_CLEAR_ONCE: f32 = 35.0;
/// 朝同台敌人方向走（每 tick 微量，引导靠拢）。
const PTS_WALK_TOWARD_ENEMY: f32 = 2.0;
const WALK_TOWARD_ENEMY_CAP: f32 = 120.0;
/// 清台后走向平台边缘（每 tick，引导去跳台）。
const PTS_WALK_TOWARD_EDGE: f32 = 0.3;
const WALK_TOWARD_EDGE_CAP: f32 = 100.0;
/// 在边缘跳向有平台的方向（一次性）。
const PTS_JUMP_TOWARD_PLATFORM: f32 = 12.0;
/// 不在边缘乱跳惩罚（每跳一次）。
const PENALTY_JUMP_NOT_EDGE: f32 = 5.0;
const JUMP_NOT_EDGE_CAP: f32 = 80.0;
/// 有怪时跳跃惩罚（无论是否在边缘：清怪前禁止跳台）。
const PENALTY_JUMP_WHEN_ENEMIES: f32 = 8.0;
const JUMP_WHEN_ENEMIES_CAP: f32 = 120.0;
/// 行走时前方无地板（即将掉下边缘）但未跳：小罚，引导在边缘起跳。
const PENALTY_WALK_INTO_VOID: f32 = 1.0;
const WALK_INTO_VOID_CAP: f32 = 30.0;
/// 有怪时朝远离敌人方向走（每 tick）。
const PENALTY_WALK_AWAY_FROM_ENEMY: f32 = 0.3;
const WALK_AWAY_FROM_ENEMY_CAP: f32 = 60.0;
/// 有怪时尝试攻击（无论是否命中）：鼓励网络使用攻击动作。
const PTS_ATTACK_ATTEMPT: f32 = 1.0;
const ATTACK_ATTEMPT_CAP: f32 = 60.0;
/// 在边缘跳向前方无平台处：惩罚，防止卡死在地图边缘反复跳。
/// 无 cap — 永远不能靠跳够次数来豁免惩罚。
const PENALTY_JUMP_AT_DEAD_END: f32 = 20.0;
/// 返回最近离开的 y-band 惩罚：防止在相邻平台间反复弹跳刷分。
const PENALTY_Y_BAND_OSCILLATION: f32 = 15.0;
const Y_BAND_OSCILLATION_TICKS: u64 = 300; // 5 秒内返回即罚
/// 跳台后落回原地（YOLO 看到平台但实际够不到）：惩罚。
const PENALTY_FAILED_JUMP: f32 = 15.0;
/// 教师奖励：动作匹配启发式"正确动作"时给。
const PTS_TEACHER_BONUS: f32 = 2.0;
const TEACHER_CAP: f32 = 120.0;

/// 与 rule_bot 探索网格一致（80×120 px）。
const X_CELL_PX: f32 = 80.0;
const ALTITUDE_BAND_PX: f32 = 120.0;

/// sim 记忆 shaping 权重（击杀/命中；主分仍为 YOLO 可见拾取）。
const DEFAULT_MEMORY_WEIGHT: f32 = 0.5;

#[derive(Debug, Clone, Copy)]
pub struct FitnessShapingConfig {
    pub memory_weight: f32,
}

impl Default for FitnessShapingConfig {
    fn default() -> Self {
        Self {
            memory_weight: DEFAULT_MEMORY_WEIGHT,
        }
    }
}

impl FitnessShapingConfig {
    pub fn disabled() -> Self {
        Self { memory_weight: 0.0 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct VisibleLoot {
    pub meso: Vec<[f32; 4]>,
    pub potions: Vec<[f32; 4]>,
}

#[derive(Debug, Clone, Copy)]
pub struct FitnessPreviewDiag {
    pub score: f32,
    pub empty_attack_penalty: f32,
    pub blocked_walk_penalty: f32,
    pub touch_hit_penalty: f32,
    pub explore_score: f32,
    pub stagnation_ticks: u32,
    pub wall_push_ticks: u32,
    pub no_near_enemy_ticks: u32,
    pub empty_attack_streak: u32,
    pub band_cleared: bool,
    pub left_spawn: bool,
    pub y_bands: u32,
    pub pressed_left: bool,
    pub pressed_right: bool,
    pub pressing_stuck: bool,
    pub idle_forfeit: bool,
    pub kills: u32,
    pub meso_events: u32,
    pub jump_not_edge_penalty: f32,
    pub jump_toward_platform_bonus: f32,
    pub walk_toward_enemy_score: f32,
    pub walk_toward_edge_score: f32,
    pub walk_into_void_penalty: f32,
    pub jump_when_enemies_penalty: f32,
    pub walk_away_from_enemy_penalty: f32,
    pub attack_attempt_score: f32,
    pub jump_at_dead_end_penalty: f32,
    pub platform_clear_bonus_given: bool,
    pub failed_jump_penalty: f32,
    pub teacher_score: f32,
}

#[derive(Debug, Clone)]
pub struct TrainingFitness {
    pub score: f32,
    pub pickup_score: f32,
    pub explore_score: f32,
    pub meso_events: u32,
    pub potion_events: u32,
    pub meso_units: u32,
    pub mob_hit_events: u32,
    pub mob_kill_events: u32,
    pub stagnation_penalty: f32,
    pub empty_attack_penalty: f32,
    pub blocked_walk_penalty: f32,
    pub touch_hit_penalty: f32,
    pub survival_bonus: f32,
    pub platform_change_score: f32,
    pub leave_spawn_bonus: f32,
    pub idle_forfeit: bool,
    /// 本帧 OCR 判定「按了方向却卡住」。
    pressing_while_stuck: bool,
    pressed_left: bool,
    pressed_right: bool,
    shaping: FitnessShapingConfig,
    last_obs: [f32; OBS_DIM],
    last_visible: VisibleLoot,
    stagnation_anchor_x: f32,
    stagnation_anchor_y: f32,
    stagnation_ticks: u32,
    stagnation_initialized: bool,
    last_x: f32,
    last_y: f32,
    last_pickup_tick: u64,
    last_mob_kill_tick: u64,
    last_episode_tick: u64,
    episode_finalized: bool,
    visited_cells: HashSet<(i32, i32)>,
    visited_y_bands: HashSet<i32>,
    spawn_y_band: Option<i32>,
    /// 之前所在的 y-band（用于振荡检测）。
    last_y_band: Option<i32>,
    /// 再上一个 y-band（振荡检测：检测 A→B→A 模式）。
    prev_y_band: Option<i32>,
    /// y-band 最近一次变化的 tick。
    y_band_changed_at_tick: u64,
    /// 已跳过的平台对 (from_y_band, direction_sign)，跳台奖励只给一次。
    jumped_pairs: HashSet<(i32, i32)>,
    /// 上一个输入帧是否包含跳跃（用于检测跳跃结束）。
    was_jumping: bool,
    /// 跳跃开始时所在的 y-band（用于检测失败跳台）。
    jump_start_y_band: Option<i32>,
    /// 失败跳台惩罚累计（诊断用）。
    failed_jump_penalty: f32,
    /// 教师奖励累计（诊断用）。
    teacher_score: f32,
    spawn_x: Option<f32>,
    left_spawn_band: bool,
    no_same_level_enemy_ticks: u32,
    wall_push_ticks: u32,
    /// 上一 tick 是否按着攻击键：空砍按「刀」计而不是按 tick 计。
    prev_attack: bool,
    empty_swings: u32,
    empty_attack_streak: u32,
    /// 上一 tick 是否按着跳跃键：跳边判定按「次」计。
    prev_jump: bool,
    /// 非边缘跳累计惩罚。
    jump_not_edge_penalty: f32,
    /// 边缘跳向平台累计奖励。
    jump_toward_platform_bonus: f32,
    /// 走向敌人累计奖励。
    walk_toward_enemy_score: f32,
    /// 清台后走向边缘累计奖励。
    walk_toward_edge_score: f32,
    /// 走向虚空（该跳却没跳）累计惩罚。
    walk_into_void_penalty: f32,
    /// 有怪时跳跃累计惩罚。
    jump_when_enemies_penalty: f32,
    /// 有怪时远离敌人累计惩罚。
    walk_away_from_enemy_penalty: f32,
    /// 有怪时尝试攻击累计奖励。
    attack_attempt_score: f32,
    /// 在死胡同边缘跳跃累计惩罚。
    jump_at_dead_end_penalty: f32,
    /// 当前平台是否已给过清台奖励。
    platform_clear_bonus_given: bool,
}

impl Default for TrainingFitness {
    fn default() -> Self {
        Self::with_shaping(FitnessShapingConfig::default())
    }
}

impl TrainingFitness {
    pub fn with_shaping(shaping: FitnessShapingConfig) -> Self {
        Self {
            score: 0.0,
            pickup_score: 0.0,
            explore_score: 0.0,
            meso_events: 0,
            potion_events: 0,
            meso_units: 0,
            mob_hit_events: 0,
            mob_kill_events: 0,
            stagnation_penalty: 0.0,
            empty_attack_penalty: 0.0,
            blocked_walk_penalty: 0.0,
            touch_hit_penalty: 0.0,
            survival_bonus: 0.0,
            platform_change_score: 0.0,
            leave_spawn_bonus: 0.0,
            idle_forfeit: false,
            pressing_while_stuck: false,
            pressed_left: false,
            pressed_right: false,
            shaping,
            last_obs: [0.0; OBS_DIM],
            last_visible: VisibleLoot::default(),
            stagnation_anchor_x: 0.0,
            stagnation_anchor_y: 0.0,
            stagnation_ticks: 0,
            stagnation_initialized: false,
            last_x: 0.0,
            last_y: 0.0,
            last_pickup_tick: 0,
            last_mob_kill_tick: 0,
            last_episode_tick: 0,
            episode_finalized: false,
            visited_cells: HashSet::new(),
            visited_y_bands: HashSet::new(),
            spawn_y_band: None,
            last_y_band: None,
            prev_y_band: None,
            y_band_changed_at_tick: 0,
            jumped_pairs: HashSet::new(),
            was_jumping: false,
            jump_start_y_band: None,
            failed_jump_penalty: 0.0,
            teacher_score: 0.0,
            spawn_x: None,
            left_spawn_band: false,
            no_same_level_enemy_ticks: 0,
            wall_push_ticks: 0,
            prev_attack: false,
            empty_swings: 0,
            empty_attack_streak: 0,
            prev_jump: false,
            jump_not_edge_penalty: 0.0,
            jump_toward_platform_bonus: 0.0,
            walk_toward_enemy_score: 0.0,
            walk_toward_edge_score: 0.0,
            walk_into_void_penalty: 0.0,
            jump_when_enemies_penalty: 0.0,
            walk_away_from_enemy_penalty: 0.0,
            attack_attempt_score: 0.0,
            jump_at_dead_end_penalty: 0.0,
            platform_clear_bonus_given: false,
        }
    }

    pub fn configure_shaping(&mut self, shaping: FitnessShapingConfig) {
        self.shaping = shaping;
    }

    pub fn shaping_config(&self) -> FitnessShapingConfig {
        self.shaping
    }

    pub fn record_visible_drops(&mut self, detections: &[Detection]) {
        let mut vis = VisibleLoot::default();
        for d in detections {
            let b = [d.x1, d.y1, d.x2, d.y2];
            match d.label {
                MESO_LABEL => vis.meso.push(b),
                POTION_LABEL => vis.potions.push(b),
                _ => {}
            }
        }
        self.last_visible = vis;
    }

    pub fn set_last_observation(&mut self, obs: &[f32]) {
        let n = obs.len().min(OBS_DIM);
        self.last_obs[..n].copy_from_slice(&obs[..n]);
    }

    pub fn try_score_input(&mut self, input: &super::InputFrame, _episode_tick: u64) {
        self.pressing_while_stuck = false;
        if input.left && !input.right {
            self.pressed_left = true;
        }
        if input.right && !input.left {
            self.pressed_right = true;
        }

        // ── 失败跳台检测：跳完后落回原地 → 惩罚 ──
        if input.jump {
            if !self.was_jumping {
                self.was_jumping = true;
                self.jump_start_y_band = Some(visit_key(self.last_x, self.last_y).1);
            }
        } else if self.was_jumping {
            self.was_jumping = false;
            if let Some(start_band) = self.jump_start_y_band.take() {
                let current_band = visit_key(self.last_x, self.last_y).1;
                if current_band == start_band {
                    // 跳台失败：落回同一平台（YOLO 看到平台但实际够不到）
                    self.failed_jump_penalty += PENALTY_FAILED_JUMP;
                    self.score -= PENALTY_FAILED_JUMP;
                }
            }
        }

        // ── 空砍只在按下那一刻结算 ──
        let swing = input.attack && !self.prev_attack;
        self.prev_attack = input.attack;
        if swing {
            let has_enemies = obs_has_nearby_platform_enemy(&self.last_obs);
            let facing = self.attack_facing(input);
            if obs_enemy_in_attack_range(&self.last_obs, facing) {
                self.empty_attack_streak = 0;
            } else {
                self.empty_swings += 1;
                self.empty_attack_streak = self.empty_attack_streak.saturating_add(1);
                if self.empty_attack_penalty < EMPTY_SWING_PENALTY_CAP {
                    let applied = PENALTY_EMPTY_SWING
                        .min(EMPTY_SWING_PENALTY_CAP - self.empty_attack_penalty);
                    self.empty_attack_penalty += applied;
                    self.score -= applied;
                }
            }
            // 有怪时尝试攻击就给奖励，鼓励网络使用攻击动作（与空砍惩罚独立）。
            if has_enemies && self.attack_attempt_score < ATTACK_ATTEMPT_CAP {
                let applied =
                    PTS_ATTACK_ATTEMPT.min(ATTACK_ATTEMPT_CAP - self.attack_attempt_score);
                self.attack_attempt_score += applied;
                self.score += applied;
            }
        }

        // ── 跳跃边缘引导 ──
        let jump_pressed = input.jump && !self.prev_jump;
        self.prev_jump = input.jump;
        if jump_pressed {
            let walking_left = input.left && !input.right;
            let walking_right = input.right && !input.left;
            let at_left_edge = obs_platform_edge(&self.last_obs, -1.0);
            let at_right_edge = obs_platform_edge(&self.last_obs, 1.0);
            let at_edge = (walking_left && at_left_edge) || (walking_right && at_right_edge);
            let has_enemies = obs_has_nearby_platform_enemy(&self.last_obs);

            if has_enemies {
                // 有怪时：禁止跳跃，无论是否在边缘。清怪是第一要务。
                if self.jump_when_enemies_penalty < JUMP_WHEN_ENEMIES_CAP {
                    let applied = PENALTY_JUMP_WHEN_ENEMIES
                        .min(JUMP_WHEN_ENEMIES_CAP - self.jump_when_enemies_penalty);
                    self.jump_when_enemies_penalty += applied;
                    self.score -= applied;
                }
            } else if at_edge {
                // 清台后：在边缘跳向有平台的方向 → 奖励，跳向无平台处 → 惩罚
                let dir = if walking_left { -1.0_f32 } else { 1.0_f32 };
                if obs_has_platform_in_direction(&self.last_obs, dir, WINDOW_W, WINDOW_H) {
                    // 每个 (y_band, 方向) 对只奖励一次，防止反复弹跳刷分。
                    let y_band = visit_key(self.last_x, self.last_y).1;
                    let dir_sign = dir as i32;
                    if self.jumped_pairs.insert((y_band, dir_sign)) {
                        self.jump_toward_platform_bonus += PTS_JUMP_TOWARD_PLATFORM;
                        self.score += PTS_JUMP_TOWARD_PLATFORM;
                    }
                } else {
                    // 防止卡死在地图边缘反复跳（无 cap，永远不能豁免）
                    self.jump_at_dead_end_penalty += PENALTY_JUMP_AT_DEAD_END;
                    self.score -= PENALTY_JUMP_AT_DEAD_END;
                }
            } else if walking_left || walking_right {
                // 清台但非边缘乱跳：小额惩罚
                if self.jump_not_edge_penalty < JUMP_NOT_EDGE_CAP {
                    let applied = PENALTY_JUMP_NOT_EDGE
                        .min(JUMP_NOT_EDGE_CAP - self.jump_not_edge_penalty);
                    self.jump_not_edge_penalty += applied;
                    self.score -= applied;
                }
            }
        }

        // ── 行走引导：朝敌人 / 朝有平台的边缘 / 远离敌人惩罚 ──
        let walking_left = input.left && !input.right;
        let walking_right = input.right && !input.left;
        if walking_left || walking_right {
            let dir = if walking_left { -1.0_f32 } else { 1.0_f32 };
            let at_edge = if walking_left {
                obs_platform_edge(&self.last_obs, -1.0)
            } else {
                obs_platform_edge(&self.last_obs, 1.0)
            };
            let has_enemies = obs_has_nearby_platform_enemy(&self.last_obs);

            if has_enemies {
                // 有敌人在附近：朝敌人走给奖励，远离敌人给惩罚
                if let Some((dx, _dy)) =
                    obs_nearest_same_level_enemy_px(&self.last_obs, WINDOW_W, WINDOW_H)
                {
                    if dx.signum() == dir {
                        // 朝敌人走：奖励
                        if self.walk_toward_enemy_score < WALK_TOWARD_ENEMY_CAP {
                            let applied = PTS_WALK_TOWARD_ENEMY
                                .min(WALK_TOWARD_ENEMY_CAP - self.walk_toward_enemy_score);
                            self.walk_toward_enemy_score += applied;
                            self.score += applied;
                        }
                    } else if dx.abs() > 4.0 {
                        // 远离敌人（且不是贴脸微调）：惩罚
                        if self.walk_away_from_enemy_penalty < WALK_AWAY_FROM_ENEMY_CAP {
                            let applied = PENALTY_WALK_AWAY_FROM_ENEMY.min(
                                WALK_AWAY_FROM_ENEMY_CAP - self.walk_away_from_enemy_penalty,
                            );
                            self.walk_away_from_enemy_penalty += applied;
                            self.score -= applied;
                        }
                    }
                }
            } else if self.band_is_cleared() {
                // 清台后：只奖励朝有平台的方向走，惩罚朝无平台方向走。
                let dir_has = obs_has_platform_in_direction(&self.last_obs, dir, WINDOW_W, WINDOW_H);
                let other_has = obs_has_platform_in_direction(&self.last_obs, -dir, WINDOW_W, WINDOW_H);

                if dir_has && !at_edge {
                    // 朝有平台边缘走 → 奖励
                    if self.walk_toward_edge_score < WALK_TOWARD_EDGE_CAP {
                        let applied = PTS_WALK_TOWARD_EDGE
                            .min(WALK_TOWARD_EDGE_CAP - self.walk_toward_edge_score);
                        self.walk_toward_edge_score += applied;
                        self.score += applied;
                    }
                } else if !dir_has && other_has {
                    // 朝无平台方向走，但另一边有平台 → 惩罚（引导掉头）
                    if self.walk_away_from_enemy_penalty < WALK_AWAY_FROM_ENEMY_CAP {
                        let applied = PENALTY_WALK_AWAY_FROM_ENEMY.min(
                            WALK_AWAY_FROM_ENEMY_CAP - self.walk_away_from_enemy_penalty,
                        );
                        self.walk_away_from_enemy_penalty += applied;
                        self.score -= applied;
                    }
                }

                // 在边缘却没跳 → 小罚，引导在边缘起跳
                if at_edge && !input.jump && self.walk_into_void_penalty < WALK_INTO_VOID_CAP {
                    let applied = PENALTY_WALK_INTO_VOID
                        .min(WALK_INTO_VOID_CAP - self.walk_into_void_penalty);
                    self.walk_into_void_penalty += applied;
                    self.score -= applied;
                }

                // 前方无地板且无 jump 缓冲 → 即将掉下边缘，引导起跳
                if !at_edge && !obs_floor_ahead(&self.last_obs, dir) {
                    if !obs_floor_drop_ahead(&self.last_obs, dir)
                        && !input.jump
                        && self.walk_into_void_penalty < WALK_INTO_VOID_CAP
                    {
                        let applied = PENALTY_WALK_INTO_VOID
                            .min(WALK_INTO_VOID_CAP - self.walk_into_void_penalty);
                        self.walk_into_void_penalty += applied;
                        self.score -= applied;
                    }
                }
            }
        }

        self.tick_blocked_walk(input);
    }

    fn tick_blocked_walk(&mut self, input: &super::InputFrame) {
        let blocked_left = self.proprio_flag(2);
        let blocked_right = self.proprio_flag(3);
        let pressing = (input.left && !input.right && blocked_left)
            || (input.right && !input.left && blocked_right);
        if !pressing {
            self.wall_push_ticks = 0;
            return;
        }
        self.pressing_while_stuck = true;
        self.wall_push_ticks = self.wall_push_ticks.saturating_add(1);
        if self.wall_push_ticks >= WALL_PUSH_FORFEIT_TICKS {
            self.idle_forfeit = true;
        }
        if self.blocked_walk_penalty < BLOCKED_WALK_PENALTY_CAP {
            let applied =
                PENALTY_BLOCKED_WALK.min(BLOCKED_WALK_PENALTY_CAP - self.blocked_walk_penalty);
            self.blocked_walk_penalty += applied;
            self.score -= applied;
        }

        // ── 教师奖励：动作匹配启发式"正确动作"时给 +2 ──
        if self.teacher_score < TEACHER_CAP {
            if let Some(_) = self.teacher_action_match(input) {
                self.teacher_score += PTS_TEACHER_BONUS;
                self.score += PTS_TEACHER_BONUS;
            }
        }
    }

    /// 返回 Some(()) 如果当前动作匹配启发式"正确动作"。
    fn teacher_action_match(&self, input: &super::InputFrame) -> Option<()> {
        let obs = &self.last_obs;
        // 有怪时：朝最近的平台怪走。
        if obs_has_nearby_platform_enemy(obs) {
            if let Some((dx, _)) =
                obs_nearest_same_level_enemy_px(obs, crate::game::WINDOW_W, crate::game::WINDOW_H)
            {
                if dx < 0.0 && input.left && !input.right && !input.jump && !input.attack {
                    return Some(());
                }
                if dx > 0.0 && input.right && !input.left && !input.jump && !input.attack {
                    return Some(());
                }
            }
        }
        // 清台后：朝边缘走，或在边缘朝有平台方向跳。
        if self.band_is_cleared() {
            let at_left = obs_platform_edge(obs, -1.0);
            let at_right = obs_platform_edge(obs, 1.0);
            if at_right && obs_has_platform_in_direction(obs, 1.0, WINDOW_W, WINDOW_H) && input.jump && input.right {
                return Some(());
            }
            if at_left && obs_has_platform_in_direction(obs, -1.0, WINDOW_W, WINDOW_H) && input.jump && input.left {
                return Some(());
            }
            if !at_left && !at_right && !input.jump && !input.attack {
                let has_right = obs_has_platform_in_direction(obs, 1.0, WINDOW_W, WINDOW_H);
                let has_left = obs_has_platform_in_direction(obs, -1.0, WINDOW_W, WINDOW_H);
                // 优先朝有平台的方向走
                if has_right && !has_left && input.right && !input.left {
                    return Some(());
                }
                if has_left && !has_right && input.left && !input.right {
                    return Some(());
                }
                // 两侧都有/都没有平台：任意方向走均可
                if (input.right && !input.left) || (input.left && !input.right) {
                    return Some(());
                }
            }
        }
        None
    }

    fn proprio_flag(&self, offset: usize) -> bool {
        self.last_obs
            .get(OBS_PROPRIO_START + offset)
            .copied()
            .unwrap_or(0.0)
            >= 0.5
    }

    /// 挥砍朝向：本帧方向键优先，否则用 OCR 反馈的上一帧方向。
    fn attack_facing(&self, input: &super::InputFrame) -> f32 {
        if input.right && !input.left {
            return 1.0;
        }
        if input.left && !input.right {
            return -1.0;
        }
        if self.proprio_flag(4) && !self.proprio_flag(5) {
            -1.0
        } else {
            1.0
        }
    }

    fn band_is_cleared(&self) -> bool {
        self.no_same_level_enemy_ticks >= CLEAR_BAND_NO_ENEMY_TICKS
            || (self.mob_kill_events > 0 && !obs_has_nearby_platform_enemy(&self.last_obs))
    }

    pub fn record_mob_hit(&mut self, _episode_tick: u64) {
        self.mob_hit_events += 1;
        if self.shaping.memory_weight <= 0.0 {
            return;
        }
        self.score += self.shaping.memory_weight * PTS_MOB_HIT;
    }

    pub fn record_mob_kill(&mut self, episode_tick: u64) {
        self.mob_kill_events += 1;
        self.last_mob_kill_tick = episode_tick;
        self.note_stagnation_activity();
        if self.shaping.memory_weight <= 0.0 {
            return;
        }
        self.score += self.shaping.memory_weight * PTS_MOB_KILL;
    }

    /// 被怪碰到扣分：引导走位躲怪。
    pub fn record_player_hurt(&mut self, _damage: i32) {
        if self.touch_hit_penalty >= TOUCH_HIT_PENALTY_CAP {
            return;
        }
        let applied = PENALTY_TOUCH_HIT.min(TOUCH_HIT_PENALTY_CAP - self.touch_hit_penalty);
        self.touch_hit_penalty += applied;
        self.score -= applied;
    }

    pub fn tick_stagnation(
        &mut self,
        x: f32,
        y: f32,
        episode_tick: u64,
        on_ground: bool,
        climbing: bool,
    ) -> bool {
        self.last_x = x;
        self.last_y = y;
        self.last_episode_tick = episode_tick;
        self.tick_exploration(x, y, on_ground, climbing);
        self.tick_band_clear();
        if !self.stagnation_initialized {
            self.reset_stagnation_anchor(x, y);
            self.stagnation_initialized = true;
            return false;
        }
        let dx = x - self.stagnation_anchor_x;
        let dy = y - self.stagnation_anchor_y;
        let moved = if climbing {
            dy.abs() > STAGNATION_MOVE_THRESHOLD * 0.5
        } else {
            dx.abs() > STAGNATION_MOVE_THRESHOLD || dy.abs() > STAGNATION_MOVE_THRESHOLD
        };
        if moved {
            self.reset_stagnation_anchor(x, y);
            return false;
        }
        self.stagnation_ticks += 1;
        if self.stagnation_ticks < STAGNATION_TICKS {
            return false;
        }
        if episode_tick >= IDLE_FORFEIT_GRACE_TICKS as u64 && !self.recently_productive(episode_tick)
        {
            self.idle_forfeit = true;
            return true;
        }
        if self.stagnation_penalty < STAGNATION_PENALTY_CAP {
            let applied = STAGNATION_PENALTY.min(STAGNATION_PENALTY_CAP - self.stagnation_penalty);
            self.stagnation_penalty += applied;
            self.score -= applied;
        }
        self.stagnation_ticks = 0;
        false
    }

    /// peak 仅用于训练日志；顶墙帧不刷新，避免把卡墙瞬间的分数当峰值。
    pub fn allows_peak_update(&self) -> bool {
        !self.pressing_while_stuck
    }

    pub fn preview_diag(&self) -> FitnessPreviewDiag {
        FitnessPreviewDiag {
            score: self.score,
            empty_attack_penalty: self.empty_attack_penalty,
            blocked_walk_penalty: self.blocked_walk_penalty,
            touch_hit_penalty: self.touch_hit_penalty,
            explore_score: self.explore_score,
            stagnation_ticks: self.stagnation_ticks,
            wall_push_ticks: self.wall_push_ticks,
            no_near_enemy_ticks: self.no_same_level_enemy_ticks,
            empty_attack_streak: self.empty_attack_streak,
            band_cleared: self.band_is_cleared(),
            left_spawn: self.left_spawn_band,
            y_bands: self.visited_y_bands.len() as u32,
            pressed_left: self.pressed_left,
            pressed_right: self.pressed_right,
            pressing_stuck: self.pressing_while_stuck,
            idle_forfeit: self.idle_forfeit,
            kills: self.mob_kill_events,
            meso_events: self.meso_events,
            jump_not_edge_penalty: self.jump_not_edge_penalty,
            jump_toward_platform_bonus: self.jump_toward_platform_bonus,
            walk_toward_enemy_score: self.walk_toward_enemy_score,
            walk_toward_edge_score: self.walk_toward_edge_score,
            walk_into_void_penalty: self.walk_into_void_penalty,
            jump_when_enemies_penalty: self.jump_when_enemies_penalty,
            walk_away_from_enemy_penalty: self.walk_away_from_enemy_penalty,
            attack_attempt_score: self.attack_attempt_score,
            jump_at_dead_end_penalty: self.jump_at_dead_end_penalty,
            platform_clear_bonus_given: self.platform_clear_bonus_given,
            failed_jump_penalty: self.failed_jump_penalty,
            teacher_score: self.teacher_score,
        }
    }

    fn tick_exploration(&mut self, x: f32, y: f32, on_ground: bool, climbing: bool) {
        if self.spawn_x.is_none() {
            self.spawn_x = Some(x);
        }
        // 空中不记探索：否则跳跃途中的高度带也会被算成换平台。
        if !(on_ground || climbing) {
            return;
        }
        let key = visit_key(x, y);
        // 新网格：小额奖励，不受清台限制（鼓励移动）。
        if self.visited_cells.insert(key) && self.explore_score < NEW_CELL_CAP {
            let applied = PTS_NEW_CELL.min(NEW_CELL_CAP - self.explore_score);
            self.explore_score += applied;
            self.score += applied;
        }
        let y_band = key.1;
        if self.spawn_y_band.is_none() {
            self.spawn_y_band = Some(y_band);
        }
        let is_new_band = self.visited_y_bands.insert(y_band);

        // 振荡检测 A→B→A：检测是否返回了上上个 y-band。
        if let Some(current_band) = self.last_y_band {
            if y_band != current_band {
                // y-band 变化
                if let Some(prev) = self.prev_y_band {
                    if y_band == prev {
                        let elapsed =
                            self.last_episode_tick.saturating_sub(self.y_band_changed_at_tick);
                        if elapsed < Y_BAND_OSCILLATION_TICKS {
                            self.score -= PENALTY_Y_BAND_OSCILLATION;
                        }
                    }
                }
                // 推移历史：prev ← last, last ← current
                self.prev_y_band = Some(current_band);
                self.last_y_band = Some(y_band);
                self.y_band_changed_at_tick = self.last_episode_tick;
            }
        } else {
            self.last_y_band = Some(y_band);
        }

        // 平台有怪时：不给 y-band 和平台切换奖励，强制先清怪。
        let band_clear = self.band_is_cleared();

        if is_new_band && band_clear {
            self.explore_score += PTS_NEW_Y_BAND;
            self.score += PTS_NEW_Y_BAND;
        }
        let left_horizontally = self
            .spawn_x
            .map(|sx| (x - sx).abs() >= LEAVE_SPAWN_MIN_DX_PX)
            .unwrap_or(false);
        let Some(spawn) = self.spawn_y_band else {
            return;
        };
        let changed_band = y_band != spawn;
        let left_by_climb = climbing && changed_band;
        let left_by_platform = on_ground && changed_band && left_horizontally;
        if (left_by_climb || left_by_platform) && !self.left_spawn_band {
            self.left_spawn_band = true;
            self.leave_spawn_bonus += PTS_LEAVE_SPAWN_BAND;
            self.score += PTS_LEAVE_SPAWN_BAND;
        }
        if self.left_spawn_band
            && is_new_band
            && changed_band
            && (climbing || left_horizontally)
            && band_clear
            && self.platform_change_score < PLATFORM_CHANGE_CAP
        {
            let applied =
                PTS_EXTRA_PLATFORM.min(PLATFORM_CHANGE_CAP - self.platform_change_score);
            self.platform_change_score += applied;
            self.score += applied;
        }
    }

    fn tick_band_clear(&mut self) {
        if obs_has_nearby_platform_enemy(&self.last_obs) {
            if self.no_same_level_enemy_ticks > 0 {
                // 敌人重新出现（到了新平台），重置清台奖励标记
                self.platform_clear_bonus_given = false;
            }
            self.no_same_level_enemy_ticks = 0;
        } else {
            self.no_same_level_enemy_ticks = self.no_same_level_enemy_ticks.saturating_add(1);
            // 首次达到清台阈值：给一次性奖励
            if self.no_same_level_enemy_ticks == CLEAR_BAND_NO_ENEMY_TICKS
                && !self.platform_clear_bonus_given
            {
                self.platform_clear_bonus_given = true;
                self.score += PTS_PLATFORM_CLEAR_ONCE;
            }
        }
    }

    fn recently_productive(&self, episode_tick: u64) -> bool {
        let window = u64::from(STAGNATION_TICKS);
        if self.last_pickup_tick > 0 && episode_tick.saturating_sub(self.last_pickup_tick) < window {
            return true;
        }
        self.last_mob_kill_tick > 0
            && episode_tick.saturating_sub(self.last_mob_kill_tick) < window
    }

    pub fn finalize_episode(&mut self) {
        if self.episode_finalized {
            return;
        }
        self.episode_finalized = true;
        let ticks = self.last_episode_tick.max(1);
        let bands = self.visited_y_bands.len() as u32;

        // 存活基线无条件给：摆烂个体分数趋近小正数，而不是靠重罚拉到 -500。
        let secs = ticks as f32 / 60.0;
        let bonus = (secs * PTS_SURVIVAL_PER_SEC).min(SURVIVAL_BONUS_CAP);
        self.survival_bonus += bonus;
        self.score += bonus;

        if bands >= 2 && self.left_spawn_band {
            let extra = (bands - 1).min(3) as f32 * PTS_MULTI_BAND;
            let applied = extra.min(MULTI_BAND_BONUS_CAP);
            self.platform_change_score += applied;
            self.score += applied;
        }
    }

    fn note_stagnation_activity(&mut self) {
        if self.stagnation_initialized {
            self.reset_stagnation_anchor(self.last_x, self.last_y);
        }
    }

    fn reset_stagnation_anchor(&mut self, x: f32, y: f32) {
        self.stagnation_anchor_x = x;
        self.stagnation_anchor_y = y;
        self.stagnation_ticks = 0;
    }

    pub fn try_score_pickup(
        &mut self,
        kind: DropKind,
        x: f32,
        y: f32,
        meso_amount: u32,
        episode_tick: u64,
    ) -> f32 {
        let gained = match kind {
            DropKind::Meso => {
                if !point_in_any_box(x, y, &self.last_visible.meso) {
                    return 0.0;
                }
                self.meso_events += 1;
                self.meso_units += meso_amount;
                meso_amount as f32 * PTS_MESO_UNIT
            }
            DropKind::RedPotion => {
                if !point_in_any_box(x, y, &self.last_visible.potions) {
                    return 0.0;
                }
                self.potion_events += 1;
                PTS_POTION_PICKUP
            }
        };
        if gained > 0.0 {
            self.last_pickup_tick = episode_tick;
            self.note_stagnation_activity();
            if self.last_mob_kill_tick > 0
                && episode_tick.saturating_sub(self.last_mob_kill_tick) <= KILL_LOOT_CHAIN_TICKS
            {
                self.score += PTS_KILL_LOOT_CHAIN;
            }
        }
        self.pickup_score += gained;
        self.score += gained;
        gained
    }

    pub fn reset(&mut self) {
        let shaping = self.shaping;
        *self = Self::with_shaping(shaping);
    }
}

fn visit_key(x: f32, y: f32) -> (i32, i32) {
    (
        (x / X_CELL_PX).floor() as i32,
        (y / ALTITUDE_BAND_PX).floor() as i32,
    )
}

fn point_in_any_box(x: f32, y: f32, boxes: &[[f32; 4]]) -> bool {
    boxes
        .iter()
        .any(|b| x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::observation::{OBS_ENEMY_START, OBS_FLOOR_START};
    use crate::game::InputFrame;

    fn floor_slot(values: &mut [f32]) {
        values[OBS_FLOOR_START] = 0.0;
        values[OBS_FLOOR_START + 1] = 0.02;
        values[OBS_FLOOR_START + 2] = 0.3;
        values[OBS_FLOOR_START + 3] = 0.05;
    }

    fn same_level_enemy_slot(values: &mut [f32], dx: f32) {
        values[OBS_ENEMY_START] = dx;
        values[OBS_ENEMY_START + 1] = 0.0;
        values[OBS_ENEMY_START + 2] = 0.05;
        values[OBS_ENEMY_START + 3] = 0.05;
    }

    fn swing(f: &mut TrainingFitness, tick: u64) {
        // 动作宏：按住 3 tick 出一刀，再松开。
        for _ in 0..3 {
            f.try_score_input(
                &InputFrame {
                    attack: true,
                    ..Default::default()
                },
                tick,
            );
        }
        f.try_score_input(&InputFrame::default(), tick);
    }

    #[test]
    fn empty_swing_penalized_once_per_swing() {
        let mut f = TrainingFitness::default();
        let mut obs = [0.0_f32; OBS_DIM];
        floor_slot(&mut obs);
        f.set_last_observation(&obs);
        swing(&mut f, 1);
        assert_eq!(f.empty_swings, 1);
        assert_eq!(f.empty_attack_penalty, PENALTY_EMPTY_SWING);
        swing(&mut f, 30);
        assert_eq!(f.empty_swings, 2, "按住多 tick 不应重复计一刀");
    }

    #[test]
    fn swing_with_enemy_in_range_is_free() {
        let mut f = TrainingFitness::default();
        let mut obs = [0.0_f32; OBS_DIM];
        floor_slot(&mut obs);
        same_level_enemy_slot(&mut obs, 0.03);
        f.set_last_observation(&obs);
        f.try_score_input(
            &InputFrame {
                attack: true,
                right: true,
                ..Default::default()
            },
            1,
        );
        assert_eq!(f.empty_attack_penalty, 0.0);
    }

    #[test]
    fn idle_genome_finishes_near_zero() {
        // 什么都不做：存活基线与挂机罚大致抵消，落在 0 附近，不再是 -500。
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        let mut obs = [0.0_f32; OBS_DIM];
        floor_slot(&mut obs);
        f.set_last_observation(&obs);
        for t in 1..=200u64 {
            f.try_score_input(&InputFrame::default(), t);
            f.tick_stagnation(100.0, 1000.0, t, true, false);
        }
        f.finalize_episode();
        assert!(f.score.abs() < 30.0, "score={} 应落在 0 附近", f.score);
    }

    #[test]
    fn farming_beats_idling() {
        let mut idle = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        let mut farm = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        let mut obs = [0.0_f32; OBS_DIM];
        floor_slot(&mut obs);
        idle.set_last_observation(&obs);
        farm.set_last_observation(&obs);
        for t in 1..=200u64 {
            idle.tick_stagnation(100.0, 1000.0, t, true, false);
            farm.tick_stagnation(100.0 + t as f32, 1000.0, t, true, false);
        }
        farm.last_visible.meso.push([0.0, 0.0, 100.0, 100.0]);
        farm.try_score_pickup(DropKind::Meso, 50.0, 50.0, 3, 100);
        idle.finalize_episode();
        farm.finalize_episode();
        assert!(farm.score > idle.score);
    }

    #[test]
    fn wall_push_forfeits_after_threshold() {
        let mut f = TrainingFitness::default();
        let mut obs = [0.0_f32; OBS_DIM];
        floor_slot(&mut obs);
        obs[OBS_PROPRIO_START + 2] = 1.0;
        f.set_last_observation(&obs);
        for t in 1..=WALL_PUSH_FORFEIT_TICKS {
            f.try_score_input(
                &InputFrame {
                    left: true,
                    ..Default::default()
                },
                t as u64,
            );
        }
        assert!(f.idle_forfeit);
        assert!(!f.allows_peak_update());
    }

    #[test]
    fn meso_pickup_scores_more_than_potion_per_unit() {
        let mut f = TrainingFitness::default();
        f.last_visible.meso.push([0.0, 0.0, 100.0, 100.0]);
        let gained = f.try_score_pickup(DropKind::Meso, 50.0, 50.0, 5, 1);
        assert_eq!(gained, 75.0);
        assert!(gained > PTS_POTION_PICKUP);
    }

    #[test]
    fn exploration_rewards_new_cell_and_y_band() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        // 无怪场景：拨快「清台」时钟，让 band_is_cleared() 返回 true。
        f.mob_kill_events = 1;
        f.last_obs = [0.0; OBS_DIM];
        f.tick_stagnation(100.0, 1225.0, 1, true, false);
        assert!(f.explore_score >= PTS_NEW_CELL + PTS_NEW_Y_BAND);
    }

    #[test]
    fn airborne_jump_does_not_grant_leave_spawn() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        f.tick_stagnation(100.0, 1100.0, 1, true, false);
        f.tick_stagnation(100.0, 900.0, 2, false, false);
        assert!(!f.left_spawn_band);
        f.tick_stagnation(100.0 + LEAVE_SPAWN_MIN_DX_PX + 1.0, 1100.0, 3, true, false);
        assert!(!f.left_spawn_band);
        f.tick_stagnation(100.0 + LEAVE_SPAWN_MIN_DX_PX + 1.0, 900.0, 4, true, false);
        assert!(f.left_spawn_band);
    }

    #[test]
    fn stagnation_forfeits_only_after_grace() {
        let mut f = TrainingFitness::with_shaping(FitnessShapingConfig::disabled());
        for t in 1..=STAGNATION_TICKS as u64 + 2 {
            f.tick_stagnation(100.0, 1000.0, t, true, false);
        }
        assert!(!f.idle_forfeit, "宽限期内不认输");
        let mut forfeited = false;
        for t in (STAGNATION_TICKS as u64 + 3)..=(IDLE_FORFEIT_GRACE_TICKS as u64 * 2) {
            if f.tick_stagnation(100.0, 1000.0, t, true, false) {
                forfeited = true;
                break;
            }
        }
        assert!(forfeited, "长时间原地不动必须认输");
    }
}
