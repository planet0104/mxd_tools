use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::game::camera::WorldCamera;
use crate::game::config::GameSimConfig;
use crate::game::input::InputFrame;
use crate::game::map::{GameMap, WalkAhead};
use crate::game::movement_gate::{MovementGate, MovementGateCtx};
use crate::game::npc::{self, NpcPlayerState};
use crate::game::types::*;
use crate::game::vision::SimVisionSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameModal {
    None,
    Inventory,
    GameOver,
}

#[derive(Debug, Clone)]
pub struct PlayerState {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub facing: f32,
    pub hp: i32,
    pub max_hp: i32,
    pub on_ground: bool,
    /// 离地后仍可起跳的剩余时间
    pub coyote_t: f32,
    /// 跳跃输入缓冲剩余时间
    pub jump_buf_t: f32,
    /// 当前站立 foothold 的 layer/group（来自 WZ）；空中保留上次以便短时间侧墙判断可不用
    pub fh_layer: i32,
    pub fh_group: i32,
    pub fh_id: u32,
    pub climbing: bool,
    pub climb_kind: String,
    pub anim: PlayerAnim,
    pub anim_t: f32,
    pub attack_t: f32,
    pub attack_cd: f32,
    pub hurt_t: f32,
    pub invuln_t: f32,
}

#[derive(Debug, Clone)]
pub struct MobState {
    pub mob_id: u32,
    pub x: f32,
    pub y: f32,
    pub hp: i32,
    pub max_hp: i32,
    pub vx: f32,
    pub walk_x1: f32,
    pub walk_x2: f32,
    pub alive: bool,
    pub hit_t: f32,
    pub die_t: f32,
    pub anim: MobAnim,
    pub anim_t: f32,
    pub touch_damage: i32,
}

/// 接战提示：相对位移与怪行进方向（用于正面接战）。
#[derive(Debug, Clone, Copy)]
pub struct EngageHint {
    /// mob.x - player.x
    pub dx: f32,
    /// mob.y - player.y（正值=怪在下方）
    pub dy: f32,
    /// 怪水平行进方向（±1）
    pub mob_dir: f32,
}

impl EngageHint {
    /// 玩家是否在怪背后（怪正背离玩家走 / 同向追尾）。
    pub fn player_behind(&self) -> bool {
        let dir = self.mob_dir.signum();
        if dir == 0.0 || self.dx.abs() < 4.0 {
            return false;
        }
        (dir > 0.0 && self.dx > 0.0) || (dir < 0.0 && self.dx < 0.0)
    }

    /// 怪是否正朝玩家走来（迎面）。
    pub fn mob_approaching(&self) -> bool {
        let dir = self.mob_dir.signum();
        if dir == 0.0 || self.dx.abs() < 4.0 {
            return false;
        }
        (dir < 0.0 && self.dx > 0.0) || (dir > 0.0 && self.dx < 0.0)
    }

    /// 朝怪所在水平方向。
    pub fn toward_mob(&self) -> f32 {
        if self.dx.abs() < 1.0 {
            0.0
        } else {
            self.dx.signum()
        }
    }

    /// 怪是否在玩家脚下更低的一层（高台避险视角）。
    pub fn mob_below(&self) -> bool {
        self.dy > 28.0
    }
}

#[derive(Debug, Clone)]
pub struct DropState {
    pub kind: DropKind,
    pub x: f32,
    pub y: f32,
    pub alive: bool,
    pub bob_t: f32,
}

#[derive(Debug, Clone)]
pub struct GameState {
    pub player: PlayerState,
    pub mobs: Vec<MobState>,
    pub drops: Vec<DropState>,
    pub meso: u32,
    pub potions: u32,
    pub kills: u32,
    /// 本局被怪 touch 掉血次数。
    pub touch_hits: u32,
    pub modal: GameModal,
    pub cam_x: f32,
    pub cam_y: f32,
    pub tick: u64,
    pub portal_hint: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GroundTruth {
    pub player_x: f32,
    pub player_y: f32,
    pub hp: i32,
    pub max_hp: i32,
    pub potions: u32,
    pub meso: u32,
    pub mob_count: usize,
}

pub struct GameSim {
    pub map: GameMap,
    pub state: GameState,
    pub config: GameSimConfig,
    /// 本局 episode 种子（锚点抖动等域随机化）。
    pub episode_seed: u64,
    /// 训练用装饰玩家（YOLO「玩家」干扰，OCR 排除自身）
    pub npc_players: Vec<NpcPlayerState>,
    pub movement_gate: MovementGate,
    spawn_x: f32,
    spawn_y: f32,
    rng: StdRng,
}

impl GameSim {
    pub fn new(map: GameMap, seed: u64) -> Self {
        Self::new_with_config(map, seed, GameSimConfig::default())
    }

    /// 规则 bot 自动玩：装饰 NPC、波次刷怪、零初始药水。
    pub fn new_bot_play(map: GameMap, seed: u64) -> Self {
        Self::new_with_config(map, seed, GameSimConfig::bot_play())
    }

    /// 预览模式：自动玩环境 + 受击不死，便于持续观察。
    pub fn new_preview(map: GameMap, seed: u64) -> Self {
        Self::new_with_config(map, seed, GameSimConfig::preview())
    }

    pub fn new_with_config(map: GameMap, seed: u64, config: GameSimConfig) -> Self {
        let (spawn_x, spawn_y) = map.default_spawn();
        let rng = StdRng::seed_from_u64(seed);
        let start_potions = if config.bot_play { 0 } else { 5 };
        let mut sim = Self {
            map,
            state: GameState {
                player: PlayerState {
                    x: spawn_x,
                    y: spawn_y,
                    vx: 0.0,
                    vy: 0.0,
                    facing: 1.0,
                    hp: PLAYER_MAX_HP,
                    max_hp: PLAYER_MAX_HP,
                    on_ground: false,
                    coyote_t: 0.0,
                    jump_buf_t: 0.0,
                    fh_layer: -1,
                    fh_group: -1,
                    fh_id: 0,
                    climbing: false,
                    climb_kind: String::new(),
                    anim: PlayerAnim::Stand,
                    anim_t: 0.0,
                    attack_t: 0.0,
                    attack_cd: 0.0,
                    hurt_t: 0.0,
                    invuln_t: 0.0,
                },
                mobs: Vec::new(),
                drops: Vec::new(),
                meso: 0,
                potions: start_potions,
                kills: 0,
                touch_hits: 0,
                modal: GameModal::None,
                cam_x: 0.0,
                cam_y: 0.0,
                tick: 0,
                portal_hint: None,
            },
            config,
            episode_seed: seed,
            npc_players: Vec::new(),
            movement_gate: MovementGate::default(),
            spawn_x,
            spawn_y,
            rng,
        };
        sim.spawn_mobs();
        if sim.config.bot_play {
            sim.npc_players = npc::spawn_training_npcs(&sim.map, &mut sim.rng);
        }
        sim.snap_player_to_ground();
        sim.update_camera();
        sim
    }

    fn snap_player_to_ground(&mut self) {
        let x = self.state.player.x;
        if let Some(st) = self.map.stand_at(x, self.state.player.y + 40.0, 120.0) {
            let p = &mut self.state.player;
            p.y = st.y;
            p.on_ground = true;
            p.fh_id = st.id;
            p.fh_layer = st.layer;
            p.fh_group = st.group;
            p.coyote_t = COYOTE_TIME;
        }
    }

    fn apply_stand(p: &mut PlayerState, st: Option<crate::game::map::StandInfo>) {
        if let Some(st) = st {
            p.y = st.y;
            p.vy = 0.0;
            p.on_ground = true;
            p.fh_id = st.id;
            p.fh_layer = st.layer;
            p.fh_group = st.group;
            p.coyote_t = COYOTE_TIME;
        } else {
            p.on_ground = false;
        }
    }

    fn check_void_fall(&mut self) {
        let p = &self.state.player;
        if p.climbing {
            return;
        }
        // 只有明显掉出最低平台以下才救援；本预览/训练图不允许虚空死。
        let too_low = p.y > self.map.death_y();
        if too_low {
            self.rescue_from_void();
        }
    }

    /// 掉出可玩区：吸到最近脚点，禁止虚空重生（地图不应有「摔死」路径）。
    fn rescue_from_void(&mut self) {
        let (x, y) = {
            let p = &self.state.player;
            (p.x, p.y)
        };
        if let Some((sx, st)) = self.map.nearest_stand(x, y) {
            {
                let p = &mut self.state.player;
                p.x = sx;
                p.vx = 0.0;
                p.vy = 0.0;
                p.climbing = false;
                p.climb_kind.clear();
            }
            Self::apply_stand(&mut self.state.player, Some(st));
            return;
        }
        // 极端：地图无脚点才回出生点。
        self.respawn_player();
    }

    fn respawn_player(&mut self) {
        let (sx, sy) = self.map.default_spawn();
        self.spawn_x = sx;
        self.spawn_y = sy;
        {
            let p = &mut self.state.player;
            p.x = sx;
            p.y = sy;
            p.vx = 0.0;
            p.vy = 0.0;
            p.climbing = false;
            p.climb_kind.clear();
            p.hurt_t = 0.0;
            p.invuln_t = 0.0;
            p.on_ground = false;
        }
        self.snap_player_to_ground();
    }

    fn spawn_mobs(&mut self) {
        for sp in &self.map.spawns.clone() {
            let stats = mob_stats(sp.mob_id);
            // 以刷怪点所在连续平台为准，避免 JSON 里误写成全图宽度
            let (walk_x1, walk_x2) = self.map.walk_range_at(sp.x, sp.y);
            self.state.mobs.push(MobState {
                mob_id: sp.mob_id,
                x: sp.x.clamp(walk_x1, walk_x2),
                y: sp.y,
                hp: stats.hp,
                max_hp: stats.hp,
                vx: stats.walk_speed() * if self.rng.gen_bool(0.5) { 1.0 } else { -1.0 },
                walk_x1,
                walk_x2,
                alive: true,
                hit_t: 0.0,
                die_t: 0.0,
                anim: MobAnim::Move,
                anim_t: 0.0,
                touch_damage: stats.touch_damage,
            });
        }
    }

    pub fn ground_truth(&self) -> GroundTruth {
        GroundTruth {
            player_x: self.state.player.x,
            player_y: self.state.player.y,
            hp: self.state.player.hp,
            max_hp: self.state.player.max_hp,
            potions: self.state.potions,
            meso: self.state.meso,
            mob_count: self.state.mobs.iter().filter(|m| m.alive).count(),
        }
    }

    /// 合成 YOLO 观测锚点：当前帧 sim 投影用快照。
    pub fn vision_snapshot(&self) -> SimVisionSnapshot {
        SimVisionSnapshot {
            player_x: self.state.player.x,
            player_y: self.state.player.y,
            cam_x: self.state.cam_x,
            cam_y: self.state.cam_y,
            episode_seed: self.episode_seed,
        }
    }

    /// 玩家是否已死亡（HP 归零）。
    pub fn is_episode_over(&self) -> bool {
        self.state.modal == GameModal::GameOver
    }

    pub fn tick(&mut self, input: &InputFrame) {
        self.tick_with_action(input);
    }

    /// 规则 bot：返回门控后的有效输入（用于日志对比）。
    pub fn effective_bot_input(&self, input: &InputFrame) -> InputFrame {
        if !self.config.bot_play {
            return *input;
        }
        let gate_ctx = self.movement_gate_ctx();
        self.movement_gate.filter_input(input, gate_ctx)
    }

    /// 游戏世界自身的物理碰撞规则（是否能走/能跳/能砍），与 Bot 感知无关，
    /// 必须用真实模拟状态：否则视觉延迟一帧就会让本该能砍/能爬的输入被误滤掉。
    fn movement_gate_ctx(&self) -> MovementGateCtx {
        let (physics_right_ok, physics_left_ok) = self.physics_walk_ok_pair();
        let (physics_drop_right, physics_drop_left) = self.physics_drop_ok_pair();
        MovementGateCtx {
            facing: self.state.player.facing,
            on_ground: self.state.player.on_ground,
            climbing: self.state.player.climbing,
            can_use_potion: self.state.potions > 0
                && self.state.player.hp < self.state.player.max_hp,
            physics_right_ok,
            physics_left_ok,
            physics_drop_right,
            physics_drop_left,
            sim_mob_in_melee: self.mob_in_strike_band(),
            allow_combat_leap: self
                .nearest_engage_hint()
                .map(|h| h.player_behind())
                .unwrap_or(false),
            adjacent_climb: self.nearest_adjacent_climb().is_some(),
            allow_step_up: self.nearest_step_up_dx().is_some(),
        }
    }

    /// 物理前方是否为可落下缘（下方有更低平台）。
    pub fn physics_drop_ok_pair(&self) -> (Option<bool>, Option<bool>) {
        let p = &self.state.player;
        if !p.on_ground || p.climbing {
            return (None, None);
        }
        let fh = if p.fh_layer >= 0 {
            Some((p.fh_layer, p.fh_group))
        } else {
            None
        };
        const PROBE: f32 = 24.0;
        let drop_dir = |dir: f32| -> bool {
            let to_x = p.x + dir * PROBE;
            matches!(
                self.map.walk_ahead(p.x, p.y, to_x, fh),
                super::map::WalkAhead::Fall
            )
        };
        (Some(drop_dir(1.0)), Some(drop_dir(-1.0)))
    }

    /// 物理同层前方是否可走（与 YOLO 无关，训练/预览一致）。
    pub fn physics_walk_ok_pair(&self) -> (Option<bool>, Option<bool>) {
        let p = &self.state.player;
        if !p.on_ground || p.climbing {
            return (None, None);
        }
        let fh = if p.fh_layer >= 0 {
            Some((p.fh_layer, p.fh_group))
        } else {
            None
        };
        const PROBE: f32 = 24.0;
        let ok_dir = |dir: f32| -> bool {
            let to_x = p.x + dir * PROBE;
            matches!(
                self.map.walk_ahead(p.x, p.y, to_x, fh),
                WalkAhead::SameLevel(_)
            )
        };
        (Some(ok_dir(1.0)), Some(ok_dir(-1.0)))
    }

    /// 紧邻当前层的绳/梯；远处上层绳梯返回 None。
    pub fn nearest_adjacent_climb(&self) -> Option<crate::game::map::ClimbHint> {
        let p = &self.state.player;
        self.map.nearest_adjacent_climb(p.x, p.y)
    }

    /// 可跳上一层台阶的相对 dx。
    pub fn nearest_step_up_dx(&self) -> Option<f32> {
        let p = &self.state.player;
        if !p.on_ground || p.climbing {
            return None;
        }
        self.map.nearest_step_up_dx(p.x, p.y)
    }

    /// 传入本帧 bot 意图，经门控后驱动模拟。
    pub fn tick_with_action(&mut self, input: &InputFrame) {
        if input.restart && self.state.modal == GameModal::GameOver {
            *self = GameSim::new_with_config(self.map.clone(), self.rng.gen(), self.config);
            return;
        }

        if self.state.modal == GameModal::GameOver {
            return;
        }

        if input.open_inventory {
            self.state.modal = match self.state.modal {
                GameModal::Inventory => GameModal::None,
                _ => GameModal::Inventory,
            };
        }

        if self.state.modal == GameModal::Inventory {
            if input.use_potion || input.inventory_click.is_some() {
                self.use_potion();
            }
            return;
        }

        self.state.tick += 1;
        let dt = LOGIC_DT;
        let gate_ctx = self.movement_gate_ctx();
        let mut effective = *input;
        if self.config.bot_play {
            effective = self.movement_gate.filter_input(input, gate_ctx);
        }
        if effective.use_potion {
            self.use_potion();
        }
        self.tick_player(&effective, dt);
        if self.config.bot_play {
            npc::tick_npc_players(
                &mut self.npc_players,
                &mut self.state.mobs,
                dt,
                &mut self.rng,
            );
        }
        self.tick_mobs(dt);
        self.tick_drops(input, dt);
        self.update_camera();
    }

    fn use_potion(&mut self) {
        if self.state.potions == 0 || self.state.player.hp >= self.state.player.max_hp {
            return;
        }
        self.state.potions -= 1;
        self.state.player.hp = (self.state.player.hp + POTION_HEAL).min(self.state.player.max_hp);
    }

    fn tick_player(&mut self, input: &InputFrame, dt: f32) {
        if self.state.player.hp <= 0 {
            if self.config.preview {
                self.state.player.hp = 1;
                self.state.player.invuln_t = 1.0;
            } else {
                self.state.modal = GameModal::GameOver;
                return;
            }
        }

        {
            let p = &mut self.state.player;
            p.attack_cd = (p.attack_cd - dt).max(0.0);
            p.invuln_t = (p.invuln_t - dt).max(0.0);
            p.hurt_t = (p.hurt_t - dt).max(0.0);
            p.attack_t = (p.attack_t - dt).max(0.0);
            p.anim_t += dt;
        }

        let (px, py) = (self.state.player.x, self.state.player.y);
        self.state.portal_hint = self
            .map
            .portal_near(px, py)
            .map(|pt| format!("传送门 {} → 地图 {}", pt.name, pt.to_map));

        // 挥砍期间可锁位移；受击（被怪碰到）不锁走，否则出生点贴怪会一直钉死。
        let attack_lock = self.state.player.attack_t > 0.0;
        let do_attack = !attack_lock
            && !self.state.player.climbing
            && input.attack
            && self.state.player.attack_cd <= 0.0;

        if do_attack {
            let face_dx = self.nearest_engage_hint().map(|e| e.dx);
            let p = &mut self.state.player;
            // 挥砍瞬间按最近同层怪自动转向，避免靠方向键转身变成追怪贴脸。
            if let Some(dx) = face_dx {
                if dx.abs() > 2.0 {
                    p.facing = dx.signum();
                }
            }
            p.attack_t = ATTACK_DURATION;
            p.attack_cd = ATTACK_COOLDOWN;
            p.anim = PlayerAnim::Attack;
            p.anim_t = 0.0;
        }

        if self.state.player.climbing {
            self.tick_player_climb(input, dt);
        } else if !attack_lock {
            let want_climb = input.up || input.down;
            let grab = if want_climb {
                self.map
                    .rope_at(self.state.player.x, self.state.player.y)
                    .and_then(|r| {
                        let top = r.y1.min(r.y2);
                        // 已站在绳顶平台：仅按↑不再抓绳（否则到顶落地后立刻又挂回去）。
                        if self.state.player.on_ground
                            && input.up
                            && !input.down
                            && self.state.player.y <= top + 12.0
                        {
                            return None;
                        }
                        Some((r.x, r.kind.clone()))
                    })
            } else {
                None
            };
            if let Some((rx, kind)) = grab {
                {
                    let p = &mut self.state.player;
                    p.climbing = true;
                    p.climb_kind = kind;
                    p.on_ground = false;
                    p.vx = 0.0;
                    p.vy = 0.0;
                    p.x = rx;
                }
                self.tick_player_climb(input, dt);
            } else {
                self.tick_player_move(input, dt);
            }
        } else {
            // 挥砍硬直：保持脚点，不清 on_ground，结束挥砍后才能立刻走路。
            let (prev_x, prev_y, on_ground, vy0) = {
                let p = &self.state.player;
                (p.x, p.y, p.on_ground, p.vy)
            };
            if on_ground {
                let stand = self.map.strict_stand_at(prev_x, prev_y);
                let p = &mut self.state.player;
                p.vx = 0.0;
                if let Some(st) = stand {
                    Self::apply_stand(p, Some(st));
                } else {
                    p.y = prev_y;
                    p.vy = 0.0;
                    p.on_ground = true;
                }
            } else {
                let mut vy = vy0 + GRAVITY * dt;
                let next_y = prev_y + vy * dt;
                let landed = self.map.land_at(prev_x, prev_y, next_y);
                let p = &mut self.state.player;
                p.vx = 0.0;
                if let Some(st) = landed {
                    Self::apply_stand(p, Some(st));
                } else {
                    p.y = next_y;
                    p.vy = vy;
                    p.on_ground = false;
                }
            }
        }

        {
            let (x, y, on_ground) = {
                let p = &self.state.player;
                (p.x, p.y, p.on_ground)
            };
            const BODY_INSET: f32 = 4.0;
            const EXIT_PROBE: f32 = 6.0;
            if on_ground {
                if let Some((lo, hi)) = self.map.platform_span_at(x, y) {
                    let can_exit = |probe_x: f32| -> bool {
                        self.map.strict_stand_at(probe_x, y).is_none()
                            && self
                                .map
                                .ground_below_at(probe_x, y + 2.0, FALL_PROBE)
                                .is_some()
                    };
                    let min_x = if can_exit(lo - EXIT_PROBE) {
                        lo - EXIT_PROBE - BODY_INSET
                    } else {
                        lo + BODY_INSET
                    };
                    let max_x = if can_exit(hi + EXIT_PROBE) {
                        hi + EXIT_PROBE + BODY_INSET
                    } else {
                        hi - BODY_INSET
                    };
                    let p = &mut self.state.player;
                    p.x = p.x.clamp(min_x, max_x);
                }
            } else {
                let (px_lo, px_hi) = self.map.playable_x_bounds();
                let p = &mut self.state.player;
                p.x = p.x.clamp(px_lo, px_hi);
            }
            // 只限制上边界；下边界由虚空重生处理，避免卡在泥土高度
            self.state.player.y = self.state.player.y.max(16.0);
        }

        self.check_void_fall();

        {
            let p = &mut self.state.player;
            if p.climbing {
                p.anim = PlayerAnim::Climb;
            } else if p.hurt_t <= 0.0 && p.attack_t <= 0.0 {
                p.anim = if !p.on_ground {
                    PlayerAnim::Jump
                } else if p.vx.abs() > 10.0 {
                    PlayerAnim::Walk
                } else {
                    PlayerAnim::Stand
                };
            } else if p.hurt_t > 0.0 {
                p.anim = PlayerAnim::Hurt;
            }
        }

        self.check_mob_touch();

        if do_attack {
            self.try_attack_mobs();
        }
    }

    fn tick_player_move(&mut self, input: &InputFrame, dt: f32) {
        let h = input.horizontal();
        let (
            prev_x,
            prev_y,
            mut vy,
            mut on_ground,
            mut facing,
            mut coyote_t,
            mut jump_buf_t,
            fh_layer,
            fh_group,
        ) = {
            let p = &self.state.player;
            (
                p.x,
                p.y,
                p.vy,
                p.on_ground,
                p.facing,
                p.coyote_t,
                p.jump_buf_t,
                p.fh_layer,
                p.fh_group,
            )
        };
        let mut vx = 0.0_f32;
        // 站在某组平台上才用该组侧墙；空中不挡其它层级侧面
        let fh = if on_ground && fh_layer >= 0 {
            Some((fh_layer, fh_group))
        } else {
            None
        };

        if h.abs() > 0.01 {
            vx = h * PLAYER_SPEED;
            facing = h.signum();
        } else {
            vx = 0.0;
        }

        if input.jump {
            jump_buf_t = JUMP_BUFFER;
        } else {
            jump_buf_t = (jump_buf_t - dt).max(0.0);
        }

        if on_ground {
            coyote_t = COYOTE_TIME;
        } else {
            coyote_t = (coyote_t - dt).max(0.0);
        }

        let can_jump = on_ground || coyote_t > 0.0;
        if jump_buf_t > 0.0 && can_jump {
            vy = JUMP_VY;
            on_ground = false;
            coyote_t = 0.0;
            jump_buf_t = 0.0;
        }

        let desire_x = prev_x + vx * dt;
        let y_hi = prev_y - WALL_HIT_H;
        let wall_fh = if on_ground { fh } else { None };
        let mut new_x = self
            .map
            .resolve_wall_x(prev_x, desire_x, y_hi, prev_y - 2.0, wall_fh);

        // 腾空：禁止水平飞入「脚下到虚空线都无脚点」的竖直列（跳二台擦边掉缝的根因）。
        if !on_ground && (new_x - prev_x).abs() > 0.01 {
            if !self.map.has_support_column(new_x, prev_y) {
                new_x = prev_x;
                vx = 0.0;
            }
        }

        let mut align_y: Option<f32> = None;
        if on_ground && vy >= 0.0 && (new_x - prev_x).abs() > 0.01 {
            match self.map.walk_ahead(prev_x, prev_y, new_x, fh) {
                WalkAhead::SameLevel(gy) => {
                    align_y = Some(gy);
                }
                WalkAhead::Fall => {
                    on_ground = false;
                }
                WalkAhead::Blocked => {
                    new_x = prev_x;
                    vx = 0.0;
                }
            }
        }

        let feet_y = align_y.unwrap_or(prev_y);

        if !on_ground || vy > 0.0 {
            vy += GRAVITY * dt;
        }

        // 已在地面：用同层脚点确认（strict），不能用 stand_at——它会跳过当前脚底平台，
        // 导致每帧 on_ground 抖动、Stand/Jump 动画狂闪。
        let stand = if on_ground && vy >= 0.0 {
            self.map.strict_stand_at(new_x, feet_y)
        } else {
            None
        };
        let next_y = if on_ground && vy >= 0.0 {
            feet_y + vy * dt
        } else {
            prev_y + vy * dt
        };
        let mut landed = if !(on_ground && vy >= 0.0) && vy >= 0.0 {
            self.map.land_at(new_x, prev_y, next_y)
        } else {
            None
        };
        // 脚已略低于平台顶（水平先到位、竖直已擦过）：吸附，防掉虚空。
        if landed.is_none() && !on_ground && vy >= 0.0 {
            landed = self.map.ledge_snap_at(new_x, next_y);
        }

        let p = &mut self.state.player;
        p.x = new_x;
        p.vx = vx;
        p.facing = facing;
        p.jump_buf_t = jump_buf_t;

        if on_ground && vy >= 0.0 {
            if stand.is_some() {
                Self::apply_stand(p, stand);
            } else {
                p.y = next_y;
                p.vy = vy;
                p.on_ground = false;
                p.coyote_t = coyote_t;
            }
        } else if let Some(st) = landed {
            Self::apply_stand(p, Some(st));
        } else {
            p.y = next_y;
            p.vy = vy;
            p.on_ground = false;
            p.coyote_t = coyote_t;
        }
    }

    fn tick_player_climb(&mut self, input: &InputFrame, dt: f32) {
        let px = self.state.player.x;
        let py = self.state.player.y;
        let Some(rope) = self.map.rope_at(px, py) else {
            self.state.player.climbing = false;
            return;
        };
        let rx = rope.x;
        let ymin = rope.y1.min(rope.y2);
        let ymax = rope.y1.max(rope.y2);

        let p = &mut self.state.player;
        p.x = rx;
        p.vx = 0.0;
        p.vy = 0.0;
        p.on_ground = false;

        let dir = (input.up as i32 - input.down as i32) as f32;
        if dir.abs() > 0.01 {
            p.y += -dir * CLIMB_SPEED * dt;
        }
        p.y = p.y.clamp(ymin, ymax);

        let at_bottom = p.y >= ymax - 4.0;
        let at_top = p.y <= ymin + 4.0;
        let jump_off = input.jump;
        let leave_bottom = at_bottom && input.down && !input.up;

        if jump_off {
            p.climbing = false;
            p.vy = JUMP_VY * 0.85;
            p.on_ground = false;
        } else if leave_bottom {
            p.climbing = false;
            if let Some(st) = self.map.stand_at(p.x, p.y + 2.0, 48.0) {
                Self::apply_stand(p, Some(st));
            } else {
                p.on_ground = false;
            }
        } else if at_top {
            // 到顶自动站上顶端平台（不再要求松手↑；stand_at 只向下找会漏掉略高的顶板）。
            if let Some(st) = self.map.stand_at_climb_exit(p.x, ymin) {
                p.climbing = false;
                Self::apply_stand(p, Some(st));
            }
        }

        if p.climbing {
            p.anim = PlayerAnim::Climb;
        }
    }

    /// 物理近战框（与 `try_attack_mobs` 一致）。
    fn player_melee_aabb(p: &PlayerState) -> (f32, f32, f32, f32) {
        // 前伸约 90：在 touch(28) 之前就能连砍，避免贴脸才出手。
        let (x1, x2) = if p.facing > 0.0 {
            (p.x - 8.0, p.x + 90.0)
        } else {
            (p.x - 90.0, p.x + 8.0)
        };
        (x1, x2, p.y - 80.0, p.y + 24.0)
    }

    /// 是否有怪落在指定朝向挥砍框内。
    pub fn mob_in_melee_facing(&self, facing: f32) -> bool {
        let p = &self.state.player;
        let (x1, x2) = if facing > 0.0 {
            (p.x - 8.0, p.x + 90.0)
        } else {
            (p.x - 90.0, p.x + 8.0)
        };
        let y1 = p.y - 80.0;
        let y2 = p.y + 24.0;
        self.state
            .mobs
            .iter()
            .any(|m| m.alive && m.x >= x1 && m.x <= x2 && m.y >= y1 && m.y <= y2)
    }

    /// 是否有怪落在当前挥砍命中框内（YOLO 误判时的最终裁决）。
    pub fn mob_in_melee_range(&self) -> bool {
        self.mob_in_melee_facing(self.state.player.facing)
    }

    /// 距离判定：任一活怪进入可砍带（不依赖朝向 AABB），供门控放行。
    pub fn mob_in_strike_band(&self) -> bool {
        const STRIKE_DX: f32 = 90.0;
        const STRIKE_DY: f32 = 40.0;
        let p = &self.state.player;
        self.state
            .mobs
            .iter()
            .any(|m| m.alive && (m.x - p.x).abs() <= STRIKE_DX && (m.y - p.y).abs() <= STRIKE_DY)
    }

    /// 最近同层可接战怪：用于正面接战 / 避免背后追。
    pub fn nearest_engage_hint(&self) -> Option<EngageHint> {
        self.nearest_mob_hint(36.0)
    }

    /// 较宽垂直范围（含脚下低一层怪），供高台避险判断。
    pub fn nearest_engage_hint_wide(&self) -> Option<EngageHint> {
        self.nearest_mob_hint(140.0)
    }

    fn nearest_mob_hint(&self, max_dy: f32) -> Option<EngageHint> {
        let p = &self.state.player;
        let mut best: Option<(f32, EngageHint)> = None;
        for m in &self.state.mobs {
            if !m.alive {
                continue;
            }
            let dy = m.y - p.y;
            if dy.abs() > max_dy {
                continue;
            }
            let dx = m.x - p.x;
            let dist = dx.abs() + dy.abs() * 0.25;
            let mob_dir = if m.vx.abs() > 8.0 {
                m.vx.signum()
            } else if dx.abs() > 1.0 {
                -dx.signum()
            } else {
                p.facing.signum()
            };
            let hint = EngageHint { dx, dy, mob_dir };
            match best {
                None => best = Some((dist, hint)),
                Some((bd, _)) if dist < bd => best = Some((dist, hint)),
                _ => {}
            }
        }
        best.map(|(_, h)| h)
    }

    /// 指定高度带附近是否还有活怪（清层判定）。
    pub fn mobs_near_y(&self, band_y: f32, tol: f32) -> bool {
        self.state
            .mobs
            .iter()
            .any(|m| m.alive && (m.y - band_y).abs() <= tol)
    }

    /// 高度带 + 水平半径内是否有活怪（本段农怪未清判定）。
    pub fn mobs_near_xy(&self, band_y: f32, y_tol: f32, x: f32, x_tol: f32) -> bool {
        self.state
            .mobs
            .iter()
            .any(|m| m.alive && (m.y - band_y).abs() <= y_tol && (m.x - x).abs() <= x_tol)
    }

    /// 是否有怪在与玩家可普攻的同一高度带（整层平台，不限水平距离）。
    pub fn mob_on_attackable_footing(&self) -> bool {
        const ENGAGE_DY: f32 = 36.0;
        let p = &self.state.player;
        self.state
            .mobs
            .iter()
            .any(|m| m.alive && (m.y - p.y).abs() <= ENGAGE_DY)
    }

    fn try_attack_mobs(&mut self) {
        let p = &self.state.player;
        let (x1, x2, y1, y2) = Self::player_melee_aabb(p);
        let mut loot: Vec<(f32, f32)> = Vec::new();
        let mut kills = 0u32;
        for mob in &mut self.state.mobs {
            if !mob.alive {
                continue;
            }
            if mob.x >= x1 && mob.x <= x2 && mob.y >= y1 && mob.y <= y2 {
                mob.hp -= PLAYER_ATTACK_DAMAGE;
                mob.hit_t = 0.15;
                mob.anim = MobAnim::Hit;
                let kb = 28.0 * p.facing;
                mob.x += kb;
                if mob.hp <= 0 {
                    mob.alive = false;
                    mob.die_t = 0.5;
                    mob.anim = MobAnim::Die;
                    self.state.kills += 1;
                    kills += 1;
                    loot.push((mob.x, mob.y));
                }
            }
        }
        for (x, y) in loot {
            self.spawn_loot(x, y);
        }
    }

    fn spawn_loot(&mut self, x: f32, y: f32) {
        self.state.drops.push(DropState {
            kind: DropKind::Meso,
            x,
            y: y - 8.0,
            alive: true,
            bob_t: 0.0,
        });
        let potion_chance = if self.config.bot_play {
            TRAINING_POTION_DROP_CHANCE
        } else {
            NORMAL_POTION_DROP_CHANCE
        };
        if self.rng.gen_bool(potion_chance as f64) {
            self.state.drops.push(DropState {
                kind: DropKind::RedPotion,
                x: x + 12.0,
                y: y - 8.0,
                alive: true,
                bob_t: 0.3,
            });
        }
    }

    fn check_mob_touch(&mut self) {
        if self.state.player.invuln_t > 0.0 || self.state.player.hurt_t > 0.0 {
            return;
        }
        let px = self.state.player.x;
        let py = self.state.player.y;
        let facing = self.state.player.facing;
        let mut hit: Option<(i32, f32, f32)> = None;
        for mob in &self.state.mobs {
            if !mob.alive {
                continue;
            }
            let dx = px - mob.x;
            let dy = py - mob.y;
            if dx.abs() < 28.0 && dy.abs() < 36.0 {
                let knock_dir = if dx.abs() < 0.01 {
                    mob.vx.signum()
                } else {
                    dx.signum()
                };
                // mob_dx: 怪相对玩家（正=怪在右）
                hit = Some((mob.touch_damage, knock_dir, mob.x - px));
                break;
            }
        }
        let Some((damage, knock_dir, mob_dx)) = hit else {
            return;
        };

        let p = &mut self.state.player;
        let hp_before = p.hp;
        p.hp -= damage;
        if self.config.preview {
            p.hp = p.hp.max(1);
        }
        let hp_after = p.hp;
        p.hurt_t = HURT_DURATION;
        p.invuln_t = INVULN_DURATION;
        p.anim = PlayerAnim::Hurt;
        let feet_y = p.y;
        let old_x = p.x;

        self.state.touch_hits = self.state.touch_hits.saturating_add(1);
        let tick = self.state.tick;
        self.state.player.x = Self::safe_hurt_knockback_x(&self.map, old_x, feet_y, knock_dir);

        if self.config.preview || self.config.bot_play {
            eprintln!(
                "DMG tick={} touch -{} hp {}→{} hits={} player=({:.0},{:.0}) mob_dx={:.0} facing={}",
                tick,
                damage,
                hp_before,
                hp_after,
                self.state.touch_hits,
                px,
                py,
                mob_dx,
                if facing >= 0.0 { "R" } else { "L" }
            );
        }
    }

    /// 受击水平击退：仅当落点仍在脚下平台时才位移，避免角落被顶出平台坠亡。
    fn safe_hurt_knockback_x(map: &GameMap, x: f32, feet_y: f32, knock_dir: f32) -> f32 {
        const KNOCK_DIST: f32 = 40.0;
        const GROUND_PROBE: f32 = 64.0;
        if knock_dir.abs() < 0.01 {
            return x;
        }
        let proposed = x + knock_dir * KNOCK_DIST;
        if map
            .stand_at(proposed, feet_y + 40.0, GROUND_PROBE)
            .is_some()
        {
            if let Some((lo, hi)) = map.platform_span_at(x, feet_y) {
                const BODY_INSET: f32 = 4.0;
                return proposed.clamp(lo + BODY_INSET, hi - BODY_INSET);
            }
            let (px_lo, px_hi) = map.playable_x_bounds();
            return proposed.clamp(px_lo, px_hi);
        }
        x
    }

    fn tick_mobs(&mut self, dt: f32) {
        for mob in &mut self.state.mobs {
            if !mob.alive {
                mob.die_t -= dt;
                mob.anim_t += dt;
                continue;
            }
            mob.hit_t = (mob.hit_t - dt).max(0.0);
            mob.anim_t += dt;
            if mob.hit_t <= 0.0 {
                mob.anim = MobAnim::Move;
                mob.x += mob.vx * dt;
                if mob.x <= mob.walk_x1 {
                    mob.x = mob.walk_x1;
                    mob.vx = mob.vx.abs();
                }
                if mob.x >= mob.walk_x2 {
                    mob.x = mob.walk_x2;
                    mob.vx = -mob.vx.abs();
                }
            }
        }
        self.state.mobs.retain(|m| m.alive || m.die_t > 0.0);
        // 训练/预览：整图杀光后才波次重生。预览须等角色离开出生高度带，
        // 否则首台清完立刻刷回，永远 SeekVertical 不出去。
        if self.config.bot_play && self.state.mobs.is_empty() && !self.map.spawns.is_empty() {
            let (spawn_x, spawn_y) = self.map.default_spawn();
            let _ = spawn_x;
            let left_spawn_band = (self.state.player.y - spawn_y).abs() > 80.0;
            if !self.config.preview || left_spawn_band {
                self.spawn_mobs();
            }
        }
    }

    fn tick_drops(&mut self, input: &InputFrame, dt: f32) {
        let px = self.state.player.x;
        let py = self.state.player.y;
        // 怀旧版：走近不会自动捡，需按拾取键（默认 Z）
        if !input.pick_up {
            for drop in &mut self.state.drops {
                drop.bob_t += dt;
            }
            return;
        }
        for drop in &mut self.state.drops {
            if !drop.alive {
                continue;
            }
            drop.bob_t += dt;
            let dy = (drop.y - 8.0) - py;
            let dx = drop.x - px;
            if (dx * dx + dy * dy).sqrt() > 40.0 {
                continue;
            }
            let (_drop_sx, _drop_sy) = (drop.x - self.state.cam_x, drop.y - self.state.cam_y);
            match drop.kind {
                DropKind::Meso => {
                    let amount = self.rng.gen_range(1..=5);
                    self.state.meso += amount;
                }
                DropKind::RedPotion => {
                    self.state.potions += 1;
                }
            }
            drop.alive = false;
        }
        self.state.drops.retain(|d| d.alive);
    }

    fn update_camera(&mut self) {
        let p = &self.state.player;
        let mut cam = WorldCamera {
            cam_x: self.state.cam_x,
            cam_y: self.state.cam_y,
        };
        cam.follow(self.map.width, self.map.height, p.x, p.y);
        self.state.cam_x = cam.cam_x;
        self.state.cam_y = cam.cam_y;
    }
}

#[cfg(test)]
mod control_tests {
    use super::*;
    use crate::game::load_default_map;

    fn sim(seed: u64) -> GameSim {
        let map = load_default_map().expect("default map");
        GameSim::new(map, seed)
    }

    fn tick_n(sim: &mut GameSim, input: &InputFrame, n: usize) {
        for _ in 0..n {
            sim.tick(input);
        }
    }

    #[test]
    fn idle_on_ground_stays_stand_not_jump_flicker() {
        let mut s = sim(42);
        tick_n(&mut s, &InputFrame::default(), 30);
        let mut jump_frames = 0u32;
        for _ in 0..120 {
            s.tick(&InputFrame::default());
            assert!(
                s.state.player.on_ground,
                "idle should stay grounded, y={}",
                s.state.player.y
            );
            if s.state.player.anim == PlayerAnim::Jump {
                jump_frames += 1;
            }
        }
        assert_eq!(
            jump_frames, 0,
            "idle must not flicker Stand/Jump (was stand_at skipping current foothold)"
        );
        assert_eq!(s.state.player.anim, PlayerAnim::Stand);
    }

    #[test]
    fn walk_right_moves_player() {
        let mut s = sim(1);
        let x0 = s.state.player.x;
        tick_n(
            &mut s,
            &InputFrame {
                right: true,
                ..Default::default()
            },
            30,
        );
        assert!(
            s.state.player.x > x0 + 10.0,
            "walk right: x0={x0} x1={}",
            s.state.player.x
        );
    }

    #[test]
    fn jump_leaves_ground() {
        let mut s = sim(42);
        // 与 mini_game 窗口模式相同 seed；站在主地面 y≈1225
        s.state.player.x = 500.0;
        s.state.player.y = 1225.0;
        s.state.player.on_ground = true;
        s.state.player.vy = 0.0;
        let y0 = s.state.player.y;
        s.tick(&InputFrame {
            jump: true,
            ..Default::default()
        });
        assert!(
            !s.state.player.on_ground || s.state.player.vy < 0.0,
            "jump should leave ground or set upward vy"
        );
        tick_n(&mut s, &InputFrame::default(), 10);
        assert!(s.state.player.y < y0 - 5.0, "jump should raise player");
    }

    #[test]
    fn climb_to_top_auto_stands_on_platform() {
        let mut s = sim(3);
        // rope x=1770, y1=567..679；顶板约 y=565
        s.state.player.x = 1770.0;
        s.state.player.y = 650.0;
        s.state.player.on_ground = false;
        s.state.player.climbing = false;
        // 先抓住绳
        tick_n(
            &mut s,
            &InputFrame {
                up: true,
                ..Default::default()
            },
            5,
        );
        assert!(s.state.player.climbing, "should be climbing");
        // 持续上爬直到顶端
        tick_n(
            &mut s,
            &InputFrame {
                up: true,
                ..Default::default()
            },
            120,
        );
        assert!(
            !s.state.player.climbing,
            "reaching rope top should auto-dismount"
        );
        assert!(
            s.state.player.on_ground,
            "should stand on top platform, y={}",
            s.state.player.y
        );
        assert!(
            (s.state.player.y - 565.0).abs() < 4.0,
            "feet should snap to top plat ~565, got {}",
            s.state.player.y
        );
        // 落地后可左右走，无需先跳
        tick_n(
            &mut s,
            &InputFrame {
                right: true,
                ..Default::default()
            },
            10,
        );
        assert!(
            !s.state.player.climbing,
            "walking right must not re-grab while holding nothing vertical"
        );
        assert!(
            s.state.player.x > 1770.0,
            "should walk right after dismount, x={}",
            s.state.player.x
        );
    }

    #[test]
    fn rope_climb_moves_up() {
        let mut s = sim(3);
        // map_50001 绳 x=1770, y1=567..679
        s.state.player.x = 1770.0;
        s.state.player.y = 650.0;
        s.state.player.on_ground = false;
        s.state.player.climbing = false;
        let y0 = s.state.player.y;
        tick_n(
            &mut s,
            &InputFrame {
                up: true,
                ..Default::default()
            },
            20,
        );
        assert!(s.state.player.climbing, "should grab rope");
        assert!(
            s.state.player.y < y0 - 5.0,
            "climb up: y0={y0} y1={}",
            s.state.player.y
        );
        assert_eq!(s.state.player.climb_kind, "rope");
    }

    #[test]
    fn ladder_climb_moves_up() {
        let mut s = sim(4);
        // ladder x=1477, y1=987..1191
        s.state.player.x = 1477.0;
        s.state.player.y = 1100.0;
        s.state.player.on_ground = false;
        s.state.player.climbing = false;
        let y0 = s.state.player.y;
        tick_n(
            &mut s,
            &InputFrame {
                up: true,
                ..Default::default()
            },
            20,
        );
        assert!(s.state.player.climbing, "should grab ladder");
        assert!(s.state.player.y < y0 - 5.0);
        assert_eq!(s.state.player.climb_kind, "ladder");
    }

    #[test]
    fn attack_damages_mob_in_front() {
        let mut s = sim(5);
        let px = s.state.player.x;
        let py = s.state.player.y;
        s.state.player.facing = 1.0;
        s.state.mobs.clear();
        s.state.mobs.push(MobState {
            mob_id: 130101,
            x: px + 30.0,
            y: py,
            hp: 50,
            max_hp: 50,
            vx: 0.0,
            walk_x1: px - 50.0,
            walk_x2: px + 100.0,
            alive: true,
            hit_t: 0.0,
            die_t: 0.0,
            anim: MobAnim::Move,
            anim_t: 0.0,
            touch_damage: 5,
        });
        s.tick(&InputFrame {
            attack: true,
            ..Default::default()
        });
        let hp = s.state.mobs[0].hp;
        assert!(hp < 50, "attack should damage mob, hp={hp}");
    }

    #[test]
    fn attack_auto_faces_mob_on_the_left() {
        let mut s = sim(5);
        let px = s.state.player.x;
        let py = s.state.player.y;
        s.state.player.facing = 1.0;
        s.state.mobs.clear();
        s.state.mobs.push(MobState {
            mob_id: 130101,
            x: px - 30.0,
            y: py,
            hp: 50,
            max_hp: 50,
            vx: 0.0,
            walk_x1: px - 100.0,
            walk_x2: px + 50.0,
            alive: true,
            hit_t: 0.0,
            die_t: 0.0,
            anim: MobAnim::Move,
            anim_t: 0.0,
            touch_damage: 5,
        });
        s.tick(&InputFrame {
            attack: true,
            ..Default::default()
        });
        assert!(
            s.state.player.facing < 0.0,
            "attack should auto-face left toward mob"
        );
        assert!(s.state.mobs[0].hp < 50, "should still hit after auto-face");
    }

    #[test]
    fn pick_up_collects_nearby_drop() {
        let mut s = sim(6);
        let px = s.state.player.x;
        let py = s.state.player.y;
        s.state.drops.push(DropState {
            kind: DropKind::Meso,
            x: px + 10.0,
            y: py,
            alive: true,
            bob_t: 0.0,
        });
        let meso0 = s.state.meso;
        s.tick(&InputFrame {
            pick_up: true,
            ..Default::default()
        });
        assert!(s.state.meso > meso0, "pick_up should collect meso");
        assert!(s.state.drops.is_empty());
    }

    #[test]
    fn use_potion_heals_player() {
        let mut s = sim(7);
        s.state.mobs.clear();
        s.state.player.hp = 40;
        let potions0 = s.state.potions;
        s.tick(&InputFrame {
            use_potion: true,
            ..Default::default()
        });
        assert_eq!(s.state.player.hp, 70);
        assert_eq!(s.state.potions, potions0 - 1);
    }

    #[test]
    fn ground_truth_exports_state() {
        let s = sim(8);
        let gt = s.ground_truth();
        assert!(gt.max_hp > 0);
        assert!(gt.mob_count > 0);
    }

    #[test]
    fn bot_play_starts_without_potions_and_spawns_npcs() {
        let map = load_default_map().expect("default map");
        let s = GameSim::new_bot_play(map, 99);
        assert_eq!(s.state.potions, 0);
        assert_eq!(s.npc_players.len(), TRAINING_NPC_COUNT);
    }

    #[test]
    fn bot_play_hp_zero_ends_episode() {
        let map = load_default_map().expect("default map");
        let mut s = GameSim::new_bot_play(map, 7);
        s.state.player.hp = 0;
        s.tick(&InputFrame::default());
        assert!(s.is_episode_over());
    }

    #[test]
    fn safe_hurt_knockback_stays_on_ground() {
        let map = load_default_map().expect("default map");
        let s = GameSim::new_bot_play(map.clone(), 1);
        let x = s.state.player.x;
        let y = s.state.player.y;
        for dir in [-1.0_f32, 1.0] {
            let proposed = x + dir * 40.0;
            let result = GameSim::safe_hurt_knockback_x(&map, x, y, dir);
            if map.stand_at(proposed, y + 40.0, 64.0).is_none() {
                assert_eq!(result, x, "knockback off platform should not move player");
            } else {
                assert!((result - proposed).abs() < 0.01);
            }
        }
    }

    #[test]
    fn bot_play_wave_respawns_after_all_mobs_cleared() {
        let map = load_default_map().expect("default map");
        let mut s = GameSim::new_bot_play(map, 3);
        let n0 = s.state.mobs.len();
        assert!(n0 > 0);
        for m in &mut s.state.mobs {
            m.alive = false;
            m.die_t = 0.0;
        }
        s.tick(&InputFrame::default());
        assert_eq!(
            s.state.mobs.iter().filter(|m| m.alive).count(),
            n0,
            "all mobs dead should wave respawn"
        );
    }

    #[test]
    fn void_fall_rescues_onto_foothold_not_spawn_teleport() {
        let map = load_default_map().expect("default map");
        let mut s = GameSim::new_preview(map, 0);
        let spawn_x = s.state.player.x;
        // 复现日志：跳到一层右缘外、二台高度以下的虚空列
        {
            let p = &mut s.state.player;
            p.x = 780.0;
            p.y = s.map.death_y() + 20.0;
            p.vx = 80.0;
            p.vy = 200.0;
            p.on_ground = false;
        }
        s.tick(&InputFrame {
            right: true,
            ..InputFrame::default()
        });
        assert!(
            s.state.player.y <= s.map.death_y(),
            "must not remain below death_y"
        );
        assert!(
            s.state.player.on_ground,
            "void rescue should stand on a foothold"
        );
        // 不应整段传送回出生点（旧 check_void_fall 行为）
        assert!(
            (s.state.player.x - spawn_x).abs() > 80.0 || s.state.player.x > 600.0,
            "rescue should prefer nearby foothold over spawn; x={} spawn={}",
            s.state.player.x,
            spawn_x
        );
    }

    #[test]
    fn airborne_cannot_strafe_into_unsupported_column() {
        let map = load_default_map().expect("default map");
        let mut s = GameSim::new_preview(map, 0);
        // 站在一层上方腾空（仍有下方一层），尝试冲进 x=780 虚空列
        {
            let p = &mut s.state.player;
            p.x = 740.0;
            p.y = 1180.0;
            p.vx = 0.0;
            p.vy = 50.0;
            p.on_ground = false;
        }
        for _ in 0..30 {
            s.tick(&InputFrame {
                right: true,
                ..InputFrame::default()
            });
            if s.state.player.on_ground {
                break;
            }
        }
        assert!(
            s.state.player.x < 770.0 || s.state.player.on_ground,
            "must not deep-strafe into void column; x={} y={} ground={}",
            s.state.player.x,
            s.state.player.y,
            s.state.player.on_ground
        );
        assert!(
            s.state.player.y <= s.map.death_y(),
            "must never cross death_y"
        );
    }
}
