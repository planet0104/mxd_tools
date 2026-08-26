use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::game::camera::WorldCamera;
use crate::game::input::InputFrame;
use crate::game::map::{GameMap, WalkAhead};
use crate::game::types::*;

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
    spawn_x: f32,
    spawn_y: f32,
    rng: StdRng,
}

impl GameSim {
    pub fn new(map: GameMap, seed: u64) -> Self {
        let (spawn_x, spawn_y) = map.default_spawn();
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
                potions: 5,
                kills: 0,
                modal: GameModal::None,
                cam_x: 0.0,
                cam_y: 0.0,
                tick: 0,
                portal_hint: None,
            },
            spawn_x,
            spawn_y,
            rng: StdRng::seed_from_u64(seed),
        };
        sim.spawn_mobs();
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
        // 只有明显掉出最低平台以下才重生，避免站立时误触发
        let too_low = p.y > self.map.death_y();
        if too_low {
            self.respawn_player();
        }
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

    pub fn tick(&mut self, input: &InputFrame) {
        if input.restart && self.state.modal == GameModal::GameOver {
            *self = GameSim::new(self.map.clone(), self.rng.gen());
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

        if input.use_potion {
            self.use_potion();
        }

        self.state.tick += 1;
        let dt = LOGIC_DT;
        self.tick_player(input, dt);
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
            self.state.modal = GameModal::GameOver;
            return;
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

        let can_move = self.state.player.hurt_t <= 0.0 && self.state.player.attack_t <= 0.0;
        let do_attack =
            can_move && !self.state.player.climbing && input.attack && self.state.player.attack_cd <= 0.0;

        if do_attack {
            let p = &mut self.state.player;
            p.attack_t = ATTACK_DURATION;
            p.attack_cd = ATTACK_COOLDOWN;
            p.anim = PlayerAnim::Attack;
            p.anim_t = 0.0;
        }

        if self.state.player.climbing {
            self.tick_player_climb(input, dt);
        } else if can_move {
            let grab = self
                .map
                .rope_at(self.state.player.x, self.state.player.y)
                .map(|r| (r.x, r.kind.clone()))
                .filter(|_| input.up || input.down);
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
            let (prev_x, prev_y, vx, vy, on_ground, fh) = {
                let p = &self.state.player;
                let fh = if p.on_ground && p.fh_layer >= 0 {
                    Some((p.fh_layer, p.fh_group))
                } else {
                    None
                };
                (p.x, p.y, p.vx * 0.5, p.vy, p.on_ground, fh)
            };
            let mut vy = vy;
            if !on_ground || vy > 0.0 {
                vy += GRAVITY * dt;
            }
            let desire_x = prev_x + vx * dt;
            let y_hi = prev_y - WALL_HIT_H;
            let new_x = self
                .map
                .resolve_wall_x(prev_x, desire_x, y_hi, prev_y - 2.0, fh);
            let next_y = prev_y + vy * dt;
            let landed = if vy >= 0.0 {
                self.map.land_at(new_x, prev_y, next_y)
            } else {
                None
            };
            let p = &mut self.state.player;
            p.vx = vx;
            p.x = new_x;
            if let Some(st) = landed {
                Self::apply_stand(p, Some(st));
            } else {
                p.y = next_y;
                p.vy = vy;
                p.on_ground = false;
            }
        }

        {
            let p = &mut self.state.player;
            p.x = p.x.clamp(16.0, self.map.width - 16.0);
            // 只限制上边界；下边界由虚空重生处理，避免卡在泥土高度
            p.y = p.y.max(16.0);
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
        let mut new_x =
            self.map
                .resolve_wall_x(prev_x, desire_x, y_hi, prev_y - 2.0, wall_fh);

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

        let stand = if on_ground && vy >= 0.0 {
            self.map.stand_at(new_x, feet_y, SAME_LEVEL_TOL)
        } else {
            None
        };
        let next_y = if on_ground && vy >= 0.0 {
            feet_y + vy * dt
        } else {
            prev_y + vy * dt
        };
        let landed = if !(on_ground && vy >= 0.0) && vy >= 0.0 {
            self.map.land_at(new_x, prev_y, next_y)
        } else {
            None
        };

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
        let leave = jump_off || (at_bottom && input.down && !input.up);

        if jump_off {
            p.climbing = false;
            p.vy = JUMP_VY * 0.85;
            p.on_ground = false;
        } else if leave && at_bottom {
            p.climbing = false;
            if let Some(st) = self.map.stand_at(p.x, p.y + 2.0, 48.0) {
                Self::apply_stand(p, Some(st));
            } else {
                p.on_ground = false;
            }
        } else if at_top && !input.up {
            if let Some(st) = self.map.stand_at(p.x, p.y + 8.0, 24.0) {
                p.climbing = false;
                Self::apply_stand(p, Some(st));
            }
        }

        if p.climbing {
            p.anim = PlayerAnim::Climb;
        }
    }

    fn try_attack_mobs(&mut self) {
        let p = &self.state.player;
        // 身前攻击框（脚底坐标）：水平朝向 + 覆盖同台怪身高
        let (x1, x2) = if p.facing > 0.0 {
            (p.x + 4.0, p.x + 56.0)
        } else {
            (p.x - 56.0, p.x - 4.0)
        };
        let y1 = p.y - 72.0;
        let y2 = p.y + 20.0;
        let mut loot: Vec<(f32, f32)> = Vec::new();
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
        if self.rng.gen_bool(0.3) {
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
        let p = &mut self.state.player;
        if p.invuln_t > 0.0 || p.hurt_t > 0.0 {
            return;
        }
        for mob in &self.state.mobs {
            if !mob.alive {
                continue;
            }
            let dx = p.x - mob.x;
            let dy = p.y - mob.y;
            if dx.abs() < 28.0 && dy.abs() < 36.0 {
                p.hp -= mob.touch_damage;
                p.hurt_t = HURT_DURATION;
                p.invuln_t = INVULN_DURATION;
                p.anim = PlayerAnim::Hurt;
                p.x += dx.signum() * 40.0;
                break;
            }
        }
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
            match drop.kind {
                DropKind::Meso => {
                    self.state.meso += self.rng.gen_range(1..=5);
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
    fn walk_right_moves_player() {
        let mut s = sim(1);
        let x0 = s.state.player.x;
        tick_n(&mut s, &InputFrame { right: true, ..Default::default() }, 30);
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
        assert!(
            hp < 50,
            "attack should damage mob, hp={hp}"
        );
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
}
