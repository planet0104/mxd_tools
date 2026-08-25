use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::game::camera::WorldCamera;
use crate::game::input::InputFrame;
use crate::game::map::GameMap;
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
        if let Some(gy) = self.map.ground_at(x, self.state.player.y + 40.0, 120.0) {
            self.state.player.y = gy;
            self.state.player.on_ground = true;
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
            self.state.mobs.push(MobState {
                mob_id: sp.mob_id,
                x: sp.x,
                y: sp.y,
                hp: stats.hp,
                max_hp: stats.hp,
                vx: stats.speed_factor * PLAYER_SPEED * if self.rng.gen_bool(0.5) { 1.0 } else { -1.0 },
                walk_x1: sp.walk_x1.min(sp.walk_x2),
                walk_x2: sp.walk_x1.max(sp.walk_x2),
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
            let p = &mut self.state.player;
            p.vx *= 0.5;
            if !p.on_ground || p.vy > 0.0 {
                p.vy += GRAVITY * dt;
            }
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            let gy = self.map.ground_at(p.x, p.y + 2.0, 48.0);
            Self::apply_ground(p, gy);
        }

        {
            let p = &mut self.state.player;
            p.x = p.x.clamp(16.0, self.map.width - 16.0);
            let y_max = if p.climbing {
                self.map.height - 16.0
            } else {
                self.map.death_y()
            };
            p.y = p.y.clamp(16.0, y_max);
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

    fn apply_ground(p: &mut PlayerState, gy: Option<f32>) {
        if let Some(gy) = gy {
            // 脚在平台附近（略上或略下）且非上升时吸附
            if p.vy >= -10.0 && p.y >= gy - 16.0 && p.y <= gy + 24.0 {
                p.y = gy;
                p.vy = 0.0;
                p.on_ground = true;
                return;
            }
        }
        // 有平台但未吸附，或脚下无平台
        if p.vy < -10.0 {
            // 上升中保持离地
            p.on_ground = false;
        } else if gy.is_none() {
            p.on_ground = false;
        }
    }

    fn tick_player_move(&mut self, input: &InputFrame, dt: f32) {
        let p = &mut self.state.player;
        let h = input.horizontal();
        if h.abs() > 0.01 {
            p.vx = h * PLAYER_SPEED;
            p.facing = h.signum();
        } else {
            p.vx = 0.0;
        }
        if input.jump && p.on_ground {
            p.vy = JUMP_VY;
            p.on_ground = false;
        }
        if !p.on_ground || p.vy > 0.0 {
            p.vy += GRAVITY * dt;
        }
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        // 下落越快探测越深，避免高速穿地
        let max_drop = 48.0_f32.max(p.vy.abs() * dt + 24.0);
        let gy = self.map.ground_at(p.x, p.y + 2.0, max_drop);
        Self::apply_ground(p, gy);
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
            let gy = self.map.ground_at(p.x, p.y + 2.0, 48.0);
            Self::apply_ground(p, gy);
        } else if at_top && !input.up {
            if let Some(gy) = self.map.ground_at(p.x, p.y + 8.0, 24.0) {
                p.climbing = false;
                p.y = gy;
                p.on_ground = true;
            }
        }

        if p.climbing {
            p.anim = PlayerAnim::Climb;
        }
    }

    fn try_attack_mobs(&mut self) {
        let p = &self.state.player;
        let (x1, x2) = if p.facing > 0.0 {
            (p.x, p.x + 48.0)
        } else {
            (p.x - 48.0, p.x)
        };
        let y1 = p.y - 48.0;
        let y2 = p.y - 8.0;
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
        for drop in &mut self.state.drops {
            if !drop.alive {
                continue;
            }
            drop.bob_t += dt;
            let dy = (drop.y - 8.0) - py;
            let dx = drop.x - px;
            let pick = input.pick_up || (dx * dx + dy * dy).sqrt() < 36.0;
            if pick {
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
