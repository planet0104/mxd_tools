//! 训练用装饰玩家（YOLO「玩家」干扰项，由 SelfTracker 运动确认排除自身）。
//!
//! - 不受怪物碰撞伤害（无 HP，不参与 `check_mob_touch`）
//! - 偶发普攻仅作画面真实感，**不可击杀怪物**（怪物 HP 最低保留 1），避免抢怪

use rand::Rng;

use super::map::GameMap;
use super::types::{
    PlayerAnim, ATTACK_DURATION, TRAINING_NPC_COUNT, TRAINING_NPC_NAMES, TRAINING_NPC_SPRITES,
};
use super::{MobAnim, MobState};

/// 每只 NPC 每秒约 3% 概率尝试普攻（平均 ~30s 一次，偏低）
pub const NPC_ATTACK_CHANCE_PER_SEC: f32 = 0.03;
/// 普攻冷却（秒）
pub const NPC_ATTACK_COOLDOWN: f32 = 2.8;
/// 装饰玩家伤害（仅作表现，不可击杀）
pub const NPC_ATTACK_DAMAGE: i32 = 8;

#[derive(Debug, Clone)]
pub struct NpcPlayerState {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub facing: f32,
    pub walk_x1: f32,
    pub walk_x2: f32,
    pub anim: PlayerAnim,
    pub anim_t: f32,
    pub attack_t: f32,
    pub attack_cd: f32,
    pub name: String,
    /// `assets/player/<sprite_dir>/`
    pub sprite_dir: String,
}

/// 在若干不同高度平台上生成装饰玩家，平台间巡逻。
pub fn spawn_training_npcs(map: &GameMap, rng: &mut impl Rng) -> Vec<NpcPlayerState> {
    let anchors: [(f32, f32); TRAINING_NPC_COUNT] = [
        (762.0, 805.0),
        (1200.0, 1105.0),
        (480.0, 1225.0),
        (1580.0, 1225.0),
    ];
    let mut names: Vec<&str> = TRAINING_NPC_NAMES.to_vec();
    for i in 0..names.len() {
        let j = rng.gen_range(0..names.len());
        names.swap(i, j);
    }

    anchors
        .into_iter()
        .enumerate()
        .map(|(i, (x, y))| {
            let (walk_x1, walk_x2) = map.walk_range_at(x, y);
            let x = x.clamp(walk_x1, walk_x2);
            let vx = if rng.gen_bool(0.5) { 62.5 } else { -62.5 };
            NpcPlayerState {
                x,
                y,
                vx,
                facing: if vx > 0.0 { 1.0 } else { -1.0 },
                walk_x1,
                walk_x2,
                anim: PlayerAnim::Walk,
                anim_t: 0.0,
                attack_t: 0.0,
                attack_cd: rng.gen_range(0.0..1.5),
                name: names[i % names.len()].to_string(),
                sprite_dir: TRAINING_NPC_SPRITES[i % TRAINING_NPC_SPRITES.len()].to_string(),
            }
        })
        .collect()
}

pub fn tick_npc_players(
    npcs: &mut [NpcPlayerState],
    mobs: &mut [MobState],
    dt: f32,
    rng: &mut impl Rng,
) {
    for npc in npcs {
        npc.anim_t += dt;
        npc.attack_t = (npc.attack_t - dt).max(0.0);
        npc.attack_cd = (npc.attack_cd - dt).max(0.0);

        if npc.attack_t > 0.0 {
            npc.anim = PlayerAnim::Attack;
            continue;
        }

        if npc.attack_cd <= 0.0
            && rng.gen_bool((NPC_ATTACK_CHANCE_PER_SEC * dt) as f64)
            && try_npc_attack(npc, mobs)
        {
            npc.attack_t = ATTACK_DURATION;
            npc.attack_cd = NPC_ATTACK_COOLDOWN;
            npc.anim = PlayerAnim::Attack;
            continue;
        }

        npc.x += npc.vx * dt;
        if npc.x <= npc.walk_x1 {
            npc.x = npc.walk_x1;
            npc.vx = npc.vx.abs();
            npc.facing = 1.0;
        }
        if npc.x >= npc.walk_x2 {
            npc.x = npc.walk_x2;
            npc.vx = -npc.vx.abs();
            npc.facing = -1.0;
        }
        npc.anim = PlayerAnim::Walk;
    }
}

/// 身前短距离普攻；怪物 HP 不低于 1，不掉落、不触发波次清空。
fn try_npc_attack(npc: &NpcPlayerState, mobs: &mut [MobState]) -> bool {
    let (x1, x2) = if npc.facing > 0.0 {
        (npc.x + 4.0, npc.x + 52.0)
    } else {
        (npc.x - 52.0, npc.x - 4.0)
    };
    let y1 = npc.y - 68.0;
    let y2 = npc.y + 16.0;
    let mut hit = false;
    for mob in mobs.iter_mut() {
        if !mob.alive {
            continue;
        }
        if mob.x >= x1 && mob.x <= x2 && mob.y >= y1 && mob.y <= y2 {
            mob.hp = (mob.hp - NPC_ATTACK_DAMAGE).max(1);
            mob.hit_t = 0.15;
            mob.anim = MobAnim::Hit;
            hit = true;
        }
    }
    hit
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::types::mob_stats;

    #[test]
    fn npc_attack_cannot_kill_mob() {
        let mut npc = NpcPlayerState {
            x: 100.0,
            y: 200.0,
            vx: 0.0,
            facing: 1.0,
            walk_x1: 0.0,
            walk_x2: 300.0,
            anim: PlayerAnim::Stand,
            anim_t: 0.0,
            attack_t: 0.0,
            attack_cd: 0.0,
            name: "test".into(),
            sprite_dir: "男战士".into(),
        };
        let stats = mob_stats(100101);
        let mut mobs = vec![MobState {
            mob_id: 100101,
            x: 120.0,
            y: 200.0,
            hp: 5,
            max_hp: stats.hp,
            vx: 0.0,
            walk_x1: 0.0,
            walk_x2: 200.0,
            alive: true,
            hit_t: 0.0,
            die_t: 0.0,
            anim: MobAnim::Move,
            anim_t: 0.0,
            touch_damage: stats.touch_damage,
        }];
        assert!(try_npc_attack(&npc, &mut mobs));
        assert_eq!(mobs[0].hp, 1);
        assert!(mobs[0].alive);
    }
}
