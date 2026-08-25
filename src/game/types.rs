pub const WINDOW_W: f32 = 1368.0;
pub const WINDOW_H: f32 = 768.0;
pub const WORLD_VIEW_H: f32 = 696.0;
pub const LOGIC_HZ: f32 = 60.0;
pub const LOGIC_DT: f32 = 1.0 / LOGIC_HZ;

pub const PLAYER_SPEED: f32 = 180.0;
pub const GRAVITY: f32 = 1200.0;
pub const JUMP_VY: f32 = -420.0;
pub const PLAYER_MAX_HP: i32 = 100;
pub const PLAYER_ATTACK_DAMAGE: i32 = 15;
pub const ATTACK_COOLDOWN: f32 = 0.35;
pub const ATTACK_DURATION: f32 = 0.25;
pub const HURT_DURATION: f32 = 0.3;
pub const INVULN_DURATION: f32 = 1.5;
pub const POTION_HEAL: i32 = 30;
pub const CLIMB_SPEED: f32 = 140.0;
pub const ROPE_GRAB_X: f32 = 18.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerAnim {
    Stand,
    Walk,
    Jump,
    Attack,
    Hurt,
    Climb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobAnim {
    Move,
    Hit,
    Die,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropKind {
    Meso,
    RedPotion,
}

#[derive(Debug, Clone, Copy)]
pub struct MobStats {
    pub hp: i32,
    pub touch_damage: i32,
    pub speed_factor: f32,
}

pub fn mob_stats(mob_id: u32) -> MobStats {
    match mob_id {
        100101 => MobStats {
            hp: 30,
            touch_damage: 8,
            speed_factor: 0.6,
        },
        130101 => MobStats {
            hp: 40,
            touch_damage: 10,
            speed_factor: 0.65,
        },
        1210102 => MobStats {
            hp: 50,
            touch_damage: 12,
            speed_factor: 0.7,
        },
        130100 => MobStats {
            hp: 60,
            touch_damage: 15,
            speed_factor: 0.5,
        },
        _ => MobStats {
            hp: 30,
            touch_damage: 8,
            speed_factor: 0.6,
        },
    }
}

pub fn mob_dir_name(mob_id: u32) -> &'static str {
    match mob_id {
        100101 => "100101_蓝蜗牛",
        130101 => "130101_红蜗牛",
        1210102 => "1210102_花蘑菇",
        130100 => "130100_树怪",
        _ => "100101_蓝蜗牛",
    }
}
