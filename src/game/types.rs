pub const WINDOW_W: f32 = 1368.0;
pub const WINDOW_H: f32 = 768.0;
pub const WORLD_VIEW_H: f32 = 697.0;

/// 玩家头顶/脚下名牌（仿真渲染用）
pub const DEFAULT_PLAYER_NAME: &str = "光头强加强版";

/// 自动玩模式：装饰玩家数量与资源（`assets/player/` 下目录名）
pub const TRAINING_NPC_COUNT: usize = 4;
/// 训练装饰玩家名牌
pub const TRAINING_NPC_NAMES: [&str; TRAINING_NPC_COUNT] =
    ["南港商人", "冒险萌新", "路过的骑士", "摆摊小贩"];
pub const TRAINING_NPC_SPRITES: [&str; TRAINING_NPC_COUNT] =
    ["默认女新手", "男战士", "男魔法师", "男弓箭手"];
/// 训练模式怪物掉落红药概率（少量红药够学盲喝即可；主奖励为金币）
pub const TRAINING_POTION_DROP_CHANCE: f32 = 0.04;
/// NEAT 训练：每只怪金币堆数（含首堆）
pub const TRAINING_MESO_PILES_MIN: u32 = 2;
pub const TRAINING_MESO_PILES_MAX: u32 = 3;
/// NEAT 训练：单堆金币数量
pub const TRAINING_MESO_AMOUNT_MIN: u32 = 3;
pub const TRAINING_MESO_AMOUNT_MAX: u32 = 8;
/// 训练/自动玩：是否在多平台候选点中随机出生（增强泛化）
pub const TRAINING_RANDOM_PLAYER_SPAWN: bool = true;
/// 候选平台最小宽度（px）
pub const PLAYER_SPAWN_MIN_PLATFORM_W: f32 = 72.0;
/// 出生点距平台端内缩
pub const PLAYER_SPAWN_EDGE_PAD: f32 = 20.0;
/// 同高度合并容差（量化到出生候选）
pub const PLAYER_SPAWN_Y_BAND: f32 = 24.0;
/// 普通/小游戏模式：红药少见，主掉落为金币
pub const NORMAL_POTION_DROP_CHANCE: f32 = 0.05;
/// 普通模式每只怪金币堆数
pub const NORMAL_MESO_PILES_MIN: u32 = 1;
pub const NORMAL_MESO_PILES_MAX: u32 = 2;
/// 名牌：脚点下方间距、内边距、字号（凤凰点阵体 16px，对齐原版 UIImageText）
pub const NAME_TAG_GAP_BELOW_FEET: f32 = 3.0;
pub const NAME_TAG_PAD_X: f32 = 4.0;
pub const NAME_TAG_PAD_Y: f32 = 2.0;
pub const NAME_TAG_FONT_SIZE: u16 = 16;
pub const NAME_TAG_BG_ALPHA: f32 = 0.58;
pub const LOGIC_HZ: f32 = 60.0;
pub const LOGIC_DT: f32 = 1.0 / LOGIC_HZ;

// —— 原版 Map.wz / Physics.img（GMS v83）——
/// 玩家基础行走速度（Speed 属性 100%），单位 px/s
pub const PLAYER_WALK_SPEED: f32 = 125.0;
/// 兼容旧名：等同 PLAYER_WALK_SPEED
pub const PLAYER_SPEED: f32 = PLAYER_WALK_SPEED;
/// 重力加速度（Physics.img gravityAcc）
pub const GRAVITY: f32 = 2000.0;
/// 起跳初速度（Physics / 客户端常见值，向上为负）
pub const JUMP_VY: f32 = -555.0;
/// 绳梯攀爬（客户端约每输入帧数像素，折合到 60Hz）
pub const CLIMB_SPEED: f32 = 100.0;

/// 将 Mob.wz `info/speed` 转为水平速度 px/s。
/// 公式：walkSpeed × max(0.05, (100 + speed) / 100)
/// speed 为负表示比基准慢（蜗牛约 -50～-70，花蘑菇 0）。
pub fn mob_walk_speed_from_wz(wz_speed: i32) -> f32 {
    let ratio = ((100 + wz_speed) as f32 / 100.0).max(0.05);
    PLAYER_WALK_SPEED * ratio
}
/// 离地后仍可起跳的宽限时间（土狼时间）
pub const COYOTE_TIME: f32 = 0.10;
/// 落地前提前按跳仍生效的缓冲
pub const JUMP_BUFFER: f32 = 0.14;
pub const PLAYER_MAX_HP: i32 = 100;
pub const PLAYER_ATTACK_DAMAGE: i32 = 15;
pub const ATTACK_COOLDOWN: f32 = 0.35;
pub const ATTACK_DURATION: f32 = 0.25;
pub const HURT_DURATION: f32 = 0.3;
pub const INVULN_DURATION: f32 = 1.5;
pub const POTION_HEAL: i32 = 30;
pub const ROPE_GRAB_X: f32 = 18.0;
/// 碰撞用近似身高（脚底向上）
pub const PLAYER_BODY_H: f32 = 56.0;
/// 侧墙阻挡只检测脚边这一段（不含头顶），避免头顶平台侧面误挡
pub const WALL_HIT_H: f32 = 28.0;
/// 同高度平台容差（可吸附/微台阶）
pub const SAME_LEVEL_TOL: f32 = 12.0;
/// 探测下方是否有可落地平台的最大落差
pub const FALL_PROBE: f32 = 720.0;

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
    /// Mob.wz info/speed（原版相对速度）
    pub wz_speed: i32,
}

impl MobStats {
    pub fn walk_speed(self) -> f32 {
        mob_walk_speed_from_wz(self.wz_speed)
    }
}

pub fn mob_stats(mob_id: u32) -> MobStats {
    // wz_speed 来自 maplestory.io GMS/83 mob meta.speed
    match mob_id {
        100101 => MobStats {
            hp: 30,
            touch_damage: 8,
            wz_speed: -50, // 蓝蜗牛 → 62.5 px/s
        },
        100100 => MobStats {
            hp: 25,
            touch_damage: 6,
            wz_speed: -65, // 绿蜗牛 → 43.75 px/s
        },
        130101 => MobStats {
            hp: 40,
            touch_damage: 10,
            wz_speed: -50, // 红蜗牛 → 62.5 px/s
        },
        1210102 => MobStats {
            hp: 50,
            touch_damage: 12,
            wz_speed: 0, // 花蘑菇 → 125 px/s
        },
        130100 => MobStats {
            hp: 60,
            touch_damage: 15,
            wz_speed: -70, // 树怪 → 37.5 px/s
        },
        _ => MobStats {
            hp: 30,
            touch_damage: 8,
            wz_speed: -50,
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
