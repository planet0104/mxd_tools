//! 类人操作节奏：随机短休息、反应延迟、移动/攻击脉冲，避免 60Hz 疯狂连按。
//!
//! 休息与反应延迟只影响**寻路走位**；砍怪 / 喝药 / 爬绳 / 跳跃始终可立刻生效。

use rand::prelude::*;
use rand::rngs::StdRng;

use super::input::InputFrame;

/// 中等刷怪节奏：偏稳、偶尔发呆，仍保持一定推进力。
#[derive(Debug, Clone)]
pub struct HumanPace {
    rng: StdRng,
    idle_until: u32,
    reaction_until: u32,
    last_intent: InputFrame,
    move_tick: u32,
    attack_refractory: u32,
    pick_up_cooldown: u32,
    next_rest_check: u32,
}

/// 移动：约 18/20 帧按住，减少走停抖动。
const MOVE_CYCLE: u32 = 20;
const MOVE_ON: u32 = 18;
/// 攻击：通过后冷却约 0.4s，接近 sim 攻击 CD。
const ATTACK_REFRACTORY: u32 = 18;
/// 捡取：最多约 4Hz。
const PICK_UP_INTERVAL: u32 = 15;
/// 意图切换反应延迟（逻辑帧）——仅延迟左右走，不延迟攻击。
const REACTION_MIN: u32 = 1;
const REACTION_MAX: u32 = 3;
/// 随机休息：仅寻路时触发，不打断战斗。
const REST_CHECK_MIN: u32 = 360;
const REST_CHECK_MAX: u32 = 600;
const REST_CHANCE: f64 = 0.08;
const REST_IDLE_MIN: u32 = 12;
const REST_IDLE_MAX: u32 = 30;

impl Default for HumanPace {
    fn default() -> Self {
        Self::new(0)
    }
}

impl HumanPace {
    pub fn new(episode_seed: u64) -> Self {
        let mut s = Self {
            rng: StdRng::seed_from_u64(episode_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            idle_until: 0,
            reaction_until: 0,
            last_intent: InputFrame::default(),
            move_tick: 0,
            attack_refractory: 0,
            pick_up_cooldown: 0,
            next_rest_check: REST_CHECK_MIN,
        };
        s.next_rest_check = s.rng.gen_range(REST_CHECK_MIN..=REST_CHECK_MAX);
        s
    }

    pub fn reset(&mut self, episode_seed: u64) {
        *self = Self::new(episode_seed);
    }

    /// 视觉帧更新意图时调用（约 5Hz）。
    pub fn on_intent(&mut self, raw: InputFrame, tick: u32) {
        // 战斗意图立刻取消休息，避免站等时进入 idle 后砍不到。
        if is_combat_action(raw) {
            self.idle_until = 0;
        }

        if intent_changed(raw, self.last_intent) {
            let delay = self.rng.gen_range(REACTION_MIN..=REACTION_MAX);
            self.reaction_until = tick.saturating_add(delay);
        }
        self.last_intent = raw;

        if tick >= self.next_rest_check {
            self.next_rest_check = tick + self.rng.gen_range(REST_CHECK_MIN..=REST_CHECK_MAX);
            // 仅纯寻路走位才休息；站桩等怪 / 砍怪 / 躲怪不休息。
            if is_explore_locomotion(raw) && self.rng.gen_bool(REST_CHANCE) {
                let rest = self.rng.gen_range(REST_IDLE_MIN..=REST_IDLE_MAX);
                self.idle_until = tick.saturating_add(rest);
            }
        }
    }

    /// 门控剥掉本帧攻击时退回 refractory，避免空挥占 CD。
    pub fn refund_attack(&mut self) {
        self.attack_refractory = 0;
    }

    /// 每个逻辑帧（60Hz）将 bot 意图转为实际按键。
    pub fn apply(&mut self, raw: InputFrame, tick: u32) -> InputFrame {
        let mut out = raw;

        let resting = tick < self.idle_until;
        let reacting = tick < self.reaction_until;

        // 休息/反应：只屏蔽左右走，砍怪与垂直机动始终放行。
        if resting || reacting {
            out.left = false;
            out.right = false;
        }

        if out.left || out.right {
            self.move_tick = self.move_tick.wrapping_add(1);
            if self.move_tick % MOVE_CYCLE >= MOVE_ON {
                out.left = false;
                out.right = false;
            }
        } else {
            self.move_tick = 0;
        }

        if out.attack {
            if self.attack_refractory > 0 {
                out.attack = false;
            } else {
                // 先占坑；若随后门控剥掉攻击，调用方应 refund_attack。
                self.attack_refractory = ATTACK_REFRACTORY;
            }
        }
        if self.attack_refractory > 0 {
            self.attack_refractory -= 1;
        }

        if out.pick_up {
            if self.pick_up_cooldown > 0 {
                out.pick_up = false;
            } else {
                self.pick_up_cooldown = PICK_UP_INTERVAL;
            }
        }
        if self.pick_up_cooldown > 0 {
            self.pick_up_cooldown -= 1;
        }

        out
    }
}

/// 砍怪 / 吃药 / 跳 / 爬：不受随机休息影响。
fn is_combat_action(raw: InputFrame) -> bool {
    raw.use_potion || raw.jump || raw.attack || raw.up || raw.down
}

/// 纯左右寻路（无战斗键）才允许进入随机休息。
fn is_explore_locomotion(raw: InputFrame) -> bool {
    (raw.left || raw.right) && !is_combat_action(raw) && !raw.pick_up
}

fn intent_changed(a: InputFrame, b: InputFrame) -> bool {
    a.left != b.left
        || a.right != b.right
        || a.jump != b.jump
        || a.attack != b.attack
        || a.up != b.up
        || a.down != b.down
        || a.use_potion != b.use_potion
        || a.pick_up != b.pick_up
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_only_suppresses_locomotion_not_attack() {
        let mut pace = HumanPace::new(1);
        pace.idle_until = 100;
        let raw = InputFrame {
            right: true,
            attack: true,
            ..Default::default()
        };
        let out = pace.apply(raw, 50);
        assert!(!out.right, "rest should block walk");
        assert!(out.attack, "rest must not block attack");
    }

    #[test]
    fn idle_suppresses_pure_walk() {
        let mut pace = HumanPace::new(1);
        pace.idle_until = 100;
        let raw = InputFrame {
            right: true,
            ..Default::default()
        };
        let out = pace.apply(raw, 50);
        assert!(!out.right && !out.left && !out.attack);
    }

    #[test]
    fn combat_intent_cancels_rest() {
        let mut pace = HumanPace::new(1);
        pace.idle_until = 1000;
        pace.on_intent(
            InputFrame {
                attack: true,
                ..Default::default()
            },
            10,
        );
        assert_eq!(pace.idle_until, 0);
        let out = pace.apply(
            InputFrame {
                attack: true,
                ..Default::default()
            },
            10,
        );
        assert!(out.attack);
    }

    #[test]
    fn rest_not_scheduled_on_noop_stand() {
        let mut pace = HumanPace::new(2);
        pace.next_rest_check = 0;
        // 站桩等怪：noop，不应开休息。
        for tick in 0..50 {
            pace.on_intent(InputFrame::default(), tick);
        }
        assert_eq!(pace.idle_until, 0, "standing wait must not start explore rest");
    }

    #[test]
    fn movement_pulses_off_phase() {
        let mut pace = HumanPace::new(2);
        let raw = InputFrame {
            right: true,
            ..Default::default()
        };
        pace.move_tick = MOVE_ON;
        let out = pace.apply(raw, 200);
        assert!(!out.right);
    }

    #[test]
    fn attack_refractory_limits_spam() {
        let mut pace = HumanPace::new(3);
        let raw = InputFrame {
            attack: true,
            ..Default::default()
        };
        assert!(pace.apply(raw, 10).attack);
        assert!(!pace.apply(raw, 11).attack);
        assert!(!pace.apply(raw, 12).attack);
    }

    #[test]
    fn refund_attack_allows_immediate_retry() {
        let mut pace = HumanPace::new(4);
        let raw = InputFrame {
            attack: true,
            ..Default::default()
        };
        assert!(pace.apply(raw, 10).attack);
        pace.refund_attack();
        assert!(
            pace.apply(raw, 11).attack,
            "after gate strip refund, next frame may attack again"
        );
    }
}
