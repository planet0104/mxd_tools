//! 类人操作节奏：随机短休息、反应延迟、移动/攻击脉冲，避免 60Hz 疯狂连按。
//!
//! 休息与反应延迟只影响**寻路走位**；砍怪 / 喝药始终可立刻生效。
//! 爬绳抓绳与左右转向另有 0.5s 量级的保持/节流（已挂绳时仍每帧传 up/down）。
//! 输出端另有硬闸：移动/跳/垂直组合切换不超过约 4Hz，从根本上杜绝每秒 5～10+ 次换键。

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
    jump_refractory: u32,
    pick_up_cooldown: u32,
    next_rest_check: u32,
    /// 生效中的左右方向：-1 左 / 0 无 / 1 右
    latched_h: i8,
    h_hold_until: u32,
    /// 下次允许 up/down/jump 抓绳脉冲
    climb_pulse_allowed_at: u32,
    /// 转向/爬绳抓绳脉冲间隔（逻辑帧，60Hz 下 30≈0.5s）
    pub locomotion_hold_ticks: u32,
    /// 上一帧真正发出的按键（输出硬闸）。
    last_emit: InputFrame,
    /// 上次「移动/跳/垂直」组合发生变化的逻辑帧。
    last_control_change_tick: u32,
}

/// 移动：约 18/20 帧按住，减少走停抖动。
const MOVE_CYCLE: u32 = 20;
const MOVE_ON: u32 = 18;
/// 攻击：通过后冷却约 0.4s，接近 sim 攻击 CD。
const ATTACK_REFRACTORY: u32 = 18;
/// 跳跃：约 0.45s 内不再起跳，避免 60Hz 连跳。
const JUMP_REFRACTORY: u32 = 28;
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
/// 转向保持 / 地面抓绳脉冲间隔（逻辑帧，60Hz 下 30≈0.5s）
pub const DEFAULT_LOCOMOTION_HOLD_TICKS: u32 = 30;
/// 输出硬闸：移动/跳/垂直签名最短切换间隔（15 帧≈4Hz）。高于此即非人类连打。
pub const MIN_CONTROL_CHANGE_TICKS: u32 = 15;

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
            jump_refractory: 0,
            pick_up_cooldown: 0,
            next_rest_check: REST_CHECK_MIN,
            latched_h: 0,
            h_hold_until: 0,
            climb_pulse_allowed_at: 0,
            locomotion_hold_ticks: DEFAULT_LOCOMOTION_HOLD_TICKS,
            last_emit: InputFrame::default(),
            last_control_change_tick: 0,
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

        if out.jump {
            if self.jump_refractory > 0 {
                out.jump = false;
            } else {
                self.jump_refractory = JUMP_REFRACTORY;
            }
        }
        if self.jump_refractory > 0 {
            self.jump_refractory -= 1;
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

    /// 在 [apply] 之后：保持转向、节流地面抓绳；**已挂绳时**不节流 up/down。
    /// 换向须等满 hold（默认约 0.5s），禁止半途提前解锁造成左右对抽。
    pub fn apply_locomotion_hold(
        &mut self,
        mut out: InputFrame,
        tick: u32,
        climbing: bool,
        intent: InputFrame,
    ) -> InputFrame {
        let want_h = horizontal_dir(intent);
        // 意图无左右（战斗站桩 / noop）：立刻清 latch，避免旧方向继续顶人乱走。
        // 攻击帧也要更新 latch，否则挥砍期间换向丢失，挥完又抖方向。
        if want_h == 0 {
            self.latched_h = 0;
            self.h_hold_until = tick;
            if !out.attack {
                out.left = false;
                out.right = false;
            }
        } else if self.latched_h == 0 || tick >= self.h_hold_until {
            self.latched_h = want_h;
            self.h_hold_until = tick.saturating_add(self.locomotion_hold_ticks);
            if !out.attack {
                out.left = self.latched_h < 0;
                out.right = self.latched_h > 0;
            }
        } else if !out.attack {
            out.left = self.latched_h < 0;
            out.right = self.latched_h > 0;
        }

        if !climbing && (intent.up || intent.down) {
            if tick >= self.climb_pulse_allowed_at {
                self.climb_pulse_allowed_at =
                    tick.saturating_add(self.locomotion_hold_ticks);
            } else {
                out.up = false;
                out.down = false;
                out.jump = false;
            }
        }

        out
    }

    /// 输出端硬闸：限制左右/上下/跳组合的切换频率（约 ≤4Hz）。
    /// 攻击/吃药走各自 refractory，不被此闸推迟；同组合连续按住不受影响。
    pub fn finalize_output(&mut self, mut out: InputFrame, tick: u32) -> InputFrame {
        let want_sig = control_sig(out);
        let prev_sig = control_sig(self.last_emit);
        if want_sig != prev_sig {
            let dt = tick.saturating_sub(self.last_control_change_tick);
            if dt < MIN_CONTROL_CHANGE_TICKS && tick > 0 {
                // 锁住上一拍的移动/跳/垂直；攻击与吃药仍用本帧结果。
                let attack = out.attack;
                let potion = out.use_potion;
                let pick = out.pick_up;
                out = self.last_emit;
                out.attack = attack;
                out.use_potion = potion;
                out.pick_up = pick;
            } else {
                self.last_control_change_tick = tick;
            }
        }
        self.last_emit = out;
        out
    }
}

/// 移动/垂直/跳签名（不含攻击与吃药）。
fn control_sig(f: InputFrame) -> u8 {
    let mut b = 0u8;
    if f.left {
        b |= 1;
    }
    if f.right {
        b |= 2;
    }
    if f.up {
        b |= 4;
    }
    if f.down {
        b |= 8;
    }
    if f.jump {
        b |= 16;
    }
    b
}

fn horizontal_dir(f: InputFrame) -> i8 {
    if f.right {
        1
    } else if f.left {
        -1
    } else {
        0
    }
}

/// 砍怪 / 吃药：不受随机休息影响；爬绳意图仍参与休息判定取消。
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
        assert_eq!(
            pace.idle_until, 0,
            "standing wait must not start explore rest"
        );
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

    #[test]
    fn direction_hold_latches_for_interval() {
        let mut pace = HumanPace::new(5);
        pace.locomotion_hold_ticks = 30;
        let intent = InputFrame {
            right: true,
            ..Default::default()
        };
        let base0 = pace.apply(intent, 0);
        let out0 = pace.apply_locomotion_hold(base0, 0, false, intent);
        assert!(out0.right);
        let intent_left = InputFrame {
            left: true,
            ..Default::default()
        };
        // hold 未满仍向右
        let base15 = pace.apply(intent_left, 15);
        let out15 = pace.apply_locomotion_hold(base15, 15, false, intent_left);
        assert!(out15.right && !out15.left);
        // 满 30 帧后才允许换向
        let base30 = pace.apply(intent_left, 30);
        let out30 = pace.apply_locomotion_hold(base30, 30, false, intent_left);
        assert!(out30.left && !out30.right);
    }

    #[test]
    fn climb_axis_throttled_on_ground_not_while_climbing() {
        let mut pace = HumanPace::new(6);
        pace.locomotion_hold_ticks = 30;
        let intent = InputFrame {
            up: true,
            jump: true,
            ..Default::default()
        };
        let base = pace.apply(intent, 0);
        let pulse = pace.apply_locomotion_hold(base, 0, false, intent);
        assert!(pulse.up);
        let base1 = pace.apply(intent, 1);
        let blocked = pace.apply_locomotion_hold(base1, 1, false, intent);
        assert!(!blocked.up && !blocked.jump);
        let base2 = pace.apply(intent, 1);
        let climbing = pace.apply_locomotion_hold(base2, 1, true, intent);
        assert!(climbing.up);
    }

    #[test]
    fn finalize_caps_control_changes_under_4hz() {
        let mut pace = HumanPace::new(7);
        let right = InputFrame {
            right: true,
            ..Default::default()
        };
        let left = InputFrame {
            left: true,
            ..Default::default()
        };
        let o0 = pace.finalize_output(right, 0);
        assert!(o0.right);
        // 立刻换向：硬闸应锁住右转
        let o1 = pace.finalize_output(left, 1);
        assert!(o1.right && !o1.left, "too-soon reverse must be held");
        // 未满间隔仍锁
        let o8 = pace.finalize_output(left, 8);
        assert!(o8.right && !o8.left);
        // 满 MIN_CONTROL_CHANGE_TICKS 后允许换向
        let o15 = pace.finalize_output(left, MIN_CONTROL_CHANGE_TICKS);
        assert!(o15.left && !o15.right);
    }

    #[test]
    fn finalize_allows_attack_during_locomotion_hold() {
        let mut pace = HumanPace::new(8);
        let walk = InputFrame {
            right: true,
            ..Default::default()
        };
        let _ = pace.finalize_output(walk, 0);
        let swing = InputFrame {
            left: true,
            attack: true,
            ..Default::default()
        };
        let out = pace.finalize_output(swing, 2);
        assert!(out.attack, "attack must pass");
        assert!(out.right && !out.left, "locomotion still held");
    }
}
