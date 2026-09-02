//! NEAT 寻路动作宏：把「走 / 跳台 / 攀爬」封成多 tick 意图，网络每次只选一个。
//!
//! 完成与失败一律由 YOLO 槽位 + 地标里程计位移 + tick 超时判定，不读任何 sim 物理状态；
//! 训练与真机部署共用同一套判定。
//!
//! 视觉有延迟：按键期间收到的观测多半是按键前拍的。所以按键结束后再等一个感知帧（settle）
//! 才结算位移，否则「刚走完就判没动」会把正常走路误判成撞墙。
//!
//! 卡死防护在这一层用代码兜底，不留给进化去试：
//! - 前置条件可见不成立的动作直接屏蔽（没绳梯不让选爬；无 step_up 不让朝该方向跳）；
//! - 死胡同高崖不可走过去；可落缘仍允许走（会掉下去，不需要跳）；
//! - 同一动作连续失败两次进入冷却，几次决策内不可再选，argmax 自然落到别的动作上；
//! - 走路失败即视为该方向被挡，作为 blocked 标志喂回网络。

use super::input::InputFrame;
use super::map::ClimbDir;
use super::observation::{
    obs_climb_hint, obs_floor_drop_ahead, obs_jump_target_ahead, obs_platform_edge,
    OBS_PROPRIO_START,
};
use super::types::{WINDOW_H, WINDOW_W};

pub const MACRO_ACTION_COUNT: usize = 5;

/// 走一段：60Hz 下约 25px，足够短以保证网络每 ~0.3s 重新决策。
const WALK_TICKS: u32 = 12;
/// 跳跃全程约 0.555s≈33 tick，留余量到落地。
const JUMP_TICKS: u32 = 40;
const JUMP_PRESS_TICKS: u32 = 4;
const CLIMB_TICKS: u32 = 90;
/// 按键结束后等待结算的感知帧数；感知长时间不到则按 tick 超时结算。
/// detect_hz=10 时感知间隔约 6 tick；至少等 2 帧再结算位移。
const SETTLE_FRAMES: u8 = 2;
const SETTLE_TIMEOUT_TICKS: u32 = 30;
/// 里程计垂直位移达到此值即视为换层成功。
const CLIMB_DONE_DY_PX: f32 = 40.0;
/// 抓不上绳/爬不动：连续感知帧几乎无垂直位移即失败。
const CLIMB_STALL_FRAMES: u32 = 4;
const CLIMB_FRAME_DY_PX: f32 = 3.0;
/// 走/跳判定「真的动了」的位移门槛。
const WALK_DONE_DX_PX: f32 = 8.0;
const JUMP_DONE_MOVE_PX: f32 = 20.0;
/// 同一动作连续失败到此次数进入冷却；冷却按决策次数计。
const FAIL_STREAK_TO_MASK: u8 = 2;
const MASK_DECISIONS: u8 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroAction {
    WalkLeft,
    WalkRight,
    JumpLeft,
    JumpRight,
    Climb,
}

impl MacroAction {
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => MacroAction::WalkLeft,
            1 => MacroAction::WalkRight,
            2 => MacroAction::JumpLeft,
            3 => MacroAction::JumpRight,
            _ => MacroAction::Climb,
        }
    }

    pub fn index(self) -> usize {
        match self {
            MacroAction::WalkLeft => 0,
            MacroAction::WalkRight => 1,
            MacroAction::JumpLeft => 2,
            MacroAction::JumpRight => 3,
            MacroAction::Climb => 4,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            MacroAction::WalkLeft => "walk_left",
            MacroAction::WalkRight => "walk_right",
            MacroAction::JumpLeft => "jump_left",
            MacroAction::JumpRight => "jump_right",
            MacroAction::Climb => "climb",
        }
    }

    fn budget(self) -> u32 {
        match self {
            MacroAction::WalkLeft | MacroAction::WalkRight => WALK_TICKS,
            MacroAction::JumpLeft | MacroAction::JumpRight => JUMP_TICKS,
            MacroAction::Climb => CLIMB_TICKS,
        }
    }

    /// 跳/爬中途打断会摔落或半挂绳上，必须跑完；走路可随时被战斗接管。
    pub fn interruptible(self) -> bool {
        matches!(self, MacroAction::WalkLeft | MacroAction::WalkRight)
    }
}

#[derive(Debug)]
struct Active {
    action: MacroAction,
    ticks: u32,
    climb_down: bool,
    net_dx: f32,
    net_dy: f32,
    stall_frames: u32,
    keys_done: bool,
    settle_left: u8,
    settle_ticks: u32,
    done: Option<bool>,
}

impl Active {
    fn judge(&self) -> bool {
        match self.action {
            MacroAction::WalkLeft | MacroAction::WalkRight => self.net_dx.abs() >= WALK_DONE_DX_PX,
            MacroAction::JumpLeft | MacroAction::JumpRight => {
                self.net_dx.abs() >= JUMP_DONE_MOVE_PX || self.net_dy.abs() >= JUMP_DONE_MOVE_PX
            }
            MacroAction::Climb => self.net_dy.abs() >= CLIMB_DONE_DY_PX,
        }
    }
}

#[derive(Debug, Default)]
pub struct MacroRunner {
    active: Option<Active>,
    last_action: Option<MacroAction>,
    last_failed: bool,
    fail_streak: [u8; MACRO_ACTION_COUNT],
    cooldown: [u8; MACRO_ACTION_COUNT],
}

impl MacroRunner {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// 宏执行完毕（成功或失败）才允许网络重新决策。
    pub fn is_idle(&self) -> bool {
        self.active.as_ref().map_or(true, |a| a.done.is_some())
    }

    /// 当前宏可否被战斗接管（空闲或正在走路）。
    pub fn interruptible(&self) -> bool {
        self.active
            .as_ref()
            .map_or(true, |a| a.done.is_some() || a.action.interruptible())
    }

    pub fn cancel(&mut self) {
        self.active = None;
    }

    pub fn last_action(&self) -> Option<MacroAction> {
        self.last_action
    }

    pub fn last_failed(&self) -> bool {
        self.last_failed
    }

    /// 走路失败即视为该方向被挡，直到同方向再次走成功。
    pub fn blocked_left(&self) -> bool {
        self.fail_streak[MacroAction::WalkLeft.index()] >= 1
    }

    pub fn blocked_right(&self) -> bool {
        self.fail_streak[MacroAction::WalkRight.index()] >= 1
    }

    /// 本次决策可选的动作：屏蔽前置条件不成立、撞墙方向、平台边缘无效走/跳与冷却中的动作。
    /// `seek_vertical`：局部无新格时放宽落缘跳/爬绳（与 rule_bot SeekVertical 对齐）。
    pub fn allowed(&self, obs: &[f32], seek_vertical: bool) -> [bool; MACRO_ACTION_COUNT] {
        let mut ok = [true; MACRO_ACTION_COUNT];
        for (i, c) in self.cooldown.iter().enumerate() {
            if *c > 0 {
                ok[i] = false;
            }
        }
        if obs_climb_hint(obs, WINDOW_W, WINDOW_H).is_none() {
            ok[MacroAction::Climb.index()] = false;
        }
        // 死胡同高崖：前方无同层地板且下方也接不住 → 不可走过去。
        // 可落缘（drop）仍允许走，角色会掉下去，不需要跳。
        if obs_platform_edge(obs, 1.0) && !obs_floor_drop_ahead(obs, 1.0) {
            ok[MacroAction::WalkRight.index()] = false;
        }
        if obs_platform_edge(obs, -1.0) && !obs_floor_drop_ahead(obs, -1.0) {
            ok[MacroAction::WalkLeft.index()] = false;
        }
        // 跳只允许 YOLO 可见的紧邻 step_up（≤80px）；太高/纯边沿 → 改爬绳或换向。
        if !obs_jump_target_ahead(obs, 1.0, WINDOW_W, WINDOW_H) {
            ok[MacroAction::JumpRight.index()] = false;
        }
        if !obs_jump_target_ahead(obs, -1.0, WINDOW_W, WINDOW_H) {
            ok[MacroAction::JumpLeft.index()] = false;
        }
        if seek_vertical {
            for (dir, jump) in [(-1.0, MacroAction::JumpLeft), (1.0, MacroAction::JumpRight)] {
                if obs_platform_edge(obs, dir) || obs_floor_drop_ahead(obs, dir) {
                    ok[jump.index()] = true;
                }
            }
        }
        ok
    }

    pub fn begin(&mut self, action: MacroAction, obs: &[f32]) {
        for c in self.cooldown.iter_mut() {
            *c = c.saturating_sub(1);
        }
        let mut climb_down = false;
        let mut done = None;
        if action == MacroAction::Climb {
            match obs_climb_hint(obs, WINDOW_W, WINDOW_H) {
                Some(hint) => climb_down = matches!(hint.dir, ClimbDir::Down),
                // 掩码兜不住的情况（全部被屏蔽时回退到原始 argmax）：当场失败。
                None => done = Some(false),
            }
        }
        self.last_action = Some(action);
        self.active = Some(Active {
            action,
            ticks: 0,
            climb_down,
            net_dx: 0.0,
            net_dy: 0.0,
            stall_frames: 0,
            keys_done: false,
            settle_left: SETTLE_FRAMES,
            settle_ticks: 0,
            done: None,
        });
        if let Some(ok) = done {
            self.finish(ok);
        }
    }

    /// 每个 sim tick 调用：输出本 tick 按键并推进宏。
    pub fn next_frame(&mut self) -> InputFrame {
        let mut frame = InputFrame {
            pick_up: true,
            ..Default::default()
        };
        let mut finished = None;
        {
            let Some(a) = self.active.as_mut() else {
                return frame;
            };
            if a.done.is_some() {
                return frame;
            }
            if a.keys_done {
                a.settle_ticks += 1;
                if a.settle_ticks >= SETTLE_TIMEOUT_TICKS {
                    finished = Some(a.judge());
                }
            } else {
                match a.action {
                    MacroAction::WalkLeft => frame.left = true,
                    MacroAction::WalkRight => frame.right = true,
                    MacroAction::JumpLeft | MacroAction::JumpRight => {
                        if a.action == MacroAction::JumpLeft {
                            frame.left = true;
                        } else {
                            frame.right = true;
                        }
                        if a.ticks < JUMP_PRESS_TICKS {
                            frame.jump = true;
                        }
                    }
                    MacroAction::Climb => {
                        if a.climb_down {
                            frame.down = true;
                        } else {
                            frame.up = true;
                        }
                    }
                }
                a.ticks += 1;
                if a.ticks >= a.action.budget() {
                    a.keys_done = true;
                }
            }
        }
        if let Some(ok) = finished {
            self.finish(ok);
        }
        frame
    }

    /// 每个感知帧调用：累计里程计位移，必要时提前判定完成/失败。
    pub fn observe(&mut self, obs: &[f32]) {
        let mut finished = None;
        {
            let Some(a) = self.active.as_mut() else {
                return;
            };
            if a.done.is_some() {
                return;
            }
            let dx = obs.get(OBS_PROPRIO_START).copied().unwrap_or(0.0) * WINDOW_W;
            let dy = obs.get(OBS_PROPRIO_START + 1).copied().unwrap_or(0.0) * WINDOW_H;
            a.net_dx += dx;
            a.net_dy += dy;
            if a.action == MacroAction::Climb {
                if a.net_dy.abs() >= CLIMB_DONE_DY_PX {
                    finished = Some(true);
                } else {
                    if dy.abs() < CLIMB_FRAME_DY_PX {
                        a.stall_frames += 1;
                    } else {
                        a.stall_frames = 0;
                    }
                    if a.stall_frames >= CLIMB_STALL_FRAMES {
                        finished = Some(false);
                    }
                }
            }
            if finished.is_none() && a.action.interruptible() {
                let cliff = match a.action {
                    MacroAction::WalkRight => {
                        obs_platform_edge(obs, 1.0) && !obs_floor_drop_ahead(obs, 1.0)
                    }
                    MacroAction::WalkLeft => {
                        obs_platform_edge(obs, -1.0) && !obs_floor_drop_ahead(obs, -1.0)
                    }
                    _ => false,
                };
                if cliff && a.ticks >= 2 {
                    finished = Some(false);
                }
            }
            if finished.is_none() && a.keys_done {
                a.settle_left = a.settle_left.saturating_sub(1);
                if a.settle_left == 0 {
                    finished = Some(a.judge());
                }
            }
        }
        if let Some(ok) = finished {
            self.finish(ok);
        }
    }

    fn finish(&mut self, ok: bool) {
        let Some(a) = self.active.as_mut() else {
            return;
        };
        a.done = Some(ok);
        let idx = a.action.index();
        self.last_failed = !ok;
        if ok {
            self.fail_streak[idx] = 0;
        } else {
            self.fail_streak[idx] = self.fail_streak[idx].saturating_add(1);
            if self.fail_streak[idx] >= FAIL_STREAK_TO_MASK {
                self.cooldown[idx] = MASK_DECISIONS;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::observation::{OBS_DIM, OBS_FLOOR_START, OBS_ROPE_START, OBS_SLOT_DIM};

    /// 脚点正上方一段可上爬的绳：框底略高于脚点，`obs_climb_hint` 才会给出 Up。
    fn rope_above(obs: &mut [f32; OBS_DIM]) {
        obs[OBS_ROPE_START] = 0.0;
        obs[OBS_ROPE_START + 1] = -60.0 / WINDOW_H;
        obs[OBS_ROPE_START + 2] = 0.01;
        obs[OBS_ROPE_START + 3] = 80.0 / WINDOW_H;
    }

    fn delta(dx_px: f32, dy_px: f32) -> [f32; OBS_DIM] {
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_PROPRIO_START] = dx_px / WINDOW_W;
        obs[OBS_PROPRIO_START + 1] = dy_px / WINDOW_H;
        obs
    }

    /// 模拟视觉延迟：按键期间收到的帧还没动，settle 帧才带出位移。
    fn run_walk(r: &mut MacroRunner, action: MacroAction, moved_px: f32) {
        let obs = [0.0_f32; OBS_DIM];
        r.begin(action, &obs);
        for t in 0..WALK_TICKS {
            r.next_frame();
            if t == 5 {
                r.observe(&delta(0.0, 0.0));
            }
        }
        assert!(!r.is_idle(), "按键结束还要等 settle 帧");
        r.observe(&delta(0.0, 0.0));
        r.observe(&delta(moved_px, 0.0));
        assert!(r.is_idle());
    }

    #[test]
    fn climb_masked_without_rope_signal() {
        let obs = [0.0_f32; OBS_DIM];
        let r = MacroRunner::default();
        let ok = r.allowed(&obs, false);
        assert!(!ok[MacroAction::Climb.index()]);
        assert!(ok[MacroAction::WalkLeft.index()]);
    }

    #[test]
    fn climb_allowed_with_rope_and_completes_on_progress() {
        let mut obs = [0.0_f32; OBS_DIM];
        rope_above(&mut obs);
        let mut r = MacroRunner::default();
        assert!(r.allowed(&obs, false)[MacroAction::Climb.index()]);
        r.begin(MacroAction::Climb, &obs);
        assert!(r.next_frame().up);
        r.observe(&delta(0.0, -(CLIMB_DONE_DY_PX + 1.0)));
        assert!(r.is_idle());
        assert!(!r.last_failed());
    }

    #[test]
    fn climb_stall_marks_failure() {
        let mut obs = [0.0_f32; OBS_DIM];
        rope_above(&mut obs);
        let mut r = MacroRunner::default();
        r.begin(MacroAction::Climb, &obs);
        for _ in 0..CLIMB_STALL_FRAMES {
            r.observe(&delta(0.0, 0.0));
        }
        assert!(r.is_idle());
        assert!(r.last_failed());
    }

    #[test]
    fn walk_without_progress_fails_and_marks_blocked() {
        let mut r = MacroRunner::default();
        run_walk(&mut r, MacroAction::WalkRight, 0.0);
        assert!(r.last_failed());
        assert!(r.blocked_right());
        assert!(!r.blocked_left());
    }

    #[test]
    fn delayed_frame_still_counts_as_progress() {
        let mut r = MacroRunner::default();
        run_walk(&mut r, MacroAction::WalkRight, 12.0);
        assert!(!r.last_failed());
        assert!(!r.blocked_right());
    }

    #[test]
    fn successful_walk_clears_blocked() {
        let mut r = MacroRunner::default();
        run_walk(&mut r, MacroAction::WalkRight, 0.0);
        assert!(r.blocked_right());
        run_walk(&mut r, MacroAction::WalkRight, 25.0);
        assert!(!r.blocked_right());
    }

    #[test]
    fn settle_times_out_without_vision() {
        let obs = [0.0_f32; OBS_DIM];
        let mut r = MacroRunner::default();
        r.begin(MacroAction::WalkLeft, &obs);
        for _ in 0..WALK_TICKS + SETTLE_TIMEOUT_TICKS {
            r.next_frame();
        }
        assert!(r.is_idle());
        assert!(r.last_failed());
    }

    #[test]
    fn repeated_failure_masks_action_for_a_few_decisions() {
        let obs = [0.0_f32; OBS_DIM];
        let mut r = MacroRunner::default();
        run_walk(&mut r, MacroAction::WalkRight, 0.0);
        assert!(r.allowed(&obs, false)[MacroAction::WalkRight.index()], "失败一次还不屏蔽");
        run_walk(&mut r, MacroAction::WalkRight, 0.0);
        assert!(!r.allowed(&obs, false)[MacroAction::WalkRight.index()], "连续两次失败进入冷却");
        for _ in 0..MASK_DECISIONS {
            run_walk(&mut r, MacroAction::WalkLeft, 25.0);
        }
        assert!(r.allowed(&obs, false)[MacroAction::WalkRight.index()]);
    }

    #[test]
    fn jump_presses_jump_only_at_start_then_holds_direction() {
        let obs = [0.0_f32; OBS_DIM];
        let mut r = MacroRunner::default();
        r.begin(MacroAction::JumpRight, &obs);
        for _ in 0..JUMP_PRESS_TICKS {
            let f = r.next_frame();
            assert!(f.jump && f.right);
        }
        let f = r.next_frame();
        assert!(!f.jump, "起跳后不再按跳，避免落地立刻再跳");
        assert!(f.right, "空中保持方向以跨过间隙");
        assert!(!r.interruptible(), "跳跃中不可被战斗打断");
    }

    #[test]
    fn jump_into_wall_fails() {
        let obs = [0.0_f32; OBS_DIM];
        let mut r = MacroRunner::default();
        r.begin(MacroAction::JumpLeft, &obs);
        r.observe(&delta(0.0, -30.0));
        r.observe(&delta(0.0, 30.0));
        for _ in 0..JUMP_TICKS {
            r.next_frame();
        }
        r.observe(&delta(0.0, 0.0));
        r.observe(&delta(0.0, 0.0));
        assert!(r.is_idle());
        assert!(r.last_failed());
    }

    #[test]
    fn jump_that_moved_is_success() {
        let obs = [0.0_f32; OBS_DIM];
        let mut r = MacroRunner::default();
        r.begin(MacroAction::JumpLeft, &obs);
        for _ in 0..JUMP_TICKS {
            r.next_frame();
        }
        r.observe(&delta(0.0, 0.0));
        r.observe(&delta(-(JUMP_DONE_MOVE_PX + 5.0), 0.0));
        assert!(r.is_idle());
        assert!(!r.last_failed());
    }

    #[test]
    fn walk_is_interruptible_and_cancel_idles_runner() {
        let obs = [0.0_f32; OBS_DIM];
        let mut r = MacroRunner::default();
        r.begin(MacroAction::WalkLeft, &obs);
        assert!(r.interruptible());
        r.cancel();
        assert!(r.is_idle());
        assert!(!r.next_frame().left);
    }

    fn right_platform_edge_obs() -> [f32; OBS_DIM] {
        let mut v = [0.0_f32; OBS_DIM];
        v[OBS_FLOOR_START] = -20.0 / WINDOW_W;
        v[OBS_FLOOR_START + 1] = 0.01;
        v[OBS_FLOOR_START + 2] = 40.0 / WINDOW_W;
        v[OBS_FLOOR_START + 3] = 0.02;
        v
    }

    fn step_up_right_obs() -> [f32; OBS_DIM] {
        let mut v = right_platform_edge_obs();
        let b = OBS_FLOOR_START + OBS_SLOT_DIM;
        v[b] = 0.10;
        v[b + 1] = -0.055;
        v[b + 2] = 0.12;
        v[b + 3] = 0.04;
        v
    }

    fn drop_ahead_right_obs() -> [f32; OBS_DIM] {
        let mut v = right_platform_edge_obs();
        let b = OBS_FLOOR_START + OBS_SLOT_DIM;
        v[b] = 0.12;
        v[b + 1] = 0.08;
        v[b + 2] = 0.25;
        v[b + 3] = 0.04;
        v
    }

    #[test]
    fn high_cliff_masks_walk_and_jump_but_allows_turn() {
        let obs = right_platform_edge_obs();
        let r = MacroRunner::default();
        let ok = r.allowed(&obs, false);
        assert!(!ok[MacroAction::WalkRight.index()]);
        assert!(!ok[MacroAction::JumpRight.index()]);
        assert!(ok[MacroAction::WalkLeft.index()]);
    }

    #[test]
    fn reachable_step_up_allows_jump_right() {
        let obs = step_up_right_obs();
        let r = MacroRunner::default();
        let ok = r.allowed(&obs, false);
        assert!(ok[MacroAction::JumpRight.index()]);
    }

    #[test]
    fn drop_edge_allows_walk_not_jump() {
        let obs = drop_ahead_right_obs();
        let r = MacroRunner::default();
        let ok = r.allowed(&obs, false);
        assert!(ok[MacroAction::WalkRight.index()]);
        assert!(!ok[MacroAction::JumpRight.index()]);
    }

    #[test]
    fn seek_vertical_allows_cliff_jump() {
        let obs = right_platform_edge_obs();
        let r = MacroRunner::default();
        let ok = r.allowed(&obs, true);
        assert!(ok[MacroAction::JumpRight.index()]);
    }

    #[test]
    fn cliff_with_rope_allows_climb() {
        let mut obs = right_platform_edge_obs();
        rope_above(&mut obs);
        let r = MacroRunner::default();
        let ok = r.allowed(&obs, false);
        assert!(ok[MacroAction::Climb.index()]);
        assert!(!ok[MacroAction::JumpRight.index()]);
    }

    #[test]
    fn walk_aborts_early_at_platform_edge() {
        let obs = right_platform_edge_obs();
        let mut r = MacroRunner::default();
        r.begin(MacroAction::WalkRight, &obs);
        r.next_frame();
        r.next_frame();
        r.observe(&obs);
        assert!(r.is_idle(), "到边应提前结束走路宏");
        assert!(r.last_failed());
    }
}
