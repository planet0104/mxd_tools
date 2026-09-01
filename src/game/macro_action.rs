//! NEAT 动作宏：把「走 / 砍 / 跳台 / 攀爬」封成多 tick 意图，网络每次只选一个。
//!
//! 完成与失败一律由 YOLO 槽位 + OCR 脚点位移 + tick 超时判定，不读任何 sim 物理状态；
//! 训练与真机部署共用同一套判定，宏在真机上失败时行为一致。

use super::input::InputFrame;
use super::map::ClimbDir;
use super::observation::{
    obs_climb_hint, obs_nearest_same_level_enemy_px, ENEMY_PLATFORM_DY, OBS_PROPRIO_START,
};
use super::types::{WINDOW_H, WINDOW_W};

pub const MACRO_ACTION_COUNT: usize = 6;

/// 走一段：60Hz 下约 25px，足够短以保证网络每 ~0.2s 重新决策。
const WALK_TICKS: u32 = 12;
/// 一次挥砍：攻击冷却 0.35s≈21 tick，按键只给前几 tick，边沿触发恰好一刀。
const ATTACK_TICKS: u32 = 21;
const ATTACK_PRESS_TICKS: u32 = 3;
/// 跳跃全程约 0.555s≈33 tick，留余量到落地。
const JUMP_TICKS: u32 = 40;
const JUMP_PRESS_TICKS: u32 = 4;
const CLIMB_TICKS: u32 = 90;
/// OCR 垂直位移达到此值即视为换层成功。
const CLIMB_DONE_DY_PX: f32 = 40.0;
/// 抓不上绳/爬不动：连续感知帧几乎无垂直位移即失败。
const CLIMB_STALL_FRAMES: u32 = 4;
const CLIMB_FRAME_DY_PX: f32 = 3.0;
/// 走/跳判定「真的动了」的 OCR 位移门槛。
const WALK_DONE_DX_PX: f32 = 8.0;
const JUMP_DONE_MOVE_PX: f32 = 20.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroAction {
    WalkLeft,
    WalkRight,
    Attack,
    JumpLeft,
    JumpRight,
    Climb,
}

impl MacroAction {
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => MacroAction::WalkLeft,
            1 => MacroAction::WalkRight,
            2 => MacroAction::Attack,
            3 => MacroAction::JumpLeft,
            4 => MacroAction::JumpRight,
            _ => MacroAction::Climb,
        }
    }

    pub fn index(self) -> usize {
        match self {
            MacroAction::WalkLeft => 0,
            MacroAction::WalkRight => 1,
            MacroAction::Attack => 2,
            MacroAction::JumpLeft => 3,
            MacroAction::JumpRight => 4,
            MacroAction::Climb => 5,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            MacroAction::WalkLeft => "walk_left",
            MacroAction::WalkRight => "walk_right",
            MacroAction::Attack => "attack",
            MacroAction::JumpLeft => "jump_left",
            MacroAction::JumpRight => "jump_right",
            MacroAction::Climb => "climb",
        }
    }

    fn budget(self) -> u32 {
        match self {
            MacroAction::WalkLeft | MacroAction::WalkRight => WALK_TICKS,
            MacroAction::Attack => ATTACK_TICKS,
            MacroAction::JumpLeft | MacroAction::JumpRight => JUMP_TICKS,
            MacroAction::Climb => CLIMB_TICKS,
        }
    }
}

#[derive(Debug)]
struct Active {
    action: MacroAction,
    ticks: u32,
    /// 挥砍朝向：来自 YOLO 最近本台怪；0 表示保持当前朝向。
    face: f32,
    climb_down: bool,
    net_dx: f32,
    net_dy: f32,
    stall_frames: u32,
    done: Option<bool>,
}

#[derive(Debug, Default)]
pub struct MacroRunner {
    active: Option<Active>,
    last_action: Option<MacroAction>,
    last_failed: bool,
}

impl MacroRunner {
    pub fn reset(&mut self) {
        self.active = None;
        self.last_action = None;
        self.last_failed = false;
    }

    /// 宏执行完毕（成功或失败）才允许网络重新决策。
    pub fn is_idle(&self) -> bool {
        self.active
            .as_ref()
            .map_or(true, |a| a.done.is_some())
    }

    pub fn last_action(&self) -> Option<MacroAction> {
        self.last_action
    }

    pub fn last_failed(&self) -> bool {
        self.last_failed
    }

    pub fn begin(&mut self, action: MacroAction, obs: &[f32]) {
        let mut face = 0.0_f32;
        let mut climb_down = false;
        let mut done = None;
        match action {
            MacroAction::Attack => face = nearest_platform_enemy_dir(obs),
            MacroAction::Climb => match obs_climb_hint(obs, WINDOW_W, WINDOW_H) {
                Some(hint) => climb_down = matches!(hint.dir, ClimbDir::Down),
                // 视野里没有可攀目标：动作当场失败，网络下一帧能看到 last_failed。
                None => done = Some(false),
            },
            _ => {}
        }
        self.last_action = Some(action);
        self.last_failed = done == Some(false);
        self.active = Some(Active {
            action,
            ticks: 0,
            face,
            climb_down,
            net_dx: 0.0,
            net_dy: 0.0,
            stall_frames: 0,
            done,
        });
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
            match a.action {
                MacroAction::WalkLeft => frame.left = true,
                MacroAction::WalkRight => frame.right = true,
                MacroAction::Attack => {
                    if a.ticks < ATTACK_PRESS_TICKS {
                        frame.attack = true;
                        if a.face > 0.0 {
                            frame.right = true;
                        } else if a.face < 0.0 {
                            frame.left = true;
                        }
                    }
                }
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
                finished = Some(match a.action {
                    MacroAction::WalkLeft | MacroAction::WalkRight => {
                        a.net_dx.abs() >= WALK_DONE_DX_PX
                    }
                    MacroAction::Attack => true,
                    MacroAction::JumpLeft | MacroAction::JumpRight => {
                        a.net_dx.abs() >= JUMP_DONE_MOVE_PX || a.net_dy.abs() >= JUMP_DONE_MOVE_PX
                    }
                    MacroAction::Climb => a.net_dy.abs() >= CLIMB_DONE_DY_PX,
                });
            }
        }
        if let Some(ok) = finished {
            self.finish(ok);
        }
        frame
    }

    /// 每个感知帧调用：累计 OCR 位移，必要时提前判定完成/失败。
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
            match a.action {
                MacroAction::Climb => {
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
                MacroAction::WalkLeft => {
                    if blocked_left(obs) {
                        finished = Some(false);
                    }
                }
                MacroAction::WalkRight => {
                    if blocked_right(obs) {
                        finished = Some(false);
                    }
                }
                _ => {}
            }
        }
        if let Some(ok) = finished {
            self.finish(ok);
        }
    }

    fn finish(&mut self, ok: bool) {
        if let Some(a) = self.active.as_mut() {
            a.done = Some(ok);
        }
        self.last_failed = !ok;
    }
}

fn blocked_left(obs: &[f32]) -> bool {
    obs.get(OBS_PROPRIO_START + 2).copied().unwrap_or(0.0) >= 0.5
}

fn blocked_right(obs: &[f32]) -> bool {
    obs.get(OBS_PROPRIO_START + 3).copied().unwrap_or(0.0) >= 0.5
}

/// 最近「本台」怪的水平方向；无本台怪时返回 0（保持当前朝向挥砍）。
fn nearest_platform_enemy_dir(obs: &[f32]) -> f32 {
    let Some((dx, dy)) = obs_nearest_same_level_enemy_px(obs, WINDOW_W, WINDOW_H) else {
        return 0.0;
    };
    if dy.abs() > ENEMY_PLATFORM_DY * WINDOW_H {
        return 0.0;
    }
    if dx.abs() < 1.0 {
        return 0.0;
    }
    dx.signum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::observation::{OBS_DIM, OBS_ENEMY_START, OBS_ROPE_START};

    fn enemy_at(obs: &mut [f32; OBS_DIM], dx_px: f32, dy_px: f32) {
        obs[OBS_ENEMY_START] = dx_px / WINDOW_W;
        obs[OBS_ENEMY_START + 1] = dy_px / WINDOW_H;
        obs[OBS_ENEMY_START + 2] = 0.05;
        obs[OBS_ENEMY_START + 3] = 0.05;
    }

    /// 脚点正上方一段可上爬的绳：框底略高于脚点，`obs_climb_hint` 才会给出 Up。
    fn rope_above(obs: &mut [f32; OBS_DIM]) {
        obs[OBS_ROPE_START] = 0.0;
        obs[OBS_ROPE_START + 1] = -60.0 / WINDOW_H;
        obs[OBS_ROPE_START + 2] = 0.01;
        obs[OBS_ROPE_START + 3] = 80.0 / WINDOW_H;
    }

    fn ocr_delta(obs: &mut [f32; OBS_DIM], dx_px: f32, dy_px: f32) {
        obs[OBS_PROPRIO_START] = dx_px / WINDOW_W;
        obs[OBS_PROPRIO_START + 1] = dy_px / WINDOW_H;
    }

    #[test]
    fn attack_presses_direction_of_nearest_platform_enemy() {
        let mut obs = [0.0_f32; OBS_DIM];
        enemy_at(&mut obs, -40.0, 0.0);
        let mut r = MacroRunner::default();
        r.begin(MacroAction::Attack, &obs);
        let f = r.next_frame();
        assert!(f.attack);
        assert!(f.left, "怪在左侧应按左转向再出手");
        assert!(!f.right);
    }

    #[test]
    fn attack_releases_keys_after_press_window() {
        let mut obs = [0.0_f32; OBS_DIM];
        enemy_at(&mut obs, 40.0, 0.0);
        let mut r = MacroRunner::default();
        r.begin(MacroAction::Attack, &obs);
        for _ in 0..ATTACK_PRESS_TICKS {
            assert!(r.next_frame().attack);
        }
        let f = r.next_frame();
        assert!(!f.attack, "一次动作只出一刀");
        assert!(f.pick_up, "拾取常按，不占用网络输出");
    }

    #[test]
    fn attack_always_succeeds_and_ends_after_cooldown() {
        let obs = [0.0_f32; OBS_DIM];
        let mut r = MacroRunner::default();
        r.begin(MacroAction::Attack, &obs);
        for _ in 0..ATTACK_TICKS {
            assert!(!r.is_idle());
            r.next_frame();
        }
        assert!(r.is_idle());
        assert!(!r.last_failed());
    }

    #[test]
    fn climb_without_rope_signal_fails_immediately() {
        let obs = [0.0_f32; OBS_DIM];
        let mut r = MacroRunner::default();
        r.begin(MacroAction::Climb, &obs);
        assert!(r.is_idle());
        assert!(r.last_failed());
    }

    #[test]
    fn climb_completes_on_vertical_progress() {
        let mut obs = [0.0_f32; OBS_DIM];
        rope_above(&mut obs);
        let mut r = MacroRunner::default();
        r.begin(MacroAction::Climb, &obs);
        assert!(!r.is_idle());
        assert!(r.next_frame().up, "上爬应按 up");
        let mut moved = [0.0_f32; OBS_DIM];
        ocr_delta(&mut moved, 0.0, -(CLIMB_DONE_DY_PX + 1.0));
        r.observe(&moved);
        assert!(r.is_idle());
        assert!(!r.last_failed());
    }

    #[test]
    fn climb_stall_marks_failure() {
        let mut obs = [0.0_f32; OBS_DIM];
        rope_above(&mut obs);
        let mut r = MacroRunner::default();
        r.begin(MacroAction::Climb, &obs);
        let mut still = [0.0_f32; OBS_DIM];
        ocr_delta(&mut still, 0.0, 0.0);
        for _ in 0..CLIMB_STALL_FRAMES {
            r.observe(&still);
        }
        assert!(r.is_idle());
        assert!(r.last_failed(), "爬不动应判失败而不是耗满 90 tick");
    }

    #[test]
    fn walk_blocked_by_ocr_marks_failure() {
        let obs = [0.0_f32; OBS_DIM];
        let mut r = MacroRunner::default();
        r.begin(MacroAction::WalkLeft, &obs);
        r.next_frame();
        let mut blocked = [0.0_f32; OBS_DIM];
        blocked[OBS_PROPRIO_START + 2] = 1.0;
        r.observe(&blocked);
        assert!(r.is_idle());
        assert!(r.last_failed());
    }

    #[test]
    fn walk_without_progress_fails_at_timeout() {
        let obs = [0.0_f32; OBS_DIM];
        let mut r = MacroRunner::default();
        r.begin(MacroAction::WalkRight, &obs);
        for _ in 0..WALK_TICKS {
            let f = r.next_frame();
            assert!(f.right);
        }
        assert!(r.is_idle());
        assert!(r.last_failed());
    }

    #[test]
    fn jump_presses_jump_only_at_start_then_holds_direction() {
        let obs = [0.0_f32; OBS_DIM];
        let mut r = MacroRunner::default();
        r.begin(MacroAction::JumpRight, &obs);
        for _ in 0..JUMP_PRESS_TICKS {
            let f = r.next_frame();
            assert!(f.jump);
            assert!(f.right);
        }
        let f = r.next_frame();
        assert!(!f.jump, "起跳后不再按跳，避免落地立刻再跳");
        assert!(f.right, "空中保持方向以跨过间隙");
    }

    #[test]
    fn jump_that_moved_is_success() {
        let obs = [0.0_f32; OBS_DIM];
        let mut r = MacroRunner::default();
        r.begin(MacroAction::JumpLeft, &obs);
        let mut moved = [0.0_f32; OBS_DIM];
        ocr_delta(&mut moved, -(JUMP_DONE_MOVE_PX + 5.0), 0.0);
        r.observe(&moved);
        for _ in 0..JUMP_TICKS {
            r.next_frame();
        }
        assert!(r.is_idle());
        assert!(!r.last_failed());
    }
}
