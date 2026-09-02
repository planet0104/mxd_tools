//! 砍怪状态机：纯 YOLO 槽位驱动，代码主动激活，本台怪清完自动交还寻路。
//!
//! 逻辑取自 rule_bot::try_combat 的核心分支（接触必砍 / 进距站砍 / 中距接近 / 悬崖不追），
//! 去掉了与 explore_mode、perch 耦合的部分。不读任何 sim 状态，朝向由自己下过的方向键推出。

use super::input::InputFrame;
use super::observation::{
    obs_assess_enemy_contact, obs_nearest_same_level_enemy_px, obs_platform_edge,
    ENEMY_NEAR_PLATFORM_DX, ENEMY_PLATFORM_DY,
};
use super::types::{WINDOW_H, WINDOW_W};

/// 进距即砍（与挥砍前伸 ~90px 对齐）。
const STRIKE_DX_PX: f32 = 90.0;
/// 连续多少个感知帧无本台怪才退出（≈48 tick，与清层判定一致）。
const CLEAR_FRAMES: u32 = 8;
/// 一刀的节奏：攻击冷却 0.35s≈21 tick，按键只给前 3 tick。
const SWING_PERIOD_TICKS: u32 = 21;
const SWING_PRESS_TICKS: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Step {
    /// 站砍：出刀 tick 同时按朝向键，保证每一刀都朝着怪。
    Strike(f32),
    /// 中距接近。
    Approach(f32),
    /// 站着不动（怪在悬崖对面 / 暂时看不到）。
    Hold,
}

#[derive(Debug)]
pub struct CombatFsm {
    active: bool,
    step: Step,
    facing: f32,
    clear_frames: u32,
    swing_ticks: u32,
}

impl Default for CombatFsm {
    fn default() -> Self {
        Self {
            active: false,
            step: Step::Hold,
            facing: 1.0,
            clear_frames: 0,
            swing_ticks: 0,
        }
    }
}

impl CombatFsm {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// 每个感知帧调用：判定激活/退出并选定本帧动作。
    pub fn observe(&mut self, obs: &[f32]) {
        let contact = obs_assess_enemy_contact(obs);
        let target = obs_nearest_same_level_enemy_px(obs, WINDOW_W, WINDOW_H)
            .filter(|(_, dy)| dy.abs() <= ENEMY_PLATFORM_DY * WINDOW_H);
        let engage_dx = ENEMY_NEAR_PLATFORM_DX * WINDOW_W;
        let threat = contact.total > 0 || target.map_or(false, |(dx, _)| dx.abs() <= engage_dx);

        if threat {
            if !self.active {
                self.swing_ticks = 0;
            }
            self.active = true;
            self.clear_frames = 0;
        } else {
            self.clear_frames = self.clear_frames.saturating_add(1);
            if self.clear_frames >= CLEAR_FRAMES {
                self.active = false;
            }
        }
        if !self.active {
            self.step = Step::Hold;
            return;
        }

        self.step = if contact.total > 0 {
            let toward = if contact.right > contact.left {
                1.0
            } else if contact.left > contact.right {
                -1.0
            } else {
                self.facing
            };
            Step::Strike(toward)
        } else if let Some((dx, _)) = target {
            let toward = if dx >= 0.0 { 1.0 } else { -1.0 };
            if dx.abs() <= STRIKE_DX_PX {
                Step::Strike(toward)
            } else if dx.abs() <= engage_dx && !obs_platform_edge(obs, toward) {
                Step::Approach(toward)
            } else {
                Step::Hold
            }
        } else {
            Step::Hold
        };
    }

    /// 每个 sim tick 调用；仅在 `is_active()` 时使用其输出。
    pub fn next_frame(&mut self) -> InputFrame {
        let mut f = InputFrame {
            pick_up: true,
            ..Default::default()
        };
        match self.step {
            Step::Strike(toward) => {
                if self.swing_ticks % SWING_PERIOD_TICKS < SWING_PRESS_TICKS {
                    f.attack = true;
                    set_dir(&mut f, toward);
                    self.facing = toward;
                }
                self.swing_ticks += 1;
            }
            Step::Approach(toward) => {
                set_dir(&mut f, toward);
                self.facing = toward;
                self.swing_ticks = 0;
            }
            Step::Hold => self.swing_ticks = 0,
        }
        f
    }
}

fn set_dir(f: &mut InputFrame, dir: f32) {
    if dir > 0.0 {
        f.right = true;
    } else {
        f.left = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::observation::{OBS_DIM, OBS_ENEMY_START, OBS_FLOOR_START, OBS_SLOT_DIM};

    fn floor_under(obs: &mut [f32; OBS_DIM]) {
        obs[OBS_FLOOR_START] = 0.0;
        obs[OBS_FLOOR_START + 1] = 0.02;
        obs[OBS_FLOOR_START + 2] = 0.6;
        obs[OBS_FLOOR_START + 3] = 0.05;
    }

    /// 地板只延伸到脚点右侧 4px：右前方无地板 → 悬崖。
    fn narrow_floor(obs: &mut [f32; OBS_DIM]) {
        obs[OBS_FLOOR_START] = -30.0 / WINDOW_W;
        obs[OBS_FLOOR_START + 1] = 0.02;
        obs[OBS_FLOOR_START + 2] = 68.0 / WINDOW_W;
        obs[OBS_FLOOR_START + 3] = 0.05;
    }

    fn enemy(obs: &mut [f32; OBS_DIM], slot: usize, dx_px: f32, dy_px: f32) {
        let b = OBS_ENEMY_START + slot * OBS_SLOT_DIM;
        obs[b] = dx_px / WINDOW_W;
        obs[b + 1] = dy_px / WINDOW_H;
        obs[b + 2] = 40.0 / WINDOW_W;
        obs[b + 3] = 40.0 / WINDOW_H;
    }

    #[test]
    fn inactive_without_enemies() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_under(&mut obs);
        let mut fsm = CombatFsm::default();
        fsm.observe(&obs);
        assert!(!fsm.is_active());
        let f = fsm.next_frame();
        assert!(!f.attack && !f.left && !f.right);
    }

    #[test]
    fn strikes_toward_close_enemy_on_the_left() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_under(&mut obs);
        enemy(&mut obs, 0, -60.0, 0.0);
        let mut fsm = CombatFsm::default();
        fsm.observe(&obs);
        assert!(fsm.is_active());
        let f = fsm.next_frame();
        assert!(f.attack);
        assert!(f.left && !f.right, "出刀 tick 必须同时按朝向键");
    }

    #[test]
    fn swing_cadence_presses_then_releases() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_under(&mut obs);
        enemy(&mut obs, 0, 50.0, 0.0);
        let mut fsm = CombatFsm::default();
        fsm.observe(&obs);
        for _ in 0..SWING_PRESS_TICKS {
            assert!(fsm.next_frame().attack);
        }
        for _ in SWING_PRESS_TICKS..SWING_PERIOD_TICKS {
            let f = fsm.next_frame();
            assert!(!f.attack && !f.right, "冷却期间站着不动");
        }
        assert!(fsm.next_frame().attack, "冷却结束再出一刀");
    }

    #[test]
    fn approaches_mid_range_enemy() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_under(&mut obs);
        enemy(&mut obs, 0, 150.0, 0.0);
        let mut fsm = CombatFsm::default();
        fsm.observe(&obs);
        assert!(fsm.is_active());
        let f = fsm.next_frame();
        assert!(f.right && !f.attack);
    }

    #[test]
    fn does_not_approach_across_cliff() {
        let mut obs = [0.0_f32; OBS_DIM];
        narrow_floor(&mut obs);
        enemy(&mut obs, 0, 150.0, 0.0);
        let mut fsm = CombatFsm::default();
        fsm.observe(&obs);
        assert!(fsm.is_active());
        let f = fsm.next_frame();
        assert!(!f.right && !f.left && !f.attack, "怪在悬崖对面：站住不追");
    }

    #[test]
    fn ignores_enemy_on_other_platform() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_under(&mut obs);
        enemy(&mut obs, 0, 60.0, -120.0);
        let mut fsm = CombatFsm::default();
        fsm.observe(&obs);
        assert!(!fsm.is_active());
    }

    #[test]
    fn contact_from_behind_turns_to_that_side() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_under(&mut obs);
        // 怪在身后左侧、接触盒重叠。
        enemy(&mut obs, 0, -20.0, 0.0);
        let mut fsm = CombatFsm::default();
        fsm.facing = 1.0;
        fsm.observe(&obs);
        let f = fsm.next_frame();
        assert!(f.attack && f.left);
    }

    #[test]
    fn deactivates_after_clear_frames() {
        let mut obs = [0.0_f32; OBS_DIM];
        floor_under(&mut obs);
        enemy(&mut obs, 0, 50.0, 0.0);
        let mut fsm = CombatFsm::default();
        fsm.observe(&obs);
        assert!(fsm.is_active());
        let mut empty = [0.0_f32; OBS_DIM];
        floor_under(&mut empty);
        for _ in 0..CLEAR_FRAMES - 1 {
            fsm.observe(&empty);
            assert!(fsm.is_active(), "YOLO 闪断不应立刻退出");
        }
        fsm.observe(&empty);
        assert!(!fsm.is_active());
    }
}
