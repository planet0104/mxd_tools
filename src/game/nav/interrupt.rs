use super::super::input::InputFrame;
use super::super::observation::obs_enemy_in_attack_range_platform;
use super::combat_fsm::CombatFsm;
use super::types::{merge_frames, SubGoal};

pub struct InterruptArbiter;

impl InterruptArbiter {
    /// P0 脚边捡 > P1 近战 > P2 追币 > P3 导航。
    /// 换层时战斗只叠 attack；战斗中仍允许脚边 pick_up。
    /// `force_transit`：底层最右去爬绳的接近走位，禁止战斗把人继续往右拉。
    pub fn merge(
        navigate: InputFrame,
        pickup_near: Option<InputFrame>,
        pickup_chase: Option<InputFrame>,
        combat: InputFrame,
        combat_active: bool,
        goal: SubGoal,
        force_transit: bool,
    ) -> InputFrame {
        let transit = goal.is_transit() || force_transit;

        if let Some(p) = pickup_near {
            if !transit && !combat_active {
                let mut out = navigate;
                if p.pick_up {
                    out.pick_up = true;
                }
                if p.left {
                    out.left = true;
                    out.right = false;
                }
                if p.right {
                    out.right = true;
                    out.left = false;
                }
                return out;
            }
            let mut base = if combat_active {
                Self::combat_frame(navigate, combat, goal, transit)
            } else {
                navigate
            };
            if p.pick_up {
                base.pick_up = true;
            }
            return base;
        }

        if combat_active {
            return Self::combat_frame(navigate, combat, goal, transit);
        }

        if let Some(p) = pickup_chase {
            if transit {
                return navigate;
            }
            return merge_frames(navigate, p);
        }

        navigate
    }

    fn combat_frame(
        navigate: InputFrame,
        combat: InputFrame,
        goal: SubGoal,
        transit: bool,
    ) -> InputFrame {
        let has_combat_move = combat.left || combat.right || combat.attack;
        let climbing = matches!(
            goal,
            SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. }
        );
        if climbing {
            let mut out = navigate;
            if combat.attack {
                out.attack = true;
            }
            return out;
        }
        if transit {
            if combat.attack {
                let mut out = combat;
                out.left = false;
                out.right = false;
                out.jump = false;
                out.up = false;
                out.down = false;
                out.pick_up = false;
                return out;
            }
            return navigate;
        }
        if !has_combat_move {
            let mut out = navigate;
            out.left = false;
            out.right = false;
            return out;
        }
        let mut out = combat;
        out.pick_up = false;
        out.jump = false;
        out
    }

    /// 可砍带站砍；换层时只补 attack，保留移动/跳跃。
    pub fn refresh_melee_hold(
        obs: &[f32],
        facing: f32,
        navigate: InputFrame,
        goal: SubGoal,
    ) -> InputFrame {
        if !obs_enemy_in_attack_range_platform(obs, facing) {
            return navigate;
        }
        let mut out = navigate;
        out.attack = true;
        if !goal.is_transit() {
            out.left = false;
            out.right = false;
            out.jump = false;
        }
        out
    }
}

pub struct CombatAdapter {
    pub fsm: CombatFsm,
}

impl Default for CombatAdapter {
    fn default() -> Self {
        Self {
            fsm: CombatFsm::default(),
        }
    }
}

impl CombatAdapter {
    pub fn reset(&mut self) {
        self.fsm.reset();
    }

    pub fn observe(&mut self, obs: &[f32]) {
        self.fsm.observe(obs);
    }

    pub fn observe_strike_only(&mut self, obs: &[f32]) {
        self.fsm.observe_strike_only(obs);
    }

    pub fn intent_frame(&mut self) -> InputFrame {
        let mut f = self.fsm.next_frame();
        f.pick_up = false;
        f
    }

    pub fn is_active(&self) -> bool {
        self.fsm.is_active()
    }
}
