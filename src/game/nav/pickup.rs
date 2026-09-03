use super::super::input::InputFrame;
use super::super::observation::{
    obs_drop_in_pickup_range, obs_drop_is_meso, obs_has_drop, obs_has_enemy, OBS_DROP_SLOTS,
    OBS_DROP_START, OBS_SLOT_DIM,
};
use super::super::types::{WINDOW_H, WINDOW_W};
use super::executor::NavCtx;
use super::map_graph::MapGraph;
use super::types::{DROP_MEMORY_TICKS, PickupState, SAME_PLATFORM_DY_PX, SubGoal};

const CLIMB_ALIGN_OBS: f32 = 0.015;
/// 非金币掉落：只近距离短追，捡不上就放弃，不挡寻路。
const ITEM_CHASE_DX: f32 = 72.0;
const ITEM_CHASE_MAX_TICKS: u32 = 18;

pub struct PickupController {
    pub state: PickupState,
}

impl Default for PickupController {
    fn default() -> Self {
        Self {
            state: PickupState::default(),
        }
    }
}

impl PickupController {
    pub fn reset(&mut self) {
        self.state = PickupState::default();
    }

    pub fn tick_memory(&mut self, obs: &[f32]) {
        if obs_has_drop(obs) {
            self.state.drop_memory = DROP_MEMORY_TICKS;
        } else if self.state.drop_memory > 0 {
            self.state.drop_memory -= 1;
        }
    }

    pub fn try_pickup(
        &mut self,
        ctx: &NavCtx<'_>,
        graph: &MapGraph,
        config: &super::types::NavBotConfig,
        near_only: bool,
        active_goal: SubGoal,
        farm_band_mobs: bool,
    ) -> Option<InputFrame> {
        if matches!(
            active_goal,
            SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. }
        ) && ctx.climbing
        {
            return None;
        }

        let near = obs_drop_in_pickup_range(ctx.obs);
        if near_only {
            if !near {
                return None;
            }
            let mut out = InputFrame::default();
            out.pick_up = true;
            self.state.chase_ticks = 0;
            return Some(out);
        }

        if !(obs_has_drop(ctx.obs) || self.state.drop_memory > 0) {
            self.state.chase_ticks = 0;
            return None;
        }

        if near {
            let mut out = InputFrame::default();
            out.pick_up = true;
            self.state.chase_ticks = 0;
            return Some(out);
        }

        let has_meso = slot_has_meso(ctx.obs);
        // 有怪时仍追金币；其他掉落物不追，避免打断清图。
        if farm_band_mobs && config.pickup_near_only_on_farm && !has_meso {
            return None;
        }
        if obs_has_enemy(ctx.obs) && !has_meso && !near {
            return None;
        }

        if !drop_on_same_platform(ctx.obs) {
            return None;
        }

        let (max_dx, max_ticks) = if has_meso {
            (config.pickup_chase_dx, config.pickup_chase_max_ticks)
        } else {
            (ITEM_CHASE_DX, ITEM_CHASE_MAX_TICKS)
        };

        if !chase_in_platform_bounds(ctx, graph, max_dx, has_meso) {
            self.state.chase_ticks = 0;
            return None;
        }

        self.state.chase_ticks = self.state.chase_ticks.saturating_add(1);
        if self.state.chase_ticks > max_ticks {
            // 超时放弃，不改 navigator；下一帧可继续寻路。
            self.state.chase_ticks = 0;
            return None;
        }

        let mut out = InputFrame::default();
        out.pick_up = true;
        steer_toward_drop(ctx.obs, &mut out, ctx.facing, has_meso);
        Some(out)
    }
}

fn drop_on_same_platform(obs: &[f32]) -> bool {
    let max_dy = SAME_PLATFORM_DY_PX / WINDOW_H;
    for i in 0..OBS_DROP_SLOTS {
        let base = OBS_DROP_START + i * OBS_SLOT_DIM;
        if base + 4 > obs.len() {
            break;
        }
        let dy = obs[base + 1];
        let w = obs[base + 2];
        let h = obs[base + 3];
        if w.abs() <= 1e-4 && h.abs() <= 1e-4 {
            continue;
        }
        if dy.abs() <= max_dy {
            return true;
        }
    }
    false
}

fn slot_has_meso(obs: &[f32]) -> bool {
    for i in 0..OBS_DROP_SLOTS {
        let base = OBS_DROP_START + i * OBS_SLOT_DIM;
        if base + 4 > obs.len() {
            break;
        }
        let w = obs[base + 2];
        let h = obs[base + 3];
        if w.abs() <= 1e-4 && h.abs() <= 1e-4 {
            continue;
        }
        if obs_drop_is_meso(w) {
            return true;
        }
    }
    false
}

fn chase_in_platform_bounds(
    ctx: &NavCtx<'_>,
    graph: &MapGraph,
    max_dx: f32,
    meso_only: bool,
) -> bool {
    let Some((dx, _)) = nearest_drop(ctx.obs, meso_only) else {
        return false;
    };
    if dx.abs() > max_dx / WINDOW_W {
        return false;
    }
    let Some(node) = graph.get(ctx.node_id) else {
        return true;
    };
    let target_x = ctx.world_x + dx * WINDOW_W;
    target_x >= node.x_min - 8.0 && target_x <= node.x_max + 8.0
}

/// `meso_only`：只看金币；否则任意掉落。
fn nearest_drop(obs: &[f32], meso_only: bool) -> Option<(f32, f32)> {
    let mut best: Option<(f32, f32, f32)> = None;
    for i in 0..OBS_DROP_SLOTS {
        let base = OBS_DROP_START + i * OBS_SLOT_DIM;
        if base + 4 > obs.len() {
            break;
        }
        let dx = obs[base];
        let dy = obs[base + 1];
        let w = obs[base + 2];
        let h = obs[base + 3];
        if w.abs() <= 1e-4 && h.abs() <= 1e-4 {
            continue;
        }
        if meso_only && !obs_drop_is_meso(w) {
            continue;
        }
        let dist = dx.abs() + dy.abs() * 0.3;
        match best {
            None => best = Some((dx, dy, dist)),
            Some((_, _, bd)) if dist < bd => best = Some((dx, dy, dist)),
            _ => {}
        }
    }
    best.map(|(dx, dy, _)| (dx, dy))
}

fn steer_toward_drop(obs: &[f32], out: &mut InputFrame, facing: f32, meso_only: bool) {
    let Some((dx, _)) = nearest_drop(obs, meso_only).or_else(|| nearest_drop(obs, false)) else {
        if facing >= 0.0 {
            out.right = true;
        } else {
            out.left = true;
        }
        return;
    };
    if dx.abs() <= CLIMB_ALIGN_OBS {
        if facing >= 0.0 {
            out.right = true;
        } else {
            out.left = true;
        }
    } else if dx > 0.0 {
        out.right = true;
    } else {
        out.left = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::observation::OBS_DIM;

    fn empty_ctx(obs: &[f32]) -> NavCtx<'_> {
        NavCtx {
            obs,
            facing: 1.0,
            on_ground: true,
            climbing: false,
            world_x: 500.0,
            world_y: 1200.0,
            node_id: 99999,
            walk_right_ok: Some(true),
            walk_left_ok: Some(true),
            drop_ahead_right: None,
            drop_ahead_left: None,
            climb: None,
            step_up_dx: None,
            farm_band_mobs: false,
            pending_target: None,
        }
    }

    #[test]
    fn near_only_requires_range() {
        let obs = [0.0_f32; OBS_DIM];
        let mut pickup = PickupController::default();
        let ctx = empty_ctx(&obs);
        let map = crate::game::load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let cfg = super::super::types::NavBotConfig::default();
        assert!(pickup
            .try_pickup(
                &ctx,
                &graph,
                &cfg,
                true,
                SubGoal::Patrol { dir: 1.0 },
                false
            )
            .is_none());
    }

    #[test]
    fn prefers_meso_over_farther_potion() {
        let mut obs = [0.0_f32; OBS_DIM];
        // 药水更近
        obs[OBS_DROP_START] = 40.0 / WINDOW_W;
        obs[OBS_DROP_START + 1] = 0.0;
        obs[OBS_DROP_START + 2] = -0.04;
        obs[OBS_DROP_START + 3] = 0.04;
        // 金币稍远
        obs[OBS_DROP_START + OBS_SLOT_DIM] = 90.0 / WINDOW_W;
        obs[OBS_DROP_START + OBS_SLOT_DIM + 1] = 0.0;
        obs[OBS_DROP_START + OBS_SLOT_DIM + 2] = 0.03;
        obs[OBS_DROP_START + OBS_SLOT_DIM + 3] = 0.03;
        let (dx, _) = nearest_drop(&obs, true).expect("meso");
        assert!((dx - 90.0 / WINDOW_W).abs() < 1e-5);
    }

    #[test]
    fn farm_mobs_still_chase_meso() {
        let mut obs = [0.0_f32; OBS_DIM];
        obs[OBS_DROP_START] = 80.0 / WINDOW_W;
        obs[OBS_DROP_START + 1] = 0.0;
        obs[OBS_DROP_START + 2] = 0.03;
        obs[OBS_DROP_START + 3] = 0.03;
        let mut pickup = PickupController::default();
        let ctx = empty_ctx(&obs);
        let map = crate::game::load_default_map().expect("map");
        let graph = MapGraph::build(&map);
        let cfg = super::super::types::NavBotConfig::default();
        let out = pickup.try_pickup(
            &ctx,
            &graph,
            &cfg,
            false,
            SubGoal::Patrol { dir: 1.0 },
            true,
        );
        assert!(out.is_some());
        assert!(out.unwrap().pick_up);
    }
}
