//! 探索记忆：维护 visit 网格与 NEAT 提示（与 fitness / rule_bot 同 80×120 网格）。
//!
//! 训练时用 sim 脚点更新网格（与计分对齐）；预览/部署用 OCR 锚定 + 里程计递推。
//! 不进网络 raw 坐标，只输出「哪边/哪层还没去过」与停滞压力。

use std::collections::HashSet;

use super::observation::{
    obs_climb_grab_ready, obs_floor_ahead_connected, obs_floor_drop_ahead, obs_has_ladder_or_rope_signal,
    obs_step_up_dx,
};
use super::types::{WINDOW_H, WINDOW_W};

/// 与 fitness 一致：80×120 访问网格。
pub const X_CELL_PX: f32 = 80.0;
pub const ALTITUDE_BAND_PX: f32 = 120.0;

pub fn visit_key(x: f32, y: f32) -> (i32, i32) {
    (
        (x / X_CELL_PX).floor() as i32,
        (y / ALTITUDE_BAND_PX).floor() as i32,
    )
}
/// 连续这么多决策帧无新格 → SeekVertical（与 rule_bot NO_NEW_CELL_DECISIONS 对齐）。
pub const NO_NEW_CELL_DECISIONS: u32 = 18;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExploreHints {
    pub unvisited_left: f32,
    pub unvisited_right: f32,
    pub unvisited_band_up: f32,
    pub unvisited_band_down: f32,
    pub stall_pressure: f32,
}

#[derive(Debug, Clone, Default)]
pub struct ExploreMemory {
    world_x: f32,
    world_y: f32,
    has_position: bool,
    visited: HashSet<(i32, i32)>,
    visited_bands: HashSet<i32>,
    ticks_without_new_cell: u32,
    current_band: i32,
    ticks_on_band: u32,
}

impl ExploreMemory {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn ticks_without_new_cell(&self) -> u32 {
        self.ticks_without_new_cell
    }

    pub fn visited_cells(&self) -> usize {
        self.visited.len()
    }

    pub fn visited_bands(&self) -> usize {
        self.visited_bands.len()
    }

    pub fn seek_vertical(&self) -> bool {
        self.ticks_without_new_cell >= NO_NEW_CELL_DECISIONS
    }

    /// 每个视觉/决策帧调用一次。
    ///
    /// `world_truth`：训练时传入 sim 脚点，与 fitness 网格对齐；部署时为 None，走里程计。
    pub fn tick(
        &mut self,
        obs: &[f32],
        delta_px: Option<(f32, f32)>,
        ocr_ok: bool,
        world_truth: Option<(f32, f32)>,
    ) -> ExploreHints {
        if let Some((x, y)) = world_truth {
            self.has_position = true;
            self.world_x = x;
            self.world_y = y;
        } else if ocr_ok && !self.has_position {
            self.has_position = true;
            self.world_x = 0.0;
            self.world_y = 0.0;
        } else if self.has_position {
            if let Some((dx, dy)) = delta_px {
                self.world_x += dx;
                self.world_y += dy;
            }
        }

        if self.has_position {
            if !self.try_mark_current() {
                self.ticks_without_new_cell = self.ticks_without_new_cell.saturating_add(1);
            }
        }
        self.compute_hints(obs)
    }

    fn try_mark_current(&mut self) -> bool {
        let (cx, cy) = visit_key(self.world_x, self.world_y);
        if cy != self.current_band {
            self.current_band = cy;
            self.ticks_on_band = 0;
        } else {
            self.ticks_on_band = self.ticks_on_band.saturating_add(1);
        }
        if self.visited.insert((cx, cy)) {
            self.ticks_without_new_cell = 0;
            self.visited_bands.insert(cy);
            true
        } else {
            false
        }
    }

    fn compute_hints(&self, obs: &[f32]) -> ExploreHints {
        if !self.has_position {
            return ExploreHints::default();
        }
        let (cx, cy) = visit_key(self.world_x, self.world_y);
        let left_key = (cx - 1, cy);
        let right_key = (cx + 1, cy);

        let can_left = obs_floor_ahead_connected(obs, -1.0);
        let can_right = obs_floor_ahead_connected(obs, 1.0);
        let unvisited_left = can_left && !self.visited.contains(&left_key);
        let unvisited_right = can_right && !self.visited.contains(&right_key);

        let band_up = cy + 1;
        let band_down = cy - 1;
        let can_go_up = obs_climb_grab_ready(obs)
            || obs_has_ladder_or_rope_signal(obs)
            || obs_step_up_dx(obs, WINDOW_W, WINDOW_H).is_some();
        let can_go_down = obs_floor_drop_ahead(obs, -1.0) || obs_floor_drop_ahead(obs, 1.0);
        let unvisited_band_up = can_go_up && !self.visited_bands.contains(&band_up);
        let unvisited_band_down = can_go_down && !self.visited_bands.contains(&band_down);

        let stall_pressure = (self.ticks_without_new_cell as f32 / NO_NEW_CELL_DECISIONS as f32)
            .clamp(0.0, 1.0);

        ExploreHints {
            unvisited_left: flag(unvisited_left),
            unvisited_right: flag(unvisited_right),
            unvisited_band_up: flag(unvisited_band_up),
            unvisited_band_down: flag(unvisited_band_down),
            stall_pressure,
        }
    }
}

fn flag(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::observation::{OBS_DIM, OBS_FLOOR_START, OBS_SLOT_DIM};

    fn floor_under(obs: &mut [f32; OBS_DIM]) {
        obs[OBS_FLOOR_START] = 0.0;
        obs[OBS_FLOOR_START + 1] = 0.02;
        obs[OBS_FLOOR_START + 2] = 0.3;
        obs[OBS_FLOOR_START + 3] = 0.05;
    }

    fn floor_ahead(obs: &mut [f32; OBS_DIM], dir: f32) {
        let slot = if dir > 0.0 { 1 } else { 0 };
        let base = OBS_FLOOR_START + slot * OBS_SLOT_DIM;
        obs[base] = dir * 0.15;
        obs[base + 1] = 0.02;
        obs[base + 2] = 0.25;
        obs[base + 3] = 0.05;
    }

    #[test]
    fn world_truth_seeds_and_marks_cell() {
        let mut mem = ExploreMemory::default();
        let obs = [0.0_f32; OBS_DIM];
        mem.tick(&obs, None, false, Some((100.0, 1000.0)));
        assert_eq!(mem.visited_cells(), 1);
    }

    #[test]
    fn horizontal_delta_marks_new_cell_and_clears_stall() {
        let mut mem = ExploreMemory::default();
        let obs = [0.0_f32; OBS_DIM];
        mem.tick(&obs, None, true, None);
        for _ in 0..5 {
            mem.tick(&obs, None, true, None);
        }
        assert!(mem.ticks_without_new_cell() >= 5);
        mem.tick(&obs, Some((X_CELL_PX, 0.0)), true, None);
        assert_eq!(mem.visited_cells(), 2);
        assert_eq!(mem.ticks_without_new_cell(), 0);
    }

    #[test]
    fn unvisited_right_when_walkable_and_never_been() {
        let mut mem = ExploreMemory::default();
        let mut obs = [0.0_f32; OBS_DIM];
        floor_under(&mut obs);
        floor_ahead(&mut obs, 1.0);
        mem.tick(&obs, None, true, None);
        let h = mem.tick(&obs, None, true, None);
        assert_eq!(h.unvisited_right, 1.0);
    }

    #[test]
    fn stall_pressure_saturates_at_no_new_cell_decisions() {
        let mut mem = ExploreMemory::default();
        let obs = [0.0_f32; OBS_DIM];
        mem.tick(&obs, None, true, None);
        for _ in 0..NO_NEW_CELL_DECISIONS {
            mem.tick(&obs, None, true, None);
        }
        let h = mem.tick(&obs, None, true, None);
        assert!(h.stall_pressure >= 1.0);
        assert!(mem.seek_vertical());
    }
}
