//! 纯视觉闭环：YOLO + SelfTracker 里程计、朝向与攀爬粘性（bot 决策唯一位置来源）。

use super::input::InputFrame;
use super::observation::{obs_floor_underfoot, obs_vertical_nav_allowed, OBS_DIM};
use super::visual_progress::{LocationNode, VisualMotionEstimator};

/// 粘性感知：推算坐标、朝向、攀爬状态。
#[derive(Debug, Clone)]
pub struct VisionSenseState {
    pub facing: f32,
    pub climbing: bool,
    pub est_x: f32,
    pub est_y: f32,
    initialized: bool,
    climb_ground_release: u32,
    motion: VisualMotionEstimator,
}

impl Default for VisionSenseState {
    fn default() -> Self {
        Self {
            facing: 1.0,
            climbing: false,
            est_x: 0.0,
            est_y: 0.0,
            initialized: false,
            climb_ground_release: 0,
            motion: VisualMotionEstimator::default(),
        }
    }
}

impl VisionSenseState {
    pub fn prepare(&mut self, obs: &[f32; OBS_DIM]) {
        self.motion.update(obs);
        self.est_x = self.motion.x;
        self.est_y = self.motion.y;
        self.initialized = true;

        let near_climb = obs_vertical_nav_allowed(obs, false);
        let under = obs_floor_underfoot(obs);
        if self.climbing {
            // 绳底仍有脚下地板 + 近绳信号：不能当落地，否则永远粘不上攀爬态。
            let can_land = under && !near_climb;
            if can_land {
                self.climb_ground_release = self.climb_ground_release.saturating_add(1);
                if self.climb_ground_release >= 2 {
                    self.climbing = false;
                    self.climb_ground_release = 0;
                }
            } else {
                self.climb_ground_release = 0;
            }
        } else if near_climb && !under {
            self.climbing = true;
            self.climb_ground_release = 0;
        }
    }

    pub fn after_decide(&mut self, out: &InputFrame, obs: &[f32; OBS_DIM]) {
        if out.right {
            self.facing = 1.0;
        } else if out.left {
            self.facing = -1.0;
        }
        let near = obs_vertical_nav_allowed(obs, false);
        let under = obs_floor_underfoot(obs);
        if (out.up || out.down || (out.jump && out.up)) && near && !under {
            self.climbing = true;
            self.climb_ground_release = 0;
        }
    }

    pub fn note_effective(&mut self, effective: &InputFrame) {
        if effective.right {
            self.facing = 1.0;
        } else if effective.left {
            self.facing = -1.0;
        }
    }

    pub fn visual_delta(&self) -> (f32, f32) {
        (self.motion.dx, self.motion.dy)
    }

    pub fn visual_confidence(&self) -> u8 {
        self.motion.confidence
    }

    pub fn location_node(&self) -> LocationNode {
        self.motion.node()
    }

    pub fn anchor_at(&mut self, x: f32, y: f32) {
        self.motion.x = x;
        self.motion.y = y;
        self.est_x = x;
        self.est_y = y;
        self.initialized = true;
    }
}
