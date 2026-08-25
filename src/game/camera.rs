use crate::game::types::{WINDOW_W, WORLD_VIEW_H};

/// 世界相机：玩家尽量位于游戏区中央；贴地图边缘时相机停住、玩家可继续走向窗口边。
#[derive(Debug, Clone, Copy)]
pub struct WorldCamera {
    pub cam_x: f32,
    pub cam_y: f32,
}

impl WorldCamera {
    pub fn new() -> Self {
        Self {
            cam_x: 0.0,
            cam_y: 0.0,
        }
    }

    pub fn follow(&mut self, map_w: f32, map_h: f32, player_x: f32, player_y: f32) {
        let ideal_x = player_x - WINDOW_W * 0.5;
        let ideal_y = player_y - WORLD_VIEW_H * 0.5;
        let max_x = (map_w - WINDOW_W).max(0.0);
        let max_y = (map_h - WORLD_VIEW_H).max(0.0);
        self.cam_x = ideal_x.clamp(0.0, max_x);
        self.cam_y = ideal_y.clamp(0.0, max_y);
    }

    /// 玩家在游戏世界层上的屏幕 x（居中时 = WINDOW_W / 2）。
    pub fn player_screen_x(&self, player_x: f32) -> f32 {
        player_x - self.cam_x
    }

    /// 玩家在游戏世界层上的屏幕 y（居中时 = WORLD_VIEW_H / 2）。
    pub fn player_screen_y(&self, player_y: f32) -> f32 {
        player_y - self.cam_y
    }
}

impl Default for WorldCamera {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_when_away_from_edges() {
        let mut cam = WorldCamera::new();
        cam.follow(2040.0, 1483.0, 800.0, 900.0);
        assert!((cam.player_screen_x(800.0) - WINDOW_W * 0.5).abs() < 0.01);
        assert!((cam.player_screen_y(900.0) - WORLD_VIEW_H * 0.5).abs() < 0.01);
    }

    #[test]
    fn left_edge_player_not_centered() {
        let mut cam = WorldCamera::new();
        cam.follow(2040.0, 1483.0, 120.0, 900.0);
        assert_eq!(cam.cam_x, 0.0);
        assert!((cam.player_screen_x(120.0) - 120.0).abs() < 0.01);
    }

    #[test]
    fn right_edge_player_not_centered() {
        let mut cam = WorldCamera::new();
        let px = 1900.0;
        cam.follow(2040.0, 1483.0, px, 900.0);
        assert_eq!(cam.cam_x, 2040.0 - WINDOW_W);
        assert!((cam.player_screen_x(px) - (px - cam.cam_x)).abs() < 0.01);
    }
}
