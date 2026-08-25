#[derive(Debug, Clone, Copy, Default)]
pub struct InputFrame {
    pub left: bool,
    pub right: bool,
    pub jump: bool,
    pub attack: bool,
    pub up: bool,
    pub down: bool,
    pub pick_up: bool,
    pub use_potion: bool,
    pub open_inventory: bool,
    pub inventory_click: Option<(u32, u32)>,
    pub restart: bool,
}

impl InputFrame {
    pub fn horizontal(&self) -> f32 {
        let mut v = 0.0;
        if self.left {
            v -= 1.0;
        }
        if self.right {
            v += 1.0;
        }
        v
    }
}
