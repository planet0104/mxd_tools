//! NEAT 离散动作 → 游戏输入帧。

use super::InputFrame;

/// NEAT 多比特输出顺序（每位 sigmoid ≥ 阈值视为按下，可同时为真）。
pub const NEAT_OUTPUT_BUTTONS: usize = 8;

/// NEAT 网络输出的离散动作（训练/预览 HUD 用；实际控制走多比特 `InputFrame`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Noop,
    Left,
    Right,
    Jump,
    Attack,
    PickUp,
    UsePotion,
    Up,
    Down,
}

impl Action {
    pub const ALL: [Action; 9] = [
        Action::Noop,
        Action::Left,
        Action::Right,
        Action::Jump,
        Action::Attack,
        Action::PickUp,
        Action::UsePotion,
        Action::Up,
        Action::Down,
    ];

    pub fn from_index(i: usize) -> Action {
        Self::ALL.get(i).copied().unwrap_or(Action::Noop)
    }

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|&a| a == self).unwrap_or(0)
    }
    pub fn label(self) -> &'static str {
        match self {
            Action::Noop => "noop",
            Action::Left => "left",
            Action::Right => "right",
            Action::Jump => "jump",
            Action::Attack => "attack",
            Action::PickUp => "pickup",
            Action::UsePotion => "potion",
            Action::Up => "up",
            Action::Down => "down",
        }
    }
}

/// 将离散动作映射为 `InputFrame`（按键为「按下」状态，持续一帧）。
pub fn action_to_input(action: Action) -> InputFrame {
    let mut f = InputFrame::default();
    match action {
        Action::Noop => {}
        Action::Left => f.left = true,
        Action::Right => f.right = true,
        Action::Jump => f.jump = true,
        Action::Attack => f.attack = true,
        Action::PickUp => f.pick_up = true,
        Action::UsePotion => f.use_potion = true,
        Action::Up => f.up = true,
        Action::Down => f.down = true,
    }
    f
}

/// 多输出 NEAT（每位一个 bool）合并为输入；冲突时 left+right 在 `horizontal()` 中抵消。
pub fn actions_from_bits(bits: &[bool]) -> InputFrame {
    let mut f = InputFrame::default();
    if bits.first().copied().unwrap_or(false) {
        f.left = true;
    }
    if bits.get(1).copied().unwrap_or(false) {
        f.right = true;
    }
    if bits.get(2).copied().unwrap_or(false) {
        f.jump = true;
    }
    if bits.get(3).copied().unwrap_or(false) {
        f.attack = true;
    }
    if bits.get(4).copied().unwrap_or(false) {
        f.pick_up = true;
    }
    if bits.get(5).copied().unwrap_or(false) {
        f.use_potion = true;
    }
    if bits.get(6).copied().unwrap_or(false) {
        f.up = true;
    }
    if bits.get(7).copied().unwrap_or(false) {
        f.down = true;
    }
    f
}

/// 将 `InputFrame` 格式化为 HUD 标签（如 `left+jump`）。
pub fn input_label(input: &InputFrame) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if input.left {
        parts.push("left");
    }
    if input.right {
        parts.push("right");
    }
    if input.jump {
        parts.push("jump");
    }
    if input.attack {
        parts.push("attack");
    }
    if input.pick_up {
        parts.push("pickup");
    }
    if input.use_potion {
        parts.push("potion");
    }
    if input.up {
        parts.push("up");
    }
    if input.down {
        parts.push("down");
    }
    if parts.is_empty() {
        "noop".to_string()
    } else {
        parts.join("+")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combo_bits_map_to_input() {
        let f = actions_from_bits(&[true, false, true, false, false, false, false, false]);
        assert!(f.left);
        assert!(f.jump);
        assert!(!f.right);
    }

    #[test]
    fn input_label_combo() {
        let mut f = InputFrame::default();
        f.left = true;
        f.jump = true;
        assert_eq!(input_label(&f), "left+jump");
    }

    #[test]
    fn action_maps_to_input() {
        let f = action_to_input(Action::Jump);
        assert!(f.jump);
        assert!(!f.left);
    }

    #[test]
    fn roundtrip_index() {
        for a in Action::ALL {
            assert_eq!(Action::from_index(a.index()), a);
        }
    }
}
