//! NEAT 离散动作 → 游戏输入帧。

use super::InputFrame;

/// NEAT 网络输出的离散动作（每逻辑帧选一个，9 选 1 argmax）。
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

/// 多输出 NEAT（每位一个 bool）合并为动作；冲突时按优先级取第一个。
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

#[cfg(test)]
mod tests {
    use super::*;

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
