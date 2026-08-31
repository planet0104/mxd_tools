//! 游戏离散动作 → 输入帧。

use super::InputFrame;

/// 离散动作（HUD / 调试）。
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
