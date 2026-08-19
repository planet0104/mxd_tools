//! 通过 Win32 SendInput 向当前前台窗口注入键盘事件（非虚拟 HID 设备）。

use std::mem::size_of;
use std::thread;
use std::time::Duration;

use windows::Win32::UI::Input::KeyboardAndMouse::{
    MapVirtualKeyW, SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, MAPVK_VK_TO_VSC, VIRTUAL_KEY, VK_A, VK_C, VK_CONTROL,
    VK_DOWN, VK_ESCAPE, VK_LEFT, VK_RETURN, VK_RIGHT, VK_S, VK_SHIFT, VK_SPACE, VK_TAB, VK_UP, VK_W,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    A,
    W,
    S,
    D,
    C,
    Space,
    Enter,
    Esc,
    Tab,
    Left,
    Right,
    Up,
    Down,
}

impl Key {
    pub fn label(self) -> &'static str {
        match self {
            Key::A => "A",
            Key::W => "W",
            Key::S => "S",
            Key::D => "D",
            Key::C => "C",
            Key::Space => "Space",
            Key::Enter => "Enter",
            Key::Esc => "Esc",
            Key::Tab => "Tab",
            Key::Left => "←左",
            Key::Right => "→右",
            Key::Up => "↑上",
            Key::Down => "↓下",
        }
    }

    fn vk(self) -> VIRTUAL_KEY {
        match self {
            Key::A => VK_A,
            Key::W => VK_W,
            Key::S => VK_S,
            Key::D => VIRTUAL_KEY(0x44),
            Key::C => VK_C,
            Key::Space => VK_SPACE,
            Key::Enter => VK_RETURN,
            Key::Esc => VK_ESCAPE,
            Key::Tab => VK_TAB,
            Key::Left => VK_LEFT,
            Key::Right => VK_RIGHT,
            Key::Up => VK_UP,
            Key::Down => VK_DOWN,
        }
    }
    fn is_extended(self) -> bool {
        matches!(self, Key::Left | Key::Right | Key::Up | Key::Down)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modifier {
    LeftCtrl,
    LeftShift,
}

impl Modifier {
    fn vk(self) -> VIRTUAL_KEY {
        match self {
            Modifier::LeftCtrl => VK_CONTROL,
            Modifier::LeftShift => VK_SHIFT,
        }
    }
}

fn key_input(vk: VIRTUAL_KEY, up: bool, extended: bool) -> INPUT {
    let mut flags = if up {
        KEYEVENTF_KEYUP
    } else {
        KEYBD_EVENT_FLAGS(0)
    };
    if extended {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    let scan = unsafe { MapVirtualKeyW(vk.0 as u32, MAPVK_VK_TO_VSC) } as u16;
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: vk,
                wScan: scan,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

fn send(inputs: &[INPUT]) -> Result<(), String> {
    let sent = unsafe { SendInput(inputs, size_of::<INPUT>() as i32) };
    if sent as usize == inputs.len() {
        Ok(())
    } else {
        Err(format!("SendInput 只发送了 {sent}/{}", inputs.len()))
    }
}

pub fn tap(modifiers: &[Modifier], key: Key) -> Result<(), String> {
    let mut down = Vec::new();
    for m in modifiers {
        down.push(key_input(m.vk(), false, false));
    }
    down.push(key_input(key.vk(), false, key.is_extended()));
    send(&down)?;
    thread::sleep(Duration::from_millis(30));

    let mut up = Vec::new();
    up.push(key_input(key.vk(), true, key.is_extended()));
    for m in modifiers.iter().rev() {
        up.push(key_input(m.vk(), true, false));
    }
    send(&up)
}

pub fn tap_keys(modifiers: &[Modifier], keys: &[Key]) -> Result<(), String> {
    let mut down = Vec::new();
    for m in modifiers {
        down.push(key_input(m.vk(), false, false));
    }
    for k in keys {
        down.push(key_input(k.vk(), false, k.is_extended()));
    }
    send(&down)?;
    thread::sleep(Duration::from_millis(30));

    let mut up = Vec::new();
    for k in keys.iter().rev() {
        up.push(key_input(k.vk(), true, k.is_extended()));
    }
    for m in modifiers.iter().rev() {
        up.push(key_input(m.vk(), true, false));
    }
    send(&up)
}
