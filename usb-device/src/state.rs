//! 键盘 / 鼠标 HID 状态缓存。
//!
//! 命令层先改这里的状态，再序列化为报告推给主机。

use usbd_hid::descriptor::{KeyboardReport, MouseReport};

/// Boot 协议键盘状态：1 字节修饰键 + 最多 6 个同时按下的键。
#[derive(Clone, Copy)]
pub struct KeyboardState {
    /// 修饰键位图：LCtrl=0x01, LShift=0x02, LAlt=0x04, LGui=0x08，右侧对应高 4 位。
    pub modifier: u8,
    /// 当前按下的 HID Usage ID，左侧紧凑排列，空位为 0。
    pub keycodes: [u8; 6],
}

impl KeyboardState {
    pub const fn new() -> Self {
        Self {
            modifier: 0,
            keycodes: [0; 6],
        }
    }

    /// 生成待发送的 HID 键盘输入报告。
    pub fn report(&self) -> KeyboardReport {
        KeyboardReport {
            modifier: self.modifier,
            reserved: 0,
            leds: 0,
            keycodes: self.keycodes,
        }
    }

    /// 是否仍有修饰键或普通键处于按下。
    pub fn is_held(&self) -> bool {
        self.modifier != 0 || self.keycodes.iter().any(|&c| c != 0)
    }

    /// 松开全部按键并清除修饰键。
    pub fn clear(&mut self) {
        self.modifier = 0;
        self.keycodes = [0; 6];
    }

    pub fn set_modifier(&mut self, modifier: u8) {
        self.modifier = modifier;
    }

    /// 按下键。已按下则视为成功；6 键槽位已满返回 `false`。
    pub fn press(&mut self, code: u8) -> bool {
        if code == 0 {
            return false;
        }
        if self.keycodes.contains(&code) {
            return true;
        }
        for slot in &mut self.keycodes {
            if *slot == 0 {
                *slot = code;
                return true;
            }
        }
        false
    }

    /// 抬起键，并把非零键码左移紧凑（Boot 协议习惯）。
    pub fn release(&mut self, code: u8) {
        for slot in &mut self.keycodes {
            if *slot == code {
                *slot = 0;
            }
        }
        let mut compact = [0u8; 6];
        let mut i = 0;
        for &c in &self.keycodes {
            if c != 0 {
                compact[i] = c;
                i += 1;
            }
        }
        self.keycodes = compact;
    }
}

/// 相对鼠标状态：按键位图 + 位移 / 滚轮 / 水平平移。
#[derive(Clone, Copy)]
pub struct MouseState {
    /// bit0 左键, bit1 右键, bit2 中键, bit3/4 侧键。
    pub buttons: u8,
    /// 相对 X（右为正）。
    pub x: i8,
    /// 相对 Y（下为正，与常见屏幕坐标系一致）。
    pub y: i8,
    /// 垂直滚轮（上为正）。
    pub wheel: i8,
    /// 水平平移（右为正）。
    pub pan: i8,
}

impl MouseState {
    pub const fn new() -> Self {
        Self {
            buttons: 0,
            x: 0,
            y: 0,
            wheel: 0,
            pan: 0,
        }
    }

    /// 生成待发送的 HID 鼠标输入报告。
    pub fn report(&self) -> MouseReport {
        MouseReport {
            buttons: self.buttons,
            x: self.x,
            y: self.y,
            wheel: self.wheel,
            pan: self.pan,
        }
    }

    /// 清零相对位移与滚轮（按键状态保留）。
    /// 相对轴发完一次后应清零，避免主机反复累加同一增量。
    pub fn clear_motion(&mut self) {
        self.x = 0;
        self.y = 0;
        self.wheel = 0;
        self.pan = 0;
    }

    pub fn clear_buttons(&mut self) {
        self.buttons = 0;
    }

    pub fn buttons_held(&self) -> bool {
        self.buttons != 0
    }

    pub fn button_down(&mut self, mask: u8) {
        self.buttons |= mask;
    }

    pub fn button_up(&mut self, mask: u8) {
        self.buttons &= !mask;
    }
}
