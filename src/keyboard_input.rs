//! 键盘注入：Win32 SendInput（可选）或 RP2040 USB HID（CDC 命令）。
//!
//! 支持点按（tap）与按住差分同步（HeldKeys），供 live NavBot / UI 测试。

use std::thread;
use std::time::{Duration, Instant};

use mxd_tools::game::InputFrame;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    MapVirtualKeyW, SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, MAPVK_VK_TO_VSC, VIRTUAL_KEY, VK_A, VK_C, VK_CONTROL,
    VK_DOWN, VK_ESCAPE, VK_LEFT, VK_MENU, VK_RETURN, VK_RIGHT, VK_S, VK_SHIFT, VK_SPACE, VK_TAB,
    VK_UP, VK_W,
};

use crate::usb_hid::UsbHidClient;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    A,
    W,
    S,
    D,
    C,
    Z,
    J,
    Digit1,
    Space,
    Enter,
    Esc,
    Tab,
    Left,
    Right,
    Up,
    Down,
    LeftCtrl,
    LeftAlt,
}

impl Key {
    pub fn label(self) -> &'static str {
        match self {
            Key::A => "A",
            Key::W => "W",
            Key::S => "S",
            Key::D => "D",
            Key::C => "C",
            Key::Z => "Z",
            Key::J => "J",
            Key::Digit1 => "1",
            Key::Space => "Space",
            Key::Enter => "Enter",
            Key::Esc => "Esc",
            Key::Tab => "Tab",
            Key::Left => "←左",
            Key::Right => "→右",
            Key::Up => "↑上",
            Key::Down => "↓下",
            Key::LeftCtrl => "Ctrl",
            Key::LeftAlt => "Alt",
        }
    }

    fn vk(self) -> VIRTUAL_KEY {
        match self {
            Key::A => VK_A,
            Key::W => VK_W,
            Key::S => VK_S,
            Key::D => VIRTUAL_KEY(0x44),
            Key::C => VK_C,
            Key::Z => VIRTUAL_KEY(0x5A),
            Key::J => VIRTUAL_KEY(0x4A),
            Key::Digit1 => VIRTUAL_KEY(0x31),
            Key::Space => VK_SPACE,
            Key::Enter => VK_RETURN,
            Key::Esc => VK_ESCAPE,
            Key::Tab => VK_TAB,
            Key::Left => VK_LEFT,
            Key::Right => VK_RIGHT,
            Key::Up => VK_UP,
            Key::Down => VK_DOWN,
            Key::LeftCtrl => VK_CONTROL,
            Key::LeftAlt => VK_MENU,
        }
    }

    fn is_extended(self) -> bool {
        matches!(self, Key::Left | Key::Right | Key::Up | Key::Down)
    }

    /// USB HID Keyboard Usage ID（Boot 协议）。
    pub fn hid_usage(self) -> u8 {
        match self {
            Key::A => 0x04,
            Key::W => 0x1A,
            Key::S => 0x16,
            Key::D => 0x07,
            Key::C => 0x06,
            Key::Z => 0x1D,
            Key::J => 0x0D,
            Key::Digit1 => 0x1E,
            Key::Space => 0x2C,
            Key::Enter => 0x28,
            Key::Esc => 0x29,
            Key::Tab => 0x2B,
            Key::Left => 0x50,
            Key::Right => 0x4F,
            Key::Up => 0x52,
            Key::Down => 0x51,
            Key::LeftCtrl => 0xE0,
            Key::LeftAlt => 0xE2,
        }
    }

    /// 是否为 HID 修饰键（Usage 0xE0~0xE7）：USB 通道必须走 modifier 字节而非 keycodes。
    pub fn is_modifier(self) -> bool {
        matches!(self, Key::LeftCtrl | Key::LeftAlt)
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

    fn hid_mask(self) -> u8 {
        match self {
            Modifier::LeftCtrl => 0x01,
            Modifier::LeftShift => 0x02,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyboardBackend {
    /// 默认：RP2040 USB 虚拟键盘（CDC）
    UsbHid,
    /// 可选：Win32 SendInput
    SendInput,
}

impl KeyboardBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::UsbHid => "USB 虚拟键盘 (RP2040)",
            Self::SendInput => "SendInput（软件注入）",
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeyboardConfig {
    pub backend: KeyboardBackend,
    /// 空 = 自动检测 VID/PID；否则为串口名如 `COM5`
    pub usb_port: String,
}

impl Default for KeyboardConfig {
    fn default() -> Self {
        Self {
            backend: KeyboardBackend::UsbHid,
            usb_port: String::new(),
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

fn send_inputs(inputs: &[INPUT]) -> Result<(), String> {
    if inputs.is_empty() {
        return Ok(());
    }
    let sent = unsafe { SendInput(inputs, std::mem::size_of::<INPUT>() as i32) };
    if sent as usize == inputs.len() {
        Ok(())
    } else {
        Err(format!("SendInput 只发送了 {sent}/{}", inputs.len()))
    }
}

fn modifiers_mask(modifiers: &[Modifier]) -> u8 {
    modifiers.iter().fold(0u8, |acc, m| acc | m.hid_mask())
}

enum Sink {
    SendInput,
    Usb(UsbHidClient),
}

impl Sink {
    fn open(cfg: &KeyboardConfig) -> Result<Self, String> {
        match cfg.backend {
            KeyboardBackend::SendInput => Ok(Self::SendInput),
            KeyboardBackend::UsbHid => {
                let client = if cfg.usb_port.trim().is_empty() {
                    UsbHidClient::open_auto()?
                } else {
                    UsbHidClient::open_named(cfg.usb_port.trim())?
                };
                Ok(Self::Usb(client))
            }
        }
    }

    fn backend_name(&self) -> &'static str {
        match self {
            Self::SendInput => "SendInput",
            Self::Usb(_) => "USB-HID",
        }
    }

    fn port_label(&self) -> Option<String> {
        match self {
            Self::Usb(c) => Some(c.port_name()),
            Self::SendInput => None,
        }
    }

    fn key_down(&mut self, key: Key) -> Result<(), String> {
        match self {
            Self::SendInput => send_inputs(&[key_input(key.vk(), false, key.is_extended())]),
            Self::Usb(c) => {
                let usage = key.hid_usage();
                // 修饰键（0xE0~0xE7）在 keycodes 数组中无效，必须置 modifier 字节。
                if key.is_modifier() {
                    c.modifier_down(1 << (usage - 0xE0))
                } else {
                    c.key_down(usage)
                }
            }
        }
    }

    fn key_up(&mut self, key: Key) -> Result<(), String> {
        match self {
            Self::SendInput => send_inputs(&[key_input(key.vk(), true, key.is_extended())]),
            Self::Usb(c) => {
                let usage = key.hid_usage();
                if key.is_modifier() {
                    c.modifier_up(1 << (usage - 0xE0))
                } else {
                    c.key_up(usage)
                }
            }
        }
    }

    fn set_modifiers(&mut self, modifiers: &[Modifier]) -> Result<(), String> {
        match self {
            Self::SendInput => {
                let mut down = Vec::new();
                for m in modifiers {
                    down.push(key_input(m.vk(), false, false));
                }
                send_inputs(&down)
            }
            Self::Usb(c) => c.set_modifier_mask(modifiers_mask(modifiers)),
        }
    }

    fn clear_modifiers(&mut self) -> Result<(), String> {
        match self {
            Self::SendInput => Ok(()),
            Self::Usb(c) => c.set_modifier_mask(0),
        }
    }

    fn clear_all_keys(&mut self) -> Result<(), String> {
        match self {
            Self::SendInput => Ok(()),
            Self::Usb(c) => c.clear_keys(),
        }
    }

    fn keepalive(&mut self) -> Result<(), String> {
        match self {
            Self::SendInput => Ok(()),
            Self::Usb(c) => c.keepalive(),
        }
    }
}

fn tap_on(sink: &mut Sink, modifiers: &[Modifier], keys: &[Key]) -> Result<(), String> {
    if keys.is_empty() {
        return Ok(());
    }
    match sink {
        Sink::SendInput => {
            let mut down = Vec::new();
            for m in modifiers {
                down.push(key_input(m.vk(), false, false));
            }
            for k in keys {
                down.push(key_input(k.vk(), false, k.is_extended()));
            }
            send_inputs(&down)?;
            thread::sleep(Duration::from_millis(30));
            let mut up = Vec::new();
            for k in keys.iter().rev() {
                up.push(key_input(k.vk(), true, k.is_extended()));
            }
            for m in modifiers.iter().rev() {
                up.push(key_input(m.vk(), true, false));
            }
            send_inputs(&up)
        }
        Sink::Usb(_) => {
            sink.set_modifiers(modifiers)?;
            for k in keys {
                sink.key_down(*k)?;
            }
            thread::sleep(Duration::from_millis(30));
            for k in keys.iter().rev() {
                sink.key_up(*k)?;
            }
            sink.clear_modifiers()
        }
    }
}

/// UI 测试用：临时打开后端，发一次点按后关闭（USB 口立即释放）。
pub fn tap(cfg: &KeyboardConfig, modifiers: &[Modifier], key: Key) -> Result<(), String> {
    let mut sink = Sink::open(cfg)?;
    tap_on(&mut sink, modifiers, &[key])
}

pub fn tap_keys(cfg: &KeyboardConfig, modifiers: &[Modifier], keys: &[Key]) -> Result<(), String> {
    let mut sink = Sink::open(cfg)?;
    tap_on(&mut sink, modifiers, keys)
}

/// 与 `mini_game::poll_input` 对齐的键位映射。
/// 冒险岛正式版键位：Alt=跳跃，Ctrl=攻击。
const NAV_KEYS: &[(Key, fn(&InputFrame) -> bool)] = &[
    (Key::Left, |f| f.left),
    (Key::Right, |f| f.right),
    (Key::Up, |f| f.up),
    (Key::Down, |f| f.down),
    (Key::LeftAlt, |f| f.jump),
    (Key::LeftCtrl, |f| f.attack),
    (Key::Z, |f| f.pick_up),
    (Key::Digit1, |f| f.use_potion),
];

/// 差分同步：只发送相对上一帧变化的 keydown/keyup。
pub struct HeldKeys {
    sink: Sink,
    down: [bool; NAV_KEYS.len()],
    last_keepalive: Instant,
}

impl HeldKeys {
    pub fn open(cfg: &KeyboardConfig) -> Result<Self, String> {
        let sink = Sink::open(cfg)?;
        Ok(Self {
            sink,
            down: [false; NAV_KEYS.len()],
            last_keepalive: Instant::now(),
        })
    }

    pub fn describe(&self) -> String {
        match self.sink.port_label() {
            Some(p) => format!("{} @ {p}", self.sink.backend_name()),
            None => self.sink.backend_name().to_string(),
        }
    }

    pub fn sync_frame(&mut self, frame: &InputFrame) -> Result<(), String> {
        // 防误触 Nvidia 录制弹窗（快捷键 Alt+Z）：跳跃(Alt) 与拾取(Z) 不允许同时按住。
        // 同帧两者同求时跳跃优先、拾取让位——落地后 bot 会持续再发拾取，不丢功能。
        let mut frame = *frame;
        if frame.jump && frame.pick_up {
            frame.pick_up = false;
        }
        // 先全部松开、再按下：避免 LZ→LJ 这类转换帧中 Alt 按下时 Z 尚未松开，形成 Alt+Z 瞬间重叠。
        for (i, (key, getter)) in NAV_KEYS.iter().enumerate() {
            if self.down[i] && !getter(&frame) {
                self.sink.key_up(*key)?;
                self.down[i] = false;
            }
        }
        for (i, (key, getter)) in NAV_KEYS.iter().enumerate() {
            if getter(&frame) && !self.down[i] {
                self.sink.key_down(*key)?;
                self.down[i] = true;
            }
        }
        // 按住期间定期 ping，喂固件 HOLD_WATCHDOG（边沿同步本身不会持续发包）
        if self.down.iter().any(|&d| d)
            && self.last_keepalive.elapsed() >= Duration::from_millis(800)
        {
            let _ = self.sink.keepalive();
            self.last_keepalive = Instant::now();
        }
        Ok(())
    }

    pub fn release_all(&mut self) -> Result<(), String> {
        match &mut self.sink {
            Sink::Usb(c) => {
                c.clear_keys()?;
                self.down = [false; NAV_KEYS.len()];
                Ok(())
            }
            Sink::SendInput => {
                let mut ups = Vec::new();
                for (i, (key, _)) in NAV_KEYS.iter().enumerate() {
                    if self.down[i] {
                        ups.push(key_input(key.vk(), true, key.is_extended()));
                        self.down[i] = false;
                    }
                }
                send_inputs(&ups)
            }
        }
    }
}

impl Drop for HeldKeys {
    fn drop(&mut self) {
        let _ = self.release_all();
    }
}

/// 保持 USB 连接的会话（UI 连续测试，避免每次重开串口）。
pub struct KeyboardSession {
    sink: Sink,
}

impl KeyboardSession {
    pub fn open(cfg: &KeyboardConfig) -> Result<Self, String> {
        Ok(Self {
            sink: Sink::open(cfg)?,
        })
    }

    pub fn describe(&self) -> String {
        match self.sink.port_label() {
            Some(p) => format!("{} @ {p}", self.sink.backend_name()),
            None => self.sink.backend_name().to_string(),
        }
    }

    pub fn tap(&mut self, modifiers: &[Modifier], key: Key) -> Result<(), String> {
        tap_on(&mut self.sink, modifiers, &[key])
    }

    pub fn tap_keys(&mut self, modifiers: &[Modifier], keys: &[Key]) -> Result<(), String> {
        tap_on(&mut self.sink, modifiers, keys)
    }

    pub fn ping(&mut self) -> Result<(), String> {
        match &mut self.sink {
            Sink::Usb(c) => c.ping(),
            Sink::SendInput => Ok(()),
        }
    }

    pub fn clear_all(&mut self) -> Result<(), String> {
        self.sink.clear_all_keys()
    }
}

impl Drop for KeyboardSession {
    fn drop(&mut self) {
        let _ = self.clear_all();
    }
}
