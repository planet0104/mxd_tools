//! CDC 串口命令解析。
//!
//! 协议：一行一条命令，以 `\n` / `\r` 结束；数字支持十进制或 `0x` 十六进制。
//! 具体命令表见 [`HELP`]，也可在串口发送 `help`。

use crate::ascii::char_to_hid;
use crate::state::{KeyboardState, MouseState};

/// 串口 `help` 命令返回的说明文本。
pub const HELP: &str = "\
USB HID+CDC commands (line-based, \\n terminated):\r\n\
  help\r\n\
  ping\r\n\
Keyboard:\r\n\
  kb <mod> <k0..k5>   full report (hex/dec)\r\n\
  km <mod>            set modifiers only\r\n\
  kd <code>           key down\r\n\
  ku <code>           key up\r\n\
  kp <code>           key press (down+up)\r\n\
  kc                  clear keys+mods\r\n\
  type <text>         type ASCII text\r\n\
Mouse:\r\n\
  ms <btn> <x> <y> <wheel> [pan]  full report\r\n\
  mm <dx> <dy>        relative move\r\n\
  md <btn>            button down (l/r/m/4/5)\r\n\
  mu <btn>            button up\r\n\
  mc <btn>            click\r\n\
  mw <delta>          vertical wheel\r\n\
  mp <delta>          horizontal pan\r\n\
  m0                  release buttons\r\n\
Modifiers: lctrl lshift lalt lgui rctrl rshift ralt rgui (or hex)\r\n\
";

/// 命令执行时需要的 USB 设备能力抽象。
///
/// 由 `main` 里的 `UsbIo` 实现，便于在延时 / 重试期间持续 poll USB。
pub trait DeviceIo {
    /// 轮询 USB 总线（处理枚举、传输完成等）。
    fn poll_usb(&mut self);
    /// 推送键盘报告；端点忙时返回 `false`。
    fn push_keyboard(&mut self, kb: &KeyboardState) -> bool;
    /// 推送鼠标报告；端点忙时返回 `false`。
    fn push_mouse(&mut self, ms: &MouseState) -> bool;
    /// 向 CDC 串口写回文本（如 OK / ERR / help）。
    fn write_reply(&mut self, msg: &str);
    /// 延时，期间持续 poll USB，避免主机超时。
    fn delay_ms(&mut self, ms: u32);
}

fn skip_ws(s: &str) -> &str {
    s.trim_start()
}

/// 取出下一个空白分隔的 token，返回 `(token, 剩余字符串)`。
fn next_token<'a>(s: &'a str) -> Option<(&'a str, &'a str)> {
    let s = skip_ws(s);
    if s.is_empty() {
        return None;
    }
    match s.find(|c: char| c.is_whitespace()) {
        Some(i) => Some((&s[..i], &s[i..])),
        None => Some((s, "")),
    }
}

/// 解析有符号整数（十进制或 `0x`/`0X` 十六进制）。
fn parse_i32(tok: &str) -> Option<i32> {
    let t = tok.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        i32::from_str_radix(hex, 16).ok()
    } else {
        t.parse::<i32>().ok()
    }
}

fn parse_u8(tok: &str) -> Option<u8> {
    parse_i32(tok)?.try_into().ok()
}

fn parse_i8(tok: &str) -> Option<i8> {
    parse_i32(tok)?.try_into().ok()
}

/// 单个修饰键名称 → 位掩码。
fn mod_bit(name: &str) -> Option<u8> {
    if name.eq_ignore_ascii_case("lctrl") || name.eq_ignore_ascii_case("ctrl") {
        Some(0x01)
    } else if name.eq_ignore_ascii_case("lshift") || name.eq_ignore_ascii_case("shift") {
        Some(0x02)
    } else if name.eq_ignore_ascii_case("lalt") || name.eq_ignore_ascii_case("alt") {
        Some(0x04)
    } else if name.eq_ignore_ascii_case("lgui")
        || name.eq_ignore_ascii_case("gui")
        || name.eq_ignore_ascii_case("win")
        || name.eq_ignore_ascii_case("cmd")
    {
        Some(0x08)
    } else if name.eq_ignore_ascii_case("rctrl") {
        Some(0x10)
    } else if name.eq_ignore_ascii_case("rshift") {
        Some(0x20)
    } else if name.eq_ignore_ascii_case("ralt") {
        Some(0x40)
    } else if name.eq_ignore_ascii_case("rgui") {
        Some(0x80)
    } else if name.is_empty() || name == "0" || name.eq_ignore_ascii_case("none") {
        Some(0)
    } else {
        None
    }
}

/// 解析修饰键：可以是数字掩码，或 `lctrl+lshift` 这类名称组合。
fn parse_modifier(tok: &str) -> Option<u8> {
    if let Some(v) = parse_u8(tok) {
        return Some(v);
    }
    let mut mask = 0u8;
    for part in tok.split(|c| c == '+' || c == '|' || c == ',') {
        mask |= mod_bit(part.trim())?;
    }
    Some(mask)
}

/// 解析鼠标按键：名称（l/r/m…）或数字掩码。
fn parse_button(tok: &str) -> Option<u8> {
    if tok.eq_ignore_ascii_case("l") || tok.eq_ignore_ascii_case("left") || tok == "1" {
        Some(0x01)
    } else if tok.eq_ignore_ascii_case("r") || tok.eq_ignore_ascii_case("right") || tok == "2" {
        Some(0x02)
    } else if tok.eq_ignore_ascii_case("m") || tok.eq_ignore_ascii_case("middle") || tok == "3" {
        Some(0x04)
    } else if tok == "4" || tok.eq_ignore_ascii_case("back") {
        Some(0x08)
    } else if tok == "5" || tok.eq_ignore_ascii_case("forward") {
        Some(0x10)
    } else {
        parse_u8(tok)
    }
}

/// 推送键盘报告，端点忙则 poll 后重试。
fn push_kb_retry<D: DeviceIo>(dev: &mut D, kb: &KeyboardState) {
    for _ in 0..64 {
        if dev.push_keyboard(kb) {
            return;
        }
        dev.poll_usb();
    }
}

/// 推送鼠标报告，端点忙则 poll 后重试。
fn push_ms_retry<D: DeviceIo>(dev: &mut D, ms: &MouseState) {
    for _ in 0..64 {
        if dev.push_mouse(ms) {
            return;
        }
        dev.poll_usb();
    }
}

/// 单击：按下 → 短暂保持 → 抬起。
fn key_tap<D: DeviceIo>(dev: &mut D, kb: &mut KeyboardState, code: u8) {
    if !kb.press(code) {
        dev.write_reply("ERR key slots full\r\n");
        return;
    }
    push_kb_retry(dev, kb);
    dev.delay_ms(20);
    kb.release(code);
    push_kb_retry(dev, kb);
    dev.delay_ms(10);
}

/// 按 ASCII 逐字输入；大写 / 部分标点自动加 Left Shift。
fn type_text<D: DeviceIo>(dev: &mut D, kb: &mut KeyboardState, text: &str) {
    let saved_mod = kb.modifier;
    for ch in text.chars() {
        let Some((code, shift)) = char_to_hid(ch) else {
            continue;
        };
        if shift {
            kb.modifier |= 0x02; // Left Shift
        } else {
            kb.modifier &= !0x02;
        }
        if !kb.press(code) {
            dev.write_reply("ERR key slots full\r\n");
            break;
        }
        push_kb_retry(dev, kb);
        dev.delay_ms(15);
        kb.release(code);
        push_kb_retry(dev, kb);
        kb.modifier = saved_mod;
        dev.delay_ms(10);
    }
    kb.modifier = saved_mod;
    push_kb_retry(dev, kb);
}

/// 鼠标单击。
fn mouse_click<D: DeviceIo>(dev: &mut D, ms: &mut MouseState, mask: u8) {
    ms.clear_motion();
    ms.button_down(mask);
    push_ms_retry(dev, ms);
    dev.delay_ms(30);
    ms.button_up(mask);
    push_ms_retry(dev, ms);
    ms.clear_motion();
}

/// 处理一行完整命令（不含行尾换行符）。
pub fn handle_line<D: DeviceIo>(
    line: &str,
    dev: &mut D,
    kb: &mut KeyboardState,
    ms: &mut MouseState,
) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }

    let Some((cmd, rest)) = next_token(line) else {
        return;
    };

    if cmd.eq_ignore_ascii_case("help") || cmd == "?" {
        // 打印命令帮助
        dev.write_reply(HELP);
    } else if cmd.eq_ignore_ascii_case("ping") {
        dev.write_reply("pong\r\n");
    } else if cmd.eq_ignore_ascii_case("kb") {
        // kb <mod> <k0..k5>：一次性写入完整键盘报告
        let mut vals = [0u8; 7];
        let mut s = rest;
        for i in 0..7 {
            let Some((tok, next)) = next_token(s) else {
                dev.write_reply("ERR kb needs 7 args\r\n");
                return;
            };
            let Some(v) = parse_u8(tok) else {
                dev.write_reply("ERR bad number\r\n");
                return;
            };
            vals[i] = v;
            s = next;
        }
        kb.modifier = vals[0];
        kb.keycodes = [vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]];
        push_kb_retry(dev, kb);
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("km") {
        // km <mod>：只改修饰键
        let Some((tok, _)) = next_token(rest) else {
            dev.write_reply("ERR km <mod>\r\n");
            return;
        };
        let Some(m) = parse_modifier(tok) else {
            dev.write_reply("ERR bad mod\r\n");
            return;
        };
        kb.set_modifier(m);
        push_kb_retry(dev, kb);
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("kd") {
        // kd <code>：按下
        let Some((tok, _)) = next_token(rest) else {
            dev.write_reply("ERR kd <code>\r\n");
            return;
        };
        let Some(code) = parse_u8(tok) else {
            dev.write_reply("ERR bad code\r\n");
            return;
        };
        if !kb.press(code) {
            dev.write_reply("ERR key slots full\r\n");
            return;
        }
        push_kb_retry(dev, kb);
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("ku") {
        // ku <code>：抬起
        let Some((tok, _)) = next_token(rest) else {
            dev.write_reply("ERR ku <code>\r\n");
            return;
        };
        let Some(code) = parse_u8(tok) else {
            dev.write_reply("ERR bad code\r\n");
            return;
        };
        kb.release(code);
        push_kb_retry(dev, kb);
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("kp") {
        // kp <code>：单击
        let Some((tok, _)) = next_token(rest) else {
            dev.write_reply("ERR kp <code>\r\n");
            return;
        };
        let Some(code) = parse_u8(tok) else {
            dev.write_reply("ERR bad code\r\n");
            return;
        };
        key_tap(dev, kb, code);
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("kc") {
        // kc：清空键盘
        kb.clear();
        push_kb_retry(dev, kb);
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("type") {
        // type <text>：打字
        let text = skip_ws(rest);
        if text.is_empty() {
            dev.write_reply("ERR type <text>\r\n");
            return;
        }
        type_text(dev, kb, text);
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("ms") {
        // ms <btn> <x> <y> <wheel> [pan]：完整鼠标报告
        let mut s = rest;
        let mut nums = [0i32; 5];
        let mut n = 0;
        while n < 5 {
            let Some((tok, next)) = next_token(s) else {
                break;
            };
            let Some(v) = parse_i32(tok) else {
                dev.write_reply("ERR bad number\r\n");
                return;
            };
            nums[n] = v;
            n += 1;
            s = next;
        }
        if n < 4 {
            dev.write_reply("ERR ms <btn> <x> <y> <wheel> [pan]\r\n");
            return;
        }
        ms.buttons = nums[0].clamp(0, 255) as u8;
        ms.x = nums[1].clamp(-127, 127) as i8;
        ms.y = nums[2].clamp(-127, 127) as i8;
        ms.wheel = nums[3].clamp(-127, 127) as i8;
        ms.pan = if n >= 5 {
            nums[4].clamp(-127, 127) as i8
        } else {
            0
        };
        push_ms_retry(dev, ms);
        // 相对轴只生效一次，发完清零
        ms.clear_motion();
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("mm") {
        // mm <dx> <dy>：相对移动
        let Some((tx, s)) = next_token(rest) else {
            dev.write_reply("ERR mm <dx> <dy>\r\n");
            return;
        };
        let Some((ty, _)) = next_token(s) else {
            dev.write_reply("ERR mm <dx> <dy>\r\n");
            return;
        };
        let (Some(dx), Some(dy)) = (parse_i8(tx), parse_i8(ty)) else {
            dev.write_reply("ERR bad number\r\n");
            return;
        };
        ms.x = dx;
        ms.y = dy;
        ms.wheel = 0;
        ms.pan = 0;
        push_ms_retry(dev, ms);
        ms.clear_motion();
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("md") {
        // md <btn>：按下鼠标键
        let Some((tok, _)) = next_token(rest) else {
            dev.write_reply("ERR md <btn>\r\n");
            return;
        };
        let Some(mask) = parse_button(tok) else {
            dev.write_reply("ERR bad btn\r\n");
            return;
        };
        ms.clear_motion();
        ms.button_down(mask);
        push_ms_retry(dev, ms);
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("mu") {
        // mu <btn>：抬起鼠标键
        let Some((tok, _)) = next_token(rest) else {
            dev.write_reply("ERR mu <btn>\r\n");
            return;
        };
        let Some(mask) = parse_button(tok) else {
            dev.write_reply("ERR bad btn\r\n");
            return;
        };
        ms.clear_motion();
        ms.button_up(mask);
        push_ms_retry(dev, ms);
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("mc") {
        // mc <btn>：单击
        let Some((tok, _)) = next_token(rest) else {
            dev.write_reply("ERR mc <btn>\r\n");
            return;
        };
        let Some(mask) = parse_button(tok) else {
            dev.write_reply("ERR bad btn\r\n");
            return;
        };
        mouse_click(dev, ms, mask);
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("mw") {
        // mw <delta>：垂直滚轮
        let Some((tok, _)) = next_token(rest) else {
            dev.write_reply("ERR mw <delta>\r\n");
            return;
        };
        let Some(delta) = parse_i8(tok) else {
            dev.write_reply("ERR bad number\r\n");
            return;
        };
        ms.x = 0;
        ms.y = 0;
        ms.wheel = delta;
        ms.pan = 0;
        push_ms_retry(dev, ms);
        ms.clear_motion();
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("mp") {
        // mp <delta>：水平平移
        let Some((tok, _)) = next_token(rest) else {
            dev.write_reply("ERR mp <delta>\r\n");
            return;
        };
        let Some(delta) = parse_i8(tok) else {
            dev.write_reply("ERR bad number\r\n");
            return;
        };
        ms.x = 0;
        ms.y = 0;
        ms.wheel = 0;
        ms.pan = delta;
        push_ms_retry(dev, ms);
        ms.clear_motion();
        dev.write_reply("OK\r\n");
    } else if cmd.eq_ignore_ascii_case("m0") {
        // m0：松开全部鼠标键
        ms.clear_buttons();
        ms.clear_motion();
        push_ms_retry(dev, ms);
        dev.write_reply("OK\r\n");
    } else {
        dev.write_reply("ERR unknown cmd, try help\r\n");
    }
}
