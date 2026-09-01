//! CDC 串口命令解析（平台无关部分）。
//!
//! 协议：一行一条命令，以 `\n` / `\r` 结束；数字支持十进制或 `0x` 十六进制。
//! 具体命令表见 [`HELP`]，也可在串口发送 `help`。
//! 实际执行见 [`crate::io::handle_line`]。

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

pub(crate) fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
    use core::cmp::{max, min};
    max(lo, min(hi, v))
}

pub(crate) fn skip_ws(s: &str) -> &str {
    s.trim_start()
}

pub(crate) fn next_token<'a>(s: &'a str) -> Option<(&'a str, &'a str)> {
    let s = skip_ws(s);
    if s.is_empty() {
        return None;
    }
    match s.find(|c: char| c.is_whitespace()) {
        Some(i) => Some((&s[..i], &s[i..])),
        None => Some((s, "")),
    }
}

pub(crate) fn parse_i32(tok: &str) -> Option<i32> {
    let t = tok.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        i32::from_str_radix(hex, 16).ok()
    } else {
        t.parse::<i32>().ok()
    }
}

pub(crate) fn parse_u8(tok: &str) -> Option<u8> {
    parse_i32(tok)?.try_into().ok()
}

pub(crate) fn parse_i8(tok: &str) -> Option<i8> {
    parse_i32(tok)?.try_into().ok()
}

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

pub(crate) fn parse_modifier(tok: &str) -> Option<u8> {
    if let Some(v) = parse_u8(tok) {
        return Some(v);
    }
    let mut mask = 0u8;
    for part in tok.split(|c| c == '+' || c == '|' || c == ',') {
        mask |= mod_bit(part.trim())?;
    }
    Some(mask)
}

pub(crate) fn parse_button(tok: &str) -> Option<u8> {
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
