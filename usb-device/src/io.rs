//! Embassy USB 写入与命令执行。

use embassy_rp::usb::{Driver, Instance};
use embassy_time::Timer;
use embassy_usb::class::cdc_acm::CdcAcmClass;
use embassy_usb::class::hid::HidWriter;
use embassy_usb::driver::EndpointError;

use crate::ascii::char_to_hid;
use crate::cmd::{
    clamp_i32, next_token, parse_button, parse_i8, parse_i32, parse_modifier, parse_u8, skip_ws,
    HELP,
};
use crate::state::{KeyboardState, MouseState};

/// 向 CDC 串口写文本。
pub async fn write_cdc<'a, T: Instance>(cdc: &mut CdcAcmClass<'a, Driver<'a, T>>, msg: &str) {
    for chunk in msg.as_bytes().chunks(64) {
        loop {
            match cdc.write_packet(chunk).await {
                Ok(()) => break,
                Err(EndpointError::Disabled) => return,
                Err(EndpointError::BufferOverflow) => {
                    Timer::after_millis(1).await;
                }
            }
        }
    }
}

/// 推送键盘 HID 报告。
pub async fn push_keyboard<'a, T: Instance>(
    keyboard: &mut HidWriter<'a, Driver<'a, T>, 8>,
    kb: &KeyboardState,
) {
    keyboard.ready().await;
    let _ = keyboard.write_serialize(&kb.report()).await;
}

/// 推送鼠标 HID 报告。
pub async fn push_mouse<'a, T: Instance>(
    mouse: &mut HidWriter<'a, Driver<'a, T>, 8>,
    ms: &MouseState,
) {
    mouse.ready().await;
    let _ = mouse.write_serialize(&ms.report()).await;
}

async fn delay_ms(ms: u32) {
    Timer::after_millis(ms as u64).await;
}

/// HID 修饰键 Usage（0xE0~0xE7）转 modifier 位图位；普通键返回 `None`。
/// 按 HID 规范修饰键必须出现在报告的 modifier 字节，放进 keycodes 数组主机会忽略。
fn modifier_bit(code: u8) -> Option<u8> {
    if (0xE0..=0xE7).contains(&code) {
        Some(1 << (code - 0xE0))
    } else {
        None
    }
}

async fn key_tap<'a, T: Instance>(
    keyboard: &mut HidWriter<'a, Driver<'a, T>, 8>,
    cdc: &mut CdcAcmClass<'a, Driver<'a, T>>,
    kb: &mut KeyboardState,
    code: u8,
) {
    if let Some(bit) = modifier_bit(code) {
        kb.modifier |= bit;
        push_keyboard(keyboard, kb).await;
        delay_ms(20).await;
        kb.modifier &= !bit;
        push_keyboard(keyboard, kb).await;
        delay_ms(10).await;
        return;
    }
    if !kb.press(code) {
        write_cdc(cdc, "ERR key slots full\r\n").await;
        return;
    }
    push_keyboard(keyboard, kb).await;
    delay_ms(20).await;
    kb.release(code);
    push_keyboard(keyboard, kb).await;
    delay_ms(10).await;
}

async fn type_text<'a, T: Instance>(
    keyboard: &mut HidWriter<'a, Driver<'a, T>, 8>,
    cdc: &mut CdcAcmClass<'a, Driver<'a, T>>,
    kb: &mut KeyboardState,
    text: &str,
) {
    let saved_mod = kb.modifier;
    for ch in text.chars() {
        let Some((code, shift)) = char_to_hid(ch) else {
            continue;
        };
        if shift {
            kb.modifier |= 0x02;
        } else {
            kb.modifier &= !0x02;
        }
        if !kb.press(code) {
            write_cdc(cdc, "ERR key slots full\r\n").await;
            break;
        }
        push_keyboard(keyboard, kb).await;
        delay_ms(15).await;
        kb.release(code);
        push_keyboard(keyboard, kb).await;
        kb.modifier = saved_mod;
        delay_ms(10).await;
    }
    kb.modifier = saved_mod;
    push_keyboard(keyboard, kb).await;
}

async fn mouse_click<'a, T: Instance>(
    mouse: &mut HidWriter<'a, Driver<'a, T>, 8>,
    ms: &mut MouseState,
    mask: u8,
) {
    ms.clear_motion();
    ms.button_down(mask);
    push_mouse(mouse, ms).await;
    delay_ms(30).await;
    ms.button_up(mask);
    push_mouse(mouse, ms).await;
    ms.clear_motion();
}

/// 处理一行 CDC 命令并驱动 HID / 串口回复。
pub async fn handle_line<'a, T: Instance>(
    keyboard: &mut HidWriter<'a, Driver<'a, T>, 8>,
    mouse: &mut HidWriter<'a, Driver<'a, T>, 8>,
    cdc: &mut CdcAcmClass<'a, Driver<'a, T>>,
    line: &str,
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
        write_cdc(cdc, HELP).await;
    } else if cmd.eq_ignore_ascii_case("ping") {
        write_cdc(cdc, "pong\r\n").await;
    } else if cmd.eq_ignore_ascii_case("kb") {
        let mut vals = [0u8; 7];
        let mut s = rest;
        for i in 0..7 {
            let Some((tok, next)) = next_token(s) else {
                write_cdc(cdc, "ERR kb needs 7 args\r\n").await;
                return;
            };
            let Some(v) = parse_u8(tok) else {
                write_cdc(cdc, "ERR bad number\r\n").await;
                return;
            };
            vals[i] = v;
            s = next;
        }
        kb.modifier = vals[0];
        kb.keycodes = [vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]];
        push_keyboard(keyboard, kb).await;
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("km") {
        let Some((tok, _)) = next_token(rest) else {
            write_cdc(cdc, "ERR km <mod>\r\n").await;
            return;
        };
        let Some(m) = parse_modifier(tok) else {
            write_cdc(cdc, "ERR bad mod\r\n").await;
            return;
        };
        kb.set_modifier(m);
        push_keyboard(keyboard, kb).await;
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("kd") {
        let Some((tok, _)) = next_token(rest) else {
            write_cdc(cdc, "ERR kd <code>\r\n").await;
            return;
        };
        let Some(code) = parse_u8(tok) else {
            write_cdc(cdc, "ERR bad code\r\n").await;
            return;
        };
        if let Some(bit) = modifier_bit(code) {
            kb.modifier |= bit;
        } else if !kb.press(code) {
            write_cdc(cdc, "ERR key slots full\r\n").await;
            return;
        }
        push_keyboard(keyboard, kb).await;
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("ku") {
        let Some((tok, _)) = next_token(rest) else {
            write_cdc(cdc, "ERR ku <code>\r\n").await;
            return;
        };
        let Some(code) = parse_u8(tok) else {
            write_cdc(cdc, "ERR bad code\r\n").await;
            return;
        };
        if let Some(bit) = modifier_bit(code) {
            kb.modifier &= !bit;
        } else {
            kb.release(code);
        }
        push_keyboard(keyboard, kb).await;
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("kp") {
        let Some((tok, _)) = next_token(rest) else {
            write_cdc(cdc, "ERR kp <code>\r\n").await;
            return;
        };
        let Some(code) = parse_u8(tok) else {
            write_cdc(cdc, "ERR bad code\r\n").await;
            return;
        };
        key_tap(keyboard, cdc, kb, code).await;
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("kc") {
        kb.clear();
        push_keyboard(keyboard, kb).await;
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("type") {
        let text = skip_ws(rest);
        if text.is_empty() {
            write_cdc(cdc, "ERR type <text>\r\n").await;
            return;
        }
        type_text(keyboard, cdc, kb, text).await;
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("ms") {
        let mut s = rest;
        let mut nums = [0i32; 5];
        let mut n = 0;
        while n < 5 {
            let Some((tok, next)) = next_token(s) else {
                break;
            };
            let Some(v) = parse_i32(tok) else {
                write_cdc(cdc, "ERR bad number\r\n").await;
                return;
            };
            nums[n] = v;
            n += 1;
            s = next;
        }
        if n < 4 {
            write_cdc(cdc, "ERR ms <btn> <x> <y> <wheel> [pan]\r\n").await;
            return;
        }
        ms.buttons = clamp_i32(nums[0], 0, 255) as u8;
        ms.x = clamp_i32(nums[1], -127, 127) as i8;
        ms.y = clamp_i32(nums[2], -127, 127) as i8;
        ms.wheel = clamp_i32(nums[3], -127, 127) as i8;
        ms.pan = if n >= 5 {
            clamp_i32(nums[4], -127, 127) as i8
        } else {
            0
        };
        push_mouse(mouse, ms).await;
        ms.clear_motion();
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("mm") {
        let Some((tx, s)) = next_token(rest) else {
            write_cdc(cdc, "ERR mm <dx> <dy>\r\n").await;
            return;
        };
        let Some((ty, _)) = next_token(s) else {
            write_cdc(cdc, "ERR mm <dx> <dy>\r\n").await;
            return;
        };
        let (Some(dx), Some(dy)) = (parse_i8(tx), parse_i8(ty)) else {
            write_cdc(cdc, "ERR bad number\r\n").await;
            return;
        };
        ms.x = dx;
        ms.y = dy;
        ms.wheel = 0;
        ms.pan = 0;
        push_mouse(mouse, ms).await;
        ms.clear_motion();
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("md") {
        let Some((tok, _)) = next_token(rest) else {
            write_cdc(cdc, "ERR md <btn>\r\n").await;
            return;
        };
        let Some(mask) = parse_button(tok) else {
            write_cdc(cdc, "ERR bad btn\r\n").await;
            return;
        };
        ms.clear_motion();
        ms.button_down(mask);
        push_mouse(mouse, ms).await;
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("mu") {
        let Some((tok, _)) = next_token(rest) else {
            write_cdc(cdc, "ERR mu <btn>\r\n").await;
            return;
        };
        let Some(mask) = parse_button(tok) else {
            write_cdc(cdc, "ERR bad btn\r\n").await;
            return;
        };
        ms.clear_motion();
        ms.button_up(mask);
        push_mouse(mouse, ms).await;
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("mc") {
        let Some((tok, _)) = next_token(rest) else {
            write_cdc(cdc, "ERR mc <btn>\r\n").await;
            return;
        };
        let Some(mask) = parse_button(tok) else {
            write_cdc(cdc, "ERR bad btn\r\n").await;
            return;
        };
        mouse_click(mouse, ms, mask).await;
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("mw") {
        let Some((tok, _)) = next_token(rest) else {
            write_cdc(cdc, "ERR mw <delta>\r\n").await;
            return;
        };
        let Some(delta) = parse_i8(tok) else {
            write_cdc(cdc, "ERR bad number\r\n").await;
            return;
        };
        ms.x = 0;
        ms.y = 0;
        ms.wheel = delta;
        ms.pan = 0;
        push_mouse(mouse, ms).await;
        ms.clear_motion();
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("mp") {
        let Some((tok, _)) = next_token(rest) else {
            write_cdc(cdc, "ERR mp <delta>\r\n").await;
            return;
        };
        let Some(delta) = parse_i8(tok) else {
            write_cdc(cdc, "ERR bad number\r\n").await;
            return;
        };
        ms.x = 0;
        ms.y = 0;
        ms.wheel = 0;
        ms.pan = delta;
        push_mouse(mouse, ms).await;
        ms.clear_motion();
        write_cdc(cdc, "OK\r\n").await;
    } else if cmd.eq_ignore_ascii_case("m0") {
        ms.clear_buttons();
        ms.clear_motion();
        push_mouse(mouse, ms).await;
        write_cdc(cdc, "OK\r\n").await;
    } else {
        write_cdc(cdc, "ERR unknown cmd, try help\r\n").await;
    }
}
