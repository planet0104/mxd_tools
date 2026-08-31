//! ASCII / 常用控制字符 → USB HID 键盘 Usage ID。
//!
//! 返回值第二项表示是否需要同时按下 Left Shift。

/// 将单个字符映射为 `(HID 键码, 是否需要 Shift)`。
/// 无法映射的字符返回 `None`（调用方应跳过）。
pub fn char_to_hid(ch: char) -> Option<(u8, bool)> {
    match ch {
        // 字母：小写不带 Shift，大写带 Shift
        'a'..='z' => Some((0x04 + (ch as u8 - b'a'), false)),
        'A'..='Z' => Some((0x04 + (ch as u8 - b'A'), true)),
        // 数字行
        '1'..='9' => Some((0x1E + (ch as u8 - b'1'), false)),
        '0' => Some((0x27, false)),
        // 常用控制键
        '\n' | '\r' => Some((0x28, false)), // Enter
        '\x1b' => Some((0x29, false)),      // Esc
        '\x08' | '\x7f' => Some((0x2A, false)), // Backspace / DEL
        '\t' => Some((0x2B, false)),        // Tab
        ' ' => Some((0x2C, false)),         // Space
        // 标点（不带 Shift）
        '-' => Some((0x2D, false)),
        '=' => Some((0x2E, false)),
        '[' => Some((0x2F, false)),
        ']' => Some((0x30, false)),
        '\\' => Some((0x31, false)),
        ';' => Some((0x33, false)),
        '\'' => Some((0x34, false)),
        '`' => Some((0x35, false)),
        ',' => Some((0x36, false)),
        '.' => Some((0x37, false)),
        '/' => Some((0x38, false)),
        // 标点（需要 Shift）
        '!' => Some((0x1E, true)),
        '@' => Some((0x1F, true)),
        '#' => Some((0x20, true)),
        '$' => Some((0x21, true)),
        '%' => Some((0x22, true)),
        '^' => Some((0x23, true)),
        '&' => Some((0x24, true)),
        '*' => Some((0x25, true)),
        '(' => Some((0x26, true)),
        ')' => Some((0x27, true)),
        '_' => Some((0x2D, true)),
        '+' => Some((0x2E, true)),
        '{' => Some((0x2F, true)),
        '}' => Some((0x30, true)),
        '|' => Some((0x31, true)),
        ':' => Some((0x33, true)),
        '"' => Some((0x34, true)),
        '~' => Some((0x35, true)),
        '<' => Some((0x36, true)),
        '>' => Some((0x37, true)),
        '?' => Some((0x38, true)),
        _ => None,
    }
}
