#![no_std]

//! RP2040 USB HID 键盘 + HID 鼠标 + CDC 串口复合设备库。
//!
//! 通过 CDC 串口接收文本命令，固件据此发送 HID 报告。

/// ASCII 字符到 HID Usage ID 的映射。
pub mod ascii;
/// CDC 行命令解析与执行。
pub mod cmd;
/// Embassy USB 写入端包装。
pub mod io;
/// 键盘 / 鼠标当前按下状态。
pub mod state;
