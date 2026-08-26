//! 训练 CLI 带时间戳的 stderr 日志。

use chrono::Local;

pub fn ts() -> String {
    Local::now().format("%H:%M:%S").to_string()
}

pub fn log_line(msg: impl AsRef<str>) {
    eprintln!("[{}] {}", ts(), msg.as_ref());
}

#[macro_export]
macro_rules! train_log {
    ($($arg:tt)*) => {{
        eprintln!("[{}] {}", $crate::trainer::log::ts(), format!($($arg)*))
    }};
}
