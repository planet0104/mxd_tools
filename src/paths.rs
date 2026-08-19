use std::path::{Path, PathBuf};

/// 以程序所在目录为工作目录；找不到 exe 时用当前目录。
/// 不再依赖 save_map.py / 预置 maps 等本地文件。
pub fn workspace_root() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            return dir.to_path_buf();
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

pub fn maps_dir(root: &Path) -> PathBuf {
    root.join("maps")
}

pub fn minimap_shots_dir(root: &Path) -> PathBuf {
    root.join("minimap_shots")
}

pub fn safe_filename(text: &str) -> String {
    let cleaned: String = text
        .chars()
        .map(|c| match c {
            '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => c,
        })
        .collect();
    let trimmed = cleaned.trim_matches(|c: char| c == ' ' || c == '.' || c == '_');
    if trimmed.is_empty() {
        "map".into()
    } else {
        trimmed.to_string()
    }
}
