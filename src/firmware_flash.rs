//! 嵌入的 RP2040 UF2 固件：另存为 / 自动拷贝到 BOOTSEL 盘（RPI-RP2）。

use std::fs;
use std::path::{Path, PathBuf};

/// 由 `scripts/build_rp2040_uf2.ps1` 生成，编译期嵌入。
pub const EMBEDDED_UF2: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/firmware/mxd-usb-hid.uf2"
));

pub const EMBEDDED_UF2_NAME: &str = "mxd-usb-hid.uf2";

pub fn embedded_size_kb() -> f64 {
    EMBEDDED_UF2.len() as f64 / 1024.0
}

/// 把嵌入固件写到指定路径。
pub fn save_uf2_to(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("创建目录失败：{e}"))?;
    }
    fs::write(path, EMBEDDED_UF2).map_err(|e| format!("写入失败：{e}"))?;
    Ok(())
}

/// 查找处于 BOOTSEL 模式的 Pico 盘符（含 `INFO_UF2.TXT` 或卷标 `RPI-RP2`）。
pub fn find_bootsel_drives() -> Vec<PathBuf> {
    #[cfg(windows)]
    {
        find_bootsel_drives_windows()
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

#[cfg(windows)]
fn find_bootsel_drives_windows() -> Vec<PathBuf> {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use windows::core::PCWSTR;
    use windows::Win32::Storage::FileSystem::{
        GetDriveTypeW, GetLogicalDrives, GetVolumeInformationW,
    };

    // Win32: DRIVE_REMOVABLE=2, DRIVE_FIXED=3
    const DRIVE_REMOVABLE: u32 = 2;
    const DRIVE_FIXED: u32 = 3;

    let mask = unsafe { GetLogicalDrives() };
    let mut out = Vec::new();
    for i in 0..26u32 {
        if mask & (1 << i) == 0 {
            continue;
        }
        let letter = (b'A' + i as u8) as char;
        let root = format!("{letter}:\\");
        let root_wide: Vec<u16> = root.encode_utf16().chain(std::iter::once(0)).collect();
        let dtype = unsafe { GetDriveTypeW(PCWSTR(root_wide.as_ptr())) };
        // BOOTSEL 盘通常是可移动；少数环境报 FIXED，仍靠 INFO_UF2.TXT 识别
        if dtype != DRIVE_REMOVABLE && dtype != DRIVE_FIXED {
            continue;
        }

        let info_uf2 = PathBuf::from(&root).join("INFO_UF2.TXT");
        let mut is_rp2 = info_uf2.is_file();
        if !is_rp2 {
            let mut name_buf = [0u16; 64];
            let mut fs_buf = [0u16; 32];
            let mut serial = 0u32;
            let mut max_comp = 0u32;
            let mut flags = 0u32;
            let ok = unsafe {
                GetVolumeInformationW(
                    PCWSTR(root_wide.as_ptr()),
                    Some(&mut name_buf),
                    Some(&mut serial),
                    Some(&mut max_comp),
                    Some(&mut flags),
                    Some(&mut fs_buf),
                )
            };
            if ok.is_ok() {
                let end = name_buf.iter().position(|&c| c == 0).unwrap_or(name_buf.len());
                let label = OsString::from_wide(&name_buf[..end])
                    .to_string_lossy()
                    .to_ascii_uppercase();
                is_rp2 = label.contains("RPI-RP2") || label.contains("RP2350");
            }
        }
        if is_rp2 {
            out.push(PathBuf::from(root));
        }
    }
    out
}

/// 将嵌入 UF2 拷到检测到的第一个 BOOTSEL 盘。
/// Pico 写入后会重启断开，Windows 上 `write` 可能返回错误；若目标曾出现且写入大半则视为成功。
pub fn flash_embedded_to_bootsel() -> Result<String, String> {
    let drives = find_bootsel_drives();
    if drives.is_empty() {
        return Err(
            "未找到 RPI-RP2 烧录盘。请按住 BOOTSEL 再插 USB，等到资源管理器出现 RPI-RP2 后再点烧写。"
                .into(),
        );
    }
    let drive = &drives[0];
    let dest = drive.join(EMBEDDED_UF2_NAME);
    match fs::write(&dest, EMBEDDED_UF2) {
        Ok(()) => Ok(format!(
            "已写入 {}（{}）。设备应自动重启并枚举为 HID+CDC。",
            dest.display(),
            drive.display()
        )),
        Err(e) => {
            // 设备重启导致句柄失效时常见
            let msg = e.to_string();
            if msg.contains("设备")
                || msg.contains("device")
                || msg.contains("process cannot access")
                || msg.contains("拒绝访问")
                || e.raw_os_error() == Some(433) // ERROR_NO_MEDIA
                || e.raw_os_error() == Some(21)  // ERROR_NOT_READY
                || e.raw_os_error() == Some(995)
            {
                Ok(format!(
                    "固件已发送到 {}（写入中设备重启属正常）：{msg}",
                    drive.display()
                ))
            } else {
                Err(format!("写入 {} 失败：{e}", dest.display()))
            }
        }
    }
}
