use std::path::{Path, PathBuf};

use chrono::Local;
use image::{ImageBuffer, Rgb, RgbImage};
use windows::Win32::Foundation::{HWND, LPARAM, MAX_PATH, POINT, RECT};
use windows::Win32::Graphics::Gdi::{
    BitBlt, ClientToScreen, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject,
    GetDC, GetDIBits, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS, HGDIOBJ,
    SRCCOPY,
};
use windows::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::HiDpi::{SetProcessDpiAwareness, PROCESS_PER_MONITOR_DPI_AWARE};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetClassNameW, GetClientRect, GetWindow, GetWindowThreadProcessId, IsIconic,
    IsWindowVisible, GW_OWNER,
};

const EXE_NAME: &str = "Maplestory_Classic.exe";
const SIZE: i32 = 222;

fn enable_dpi_aware() {
    unsafe {
        let _ = SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
    }
}

fn process_exe_name(pid: u32) -> String {
    unsafe {
        let Ok(handle) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
            return String::new();
        };
        let mut buf = [0u16; MAX_PATH as usize];
        let mut size = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_WIN32,
            windows::core::PWSTR(buf.as_mut_ptr()),
            &mut size,
        );
        let _ = windows::Win32::Foundation::CloseHandle(handle);
        if ok.is_err() {
            return String::new();
        }
        let path = String::from_utf16_lossy(&buf[..size as usize]);
        Path::new(&path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string()
    }
}

struct FoundHwnd {
    area: i32,
    unity: i32,
    hwnd: HWND,
}

unsafe extern "system" fn enum_proc(hwnd: HWND, lparam: LPARAM) -> windows::core::BOOL {
    unsafe {
        if !IsWindowVisible(hwnd).as_bool() {
            return true.into();
        }
        if let Ok(owner) = GetWindow(hwnd, GW_OWNER) {
            if !owner.0.is_null() {
                return true.into();
            }
        }
        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        if !process_exe_name(pid).eq_ignore_ascii_case(EXE_NAME) {
            return true.into();
        }
        let mut class = [0u16; 256];
        let n = GetClassNameW(hwnd, &mut class);
        let class_name = String::from_utf16_lossy(&class[..n as usize]);
        let (w, h) = client_size(hwnd);
        let area = w.max(0) * h.max(0);
        let list = &mut *(lparam.0 as *mut Vec<FoundHwnd>);
        list.push(FoundHwnd {
            area,
            unity: if class_name.eq_ignore_ascii_case("UnityWndClass") {
                1
            } else {
                0
            },
            hwnd,
        });
        true.into()
    }
}

fn find_game_hwnd() -> Option<HWND> {
    let mut found: Vec<FoundHwnd> = Vec::new();
    unsafe {
        let _ = EnumWindows(Some(enum_proc), LPARAM(&mut found as *mut _ as isize));
    }
    found.retain(|item| item.area >= SIZE);
    found.sort_by(|a, b| b.area.cmp(&a.area).then(b.unity.cmp(&a.unity)));
    found.first().map(|item| item.hwnd)
}

fn client_size(hwnd: HWND) -> (i32, i32) {
    let mut rect = RECT::default();
    unsafe {
        let _ = GetClientRect(hwnd, &mut rect);
    }
    (rect.right - rect.left, rect.bottom - rect.top)
}

fn capture_client_square(hwnd: HWND, size: i32) -> Result<RgbImage, String> {
    unsafe {
        let hdc = GetDC(Some(hwnd));
        if hdc.is_invalid() {
            return Err("GetDC 失败".into());
        }
        let memdc = CreateCompatibleDC(Some(hdc));
        let hbmp = CreateCompatibleBitmap(hdc, size, size);
        let old = SelectObject(memdc, HGDIOBJ(hbmp.0));
        let result = (|| {
            BitBlt(memdc, 0, 0, size, size, Some(hdc), 0, 0, SRCCOPY)
                .map_err(|e| e.to_string())?;
            let mut bmi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: size,
                    biHeight: -size,
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0 as u32,
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut buf = vec![0u8; (size * size * 4) as usize];
            let got = GetDIBits(
                memdc,
                hbmp,
                0,
                size as u32,
                Some(buf.as_mut_ptr() as *mut _),
                &mut bmi,
                DIB_RGB_COLORS,
            );
            if got == 0 {
                return Err("GetDIBits 失败".into());
            }
            let mut img = ImageBuffer::new(size as u32, size as u32);
            for y in 0..size as u32 {
                for x in 0..size as u32 {
                    let i = ((y * size as u32 + x) * 4) as usize;
                    let b = buf[i];
                    let g = buf[i + 1];
                    let r = buf[i + 2];
                    img.put_pixel(x, y, Rgb([r, g, b]));
                }
            }
            Ok(img)
        })();
        SelectObject(memdc, old);
        let _ = DeleteObject(HGDIOBJ(hbmp.0));
        let _ = DeleteDC(memdc);
        let _ = windows::Win32::Graphics::Gdi::ReleaseDC(Some(hwnd), hdc);
        result
    }
}

fn capture_screen_square(hwnd: HWND, size: i32) -> Result<RgbImage, String> {
    let mut pt = POINT { x: 0, y: 0 };
    unsafe {
        if !ClientToScreen(hwnd, &mut pt).as_bool() {
            return Err("ClientToScreen 失败".into());
        }
        let hdc = GetDC(None);
        if hdc.is_invalid() {
            return Err("GetDC(screen) 失败".into());
        }
        let memdc = CreateCompatibleDC(Some(hdc));
        let hbmp = CreateCompatibleBitmap(hdc, size, size);
        let old = SelectObject(memdc, HGDIOBJ(hbmp.0));
        let result = (|| {
            BitBlt(memdc, 0, 0, size, size, Some(hdc), pt.x, pt.y, SRCCOPY)
                .map_err(|e| e.to_string())?;
            let mut bmi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: size,
                    biHeight: -size,
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0 as u32,
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut buf = vec![0u8; (size * size * 4) as usize];
            let got = GetDIBits(
                memdc,
                hbmp,
                0,
                size as u32,
                Some(buf.as_mut_ptr() as *mut _),
                &mut bmi,
                DIB_RGB_COLORS,
            );
            if got == 0 {
                return Err("GetDIBits 失败".into());
            }
            let mut img = ImageBuffer::new(size as u32, size as u32);
            for y in 0..size as u32 {
                for x in 0..size as u32 {
                    let i = ((y * size as u32 + x) * 4) as usize;
                    img.put_pixel(x, y, Rgb([buf[i + 2], buf[i + 1], buf[i]]));
                }
            }
            Ok(img)
        })();
        SelectObject(memdc, old);
        let _ = DeleteObject(HGDIOBJ(hbmp.0));
        let _ = DeleteDC(memdc);
        let _ = windows::Win32::Graphics::Gdi::ReleaseDC(None, hdc);
        result
    }
}

fn almost_black(img: &RgbImage) -> bool {
    img.pixels()
        .map(|p| p.0[0].max(p.0[1]).max(p.0[2]))
        .max()
        .unwrap_or(0)
        < 8
}

/// 查找正在运行的 Maplestory_Classic.exe，截取客户区左上角 222×222，仅返回内存图像。
pub fn capture_minimap_image() -> Result<RgbImage, String> {
    enable_dpi_aware();
    let hwnd = find_game_hwnd().ok_or_else(|| {
        format!("未找到正在运行的 {EXE_NAME}，请先启动游戏并保持窗口可见")
    })?;
    unsafe {
        if IsIconic(hwnd).as_bool() {
            return Err("游戏窗口已最小化，请先还原".into());
        }
    }
    let (width, height) = client_size(hwnd);
    if width < SIZE || height < SIZE {
        return Err(format!(
            "客户区太小：{width}x{height}，需要至少 {SIZE}x{SIZE}"
        ));
    }
    let mut img = capture_client_square(hwnd, SIZE)?;
    if almost_black(&img) {
        img = capture_screen_square(hwnd, SIZE)?;
    }
    Ok(img)
}

/// 现场截取并可选保存到目录（仅「截取小地图」按钮使用）。
pub fn capture_minimap(out_dir: &Path) -> Result<PathBuf, String> {
    let img = capture_minimap_image()?;
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let stamp = Local::now().format("%Y%m%d_%H%M%S");
    let path = out_dir.join(format!("minimap_{stamp}.png"));
    img.save(&path).map_err(|e| e.to_string())?;
    Ok(path)
}
