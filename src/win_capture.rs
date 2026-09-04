//! 查找 mini_game 窗口并截取客户区 RGB（与 NavBot 视觉分辨率对齐）。

use std::ffi::OsString;
use std::os::windows::ffi::OsStringExt;
use std::path::Path;

use image::RgbImage;
use windows::core::{BOOL, PWSTR};
use windows::Win32::Foundation::{CloseHandle, HWND, LPARAM, POINT, RECT};
use windows::Win32::Graphics::Gdi::{
    BitBlt, ClientToScreen, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject,
    GetDC, GetDIBits, ReleaseDC, SelectObject, SetStretchBltMode, StretchBlt, BITMAPINFO,
    BITMAPINFOHEADER, BI_RGB, COLORONCOLOR, DIB_RGB_COLORS, HBITMAP, HDC, HGDIOBJ, SRCCOPY,
};
use windows::Win32::Storage::Xps::{PrintWindow, PRINT_WINDOW_FLAGS};
use windows::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetClientRect, GetWindowTextW, GetWindowThreadProcessId, IsIconic, IsWindow,
    IsWindowVisible, SetForegroundWindow,
};

/// mini_game 默认窗口标题（见 `src/bin/mini_game.rs`）。
pub const MINI_GAME_TITLE: &str = "mxd小游戏";
const PROCESS_HINT: &str = "mini_game";

#[derive(Debug, Clone)]
pub struct GameWindow {
    pub hwnd: isize,
    pub title: String,
    pub process_path: Option<String>,
}

impl GameWindow {
    pub fn as_hwnd(&self) -> HWND {
        HWND(self.hwnd as *mut _)
    }
}

/// 枚举可见顶层窗口，优先标题匹配，其次进程名含 `mini_game`。
pub fn find_mini_game_window() -> Option<GameWindow> {
    let mut found: Vec<GameWindow> = Vec::new();
    unsafe {
        let _ = EnumWindows(
            Some(enum_callback),
            LPARAM(&mut found as *mut Vec<GameWindow> as isize),
        );
    }
    let by_title = found
        .iter()
        .find(|w| w.title.contains(MINI_GAME_TITLE))
        .cloned();
    by_title.or_else(|| {
        found.into_iter().find(|w| {
            w.process_path
                .as_deref()
                .map(process_looks_like_mini_game)
                .unwrap_or(false)
        })
    })
}

unsafe extern "system" fn enum_callback(hwnd: HWND, lparam: LPARAM) -> BOOL {
    let list = &mut *(lparam.0 as *mut Vec<GameWindow>);
    if !IsWindowVisible(hwnd).as_bool() || IsIconic(hwnd).as_bool() {
        return BOOL(1);
    }
    let title = window_title(hwnd);
    if title.is_empty() {
        return BOOL(1);
    }
    let process_path = process_image_path(hwnd);
    let title_hit = title.contains(MINI_GAME_TITLE);
    let proc_hit = process_path
        .as_deref()
        .map(process_looks_like_mini_game)
        .unwrap_or(false);
    if title_hit || proc_hit {
        list.push(GameWindow {
            hwnd: hwnd.0 as isize,
            title,
            process_path,
        });
    }
    BOOL(1)
}

fn process_looks_like_mini_game(path: &str) -> bool {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case(PROCESS_HINT))
        .unwrap_or(false)
}

fn window_title(hwnd: HWND) -> String {
    let mut buf = [0u16; 256];
    let n = unsafe { GetWindowTextW(hwnd, &mut buf) };
    if n <= 0 {
        return String::new();
    }
    OsString::from_wide(&buf[..n as usize])
        .to_string_lossy()
        .into_owned()
}

fn process_image_path(hwnd: HWND) -> Option<String> {
    let mut pid = 0u32;
    unsafe {
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
    }
    if pid == 0 {
        return None;
    }
    let handle =
        unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
    let mut buf = vec![0u16; 512];
    let mut size = buf.len() as u32;
    let ok = unsafe {
        QueryFullProcessImageNameW(
            handle,
            Default::default(),
            PWSTR(buf.as_mut_ptr()),
            &mut size,
        )
    };
    let _ = unsafe { CloseHandle(handle) };
    if ok.is_err() || size == 0 {
        return None;
    }
    Some(
        OsString::from_wide(&buf[..size as usize])
            .to_string_lossy()
            .into_owned(),
    )
}

pub fn window_alive(hwnd: HWND) -> bool {
    unsafe { IsWindow(Some(hwnd)).as_bool() && IsWindowVisible(hwnd).as_bool() }
}

/// 尽量把游戏窗口拉到前台，便于 SendInput 打到正确目标。
pub fn focus_window(hwnd: HWND) -> bool {
    unsafe { SetForegroundWindow(hwnd).as_bool() }
}

struct MemBitmap {
    hdc: HDC,
    hbmp: HBITMAP,
    old: HGDIOBJ,
    w: i32,
    h: i32,
}

impl MemBitmap {
    unsafe fn create(hdc_ref: HDC, w: i32, h: i32) -> Result<Self, String> {
        let hdc = CreateCompatibleDC(Some(hdc_ref));
        if hdc.is_invalid() {
            return Err("CreateCompatibleDC 失败".into());
        }
        let hbmp = CreateCompatibleBitmap(hdc_ref, w, h);
        if hbmp.is_invalid() {
            let _ = DeleteDC(hdc);
            return Err("CreateCompatibleBitmap 失败".into());
        }
        let old = SelectObject(hdc, HGDIOBJ(hbmp.0));
        Ok(Self {
            hdc,
            hbmp,
            old,
            w,
            h,
        })
    }

    unsafe fn destroy(self) {
        SelectObject(self.hdc, self.old);
        let _ = DeleteObject(HGDIOBJ(self.hbmp.0));
        let _ = DeleteDC(self.hdc);
    }
}

/// 截取客户区，GDI StretchBlt 缩到目标尺寸，再整块 BGRA→RGB。
pub fn capture_client_rgb(
    hwnd: HWND,
    target_w: u32,
    target_h: u32,
) -> Result<RgbImage, String> {
    if !window_alive(hwnd) {
        return Err("游戏窗口已关闭".into());
    }
    let target_w = target_w.max(1);
    let target_h = target_h.max(1);
    let mut rect = RECT::default();
    unsafe {
        GetClientRect(hwnd, &mut rect).map_err(|e| format!("GetClientRect: {e}"))?;
    }
    let src_w = (rect.right - rect.left).max(1);
    let src_h = (rect.bottom - rect.top).max(1);

    let hdc_win = unsafe { GetDC(Some(hwnd)) };
    if hdc_win.is_invalid() {
        return Err("GetDC 失败".into());
    }

    let result = unsafe { capture_inner(hwnd, hdc_win, src_w, src_h, target_w, target_h) };
    unsafe {
        ReleaseDC(Some(hwnd), hdc_win);
    }
    result
}

unsafe fn capture_inner(
    hwnd: HWND,
    hdc_win: HDC,
    src_w: i32,
    src_h: i32,
    target_w: u32,
    target_h: u32,
) -> Result<RgbImage, String> {
    let src = MemBitmap::create(hdc_win, src_w, src_h)?;
    blit_client_to_dc(hwnd, hdc_win, src.hdc, src_w, src_h);

    let same_size = src_w as u32 == target_w && src_h as u32 == target_h;
    let rgb = if same_size {
        let out = dib_bgra_to_rgb(src.hdc, src.hbmp, target_w, target_h)?;
        src.destroy();
        out
    } else {
        let dst = MemBitmap::create(hdc_win, target_w as i32, target_h as i32)?;
        let _ = SetStretchBltMode(dst.hdc, COLORONCOLOR);
        let ok = StretchBlt(
            dst.hdc,
            0,
            0,
            target_w as i32,
            target_h as i32,
            Some(src.hdc),
            0,
            0,
            src_w,
            src_h,
            SRCCOPY,
        );
        if !ok.as_bool() {
            dst.destroy();
            src.destroy();
            return Err("StretchBlt 失败".into());
        }
        let out = dib_bgra_to_rgb(dst.hdc, dst.hbmp, target_w, target_h)?;
        dst.destroy();
        src.destroy();
        out
    };
    Ok(rgb)
}

unsafe fn blit_client_to_dc(hwnd: HWND, hdc_win: HDC, hdc_dst: HDC, w: i32, h: i32) {
    // wgpu/GL：优先 PrintWindow；失败再 BitBlt 客户区 / 屏幕坐标。
    let printed =
        PrintWindow(hwnd, hdc_dst, PRINT_WINDOW_FLAGS(1 | 2)).as_bool(); // CLIENTONLY|RENDERFULLCONTENT
    if printed {
        return;
    }
    if BitBlt(hdc_dst, 0, 0, w, h, Some(hdc_win), 0, 0, SRCCOPY).is_ok() {
        return;
    }
    let hdc_screen = GetDC(None);
    if hdc_screen.is_invalid() {
        return;
    }
    let mut pt = POINT { x: 0, y: 0 };
    let _ = ClientToScreen(hwnd, &mut pt);
    let _ = BitBlt(hdc_dst, 0, 0, w, h, Some(hdc_screen), pt.x, pt.y, SRCCOPY);
    ReleaseDC(None, hdc_screen);
}

unsafe fn dib_bgra_to_rgb(
    hdc: HDC,
    hbmp: HBITMAP,
    w: u32,
    h: u32,
) -> Result<RgbImage, String> {
    let mut bmi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: w as i32,
            biHeight: -(h as i32), // top-down
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0 as u32,
            ..Default::default()
        },
        ..Default::default()
    };
    let px = (w * h) as usize;
    let mut bgra = vec![0u8; px * 4];
    let got = GetDIBits(
        hdc,
        hbmp,
        0,
        h,
        Some(bgra.as_mut_ptr() as *mut _),
        &mut bmi,
        DIB_RGB_COLORS,
    );
    if got == 0 {
        return Err("GetDIBits 失败".into());
    }
    Ok(bgra32_to_rgb_image(&bgra, w, h))
}

/// 整块 BGRA→RGB（无逐像素 put_pixel）。
fn bgra32_to_rgb_image(bgra: &[u8], w: u32, h: u32) -> RgbImage {
    let px = (w * h) as usize;
    debug_assert!(bgra.len() >= px * 4);
    let mut rgb = vec![0u8; px * 3];
    let mut s = 0usize;
    let mut d = 0usize;
    while d < px * 3 {
        rgb[d] = bgra[s + 2];
        rgb[d + 1] = bgra[s + 1];
        rgb[d + 2] = bgra[s];
        s += 4;
        d += 3;
    }
    RgbImage::from_raw(w, h, rgb).expect("rgb buffer size matches")
}
