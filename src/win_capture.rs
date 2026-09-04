//! 查找游戏窗口并截取客户区 RGB（与 NavBot 视觉分辨率对齐）。
//!
//! 支持：
//! - 复刻版 `mini_game`（标题「mxd小游戏」）
//! - 正式客户端（标题「冒险岛怀旧服」/ 进程 Maplestory_Classic）

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

/// 复刻版 mini_game 默认窗口标题（见 `src/bin/mini_game.rs`）。
pub const MINI_GAME_TITLE: &str = "mxd小游戏";
const MINI_GAME_PROCESS: &str = "mini_game";

/// 正式怀旧服客户端窗口标题（当前线上窗口名）。
pub const CLASSIC_CLIENT_TITLE: &str = "冒险岛怀旧服";
const CLASSIC_CLIENT_PROCESS: &str = "Maplestory_Classic";

/// 正式客户端默认安装目录（用于进程路径提示/匹配）。
pub const CLASSIC_INSTALL_DIR: &str =
    r"D:\Program Files\上海数龙科技有限公司\冒险岛online\mxdclassic";

/// Bot 附着的截图目标。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureTarget {
    /// 本地复刻版 `mini_game`
    MiniGame,
    /// 正式「冒险岛怀旧服」客户端
    ClassicClient,
}

impl CaptureTarget {
    pub fn label(self) -> &'static str {
        match self {
            Self::MiniGame => "复刻版 mini_game",
            Self::ClassicClient => "正式客户端（冒险岛怀旧服）",
        }
    }

    pub fn title_hint(self) -> &'static str {
        match self {
            Self::MiniGame => MINI_GAME_TITLE,
            Self::ClassicClient => CLASSIC_CLIENT_TITLE,
        }
    }
}

impl Default for CaptureTarget {
    fn default() -> Self {
        Self::MiniGame
    }
}

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

    pub fn short_desc(&self) -> String {
        match &self.process_path {
            Some(p) => {
                let name = Path::new(p)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(p);
                format!("「{}」 hwnd={:#x} ({name})", self.title, self.hwnd as u64)
            }
            None => format!("「{}」 hwnd={:#x}", self.title, self.hwnd as u64),
        }
    }
}

/// 兼容旧调用：只找 mini_game。
pub fn find_mini_game_window() -> Option<GameWindow> {
    find_game_window(CaptureTarget::MiniGame)
}

/// 按目标查找最佳匹配窗口。
pub fn find_game_window(target: CaptureTarget) -> Option<GameWindow> {
    let found = list_candidate_windows(target);
    match target {
        CaptureTarget::MiniGame => {
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
        CaptureTarget::ClassicClient => {
            let by_title = found
                .iter()
                .find(|w| w.title.contains(CLASSIC_CLIENT_TITLE))
                .cloned();
            by_title
                .or_else(|| {
                    found.iter().find(|w| {
                        w.title.contains("冒险岛")
                            && w.process_path
                                .as_deref()
                                .map(process_looks_like_classic)
                                .unwrap_or(false)
                    }).cloned()
                })
                .or_else(|| {
                    found.into_iter().find(|w| {
                        w.process_path
                            .as_deref()
                            .map(process_looks_like_classic)
                            .unwrap_or(false)
                    })
                })
        }
    }
}

/// 列出当前目标下所有候选窗口（供 UI 展示）。
pub fn list_candidate_windows(target: CaptureTarget) -> Vec<GameWindow> {
    let mut found: Vec<GameWindow> = Vec::new();
    let mut ctx = EnumCtx {
        target,
        out: &mut found,
    };
    unsafe {
        let _ = EnumWindows(Some(enum_callback), LPARAM(&mut ctx as *mut EnumCtx as isize));
    }
    found
}

struct EnumCtx<'a> {
    target: CaptureTarget,
    out: &'a mut Vec<GameWindow>,
}

unsafe extern "system" fn enum_callback(hwnd: HWND, lparam: LPARAM) -> BOOL {
    let ctx = &mut *(lparam.0 as *mut EnumCtx);
    if !IsWindowVisible(hwnd).as_bool() || IsIconic(hwnd).as_bool() {
        return BOOL(1);
    }
    let title = window_title(hwnd);
    if title.is_empty() {
        return BOOL(1);
    }
    let process_path = process_image_path(hwnd);
    let hit = match ctx.target {
        CaptureTarget::MiniGame => {
            title.contains(MINI_GAME_TITLE)
                || process_path
                    .as_deref()
                    .map(process_looks_like_mini_game)
                    .unwrap_or(false)
        }
        CaptureTarget::ClassicClient => {
            title.contains(CLASSIC_CLIENT_TITLE)
                || (title.contains("冒险岛")
                    && process_path
                        .as_deref()
                        .map(process_looks_like_classic)
                        .unwrap_or(false))
                || process_path
                    .as_deref()
                    .map(process_looks_like_classic)
                    .unwrap_or(false)
        }
    };
    if hit {
        ctx.out.push(GameWindow {
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
        .map(|s| s.eq_ignore_ascii_case(MINI_GAME_PROCESS))
        .unwrap_or(false)
}

fn process_looks_like_classic(path: &str) -> bool {
    let p = Path::new(path);
    let stem_ok = p
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| {
            s.eq_ignore_ascii_case(CLASSIC_CLIENT_PROCESS)
                || s.to_ascii_lowercase().contains("maplestory")
        })
        .unwrap_or(false);
    if stem_ok {
        return true;
    }
    let lower = path.replace('/', "\\").to_ascii_lowercase();
    lower.contains("mxdclassic") || lower.contains("冒险岛online")
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

/// 尽量把游戏窗口拉到前台，便于按键打到正确目标。
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
    // Unity/GL：优先 PrintWindow；失败再 BitBlt 客户区 / 屏幕坐标。
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
