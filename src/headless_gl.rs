//! macroquad / miniquad 训练用 headless 初始化：1×1 隐藏窗 + 关 VSync。
//!
//! WGL 仍需要绑定 HWND，无法完全无窗；此处把占位窗隐藏并从任务栏移除，
//! 实际 1368×768 画面仍由 `game::view` 的离屏 `RenderTarget` 产出。
//! 可视化请用 `game_preview`。

use macroquad::miniquad::conf::{Conf, Platform};

/// miniquad 在 Windows 上注册的窗口类名（固定值）。
const MINIQUAD_CLASS: &str = "MINIQUADAPP";

/// 训练 / 截图 headless 用的 macroquad 窗口配置。
pub fn headless_window_conf(window_title: impl Into<String>) -> Conf {
    Conf {
        window_title: window_title.into(),
        window_width: 1,
        window_height: 1,
        window_resizable: false,
        high_dpi: false,
        icon: None,
        platform: Platform {
            swap_interval: Some(0),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// 在 `#[macroquad::main]` 进入 `async fn main` 后立刻调用：隐藏 GL 占位窗。
pub fn hide_gl_window() -> bool {
    #[cfg(windows)]
    {
        return hide_gl_window_windows();
    }
    #[cfg(not(windows))]
    {
        macroquad::miniquad::window::set_window_position(32_000, 32_000);
        true
    }
}

#[cfg(windows)]
fn hide_gl_window_windows() -> bool {
    use windows::Win32::Foundation::{FALSE, HWND, LPARAM, TRUE};
    use windows::Win32::System::Threading::GetCurrentProcessId;
    use windows::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetClassNameW, GetWindowLongPtrW, GetWindowThreadProcessId, SetWindowLongPtrW,
        SetWindowPos, ShowWindow, GWL_EXSTYLE, SWP_FRAMECHANGED, SWP_NOACTIVATE, SWP_NOMOVE,
        SWP_NOSIZE, SWP_NOZORDER, SW_HIDE, WS_EX_APPWINDOW, WS_EX_TOOLWINDOW,
    };

    struct EnumCtx {
        pid: u32,
        hwnd: HWND,
    }

    unsafe extern "system" fn enum_miniquad_window(
        hwnd: HWND,
        lparam: LPARAM,
    ) -> windows::core::BOOL {
        let ctx = &mut *(lparam.0 as *mut EnumCtx);
        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        if pid != ctx.pid {
            return TRUE;
        }

        let mut class_buf = [0u16; 32];
        let len = GetClassNameW(hwnd, &mut class_buf);
        if len == 0 {
            return TRUE;
        }
        let class_name = String::from_utf16_lossy(&class_buf[..len as usize]);
        if class_name != MINIQUAD_CLASS {
            return TRUE;
        }

        ctx.hwnd = hwnd;
        FALSE
    }

    let mut ctx = EnumCtx {
        pid: unsafe { GetCurrentProcessId() },
        hwnd: HWND::default(),
    };
    unsafe {
        let _ = EnumWindows(
            Some(enum_miniquad_window),
            LPARAM(&mut ctx as *mut EnumCtx as isize),
        );
    }

    if ctx.hwnd.0.is_null() {
        return false;
    }

    unsafe {
        let _ = ShowWindow(ctx.hwnd, SW_HIDE);
        let ex = GetWindowLongPtrW(ctx.hwnd, GWL_EXSTYLE) as u32;
        let next = (ex | WS_EX_TOOLWINDOW.0) & !WS_EX_APPWINDOW.0;
        SetWindowLongPtrW(ctx.hwnd, GWL_EXSTYLE, next as isize);
        let _ = SetWindowPos(
            ctx.hwnd,
            None,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        );
    }
    true
}
