//! 让窗口像屏幕键盘一样：点击不夺取前台焦点，按键可打到当前活动窗口。

use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};

use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    CallWindowProcW, GetWindowLongPtrW, SetWindowLongPtrW, SetWindowPos, GWLP_WNDPROC, GWL_EXSTYLE,
    MA_NOACTIVATE, SWP_FRAMECHANGED, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER,
    WM_MOUSEACTIVATE, WNDPROC, WS_EX_NOACTIVATE,
};

static ENABLED: AtomicBool = AtomicBool::new(true);
static ORIG_WNDPROC: AtomicIsize = AtomicIsize::new(0);
static SUBCLASSED: AtomicBool = AtomicBool::new(false);

pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::SeqCst);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::SeqCst)
}

pub fn sync_from_handle(handle: &impl HasWindowHandle) {
    let Ok(wh) = handle.window_handle() else {
        return;
    };
    let RawWindowHandle::Win32(win) = wh.as_raw() else {
        return;
    };
    let hwnd = HWND(win.hwnd.get() as *mut _);
    unsafe {
        ensure_subclass(hwnd);
        apply_exstyle(hwnd, enabled());
    }
}

unsafe fn apply_exstyle(hwnd: HWND, no_activate: bool) {
    let current = GetWindowLongPtrW(hwnd, GWL_EXSTYLE) as u32;
    let next = if no_activate {
        current | WS_EX_NOACTIVATE.0
    } else {
        current & !WS_EX_NOACTIVATE.0
    };
    if next == current {
        return;
    }
    SetWindowLongPtrW(hwnd, GWL_EXSTYLE, next as isize);
    let _ = SetWindowPos(
        hwnd,
        None,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
    );
}

unsafe fn ensure_subclass(hwnd: HWND) {
    if SUBCLASSED.swap(true, Ordering::SeqCst) {
        return;
    }
    let prev = GetWindowLongPtrW(hwnd, GWLP_WNDPROC);
    ORIG_WNDPROC.store(prev, Ordering::SeqCst);
    SetWindowLongPtrW(hwnd, GWLP_WNDPROC, wnd_proc as *const () as usize as isize);
}

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if msg == WM_MOUSEACTIVATE && ENABLED.load(Ordering::SeqCst) {
        return LRESULT(MA_NOACTIVATE as isize);
    }

    let orig = ORIG_WNDPROC.load(Ordering::SeqCst);
    if orig == 0 {
        return LRESULT(0);
    }
    let prev: WNDPROC = std::mem::transmute(orig);
    CallWindowProcW(prev, hwnd, msg, wparam, lparam)
}
