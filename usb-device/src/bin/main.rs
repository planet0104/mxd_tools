//! RP2040 USB 复合设备固件入口。
//!
//! 枚举为：HID 键盘 + HID 鼠标 + CDC 串口。
//! 主机通过 CDC 发命令，本程序据此发送 HID 报告。
//!
//! 安全：CDC 断开或长时间无命令时自动松开按键，避免宿主崩溃/关窗导致卡键。

#![no_std]
#![no_main]

use embassy_executor::Spawner;
use embassy_futures::join::join;
use embassy_futures::select::{select, Either};
use embassy_rp::bind_interrupts;
use embassy_rp::peripherals::USB;
use embassy_rp::usb::{Driver, InterruptHandler};
use embassy_time::{Duration, Instant, Timer};
use embassy_usb::class::cdc_acm::{CdcAcmClass, State as CdcState};
use embassy_usb::class::hid::{HidBootProtocol, HidSubclass, HidWriter, State as HidState};
use embassy_usb::driver::EndpointError;
use embassy_usb::{Builder, Config};
use mxd_usb_hid::io::{handle_line, push_keyboard, push_mouse, write_cdc};
use mxd_usb_hid::state::{KeyboardState, MouseState};
use static_cell::StaticCell;
use usbd_hid::descriptor::{KeyboardReport, MouseReport, SerializedDescriptor};

use panic_halt as _;

bind_interrupts!(struct Irqs {
    USBCTRL_IRQ => InterruptHandler<USB>;
});

/// 宿主按住键时只发边沿，不会持续刷命令；因此需 keepalive。
/// 超过该空闲时间仍有按键按下 → 强制松开（宿主应 ≤1s ping 一次）。
const HOLD_WATCHDOG: Duration = Duration::from_millis(2500);

async fn release_all_inputs<'a>(
    keyboard: &mut HidWriter<'a, Driver<'a, USB>, 8>,
    mouse: &mut HidWriter<'a, Driver<'a, USB>, 8>,
    kb: &mut KeyboardState,
    ms: &mut MouseState,
) {
    if kb.is_held() {
        kb.clear();
        push_keyboard(keyboard, kb).await;
    }
    if ms.buttons_held() {
        ms.clear_buttons();
        ms.clear_motion();
        push_mouse(mouse, ms).await;
    }
}

async fn cmd_loop<'a>(
    mut cdc: CdcAcmClass<'a, Driver<'a, USB>>,
    mut keyboard: HidWriter<'a, Driver<'a, USB>, 8>,
    mut mouse: HidWriter<'a, Driver<'a, USB>, 8>,
) {
    let mut kb_state = KeyboardState::new();
    let mut ms_state = MouseState::new();
    let mut line_buf = [0u8; 128];

    loop {
        cdc.wait_connection().await;
        // 新连接：确保上一会话残留按键已松开
        release_all_inputs(&mut keyboard, &mut mouse, &mut kb_state, &mut ms_state).await;
        let mut line_len = 0usize;
        write_cdc(&mut cdc, "RP2040 USB HID+CDC ready. Type help\r\n").await;

        let mut pkt = [0u8; 64];
        let mut last_rx = Instant::now();
        loop {
            match select(
                cdc.read_packet(&mut pkt),
                Timer::after(Duration::from_millis(100)),
            )
            .await
            {
                Either::First(Ok(n)) => {
                    last_rx = Instant::now();
                    for &b in &pkt[..n] {
                        if b == b'\n' || b == b'\r' {
                            if line_len > 0 {
                                let line =
                                    core::str::from_utf8(&line_buf[..line_len]).unwrap_or("");
                                handle_line(
                                    &mut keyboard,
                                    &mut mouse,
                                    &mut cdc,
                                    line,
                                    &mut kb_state,
                                    &mut ms_state,
                                )
                                .await;
                                line_len = 0;
                            }
                        } else if line_len < line_buf.len() {
                            line_buf[line_len] = b;
                            line_len += 1;
                        } else {
                            line_len = 0;
                        }
                    }
                }
                Either::First(Err(EndpointError::Disabled)) => {
                    // 宿主关串口 / 杀进程：立刻松开，避免卡键
                    release_all_inputs(&mut keyboard, &mut mouse, &mut kb_state, &mut ms_state)
                        .await;
                    break;
                }
                Either::First(Err(EndpointError::BufferOverflow)) => continue,
                Either::Second(_) => {
                    if (kb_state.is_held() || ms_state.buttons_held())
                        && last_rx.elapsed() >= HOLD_WATCHDOG
                    {
                        release_all_inputs(
                            &mut keyboard,
                            &mut mouse,
                            &mut kb_state,
                            &mut ms_state,
                        )
                        .await;
                    }
                }
            }
        }
    }
}

#[embassy_executor::main(
    executor = "embassy_rp::executor::Executor",
    entry = "cortex_m_rt::entry"
)]
async fn main(_spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    let driver = Driver::new(p.USB, Irqs);

    let config = {
        let mut config = Config::new(0x2E8A, 0x4002);
        config.manufacturer = Some("mxd-tools");
        config.product = Some("RP2040 HID+CDC");
        config.serial_number = Some("0001");
        config.max_power = 100;
        config.max_packet_size_0 = 64;
        config
    };

    static CONFIG_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static BOS_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static CONTROL_BUF: StaticCell<[u8; 64]> = StaticCell::new();
    static CDC_STATE: StaticCell<CdcState> = StaticCell::new();
    static KB_HID_STATE: StaticCell<HidState> = StaticCell::new();
    static MOUSE_HID_STATE: StaticCell<HidState> = StaticCell::new();

    let mut builder = Builder::new(
        driver,
        config,
        CONFIG_DESC.init([0; 256]),
        BOS_DESC.init([0; 256]),
        &mut [],
        CONTROL_BUF.init([0; 64]),
    );

    let cdc = CdcAcmClass::new(&mut builder, CDC_STATE.init(CdcState::new()), 64);

    let kb_config = embassy_usb::class::hid::Config {
        report_descriptor: KeyboardReport::desc(),
        request_handler: None,
        poll_ms: 10,
        max_packet_size: 64,
        hid_subclass: HidSubclass::Boot,
        hid_boot_protocol: HidBootProtocol::Keyboard,
    };
    let keyboard = HidWriter::new(&mut builder, KB_HID_STATE.init(HidState::new()), kb_config);

    let mouse_config = embassy_usb::class::hid::Config {
        report_descriptor: MouseReport::desc(),
        request_handler: None,
        poll_ms: 10,
        max_packet_size: 64,
        hid_subclass: HidSubclass::Boot,
        hid_boot_protocol: HidBootProtocol::Mouse,
    };
    let mouse = HidWriter::new(&mut builder, MOUSE_HID_STATE.init(HidState::new()), mouse_config);

    let mut usb = builder.build();
    join(usb.run(), cmd_loop(cdc, keyboard, mouse)).await;
}
