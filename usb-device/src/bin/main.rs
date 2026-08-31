//! ESP32-S2 USB 复合设备固件入口。
//!
//! 枚举为：HID 键盘 + HID 鼠标 + CDC 串口。
//! 主机通过 CDC 发命令，本程序据此发送 HID 报告。
//!
//! 接线（USB-OTG）：DP = GPIO20，DM = GPIO19。

#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

use core::ptr::addr_of_mut;

use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::otg_fs::{Usb, UsbBus};
use esp_hal::time::{Duration, Instant};
use usb_device::cmd::{self, DeviceIo};
use usb_device::state::{KeyboardState, MouseState};
use usb_device_stack::prelude::{
    StringDescriptors, UsbDevice, UsbDeviceBuilder, UsbDeviceState, UsbVidPid,
};
use usbd_hid::descriptor::{KeyboardReport, MouseReport, SerializedDescriptor};
use usbd_hid::hid_class::{HidClassSettings, HidProtocol, HidSubClass, HIDClass};
use usbd_serial::SerialPort;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// 生成 esp-idf 兼容的 app descriptor，供官方 bootloader 校验镜像。
esp_bootloader_esp_idf::esp_app_desc!();

/// Synopsys OTG 端点缓冲（单位：字）。复合设备接口较多，给足余量。
static mut EP_MEMORY: [u32; 2048] = [0; 2048];

/// 把 USB 设备各接口捆在一起，供命令层在延时中持续 poll / 收发。
///
/// `'a`：本次短借用；`'d`：设备对象自身生命周期。
struct UsbIo<'a, 'd, B>
where
    B: usb_device_stack::bus::UsbBus,
{
    usb_dev: &'a mut UsbDevice<'d, B>,
    keyboard: &'a mut HIDClass<'d, B>,
    mouse: &'a mut HIDClass<'d, B>,
    serial: &'a mut SerialPort<'d, B>,
}

impl<B> DeviceIo for UsbIo<'_, '_, B>
where
    B: usb_device_stack::bus::UsbBus,
{
    fn poll_usb(&mut self) {
        self.usb_dev
            .poll(&mut [self.keyboard, self.mouse, self.serial]);
    }

    fn push_keyboard(&mut self, kb: &KeyboardState) -> bool {
        matches!(self.keyboard.push_input(&kb.report()), Ok(_))
    }

    fn push_mouse(&mut self, ms: &MouseState) -> bool {
        matches!(self.mouse.push_input(&ms.report()), Ok(_))
    }

    fn write_reply(&mut self, msg: &str) {
        let bytes = msg.as_bytes();
        let mut off = 0;
        let mut spins = 0;
        // CDC 写缓冲可能暂时满，边 poll 边重试。
        while off < bytes.len() && spins < 10_000 {
            self.poll_usb();
            match self.serial.write(&bytes[off..]) {
                Ok(n) if n > 0 => {
                    off += n;
                    spins = 0;
                }
                _ => {
                    spins += 1;
                }
            }
        }
    }

    fn delay_ms(&mut self, ms: u32) {
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(ms as u64) {
            self.poll_usb();
        }
    }
}

#[allow(
    clippy::large_stack_frames,
    reason = "USB descriptor / endpoint buffers are intentionally large"
)]
#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    // ESP32-S2 片上 USB-OTG：D+=GPIO20，D-=GPIO19
    let usb = Usb::new(peripherals.USB0, peripherals.GPIO20, peripherals.GPIO19);
    let usb_bus = UsbBus::new(usb, unsafe { &mut *addr_of_mut!(EP_MEMORY) });

    // ---- CDC 串口（复合设备里放前面，便于 Windows IAD 识别）----
    let mut serial = SerialPort::new(&usb_bus);

    // ---- HID 键盘（仅 IN 端点；Boot 协议，便于多数主机直接识别）----
    let kb_settings = HidClassSettings {
        subclass: HidSubClass::Boot,
        protocol: HidProtocol::Keyboard,
        ..Default::default()
    };
    let mut keyboard = HIDClass::new_ep_in_with_settings(
        &usb_bus,
        KeyboardReport::desc(),
        10, // 主机轮询间隔 ms
        kb_settings,
    );

    // ---- HID 鼠标 ----
    let mouse_settings = HidClassSettings {
        subclass: HidSubClass::Boot,
        protocol: HidProtocol::Mouse,
        ..Default::default()
    };
    let mut mouse =
        HIDClass::new_ep_in_with_settings(&usb_bus, MouseReport::desc(), 10, mouse_settings);

    // VID=Espressif(0x303A)，PID 自定义；composite_with_iads 满足 Windows CDC 复合设备要求
    let mut usb_dev = UsbDeviceBuilder::new(&usb_bus, UsbVidPid(0x303A, 0x4002))
        .strings(&[StringDescriptors::default()
            .manufacturer("mxd-tools")
            .product("ESP32-S2 HID+CDC")
            .serial_number("0001")])
        .unwrap()
        .composite_with_iads()
        .max_packet_size_0(64)
        .unwrap()
        .build();

    let mut kb_state = KeyboardState::new();
    let mut ms_state = MouseState::new();
    // CDC 行缓冲：收到 \n/\r 后解析一整行命令
    let mut line_buf = [0u8; 128];
    let mut line_len = 0usize;
    let mut greeted = false;

    loop {
        // 无事件则继续忙等 poll
        if !usb_dev.poll(&mut [&mut keyboard, &mut mouse, &mut serial]) {
            continue;
        }

        // 主机完成配置后，主动打一行欢迎语（只发一次）
        if usb_dev.state() == UsbDeviceState::Configured && !greeted {
            let mut io = UsbIo {
                usb_dev: &mut usb_dev,
                keyboard: &mut keyboard,
                mouse: &mut mouse,
                serial: &mut serial,
            };
            io.write_reply("ESP32-S2 USB HID+CDC ready. Type help\r\n");
            greeted = true;
        }

        // 从 CDC 读数据，按行组装命令
        let mut pkt = [0u8; 64];
        match serial.read(&mut pkt) {
            Ok(n) if n > 0 => {
                for &b in &pkt[..n] {
                    if b == b'\n' || b == b'\r' {
                        if line_len > 0 {
                            let line = core::str::from_utf8(&line_buf[..line_len]).unwrap_or("");
                            let mut io = UsbIo {
                                usb_dev: &mut usb_dev,
                                keyboard: &mut keyboard,
                                mouse: &mut mouse,
                                serial: &mut serial,
                            };
                            cmd::handle_line(line, &mut io, &mut kb_state, &mut ms_state);
                            line_len = 0;
                        }
                    } else if line_len < line_buf.len() {
                        line_buf[line_len] = b;
                        line_len += 1;
                    } else {
                        // 行过长：丢弃本行，避免脏数据继续累积
                        line_len = 0;
                    }
                }
            }
            _ => {}
        }
    }
}
