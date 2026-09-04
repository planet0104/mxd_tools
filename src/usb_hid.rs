//! RP2040 `usb-device` CDC 串口客户端（VID `0x2E8A` / PID `0x4002`）。
//!
//! 协议见 `usb-device/README.md`：一行命令，成功回 `OK`。

use std::io::{Read, Write};
use std::time::{Duration, Instant};

use serialport::{SerialPort, SerialPortType, UsbPortInfo};

pub const USB_VID: u16 = 0x2E8A;
pub const USB_PID: u16 = 0x4002;

const CMD_TIMEOUT: Duration = Duration::from_millis(800);
const OPEN_BANNER_WAIT: Duration = Duration::from_millis(200);

pub struct UsbHidClient {
    port: Box<dyn SerialPort>,
    read_buf: Vec<u8>,
}

impl UsbHidClient {
    /// 自动查找 VID/PID 匹配的 CDC 口；若无 USB 元数据则尝试打开候选口并 `ping`。
    pub fn open_auto() -> Result<Self, String> {
        let ports = list_device_ports();
        if ports.is_empty() {
            return Err(format!(
                "未找到 RP2040 USB 设备（期望 VID={USB_VID:#06x} PID={USB_PID:#06x}）。请插入 Pico 并确认已烧录 usb-device 固件"
            ));
        }
        let mut last_err = String::new();
        for name in ports {
            match Self::open_named(&name) {
                Ok(mut c) => match c.ping() {
                    Ok(()) => return Ok(c),
                    Err(e) => last_err = format!("{name}: ping 失败：{e}"),
                },
                Err(e) => last_err = format!("{name}: {e}"),
            }
        }
        Err(format!("打开 USB CDC 失败：{last_err}"))
    }

    pub fn open_named(name: &str) -> Result<Self, String> {
        let port = serialport::new(name, 115_200)
            .timeout(CMD_TIMEOUT)
            .open()
            .map_err(|e| format!("打开串口 {name} 失败：{e}"))?;
        let mut client = Self {
            port,
            read_buf: Vec::with_capacity(256),
        };
        let _ = client.drain_pending(OPEN_BANNER_WAIT);
        Ok(client)
    }

    pub fn port_name(&self) -> String {
        self.port.name().unwrap_or_else(|| "(unknown)".into())
    }

    pub fn ping(&mut self) -> Result<(), String> {
        let resp = self.transact("ping")?;
        if resp.to_ascii_lowercase().contains("pong") || resp.eq_ignore_ascii_case("ok") {
            Ok(())
        } else {
            Err(format!("意外 ping 回复：{resp}"))
        }
    }

    pub fn key_down(&mut self, hid: u8) -> Result<(), String> {
        self.expect_ok(&format!("kd 0x{hid:02X}"))
    }

    pub fn key_up(&mut self, hid: u8) -> Result<(), String> {
        self.expect_ok(&format!("ku 0x{hid:02X}"))
    }

    pub fn set_modifier_mask(&mut self, mask: u8) -> Result<(), String> {
        self.expect_ok(&format!("km 0x{mask:02X}"))
    }

    pub fn clear_keys(&mut self) -> Result<(), String> {
        self.expect_ok("kc")
    }

    /// 保活：固件按住看门狗依赖定期收包（不必 OK）。
    pub fn keepalive(&mut self) -> Result<(), String> {
        let _ = self.transact("ping");
        Ok(())
    }

    fn expect_ok(&mut self, cmd: &str) -> Result<(), String> {
        let resp = self.transact(cmd)?;
        if resp.eq_ignore_ascii_case("ok") || resp.to_ascii_uppercase().starts_with("OK") {
            Ok(())
        } else if resp.to_ascii_uppercase().starts_with("ERR") {
            Err(resp)
        } else {
            Err(format!("命令 `{cmd}` 意外回复：{resp}"))
        }
    }

    fn transact(&mut self, cmd: &str) -> Result<String, String> {
        let _ = self.drain_pending(Duration::from_millis(20));
        let line = if cmd.ends_with('\n') {
            cmd.to_string()
        } else {
            format!("{cmd}\n")
        };
        self.port
            .write_all(line.as_bytes())
            .map_err(|e| format!("写串口失败：{e}"))?;
        self.port.flush().map_err(|e| format!("flush 失败：{e}"))?;
        self.read_response_line(CMD_TIMEOUT)
    }

    fn drain_pending(&mut self, max_wait: Duration) -> Result<(), String> {
        let deadline = Instant::now() + max_wait;
        let prev = self.port.timeout();
        let _ = self.port.set_timeout(Duration::from_millis(30));
        while Instant::now() < deadline {
            let mut tmp = [0u8; 64];
            match self.port.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => self.read_buf.extend_from_slice(&tmp[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => break,
                Err(_) => break,
            }
        }
        let _ = self.port.set_timeout(prev);
        // 丢掉横幅，只保留未完成行
        if let Some(pos) = self.read_buf.iter().rposition(|&b| b == b'\n') {
            self.read_buf.drain(..=pos);
        }
        Ok(())
    }

    fn read_response_line(&mut self, timeout: Duration) -> Result<String, String> {
        let deadline = Instant::now() + timeout;
        let prev = self.port.timeout();
        let _ = self.port.set_timeout(Duration::from_millis(50));
        loop {
            if let Some(line) = self.take_line() {
                let _ = self.port.set_timeout(prev);
                let t = line.trim();
                // 跳过固件欢迎语
                if t.is_empty()
                    || t.to_ascii_lowercase().contains("ready")
                    || t.to_ascii_lowercase().starts_with("rp2040")
                {
                    if Instant::now() >= deadline {
                        return Err("读回复超时（仅收到横幅）".into());
                    }
                    continue;
                }
                return Ok(t.to_string());
            }
            if Instant::now() >= deadline {
                let _ = self.port.set_timeout(prev);
                return Err("读串口回复超时".into());
            }
            let mut tmp = [0u8; 64];
            match self.port.read(&mut tmp) {
                Ok(n) if n > 0 => self.read_buf.extend_from_slice(&tmp[..n]),
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => {
                    let _ = self.port.set_timeout(prev);
                    return Err(format!("读串口失败：{e}"));
                }
            }
        }
    }

    fn take_line(&mut self) -> Option<String> {
        let pos = self.read_buf.iter().position(|&b| b == b'\n')?;
        let raw: Vec<u8> = self.read_buf.drain(..=pos).collect();
        let s = String::from_utf8_lossy(&raw).to_string();
        Some(s.trim_end_matches(['\r', '\n']).to_string())
    }
}

/// 列出可能的设备串口名（优先匹配 VID/PID）。
pub fn list_device_ports() -> Vec<String> {
    let Ok(ports) = serialport::available_ports() else {
        return Vec::new();
    };
    let mut matched = Vec::new();
    let mut others = Vec::new();
    for p in ports {
        match &p.port_type {
            SerialPortType::UsbPort(UsbPortInfo { vid, pid, .. })
                if *vid == USB_VID && *pid == USB_PID =>
            {
                matched.push(p.port_name);
            }
            SerialPortType::UsbPort(_) => others.push(p.port_name),
            _ => {}
        }
    }
    if !matched.is_empty() {
        matched
    } else {
        others
    }
}

/// 供 UI 展示的串口列表（全部可用口，匹配设备排前）。
pub fn list_ports_for_ui() -> Vec<(String, bool)> {
    let Ok(ports) = serialport::available_ports() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for p in ports {
        let is_dev = matches!(
            &p.port_type,
            SerialPortType::UsbPort(UsbPortInfo { vid, pid, .. })
                if *vid == USB_VID && *pid == USB_PID
        );
        out.push((p.port_name, is_dev));
    }
    out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}
