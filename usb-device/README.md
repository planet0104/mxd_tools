# usb-device

ESP32-S2 纯 Rust（`no_std` / bare-metal）USB 复合设备固件。

电脑枚举后同时提供：

- USB HID 键盘
- USB HID 鼠标
- USB CDC 虚拟串口

通过串口发送文本命令，即可模拟键盘按键与鼠标移动/按键/滚轮。

> 不依赖 ESP-IDF。基于 [`esp-hal`](https://github.com/esp-rs/esp-hal) + [`usb-device`](https://crates.io/crates/usb-device) + [`usbd-hid`](https://crates.io/crates/usbd-hid) + [`usbd-serial`](https://crates.io/crates/usbd-serial)。

本目录是独立 Cargo 工程，与上级 `mxd_tools` **互不关联**：各自有独立的 `Cargo.toml`、`target/` 与 `.cargo/config.toml`。请始终在本目录内编译/烧录。

## 硬件

| 信号 | ESP32-S2 引脚 |
|------|----------------|
| USB D+ | GPIO20 |
| USB D- | GPIO19 |

请使用板载 **USB-OTG / native USB** 口接到电脑，不要用 UART 下载口当 HID/CDC 设备口。

VID/PID：`0x303A` / `0x4002`（Espressif VID + 自定义 PID）。

## 开发环境

1. 安装 [espup](https://github.com/esp-rs/espup) 并安装 Espressif Rust 工具链：

   ```powershell
   cargo install espup --locked
   espup install
   # Windows 还需按提示执行 export 脚本，或确保 rustup 能使用 channel = "esp"
   ```

2. 安装烧录工具：

   ```powershell
   cargo install espflash --locked
   ```

3. 本工程已通过 `rust-toolchain.toml` 指定 `channel = "esp"`，进入目录后 cargo 会自动选用。

## 编译与烧录

```powershell
cd usb-device
cargo build
cargo run          # 等价于编译 + espflash flash --monitor
cargo build --release
```

首次接入电脑时，系统应出现键盘、鼠标各一个，以及一个 CDC 串口。用任意串口工具（115200 波特率通常即可，CDC 实际由 USB 驱动）打开该串口。

连接成功后设备会回复：

```text
ESP32-S2 USB HID+CDC ready. Type help
```

## 串口命令协议

- 一行一条命令，以 `\n` 或 `\r` 结束
- 数字支持十进制或 `0x` 十六进制
- 成功回复 `OK`，失败回复 `ERR ...`
- 发送 `help` 可查看内置说明

### 通用

| 命令 | 说明 |
|------|------|
| `help` / `?` | 打印命令帮助 |
| `ping` | 回复 `pong` |

### 键盘

| 命令 | 说明 |
|------|------|
| `kb <mod> <k0> <k1> <k2> <k3> <k4> <k5>` | 发送完整键盘报告（Boot 协议，最多 6 键） |
| `km <mod>` | 只设置修饰键 |
| `kd <code>` | 按下键（加入当前按下集合） |
| `ku <code>` | 抬起键 |
| `kp <code>` | 单击（按下再抬起） |
| `kc` | 清空所有按键与修饰键 |
| `type <text>` | 按 ASCII 逐字输入（自动处理 Shift） |

修饰键 `<mod>` 可为十六进制掩码，或名称组合（`+` / `|` / `,` 连接）：

`lctrl` `lshift` `lalt` `lgui` `rctrl` `rshift` `ralt` `rgui`  
（也可用 `ctrl` / `shift` / `alt` / `gui` / `win` / `cmd`）

按键 `<code>` 为 USB HID Keyboard Usage ID，例如：

| 键 | Usage |
|----|-------|
| A | `0x04` |
| Enter | `0x28` |
| Esc | `0x29` |
| Backspace | `0x2A` |
| Tab | `0x2B` |
| Space | `0x2C` |

完整表见 [HID Usage Tables](https://usb.org/sites/default/files/hut1_5.pdf) 第 10 章。

示例：

```text
kp 0x04
type Hello, world!
km lctrl
kd 0x06
ku 0x06
kc
```

### 鼠标

| 命令 | 说明 |
|------|------|
| `ms <btn> <x> <y> <wheel> [pan]` | 完整鼠标报告 |
| `mm <dx> <dy>` | 相对移动（-127..127） |
| `md <btn>` | 按下按键 |
| `mu <btn>` | 抬起按键 |
| `mc <btn>` | 单击 |
| `mw <delta>` | 垂直滚轮（正数向上） |
| `mp <delta>` | 水平平移 |
| `m0` | 松开全部鼠标按键 |

`<btn>` 可为：

| 写法 | 含义 |
|------|------|
| `l` / `left` / `1` | 左键 |
| `r` / `right` / `2` | 右键 |
| `m` / `middle` / `3` | 中键 |
| `4` / `back` | 侧键 4 |
| `5` / `forward` | 侧键 5 |
| 数字掩码 | 直接按 bit 掩码（左=1, 右=2, 中=4, …） |

示例：

```text
mm 20 -10
mc left
mw 1
md right
mu right
m0
```

## 目录结构

```text
usb-device/
├── Cargo.toml
├── rust-toolchain.toml      # channel = "esp"
├── .cargo/config.toml       # xtensa-esp32s2-none-elf + espflash
├── build.rs
└── src/
    ├── lib.rs
    ├── ascii.rs             # ASCII → HID 键码
    ├── cmd.rs               # CDC 命令解析
    ├── state.rs             # 键盘/鼠标状态
    └── bin/main.rs          # USB 复合设备主循环
```

## 注意

- 首次枚举若串口驱动异常，可尝试重新插拔，或确认系统识别为复合设备（设备类 `0xEF` + IAD）。
- `type` / `kp` / `mc` 等命令执行期间会持续 poll USB，避免总线超时。
- 本工程与上级 `mxd_tools` 的 OpenCV / LLVM 等主机环境变量已在本目录 `.cargo/config.toml` 中显式 `unset`，避免互相干扰。
