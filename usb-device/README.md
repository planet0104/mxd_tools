# mxd-usb-hid（usb-device）

RP2040 纯 Rust（`no_std` / bare-metal）USB 复合设备固件。

电脑枚举后同时提供：

- USB HID 键盘
- USB HID 鼠标
- USB CDC 虚拟串口

通过串口发送文本命令，即可模拟键盘按键与鼠标移动/按键/滚轮。

> 基于 [Embassy](https://github.com/embassy-rs/embassy)（`embassy-rp` + `embassy-usb`）+ [`usbd-hid`](https://crates.io/crates/usbd-hid)。

本目录是独立 Cargo 工程，与上级 `mxd_tools` **互不关联**：各自有独立的 `Cargo.toml`、`target/` 与 `.cargo/config.toml`。请始终在本目录内编译/烧录。

## 硬件

| 项目 | 说明 |
|------|------|
| MCU | RP2040（Raspberry Pi Pico 等） |
| USB | 板载 USB 口直连电脑（Pico 的 USB 口，非 UART 调试口） |

VID/PID：`0x2E8A` / `0x4002`（Raspberry Pi 官方 VID + 自定义 PID）。

## 开发环境

1. 安装 Rust stable 与 RP2040 目标：

   ```powershell
   rustup target add thumbv6m-none-eabi
   rustup component add rust-src
   ```

   本工程已通过 `rust-toolchain.toml` 自动选用 stable + `thumbv6m-none-eabi`。

2. 烧录工具（二选一）：

   ```powershell
   # probe-rs（默认 runner，适合 SWD 调试器）
   cargo install probe-rs-tools --locked

   # 或生成 UF2 拖拽到 Pico（需改 .cargo/config.toml 的 runner）
   cargo install elf2uf2-rs --locked
   ```

## 编译与烧录

```powershell
cd usb-device
cargo build
cargo build --release
cargo run          # probe-rs 编译并烧录
```

若使用 Pico 拖拽 UF2，将 `.cargo/config.toml` 中 runner 改为：

```toml
runner = "elf2uf2-rs -d"
```

然后 `cargo run --release`，把生成的 UF2 拖入 Pico 的 U 盘。

首次接入电脑时，系统应出现键盘、鼠标各一个，以及一个 CDC 串口。用任意串口工具打开该 CDC 口（波特率设置通常不影响 CDC，选 115200 即可）。

连接成功后设备会回复：

```text
RP2040 USB HID+CDC ready. Type help
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
├── rust-toolchain.toml      # stable + thumbv6m-none-eabi
├── .cargo/config.toml       # probe-rs / elf2uf2 runner
└── src/
    ├── lib.rs
    ├── ascii.rs             # ASCII → HID 键码
    ├── cmd.rs               # 命令解析辅助
    ├── io.rs                # Embassy USB 命令执行
    ├── state.rs             # 键盘/鼠标状态
    └── bin/main.rs          # USB 复合设备主循环
```

## 注意

- Pico 请用 **USB 口**（GPIO 复用 USB），不要用 3-pin UART 口当 HID 设备。
- 首次枚举若串口驱动异常，可尝试重新插拔。
- `type` / `kp` / `mc` 等命令执行期间 Embassy 异步栈持续运行，避免 USB 超时。
