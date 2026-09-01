@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

rem 切换到脚本所在目录（usb-device 工程根）
cd /d "%~dp0"

set ELF=target\thumbv6m-none-eabi\release\usb-device
set UF2=target\thumbv6m-none-eabi\release\usb-device.uf2
set BOOT_LABEL=RPI-RP2

echo.
echo [1/3] 编译 release 固件...
cargo build --release
if errorlevel 1 (
    echo.
    echo 编译失败。
    exit /b 1
)

if not exist "%ELF%" (
    echo.
    echo 找不到 ELF: %ELF%
    exit /b 1
)

where elf2uf2-rs >nul 2>&1
if errorlevel 1 (
    echo.
    echo 未找到 elf2uf2-rs，正在安装...
    cargo install elf2uf2-rs --locked
    if errorlevel 1 (
        echo.
        echo 安装 elf2uf2-rs 失败。可手动执行: cargo install elf2uf2-rs --locked
        exit /b 1
    )
)

echo.
echo [2/3] 生成 UF2...
elf2uf2-rs "%ELF%" "%UF2%"
if errorlevel 1 (
    echo.
    echo UF2 转换失败。
    exit /b 1
)

echo.
echo [3/3] 查找 RP2040 BOOTSEL 盘符（卷标 %BOOT_LABEL%）...

set BOOT_DRIVE=

rem 优先用 PowerShell 查卷标（Win10/11 更可靠）
for /f "usebackq delims=" %%d in (`powershell -NoProfile -Command "(Get-Volume -FileSystemLabel '%BOOT_LABEL%' -ErrorAction SilentlyContinue).DriveLetter"`) do (
    if not "%%d"=="" set BOOT_DRIVE=%%d:
)

rem PowerShell 失败时回退 wmic
if not defined BOOT_DRIVE (
    for /f "tokens=2 delims==" %%a in ('wmic logicaldisk where "VolumeName='%BOOT_LABEL%'" get DeviceID /value 2^>nul ^| find "="') do (
        set BOOT_DRIVE=%%a
    )
)

if not defined BOOT_DRIVE (
    echo.
    echo 未找到 %BOOT_LABEL% 盘。请确认:
    echo   1. RP2040 已按住 BOOTSEL 再插入 USB（或双击 RESET 进 BOOTSEL）
    echo   2. 资源管理器里能看到名为 %BOOT_LABEL% 的 U 盘
    echo.
    echo UF2 已生成，可手动拖入 U 盘: %UF2%
    exit /b 1
)

echo 检测到 BOOTSEL 盘: %BOOT_DRIVE%
echo 正在复制 %UF2% ...
copy /y "%UF2%" "%BOOT_DRIVE%\" >nul
if errorlevel 1 (
    echo.
    echo 复制失败。请检查 U 盘是否仍挂载。
    exit /b 1
)

echo.
echo 烧录完成。设备会自动重启并运行新固件。
echo CDC 串口连接后应收到: RP2040 USB HID+CDC ready. Type help
echo.
exit /b 0
