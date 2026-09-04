# 构建 RP2040 UF2 并写入 mxd_tools/firmware/mxd-usb-hid.uf2
# 依赖：rustup target thumbv6m-none-eabi、rust-src、elf2uf2-rs
#
#   powershell -ExecutionPolicy Bypass -File scripts/build_rp2040_uf2.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Usb = Join-Path $Root "usb-device"
$OutDir = Join-Path $Root "firmware"
$Uf2 = Join-Path $OutDir "mxd-usb-hid.uf2"
$Elf = Join-Path $Usb "target\thumbv6m-none-eabi\release\usb-device"

Write-Host "==> build usb-device (release)"
Push-Location $Usb
try {
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
} finally {
    Pop-Location
}

if (-not (Test-Path $Elf)) {
    throw "找不到 ELF: $Elf"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "==> elf2uf2-rs -> $Uf2"
& elf2uf2-rs $Elf $Uf2
if ($LASTEXITCODE -ne 0) { throw "elf2uf2-rs failed" }

$item = Get-Item $Uf2
Write-Host ("OK: {0} ({1} bytes)" -f $item.FullName, $item.Length)
