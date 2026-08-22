# 安装静态 OpenCV（vcpkg）供 opencv-rust + ort 静态链接。
# 默认 triplet = x64-windows-static-md（静态 OpenCV + 动态 CRT /MD），
# 与 pyke 预编译静态 onnxruntime 一致，部署时 CPU 推理只需单个 exe。
#
# 用法（在 mxd_tools 目录）:
#   powershell -ExecutionPolicy Bypass -File scripts/setup_opencv_static.ps1
# 若仍要用旧的 /MT 全静态 CRT（与 ort 预编译包不兼容）:
#   $env:OPENCV_TRIPLET='x64-windows-static'; powershell -File scripts/setup_opencv_static.ps1

$ErrorActionPreference = "Stop"
$VcpkgRoot = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { Join-Path $env:USERPROFILE "vcpkg" }
$Triplet = if ($env:OPENCV_TRIPLET) { $env:OPENCV_TRIPLET } else { "x64-windows-static-md" }

Write-Host "==> vcpkg root: $VcpkgRoot"
Write-Host "==> triplet:    $Triplet"

if (-not (Test-Path (Join-Path $VcpkgRoot "vcpkg.exe"))) {
    if (-not (Test-Path (Join-Path $VcpkgRoot ".git"))) {
        Write-Host "==> cloning vcpkg..."
        git clone --depth 1 https://github.com/microsoft/vcpkg.git $VcpkgRoot
    }
    Write-Host "==> bootstrap..."
    & (Join-Path $VcpkgRoot "bootstrap-vcpkg.bat") -disableMetrics
}

$vcpkg = Join-Path $VcpkgRoot "vcpkg.exe"
Write-Host "==> install opencv4:$Triplet (this takes a long time)..."
& $vcpkg install "opencv4:$Triplet"
if ($LASTEXITCODE -ne 0) { throw "vcpkg install failed" }

$inc = Join-Path $VcpkgRoot "installed\$Triplet\include"
$lib = Join-Path $VcpkgRoot "installed\$Triplet\lib"
Write-Host "==> include: $inc"
Write-Host "==> lib:     $lib"
Write-Host "==> libs:"
Get-ChildItem $lib -Filter "opencv*.lib" | ForEach-Object { $_.Name }
Write-Host "==> zlib/zs:"
Get-ChildItem $lib -Filter "z*.lib" | ForEach-Object { $_.Name }

$llvm = "C:\Program Files\LLVM\bin"
if (-not (Test-Path (Join-Path $llvm "libclang.dll"))) {
    Write-Host "WARN: 未找到 $llvm\libclang.dll，请先安装 LLVM（winget install LLVM.LLVM）"
}

$crt = if ($Triplet -like "*-static-md") { "dynamic" } else { "static" }
$zlibName = if (Test-Path (Join-Path $lib "zlib.lib")) { "zlib" } elseif (Test-Path (Join-Path $lib "zs.lib")) { "zs" } else { "zlib" }

Write-Host ""
Write-Host "安装完成。请把 .cargo/config.toml 中路径改为:"
Write-Host "  OPENCV_INCLUDE_PATHS = ...\installed\$Triplet\include\opencv4"
Write-Host "  OPENCV_LINK_PATHS    = ...\installed\$Triplet\lib"
Write-Host "  OPENCV_MSVC_CRT      = $crt"
Write-Host "  OPENCV_LINK_LIBS 末尾压缩库名 = $zlibName"
if ($crt -eq "dynamic") {
    Write-Host "  不要设置 rustflags +crt-static（否则与 ort 静态库冲突）"
} else {
    Write-Host "  警告: x64-windows-static (/MT) 与 pyke ort 预编译静态库不兼容；CPU 单 exe 请用 static-md。"
}
Write-Host "然后: cargo build --release"
Write-Host "可用: powershell -File scripts/list_opencv_libs.ps1"
