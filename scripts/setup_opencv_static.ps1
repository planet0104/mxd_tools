# 安装静态 OpenCV（vcpkg x64-windows-static）供 opencv-rust 链接。
# 首次可能需要 1～3 小时，请保持网络畅通。
#
# 用法（在 mxd_tools 目录）:
#   powershell -ExecutionPolicy Bypass -File scripts/setup_opencv_static.ps1

$ErrorActionPreference = "Stop"
$VcpkgRoot = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { Join-Path $env:USERPROFILE "vcpkg" }
$Triplet = "x64-windows-static"

Write-Host "==> vcpkg root: $VcpkgRoot"

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

$llvm = "C:\Program Files\LLVM\bin"
if (-not (Test-Path (Join-Path $llvm "libclang.dll"))) {
    Write-Host "WARN: 未找到 $llvm\libclang.dll，请先安装 LLVM（winget install LLVM.LLVM）"
}

Write-Host ""
Write-Host "安装完成。请确认 .cargo/config.toml 中路径与 OPENCV_LINK_LIBS 与上面 lib 文件名一致。"
Write-Host "然后: cargo build --release"
Write-Host "可用: powershell -File scripts/list_opencv_libs.ps1"
