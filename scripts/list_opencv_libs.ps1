# 列出 vcpkg 静态 OpenCV 的 .lib，便于填写 OPENCV_LINK_LIBS
$VcpkgRoot = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { Join-Path $env:USERPROFILE "vcpkg" }
$lib = Join-Path $VcpkgRoot "installed\x64-windows-static\lib"
if (-not (Test-Path $lib)) {
    Write-Host "未找到 $lib ，请先运行 setup_opencv_static.ps1"
    exit 1
}
Get-ChildItem $lib -Filter "*.lib" | Sort-Object Name | ForEach-Object {
    $n = $_.BaseName
    Write-Host $n
}
Write-Host ""
Write-Host "建议 OPENCV_LINK_LIBS 以 opencv_ 开头的模块为主，并加上 zlib/libpng 等依赖。"
