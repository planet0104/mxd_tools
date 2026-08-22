# 已弃用：CPU 推理改为静态链接 ort，不再需要 onnxruntime.dll。
# 见 docs/如何在Rust中做YOLO推理.md
#
# 仅在调试 load-dynamic 或旧脚本时保留。若仍要下载官方 DLL：
#   powershell -ExecutionPolicy Bypass -File scripts/fetch_onnxruntime.ps1

$ErrorActionPreference = "Stop"
Write-Host "提示: 当前工程默认静态链接 ONNX Runtime，部署 CPU 推理一般不需要本脚本。"
Write-Host "若仍要下载 DLL 用于特殊场景，继续执行..."

$Version = if ($env:ORT_VERSION) { $env:ORT_VERSION } else { "1.22.0" }
$Url = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/onnxruntime-win-x64-$Version.zip"
$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root "third_party\onnxruntime"
$Zip = Join-Path $env:TEMP "onnxruntime-win-x64-$Version.zip"

Write-Host "download $Url"
Invoke-WebRequest -Uri $Url -OutFile $Zip
if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
New-Item -ItemType Directory -Path $Dest | Out-Null
Expand-Archive -Path $Zip -DestinationPath $Dest -Force

$Dll = Get-ChildItem -Path $Dest -Recurse -Filter "onnxruntime.dll" | Select-Object -First 1
if (-not $Dll) { throw "zip 内找不到 onnxruntime.dll" }
Write-Host "extracted $($Dll.FullName)"

$Release = Join-Path $Root "target\release"
New-Item -ItemType Directory -Path $Release -Force | Out-Null
Get-ChildItem $Dll.Directory -Filter "*.dll" | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $Release $_.Name) -Force
    Write-Host "copied $($_.Name) -> $Release"
}
