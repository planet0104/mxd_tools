# Pack portable release: exe + VC runtime DLLs (no VC++ Redistributable install).
#
# Why not +crt-static?
#   pyke static onnxruntime.lib is built with /MD. Mixing with /MT (+crt-static) breaks the link.
# Bundling the redistributable DLLs next to the exe is the supported way to avoid asking users
# to install the VC++ redist separately.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/pack_release.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/pack_release.ps1 -SkipBuild

param(
    [switch]$SkipBuild,
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $OutDir) {
    $OutDir = Join-Path $Root "dist\mxd_tools"
}

if (-not $SkipBuild) {
    Write-Host "==> cargo build --release --bin mxd_tools"
    cargo build --release --bin mxd_tools
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
}

$exe = Join-Path $Root "target\release\mxd_tools.exe"
if (-not (Test-Path $exe)) {
    throw "missing $exe ; build first or omit -SkipBuild"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $exe (Join-Path $OutDir "mxd_tools.exe")

function Find-VcCrtDir {
    $roots = @()
    if ($env:ProgramFiles) { $roots += $env:ProgramFiles }
    $pf86 = ${env:ProgramFiles(x86)}
    if ($pf86) { $roots += $pf86 }

    foreach ($root in $roots) {
        foreach ($vs in @("2022", "2019")) {
            $pattern = Join-Path $root "Microsoft Visual Studio\$vs\*\VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT"
            $hit = Get-Item $pattern -ErrorAction SilentlyContinue |
                Sort-Object FullName -Descending |
                Select-Object -First 1
            if ($hit) { return $hit.FullName }
        }
    }
    return $null
}

$dllNames = @(
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "concrt140.dll"
)

$crtDir = Find-VcCrtDir
$sys32 = Join-Path $env:SystemRoot "System32"

Write-Host "==> copy runtime DLLs -> $OutDir"
foreach ($name in $dllNames) {
    $src = $null
    if ($crtDir) {
        $cand = Join-Path $crtDir $name
        if (Test-Path $cand) { $src = $cand }
    }
    if (-not $src) {
        $cand = Join-Path $sys32 $name
        if (Test-Path $cand) { $src = $cand }
    }
    if ($src) {
        Copy-Item -Force $src (Join-Path $OutDir $name)
        Write-Host "  + $name"
    } else {
        Write-Host "  ! missing $name"
    }
}

$dml = Join-Path $sys32 "DirectML.dll"
if (Test-Path $dml) {
    Copy-Item -Force $dml (Join-Path $OutDir "DirectML.dll")
    Write-Host "  + DirectML.dll"
} else {
    $ortDml = Get-ChildItem (Join-Path $env:LOCALAPPDATA "ort.pyke.io") -Recurse -Filter "DirectML.dll" -ErrorAction SilentlyContinue |
        Sort-Object Length -Descending |
        Select-Object -First 1
    if ($ortDml) {
        Copy-Item -Force $ortDml.FullName (Join-Path $OutDir "DirectML.dll")
        Write-Host "  + DirectML.dll (from ort cache)"
    } else {
        Write-Host "  ! missing DirectML.dll (often already on Win10 1903+)"
    }
}

Write-Host ""
Write-Host "OK: $OutDir"
Get-ChildItem $OutDir | Format-Table Name, @{N = "MB"; E = { [math]::Round($_.Length / 1MB, 2) } } -AutoSize
Write-Host "Ship this folder as-is. Do not enable +crt-static (breaks static ORT)."
