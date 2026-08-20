# 按地图名或地图 ID，从 maplestory.io 下载官方小地图 + 完整渲染图。
# 不依赖游戏安装目录（怀旧服客户端里通常没有现成的完整图 PNG）。
#
# 用法:
#   .\scripts\download_map.ps1 -MapId 50001
#   .\scripts\download_map.ps1 -Query "南港"
#   .\scripts\download_map.ps1 -MapId 50001 -OutDir .\tmp

param(
    [string]$Query = "",
    [long]$MapId = 0,
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$Ua = "Mozilla/5.0"
$WikiApi = "https://wiki.biligame.com/maplestory/api.php"
$IoBase = "https://maplestory.io/api/GMS/83/map"

function Get-MapIdFromText([string]$text) {
    if ($text -match '(?:Map/)?(\d{5,9})') {
        return [long]$Matches[1]
    }
    return $null
}

function Resolve-MapIdByWiki([string]$name) {
    $candidates = @(
        $name.Trim(),
        ($name -replace '-', ':' -replace '：', ':' -replace '/', ':')
    )
    $parts = @($name -split '[-:：/｜|]' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    if ($parts.Count -ge 1) { $candidates += $parts[-1] }
    if ($parts.Count -ge 2) { $candidates += "$($parts[0]):$($parts[-1])" }

    $seen = @{}
    foreach ($cand in $candidates) {
        if ([string]::IsNullOrWhiteSpace($cand) -or $seen.ContainsKey($cand)) { continue }
        $seen[$cand] = $true
        $url = "$WikiApi?action=query&format=json&redirects=1&titles=$([uri]::EscapeDataString($cand))"
        $data = Invoke-RestMethod -Uri $url -UserAgent $Ua
        if ($data.query.redirects) {
            foreach ($r in @($data.query.redirects)) {
                $id = Get-MapIdFromText $r.to
                if ($id) { return $id }
            }
        }
        if ($data.query.pages) {
            foreach ($p in $data.query.pages.PSObject.Properties.Value) {
                $id = Get-MapIdFromText $p.title
                if ($id) { return $id }
            }
        }
    }

    $searchUrl = "$WikiApi?action=query&format=json&list=search&srlimit=10&srsearch=$([uri]::EscapeDataString($name))"
    $search = Invoke-RestMethod -Uri $searchUrl -UserAgent $Ua
    foreach ($item in @($search.query.search)) {
        foreach ($field in @($item.title, $item.snippet)) {
            $id = Get-MapIdFromText $field
            if ($id) { return $id }
        }
    }
    return $null
}

if (-not $OutDir) {
    $OutDir = Join-Path $PSScriptRoot "..\tmp"
}
$OutDir = [System.IO.Path]::GetFullPath($OutDir)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if ($MapId -le 0) {
    if ([string]::IsNullOrWhiteSpace($Query)) {
        Write-Host @"
用法:
  .\scripts\download_map.ps1 -MapId 50001
  .\scripts\download_map.ps1 -Query "彩虹岛-南港"
  .\scripts\download_map.ps1 -MapId 50001 -OutDir .\tmp

说明:
  - 优先用 -MapId（最稳）。截图里 NPC「白瑞德」→ 资料站常标地图 50001（南港西郊平原）。
  - -Query 走 biligame wiki 解析；短 ID / 冷门中文名可能解析失败，请改用数字 ID。
  - 完整图来自 maplestory.io，不是游戏安装目录。
"@
        exit 1
    }
    if ($Query -match '^\d{1,9}$') {
        $MapId = [long]$Query
    } else {
        Write-Host "wiki 解析: $Query"
        $MapId = Resolve-MapIdByWiki $Query
        if (-not $MapId) {
            Write-Error "wiki 无法解析「$Query」。请改用 -MapId，或到 https://mxd.dvg.cn 用 NPC/地图名查 ID。"
        }
    }
}

Write-Host "地图 ID $MapId"

try {
    $info = Invoke-RestMethod -Uri "$IoBase/$MapId/name" -UserAgent $Ua
    Write-Host ("maplestory.io: {0} / {1}" -f $info.streetName, $info.name)
} catch {
    Write-Host "警告: 无法读取 maplestory.io 地图名（ID 可能无效）"
}

$minimapPath = Join-Path $OutDir "map_${MapId}_minimap.png"
$renderPath = Join-Path $OutDir "map_${MapId}_render.png"

Invoke-WebRequest -Uri "$IoBase/$MapId/minimap" -UserAgent $Ua -OutFile $minimapPath
Invoke-WebRequest -Uri "$IoBase/$MapId/render" -UserAgent $Ua -OutFile $renderPath

Write-Host "小地图 $minimapPath"
Write-Host "完整图 $renderPath"
