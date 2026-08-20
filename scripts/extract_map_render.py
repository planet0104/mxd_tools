#!/usr/bin/env python3
"""导出完整大地图 PNG，并附带从客户端 WZJS 解析的布局元数据。

完整像素图：maplestory.io GMS/83 `/render`（CMS 对 50001 无图；客户端无现成整图）。
WZJS：可稳定读出 miniMap / VR / portal 等整型叶子；tile/obj 字符串+坐标的完整
本地拼图仍在推进（值类型流与图集裁帧未完全对齐）。

用法:
  python scripts/extract_map_render.py 50001
  python scripts/extract_map_render.py 50001 --out assets/maps
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from collections import defaultdict
from pathlib import Path

import sys

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from extract_minimap import (  # noqa: E402
    aa_w,
    find_asset,
    find_game_root,
    mxd_tools_root,
    pad_map_id,
    parse_wzjs_minimap_meta,
    read_raw_by_key,
)

UA = "Mozilla/5.0"
GMS_RENDER = "https://maplestory.io/api/GMS/83/map/{map_id}/render"
GMS_NAME = "https://maplestory.io/api/GMS/83/map/{map_id}/name"

STRING_FIELDS = {
    "tS",
    "oS",
    "l0",
    "l1",
    "l2",
    "u",
    "bS",
    "pn",
    "tn",
    "script",
    "bgm",
    "mapMark",
    "mapDesc",
}
FLOAT_FIELDS = {"mobRate"}


def http_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def http_json(url: str) -> dict:
    return json.loads(http_bytes(url).decode("utf-8"))


def field_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def leaf_kind(path: str) -> str:
    f = field_name(path)
    if "spritesheetitem" in path:
        return "skip"
    if f in STRING_FIELDS:
        return "str"
    if path.startswith("life/") and f in ("type", "id"):
        return "str"
    if f in FLOAT_FIELDS:
        return "float"
    return "int"


def parse_wzjs_paths(raw: bytes) -> list[str]:
    """从 WZJS 的 path 拼接区 + offset 表还原路径列表。"""
    # path 文本区以层根/分区开头，例如 "00/info0/tile..."（首字符为层号 0）
    hit = raw.find(b"0/info")
    if hit < 0:
        raise RuntimeError("WZJS 中找不到 path blob")
    blob_start = hit - 1 if hit > 0 and raw[hit - 1 : hit] == b"0" else hit

    best: tuple[int, list[str]] | None = None
    # offset 表紧跟 path blob，首项为 0，单调递增
    for off_table in range(blob_start + 64, len(raw) - 64, 4):
        if struct.unpack_from("<I", raw, off_table)[0] != 0:
            continue
        blob = raw[blob_start:off_table]
        if len(blob) < 200:
            continue
        offs: list[int] = []
        i = 0
        while off_table + (i + 1) * 4 <= len(raw):
            v = struct.unpack_from("<I", raw, off_table + i * 4)[0]
            if i > 0 and v < offs[-1]:
                break
            if v > len(blob):
                break
            offs.append(v)
            i += 1
            if i > 20000:
                break
        if len(offs) < 200:
            continue
        ends = offs[1:] + [len(blob)]
        paths = [blob[a:b].decode("ascii", "replace") for a, b in zip(offs, ends)]
        while paths and paths[-1] == "":
            paths.pop()
        if "miniMap/width" in paths and "portal/0/x" in paths:
            if best is None or len(paths) > best[0]:
                best = (len(paths), paths)
    if not best:
        raise RuntimeError("无法定位 WZJS path offset 表")
    return best[1]


def parse_wzjs_layout(raw: bytes) -> dict:
    """解析 WZJS 中已验证可对齐的整型/浮点叶子（info/portal/miniMap/VR）。"""
    paths = parse_wzjs_paths(raw)
    path_set = set(paths)
    leaves = [p for p in paths if p and not any(q.startswith(p + "/") for q in path_set)]

    num_leaves = [p for p in leaves if leaf_kind(p) in ("int", "float")]
    # 用已知 miniMap 五元组锚定数值流
    mm_meta = parse_wzjs_minimap_meta(raw) or {}
    w = int(mm_meta.get("width") or 0)
    h = int(mm_meta.get("height") or 0)
    cx = int(mm_meta.get("centerX") or 0)
    cy = int(mm_meta.get("centerY") or 0)
    mag = int(mm_meta.get("magnification") or 4)
    if not w:
        raise RuntimeError("无法从 WZJS 读 miniMap 尺寸")

    needle = struct.pack("<iiiii", w, h, cx, cy, mag)
    mm_pos = raw.find(needle)
    if mm_pos < 0 or "miniMap/width" not in num_leaves:
        raise RuntimeError("无法锚定 miniMap 数值流")

    base = mm_pos - num_leaves.index("miniMap/width") * 4
    values: dict[str, int | float] = {}
    for i, p in enumerate(num_leaves):
        addr = base + i * 4
        if addr < 0 or addr + 4 > len(raw):
            continue
        if leaf_kind(p) == "float":
            values[p] = struct.unpack_from("<f", raw, addr)[0]
        else:
            values[p] = struct.unpack_from("<i", raw, addr)[0]

    info = {}
    mini_map = dict(mm_meta)
    portals: dict[str, dict] = defaultdict(dict)
    for p, v in values.items():
        parts = p.split("/")
        if parts[0] == "info" and len(parts) == 2:
            info[parts[1]] = v
        elif parts[0] == "miniMap" and len(parts) == 2:
            mini_map[parts[1]] = v
        elif parts[0] == "portal" and len(parts) == 3:
            portals[parts[1]][parts[2]] = v

    return {
        "pathCount": len(paths),
        "leafCount": len(leaves),
        "numLeafCount": len(num_leaves),
        "info": info,
        "miniMap": mini_map,
        "portals": {k: portals[k] for k in sorted(portals, key=lambda x: int(x))},
        "note": (
            "整型叶子（VR/portal/miniMap 等）已与二进制对齐；"
            "tile/obj 的字符串叶子与坐标完整拼图仍需继续解析类型流与图集。"
        ),
    }


def extract_map_render(map_id: int, out_dir: Path, game_root: Path | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("下载 GMS render ...")
    png = http_bytes(GMS_RENDER.format(map_id=map_id))
    if not png.startswith(b"\x89PNG"):
        raise RuntimeError("render 返回的不是 PNG")
    render_path = out_dir / f"map_{map_id}_render.png"
    render_path.write_bytes(png)
    print(f"  -> {render_path} ({len(png)} bytes)")

    name_info = {}
    try:
        name_info = http_json(GMS_NAME.format(map_id=map_id))
        print("  name:", name_info)
    except Exception as e:
        print("  name API 失败:", e)

    layout = None
    wzjs_src = None
    if game_root is not None:
        try:
            w_dir = aa_w(game_root)
            pad = pad_map_id(map_id)
            bundle, key = find_asset(w_dir, f"{pad}.wzjson", "json_*.bundle")
            raw = read_raw_by_key(bundle, key)
            wzjs_src = f"{bundle.name} :: {key}"
            layout = parse_wzjs_layout(raw)
            print("  WZJS layout: portals=", len(layout.get("portals") or {}))
        except Exception as e:
            print("  WZJS 布局解析跳过:", e)

    meta = {
        "mapId": map_id,
        "padId": pad_map_id(map_id),
        "name": name_info,
        "render": render_path.name,
        "renderSource": GMS_RENDER.format(map_id=map_id),
        "renderNote": "GMS/83 英文木牌；CMS 对本图无 /render",
        "wzjson": wzjs_src,
        "layout": layout,
    }
    meta_path = out_dir / f"map_{map_id}_render.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  -> {meta_path}")
    return render_path


def main() -> None:
    ap = argparse.ArgumentParser(description="导出完整大地图 render + WZJS 布局元数据")
    ap.add_argument("map_id", type=int, help="地图 ID，如 50001")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录（默认 mxd_tools/assets/maps/{id}/）",
    )
    ap.add_argument(
        "--no-client",
        action="store_true",
        help="不读客户端 WZJS，只下载 GMS render",
    )
    args = ap.parse_args()

    out = args.out or (mxd_tools_root() / "assets" / "maps" / str(args.map_id))
    game = None if args.no_client else find_game_root()
    print("输出目录:", out)
    if game:
        print("游戏目录:", game)
    extract_map_render(args.map_id, out, game)
    print("完成")


if __name__ == "__main__":
    main()
