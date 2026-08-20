#!/usr/bin/env python3
"""从怀旧服客户端导出地图 minimap（中文资源）。

WZJS（0000xxxxx.wzjson）里存布局与 miniMap 元数据；画布像素在同 ID 的
SpriteSheet/CN/Map/Map/Map0/{id}_0.png 图集中，矩形在对应 .wzspritesheet（WZSS）。

用法:
  python scripts/extract_minimap.py 50001
  python scripts/extract_minimap.py 50001 --out assets/maps
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import UnityPy
from PIL import Image

# 已知 miniMap 元数据在 WZJS 二进制中的特征（与 GMS API 一致时可定位）
# 布局: width(i32) height(i32) centerX(i32) centerY(i32) mag(i32)


def mxd_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_game_root() -> Path:
    tip = repo_root() / "安装位置.txt"
    if tip.is_file():
        p = Path(tip.read_text(encoding="utf-8").strip())
        if (p / "Maplestory_Classic_Data").is_dir():
            return p
    hit = next(Path(r"D:\Program Files").rglob("mxdclassic"), None)
    if hit and (hit / "Maplestory_Classic_Data").is_dir():
        return hit
    raise FileNotFoundError("找不到 mxdclassic，请检查 安装位置.txt")


def aa_w(game_root: Path) -> Path:
    return game_root / "Maplestory_Classic_Data" / "StreamingAssets" / "aa" / "w"


def pad_map_id(map_id: int) -> str:
    """客户端地图资源键：9 位补零，如 50001 -> 000050001。"""
    return f"{map_id:09d}"


def load_bundle_container(bundle_path: Path):
    env = UnityPy.load(str(bundle_path))
    objs = {o.path_id: o for o in env.objects}
    container = {}
    for obj in env.objects:
        if obj.type.name == "AssetBundle":
            container = {k: v for k, v in obj.read().m_Container}
            break
    return env, objs, container


def find_asset(w_dir: Path, needle: str, glob_pat: str = "*.bundle") -> tuple[Path, str]:
    """在 bundle 的 m_Container 中查找 key 含子串 needle。"""
    bundles = sorted(w_dir.glob(glob_pat), key=lambda p: -p.stat().st_size)
    nb = needle.encode("ascii")
    # 明文能搜到则只查这些；地图图集路径常被压进 bundle，需全扫
    hit_plain = [b for b in bundles if nb in b.read_bytes()]
    ordered = hit_plain if hit_plain else bundles

    for bundle in ordered:
        print(f"  扫描 {bundle.name} ...")
        _, _, container = load_bundle_container(bundle)
        for k in container:
            if needle in k:
                return bundle, k
    raise FileNotFoundError(f"找不到资源: {needle}（pattern={glob_pat}）")

def read_raw_by_key(bundle: Path, key: str) -> bytes:
    _, objs, container = load_bundle_container(bundle)
    return objs[container[key].asset.path_id].get_raw_data()


def read_texture_by_key(bundle: Path, key: str) -> Image.Image:
    _, objs, container = load_bundle_container(bundle)
    return objs[container[key].asset.path_id].read().image.convert("RGBA")


def parse_wzjs_minimap_meta(raw: bytes) -> dict | None:
    """从 WZJS 原始字节中扫描 miniMap 五元组 width/height/centerX/centerY/mag。

    经典图 mag 几乎总是 4；在 mag==4 的候选里取面积最大者。
    """
    if b"WZJS" not in raw:
        return None
    best = None
    best_area = -1
    for i in range(0, len(raw) - 20, 4):
        w, h, cx, cy, mag = struct.unpack_from("<iiiii", raw, i)
        if mag != 4:
            continue
        if not (400 <= w <= 8192 and 200 <= h <= 8192):
            continue
        if not (-4000 <= cx <= 4000 and -1000 <= cy <= 8000):
            continue
        # 其后常见 portal：x,y 落在地图范围内
        bonus = 0
        if i + 28 <= len(raw):
            nx, ny = struct.unpack_from("<ii", raw, i + 20)
            if 0 <= nx <= w and -200 <= ny <= h:
                bonus = 1
        area = w * h
        rank = (bonus, area)
        best_rank = (1 if best and best.get("_bonus") else 0, best_area)
        if best is None or rank > best_rank:
            best_area = area
            best = {
                "width": w,
                "height": h,
                "centerX": cx,
                "centerY": cy,
                "magnification": mag,
                "_offset": i,
                "_bonus": bonus,
            }
    if best:
        best.pop("_bonus", None)
        best.pop("_offset", None)
    return best


def parse_wzss_canvas_rect(raw: bytes) -> tuple[int, int, int, int] | None:
    """
    解析地图 WZSS：在 WZSS 魔数后找 canvas 宽高，原点一般为 (0,0)。
    返回 Unity 底左原点下的 (x, y, w, h)。
    """
    idx = raw.find(b"WZSS")
    if idx < 0:
        return None
    # 在 WZSS 段内找一对合理的 u32 宽高（minimap 通常 < 512）
    for i in range(idx, min(len(raw) - 8, idx + 200), 4):
        w, h = struct.unpack_from("<II", raw, i)
        if 16 <= w <= 512 and 16 <= h <= 512:
            # 前面常见 float 1.0 (0x3f800000) 与若干 0
            return (0, 0, int(w), int(h))
    return None


def crop_unity_bl(atlas: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """Unity 纹理坐标原点在左下；PIL 在左上。"""
    aw, ah = atlas.size
    y_pil = ah - y - h
    return atlas.crop((x, y_pil, x + w, y_pil + h))


def extract_minimap(map_id: int, out_dir: Path, game_root: Path) -> Path:
    w_dir = aa_w(game_root)
    pad = pad_map_id(map_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) WZJS 元数据
    json_bundle, json_key = find_asset(w_dir, f"{pad}.wzjson", "json_*.bundle")
    wzjs_raw = read_raw_by_key(json_bundle, json_key)
    meta = parse_wzjs_minimap_meta(wzjs_raw) or {}
    print(f"  WZJS: {json_bundle.name} -> {json_key}")
    print(f"  miniMap meta: {meta}")

    # 2) 图集 + WZSS 矩形
    sheet_bundle, png_key = find_asset(w_dir, f"{pad}_0.png", "spritesheet_*.bundle")
    # 规范化 key
    if not png_key.startswith("Assets/"):
        # 通过 container 再找一次
        _, _, container = load_bundle_container(sheet_bundle)
        for k in container:
            if f"{pad}_0.png" in k:
                png_key = k
                break
    atlas = read_texture_by_key(sheet_bundle, png_key)
    atlas_path = out_dir / f"map_{map_id}_atlas.png"
    atlas.save(atlas_path)

    ss_key = png_key.replace("_0.png", ".wzspritesheet")
    rect = None
    try:
        ss_raw = read_raw_by_key(sheet_bundle, ss_key)
        rect = parse_wzss_canvas_rect(ss_raw)
        print(f"  WZSS rect (BL): {rect}")
    except Exception as e:
        print(f"  WZSS 读取失败: {e}")

    if rect is None:
        # 回退：用不透明包围盒（略逊于 WZSS）
        bb = atlas.getbbox()
        if not bb:
            raise RuntimeError("图集全透明")
        x0, y0, x1, y1 = bb
        # 转成 BL 原点再按常见 125x72 对齐过于随意；直接用 bbox 顶左裁切
        mini = atlas.crop(bb)
        print(f"  fallback bbox crop {bb} -> {mini.size}")
    else:
        x, y, w, h = rect
        mini = crop_unity_bl(atlas, x, y, w, h)
        print(f"  crop PIL size {mini.size}")

    mini_path = out_dir / f"map_{map_id}_minimap.png"
    mini.save(mini_path)

    meta_out = {
        "mapId": map_id,
        "padId": pad,
        "miniMap": meta,
        "atlas": str(atlas_path.name),
        "minimap": str(mini_path.name),
        "sources": {
            "wzjson": f"{json_bundle.name} :: {json_key}",
            "atlas": f"{sheet_bundle.name} :: {png_key}",
            "wzspritesheet": ss_key,
            "rect_bl": rect,
        },
        "note": "像素来自客户端 CN SpriteSheet；WZJS 提供 width/height/center/mag",
    }
    (out_dir / f"map_{map_id}_minimap.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  -> {mini_path}")
    return mini_path


def main():
    ap = argparse.ArgumentParser(description="从客户端 WZJS/图集导出地图 minimap")
    ap.add_argument("map_id", type=int, help="地图 ID，如 50001")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录（默认 mxd_tools/assets/maps/{id}/）",
    )
    args = ap.parse_args()

    game = find_game_root()
    out = args.out or (mxd_tools_root() / "assets" / "maps" / str(args.map_id))
    print("游戏目录:", game)
    print("输出目录:", out)
    extract_minimap(args.map_id, out, game)
    print("完成")


if __name__ == "__main__":
    main()
