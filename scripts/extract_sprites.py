#!/usr/bin/env python3
"""从怀旧服客户端 Addressables 抽出怪物帧图 / 传送门精灵图。

依赖: pip install UnityPy Pillow numpy opencv-python

用法:
  python scripts/extract_sprites.py --mob 130100
  python scripts/extract_sprites.py --mob 100101 --mob 130101 --mob 1210102 --mob 130100
  python scripts/extract_sprites.py --portals
  python scripts/extract_sprites.py --map 50001
  python scripts/extract_sprites.py --map 50001 --out assets
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import UnityPy
from PIL import Image

IO_BASE = "https://maplestory.io/api/GMS/83"
UA = {"User-Agent": "Mozilla/5.0"}

# 南港西郊平原等常见中文名（仅用于目录命名；像素始终来自客户端）
MOB_CN = {
    100100: "绿蜗牛",
    100101: "蓝蜗牛",
    130100: "木妖",
    130101: "红蜗牛",
    1210102: "花蘑菇",
}

PORTAL_PV_KEY = (
    "Assets/WzAssets/SpriteSet/Common/Map/MapHelper/portal/game/pv.asset"
)
PORTAL_PH_KEYS = {
    "ph_default_start": (
        "Assets/WzAssets/SpriteSet/Common/Map/MapHelper/portal/game/ph/default/portalStart.asset"
    ),
    "ph_default_continue": (
        "Assets/WzAssets/SpriteSet/Common/Map/MapHelper/portal/game/ph/default/portalContinue.asset"
    ),
    "ph_default_exit": (
        "Assets/WzAssets/SpriteSet/Common/Map/MapHelper/portal/game/ph/default/portalExit.asset"
    ),
}


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
    raise FileNotFoundError("找不到 mxdclassic 安装目录，请检查 安装位置.txt")


def aa_w(game_root: Path) -> Path:
    return game_root / "Maplestory_Classic_Data" / "StreamingAssets" / "aa" / "w"


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def pad_mob_id(mob_id: int) -> str:
    return f"{mob_id:07d}"


def load_bundle_container(bundle_path: Path):
    env = UnityPy.load(str(bundle_path))
    objs = {o.path_id: o for o in env.objects}
    container = {}
    for obj in env.objects:
        if obj.type.name == "AssetBundle":
            container = {k: v for k, v in obj.read().m_Container}
            break
    return env, objs, container


def find_mob_atlas_key(container: dict, mob_id: int) -> str | None:
    pad = pad_mob_id(mob_id)
    want = f"/Mob/{pad}_0.png"
    for k in container:
        if k.endswith(want) or k.endswith(f"/Mob/{mob_id}_0.png"):
            return k
    return None


def find_mob_bundle(w_dir: Path, mob_id: int) -> tuple[Path, str]:
    pad = pad_mob_id(mob_id)
    needle = pad.encode("ascii")
    for bundle in sorted(w_dir.glob("spritesheet_*.bundle")):
        if needle not in bundle.read_bytes():
            continue
        _, _, container = load_bundle_container(bundle)
        key = find_mob_atlas_key(container, mob_id)
        if key:
            return bundle, key
    raise FileNotFoundError(f"客户端中找不到 Mob/{pad}_0.png（id={mob_id}）")


def load_atlas_png(bundle: Path, key: str) -> Image.Image:
    _, objs, container = load_bundle_container(bundle)
    tex = objs[container[key].asset.path_id].read()
    return tex.image.convert("RGBA")


def download_io_frames(mob_id: int, dest: Path) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    zdata = http_get(f"{IO_BASE}/mob/{mob_id}/download")
    with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
        for n in zf.namelist():
            if not n.lower().endswith(".png"):
                continue
            parts = Path(n).parts
            if len(parts) >= 2:
                out = f"{parts[-2]}_{Path(parts[-1]).stem}.png"
            else:
                out = Path(n).name
            (dest / out).write_bytes(zf.read(n))
    meta = json.loads(http_get(f"{IO_BASE}/mob/{mob_id}").decode())
    return meta.get("name") or ""


def map_mob_ids(map_id: int) -> list[int]:
    data = json.loads(http_get(f"{IO_BASE}/map/{map_id}").decode())
    ids = sorted({int(m["id"]) for m in data.get("mobs", [])})
    return ids


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / float(aw * ah + bw * bh - inter)


def cut_frames_from_atlas(atlas: Image.Image, io_dir: Path, out_dir: Path) -> tuple[int, int]:
    """用 maplestory.io 命名帧作模板，在游戏图集上 NCC 裁切（一对一、去重）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas.save(out_dir / "_atlas_from_game.png")

    rgba = np.array(atlas)
    rgb = rgba[:, :, :3].copy()
    rgb[rgba[:, :, 3] < 10] = 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    refs = sorted(io_dir.glob("*.png"))
    unique: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for ref_path in refs:
        ph = hashlib.md5(np.array(Image.open(ref_path).convert("RGBA")).tobytes()).hexdigest()
        name = ref_path.name
        if ph not in unique:
            unique[ph] = name
            aliases[name] = name
        else:
            aliases[name] = unique[ph]

    placements: dict[str, tuple[int, int, int, int, float]] = {}
    used: list[tuple[int, int, int, int]] = []
    for _, cname in unique.items():
        ref = Image.open(io_dir / cname).convert("RGBA")
        bb = ref.getbbox()
        if not bb:
            continue
        ref = ref.crop(bb)
        ref_rgba = np.array(ref)
        ref_rgb = ref_rgba[:, :, :3].copy()
        ref_rgb[ref_rgba[:, :, 3] < 10] = 0
        ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
        rh, rw = ref_gray.shape
        if rh > gray.shape[0] or rw > gray.shape[1]:
            print(f"  skip oversized {cname}")
            continue
        res = cv2.matchTemplate(gray, ref_gray, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.unravel_index(np.argsort(res, axis=None)[::-1][:30], res.shape)
        for y, x in zip(ys, xs):
            score = float(res[y, x])
            box = (int(x), int(y), rw, rh)
            if score < 0.85:
                break
            if any(_iou(box, u) > 0.3 for u in used):
                continue
            placements[cname] = (*box, score)
            used.append(box)
            break
        else:
            print(f"  NO MATCH {cname} best={float(res.max()):.3f}")

    for name, canon in aliases.items():
        if canon not in placements:
            continue
        x, y, ww, hh, _ = placements[canon]
        crop = atlas.crop((x, y, x + ww, y + hh))
        bb = crop.getbbox()
        if bb:
            crop = crop.crop(bb)
        crop.save(out_dir / name)

    return len(placements), len(aliases)


def extract_mob(mob_id: int, out_root: Path, cache_dir: Path, game_root: Path) -> Path:
    w_dir = aa_w(game_root)
    en_name = ""
    io_dir = cache_dir / f"io_mob_{mob_id}"
    if not any(io_dir.glob("*.png")):
        en_name = download_io_frames(mob_id, io_dir)
        print(f"  IO 命名帧: {mob_id} {en_name} ×{len(list(io_dir.glob('*.png')))}")
    else:
        print(f"  复用 IO 缓存: {io_dir}")

    bundle, key = find_mob_bundle(w_dir, mob_id)
    atlas = load_atlas_png(bundle, key)
    cn = MOB_CN.get(mob_id, en_name or str(mob_id))
    out_dir = out_root / "mobs" / f"{mob_id}_{cn}"
    n_unique, n_named = cut_frames_from_atlas(atlas, io_dir, out_dir)
    pad = pad_mob_id(mob_id)
    readme = (
        f"{cn}\n"
        f"怪物ID: {mob_id}（资源键 {pad}）\n"
        f"独立像素帧: {n_unique} / 命名帧: {n_named}\n"
        f"像素来源: {bundle.name} -> {key}\n"
        f"命名参考: maplestory.io GMS/83 mob {mob_id}\n"
        f"说明: 部分动作帧会共用同一张图（命名数可大于独立像素数）\n"
    )
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")
    print(f"  -> {out_dir}  unique={n_unique} named={n_named} atlas={atlas.size}")
    return out_dir


def _sprites_from_asset_raw(raw: bytes, objs: dict) -> list:
    seen = set()
    ordered = []
    for off in range(0, len(raw) - 8, 4):
        pid = struct.unpack_from("<q", raw, off)[0]
        if pid in objs and objs[pid].type.name == "Sprite" and pid not in seen:
            seen.add(pid)
            ordered.append(pid)
    return ordered


def find_portal_bundle(w_dir: Path) -> Path:
    for bundle in sorted(w_dir.glob("spriteset_*.bundle")):
        if b"portal/game" in bundle.read_bytes():
            return bundle
    raise FileNotFoundError("找不到含 portal/game 的 spriteset_*.bundle")


def extract_portals(out_root: Path, game_root: Path) -> Path:
    w_dir = aa_w(game_root)
    bundle = find_portal_bundle(w_dir)
    _, objs, container = load_bundle_container(bundle)
    out = out_root / "portals"
    out.mkdir(parents=True, exist_ok=True)

    def dump(asset_key: str, dest: Path, prefix: str):
        dest.mkdir(parents=True, exist_ok=True)
        raw = objs[container[asset_key].asset.path_id].get_raw_data()
        sprites = _sprites_from_asset_raw(raw, objs)
        print(f"  {asset_key.split('portal/game/')[-1]} ×{len(sprites)}")
        for i, pid in enumerate(sprites):
            img = objs[pid].read().image.convert("RGBA")
            img.save(dest / f"{prefix}_{i:02d}.png")

    dump(PORTAL_PV_KEY, out / "pv_可见传送门", "pv")
    for folder, key in PORTAL_PH_KEYS.items():
        prefix = folder.split("_")[-1]
        dump(key, out / folder, prefix)

    (out / "README.txt").write_text(
        "传送门精灵图\n"
        "可见入口/出口（portal type=2）使用 MapHelper/portal/game/pv\n"
        f"像素来源: {bundle.name}\n"
        "另附 ph/default（隐身门 start/continue/exit）\n",
        encoding="utf-8",
    )
    print(f"  -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="从怀旧服客户端抽取怪物/传送门精灵图")
    ap.add_argument("--mob", type=int, action="append", default=[], help="怪物 ID，可多次")
    ap.add_argument("--map", type=int, default=0, help="地图 ID，自动收集该图全部 mob")
    ap.add_argument("--portals", action="store_true", help="导出传送门 pv / ph")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出根目录（默认 mxd_tools/assets/）",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="IO 命名帧缓存目录（默认 mxd_tools/tmp/extract_probe）",
    )
    args = ap.parse_args()

    out_root = args.out or (mxd_tools_root() / "assets")
    cache_dir = args.cache or (mxd_tools_root() / "tmp" / "extract_probe")
    cache_dir.mkdir(parents=True, exist_ok=True)
    game_root = find_game_root()
    print("游戏目录:", game_root)
    print("输出目录:", out_root)

    mob_ids = list(args.mob)
    if args.map:
        ids = map_mob_ids(args.map)
        print(f"地图 {args.map} 怪物: {ids}")
        mob_ids.extend(ids)
    # 去重保序
    seen = set()
    mob_ids = [m for m in mob_ids if not (m in seen or seen.add(m))]

    if not mob_ids and not args.portals:
        ap.error("请指定 --mob / --map / --portals")

    for mid in mob_ids:
        print(f"抽取怪物 {mid} ...")
        extract_mob(mid, out_root, cache_dir, game_root)

    if args.portals or args.map:
        print("抽取传送门 ...")
        extract_portals(out_root, game_root)

    print("完成")


if __name__ == "__main__":
    main()
