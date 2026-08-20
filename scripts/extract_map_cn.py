#!/usr/bin/env python3
"""从资源拼出中文完整地图。

布局：maplestory.io WZ API（GMS/83，与怀旧服同图几何）
像素：Tile/Back/普通 Obj → GMS WZ；signboard → 客户端 CN 图集（中文木牌）

用法:
  python -u scripts/extract_map_cn.py 50001
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from extract_minimap import (  # noqa: E402
    aa_w,
    find_asset,
    find_game_root,
    mxd_tools_root,
    pad_map_id,
    read_texture_by_key,
)

UA = "Mozilla/5.0"
WZ = "https://maplestory.io/api/wz/{region}/{path}"
CN_OBJ_SETS = {"signboard"}


def log(msg: str) -> None:
    print(msg, flush=True)


class WzClient:
    def __init__(self, workers: int = 24):
        self.cache: dict[str, dict | None] = {}
        self.img_cache: dict[str, tuple[Image.Image, tuple[int, int]]] = {}
        self.workers = workers

    def _fetch(self, region: str, path: str) -> dict | None:
        url = WZ.format(region=region, path=path)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                time.sleep(0.4 * (attempt + 1))
            except Exception:
                time.sleep(0.4 * (attempt + 1))
        return None

    def get(self, region: str, path: str) -> dict:
        key = f"{region}|{path}"
        if key not in self.cache:
            self.cache[key] = self._fetch(region, path)
        node = self.cache[key]
        if node is None:
            raise FileNotFoundError(f"{region} {path}")
        return node

    def get_many(self, items: list[tuple[str, str]]) -> None:
        todo = [(r, p) for r, p in items if f"{r}|{p}" not in self.cache]
        if not todo:
            return
        log(f"  并行拉取 {len(todo)} 个节点…")
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = {pool.submit(self._fetch, r, p): (r, p) for r, p in todo}
            done = 0
            for fut in as_completed(futs):
                r, p = futs[fut]
                self.cache[f"{r}|{p}"] = fut.result()
                done += 1
                if done % 50 == 0 or done == len(todo):
                    log(f"    {done}/{len(todo)}")

    def leaf(self, region: str, path: str):
        node = self.get(region, path)
        return node.get("value")


def _load_entries(
    client: WzClient, region: str, parent_path: str, ids: list[str]
) -> list[dict]:
    """批量加载 parent/id 下各字段叶子。"""
    if not ids:
        return []
    client.get_many([(region, f"{parent_path}/{i}") for i in ids])
    fields: dict[str, list[str]] = {}
    freqs: list[tuple[str, str]] = []
    for i in ids:
        fk = client.get(region, f"{parent_path}/{i}").get("children") or []
        fields[i] = fk
        freqs.extend((region, f"{parent_path}/{i}/{k}") for k in fk)
    client.get_many(freqs)
    out = []
    for i in ids:
        item = {"_i": i}
        for k in fields[i]:
            item[k] = client.leaf(region, f"{parent_path}/{i}/{k}")
        out.append(item)
    return out


def walk_map(client: WzClient, region: str, map_path: str) -> dict:
    root = client.get(region, map_path)
    children = root.get("children") or []
    client.get_many([(region, f"{map_path}/{c}") for c in children])

    info: dict = {}
    if "info" in children:
        ikids = client.get(region, f"{map_path}/info").get("children") or []
        client.get_many([(region, f"{map_path}/info/{k}") for k in ikids])
        for k in ikids:
            info[k] = client.leaf(region, f"{map_path}/info/{k}")

    backs: list[dict] = []
    if "back" in children:
        bkids = client.get(region, f"{map_path}/back").get("children") or []
        log(f"  back x{len(bkids)}")
        backs = _load_entries(client, region, f"{map_path}/back", bkids)

    layers: dict = {}
    for L in children:
        if not L.isdigit():
            continue
        lp = f"{map_path}/{L}"
        kids = client.get(region, lp).get("children") or []
        layer: dict = {"tS": None, "tiles": [], "objs": []}
        if "info" in kids:
            client.get_many([(region, f"{lp}/info"), (region, f"{lp}/info/tS")])
            try:
                layer["tS"] = client.leaf(region, f"{lp}/info/tS")
            except Exception:
                pass
        if "tile" in kids:
            tkids = client.get(region, f"{lp}/tile").get("children") or []
            log(f"  layer {L} tiles x{len(tkids)}")
            layer["tiles"] = _load_entries(client, region, f"{lp}/tile", tkids)
        if "obj" in kids:
            okids = client.get(region, f"{lp}/obj").get("children") or []
            log(f"  layer {L} objs x{len(okids)}")
            layer["objs"] = _load_entries(client, region, f"{lp}/obj", okids)
        layers[L] = layer
        log(f"  layer {L} done tS={layer['tS']}")

    return {"info": info, "back": backs, "layers": layers}


def canvas_and_origin(client: WzClient, region: str, canvas_path: str):
    cache_key = f"{region}|{canvas_path}"
    if cache_key in client.img_cache:
        return client.img_cache[cache_key]

    node = client.get(region, canvas_path)
    if node.get("type") != 12 or "value" not in node:
        kids = node.get("children") or []
        digits = [k for k in kids if k.isdigit()]
        if digits:
            return canvas_and_origin(client, region, f"{canvas_path}/{digits[0]}")
        raise RuntimeError(f"不是 canvas: {region} {canvas_path}")

    img = Image.open(io.BytesIO(base64.b64decode(node["value"]))).convert("RGBA")
    if img.size[0] <= 2 or img.size[1] <= 2:
        raise RuntimeError(f"canvas 无效尺寸 {img.size}: {region} {canvas_path}")
    ox = oy = 0
    if "origin" in (node.get("children") or []):
        try:
            ov = client.leaf(region, f"{canvas_path}/origin")
            if isinstance(ov, dict):
                ox = int(ov.get("x") or 0)
                oy = int(ov.get("y") or 0)
        except Exception:
            pass
    client.img_cache[cache_key] = (img, (ox, oy))
    return img, (ox, oy)


def load_tile(client: WzClient, tS: str, u: str, no: int):
    return canvas_and_origin(client, "GMS/83", f"Map/Tile/{tS}.img/{u}/{no}")


class CnSignboardResolver:
    """用 GMS 英文木牌轮廓，在客户端 CN signboard 图集里找中文帧。"""

    def __init__(self):
        self._atlas: Image.Image | None = None
        self._cache: dict[str, tuple[Image.Image, tuple[int, int]]] = {}

    def _ensure_atlas(self) -> Image.Image:
        if self._atlas is not None:
            return self._atlas
        game = find_game_root()
        bundle, key = find_asset(aa_w(game), "signboard_0.png", "spritesheet_*.bundle")
        self._atlas = read_texture_by_key(bundle, key)
        log(f"  已载入 CN signboard 图集 {self._atlas.size}")
        return self._atlas

    def resolve(
        self, en_img: Image.Image, origin: tuple[int, int]
    ) -> tuple[Image.Image, tuple[int, int]]:
        key = f"{en_img.size}:{origin}"
        if key in self._cache:
            return self._cache[key]
        import cv2

        atlas = np.array(self._ensure_atlas())
        en = np.array(en_img.convert("RGBA"))
        en_a = (en[:, :, 3] > 10).astype(np.uint8)
        at_a = (atlas[:, :, 3] > 10).astype(np.uint8)
        # 同轮廓木牌很多（仅文字不同）。英文板与正确中文板像素最接近，
        # 其它中文板 diff 更大。取高轮廓分里 diff 最低且 > 阈值（排除英文原图）。
        res = cv2.matchTemplate(at_a, en_a, cv2.TM_CCOEFF_NORMED)
        h, w = en_a.shape
        cands: list[tuple[float, float, int, int, np.ndarray]] = []
        tmp = res.copy()
        for _ in range(30):
            _, maxv, _, maxl = cv2.minMaxLoc(tmp)
            if maxv < 0.9:
                break
            x, y = maxl
            crop = atlas[y : y + h, x : x + w]
            m = (en[:, :, 3] > 10) & (crop[:, :, 3] > 10)
            if m.sum() < 100:
                diff = 0.0
            else:
                diff = float(
                    np.abs(
                        en[:, :, :3].astype(np.int16) - crop[:, :, :3].astype(np.int16)
                    )[m].mean()
                )
            cands.append((maxv, diff, x, y, crop))
            y0 = max(0, y - h // 2)
            y1 = min(tmp.shape[0], y + h // 2)
            x0 = max(0, x - w // 2)
            x1 = min(tmp.shape[1], x + w // 2)
            tmp[y0:y1, x0:x1] = 0
        localized = [c for c in cands if c[1] >= 5.0]
        if not localized:
            raise RuntimeError("CN 图集中未找到对应木牌")
        _, diff, x, y, crop = min(localized, key=lambda c: c[1])
        img = Image.fromarray(crop)
        log(f"  CN 木牌匹配 atlas@({x},{y}) diff={diff:.1f}")
        self._cache[key] = (img, origin)
        return img, origin


_CN_SIGNS = CnSignboardResolver()


def load_obj(client: WzClient, oS: str, l0: str, l1: str, l2: str):
    base = f"Map/Obj/{oS}.img/{l0}/{l1}/{l2}"
    node = client.get("GMS/83", base)
    if node.get("type") == 12:
        gms_path = base
    else:
        kids = node.get("children") or []
        if not kids:
            raise RuntimeError(f"obj 无帧 {base}")
        frame = "0" if "0" in kids else kids[0]
        gms_path = f"{base}/{frame}"
    en_img, origin = canvas_and_origin(client, "GMS/83", gms_path)
    if oS in CN_OBJ_SETS:
        try:
            return _CN_SIGNS.resolve(en_img, origin)
        except Exception as e:
            log(f"  CN 木牌回退 GMS: {e}")
    return en_img, origin


def load_back(client: WzClient, bS: str, no: int, ani: int = 0):
    folder = "ani" if int(ani or 0) else "back"
    path = f"Map/Back/{bS}.img/{folder}/{no}"
    try:
        return canvas_and_origin(client, "GMS/83", path)
    except Exception:
        return canvas_and_origin(client, "GMS/83", f"{path}/0")


def paste(canvas, sprite, x, y, origin, flip):
    ox, oy = origin
    im = sprite.transpose(Image.FLIP_LEFT_RIGHT) if flip else sprite
    px, py = int(x - ox), int(y - oy)
    if px >= canvas.width or py >= canvas.height or px + im.width <= 0 or py + im.height <= 0:
        return
    canvas.alpha_composite(im, (px, py))


def _tile_axis(center: int, origin: int, size: int, step: int, canvas_size: int) -> list[int]:
    """生成贴图中心坐标，使 paste(p=pos-origin) 覆盖 [0, canvas_size)。"""
    first = center
    while first - origin + size > 0:
        first -= step
    first += step
    out = []
    pos = first
    while pos - origin < canvas_size:
        out.append(pos)
        pos += step
    return out or [center]


def paste_tiled(canvas, sprite, x, y, origin, flip, btype: int, cx: int, cy: int):
    ox, oy = origin
    im = sprite.transpose(Image.FLIP_LEFT_RIGHT) if flip else sprite
    step_x = cx if cx > 0 else max(im.width, 1)
    step_y = cy if cy > 0 else max(im.height, 1)
    tile_h = btype in (1, 3)
    tile_v = btype in (2, 3)
    if not tile_h and not tile_v:
        paste(canvas, sprite, x, y, origin, flip)
        return
    xs = _tile_axis(x, ox, im.width, step_x, canvas.width) if tile_h else [x]
    ys = _tile_axis(y, oy, im.height, step_y, canvas.height) if tile_v else [y]
    for yy in ys:
        for xx in xs:
            paste(canvas, im, xx, yy, origin, False)


def prefetch_sprites(client: WzClient, layout: dict) -> None:
    reqs: list[tuple[str, str]] = []
    for b in layout["back"]:
        folder = "ani" if int(b.get("ani") or 0) else "back"
        reqs.append(("GMS/83", f"Map/Back/{b['bS']}.img/{folder}/{int(b['no'])}"))
    for layer in layout["layers"].values():
        tS = layer.get("tS")
        if tS:
            for t in layer["tiles"]:
                reqs.append(("GMS/83", f"Map/Tile/{tS}.img/{t['u']}/{int(t['no'])}"))
        for o in layer["objs"]:
            base = f"Map/Obj/{o['oS']}.img/{o['l0']}/{o['l1']}/{o['l2']}"
            reqs.append(("GMS/83", base))
    client.get_many(list(dict.fromkeys(reqs)))


def render_map(layout: dict, client: WzClient) -> Image.Image:
    info = layout["info"]
    vr_left, vr_right = int(info["VRLeft"]), int(info["VRRight"])
    vr_top, vr_bottom = int(info["VRTop"]), int(info["VRBottom"])
    pad = 80
    width = vr_right - vr_left
    height = vr_bottom - vr_top
    canvas = Image.new("RGBA", (width + pad * 2, height + pad * 2), (0, 0, 0, 0))
    ox0, oy0 = -vr_left + pad, -vr_top + pad
    missing: dict[str, int] = defaultdict(int)

    def xy(x, y):
        return x + ox0, y + oy0

    backs = sorted(layout["back"], key=lambda b: int(b.get("_i", 0)))

    def draw_backs(front: bool):
        for b in backs:
            if bool(int(b.get("front") or 0)) != front:
                continue
            try:
                spr, origin = load_back(
                    client, str(b["bS"]), int(b["no"]), int(b.get("ani") or 0)
                )
                bx, by = xy(int(b["x"]), int(b["y"]))
                paste_tiled(
                    canvas,
                    spr,
                    bx,
                    by,
                    origin,
                    bool(int(b.get("f") or 0)),
                    int(b.get("type") or 0),
                    int(b.get("cx") or 0),
                    int(b.get("cy") or 0),
                )
            except Exception:
                missing["back"] += 1

    draw_backs(False)
    for L in sorted(layout["layers"], key=int):
        layer = layout["layers"][L]
        tS = layer.get("tS")
        log(f"  层 {L}: tiles={len(layer['tiles'])} objs={len(layer['objs'])} tS={tS}")
        for t in sorted(layer["tiles"], key=lambda t: (int(t.get("zM") or 0), int(t["_i"]))):
            if not tS:
                continue
            try:
                spr, origin = load_tile(client, str(tS), str(t["u"]), int(t["no"]))
                paste(canvas, spr, *xy(int(t["x"]), int(t["y"])), origin, False)
            except Exception:
                missing["tile"] += 1
        for o in sorted(
            layer["objs"],
            key=lambda o: (int(o.get("z") or 0), int(o.get("zM") or 0), int(o["_i"])),
        ):
            try:
                spr, origin = load_obj(
                    client, str(o["oS"]), str(o["l0"]), str(o["l1"]), str(o["l2"])
                )
                paste(
                    canvas,
                    spr,
                    *xy(int(o["x"]), int(o["y"])),
                    origin,
                    bool(int(o.get("f") or 0)),
                )
            except Exception:
                missing["obj"] += 1
    draw_backs(True)
    log(f"missing={dict(missing)}")
    bbox = canvas.getbbox()
    return canvas.crop(bbox) if bbox else canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map_id", type=int)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or (mxd_tools_root() / "assets" / "maps" / str(args.map_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    pad = pad_map_id(args.map_id)
    map_path = f"Map/Map/Map0/{pad}.img"
    cache_path = out_dir / f"map_{args.map_id}_layout_wz.json"

    client = WzClient(workers=24)
    if cache_path.is_file():
        log(f"加载布局缓存 {cache_path}")
        layout = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        log(f"拉取布局 {map_path}")
        layout = walk_map(client, "GMS/83", map_path)
        cache_path.write_text(
            json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        log(f"已缓存 {cache_path}")

    log(
        "layers "
        + str(
            {
                k: (len(v["tiles"]), len(v["objs"]), v["tS"])
                for k, v in layout["layers"].items()
            }
        )
    )
    log("预取精灵…")
    prefetch_sprites(client, layout)
    log("渲染…")
    img = render_map(layout, client)
    out_png = out_dir / f"map_{args.map_id}_render_cn.png"
    img.save(out_png)
    log(f"-> {out_png} size={img.size}")
    meta = {
        "mapId": args.map_id,
        "padId": pad,
        "size": list(img.size),
        "layoutSource": f"https://maplestory.io/api/wz/GMS/83/{map_path}",
        "spritePolicy": "Tile/Back/Obj=GMS/83；signboard=客户端 CN 图集（轮廓匹配中文帧）",
        "output": out_png.name,
    }
    (out_dir / f"map_{args.map_id}_render_cn.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log("完成")


if __name__ == "__main__":
    main()
