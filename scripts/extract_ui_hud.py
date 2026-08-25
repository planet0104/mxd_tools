#!/usr/bin/env python3
"""从怀旧服客户端抽取 mini_game 用的 HUD 底框（写入 assets/ui_game，不碰 assets/ui）。

输出到 assets/ui_game/：
  - panel_frame.png      StatusBar LayerSkin/Background backgrnd
  - keyboard_frame.png   StatusBar ShortCutKeys quickSlot
  - minimap_frame.png    MiniMap MaximizeBackground 九宫格拉伸合成
  - slices/              九宫格原片（便于以后改尺寸）
  - ui_layout.json       mini_game 专用布局（仅 ui_game）

用法：
  python scripts/extract_ui_hud.py
  python scripts/extract_ui_hud.py --game-root \"D:/.../mxdclassic\"
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import UnityPy
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "assets" / "ui_game"


def read_game_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    tip = ROOT.parent / "安装位置.txt"
    if tip.is_file():
        return Path(tip.read_text(encoding="utf-8").strip())
    raise SystemExit("请提供 --game-root 或在仓库根目录写 安装位置.txt")


def aa_w(game_root: Path) -> Path:
    return game_root / "Maplestory_Classic_Data" / "StreamingAssets" / "aa" / "w"


def find_hud_prefab(w_dir: Path) -> Path:
    for p in sorted(w_dir.glob("prefab_*.bundle")):
        data = p.read_bytes()
        if b"UIMiniMap" in data and b"UIStatusBar" in data:
            return p
    raise FileNotFoundError("找不到含 UIMiniMap/UIStatusBar 的 prefab_*.bundle")


def load_env(bundle: Path):
    env = UnityPy.load(str(bundle))
    objs = {o.path_id: o for o in env.objects}
    return env, objs


def find_gos(objs, env, name: str):
    hits = []
    for obj in env.objects:
        if obj.type.name != "GameObject":
            continue
        t = obj.read_typetree()
        if t.get("m_Name") == name:
            hits.append((obj.path_id, t))
    return hits


def transform_of(objs, got):
    for c in got.get("m_Component") or []:
        pid = c["component"]["m_PathID"]
        o = objs[pid]
        if o.type.name in ("Transform", "RectTransform"):
            return o.read_typetree()
    return None


def children_gos(objs, got):
    tr = transform_of(objs, got)
    if not tr:
        return []
    out = []
    for ch in tr.get("m_Children") or []:
        cpid = ch["m_PathID"]
        ct = objs[cpid].read_typetree()
        goid = ct["m_GameObject"]["m_PathID"]
        out.append((goid, objs[goid].read_typetree()))
    return out


def sprites_on_go(objs, got) -> list[tuple[str, Image.Image]]:
    found: list[tuple[str, Image.Image]] = []
    seen = set()
    for c in got.get("m_Component") or []:
        pid = c["component"]["m_PathID"]
        raw = objs[pid].get_raw_data()
        for off in range(0, len(raw) - 8, 4):
            spid = struct.unpack_from("<q", raw, off)[0]
            if spid in seen or spid not in objs:
                continue
            o = objs[spid]
            if o.type.name not in ("Sprite", "Texture2D"):
                continue
            seen.add(spid)
            d = o.read()
            name = getattr(d, "m_Name", "") or "spr"
            try:
                img = d.image.convert("RGBA")
            except Exception:
                continue
            found.append((name, img))
    return found


def find_child(objs, got, name: str):
    for _gid, gt in children_gos(objs, got):
        if gt.get("m_Name") == name:
            return gt
    return None


def first_sprite(objs, got, prefer: str | None = None) -> Image.Image | None:
    sprs = sprites_on_go(objs, got)
    if not sprs:
        return None
    if prefer:
        for n, im in sprs:
            if n == prefer:
                return im
    return sprs[0][1]


def collect_9slice(objs, bg_go) -> dict[str, Image.Image]:
    """从 MaximizeBackground / MediumBackground 子节点收集 nw..se。"""
    pieces = {}
    for _gid, gt in children_gos(objs, bg_go):
        key = (gt.get("m_Name") or "").lower()
        if key not in ("nw", "n", "ne", "w", "c", "e", "sw", "s", "se"):
            continue
        im = first_sprite(objs, gt, prefer=key)
        if im is not None:
            pieces[key] = im
    return pieces


def compose_9slice(pieces: dict[str, Image.Image], width: int, height: int) -> Image.Image:
    need = ("nw", "n", "ne", "w", "c", "e", "sw", "s", "se")
    missing = [k for k in need if k not in pieces]
    if missing:
        raise ValueError(f"九宫格缺少: {missing}")

    nw, n, ne = pieces["nw"], pieces["n"], pieces["ne"]
    w, c, e = pieces["w"], pieces["c"], pieces["e"]
    sw, s, se = pieces["sw"], pieces["s"], pieces["se"]

    top_h = nw.height
    bot_h = sw.height
    left_w = nw.width
    right_w = ne.width
    mid_w = width - left_w - right_w
    mid_h = height - top_h - bot_h
    if mid_w < 1 or mid_h < 1:
        raise ValueError(f"目标尺寸过小: {width}x{height}")

    out = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    def tile(src: Image.Image, box: tuple[int, int, int, int]):
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            return
        patched = Image.new("RGBA", (bw, bh))
        for yy in range(0, bh, src.height):
            for xx in range(0, bw, src.width):
                patched.paste(src, (xx, yy))
        out.paste(patched, (x0, y0), patched)

    out.paste(nw, (0, 0), nw)
    tile(n, (left_w, 0, left_w + mid_w, top_h))
    out.paste(ne, (left_w + mid_w, 0), ne)

    tile(w, (0, top_h, left_w, top_h + mid_h))
    tile(c, (left_w, top_h, left_w + mid_w, top_h + mid_h))
    tile(e, (left_w + mid_w, top_h, width, top_h + mid_h))

    out.paste(sw, (0, top_h + mid_h), sw)
    tile(s, (left_w, top_h + mid_h, left_w + mid_w, height))
    out.paste(se, (left_w + mid_w, top_h + mid_h), se)
    return out


def extract(game_root: Path, out_dir: Path, minimap_size: tuple[int, int]) -> None:
    w_dir = aa_w(game_root)
    prefab = find_hud_prefab(w_dir)
    print(f"prefab: {prefab.name}")
    env, objs = load_env(prefab)

    out_dir.mkdir(parents=True, exist_ok=True)
    slices = out_dir / "slices"
    slices.mkdir(exist_ok=True)

    # --- panel ---
    layer_skin = find_gos(objs, env, "LayerSkin")
    panel_img = None
    panel2_img = None
    for _gid, got in layer_skin:
        bg = find_child(objs, got, "Background")
        if bg:
            panel_img = first_sprite(objs, bg, prefer="backgrnd") or first_sprite(objs, bg)
        bg2 = find_child(objs, got, "Background2")
        if bg2:
            panel2_img = first_sprite(objs, bg2, prefer="backgrnd2") or first_sprite(objs, bg2)
        if panel_img is not None:
            break
    if panel_img is None:
        raise RuntimeError("未找到 StatusBar LayerSkin/Background backgrnd")
    panel_img.save(out_dir / "panel_frame.png")
    print(f"  panel_frame.png {panel_img.size}")
    if panel2_img is not None:
        panel2_img.save(slices / "panel_backgrnd2.png")

    # --- keyboard ---
    kb_img = None
    for _gid, got in find_gos(objs, env, "ShortCutKeys"):
        kb_img = first_sprite(objs, got, prefer="quickSlot") or first_sprite(objs, got)
        if kb_img is not None:
            break
    if kb_img is None:
        raise RuntimeError("未找到 ShortCutKeys/quickSlot")
    kb_img.save(out_dir / "keyboard_frame.png")
    print(f"  keyboard_frame.png {kb_img.size}")

    # --- minimap 9-slice ---
    pieces = {}
    for _gid, got in find_gos(objs, env, "MaximizeMap"):
        for bg_name in ("MaximizeBackground", "MediumBackground"):
            bg = find_child(objs, got, bg_name)
            if not bg:
                continue
            p = collect_9slice(objs, bg)
            if len(p) >= 9 and bg_name == "MaximizeBackground":
                pieces = p
            # 保存两套切片
            for k, im in p.items():
                im.save(slices / f"minimap_{bg_name}_{k}.png")
        if pieces:
            break
    if len(pieces) < 9:
        raise RuntimeError(f"MiniMap 九宫格不完整: {sorted(pieces)}")

    mw, mh = minimap_size
    mini = compose_9slice(pieces, mw, mh)
    mini.save(out_dir / "minimap_frame.png")
    print(f"  minimap_frame.png {mini.size} (from MaximizeBackground)")

    # gauge 空框（可选，方便以后画血条）
    for _gid, got in find_gos(objs, env, "Graduation"):
        g = first_sprite(objs, got, prefer="graduation")
        if g is not None:
            g.save(slices / "gauge_graduation.png")
            print(f"  slices/gauge_graduation.png {g.size}")
            break

    panel_w, panel_h = panel_img.size
    kb_w, kb_h = kb_img.size
    win_w, win_h = 1368, 768
    panel_x = (win_w - panel_w) // 2
    panel_y = win_h - panel_h
    kb_x = panel_x + panel_w - kb_w
    kb_y = panel_y - kb_h

    layout = {
        "window": [win_w, win_h],
        "note": "仅 assets/ui_game：客户端原版 HUD 底框。面板底边贴窗口底，键盘贴面板右上角外侧。",
        "world_height": panel_y,
        "source_prefab": prefab.name,
        "widgets": {
            "minimap": {
                "file": "minimap_frame.png",
                "x": 6,
                "y": 6,
                "w": mw,
                "h": mh,
            },
            "panel": {
                "file": "panel_frame.png",
                "x": panel_x,
                "y": panel_y,
                "w": panel_w,
                "h": panel_h,
            },
            "keyboard": {
                "file": "keyboard_frame.png",
                "x": kb_x,
                "y": kb_y,
                "w": kb_w,
                "h": kb_h,
            },
        },
        "dynamic_overlay": {
            "hp_bar": {"x": panel_x + 42, "y": panel_y + 12, "w": 120, "h": 12},
            "mp_bar": {"x": panel_x + 42, "y": panel_y + 28, "w": 120, "h": 12},
            "player_name": {"x": panel_x - 77, "y": panel_y + 10, "w": 100, "h": 14},
            "hotbar_slots": [
                {"slot": i + 1, "x": panel_x + 204 + i * 36, "y": panel_y + 14, "w": 32, "h": 32}
                for i in range(6)
            ],
            "inventory_button": {"x": panel_x + 744, "y": panel_y + 14, "w": 40, "h": 40},
        },
        "inventory_window": {
            "x": 484,
            "y": 200,
            "w": 400,
            "h": 360,
            "cols": 4,
            "rows": 6,
            "slot_size": 40,
        },
    }
    (out_dir / "ui_layout.json").write_text(
        json.dumps(layout, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.txt").write_text(
        "mini_game HUD 底框（客户端原版资源）\n"
        f"来源 prefab: {prefab.name}\n"
        "- panel_frame.png: UIStatusBar/LayerSkin/Background backgrnd\n"
        "- keyboard_frame.png: ShortCutKeys quickSlot\n"
        "- minimap_frame.png: MaximizeMap/MaximizeBackground 九宫格合成\n"
        "- slices/: 原始切片\n"
        "注意: 不要把这些图放进 assets/ui（该目录截图供 YOLO 训练）。\n",
        encoding="utf-8",
    )
    print(f"-> {out_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="抽取 mini_game HUD 底框到 assets/ui_game")
    ap.add_argument("--game-root", default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--minimap-w", type=int, default=136)
    ap.add_argument("--minimap-h", type=int, default=161)
    args = ap.parse_args(argv)
    game_root = read_game_root(args.game_root)
    extract(game_root, args.out, (args.minimap_w, args.minimap_h))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
