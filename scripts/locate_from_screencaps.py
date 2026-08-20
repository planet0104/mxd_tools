#!/usr/bin/env python3
"""从游戏窗口完整截图定位玩家，并标注到完整大地图。

算法：
1. 在截图左上角搜索区，对官方/客户端 minimap 画布做多尺度模板匹配，
   找到小地图「内容区」位置（适配不同分辨率/DPI；小地图可能只显示局部）。
2. 在匹配到的区域内用 #FFFF88/#FFFF00 找菱形玩家点（最密簇）。
3. 若黄点落在小地图天空，向下吸附到最近草地；否则用菱形底边。
4. 画布 → 完整大地图：先把大图缩到与小地图同宽，再 SIFT+FLANN 求单应，
   按缩放因子映回原图；失败则剪影模板 / 内容带映射。

若整张画布匹配分过低（大图滚动视口），回退：按经典 UI 比例枚举视口，
再把视口匹配进画布（与旧 locate 逻辑一致）。

详见 docs/如何从截图定位玩家.md。

用法:
  python -u scripts/locate_from_screencaps.py \\
    --caps screen_caps/彩虹岛-南港西郊平原 \\
    --minimap assets/maps/50001/map_50001_minimap.png \\
    --full assets/maps/50001/map_50001_render_cn.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# 经典客户端：客户区左上角 222 面板内的地图视口（不含窗口标题栏）
CLASSIC_VIEW = (6, 72, 210, 134)


def log(msg: str) -> None:
    print(msg, flush=True)


def find_yellow(rgb: np.ndarray) -> tuple[float, float, float, int] | None:
    """黄菱形：#FFFF88 心 + #FFFF00 边。返回 (cx, cy_mean, cy_bot, n)。"""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    m = (
        ((r >= 248) & (g >= 248) & (b >= 100) & (b <= 155))
        | ((r >= 248) & (g >= 240) & (b <= 45))
        | ((r >= 240) & (g >= 240) & (b <= 165) & (b >= 70))
    )
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    best = None
    for x, y in zip(xs, ys):
        cnt = int(((np.abs(xs - x) <= 2) & (np.abs(ys - y) <= 2)).sum())
        if best is None or cnt > best[0]:
            best = (cnt, int(x), int(y))
    _, bx, by = best
    sel = (np.abs(xs - bx) <= 3) & (np.abs(ys - by) <= 3)
    cx = float(xs[sel].mean())
    cy = float(ys[sel].mean())
    cy_bot = float(ys[sel].max())
    return cx, cy, cy_bot, int(sel.sum())


def _is_sky(p: np.ndarray) -> bool:
    return int(p.mean()) < 18


def _is_grass(p: np.ndarray) -> bool:
    r, g, b = int(p[0]), int(p[1]), int(p[2])
    return g >= r and g > 80 and b < g and (g - b) > 20


def refine_player_on_canvas(
    canvas: np.ndarray, cx: float, cy: float, cy_bot: float
) -> tuple[float, float]:
    """黄点在天空时向下吸附到草地；否则用菱形底边。"""
    ch, cw = canvas.shape[:2]
    x = int(np.clip(round(cx), 0, cw - 1))
    y = int(np.clip(round(cy), 0, ch - 1))
    y0 = int(np.clip(round(cy_bot), 0, ch - 1))
    if not _is_sky(canvas[y, x]):
        return float(cx), float(np.clip(cy_bot, 0, ch - 1))
    for dy in range(0, 15):
        yy = min(ch - 1, y0 + dy)
        run = sum(
            1
            for xx in range(max(0, x - 2), min(cw, x + 3))
            if _is_grass(canvas[yy, xx])
        )
        if run >= 2:
            return float(x), float(yy)
    return float(cx), float(np.clip(cy_bot, 0, ch - 1))


def _canvas_y_band(canvas: np.ndarray) -> tuple[float, float]:
    dens = (canvas.mean(axis=2) > 20).mean(axis=1)
    idx = np.where(dens > 0.08)[0]
    if len(idx) == 0:
        return 0.0, float(canvas.shape[0] - 1)
    return float(idx.min()), float(idx.max())


def _full_y_band(full: np.ndarray) -> tuple[float, float]:
    r, g, b = full[:, :, 0].astype(np.int16), full[:, :, 1].astype(np.int16), full[:, :, 2].astype(np.int16)
    sky = (b > 140) & (b > r + 15) & (b > g + 10)
    black = (r + g + b) < 40
    dens = ((~sky) & (~black)).mean(axis=1)
    idx = np.where(dens > 0.04)[0]
    if len(idx) == 0:
        return 0.0, float(full.shape[0] - 1)
    return float(idx.min()), float(idx.max())


def _content_band_map(
    canvas: np.ndarray, full: np.ndarray, cx: float, cy: float
) -> tuple[float, float]:
    fh, fw = full.shape[:2]
    ch, cw = canvas.shape[:2]
    fx = cx / cw * fw
    mt, mb = _canvas_y_band(canvas)
    ft, fb = _full_y_band(full)
    fy = ft + (cy - mt) / max(1.0, mb - mt) * (fb - ft)
    return float(np.clip(fx, 0, fw - 1)), float(np.clip(fy, 0, fh - 1))


def _sil_u8(rgb: np.ndarray, painted: bool) -> np.ndarray:
    if not painted:
        return (rgb.mean(axis=2) > 12).astype(np.uint8) * 255
    r, g, b = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
    sky = (b > 130) & (b > r + 10) & (b > g + 5)
    black = (r + g + b) < 40
    return ((~sky) & (~black)).astype(np.uint8) * 255


class CanvasToFullAlign:
    """先把大地图缩到与小地图同宽，再 SIFT+FLANN 求变换，最后按缩放因子映回原图。"""

    def __init__(self, canvas: np.ndarray, full: np.ndarray, boost: int = 3):
        self.canvas = canvas
        self.full = full
        self.boost = boost
        ch, cw = canvas.shape[:2]
        fh, fw = full.shape[:2]
        self.fm_w = cw
        self.fm_h = max(1, int(round(fh * cw / fw)))
        self.sx_back = fw / self.fm_w
        self.sy_back = fh / self.fm_h
        self.mode = "content_band"
        self.H = None  # mini_boost -> full_small_boost
        self.tmpl_xy = (0, 0)
        self._build()

    def _build(self) -> None:
        full_s = cv2.resize(
            self.full, (self.fm_w, self.fm_h), interpolation=cv2.INTER_AREA
        )
        b = self.boost
        mini_b = cv2.resize(
            cv2.cvtColor(self.canvas, cv2.COLOR_RGB2GRAY),
            (self.fm_w * b, self.canvas.shape[0] * b),
            interpolation=cv2.INTER_NEAREST,
        )
        full_b = cv2.resize(
            cv2.cvtColor(full_s, cv2.COLOR_RGB2GRAY),
            (self.fm_w * b, self.fm_h * b),
            interpolation=cv2.INTER_AREA,
        )
        if self._try_sift(mini_b, full_b):
            return
        # 回退：同尺度剪影模板匹配（只估平移）
        ms = _sil_u8(self.canvas, painted=False)
        fs = _sil_u8(full_s, painted=True)
        res = cv2.matchTemplate(fs, ms, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv >= 0.12:
            self.mode = "sil_template"
            self.tmpl_xy = (int(ml[0]), int(ml[1]))
            return
        self.mode = "content_band"

    def _try_sift(self, mini_b: np.ndarray, full_b: np.ndarray) -> bool:
        sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02)
        k1, d1 = sift.detectAndCompute(mini_b, None)
        k2, d2 = sift.detectAndCompute(full_b, None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return False
        flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=100)
        )
        knn = flann.knnMatch(d1, d2, k=2)
        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.8 * n.distance:
                good.append(m)
        if len(good) < 8:
            return False
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or inl is None or int(inl.sum()) < 8:
            return False
        self.H = H
        self.mode = f"sift_flann(inliers={int(inl.sum())})"
        return True

    def map_xy(self, cx: float, cy: float) -> tuple[float, float]:
        fh, fw = self.full.shape[:2]
        if self.mode.startswith("sift_flann") and self.H is not None:
            b = float(self.boost)
            pt = np.array([[[cx * b, cy * b]]], dtype=np.float32)
            p = cv2.perspectiveTransform(pt, self.H)[0, 0]
            fx = float(p[0] / b * self.sx_back)
            fy = float(p[1] / b * self.sy_back)
        elif self.mode == "sil_template":
            ox, oy = self.tmpl_xy
            fx = (ox + cx) * self.sx_back
            fy = (oy + cy) * self.sy_back
        else:
            return _content_band_map(self.canvas, self.full, cx, cy)
        return float(np.clip(fx, 0, fw - 1)), float(np.clip(fy, 0, fh - 1))


def canvas_xy_to_full(
    canvas: np.ndarray, full: np.ndarray, cx: float, cy: float,
    align: CanvasToFullAlign | None = None,
) -> tuple[float, float]:
    if align is None:
        align = CanvasToFullAlign(canvas, full)
    return align.map_xy(cx, cy)


def _match_scaled_canvas(
    search_g: np.ndarray, canvas_g: np.ndarray, mask0: np.ndarray, scale: float
) -> tuple[float, int, int, int, int] | None:
    ch, cw = canvas_g.shape
    sh, sw = search_g.shape
    tw = int(round(cw * scale))
    th = int(round(ch * scale))
    if tw < 40 or th < 24 or tw >= sw - 2 or th >= sh - 2:
        return None
    templ = cv2.resize(canvas_g, (tw, th), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask0, (tw, th), interpolation=cv2.INTER_NEAREST)
    if int(mask.sum()) < 80:
        return None
    res = cv2.matchTemplate(search_g, templ, cv2.TM_CCOEFF_NORMED, mask=mask)
    _, mv, _, ml = cv2.minMaxLoc(res)
    return float(mv), int(ml[0]), int(ml[1]), tw, th


def find_canvas_on_screen(
    shot: np.ndarray, canvas: np.ndarray
) -> dict | None:
    """多尺度：把完整 minimap 画布匹配到截图左上角。"""
    h, w = shot.shape[:2]
    sw = max(220, min(w, int(w * 0.48)))
    sh = max(220, min(h, int(h * 0.58)))
    search = shot[:sh, :sw]
    sg = cv2.cvtColor(search, cv2.COLOR_RGB2GRAY)
    ym = (
        (search[:, :, 0] >= 240)
        & (search[:, :, 1] >= 240)
        & (search[:, :, 2] <= 170)
    )
    sg = sg.copy()
    sg[ym] = 0

    cg = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    mask0 = (cg > 12).astype(np.uint8)

    best = None
    scales = list(np.linspace(0.7, 2.8, 55))
    if 1.0 not in scales:
        scales.append(1.0)
    for scale in scales:
        hit = _match_scaled_canvas(sg, cg, mask0, float(scale))
        if hit is None:
            continue
        mv, x, y, tw, th = hit
        if best is None or mv > best["score"]:
            best = {
                "mode": "canvas_on_screen",
                "score": mv,
                "scale": float(scale),
                "x": x,
                "y": y,
                "w": tw,
                "h": th,
            }
    return best


def view_to_canvas_align(view: np.ndarray, canvas: np.ndarray) -> dict | None:
    """视口 → 画布对齐（支持滚动/缩放）。"""
    vg = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
    ym = (
        (view[:, :, 0] >= 240)
        & (view[:, :, 1] >= 240)
        & (view[:, :, 2] <= 160)
    )
    vg = vg.copy()
    vg[ym] = 0
    cg = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    vh, vw = vg.shape
    ch, cw = cg.shape
    best = None

    def consider(mode: str, score: float, x: int, y: int, sx: float, sy: float):
        nonlocal best
        cand = {
            "mode": mode,
            "score": float(score),
            "loc": (int(x), int(y)),
            "scale": (float(sx), float(sy)),
        }
        if best is None or cand["score"] > best["score"]:
            best = cand

    if vw <= cw and vh <= ch:
        res = cv2.matchTemplate(cg, vg, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        consider("view", mv, ml[0], ml[1], 1.0, 1.0)

    nw = max(8, int(round(vw * ch / max(vh, 1))))
    if nw <= cw:
        scaled = cv2.resize(vg, (nw, ch), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(cg, scaled, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        consider("hscroll", mv, ml[0], ml[1], nw / vw, ch / vh)

    nh = max(8, int(round(vh * cw / max(vw, 1))))
    if nh <= ch:
        scaled = cv2.resize(vg, (cw, nh), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(cg, scaled, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        consider("vscroll", mv, ml[0], ml[1], cw / vw, nh / vh)

    if cw <= vw and ch <= vh:
        res = cv2.matchTemplate(vg, cg, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        consider("canvas_in_view", mv, ml[0], ml[1], 1.0, 1.0)

    return best


def find_view_fallback(shot: np.ndarray, canvas: np.ndarray) -> dict | None:
    """画布整图匹配不佳时：枚举经典视口比例，选对齐分最高者。"""
    best = None
    h, w = shot.shape[:2]
    for ui_scale in np.linspace(0.75, 1.5, 16):
        vx = int(round(CLASSIC_VIEW[0] * ui_scale))
        vy = int(round(CLASSIC_VIEW[1] * ui_scale))
        vw = int(round(CLASSIC_VIEW[2] * ui_scale))
        vh = int(round(CLASSIC_VIEW[3] * ui_scale))
        for y0 in range(0, 48, 2):
            for x0 in range(0, 8):
                x = x0 + vx
                y = y0 + vy
                if x + vw > w or y + vh > h:
                    continue
                view = shot[y : y + vh, x : x + vw]
                if find_yellow(view) is None:
                    continue
                align = view_to_canvas_align(view, canvas)
                if align is None:
                    continue
                score = align["score"]
                if best is None or score > best["score"]:
                    best = {
                        "mode": "view_fallback",
                        "score": score,
                        "scale": float(ui_scale),
                        "x": x,
                        "y": y,
                        "w": vw,
                        "h": vh,
                        "align": align,
                    }
    return best


def locate_one(
    shot: np.ndarray,
    canvas: np.ndarray,
    full: np.ndarray,
    map_align: CanvasToFullAlign | None = None,
) -> dict:
    hit = find_canvas_on_screen(shot, canvas)
    use_fallback = hit is None or hit["score"] < 0.42
    if use_fallback:
        fb = find_view_fallback(shot, canvas)
        if fb is not None and (hit is None or fb["score"] > hit["score"] + 0.05):
            hit = fb

    if hit is None:
        return {"ok": False, "error": "未在左上角匹配到小地图"}

    x, y, bw, bh = hit["x"], hit["y"], hit["w"], hit["h"]
    region = shot[y : y + bh, x : x + bw]
    yel = find_yellow(region)
    if yel is None:
        return {
            "ok": False,
            "error": "小地图区域内未找到玩家黄点",
            "hit": {k: hit[k] for k in hit if k != "align"},
        }

    lx, ly, ly_bot, n = yel
    ch, cw = canvas.shape[:2]

    if hit["mode"] == "canvas_on_screen":
        sc = hit["scale"]
        cx = lx / sc
        cy = ly / sc
        cy_bot = ly_bot / sc
        align_mode = "canvas_on_screen"
        align_score = hit["score"]
    else:
        align = hit["align"]
        ax, ay = align["loc"]
        sx, sy = align["scale"]
        if align["mode"] == "canvas_in_view":
            cx = (lx - ax) / sx
            cy = (ly - ay) / sy
            cy_bot = (ly_bot - ay) / sy
        else:
            cx = ax + lx * sx
            cy = ay + ly * sy
            cy_bot = ay + ly_bot * sy
        align_mode = align["mode"]
        align_score = align["score"]

    cx, cy = refine_player_on_canvas(canvas, cx, cy, cy_bot)
    cx = float(np.clip(cx, 0, cw - 1))
    cy = float(np.clip(cy, 0, ch - 1))
    if map_align is None:
        map_align = CanvasToFullAlign(canvas, full)
    fx, fy = map_align.map_xy(cx, cy)

    return {
        "ok": True,
        "hit_mode": hit["mode"],
        "match_score": round(float(hit["score"]), 4),
        "align_mode": align_mode,
        "align_score": round(float(align_score), 4),
        "map_align": map_align.mode,
        "box": [x, y, bw, bh],
        "ui_scale": round(float(hit.get("scale", 1.0)), 4),
        "yellow_in_box": [round(lx, 2), round(ly, 2), n],
        "canvas_xy": [round(cx, 2), round(cy, 2)],
        "full_xy": [round(fx, 2), round(fy, 2)],
    }


def mark_full(full: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """淡蓝色大空心菱形标注玩家。"""
    out = full.copy()
    xi, yi = int(round(fx)), int(round(fy))
    h, w = out.shape[:2]
    half = int(np.clip(round(min(w, h) * 0.028), 22, 48))
    color = (80, 220, 255)  # RGB 淡青蓝
    pts = np.array(
        [
            [xi, yi - half],
            [xi + half, yi],
            [xi, yi + half],
            [xi - half, yi],
        ],
        dtype=np.int32,
    )
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=9, lineType=cv2.LINE_AA)
    return out


def mark_tl(shot: np.ndarray, box: list[int], yel_abs: tuple[float, float]) -> np.ndarray:
    h = min(340, shot.shape[0])
    w = min(340, shot.shape[1])
    tl = shot[:h, :w].copy()
    x, y, bw, bh = box
    cv2.rectangle(tl, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
    # 小地图上也用同色空心菱形（略小）
    xi, yi = int(round(yel_abs[0])), int(round(yel_abs[1]))
    half = 14
    pts = np.array(
        [[xi, yi - half], [xi + half, yi], [xi, yi + half], [xi - half, yi]],
        dtype=np.int32,
    )
    cv2.polylines(tl, [pts], True, (80, 220, 255), 2, cv2.LINE_AA)
    return tl


def main() -> int:
    ap = argparse.ArgumentParser(description="截图小地图匹配 + 玩家标注验证")
    ap.add_argument("--caps", type=Path, required=True, help="完整窗口截图目录")
    ap.add_argument("--minimap", type=Path, required=True, help="minimap 画布 PNG")
    ap.add_argument("--full", type=Path, required=True, help="完整大地图 PNG")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录（默认 tmp/screen_cap_locate）",
    )
    args = ap.parse_args()

    caps_dir = args.caps
    if not caps_dir.is_dir():
        log(f"截图目录不存在: {caps_dir}")
        return 2

    shots = sorted({p.resolve(): p for p in caps_dir.glob("*.png")}.values(), key=lambda p: p.name.lower())
    if not shots:
        log(f"目录内无 PNG: {caps_dir}")
        return 2

    canvas = np.array(Image.open(args.minimap).convert("RGB"))
    full = np.array(Image.open(args.full).convert("RGB"))
    map_align = CanvasToFullAlign(canvas, full)
    out = args.out or (
        Path(__file__).resolve().parents[1] / "tmp" / "screen_cap_locate"
    )
    out.mkdir(parents=True, exist_ok=True)

    log(
        f"截图 {len(shots)} 张 | minimap {canvas.shape[1]}x{canvas.shape[0]} | "
        f"full {full.shape[1]}x{full.shape[0]} | map_align={map_align.mode}"
    )
    rows = []
    ok_n = 0
    for i, path in enumerate(shots):
        shot = np.array(Image.open(path).convert("RGB"))
        rec = locate_one(shot, canvas, full, map_align=map_align)
        rec["file"] = path.name
        rows.append(rec)
        if not rec.get("ok"):
            log(f"[{i}] FAIL {path.name}: {rec.get('error')}")
            continue
        ok_n += 1
        box = rec["box"]
        lx, ly, _ = rec["yellow_in_box"]
        tl = mark_tl(shot, box, (box[0] + lx, box[1] + ly))
        marked = mark_full(full, rec["full_xy"][0], rec["full_xy"][1])
        Image.fromarray(tl).save(out / f"{i:02d}_tl_{path.stem}.png")
        Image.fromarray(marked).save(out / f"{i:02d}_full_{path.stem}.png")
        log(
            f"[{i}] OK {path.name} score={rec['match_score']:.3f} "
            f"mode={rec['hit_mode']} full=({rec['full_xy'][0]:.0f},{rec['full_xy'][1]:.0f})"
        )

    summary = {
        "caps": str(caps_dir),
        "minimap": str(args.minimap),
        "full": str(args.full),
        "total": len(shots),
        "ok": ok_n,
        "results": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"完成 {ok_n}/{len(shots)} → {out}")
    return 0 if ok_n == len(shots) else 1


if __name__ == "__main__":
    sys.exit(main())
