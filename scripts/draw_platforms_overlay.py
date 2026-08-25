#!/usr/bin/env python3
"""把 map_50001_platforms.json 的平台/绳梯/刷怪点叠画到地图上，便于核对。"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "assets/maps/50001"
PLAT = MAP_DIR / "map_50001_platforms.json"
OUT = ROOT / "tmp/platform_overlay_50001.png"


def main() -> None:
    data = json.loads(PLAT.read_text(encoding="utf-8"))
    img_path = MAP_DIR / data["image"]
    im = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 平台：青绿线段（可站立）
    for p in data.get("platforms", []):
        x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
        # 竖直/过短用橙色，水平用青绿
        if abs(x2 - x1) < 8:
            color = (255, 140, 0, 220)
            w = 2
        else:
            color = (0, 255, 120, 230)
            w = 3
        draw.line([(x1, y1), (x2, y2)], fill=color, width=w)

    # 斜坡
    for s in data.get("slopes", []):
        pts = [(float(p[0]), float(p[1])) for p in s["points"]]
        if len(pts) >= 2:
            draw.polygon(pts, outline=(255, 80, 255, 200))
            # 顶边加粗
            best = min(
                range(len(pts)),
                key=lambda i: (pts[i][1] + pts[(i + 1) % len(pts)][1]) * 0.5,
            )
            a, b = pts[best], pts[(best + 1) % len(pts)]
            draw.line([a, b], fill=(255, 0, 255, 255), width=4)

    # 绳/梯
    for r in data.get("ropes", []):
        x, y1, y2 = r["x"], r["y1"], r["y2"]
        kind = r.get("kind", "rope")
        col = (80, 180, 255, 230) if kind == "rope" else (255, 220, 60, 230)
        half = float(r.get("width", 16)) * 0.5
        draw.rectangle([x - half, min(y1, y2), x + half, max(y1, y2)], outline=col, width=2)
        draw.line([(x, y1), (x, y2)], fill=col, width=2)

    # 刷怪点
    for sp in data.get("spawns", []):
        x, y = sp["x"], sp["y"]
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 60, 60, 220), outline=(255, 255, 255, 255))

    # 图例
    try:
        font = ImageFont.truetype("msyh.ttc", 22)
        font_s = ImageFont.truetype("msyh.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
        font_s = font

    legend = Image.new("RGBA", (420, 160), (0, 0, 0, 180))
    ld = ImageDraw.Draw(legend)
    ld.text((12, 8), "platforms.json 碰撞示意（网站/WZ 原版）", fill=(255, 255, 255, 255), font=font)
    ld.line([(12, 48), (60, 48)], fill=(0, 255, 120, 255), width=3)
    ld.text((70, 38), "水平平台（站立面）", fill=(220, 255, 230, 255), font=font_s)
    ld.line([(12, 78), (60, 78)], fill=(255, 140, 0, 255), width=2)
    ld.text((70, 68), "竖直/过短线段", fill=(255, 220, 180, 255), font=font_s)
    ld.rectangle([12, 100, 28, 130], outline=(80, 180, 255, 255), width=2)
    ld.text((70, 105), "绳 / 梯", fill=(180, 220, 255, 255), font=font_s)
    ld.ellipse([14, 138, 26, 150], fill=(255, 60, 60, 255))
    ld.text((70, 132), "刷怪点", fill=(255, 180, 180, 255), font=font_s)
    overlay.paste(legend, (20, 20), legend)

    out = Image.alpha_composite(im, overlay).convert("RGB")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT, quality=92)
    print(f"saved {OUT}")
    print(f"platforms={len(data.get('platforms',[]))} ropes={len(data.get('ropes',[]))} spawns={len(data.get('spawns',[]))}")


if __name__ == "__main__":
    main()
