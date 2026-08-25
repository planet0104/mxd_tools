#!/usr/bin/env python3
"""对比 dataset 手工标注与 assets/platforms.json。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "dataset/彩虹岛-南港西郊平原/map_50001_render_cn.json"
PLAT = ROOT / "assets/maps/50001/map_50001_platforms.json"


def rect_to_seg(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), max(xs), min(ys), max(ys)


def main():
    ann = json.loads(ANN.read_text(encoding="utf-8"))
    plat = json.loads(PLAT.read_text(encoding="utf-8"))

    floors = []
    ds_ropes = []
    for s in ann["shapes"]:
        xlo, xhi, yt, yb = rect_to_seg(s["points"])
        if s["label"] == "地板":
            floors.append((xlo, xhi, yb, s["shape_type"]))
        elif s["label"] in ("绳子", "梯子"):
            kind = "ladder" if s["label"] == "梯子" else "rope"
            ds_ropes.append(((xlo + xhi) / 2, yt, yb, kind))

    plat_horiz = []
    for p in plat["platforms"]:
        if abs(p["y1"] - p["y2"]) < 2:
            y = (p["y1"] + p["y2"]) / 2
            xmin, xmax = sorted([p["x1"], p["x2"]])
            plat_horiz.append((xmin, xmax, y))

    print(f"dataset: {len(floors)} floors, {len(ds_ropes)} ropes/ladder")
    print(f"platforms.json: {len(plat['platforms'])} segs, {len(plat['ropes'])} ropes")

    print("\n--- rope alignment ---")
    for cx, yt, yb, kind in ds_ropes:
        best = None
        for r in plat["ropes"]:
            d = abs(r["x"] - cx) + abs((r["y1"] + r["y2"]) / 2 - (yt + yb) / 2)
            if best is None or d < best[0]:
                best = (d, r)
        r = best[1]
        ok = best[0] < 40 and r["kind"] == kind
        mark = "OK" if ok else "DIFF"
        print(
            f"  [{mark}] ds {kind} x={cx:.0f} y=[{yt:.0f},{yb:.0f}] "
            f"-> json x={r['x']} y=[{r['y1']},{r['y2']}] dist={best[0]:.0f}"
        )

    print("\n--- floor y alignment (stand = rect bottom) ---")
    matched = 0
    for xlo, xhi, sy, stype in sorted(floors, key=lambda t: t[2]):
        cands = [
            ph
            for ph in plat_horiz
            if abs(ph[2] - sy) < 25 and ph[1] > xlo + 10 and ph[0] < xhi - 10
        ]
        if cands:
            matched += 1
            ys = sorted({round(c[2]) for c in cands})
            print(f"  OK  ds y={sy:.0f} x=[{xlo:.0f},{xhi:.0f}] ({stype}) -> plat y {ys[:6]}")
        else:
            print(f"  MISS ds y={sy:.0f} x=[{xlo:.0f},{xhi:.0f}] ({stype})")

    print(f"\nfloor match: {matched}/{len(floors)}")


if __name__ == "__main__":
    main()
