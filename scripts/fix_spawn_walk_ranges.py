#!/usr/bin/env python3
"""按同高度连续平台重算 map_50001_platforms.json 的 walk_x1/walk_x2。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAT = ROOT / "assets/maps/50001/map_50001_platforms.json"
Y_TOL = 4.0
GAP = 8.0
EDGE_PAD = 8.0
MIN_HALF = 20.0


def merge_horiz(platforms: list[dict], y: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for pl in platforms:
        if abs(pl["x2"] - pl["x1"]) < 8:
            continue
        if abs(pl["y2"] - pl["y1"]) >= 2:
            continue
        py = (pl["y1"] + pl["y2"]) * 0.5
        if abs(py - y) > Y_TOL:
            continue
        spans.append(tuple(sorted((pl["x1"], pl["x2"]))))
    if not spans:
        return []
    spans.sort()
    out: list[tuple[float, float]] = []
    lo, hi = spans[0]
    for a, b in spans[1:]:
        if a <= hi + GAP:
            hi = max(hi, b)
        else:
            out.append((lo, hi))
            lo, hi = a, b
    out.append((lo, hi))
    return out


def walk_for_spawn(platforms: list[dict], x: float, y: float) -> tuple[float, float]:
    segs = merge_horiz(platforms, y)
    for lo, hi in segs:
        if lo - 4 <= x <= hi + 4:
            w1 = lo + EDGE_PAD
            w2 = hi - EDGE_PAD
            if w2 - w1 < MIN_HALF * 2:
                mid = (lo + hi) * 0.5
                w1, w2 = mid - MIN_HALF, mid + MIN_HALF
            return round(w1, 1), round(w2, 1)
    # 找不到平台：以刷怪点为中心小范围
    return round(x - MIN_HALF, 1), round(x + MIN_HALF, 1)


def main() -> None:
    data = json.loads(PLAT.read_text(encoding="utf-8"))
    for sp in data.get("spawns", []):
        old = (sp["walk_x1"], sp["walk_x2"])
        w1, w2 = walk_for_spawn(data["platforms"], sp["x"], sp["y"])
        sp["walk_x1"], sp["walk_x2"] = w1, w2
        print(f"mob={sp['mob_id']} @({sp['x']},{sp['y']}) {old} -> ({w1},{w2})")
    PLAT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {PLAT}")


if __name__ == "__main__":
    main()
