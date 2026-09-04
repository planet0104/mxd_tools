#!/usr/bin/env python3
"""将 dataset 斜坡 + render.json 传送门合并进 map_50001_platforms.json。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "dataset/nangang_50001/map_50001_render_cn.json"
RENDER = ROOT / "assets/maps/50001/map_50001_render.json"
PLAT = ROOT / "assets/maps/50001/map_50001_platforms.json"

TRANSFORM = {"vr_left": 138, "vr_top": -780, "pad_x": 80, "pad_y": 80}


def wz_to_render(x: float, y: float) -> tuple[float, float]:
    t = TRANSFORM
    return x - t["vr_left"] + t["pad_x"], y - t["vr_top"] + t["pad_y"]


def extract_slopes() -> list[dict]:
    ann = json.loads(ANN.read_text(encoding="utf-8"))
    slopes = []
    for s in ann["shapes"]:
        if s["label"] != "地板" or s["shape_type"] != "polygon":
            continue
        pts = [[float(p[0]), float(p[1])] for p in s["points"]]
        slopes.append({"points": pts, "source": "dataset_labelme"})
    return slopes


def extract_portals() -> list[dict]:
    render = json.loads(RENDER.read_text(encoding="utf-8"))
    portals = []
    names = {1: "west00", 2: "east00", 3: "in00", 4: "in01"}
    for key, v in render["layout"]["portals"].items():
        pt = int(v.get("pt", 0))
        if pt != 2:
            continue
        x, y = wz_to_render(float(v["x"]), float(v["y"]))
        idx = int(key)
        portals.append(
            {
                "id": idx,
                "name": names.get(idx, f"portal_{idx}"),
                "x": round(x, 1),
                "y": round(y, 1),
                "to_map": int(v.get("tm", 0)),
                "kind": "visible",
            }
        )
    return portals


def main() -> None:
    plat = json.loads(PLAT.read_text(encoding="utf-8"))
    slopes = extract_slopes()
    portals = extract_portals()
    plat["slopes"] = slopes
    plat["portals"] = portals
    PLAT.write_text(json.dumps(plat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"updated {PLAT.name}: slopes={len(slopes)} portals={len(portals)}")
    for p in portals:
        print(f"  {p['name']} @ ({p['x']}, {p['y']}) -> map {p['to_map']}")


if __name__ == "__main__":
    main()
