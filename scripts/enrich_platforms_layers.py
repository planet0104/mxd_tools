#!/usr/bin/env python3
"""从缓存的 map API / 在线数据给 platforms.json 写入 layerId/groupId，并做统计。"""

from __future__ import annotations

import json
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAT = ROOT / "assets/maps/50001/map_50001_platforms.json"
CACHE = Path(
    r"C:\Users\glsa-\.cursor\projects\c-Users-glsa-Desktop-mxd-classic"
    r"\agent-tools\f54a2451-eb66-4e2e-a6c9-705bfe3fdce1.txt"
)


def load_footholds() -> dict[int, dict]:
    raw = None
    if CACHE.is_file():
        raw = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        req = urllib.request.Request(
            "https://maplestory.io/api/GMS/83/map/50001",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = json.loads(r.read().decode())
    out: dict[int, dict] = {}
    for v in (raw.get("footholds") or {}).values():
        out[int(v["id"])] = v
    return out


def to_px(x: float, y: float, t: dict) -> tuple[float, float]:
    return (
        round(x - t["vr_left"] + t["pad_x"]),
        round(y - t["vr_top"] + t["pad_y"]),
    )


def main() -> None:
    fhs = load_footholds()
    plat = json.loads(PLAT.read_text(encoding="utf-8"))
    t = plat["transform"]
    c = Counter((v["layerId"], v["groupId"]) for v in fhs.values())
    print("layer/group combos", len(c))
    for k, n in sorted(c.items()):
        print(f"  L{k[0]} G{k[1]}: {n}")

    matched = 0
    for p in plat["platforms"]:
        fid = int(p.get("id", -1))
        fh = fhs.get(fid)
        if not fh:
            # 坐标兜底匹配
            for cand in fhs.values():
                px1, py1 = to_px(cand["x1"], cand["y1"], t)
                px2, py2 = to_px(cand["x2"], cand["y2"], t)
                if (
                    abs(px1 - p["x1"]) <= 1
                    and abs(py1 - p["y1"]) <= 1
                    and abs(px2 - p["x2"]) <= 1
                    and abs(py2 - p["y2"]) <= 1
                ):
                    fh = cand
                    break
        if fh:
            p["layer"] = int(fh["layerId"])
            p["group"] = int(fh["groupId"])
            p["prev"] = int(fh.get("prev") or 0)
            p["next"] = int(fh.get("next") or 0)
            matched += 1
        else:
            p.setdefault("layer", 0)
            p.setdefault("group", 0)

    print(f"matched {matched}/{len(plat['platforms'])}")
    verts = [
        p
        for p in plat["platforms"]
        if abs(p["x2"] - p["x1"]) < 8 and abs(p["y2"] - p["y1"]) >= 8
    ]
    print("vertical walls with layer/group:")
    for v in verts[:20]:
        print(
            f"  id={v.get('id')} L{v.get('layer')}G{v.get('group')} "
            f"x={v['x1']} y={v['y1']}..{v['y2']}"
        )
    PLAT.write_text(json.dumps(plat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", PLAT)


if __name__ == "__main__":
    main()
