#!/usr/bin/env python3
"""用 platforms.json（WZ/网站碰撞）重写 LabelMe 的地板/绳子/梯子标注。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAT = ROOT / "assets/maps/50001/map_50001_platforms.json"
ANN = ROOT / "dataset/nangang_50001/map_50001_render_cn.json"
BACKUP = ANN.with_suffix(".json.bak_before_platforms_sync")

# LabelMe 矩形高度（顶边=站立线，向下延伸便于在标注工具里看见）
FLOOR_BODY_H = 56.0
ROPE_MIN_W = 16.0
LADDER_WIDTH = 50.0  # LabelMe 梯子框固定宽度
VERTICAL_TRIM_BOTTOM = 20.0  # 绳子/梯子底部裁掉空白，减少 YOLO 干扰


def shape_rect(label: str, x1: float, y1: float, x2: float, y2: float, desc: str = "") -> dict:
    return {
        "label": label,
        "points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
        "group_id": None,
        "description": desc,
        "shape_type": "rectangle",
        "flags": {},
        "mask": None,
    }


def shape_poly(label: str, points: list[list[float]], desc: str = "") -> dict:
    return {
        "label": label,
        "points": [[round(p[0], 2), round(p[1], 2)] for p in points],
        "group_id": None,
        "description": desc,
        "shape_type": "polygon",
        "flags": {},
        "mask": None,
    }


def merge_horizontal(segs: list[dict]) -> list[tuple[float, float, float]]:
    """合并同高度、近邻的水平平台 → (xlo, xhi, y)。"""
    by_y: dict[float, list[tuple[float, float]]] = {}
    for s in segs:
        y = round((s["y1"] + s["y2"]) * 0.5, 1)
        xlo, xhi = sorted([s["x1"], s["x2"]])
        by_y.setdefault(y, []).append((xlo, xhi))

    out: list[tuple[float, float, float]] = []
    for y, spans in sorted(by_y.items()):
        spans = sorted(spans)
        cur_lo, cur_hi = spans[0]
        for lo, hi in spans[1:]:
            if lo <= cur_hi + 8.0:
                cur_hi = max(cur_hi, hi)
            else:
                out.append((cur_lo, cur_hi, y))
                cur_lo, cur_hi = lo, hi
        out.append((cur_lo, cur_hi, y))
    return out


def main() -> None:
    plat = json.loads(PLAT.read_text(encoding="utf-8"))
    ann = json.loads(ANN.read_text(encoding="utf-8"))

    if not BACKUP.exists():
        shutil.copy2(ANN, BACKUP)
        print(f"backup -> {BACKUP.name}")

    horiz: list[dict] = []
    slant: list[dict] = []
    for s in plat.get("platforms", []):
        dx = abs(s["x2"] - s["x1"])
        dy = abs(s["y2"] - s["y1"])
        if dx < 8.0:
            continue  # 竖直墙，不作地板
        if dy < 2.0:
            horiz.append(s)
        else:
            slant.append(s)

    shapes: list[dict] = []

    for xlo, xhi, y in merge_horizontal(horiz):
        shapes.append(
            shape_rect("地板", xlo, y, xhi, y + FLOOR_BODY_H, desc="from platforms.json horiz")
        )

    for s in slant:
        x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
        # 顶边=站立斜线，底边平行下移
        pts = [
            [x1, y1],
            [x2, y2],
            [x2, y2 + FLOOR_BODY_H],
            [x1, y1 + FLOOR_BODY_H],
        ]
        shapes.append(shape_poly("地板", pts, desc=f"from platforms.json slant id={s.get('id','')}"))

    # slopes[] 里若混有旧 LabelMe 多边形，且与 platforms 斜段重复，则跳过
    for s in plat.get("slopes", []):
        if s.get("source") == "dataset_labelme":
            continue
        pts = s.get("points") or []
        if len(pts) >= 3:
            shapes.append(shape_poly("地板", pts, desc="from platforms.json slopes[]"))

    for r in plat.get("ropes", []):
        x = float(r["x"])
        y1, y2 = float(r["y1"]), float(r["y2"])
        kind = r.get("kind", "rope")
        label = "梯子" if kind == "ladder" else "绳子"
        if kind == "ladder":
            w = LADDER_WIDTH
        else:
            w = max(ROPE_MIN_W, float(r.get("width", 16)))
        top = min(y1, y2)
        bottom = max(y1, y2) - VERTICAL_TRIM_BOTTOM
        if bottom < top + 4.0:
            bottom = top + 4.0
        shapes.append(
            shape_rect(
                label,
                x - w * 0.5,
                top,
                x + w * 0.5,
                bottom,
                desc=f"from platforms.json {kind}",
            )
        )

    # 保留非碰撞类标注（若有）
    keep = [
        s
        for s in ann.get("shapes", [])
        if s.get("label") not in ("地板", "绳子", "梯子")
    ]
    ann["shapes"] = shapes + keep

    # ensure_ascii=True：纯 ASCII（中文用 \\u 转义），无 BOM。
    # 避免中文 Windows 上 LabelMe 用 gbk 打开失败，也避免 utf-8-sig BOM 被 json.loads 拒绝。
    ANN.write_text(json.dumps(ann, ensure_ascii=True, indent=2) + "\n", encoding="ascii")

    n_floor = sum(1 for s in shapes if s["label"] == "地板")
    n_rope = sum(1 for s in shapes if s["label"] == "绳子")
    n_ladder = sum(1 for s in shapes if s["label"] == "梯子")
    print(f"wrote {ANN}")
    print(f"地板={n_floor} (水平合并自 {len(horiz)} 段 + 斜坡 {len(slant)} + slopes {len(plat.get('slopes',[]))})")
    print(f"绳子={n_rope} 梯子={n_ladder} 保留其他={len(keep)}")


if __name__ == "__main__":
    main()
