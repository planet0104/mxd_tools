#!/usr/bin/env python3
"""验证 mini_game 所需的全部资源路径存在且 PNG 可解码。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("需要 Pillow: pip install pillow")
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def ok_png(p: Path) -> str:
    with Image.open(p) as im:
        return f"{im.size[0]}x{im.size[1]}"


def main() -> int:
    errors: list[str] = []

    plat = json.loads((ASSETS / "maps/50001/map_50001_platforms.json").read_text(encoding="utf-8"))
    checks: list[Path] = [
        ASSETS / "maps/50001" / plat["image"],
    ]
    ui = json.loads((ASSETS / "ui/ui_layout.json").read_text(encoding="utf-8"))
    for w in ui["widgets"].values():
        checks.append(ASSETS / "ui" / w["file"])

    player = ASSETS / "player/默认男新手"
    for anim in ["stand1", "walk1", "jump", "alert"]:
        checks.append(player / f"{anim}_0.png")

    for mob in ["100101_蓝蜗牛", "130101_红蜗牛", "1210102_花蘑菇", "130100_树怪"]:
        checks.append(ASSETS / "mobs" / mob / "move_0.png")

    checks.append(ASSETS / "drops/金币/meso_00.png")
    potions = list((ASSETS / "drops/药水").glob("*2000000*.png"))
    if potions:
        checks.append(potions[0])
    else:
        errors.append("drops/药水/*2000000*.png missing")

    portal_dirs = list((ASSETS / "portals").glob("pv_*"))
    if portal_dirs:
        pv = sorted(portal_dirs[0].glob("pv_*.png"))
        if pv:
            checks.append(pv[0])
        else:
            errors.append(f"{portal_dirs[0]} has no pv_*.png")
    else:
        errors.append("portals/pv_* missing")

    print("=== mini_game asset smoke ===")
    for p in checks:
        if not p.is_file():
            errors.append(f"missing: {p.relative_to(ROOT)}")
            print(f"  FAIL {p.name}")
            continue
        try:
            info = ok_png(p)
            print(f"  OK   {p.relative_to(ROOT)} ({info})")
        except Exception as e:
            errors.append(f"corrupt {p}: {e}")
            print(f"  FAIL {p.name}: {e}")

    print(f"\nmap: slopes={len(plat.get('slopes', []))} portals={len(plat.get('portals', []))} ropes={len(plat.get('ropes', []))}")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(" -", e)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
