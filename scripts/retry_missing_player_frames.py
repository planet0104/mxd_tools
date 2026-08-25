#!/usr/bin/env python3
"""补齐玩家合成立绘中缺失的帧（上次 IO 超时/5xx 跳过的）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 允许直接运行：把 scripts 目录加入 path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_sprites import (  # noqa: E402
    IO_BASE,
    PLAYER_PRESETS,
    http_get,
    player_all_animations,
)
from PIL import Image  # noqa: E402


def missing_frames(out_dir: Path, anims: dict[str, int]) -> list[tuple[str, int]]:
    miss = []
    for anim, n in anims.items():
        for i in range(n):
            p = out_dir / f"{anim}_{i}.png"
            if not p.is_file():
                miss.append((anim, i))
    return miss


def download_one(skin: int, items: list[int], anim: str, frame: int, dest: Path, retries: int = 4) -> bool:
    item_str = ",".join(str(i) for i in items)
    url = f"{IO_BASE}/Character/{skin}/{item_str}/{anim}/{frame}"
    last_err = None
    for attempt in range(retries):
        try:
            data = http_get(url)
            if not data.startswith(b"\x89PNG"):
                raise ValueError("not png")
            dest.write_bytes(data)
            Image.open(dest).convert("RGBA").save(dest)
            return True
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  FAIL {dest.name}: {last_err}")
    return False


def main() -> int:
    assets = ROOT / "assets" / "player"
    anims = player_all_animations()
    total_miss = 0
    total_ok = 0
    total_fail = 0

    for name, cfg in PLAYER_PRESETS.items():
        out_dir = assets / name
        if not out_dir.is_dir():
            print(f"[skip] {name}: 目录不存在")
            continue
        miss = missing_frames(out_dir, anims)
        if not miss:
            print(f"[ok]   {name}: 无缺失")
            continue
        print(f"[fix] {name}: 缺失 {len(miss)} 帧 -> {', '.join(f'{a}_{i}' for a,i in miss)}")
        total_miss += len(miss)
        skin = cfg["skin"]
        items = list(cfg["items"])
        for anim, i in miss:
            dest = out_dir / f"{anim}_{i}.png"
            if download_one(skin, items, anim, i, dest):
                print(f"  OK   {anim}_{i}.png")
                total_ok += 1
            else:
                total_fail += 1
            time.sleep(0.3)

    print(f"\n完成: 缺失 {total_miss}, 补齐 {total_ok}, 仍失败 {total_fail}")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
