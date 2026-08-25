#!/usr/bin/env python3
"""检查小游戏精灵资源是否满足 mini_game 加载要求。"""

from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

ROOT = Path(__file__).resolve().parents[1] / "assets"


def check_png(p: Path) -> tuple[bool, str]:
    if not p.is_file():
        return False, "missing"
    if HAS_PIL:
        try:
            with Image.open(p) as im:
                return True, f"{im.size[0]}x{im.size[1]} {im.mode}"
        except Exception as e:
            return False, str(e)
    return True, f"{p.stat().st_size} bytes"


def main() -> int:
    issues: list[str] = []
    warnings: list[str] = []

    # --- Player ---
    player = ROOT / "player" / "默认男新手"
    req_player = {"stand1": 4, "walk1": 4, "jump": 1, "alert": 3, "swingO1": 3}
    print("=== PLAYER (默认男新手) ===")
    if not player.is_dir():
        issues.append(f"player dir missing: {player}")
        print(f"  FAIL dir missing: {player}")
    else:
        for anim, n in req_player.items():
            frames = sorted(player.glob(f"{anim}_*.png"))
            if len(frames) < n:
                issues.append(f"player {anim}: need>={n}, got {len(frames)}")
                print(f"  FAIL {anim}: {len(frames)}/{n}")
            else:
                ok, info = check_png(frames[0])
                if not ok:
                    issues.append(f"player {anim} corrupt: {info}")
                print(f"  OK   {anim}: {len(frames)} frames, sample {info}")
        extra = sorted({f.name.split("_")[0] for f in player.glob("*.png")})
        print(f"  available anims: {', '.join(extra)}")

    # --- Mobs ---
    mob_map = {
        100101: "100101_蓝蜗牛",
        130101: "130101_红蜗牛",
        1210102: "1210102_花蘑菇",
        130100: "130100_树怪",
    }
    print("\n=== MOBS (地图 50001 使用) ===")
    for mid, dname in mob_map.items():
        d = ROOT / "mobs" / dname
        if not d.is_dir():
            issues.append(f"mob dir missing: {dname}")
            print(f"  FAIL {mid} {dname}: dir missing")
            continue
        for anim in ["move", "hit1", "die1"]:
            frames = sorted(d.glob(f"{anim}_*.png"))
            if not frames:
                warnings.append(f"mob {dname} {anim}: no frames (game draws empty)")
                print(f"  WARN {dname} {anim}: 0 frames")
            else:
                ok, info = check_png(frames[0])
                if not ok:
                    issues.append(f"mob {dname} {anim} corrupt: {info}")
                print(f"  OK   {dname} {anim}: {len(frames)} frames, sample {info}")

    print("\n=== MOBS (额外目录) ===")
    for d in sorted((ROOT / "mobs").iterdir()):
        if not d.is_dir():
            continue
        pngs = list(d.glob("*.png"))
        used = d.name in mob_map.values()
        note = "used" if used else "unused"
        print(f"  {d.name}: {len(pngs)} png [{note}]")

    dup_130100 = [d.name for d in (ROOT / "mobs").iterdir() if d.is_dir() and d.name.startswith("130100_")]
    if len(dup_130100) > 1:
        warnings.append(f"130100 有两个目录 {dup_130100}，游戏只用 130100_树怪")

    # --- Drops ---
    print("\n=== DROPS ===")
    meso = ROOT / "drops" / "金币" / "meso_00.png"
    ok, info = check_png(meso)
    if ok:
        print(f"  OK   meso_00.png: {info}")
    else:
        issues.append(f"meso_00: {info}")
        print(f"  FAIL meso_00.png: {info}")
    meso_all = sorted((ROOT / "drops" / "金币").glob("meso_*.png"))
    print(f"  meso 动画帧共 {len(meso_all)} 张（游戏当前只用 meso_00）")

    potion_dir = ROOT / "drops" / "药水"
    potions = sorted(potion_dir.glob("*2000000*.png")) if potion_dir.is_dir() else []
    if not potions:
        issues.append("potion 2000000 not found under drops/药水")
        print("  FAIL potion 2000000: not found")
    else:
        ok, info = check_png(potions[0])
        print(f"  OK   {potions[0].name}: {info}")

    for sub in ["其他", "武器", "装备"]:
        p = ROOT / "drops" / sub
        if p.is_dir():
            n = len(list(p.glob("*.png")))
            print(f"  extra {sub}: {n} png (背包 P2+ 可用)")

    # --- Portals ---
    print("\n=== PORTALS ===")
    portal_pngs = list((ROOT / "portals").rglob("*.png"))
    print(f"  total: {len(portal_pngs)} png")
    by_dir: dict[str, int] = {}
    for p in portal_pngs:
        by_dir[p.parent.name] = by_dir.get(p.parent.name, 0) + 1
    for name, cnt in sorted(by_dir.items()):
        sample = next(p for p in portal_pngs if p.parent.name == name)
        ok, info = check_png(sample)
        print(f"  OK   {name}: {cnt} frames, sample {info}")
    if not portal_pngs:
        warnings.append("portals 目录无 png")
    else:
        ok.append("portals sprites present (rendered in mini_game)")

    # --- Summary ---
    print("\n=== SUMMARY ===")
    if issues:
        print(f"BLOCKERS ({len(issues)}):")
        for i in issues:
            print(f"  [X] {i}")
    else:
        print("BLOCKERS: 无 — drops/player/mobs 满足 mini_game 当前加载需求")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [!] {w}")

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
