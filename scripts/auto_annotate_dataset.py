#!/usr/bin/env python3
"""根据 dataset 目录说明，用大地图 + assets 精灵自动生成 YOLO 训练数据。

流程：
1. 读取已标注地板/梯子/绳子的完整大图（labelme）
2. 生成多张「完整大图副本」：随机贴入口/出口、花蘑菇、蓝蜗牛、玩家
3. 从每张大图按游戏窗口比例随机裁切 → 最终 YOLO 数据集

用法:
  python scripts/auto_annotate_dataset.py \\
    --dataset dataset/彩虹岛-南港西郊平原

  python scripts/auto_annotate_dataset.py \\
    --dataset dataset/彩虹岛-南港西郊平原 \\
    --full-maps 40 --crops-per-map 12 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

CLASS_NAMES = [
    "地板",
    "梯子",
    "绳子",
    "入口",
    "出口",
    "花蘑菇",
    "蓝蜗牛",
    "玩家",
]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# 可用动画帧前缀（每次贴图从对应目录随机抽一帧；排除死亡/图集）
MOB_KEEP_PREFIX = ("stand", "move", "jump", "skill", "hit")
PLAYER_KEEP_PREFIX = ("stand", "walk", "jump", "alert")
PORTAL_KEEP_PREFIX = ("pv",)


@dataclass
class Box:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    shape_type: str = "rectangle"
    points: list | None = None  # 原始多边形点（可选）
    description: str = ""

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    def width(self) -> float:
        a = self.as_xyxy()
        return a[2] - a[0]

    def height(self) -> float:
        a = self.as_xyxy()
        return a[3] - a[1]

    def area(self) -> float:
        return max(0.0, self.width()) * max(0.0, self.height())

    def iou(self, other: "Box") -> float:
        ax1, ay1, ax2, ay2 = self.as_xyxy()
        bx1, by1, bx2, by2 = other.as_xyxy()
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0

    def contains_box(self, other: "Box", coverage: float = 0.92) -> bool:
        """other 被 self 覆盖的面积比例是否过高。"""
        ax1, ay1, ax2, ay2 = self.as_xyxy()
        bx1, by1, bx2, by2 = other.as_xyxy()
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        oa = other.area()
        return oa > 0 and (inter / oa) >= coverage

    def to_labelme_shape(self) -> dict:
        if self.points and self.shape_type == "polygon":
            pts = self.points
            st = "polygon"
        else:
            x1, y1, x2, y2 = self.as_xyxy()
            pts = [[x1, y1], [x2, y2]]
            st = "rectangle"
        return {
            "label": self.label,
            "points": pts,
            "group_id": None,
            "description": self.description,
            "shape_type": st,
            "flags": {},
            "mask": None,
        }


@dataclass
class Scene:
    image: Image.Image
    base_boxes: list[Box]
    sprite_boxes: list[Box] = field(default_factory=list)

    def all_boxes(self) -> list[Box]:
        return list(self.base_boxes) + list(self.sprite_boxes)


def mxd_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rect_from_points(points: list) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def load_labelme(json_path: Path, image_path: Path) -> tuple[Image.Image, list[Box]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    img = Image.open(image_path).convert("RGBA")
    boxes: list[Box] = []
    for s in data.get("shapes") or []:
        label = s.get("label") or ""
        pts = s.get("points") or []
        if len(pts) < 2:
            continue
        x1, y1, x2, y2 = rect_from_points(pts)
        boxes.append(
            Box(
                label=label,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                shape_type=s.get("shape_type") or "rectangle",
                points=pts if s.get("shape_type") == "polygon" else None,
            )
        )
    return img, boxes


def list_sprite_frames(folder: Path, keep_prefix: tuple[str, ...]) -> list[Path]:
    frames = []
    for p in sorted(folder.glob("*.png")):
        name = p.name.lower()
        if name.startswith("_") or "atlas" in name:
            continue
        if name.startswith("die"):
            continue
        if any(name.startswith(pref) for pref in keep_prefix):
            frames.append(p)
    if not frames:
        frames = [
            p
            for p in sorted(folder.glob("*.png"))
            if not p.name.startswith("_")
            and "atlas" not in p.name.lower()
            and not p.name.lower().startswith("die")
        ]
    return frames


def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def pick_sprite(
    frame_paths: list[Path],
    rng: random.Random,
    *,
    allow_flip: bool = True,
) -> tuple[Image.Image, str]:
    """每次贴图：随机选一帧，并可随机水平翻转（模拟朝向）。"""
    if not frame_paths:
        raise ValueError("frame_paths 为空")
    path = rng.choice(frame_paths)
    spr = load_rgba(path)
    flipped = False
    if allow_flip and rng.random() < 0.5:
        spr = spr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flipped = True
    desc = f"frame={path.name}"
    if flipped:
        desc += " flip=1"
    return spr, desc


def paste_sprite(canvas: Image.Image, sprite: Image.Image, x: int, y: int) -> None:
    canvas.alpha_composite(sprite, (x, y))


def place_sprite_box(
    canvas: Image.Image,
    scene: Scene,
    label: str,
    frame_paths: list[Path],
    floor: Box,
    rng: random.Random,
    *,
    map_w: int,
    map_h: int,
    blocked: list[tuple[float, float, float, float]] | None = None,
    allow_flip: bool = True,
    into_mobs: list[Box] | None = None,
) -> Box | None:
    spr, desc = pick_sprite(frame_paths, rng, allow_flip=allow_flip)
    sw, sh = spr.size
    pos = try_place_on_floor(floor, sw, sh, map_w, map_h, rng, blocked=blocked)
    if not pos:
        return None
    x, y = pos
    paste_sprite(canvas, spr, x, y)
    box = Box(
        label,
        float(x),
        float(y),
        float(x + sw),
        float(y + sh),
        description=desc,
    )
    scene.sprite_boxes.append(box)
    if into_mobs is not None:
        into_mobs.append(box)
    return box


def aabb_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def try_place_on_floor(
    floor: Box,
    sw: int,
    sh: int,
    map_w: int,
    map_h: int,
    rng: random.Random,
    blocked: list[tuple[float, float, float, float]] | None = None,
    max_tries: int = 40,
) -> tuple[int, int] | None:
    """脚底落在地板顶边附近；返回左上角。"""
    fx1, fy1, fx2, fy2 = floor.as_xyxy()
    if fx2 - fx1 < sw * 0.5:
        return None
    foot_y = fy1 + rng.uniform(0, min(12.0, max(1.0, (fy2 - fy1) * 0.15)))
    for _ in range(max_tries):
        cx = rng.uniform(fx1 + sw * 0.3, fx2 - sw * 0.3)
        x = int(round(cx - sw / 2))
        y = int(round(foot_y - sh))
        if x < 0 or y < 0 or x + sw > map_w or y + sh > map_h:
            continue
        box = (float(x), float(y), float(x + sw), float(y + sh))
        if blocked and any(aabb_overlap(box, b) for b in blocked):
            continue
        return x, y
    return None


def is_large_floor(floor: Box, min_w: float = 200.0, min_area: float = 15000.0) -> bool:
    return floor.width() >= min_w or floor.area() >= min_area


def build_full_scene(
    base_img: Image.Image,
    base_boxes: list[Box],
    sprites: dict[str, list[Path]],
    rng: random.Random,
    *,
    portals_n: tuple[int, int] = (3, 5),
    mushrooms_per_large: tuple[int, int] = (1, 2),
    snails_per_large: tuple[int, int] = (3, 4),
    players_n: tuple[int, int] = (5, 6),
) -> Scene:
    canvas = base_img.copy()
    mw, mh = canvas.size
    floors = [b for b in base_boxes if b.label == "地板"]
    large_floors = [f for f in floors if is_large_floor(f)]
    if not large_floors:
        large_floors = floors

    scene = Scene(image=canvas, base_boxes=list(base_boxes), sprite_boxes=[])
    portal_boxes: list[tuple[float, float, float, float]] = []
    mob_boxes: list[Box] = []

    # 1) 出入口：互不重叠；尽量同时有入口和出口；传送门不翻转（光柱不对称）
    n_portals = rng.randint(*portals_n)
    portal_labels: list[str] = []
    if n_portals >= 2:
        portal_labels = ["入口", "出口"] + [
            rng.choice(["入口", "出口"]) for _ in range(n_portals - 2)
        ]
        rng.shuffle(portal_labels)
    else:
        portal_labels = [rng.choice(["入口", "出口"]) for _ in range(n_portals)]
    for label in portal_labels:
        floor = rng.choice(large_floors or floors)
        box = place_sprite_box(
            canvas,
            scene,
            label,
            sprites["portal"],
            floor,
            rng,
            map_w=mw,
            map_h=mh,
            blocked=portal_boxes,
            allow_flip=False,
        )
        if box:
            portal_boxes.append(box.as_xyxy())

    # 2) 花蘑菇 / 蓝蜗牛：每次随机帧 + 随机朝向
    for floor in large_floors:
        for _ in range(rng.randint(*mushrooms_per_large)):
            place_sprite_box(
                canvas,
                scene,
                "花蘑菇",
                sprites["花蘑菇"],
                floor,
                rng,
                map_w=mw,
                map_h=mh,
                into_mobs=mob_boxes,
            )
        for _ in range(rng.randint(*snails_per_large)):
            place_sprite_box(
                canvas,
                scene,
                "蓝蜗牛",
                sprites["蓝蜗牛"],
                floor,
                rng,
                map_w=mw,
                map_h=mh,
                into_mobs=mob_boxes,
            )

    small_floors = [f for f in floors if f not in large_floors]
    for floor in small_floors:
        if rng.random() > 0.45:
            continue
        place_sprite_box(
            canvas,
            scene,
            "蓝蜗牛",
            sprites["蓝蜗牛"],
            floor,
            rng,
            map_w=mw,
            map_h=mh,
            into_mobs=mob_boxes,
        )

    # 3) 玩家：后贴（上层），5~6 个；不整框盖住怪物
    n_players = rng.randint(*players_n)
    placed_players = 0
    attempts = 0
    while placed_players < n_players and attempts < n_players * 30:
        attempts += 1
        floor = rng.choice(floors)
        spr, desc = pick_sprite(sprites["玩家"], rng, allow_flip=True)
        sw, sh = spr.size
        pos = try_place_on_floor(floor, sw, sh, mw, mh, rng, blocked=portal_boxes)
        if not pos:
            continue
        x, y = pos
        cand = Box(
            "玩家",
            float(x),
            float(y),
            float(x + sw),
            float(y + sh),
            description=desc,
        )
        if any(cand.contains_box(m) for m in mob_boxes):
            continue
        paste_sprite(canvas, spr, x, y)
        scene.sprite_boxes.append(cand)
        placed_players += 1

    return scene


def save_labelme(path: Path, image_name: str, w: int, h: int, boxes: list[Box]) -> None:
    data = {
        "version": "5.10.1",
        "flags": {},
        "shapes": [b.to_labelme_shape() for b in boxes],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clip_box_to_window(box: Box, wx: int, wy: int, ww: int, wh: int) -> Box | None:
    x1, y1, x2, y2 = box.as_xyxy()
    nx1 = max(x1, float(wx))
    ny1 = max(y1, float(wy))
    nx2 = min(x2, float(wx + ww))
    ny2 = min(y2, float(wy + wh))
    if nx2 - nx1 < 2 or ny2 - ny1 < 2:
        return None
    # 原框被裁掉太多则丢弃（避免碎标签）
    if box.area() > 0 and ((nx2 - nx1) * (ny2 - ny1) / box.area()) < 0.25:
        return None
    pts = None
    st = "rectangle"
    if box.points and box.shape_type == "polygon":
        clipped = []
        for px, py in box.points:
            if wx <= px <= wx + ww and wy <= py <= wy + wh:
                clipped.append([px - wx, py - wy])
        if len(clipped) >= 3:
            pts = clipped
            st = "polygon"
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return Box(
                box.label,
                min(xs),
                min(ys),
                max(xs),
                max(ys),
                st,
                pts,
                description=box.description,
            )
    return Box(
        box.label,
        nx1 - wx,
        ny1 - wy,
        nx2 - wx,
        ny2 - wy,
        "rectangle",
        None,
        description=box.description,
    )


def box_to_yolo_line(box: Box, img_w: int, img_h: int) -> str | None:
    cid = CLASS_TO_ID.get(box.label)
    if cid is None:
        return None
    x1, y1, x2, y2 = box.as_xyxy()
    bw = x2 - x1
    bh = y2 - y1
    if bw < 1 or bh < 1:
        return None
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def window_sizes(ref_w: int, ref_h: int) -> list[tuple[int, int]]:
    """参考截图尺寸 + 若干缩放，增加多样性。"""
    scales = (0.75, 0.9, 1.0, 1.15, 1.3)
    out = []
    for s in scales:
        w = max(320, int(round(ref_w * s)))
        h = max(240, int(round(ref_h * s)))
        out.append((w, h))
    # 额外两种比例
    out.append((int(ref_w * 0.85), int(ref_h * 1.05)))
    out.append((int(ref_w * 1.1), int(ref_h * 0.85)))
    return out


def random_crop_origin(
    map_w: int, map_h: int, ww: int, wh: int, rng: random.Random
) -> tuple[int, int] | None:
    if ww >= map_w or wh >= map_h:
        return None
    return rng.randint(0, map_w - ww), rng.randint(0, map_h - wh)


def load_all_sprites(assets: Path) -> dict[str, list[Path]]:
    """返回各类精灵的帧文件路径；贴图时再随机抽一帧加载。"""
    portal_dir = assets / "portals" / "pv_可见传送门"
    if not portal_dir.is_dir():
        cands = list((assets / "portals").glob("pv*"))
        portal_dir = cands[0] if cands else portal_dir

    mushroom_dir = assets / "mobs" / "1210102_花蘑菇"
    snail_dir = assets / "mobs" / "100101_蓝蜗牛"

    player_frames: list[Path] = []
    player_root = assets / "player"
    if player_root.is_dir():
        for sub in sorted(player_root.iterdir()):
            if not sub.is_dir():
                continue
            player_frames.extend(list_sprite_frames(sub, PLAYER_KEEP_PREFIX))

    sprites = {
        "portal": list_sprite_frames(portal_dir, PORTAL_KEEP_PREFIX),
        "花蘑菇": list_sprite_frames(mushroom_dir, MOB_KEEP_PREFIX),
        "蓝蜗牛": list_sprite_frames(snail_dir, MOB_KEEP_PREFIX),
        "玩家": player_frames,
    }
    for k, v in sprites.items():
        if not v:
            raise FileNotFoundError(f"精灵帧为空: {k}（检查 assets 目录）")
    return sprites


def find_base_pair(dataset_dir: Path) -> tuple[Path, Path]:
    pngs = sorted(dataset_dir.glob("map_*_render*.png")) + sorted(
        dataset_dir.glob("*render*.png")
    )
    pngs = [p for p in pngs if p.is_file()]
    if not pngs:
        raise FileNotFoundError(f"未找到大地图 PNG: {dataset_dir}")
    img = pngs[0]
    js = img.with_suffix(".json")
    if not js.is_file():
        raise FileNotFoundError(f"未找到对应 labelme: {js}")
    return img, js


def find_ref_shot_size(dataset_dir: Path, tools_root: Path) -> tuple[int, int]:
    # 同名 screen_caps
    name = dataset_dir.name
    caps = tools_root / "screen_caps" / name
    if caps.is_dir():
        shots = sorted(caps.glob("*.png"))
        if shots:
            with Image.open(shots[0]) as im:
                return im.size
    return 1368, 800


def write_data_yaml(path: Path, dataset_name: str) -> None:
    lines = [
        f"# auto-generated for {dataset_name}",
        "path: .",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for i, n in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="大地图自动贴图标注 → YOLO 窗口数据集")
    ap.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="dataset 子目录，如 dataset/彩虹岛-南港西郊平原",
    )
    ap.add_argument("--assets", type=Path, default=None, help="精灵根目录（默认 assets）")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录（默认 <dataset>/generated）",
    )
    ap.add_argument("--full-maps", type=int, default=40, help="完整大图副本数量")
    ap.add_argument("--crops-per-map", type=int, default=12, help="每张大图裁切窗口数")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--players-min", type=int, default=5)
    ap.add_argument("--players-max", type=int, default=6)
    ap.add_argument("--snails-per-large-min", type=int, default=3)
    ap.add_argument("--snails-per-large-max", type=int, default=4)
    ap.add_argument("--mushrooms-per-large-min", type=int, default=1)
    ap.add_argument("--mushrooms-per-large-max", type=int, default=2)
    ap.add_argument("--portals-min", type=int, default=3)
    ap.add_argument("--portals-max", type=int, default=5)
    ap.add_argument(
        "--keep-full-maps",
        action="store_true",
        help="保留中间完整大图 PNG/JSON（默认也保留）",
    )
    args = ap.parse_args()

    tools = mxd_tools_root()
    dataset_dir = args.dataset
    if not dataset_dir.is_absolute():
        dataset_dir = (tools / dataset_dir).resolve()
    assets = args.assets or (tools / "assets")
    if not assets.is_absolute():
        assets = (tools / assets).resolve()
    out_dir = args.out or (dataset_dir / "generated")
    if not out_dir.is_absolute():
        out_dir = (tools / out_dir).resolve()

    rng = random.Random(args.seed)
    img_path, json_path = find_base_pair(dataset_dir)
    base_img, base_boxes = load_labelme(json_path, img_path)
    sprites = load_all_sprites(assets)
    print(
        "帧库: "
        + ", ".join(f"{k}×{len(v)}" for k, v in sprites.items())
    )
    ref_w, ref_h = find_ref_shot_size(dataset_dir, tools)
    sizes = window_sizes(ref_w, ref_h)

    full_dir = out_dir / "full_maps"
    yolo_dir = out_dir / "yolo"
    img_train = yolo_dir / "images" / "train"
    img_val = yolo_dir / "images" / "val"
    lbl_train = yolo_dir / "labels" / "train"
    lbl_val = yolo_dir / "labels" / "val"
    for d in (full_dir, img_train, img_val, lbl_train, lbl_val):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    print(f"大地图: {img_path.name} {base_img.size}")
    print(f"基底标注: {len(base_boxes)} | 参考窗口: {ref_w}x{ref_h}")
    print(
        f"密度: 玩家 {args.players_min}~{args.players_max}, "
        f"大地板蜗牛 {args.snails_per_large_min}~{args.snails_per_large_max}, "
        f"花蘑菇 {args.mushrooms_per_large_min}~{args.mushrooms_per_large_max}"
    )

    crop_records: list[tuple[Image.Image, list[Box], str]] = []
    stats = defaultdict(int)

    for mi in range(args.full_maps):
        scene = build_full_scene(
            base_img,
            base_boxes,
            sprites,
            rng,
            portals_n=(args.portals_min, args.portals_max),
            mushrooms_per_large=(
                args.mushrooms_per_large_min,
                args.mushrooms_per_large_max,
            ),
            snails_per_large=(args.snails_per_large_min, args.snails_per_large_max),
            players_n=(args.players_min, args.players_max),
        )
        for b in scene.sprite_boxes:
            stats[b.label] += 1

        name = f"full_{mi:03d}"
        rgb = scene.image.convert("RGB")
        rgb.save(full_dir / f"{name}.png")
        save_labelme(
            full_dir / f"{name}.json",
            f"{name}.png",
            scene.image.width,
            scene.image.height,
            scene.all_boxes(),
        )

        mw, mh = scene.image.size
        for ci in range(args.crops_per_map):
            ww, wh = rng.choice(sizes)
            ww = min(ww, mw - 1)
            wh = min(wh, mh - 1)
            origin = random_crop_origin(mw, mh, ww, wh, rng)
            if origin is None:
                continue
            wx, wy = origin
            crop = scene.image.crop((wx, wy, wx + ww, wy + wh)).convert("RGB")
            cropped_boxes: list[Box] = []
            for b in scene.all_boxes():
                cb = clip_box_to_window(b, wx, wy, ww, wh)
                if cb is not None and cb.label in CLASS_TO_ID:
                    cropped_boxes.append(cb)
            if not cropped_boxes:
                continue
            crop_name = f"{name}_c{ci:02d}_{ww}x{wh}"
            crop_records.append((crop, cropped_boxes, crop_name))

        print(
            f"  [{mi+1}/{args.full_maps}] sprites="
            + ", ".join(
                f"{k}={sum(1 for b in scene.sprite_boxes if b.label==k)}"
                for k in ("入口", "出口", "花蘑菇", "蓝蜗牛", "玩家")
            )
        )

    rng.shuffle(crop_records)
    n_val = max(1, int(round(len(crop_records) * args.val_ratio))) if crop_records else 0
    val_set = set(range(len(crop_records) - n_val, len(crop_records))) if n_val else set()

    for i, (crop, boxes, crop_name) in enumerate(crop_records):
        is_val = i in val_set
        img_dir = img_val if is_val else img_train
        lbl_dir = lbl_val if is_val else lbl_train
        crop.save(img_dir / f"{crop_name}.png")
        lines = []
        for b in boxes:
            line = box_to_yolo_line(b, crop.width, crop.height)
            if line:
                lines.append(line)
        (lbl_dir / f"{crop_name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        # 同步一份 labelme 便于肉眼检查
        save_labelme(
            img_dir / f"{crop_name}.json",
            f"{crop_name}.png",
            crop.width,
            crop.height,
            boxes,
        )

    write_data_yaml(yolo_dir / "data.yaml", dataset_dir.name)
    classes_txt = yolo_dir / "classes.txt"
    classes_txt.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    summary = (
        f"完成\n"
        f"  完整大图: {args.full_maps} → {full_dir}\n"
        f"  窗口样本: {len(crop_records)} "
        f"(train={len(crop_records)-n_val}, val={n_val}) → {yolo_dir}\n"
        f"  大图精灵累计: {dict(stats)}\n"
        f"  data.yaml: {yolo_dir / 'data.yaml'}\n"
    )
    (out_dir / "README.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
