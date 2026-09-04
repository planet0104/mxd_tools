#!/usr/bin/env python3
"""根据 dataset 目录说明，用大地图 + assets 精灵自动生成 YOLO 训练数据。

流程：
1. 读取已标注地板/梯子/绳子的完整大图（labelme）
2. 生成多张「完整大图副本」：随机贴入口/出口、怪物、掉落物、玩家
3. 大地板按比例混合：正常 / 稀疏 / 仅掉落 / 空台
4. 从每张大图按游戏窗口比例随机裁切
5. 按比例在部分窗口上贴 UI → 最终 YOLO 数据集

用法:
  python scripts/auto_annotate_dataset.py \\
    --dataset dataset/彩虹岛-南港西郊平原

  python scripts/auto_annotate_dataset.py \\
    --dataset dataset/彩虹岛-南港西郊平原 \\
    --full-maps 80 --crops-per-map 15 --ui-ratio 0.65 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from hp_bar_draw import HP_CLASS_NAMES, apply_hp_to_panel, sample_hp_ratio

CLASS_NAMES = [
    "地板",
    "梯子",
    "绳子",
    "入口",
    "出口",
    "花蘑菇",
    "蓝蜗牛",
    "绿蜗牛",
    "红蜗牛",
    "树怪",
    "玩家",
    "金币",
    "药水",
    "武器",
    "装备",
    "材料",
    "小地图",
    "任务窗",
    "浮动按钮",
    "面板",
    "键盘",
    *HP_CLASS_NAMES,
]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

MOB_LABELS = ("花蘑菇", "蓝蜗牛", "绿蜗牛", "红蜗牛", "树怪")

# assets/mobs 下目录名 → 标注类（按地图 50001 + 常见蜗牛）
MOB_ASSET_DIRS: tuple[tuple[str, str], ...] = (
    ("1210102_花蘑菇", "花蘑菇"),
    ("100101_蓝蜗牛", "蓝蜗牛"),
    ("100100_绿蜗牛", "绿蜗牛"),
    ("130101_红蜗牛", "红蜗牛"),
    ("130100_树怪", "树怪"),
    ("130100_木妖", "树怪"),
)
DROP_LABELS = ("金币", "药水", "武器", "装备", "材料")

# 可用动画帧前缀（每次贴图从对应目录随机抽一帧；排除死亡/图集）
MOB_KEEP_PREFIX = ("stand", "move", "jump", "skill", "hit")
PLAYER_KEEP_PREFIX = ("stand", "walk", "jump", "alert")
PLAYER_CLIMB_PREFIX = ("ladder", "rope")
# 与 extract_sprites.PLAYER_COMBAT_ANIMS 一致：swingO1/O2/O3、swingT1、stabO1/O2、shoot1
PLAYER_COMBAT_PREFIX = ("swing", "stab", "shoot")
# 优先抽持武器更明显的帧（小写匹配文件名）；O1/O2 通常比部分中间帧更清晰
COMBAT_PREFER_PREFIX = (
    "swingo1",
    "swingo2",
    "stabo1",
    "shoot1",
    "swingo3",
    "stabo2",
    "swingt1",
)
PORTAL_KEEP_PREFIX = ("pv",)

# 精灵互遮：任一框被另一框盖住超过该比例则拒绝摆放（最多 30% 遮挡）
MAX_SPRITE_OCCLUSION = 0.30

# 大地板模式权重：normal / sparse / drops_only（仅掉落）/ empty（几乎空）
FLOOR_MODE_WEIGHTS = (
    ("normal", 0.45),
    ("sparse", 0.20),
    ("drops_only", 0.25),
    ("empty", 0.10),
)

# 掉落物标签抽样权重
DROP_LABEL_WEIGHTS = (
    ("金币", 5),
    ("药水", 4),
    ("材料", 3),
    ("武器", 3),
    ("装备", 3),
)

# 窗口内掉落框过小则跳过（像素）；小掉落难学且易与 UI/地面纹理混淆
MIN_DROP_BOX_PX = 14
# 掉落物贴图统一缩放到最长边约 30px，减少帧间尺度差
DROP_TARGET_PX = 30

# assets/drops 子目录 → 标注类名
DROP_DIR_TO_LABEL = {
    "金币": "金币",
    "药水": "药水",
    "武器": "武器",
    "装备": "装备",
    "其他": "材料",
}

# 窗口 UI 资源文件名 → 标注类名
UI_ASSET_FILES = {
    "小地图": "小地图_UI.png",
    "任务窗": "任务窗_UI.png",
    "浮动按钮": "浮动按钮_UI.png",
    "面板": "面板_UI.png",
    "键盘": "键盘_UI.png",
}


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

    def intersection_area(self, other: "Box") -> float:
        ax1, ay1, ax2, ay2 = self.as_xyxy()
        bx1, by1, bx2, by2 = other.as_xyxy()
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    def iou(self, other: "Box") -> float:
        inter = self.intersection_area(other)
        if inter <= 0:
            return 0.0
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0

    def coverage_of(self, other: "Box") -> float:
        """self 覆盖 other 的面积比例（0~1）。"""
        oa = other.area()
        if oa <= 0:
            return 0.0
        return self.intersection_area(other) / oa

    def contains_box(self, other: "Box", coverage: float = 0.92) -> bool:
        """other 被 self 覆盖的面积比例是否过高。"""
        return self.coverage_of(other) >= coverage

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


def sprite_paths_with_prefix(frame_paths: list[Path], prefix: str) -> list[Path]:
    p = prefix.lower()
    return [fp for fp in frame_paths if fp.name.lower().startswith(p)]


def paste_sprite(canvas: Image.Image, sprite: Image.Image, x: int, y: int) -> None:
    canvas.alpha_composite(sprite, (x, y))


def normalize_drop_sprite(spr: Image.Image, target: int = DROP_TARGET_PX) -> Image.Image:
    """掉落物统一缩放到最长边约 target px，减少 meso 帧间尺度差。"""
    w, h = spr.size
    if w <= 0 or h <= 0:
        return spr
    scale = target / max(w, h)
    if abs(scale - 1.0) < 0.05:
        return spr
    nw = max(MIN_DROP_BOX_PX, int(round(w * scale)))
    nh = max(MIN_DROP_BOX_PX, int(round(h * scale)))
    return spr.resize((nw, nh), Image.Resampling.LANCZOS)


def clean_ui_edge_bg(img: Image.Image, tol: int = 26) -> Image.Image:
    """从四边洪水填充相近色，去掉截图残留背景（UI 本体尽量保留）。"""
    arr = img.convert("RGBA").load()
    w, h = img.size
    out = img.convert("RGBA").copy()
    px = out.load()
    visited = [[False] * w for _ in range(h)]
    seeds: list[tuple[int, int]] = []
    for x in range(w):
        seeds.append((x, 0))
        seeds.append((x, h - 1))
    for y in range(h):
        seeds.append((0, y))
        seeds.append((w - 1, y))

    def similar(c0, c1) -> bool:
        return (
            abs(c0[0] - c1[0]) <= tol
            and abs(c0[1] - c1[1]) <= tol
            and abs(c0[2] - c1[2]) <= tol
        )

    for sx, sy in seeds:
        if visited[sy][sx]:
            continue
        seed = arr[sx, sy]
        q: deque[tuple[int, int]] = deque([(sx, sy)])
        visited[sy][sx] = True
        while q:
            x, y = q.popleft()
            px[x, y] = (seed[0], seed[1], seed[2], 0)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx]:
                    if similar(arr[nx, ny], seed):
                        visited[ny][nx] = True
                        q.append((nx, ny))
    return out


def load_ui_sprites(assets: Path) -> dict[str, Image.Image]:
    ui_dir = assets / "ui"
    out: dict[str, Image.Image] = {}
    for label, fname in UI_ASSET_FILES.items():
        path = ui_dir / fname
        if not path.is_file():
            raise FileNotFoundError(f"缺少 UI 资源: {path}")
        out[label] = clean_ui_edge_bg(load_rgba(path))
    return out


def scale_ui_set(
    ui: dict[str, Image.Image], window_w: int, window_h: int
) -> dict[str, Image.Image]:
    """面板过宽或总高度不够时，整套 UI 等比缩小。"""
    panel = ui["面板"]
    keyboard = ui["键盘"]
    minimap = ui["小地图"]
    quest = ui["任务窗"]
    scale = 1.0
    # 面板左右各留 8px
    if panel.width + 16 > window_w:
        scale = min(scale, (window_w - 16) / panel.width)
    # 底部：面板 + 键盘 + 顶边小地图/任务窗余量
    need_h = panel.height + keyboard.height + 8
    if need_h > window_h * 0.55:
        scale = min(scale, (window_h * 0.55) / need_h)
    # 顶部：小地图 / 任务窗不挤爆
    top_need = max(minimap.height, quest.height) + 16
    if top_need > window_h * 0.45:
        scale = min(scale, (window_h * 0.45) / top_need)
    if scale >= 0.999:
        return ui
    scaled: dict[str, Image.Image] = {}
    for k, im in ui.items():
        nw = max(1, int(round(im.width * scale)))
        nh = max(1, int(round(im.height * scale)))
        scaled[k] = im.resize((nw, nh), Image.Resampling.LANCZOS)
    return scaled


def paste_window_ui(
    crop: Image.Image,
    ui_raw: dict[str, Image.Image],
    rng: random.Random,
) -> tuple[Image.Image, list[Box]]:
    """按固定布局把 UI 贴到窗口裁切图上（不贴在大地图上）。"""
    ww, wh = crop.size
    ui = scale_ui_set(ui_raw, ww, wh)
    canvas = crop.convert("RGBA")
    boxes: list[Box] = []
    margin = 6

    # 小地图：左上
    mm = ui["小地图"]
    mx, my = margin, margin
    paste_sprite(canvas, mm, mx, my)
    boxes.append(
        Box("小地图", float(mx), float(my), float(mx + mm.width), float(my + mm.height))
    )

    # 任务窗：右上附近（轻微抖动）
    q = ui["任务窗"]
    qx = ww - q.width - margin - rng.randint(0, 12)
    qy = margin + rng.randint(0, 10)
    qx = max(margin, min(qx, ww - q.width - 1))
    qy = max(0, min(qy, wh - q.height - 1))
    paste_sprite(canvas, q, qx, qy)
    boxes.append(
        Box("任务窗", float(qx), float(qy), float(qx + q.width), float(qy + q.height))
    )

    # 浮动按钮：左侧竖直居中
    fb = ui["浮动按钮"]
    fx = margin
    fy = max(margin, (wh - fb.height) // 2)
    paste_sprite(canvas, fb, fx, fy)
    boxes.append(
        Box(
            "浮动按钮",
            float(fx),
            float(fy),
            float(fx + fb.width),
            float(fy + fb.height),
        )
    )

    # 面板：底部居中；重绘 0%～100% 血条并自动标注血条10%～100%
    panel = ui["面板"]
    px = max(0, (ww - panel.width) // 2)
    py = wh - panel.height
    paste_sprite(canvas, panel, px, py)
    boxes.append(
        Box(
            "面板",
            float(px),
            float(py),
            float(px + panel.width),
            float(py + panel.height),
        )
    )
    # 面板相对原始 815×72 的缩放（scale_ui_set 可能缩小）
    panel_scale = panel.width / 815.0
    hp_ratio = sample_hp_ratio(rng)
    (hx1, hy1, hx2, hy2), hp_label, _ = apply_hp_to_panel(
        canvas,
        hp_ratio,
        max_hp=rng.choice((50, 82, 100, 120, 150)),
        scale=panel_scale,
        origin_xy=(px, py),
    )
    if hx2 > hx1 and hy2 > hy1:
        boxes.append(Box(hp_label, float(hx1), float(hy1), float(hx2), float(hy2)))

    # 键盘：面板上方靠右紧贴，不遮挡面板
    kb = ui["键盘"]
    kx = px + panel.width - kb.width
    ky = py - kb.height
    kx = max(0, min(kx, ww - kb.width))
    ky = max(0, ky)
    paste_sprite(canvas, kb, kx, ky)
    boxes.append(
        Box("键盘", float(kx), float(ky), float(kx + kb.width), float(ky + kb.height))
    )

    return canvas.convert("RGB"), boxes


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
    occlude: list[Box] | None = None,
    max_occlusion: float = MAX_SPRITE_OCCLUSION,
    allow_flip: bool = True,
    into_mobs: list[Box] | None = None,
    preloaded: tuple[Image.Image, str] | None = None,
) -> Box | None:
    if preloaded:
        spr, desc = preloaded
    else:
        spr, desc = pick_sprite(frame_paths, rng, allow_flip=allow_flip)
    sw, sh = spr.size
    pos = try_place_on_floor(
        floor,
        sw,
        sh,
        map_w,
        map_h,
        rng,
        blocked=blocked,
        occlude=occlude,
        max_occlusion=max_occlusion,
    )
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


def occlusion_too_high(
    cand: Box,
    others: list[Box],
    max_occlusion: float = MAX_SPRITE_OCCLUSION,
) -> bool:
    """任一方向遮挡超过阈值则 True（新盖旧 或 旧盖新）。"""
    for o in others:
        if cand.coverage_of(o) > max_occlusion or o.coverage_of(cand) > max_occlusion:
            return True
    return False


def try_place_on_floor(
    floor: Box,
    sw: int,
    sh: int,
    map_w: int,
    map_h: int,
    rng: random.Random,
    blocked: list[tuple[float, float, float, float]] | None = None,
    occlude: list[Box] | None = None,
    max_occlusion: float = MAX_SPRITE_OCCLUSION,
    max_tries: int = 120,
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
        if occlude:
            cand = Box("_", float(x), float(y), float(x + sw), float(y + sh))
            if occlusion_too_high(cand, occlude, max_occlusion):
                continue
        return x, y
    return None


def try_place_airborne(
    floor: Box,
    sw: int,
    sh: int,
    map_w: int,
    map_h: int,
    rng: random.Random,
    blocked: list[tuple[float, float, float, float]] | None = None,
    occlude: list[Box] | None = None,
    max_occlusion: float = MAX_SPRITE_OCCLUSION,
    lift_ratio: tuple[float, float] = (0.18, 0.55),
    max_tries: int = 120,
) -> tuple[int, int] | None:
    """脚底高于地板顶边，模拟跳跃/空中（相对站立位置向上抬升）。"""
    pos = try_place_on_floor(
        floor,
        sw,
        sh,
        map_w,
        map_h,
        rng,
        blocked=blocked,
        occlude=occlude,
        max_occlusion=max_occlusion,
        max_tries=max_tries,
    )
    if not pos:
        return None
    x, y = pos
    lift = int(round(sh * rng.uniform(*lift_ratio)))
    y = max(0, y - lift)
    if y < 0 or x + sw > map_w or y + sh > map_h:
        return None
    box = (float(x), float(y), float(x + sw), float(y + sh))
    if blocked and any(aabb_overlap(box, b) for b in blocked):
        return None
    if occlude:
        cand = Box("_", float(x), float(y), float(x + sw), float(y + sh))
        if occlusion_too_high(cand, occlude, max_occlusion):
            return None
    return x, y


def try_place_on_vertical(
    vert: Box,
    sw: int,
    sh: int,
    map_w: int,
    map_h: int,
    rng: random.Random,
    occlude: list[Box] | None = None,
    max_occlusion: float = MAX_SPRITE_OCCLUSION,
    max_tries: int = 80,
) -> tuple[int, int] | None:
    """把玩家贴到绳子/梯子上，水平居中并保证与竖向结构有重叠。"""
    vx1, vy1, vx2, vy2 = vert.as_xyxy()
    vh = vy2 - vy1
    vw = vx2 - vx1
    if vh < sh * 0.3:
        return None
    for _ in range(max_tries):
        cx = (vx1 + vx2) / 2.0 + rng.uniform(-max(2.0, vw * 0.35), max(2.0, vw * 0.35))
        x = int(round(cx - sw / 2.0))
        y_lo = int(vy1 - sh * 0.12)
        y_hi = int(vy2 - sh * 0.35)
        if y_hi < y_lo:
            y_lo = int(vy1)
            y_hi = int(max(vy1, vy2 - sh * 0.4))
        y = rng.randint(y_lo, max(y_lo, y_hi))
        if x < 0 or y < 0 or x + sw > map_w or y + sh > map_h:
            continue
        ox1 = max(float(x), vx1)
        oy1 = max(float(y), vy1)
        ox2 = min(float(x + sw), vx2)
        oy2 = min(float(y + sh), vy2)
        inter = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
        if inter < sw * sh * 0.06:
            continue
        if occlude:
            cand = Box("_", float(x), float(y), float(x + sw), float(y + sh))
            if occlusion_too_high(cand, occlude, max_occlusion):
                continue
        return x, y
    return None


def is_large_floor(floor: Box, min_w: float = 200.0, min_area: float = 15000.0) -> bool:
    return floor.width() >= min_w or floor.area() >= min_area


def pick_weighted(rng: random.Random, pairs: tuple[tuple[str, float | int], ...]) -> str:
    total = sum(float(w) for _, w in pairs)
    r = rng.random() * total
    acc = 0.0
    for name, w in pairs:
        acc += float(w)
        if r <= acc:
            return name
    return pairs[-1][0]


def scale_count_range(
    rng: random.Random, lo: int, hi: int, mode: str
) -> int:
    """按地板模式缩放数量。"""
    if mode in ("drops_only", "empty"):
        return 0
    n = rng.randint(lo, hi)
    if mode == "sparse":
        return max(0, (n + 1) // 2)
    return n


def drops_count_for_mode(rng: random.Random, mode: str, dense: tuple[int, int]) -> int:
    lo, hi = dense
    if mode == "empty":
        return rng.randint(0, 1)
    if mode == "sparse":
        return rng.randint(max(1, lo // 2), max(2, (hi + 1) // 2 + 1))
    # normal / drops_only：与怪密度同量级
    return rng.randint(lo, hi)


def place_drops_on_floor(
    canvas: Image.Image,
    scene: Scene,
    sprites: dict[str, list[Path]],
    floor: Box,
    rng: random.Random,
    *,
    map_w: int,
    map_h: int,
    n: int,
    mode: str = "normal",
) -> None:
    labels = [k for k, _ in DROP_LABEL_WEIGHTS if sprites.get(k)]
    if not labels or n <= 0:
        return
    weighted = tuple((k, w) for k, w in DROP_LABEL_WEIGHTS if k in labels)
    # drops_only 地板：保底 1 武器 + 1 装备
    forced: list[str] = []
    if mode == "drops_only":
        for req in ("武器", "装备"):
            if sprites.get(req):
                forced.append(req)
    labels_to_place: list[str] = list(forced)
    for _ in range(max(0, n - len(forced))):
        labels_to_place.append(pick_weighted(rng, weighted))
    for label in labels_to_place:
        spr, desc = pick_sprite(
            sprites[label],
            rng,
            allow_flip=(label != "金币"),
        )
        spr = normalize_drop_sprite(spr)
        place_sprite_box(
            canvas,
            scene,
            label,
            sprites[label],
            floor,
            rng,
            map_w=map_w,
            map_h=map_h,
            occlude=scene.sprite_boxes,
            allow_flip=(label != "金币"),
            preloaded=(spr, desc),
        )


def build_full_scene(
    base_img: Image.Image,
    base_boxes: list[Box],
    sprites: dict[str, list[Path]],
    rng: random.Random,
    *,
    portals_n: tuple[int, int] = (1, 2),
    mushrooms_per_large: tuple[int, int] = (2, 3),
    blue_snails_per_large: tuple[int, int] = (2, 4),
    green_snails_per_large: tuple[int, int] = (2, 4),
    red_snails_per_large: tuple[int, int] = (1, 3),
    stumps_per_large: tuple[int, int] = (2, 4),
    drops_per_large: tuple[int, int] = (4, 8),
    players_n: tuple[int, int] = (6, 8),
    climbers_n: tuple[int, int] = (2, 4),
    airborne_n: tuple[int, int] = (2, 4),
    combat_n: tuple[int, int] = (2, 4),
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
            occlude=scene.sprite_boxes,
            allow_flip=False,
        )
        if box:
            portal_boxes.append(box.as_xyxy())

    # 2) 怪 + 掉落：按大地板模式（正常 / 稀疏 / 仅掉落 / 空台）
    for floor in large_floors:
        mode = pick_weighted(rng, FLOOR_MODE_WEIGHTS)
        n_mush = scale_count_range(rng, *mushrooms_per_large, mode)
        n_stump = scale_count_range(rng, *stumps_per_large, mode)
        n_blue = scale_count_range(rng, *blue_snails_per_large, mode)
        n_green = scale_count_range(rng, *green_snails_per_large, mode)
        n_red = scale_count_range(rng, *red_snails_per_large, mode)
        for _ in range(n_mush):
            place_sprite_box(
                canvas,
                scene,
                "花蘑菇",
                sprites["花蘑菇"],
                floor,
                rng,
                map_w=mw,
                map_h=mh,
                occlude=scene.sprite_boxes,
                into_mobs=mob_boxes,
            )
        for _ in range(n_stump):
            place_sprite_box(
                canvas,
                scene,
                "树怪",
                sprites["树怪"],
                floor,
                rng,
                map_w=mw,
                map_h=mh,
                occlude=scene.sprite_boxes,
                into_mobs=mob_boxes,
            )
        for _ in range(n_blue):
            place_sprite_box(
                canvas,
                scene,
                "蓝蜗牛",
                sprites["蓝蜗牛"],
                floor,
                rng,
                map_w=mw,
                map_h=mh,
                occlude=scene.sprite_boxes,
                into_mobs=mob_boxes,
            )
        for _ in range(n_green):
            place_sprite_box(
                canvas,
                scene,
                "绿蜗牛",
                sprites["绿蜗牛"],
                floor,
                rng,
                map_w=mw,
                map_h=mh,
                occlude=scene.sprite_boxes,
                into_mobs=mob_boxes,
            )
        for _ in range(n_red):
            place_sprite_box(
                canvas,
                scene,
                "红蜗牛",
                sprites["红蜗牛"],
                floor,
                rng,
                map_w=mw,
                map_h=mh,
                occlude=scene.sprite_boxes,
                into_mobs=mob_boxes,
            )
        n_drop = drops_count_for_mode(rng, mode, drops_per_large)
        place_drops_on_floor(
            canvas, scene, sprites, floor, rng,
            map_w=mw, map_h=mh, n=n_drop, mode=mode,
        )

    small_floors = [f for f in floors if f not in large_floors]
    for floor in small_floors:
        roll = rng.random()
        if roll < 0.15:
            # 小地板仅掉落
            place_drops_on_floor(
                canvas,
                scene,
                sprites,
                floor,
                rng,
                map_w=mw,
                map_h=mh,
                n=rng.randint(1, 2),
            )
        elif roll < 0.40:
            label = rng.choice(MOB_LABELS)
            place_sprite_box(
                canvas,
                scene,
                label,
                sprites[label],
                floor,
                rng,
                map_w=mw,
                map_h=mh,
                occlude=scene.sprite_boxes,
                into_mobs=mob_boxes,
            )

    # 3) 玩家：后贴（上层）；与已有精灵互遮不超过阈值
    n_players = rng.randint(*players_n)
    placed_players = 0
    attempts = 0
    while placed_players < n_players and attempts < n_players * 40:
        attempts += 1
        floor = rng.choice(floors)
        spr, desc = pick_sprite(sprites["玩家"], rng, allow_flip=True)
        sw, sh = spr.size
        pos = try_place_on_floor(
            floor,
            sw,
            sh,
            mw,
            mh,
            rng,
            blocked=portal_boxes,
            occlude=scene.sprite_boxes,
        )
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
        paste_sprite(canvas, spr, x, y)
        scene.sprite_boxes.append(cand)
        placed_players += 1

    # 3b) 半空玩家：优先 jump 帧，脚底抬离地板（模拟起跳/空中）
    jump_pool = sprite_paths_with_prefix(sprites["玩家"], "jump")
    airborne_pool = jump_pool or sprites["玩家"]
    n_airborne = rng.randint(*airborne_n)
    placed_airborne = 0
    attempts = 0
    while placed_airborne < n_airborne and attempts < n_airborne * 40:
        attempts += 1
        floor = rng.choice(floors)
        spr, desc = pick_sprite(airborne_pool, rng, allow_flip=True)
        sw, sh = spr.size
        pos = try_place_airborne(
            floor,
            sw,
            sh,
            mw,
            mh,
            rng,
            blocked=portal_boxes,
            occlude=scene.sprite_boxes,
        )
        if not pos:
            continue
        x, y = pos
        paste_sprite(canvas, spr, x, y)
        scene.sprite_boxes.append(
            Box(
                "玩家",
                float(x),
                float(y),
                float(x + sw),
                float(y + sh),
                description=f"{desc} airborne=1",
            )
        )
        placed_airborne += 1

    # 4) 攀爬玩家：贴在绳子/梯子上（优先背部 ladder/rope 帧，否则用正面帧）
    verticals = [b for b in base_boxes if b.label in ("绳子", "梯子")]
    climb_pool = sprites.get("玩家攀爬") or []
    if not climb_pool:
        climb_pool = sprites["玩家"]
    n_climb = min(rng.randint(*climbers_n), len(verticals)) if verticals else 0
    verts_shuffled = list(verticals)
    rng.shuffle(verts_shuffled)
    placed_climb = 0
    for vert in verts_shuffled:
        if placed_climb >= n_climb:
            break
        # 绳子优先用 rope 帧，梯子优先 ladder；没有则整池随机
        prefer = "rope" if vert.label == "绳子" else "ladder"
        preferred = [p for p in climb_pool if p.name.lower().startswith(prefer)]
        pool = preferred or climb_pool
        spr, desc = pick_sprite(pool, rng, allow_flip=False)
        sw, sh = spr.size
        pos = try_place_on_vertical(
            vert, sw, sh, mw, mh, rng, occlude=scene.sprite_boxes
        )
        if not pos:
            continue
        x, y = pos
        paste_sprite(canvas, spr, x, y)
        scene.sprite_boxes.append(
            Box(
                "玩家",
                float(x),
                float(y),
                float(x + sw),
                float(y + sh),
                description=f"{desc} climb_on={vert.label}",
            )
        )
        placed_climb += 1

    # 3c) 战斗姿态：swingO1/O2/O3、stab、shoot；优先 O1/O2，约一半半空
    combat_pool = sprites.get("玩家战斗") or []
    preferred_combat = [
        p
        for p in combat_pool
        if any(p.name.lower().startswith(pref) for pref in COMBAT_PREFER_PREFIX)
    ]
    # 70% 从优先池抽，30% 从全战斗池抽，保证 O1/O2/stab/shoot 覆盖且仍有多样性
    combat_pick_pool = preferred_combat or combat_pool
    n_combat = rng.randint(*combat_n) if combat_pick_pool else 0
    placed_combat = 0
    attempts = 0
    while placed_combat < n_combat and attempts < n_combat * 40:
        attempts += 1
        floor = rng.choice(floors)
        pool = (
            preferred_combat
            if preferred_combat and rng.random() < 0.7
            else combat_pick_pool
        )
        spr, desc = pick_sprite(pool, rng, allow_flip=True)
        sw, sh = spr.size
        use_airborne = rng.random() < 0.5
        if use_airborne:
            pos = try_place_airborne(
                floor,
                sw,
                sh,
                mw,
                mh,
                rng,
                blocked=portal_boxes,
                occlude=scene.sprite_boxes,
            )
            tag = f"{desc} airborne=1 combat=1"
        else:
            pos = try_place_on_floor(
                floor,
                sw,
                sh,
                mw,
                mh,
                rng,
                blocked=portal_boxes,
                occlude=scene.sprite_boxes,
            )
            tag = f"{desc} combat=1"
        if not pos:
            continue
        x, y = pos
        paste_sprite(canvas, spr, x, y)
        scene.sprite_boxes.append(
            Box(
                "玩家",
                float(x),
                float(y),
                float(x + sw),
                float(y + sh),
                description=tag,
            )
        )
        placed_combat += 1

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
    if box.label in DROP_LABELS and (bw < MIN_DROP_BOX_PX or bh < MIN_DROP_BOX_PX):
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


def dedupe_boxes(boxes: list[Box], iou_thr: float = 0.92) -> list[Box]:
    """去掉几乎完全重叠的重复框（保留面积较大者）。"""
    kept: list[Box] = []
    for b in sorted(boxes, key=lambda x: -x.area()):
        if any(b.iou(k) > iou_thr and b.label == k.label for k in kept):
            continue
        kept.append(b)
    return kept


def validate_crop_boxes(boxes: list[Box], img_w: int, img_h: int) -> list[str]:
    issues: list[str] = []
    for b in boxes:
        if b.label not in CLASS_TO_ID:
            issues.append(f"未知标签: {b.label}")
            continue
        x1, y1, x2, y2 = b.as_xyxy()
        if x1 < -1 or y1 < -1 or x2 > img_w + 1 or y2 > img_h + 1:
            issues.append(f"{b.label} 越界")
        if b.label in MOB_LABELS and (b.width() < 8 or b.height() < 8):
            issues.append(f"{b.label} 框过小")
    # 同类极高 IoU 重复
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if a.label == b.label and a.iou(b) > 0.95:
                issues.append(f"重复框: {a.label}")
                break
    return issues


def audit_yolo_dataset(yolo_dir: Path) -> dict[str, int]:
    """统计各类窗口标注数，返回计数。"""
    counts: dict[str, int] = defaultdict(int)
    for split in ("train", "val"):
        lbl_dir = yolo_dir / "labels" / split
        if not lbl_dir.is_dir():
            continue
        for txt in lbl_dir.glob("*.txt"):
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                cid = int(parts[0])
                if 0 <= cid < len(CLASS_NAMES):
                    counts[CLASS_NAMES[cid]] += 1
    return dict(counts)


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


def list_drop_frames(folder: Path) -> list[Path]:
    frames = []
    for p in sorted(folder.glob("*.png")):
        name = p.name.lower()
        if name.startswith("_") or "atlas" in name:
            continue
        try:
            w, h = Image.open(p).size
            if w < MIN_DROP_BOX_PX or h < MIN_DROP_BOX_PX:
                continue
        except OSError:
            continue
        frames.append(p)
    return frames


def load_all_sprites(assets: Path) -> dict[str, list[Path]]:
    """返回各类精灵的帧文件路径；贴图时再随机抽一帧加载。"""
    portal_dir = assets / "portals" / "pv_可见传送门"
    if not portal_dir.is_dir():
        cands = list((assets / "portals").glob("pv*"))
        portal_dir = cands[0] if cands else portal_dir

    player_frames: list[Path] = []
    climb_frames: list[Path] = []
    combat_frames: list[Path] = []
    player_root = assets / "player"
    preset_dirs: list[Path] = []
    if player_root.is_dir():
        for sub in sorted(player_root.iterdir()):
            if not sub.is_dir():
                continue
            # 跳过部件/图集备份子目录
            if sub.name in ("atlases_from_game", "parts_from_game"):
                continue
            preset_dirs.append(sub)
            player_frames.extend(list_sprite_frames(sub, PLAYER_KEEP_PREFIX))
            climb_frames.extend(list_sprite_frames(sub, PLAYER_CLIMB_PREFIX))
            combat_frames.extend(list_sprite_frames(sub, PLAYER_COMBAT_PREFIX))

    sprites: dict[str, list[Path]] = {
        "portal": list_sprite_frames(portal_dir, PORTAL_KEEP_PREFIX),
    }
    mobs_root = assets / "mobs"
    for dirname, label in MOB_ASSET_DIRS:
        if label in sprites and sprites[label]:
            continue
        folder = mobs_root / dirname
        if not folder.is_dir():
            continue
        frames = list_sprite_frames(folder, MOB_KEEP_PREFIX)
        if frames:
            sprites[label] = frames
    sprites["玩家"] = player_frames
    sprites["玩家攀爬"] = climb_frames
    sprites["玩家战斗"] = combat_frames

    drops_root = assets / "drops"
    for dirname, label in DROP_DIR_TO_LABEL.items():
        frames = list_drop_frames(drops_root / dirname) if drops_root.is_dir() else []
        sprites[label] = frames
    for label in MOB_LABELS:
        if not sprites.get(label):
            raise FileNotFoundError(f"缺少怪物精灵: {label}（请先 extract_sprites --mob ...）")
    for k in ("portal", "玩家", *DROP_LABELS):
        if not sprites.get(k):
            raise FileNotFoundError(f"精灵帧为空: {k}（检查 assets 目录）")

    if not climb_frames:
        print("提示: 无 ladder/rope 背部帧，攀爬贴图将回退为正面玩家帧")
    if not combat_frames:
        print("提示: 无 swing/stab/shoot 战斗帧，战斗玩家贴图将跳过")
    else:
        # 按动作前缀统计，确认 O1/O2 等已进入池
        pref_cnt = Counter()
        for p in combat_frames:
            n = p.name.lower()
            key = n.rsplit("_", 1)[0]
            pref_cnt[key] += 1
        top = ", ".join(f"{k}×{v}" for k, v in sorted(pref_cnt.items()))
        print(
            f"玩家预设 {len(preset_dirs)} 个 | 站立/走跳 {len(player_frames)} | "
            f"攀爬 {len(climb_frames)} | 战斗 {len(combat_frames)}（{top}）"
        )
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
    ap.add_argument("--full-maps", type=int, default=80, help="完整大图副本数量（默认 80×15≈1200 窗）")
    ap.add_argument("--crops-per-map", type=int, default=15, help="每张大图裁切窗口数")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--players-min", type=int, default=6)
    ap.add_argument("--players-max", type=int, default=8)
    ap.add_argument("--climbers-min", type=int, default=2, help="每张大图攀爬玩家最少个数")
    ap.add_argument("--climbers-max", type=int, default=4, help="每张大图攀爬玩家最多个数")
    ap.add_argument("--airborne-min", type=int, default=2, help="每张大图半空跳跃玩家最少个数")
    ap.add_argument("--airborne-max", type=int, default=4, help="每张大图半空跳跃玩家最多个数")
    ap.add_argument("--combat-min", type=int, default=2, help="每张大图战斗姿态玩家最少个数")
    ap.add_argument("--combat-max", type=int, default=4, help="每张大图战斗姿态玩家最多个数")
    ap.add_argument("--snails-per-large-min", type=int, default=2, help="蓝蜗牛/大地板下限（兼容旧参数名）")
    ap.add_argument("--snails-per-large-max", type=int, default=4, help="蓝蜗牛/大地板上限")
    ap.add_argument("--red-snails-per-large-min", type=int, default=1)
    ap.add_argument("--red-snails-per-large-max", type=int, default=3)
    ap.add_argument("--green-snails-per-large-min", type=int, default=2)
    ap.add_argument("--green-snails-per-large-max", type=int, default=4)
    ap.add_argument("--mushrooms-per-large-min", type=int, default=2)
    ap.add_argument("--mushrooms-per-large-max", type=int, default=3)
    ap.add_argument("--stumps-per-large-min", type=int, default=2)
    ap.add_argument("--stumps-per-large-max", type=int, default=4)
    ap.add_argument("--drops-per-large-min", type=int, default=4, help="正常/仅掉落地板掉落物数量下限")
    ap.add_argument("--drops-per-large-max", type=int, default=8, help="正常/仅掉落地板掉落物数量上限")
    ap.add_argument("--portals-min", type=int, default=1)
    ap.add_argument("--portals-max", type=int, default=2)
    ap.add_argument(
        "--ui-ratio",
        type=float,
        default=0.65,
        help="带 UI 的窗口样本比例（其余为无 UI，默认 0.65）",
    )
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
    ui_sprites = load_ui_sprites(assets)
    print(
        "帧库: "
        + ", ".join(f"{k}×{len(v)}" for k, v in sprites.items())
    )
    print("UI: " + ", ".join(f"{k}={v.size}" for k, v in ui_sprites.items()))
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
        f"半空 {args.airborne_min}~{args.airborne_max}, "
        f"战斗 {args.combat_min}~{args.combat_max}, "
        f"攀爬 {args.climbers_min}~{args.climbers_max}, "
        f"蓝蜗牛 {args.snails_per_large_min}~{args.snails_per_large_max}, "
        f"绿蜗牛 {args.green_snails_per_large_min}~{args.green_snails_per_large_max}, "
        f"红蜗牛 {args.red_snails_per_large_min}~{args.red_snails_per_large_max}, "
        f"花蘑菇 {args.mushrooms_per_large_min}~{args.mushrooms_per_large_max}, "
        f"树怪 {args.stumps_per_large_min}~{args.stumps_per_large_max}, "
        f"掉落 {args.drops_per_large_min}~{args.drops_per_large_max}, "
        f"UI比例 {args.ui_ratio:.0%} | "
        f"目标窗口≈{args.full_maps * args.crops_per_map}"
    )

    crop_records: list[tuple[Image.Image, list[Box], str]] = []
    stats = defaultdict(int)
    n_with_ui = 0
    n_without_ui = 0

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
            blue_snails_per_large=(args.snails_per_large_min, args.snails_per_large_max),
            red_snails_per_large=(
                args.red_snails_per_large_min,
                args.red_snails_per_large_max,
            ),
            green_snails_per_large=(
                args.green_snails_per_large_min,
                args.green_snails_per_large_max,
            ),
            stumps_per_large=(args.stumps_per_large_min, args.stumps_per_large_max),
            drops_per_large=(args.drops_per_large_min, args.drops_per_large_max),
            players_n=(args.players_min, args.players_max),
            climbers_n=(args.climbers_min, args.climbers_max),
            airborne_n=(args.airborne_min, args.airborne_max),
            combat_n=(args.combat_min, args.combat_max),
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
            cropped_boxes = dedupe_boxes(cropped_boxes)
            issues = validate_crop_boxes(cropped_boxes, ww, wh)
            if issues:
                print(f"  WARN {crop_name}: " + "; ".join(issues[:3]))
            use_ui = rng.random() < args.ui_ratio
            if use_ui:
                crop, ui_boxes = paste_window_ui(crop, ui_sprites, rng)
                cropped_boxes.extend(ui_boxes)
                crop_name = f"{name}_c{ci:02d}_{ww}x{wh}_ui"
                n_with_ui += 1
            else:
                crop_name = f"{name}_c{ci:02d}_{ww}x{wh}"
                n_without_ui += 1
            crop_records.append((crop, cropped_boxes, crop_name))

        print(
            f"  [{mi+1}/{args.full_maps}] sprites="
            + ", ".join(
                f"{k}={sum(1 for b in scene.sprite_boxes if b.label==k)}"
                for k in (
                    "入口",
                    "出口",
                    "花蘑菇",
                    "蓝蜗牛",
                    "绿蜗牛",
                    "红蜗牛",
                    "树怪",
                    "金币",
                    "药水",
                    "武器",
                    "装备",
                    "材料",
                    "玩家",
                )
            )
            + f", 攀爬={sum(1 for b in scene.sprite_boxes if b.label=='玩家' and 'climb_on=' in b.description)}"
            + f", 半空={sum(1 for b in scene.sprite_boxes if b.label=='玩家' and 'airborne=1' in b.description)}"
            + f", 战斗={sum(1 for b in scene.sprite_boxes if b.label=='玩家' and 'combat=1' in b.description)}"
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

    class_counts = audit_yolo_dataset(yolo_dir)
    print("窗口标注统计:")
    for name in CLASS_NAMES:
        if name in class_counts:
            print(f"  {name}: {class_counts[name]}")

    summary = (
        f"完成\n"
        f"  完整大图: {args.full_maps} → {full_dir}\n"
        f"  窗口样本: {len(crop_records)} "
        f"(train={len(crop_records)-n_val}, val={n_val}) → {yolo_dir}\n"
        f"  UI: 有={n_with_ui}, 无={n_without_ui} "
        f"(目标比例 {args.ui_ratio:.0%})\n"
        f"  大图精灵累计: {dict(stats)}\n"
        f"  窗口标注累计: {class_counts}\n"
        f"  data.yaml: {yolo_dir / 'data.yaml'} ({len(CLASS_NAMES)} 类)\n"
    )
    (out_dir / "README.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
