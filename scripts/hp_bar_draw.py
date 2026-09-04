"""冒险岛怀旧服风格 HP 血条绘制（与 screen_caps / 面板_UI 像素对齐）。

面板 `assets/ui/面板_UI.png`（815×72）内实测：
- 血条填充槽：相对面板 (228, 53, 104, 14)
- 标注框（含 HP 字样与数值）：(226, 38, 109, 30)

供 auto_annotate_dataset 与预览脚本共用。
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# 相对面板左上角（assets/ui/面板_UI.png）
HP_TRACK_XYWH = (228, 53, 104, 14)  # 填充槽内区
HP_WIDGET_XYWH = (226, 38, 109, 30)  # YOLO 标注框
HP_NUM_XYWH = (248, 39, 82, 12)  # 盖住旧 [cur/max] 再重绘

# 自上而下 14 行：对齐截图红条高光→暗边
HP_FILL_ROWS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),
    (255, 119, 119),
    (252, 48, 48),
    (255, 0, 0),
    (238, 0, 0),
    (238, 0, 0),
    (238, 0, 0),
    (221, 0, 0),
    (204, 0, 0),
    (170, 0, 0),
    (102, 0, 0),
    (102, 0, 0),
    (102, 0, 0),
    (170, 0, 0),
)

HP_EMPTY_ROWS: tuple[tuple[int, int, int], ...] = (
    (204, 204, 204),
    (228, 228, 228),
    (211, 211, 211),
    (204, 204, 204),
    (190, 190, 190),
    (190, 190, 190),
    (190, 190, 190),
    (177, 177, 177),
    (163, 163, 163),
    (136, 136, 136),
    (82, 82, 82),
    (82, 82, 82),
    (82, 82, 82),
    (136, 136, 136),
)

# YOLO 类别：血条10% … 血条100%
HP_PCT_STEPS: tuple[int, ...] = tuple(range(10, 101, 10))
HP_CLASS_NAMES: tuple[str, ...] = tuple(f"血条{p}%" for p in HP_PCT_STEPS)


def hp_class_name(pct_bucket: int) -> str:
    pct_bucket = max(10, min(100, int(pct_bucket)))
    pct_bucket = int(round(pct_bucket / 10.0) * 10)
    pct_bucket = max(10, min(100, pct_bucket))
    return f"血条{pct_bucket}%"


def ratio_to_hp_bucket(ratio: float) -> int:
    """将 [0,1] 血量映射到 10/20/…/100。0% 空条归入 10%（危急档）。"""
    r = max(0.0, min(1.0, float(ratio)))
    pct = int(round(r * 100.0))
    if pct <= 0:
        return 10
    bucket = int(round(pct / 10.0) * 10)
    return max(10, min(100, bucket))


def ratio_to_hp_label(ratio: float) -> str:
    return hp_class_name(ratio_to_hp_bucket(ratio))


def _put_row(
    px: Image.Image,
    x0: int,
    y: int,
    width: int,
    color: tuple[int, int, int],
) -> None:
    if width <= 0:
        return
    draw = ImageDraw.Draw(px)
    draw.line([(x0, y), (x0 + width - 1, y)], fill=color)


def draw_hp_fill(
    img: Image.Image,
    *,
    track_x: int,
    track_y: int,
    track_w: int,
    track_h: int,
    ratio: float,
) -> None:
    """在已有坐标系下绘制血条填充（先铺空槽，再按比例画红条）。"""
    ratio = max(0.0, min(1.0, float(ratio)))
    fill_w = int(round(track_w * ratio))
    h = min(track_h, len(HP_FILL_ROWS))
    # 空槽
    for i in range(h):
        _put_row(img, track_x, track_y + i, track_w, HP_EMPTY_ROWS[i])
    # 红条 + 每隔 4px 暗竖线（截图里的分格感）
    if fill_w <= 0:
        return
    for i in range(h):
        _put_row(img, track_x, track_y + i, fill_w, HP_FILL_ROWS[i])
    draw = ImageDraw.Draw(img)
    rib = (180, 0, 0)
    for x in range(track_x + 4, track_x + fill_w, 4):
        draw.line([(x, track_y + 1), (x, track_y + h - 2)], fill=rib)


def _try_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "consola.ttf",
        "consolab.ttf",
        "lucon.ttf",
        "arialbd.ttf",
        "arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/lucon.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_hp_numbers(
    img: Image.Image,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    cur: int,
    mx: int,
    bg: tuple[int, int, int] = (40, 48, 55),
) -> None:
    """盖住旧数值区域，绘制 [cur/max]。"""
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=bg)
    text = f"[{cur}/{mx}]"
    font = _try_font(11)
    # 近似垂直居中
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    tx = x + max(0, (w - tw) // 2)
    ty = y + max(0, (h - th) // 2 - 1)
    draw.text((tx, ty), text, fill=(220, 255, 180), font=font)


def apply_hp_to_panel(
    panel: Image.Image,
    ratio: float,
    *,
    max_hp: int = 100,
    scale: float = 1.0,
    origin_xy: tuple[int, int] = (0, 0),
) -> tuple[tuple[int, int, int, int], str, float]:
    """在面板图（或已贴到窗口的面板区域）上重绘血条。

    返回：(标注框 x1,y1,x2,y2 相对 img), 类别名, 实际绘制比例。
    origin_xy：面板左上角在 img 中的位置；scale：面板相对原始 815×72 的缩放。
    """
    ratio = max(0.0, min(1.0, float(ratio)))
    ox, oy = origin_xy
    s = float(scale)

    def sc_box(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
        return (
            ox + int(round(x * s)),
            oy + int(round(y * s)),
            max(1, int(round(w * s))),
            max(1, int(round(h * s))),
        )

    tx, ty, tw, th = sc_box(*HP_TRACK_XYWH)
    draw_hp_fill(panel, track_x=tx, track_y=ty, track_w=tw, track_h=th, ratio=ratio)

    nx, ny, nw, nh = sc_box(*HP_NUM_XYWH)
    cur = int(round(max_hp * ratio))
    draw_hp_numbers(panel, x=nx, y=ny, w=nw, h=nh, cur=cur, mx=max_hp)

    wx, wy, ww, wh = sc_box(*HP_WIDGET_XYWH)
    label = ratio_to_hp_label(ratio)
    return (wx, wy, wx + ww, wy + wh), label, ratio


def sample_hp_ratio(rng) -> float:
    """数据集抽样：70% 精确 0/10/…/100，其余连续。"""
    if rng.random() < 0.70:
        return rng.choice([0.0] + [p / 100.0 for p in HP_PCT_STEPS])
    return rng.random()
