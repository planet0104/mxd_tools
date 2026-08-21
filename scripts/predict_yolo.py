#!/usr/bin/env python3
"""YOLO 推理并保存标注图：小字号 + 半透明文字底，便于看清误检。

用法:
  python scripts/predict_yolo.py \\
    --model models/yolo_nangang_e5000_best.pt \\
    --source screen_caps/彩虹岛-南港西郊平原/ScreenShot_2026-08-20_095130_246.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def mxd_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(p: Path) -> Path:
    return p if p.is_absolute() else (mxd_tools_root() / p).resolve()


def find_cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path.home() / "AppData/Roaming/Ultralytics/Arial.Unicode.ttf",
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for fp in candidates:
        if fp.is_file():
            try:
                return ImageFont.truetype(str(fp), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def class_color(i: int) -> tuple[int, int, int]:
    # BGR 常见调色（与 ultralytics 类似的区分色）
    palette = [
        (255, 56, 56),
        (255, 157, 151),
        (255, 112, 31),
        (255, 178, 29),
        (207, 210, 49),
        (72, 249, 10),
        (146, 204, 23),
        (61, 219, 134),
        (26, 147, 52),
        (0, 212, 187),
        (44, 153, 168),
        (0, 194, 255),
        (52, 69, 147),
        (100, 115, 255),
        (0, 24, 236),
        (132, 56, 255),
        (82, 0, 133),
        (203, 56, 255),
        (255, 149, 200),
        (255, 55, 199),
    ]
    return palette[i % len(palette)]


def draw_detections(
    bgr: np.ndarray,
    xyxy: np.ndarray,
    cls_ids: np.ndarray,
    confs: np.ndarray,
    names: dict,
    *,
    font_size: int = 14,
    line_width: int = 2,
    bg_alpha: float = 0.45,
) -> np.ndarray:
    """画框：细线 + 小字 + 半透明标签底。"""
    out = bgr.copy()
    font = find_cjk_font(font_size)
    # 先画框
    for (x1, y1, x2, y2), cid, conf in zip(xyxy, cls_ids, confs):
        color = class_color(int(cid))
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, line_width, cv2.LINE_AA)

    # 标签用 PIL 半透明底
    rgba = cv2.cvtColor(out, cv2.COLOR_BGR2RGBA)
    base = Image.fromarray(rgba)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)

    for (x1, y1, x2, y2), cid, conf in zip(xyxy, cls_ids, confs):
        color = class_color(int(cid))
        # PIL 用 RGB
        rgb = (color[2], color[1], color[0])
        label = f"{names[int(cid)]} {conf:.2f}"
        bbox = font.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 2
        tx = int(x1)
        ty = int(y1) - th - pad * 2
        if ty < 0:
            ty = int(y1) + pad
        if tx + tw + pad * 2 > base.size[0]:
            tx = max(0, base.size[0] - tw - pad * 2)

        alpha = int(round(255 * bg_alpha))
        draw.rectangle(
            (tx, ty, tx + tw + pad * 2, ty + th + pad * 2),
            fill=(*rgb, alpha),
        )
        text_draw.text((tx + pad, ty + pad), label, font=font, fill=(255, 255, 255, 255))

    composed = Image.alpha_composite(base, overlay)
    composed = Image.alpha_composite(composed, text_layer)
    return cv2.cvtColor(np.asarray(composed.convert("RGB")), cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO 推理（小字号半透明标签）")
    ap.add_argument(
        "--model",
        type=Path,
        default=Path("models/yolo_nangang_e5000_best.pt"),
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=Path(
            "screen_caps/彩虹岛-南港西郊平原/ScreenShot_2026-08-20_095130_246.png"
        ),
    )
    ap.add_argument("--out", type=Path, default=None, help="输出图片路径")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--font-size", type=int, default=14, help="标签字号（默认 14，比 ultralytics 默认更小）")
    ap.add_argument("--line-width", type=int, default=2)
    ap.add_argument("--bg-alpha", type=float, default=0.45, help="文字底透明度 0~1")
    args = ap.parse_args()

    from ultralytics import YOLO

    model_path = resolve(args.model)
    source = resolve(args.source)
    if not model_path.is_file():
        # 兜底
        alt = mxd_tools_root() / "models/yolo_nangang_e5000/weights/best.pt"
        if alt.is_file():
            model_path = alt
        else:
            raise SystemExit(f"找不到模型: {model_path}")
    if not source.is_file():
        raise SystemExit(f"找不到图片: {source}")

    out = args.out
    if out is None:
        out = (
            mxd_tools_root()
            / "runs"
            / "detect"
            / "predict_styled"
            / f"{source.stem}_styled.jpg"
        )
    else:
        out = resolve(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    r0 = results[0]
    im = r0.orig_img.copy()
    if r0.boxes is None or len(r0.boxes) == 0:
        cv2.imwrite(str(out), im)
        print(f"无检测框，已原样保存: {out}")
        return 0

    xyxy = r0.boxes.xyxy.cpu().numpy()
    cls_ids = r0.boxes.cls.cpu().numpy().astype(int)
    confs = r0.boxes.conf.cpu().numpy()
    annotated = draw_detections(
        im,
        xyxy,
        cls_ids,
        confs,
        r0.names,
        font_size=args.font_size,
        line_width=args.line_width,
        bg_alpha=args.bg_alpha,
    )
    cv2.imwrite(str(out), annotated)
    print(f"detections={len(cls_ids)} font={args.font_size} bg_alpha={args.bg_alpha}")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
