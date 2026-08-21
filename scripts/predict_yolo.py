#!/usr/bin/env python3
"""YOLO 推理：保存原图 + labelme JSON（便于 labelme 查看）+ 可选可视化预览。

用法:
  python scripts/predict_yolo.py \\
    --model models/yolo_nangang_e1500_best.pt \\
    --source screen_caps/彩虹岛-南港西郊平原/ScreenShot_2026-08-20_095130_246.png

  # 整目录批量
  python scripts/predict_yolo.py \\
    --model models/yolo_nangang_e1500_best.pt \\
    --source screen_caps/彩虹岛-南港西郊平原 \\
    --out runs/detect/yolo_nangang_e1500_labelme
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    for (x1, y1, x2, y2), cid, conf in zip(xyxy, cls_ids, confs):
        color = class_color(int(cid))
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, line_width, cv2.LINE_AA)

    rgba = cv2.cvtColor(out, cv2.COLOR_BGR2RGBA)
    base = Image.fromarray(rgba)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)

    for (x1, y1, x2, y2), cid, conf in zip(xyxy, cls_ids, confs):
        color = class_color(int(cid))
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


def save_labelme_json(
    path: Path,
    image_name: str,
    width: int,
    height: int,
    xyxy: np.ndarray,
    cls_ids: np.ndarray,
    confs: np.ndarray,
    names: dict,
) -> None:
    shapes = []
    for (x1, y1, x2, y2), cid, conf in zip(xyxy, cls_ids, confs):
        label = str(names.get(int(cid), cid))
        shapes.append(
            {
                "label": label,
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "description": f"conf={float(conf):.4f}",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            }
        )
    data = {
        "version": "5.10.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_sources(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        imgs = sorted(
            list(source.glob("*.png"))
            + list(source.glob("*.jpg"))
            + list(source.glob("*.jpeg"))
            + list(source.glob("*.bmp"))
            + list(source.glob("*.webp"))
        )
        if not imgs:
            raise SystemExit(f"目录下没有图片: {source}")
        return imgs
    raise SystemExit(f"找不到图片或目录: {source}")


def predict_one(
    model,
    source: Path,
    out_dir: Path,
    *,
    conf: float,
    imgsz: int,
    device: str,
    font_size: int,
    line_width: int,
    bg_alpha: float,
    save_styled: bool,
) -> None:
    results = model.predict(
        source=str(source),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    r0 = results[0]
    im = r0.orig_img
    h, w = im.shape[:2]

    # 原图：尽量保持 png，便于 labelme
    img_name = f"{source.stem}.png"
    img_path = out_dir / img_name
    json_path = out_dir / f"{source.stem}.json"

    if source.suffix.lower() == ".png":
        shutil.copy2(source, img_path)
    else:
        # BGR -> 写 png
        cv2.imwrite(str(img_path), im)

    if r0.boxes is None or len(r0.boxes) == 0:
        xyxy = np.zeros((0, 4), dtype=np.float32)
        cls_ids = np.zeros((0,), dtype=int)
        confs = np.zeros((0,), dtype=np.float32)
    else:
        xyxy = r0.boxes.xyxy.cpu().numpy()
        cls_ids = r0.boxes.cls.cpu().numpy().astype(int)
        confs = r0.boxes.conf.cpu().numpy()

    save_labelme_json(
        json_path, img_name, w, h, xyxy, cls_ids, confs, r0.names
    )

    styled_path = None
    if save_styled and len(cls_ids) > 0:
        annotated = draw_detections(
            im,
            xyxy,
            cls_ids,
            confs,
            r0.names,
            font_size=font_size,
            line_width=line_width,
            bg_alpha=bg_alpha,
        )
        styled_path = out_dir / f"{source.stem}_styled.jpg"
        cv2.imwrite(str(styled_path), annotated)
    elif save_styled:
        styled_path = out_dir / f"{source.stem}_styled.jpg"
        cv2.imwrite(str(styled_path), im)

    print(
        f"{source.name}: detections={len(cls_ids)} "
        f"-> {img_path.name} + {json_path.name}"
        + (f" + {styled_path.name}" if styled_path else "")
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="YOLO 推理 → 原图 + labelme JSON（+ 可选可视化）"
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=Path("models/yolo_nangang_e1500_best.pt"),
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=Path(
            "screen_caps/彩虹岛-南港西郊平原/ScreenShot_2026-08-20_095130_246.png"
        ),
        help="单张图片或目录",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录（默认 runs/detect/predict_labelme）",
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--font-size", type=int, default=14)
    ap.add_argument("--line-width", type=int, default=2)
    ap.add_argument("--bg-alpha", type=float, default=0.45)
    ap.add_argument(
        "--no-styled",
        action="store_true",
        help="不生成 *_styled.jpg 预览图（仍写原图+json）",
    )
    args = ap.parse_args()

    from ultralytics import YOLO

    model_path = resolve(args.model)
    source = resolve(args.source)
    if not model_path.is_file():
        alt = mxd_tools_root() / "models/yolo_nangang_e1500/weights/best.pt"
        if alt.is_file():
            model_path = alt
        else:
            raise SystemExit(f"找不到模型: {model_path}")

    sources = collect_sources(source)
    out_dir = resolve(args.out) if args.out else (
        mxd_tools_root() / "runs" / "detect" / "predict_labelme"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    print(f"model: {model_path}")
    print(f"out:   {out_dir}")
    for src in sources:
        predict_one(
            model,
            src,
            out_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            font_size=args.font_size,
            line_width=args.line_width,
            bg_alpha=args.bg_alpha,
            save_styled=not args.no_styled,
        )
    print(f"完成 {len(sources)} 张。用 labelme 打开目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
