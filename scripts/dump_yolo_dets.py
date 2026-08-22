#!/usr/bin/env python3
"""把 YOLO(.pt) 推理结果导出为 JSON，供 Rust yolo_compare 比对。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [p for p in sorted(source.iterdir()) if p.suffix.lower() in exts]
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    images = collect(args.source)
    payload = {"model": str(args.model), "conf": args.conf, "iou": args.iou, "images": []}

    for path in images:
        r = model.predict(
            str(path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        h, w = r.orig_shape
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cid, conf in zip(xyxy, cls_ids, confs):
                dets.append(
                    {
                        "class_id": int(cid),
                        "label": str(r.names[int(cid)]),
                        "conf": float(conf),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )
        payload["images"].append(
            {
                "file": path.name,
                "path": str(path),
                "width": int(w),
                "height": int(h),
                "dets": dets,
            }
        )
        print(f"{path.name}: {len(dets)} boxes")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
