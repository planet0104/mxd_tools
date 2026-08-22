#!/usr/bin/env python3
"""用 ultralytics 训练 YOLO 检测模型（对接 auto_annotate 生成的数据集）。

依赖:
  pip install ultralytics

用法:
  python scripts/train_yolo.py \\
    --data dataset/彩虹岛-南港西郊平原/generated/yolo/data.yaml

  python scripts/train_yolo.py \\
    --data dataset/彩虹岛-南港西郊平原/generated/yolo/data.yaml \\
    --model yolo11n.pt --epochs 100 --imgsz 640 --batch 8
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def mxd_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (mxd_tools_root() / p).resolve()


def prepare_data_yaml(src: Path, out_dir: Path) -> Path:
    """把 data.yaml 的 path 改成绝对路径，避免 cwd 影响。"""
    try:
        import yaml
    except ImportError:
        # ultralytics 一般会带上 pyyaml；没有则手写最小解析
        yaml = None

    text = src.read_text(encoding="utf-8")
    data_root = src.parent.resolve()

    if yaml is not None:
        cfg = yaml.safe_load(text) or {}
        cfg["path"] = str(data_root)
        if "train" not in cfg:
            cfg["train"] = "images/train"
        if "val" not in cfg:
            cfg["val"] = "images/val"
        out = out_dir / "data.abs.yaml"
        out.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return out

    # 无 pyyaml：只替换 path 行
    lines = []
    replaced = False
    for line in text.splitlines():
        if line.strip().startswith("path:"):
            lines.append(f"path: {data_root.as_posix()}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.insert(0, f"path: {data_root.as_posix()}")
    out = out_dir / "data.abs.yaml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def count_split(data_yaml: Path, split: str) -> int:
    root = data_yaml.parent
    img_dir = root / "images" / split
    if not img_dir.is_dir():
        return 0
    return len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))


def main() -> int:
    ap = argparse.ArgumentParser(description="训练 YOLO 检测模型")
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/彩虹岛-南港西郊平原/generated/yolo/data.yaml"),
        help="data.yaml 路径",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="预训练权重，如 yolo11n.pt / yolo11s.pt / yolov8n.pt",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda 设备号如 0，或 cpu；默认优先 GPU",
    )
    ap.add_argument(
        "--project",
        type=Path,
        default=None,
        help="训练输出根目录（默认 mxd_tools/models）",
    )
    ap.add_argument(
        "--name",
        type=str,
        default="yolo_nangang",
        help="本次 run 名称",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=100, help="早停耐心")
    ap.add_argument(
        "--cache",
        type=str,
        default="ram",
        choices=["", "ram", "disk"],
        help="缓存图片加速读盘：ram / disk / 空=关闭（默认 ram）",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="从 project/name 下最近一次未完成训练继续",
    )
    ap.add_argument(
        "--export-onnx",
        action="store_true",
        help="训练结束后导出 ONNX",
    )
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("请先安装: pip install ultralytics", file=sys.stderr)
        return 2

    data_yaml = resolve_path(args.data)
    if not data_yaml.is_file():
        print(f"找不到 data.yaml: {data_yaml}", file=sys.stderr)
        return 1

    n_train = count_split(data_yaml, "train")
    n_val = count_split(data_yaml, "val")
    if n_train < 10:
        print(
            f"警告: train 仅 {n_train} 张，建议先加大 auto_annotate 产量 "
            f"(--full-maps / --crops-per-map)",
            file=sys.stderr,
        )

    project = resolve_path(args.project or (mxd_tools_root() / "models"))
    project.mkdir(parents=True, exist_ok=True)
    run_dir = project / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    abs_yaml = prepare_data_yaml(data_yaml, run_dir)

    device = args.device
    if not device:
        try:
            import torch

            if torch.cuda.is_available():
                device = "0"
                print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print(
                    "未检测到 CUDA。若本机有 NVIDIA 显卡，请改装 GPU 版 PyTorch：\n"
                    "  pip install --upgrade torch torchvision "
                    "--index-url https://download.pytorch.org/whl/cu126"
                )
        except Exception:
            device = "cpu"

    print(f"data:   {data_yaml}")
    print(f"abs:    {abs_yaml}")
    print(f"train/val images: {n_train} / {n_val}")
    print(f"model:  {args.model}")
    print(
        f"device: {device} | epochs={args.epochs} imgsz={args.imgsz} "
        f"batch={args.batch} workers={args.workers} cache={args.cache or 'off'}"
    )
    print(f"out:    {run_dir}")

    train_kw: dict = dict(
        data=str(abs_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(project),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        patience=args.patience,
        resume=args.resume,
        plots=True,
        save=True,
        amp=True,
    )
    if args.cache:
        train_kw["cache"] = args.cache

    model = YOLO(args.model)
    results = model.train(**train_kw)

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if best.is_file():
        # 方便拷贝的别名
        alias = project / f"{args.name}_best.pt"
        shutil.copy2(best, alias)
        print(f"best -> {best}")
        print(f"copy -> {alias}")
    elif last.is_file():
        print(f"last -> {last}")

    if args.export_onnx and best.is_file():
        print("导出 ONNX ...")
        YOLO(str(best)).export(format="onnx", imgsz=args.imgsz)

    print("训练完成")
    # results 可能很大，只提示路径
    _ = results
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
