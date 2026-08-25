#!/usr/bin/env python3
"""对比 Rust ONNX / Python Ultralytics 推理速度（CPU / GPU）。"""

from __future__ import annotations

import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "screen_caps/彩虹岛-南港西郊平原"
ONNX = ROOT / "models/yolo_nangang_e2000_best.onnx"
PT = ROOT / "models/yolo_nangang_e2000_best.pt"
RUST_CPU = ROOT / "target/release/yolo_infer.exe"
RUST_CUDA = ROOT / "target/release/yolo_infer.exe"  # same binary, cuda feature at build time
WARMUP = 2
ROUNDS = 3


def list_images() -> list[Path]:
    imgs = sorted(
        p
        for p in SOURCE.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if not imgs:
        raise FileNotFoundError(f"未找到图片: {SOURCE}")
    return imgs


def bench_python(device: str) -> dict:
    images = list_images()
    model = YOLO(str(PT))
    names = model.names

    # warmup
    for p in images[:WARMUP]:
        model.predict(str(p), device=device, imgsz=640, conf=0.25, iou=0.7, verbose=False)

    per_image: list[float] = []
    for _ in range(ROUNDS):
        for p in images:
            t0 = time.perf_counter()
            model.predict(str(p), device=device, imgsz=640, conf=0.25, iou=0.7, verbose=False)
            per_image.append((time.perf_counter() - t0) * 1000.0)

    return {
        "engine": "Python Ultralytics (.pt)",
        "device": device,
        "images": len(images),
        "rounds": ROUNDS,
        "avg_ms": statistics.mean(per_image),
        "min_ms": min(per_image),
        "max_ms": max(per_image),
        "classes": len(names),
    }


def bench_rust(exe: Path, device: str, cuda_build: bool) -> dict:
    if not exe.is_file():
        raise FileNotFoundError(f"缺少可执行文件: {exe}")

    images = list_images()
    cmd = [
        str(exe),
        "--model",
        str(ONNX),
        "--source",
        str(SOURCE),
        "--device",
        device,
        "--bench",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Rust 推理失败 ({device}):\n{proc.stderr}\n{proc.stdout}")

    ms_vals = [float(m) for m in re.findall(r"^BENCH .+? ([0-9.]+) ms", proc.stdout, re.M)]
    avg_line = re.search(r"infer_ms_avg: ([0-9.]+)", proc.stdout)
    if not ms_vals or not avg_line:
        raise RuntimeError(f"未解析到 BENCH 输出:\n{proc.stdout}")

    label = "Rust ONNX (cuda build)" if cuda_build else "Rust ONNX (cpu build)"
    return {
        "engine": label,
        "device": device,
        "images": len(images),
        "rounds": 1,
        "avg_ms": float(avg_line.group(1)),
        "min_ms": min(ms_vals),
        "max_ms": max(ms_vals),
        "classes": 21,
    }


def print_row(r: dict) -> None:
    print(
        f"{r['engine']:<28} {r['device']:<8} "
        f"avg={r['avg_ms']:7.1f} ms  min={r['min_ms']:7.1f}  max={r['max_ms']:7.1f}  "
        f"imgs={r['images']} rounds={r['rounds']} classes={r['classes']}"
    )


def main() -> int:
    print(f"数据源: {SOURCE}")
    print(f"图片数: {len(list_images())}")
    print(f"模型: {PT.name} / {ONNX.name}")
    print(f"Python 每项跑 {ROUNDS} 轮取平均；Rust 单进程跑 {len(list_images())} 张（含读图+推理）\n")

    rows: list[dict] = []

    print("== Python ==")
    for dev in ("cpu", "0"):
        label = "cuda:0" if dev == "0" else "cpu"
        r = bench_python(dev)
        r["device"] = label
        rows.append(r)
        print_row(r)

    print("\n== Rust ==")
    r_cpu = bench_rust(RUST_CPU, "cpu", cuda_build=False)
    rows.append(r_cpu)
    print_row(r_cpu)

    try:
        r_cuda = bench_rust(RUST_CUDA, "cuda:0", cuda_build=True)
        rows.append(r_cuda)
        print_row(r_cuda)
    except Exception as e:
        print(f"Rust CUDA 跳过: {e}")

    print("\n===== 汇总（平均每张图推理耗时）=====")
    for r in rows:
        print(f"  {r['engine']} [{r['device']}]: {r['avg_ms']:.1f} ms/张")

    return 0


if __name__ == "__main__":
    sys.exit(main())
