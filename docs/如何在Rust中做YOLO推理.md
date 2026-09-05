# 如何在 Rust 中做 YOLO 推理

训练仍用 Python（Ultralytics）；**游戏侧推理用 Rust + ONNX Runtime（[`ort`](https://crates.io/crates/ort)）**，运行时不依赖 Python。Letterbox 预处理为纯 Rust（`image` crate），无 OpenCV。

## 部署目标：单个 exe

CPU 推理路径下，**ONNX Runtime、默认 YOLO 权重、默认地图 50001 platforms** 均编进主程序。已验证：去掉旁路 `onnxruntime.dll` / `.onnx` 后仍可推理与 Live Nav。

发布时通常需要：

| 文件 | 说明 |
|------|------|
| `mxd_tools.exe`（或 `game_preview.exe`） | 已内嵌 ORT + 默认 YOLO + 默认地图 |
| VC 运行库 | `/MD` 依赖 `VCRUNTIME140.dll` 等；目标机装 [VC++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) 即可 |
| `DirectML.dll`（偶发） | pyke ORT 预编译库会链到 DirectML；Win10+ 系统目录常已有，若启动报缺再拷一份到 exe 旁 |

**不需要**再带 `onnxruntime.dll` / `.onnx` 模型文件。

换模型时：重新导出 ONNX 覆盖 `onnx/best.onnx` 后重新 `cargo build --release`；或运行时传 `--model path/to.onnx`（工具 bin）覆盖嵌入权重。

技术要点：

- `ort`：`download-binaries` 的 **静态 `onnxruntime.lib`** 链进 exe（**禁用** `load-dynamic`）
- YOLO：`include_bytes!` → `commit_from_memory`
- Letterbox：`image` crate（`FilterType::Triangle` ≈ 双线性）
- **不要**开 `rustflags = +crt-static`（与 pyke ORT 冲突）

可选 `--features cuda` 时往往还要 CUDA provider / 本机 CUDA，**不再保证单文件**；默认 CPU 按上表部署。

## 1. 导出 ONNX

```powershell
python scripts/train_yolo.py ... --export-onnx
# 或
python -c "from ultralytics import YOLO; YOLO('models/yolo_nangang_e1000_best.pt').export(format='onnx', imgsz=640, simplify=True, opset=12)"
```

约定路径：`models/yolo_nangang_e1000.onnx`（imgsz=640，输出约 `[1, 23, 8400]` = 4 + 19 类）。

## 2. 编译

```powershell
cargo build --release --bin yolo_predict
```

产物：`target/release/yolo_predict.exe`（ORT 已静态链接）。

可选 CUDA（失败回退 CPU）：

```powershell
cargo build --release --features cuda --bin yolo_predict
```

## 与 Python 对齐验框

```powershell
cargo run --release --bin yolo_compare -- `
  --source screen_caps/nangang_50001 `
  --onnx models/yolo_nangang_e1000.onnx `
  --pt models/yolo_nangang_e1000_best.pt `
  --conf 0.25 --iou 0.7 --device cpu --py-device cpu
```

会先跑 `scripts/dump_yolo_dets.py`（Ultralytics `.pt`），再与 Rust ONNX 框做 IoU 匹配；默认要求 recall/precision ≥ 0.85 且 mean_iou ≥ 0.70。

## 4. 库 API

```rust
use mxd_tools::yolo::{YoloDetector, YoloDevice};

// 嵌入默认模型（单 exe）
let mut det = YoloDetector::load_embedded(YoloDevice::Cpu)?;
// 或外置文件（调试换权重）
// let mut det = YoloDetector::load(Path::new("onnx/foo.onnx"), YoloDevice::Cpu)?;
let boxes = det.detect_rgb8(w, h, &rgb)?;
```

模块：`src/yolo/`（letterbox → ORT → 解码 + NMS）。默认嵌入 `onnx/best.onnx`。

## 5. Cargo 依赖

- `ort`：静态链接（`download-binaries`，**无** `load-dynamic`）+ 可选 feature `cuda`
- `ndarray`、`image`（letterbox / 读写图）

## 常见问题

| 现象 | 处理 |
|------|------|
| 链接 `ort` 报 `__imp_*` / CRT 冲突 | 确认未启用 `+crt-static` |
| `--device cuda` 仍走 CPU | 未加 `--features cuda`，或本机 CUDA 不满足 ort 要求 |
| 想用旧的 `/MT` 全静态 CRT | 需自行用 `--enable_msvc_static_runtime` 编译 ORT；预编译 pyke 包不支持 |
