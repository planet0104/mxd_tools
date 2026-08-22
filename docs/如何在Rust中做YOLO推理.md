# 如何在 Rust 中做 YOLO 推理

训练仍用 Python（Ultralytics）；**游戏侧推理用 Rust + ONNX Runtime（[`ort`](https://crates.io/crates/ort)）**，运行时不依赖 Python。

## 部署目标：单个 exe

CPU 推理路径下，**OpenCV 与 ONNX Runtime 均静态链进 exe**。已验证：去掉同目录的 `onnxruntime.dll` 后仍可推理。

发布时通常需要：

| 文件 | 说明 |
|------|------|
| `yolo_predict.exe`（或主程序） | 已内嵌 OpenCV + ORT |
| `models/*.onnx` | 模型权重（不编进二进制） |
| VC 运行库 | `/MD` 依赖 `VCRUNTIME140.dll` 等；目标机装 [VC++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) 即可 |
| `DirectML.dll`（偶发） | pyke ORT 预编译库会链到 DirectML；Win10+ 系统目录常已有，若启动报缺再拷一份到 exe 旁 |

**不需要**再带 `onnxruntime.dll` / OpenCV DLL。

技术要点：

- OpenCV：`vcpkg` triplet **`x64-windows-static-md`**（静态库 + `/MD`）
- `ort`：`download-binaries` 的 **静态 `onnxruntime.lib`** 链进 exe（**禁用** `load-dynamic`）
- **不要**开 `rustflags = +crt-static`（与 pyke ORT 冲突）
- 环境：`powershell -File scripts/setup_opencv_static.ps1`

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
# LLVM 在 PATH（OpenCV 绑定生成）
$env:PATH = "C:\Program Files\LLVM\bin;" + $env:PATH

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
  --source screen_caps/彩虹岛-南港西郊平原 `
  --onnx models/yolo_nangang_e1000.onnx `
  --pt models/yolo_nangang_e1000_best.pt `
  --conf 0.25 --iou 0.7 --device cpu --py-device cpu
```

会先跑 `scripts/dump_yolo_dets.py`（Ultralytics `.pt`），再与 Rust ONNX 框做 IoU 匹配；默认要求 recall/precision ≥ 0.85 且 mean_iou ≥ 0.70。

## 4. 库 API

```rust
use mxd_tools::yolo::{YoloDetector, YoloDevice};

let mut det = YoloDetector::load(
    Path::new("models/yolo_nangang_e1000.onnx"),
    YoloDevice::Cpu, // 或 YoloDevice::Cuda(0)
)?;
let boxes = det.detect_rgb8(w, h, &rgb)?;
```

模块：`src/yolo/`（letterbox → ORT → 解码 + NMS）。类别与 `dataset/.../generated/yolo/data.yaml` 19 类一致。

## 5. Cargo 依赖

- `ort`：静态链接（`download-binaries`，**无** `load-dynamic`）+ 可选 feature `cuda`
- `ndarray`、`image`

`.cargo/config.toml` 指向 `x64-windows-static-md`，`OPENCV_MSVC_CRT=dynamic`。

## 常见问题

| 现象 | 处理 |
|------|------|
| 链接 `ort` 报 `__imp_*` / CRT 冲突 | 确认未启用 `+crt-static`，OpenCV 为 `static-md` |
| 找不到 OpenCV | `scripts/setup_opencv_static.ps1` |
| OpenCV 构建找不到 clang | 安装 LLVM，`PATH` 含 `C:\Program Files\LLVM\bin` |
| `--device cuda` 仍走 CPU | 未加 `--features cuda`，或本机 CUDA 不满足 ort 要求 |
| 想用旧的 `/MT` 全静态 CRT | 需自行用 `--enable_msvc_static_runtime` 编译 ORT；预编译 pyke 包不支持 |
