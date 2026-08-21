# 训练 YOLO 检测模型

依赖合成数据集（见 `如何自动贴图标注数据集.md`）。

## 安装

```powershell
cd mxd_tools
pip install ultralytics

# 重要：默认 pip 的 torch 常是 CPU 版。有 NVIDIA 显卡请改装 CUDA 版：
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

本机已验证：RTX 4060 Laptop → `torch 2.4.1+cu121`，`cuda True`。

## 训练

```powershell
python .\scripts\train_yolo.py `
  --data dataset/彩虹岛-南港西郊平原/generated/yolo/data.yaml
```

常用参数：

```powershell
python .\scripts\train_yolo.py `
  --data dataset/彩虹岛-南港西郊平原/generated/yolo/data.yaml `
  --model yolo11n.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 8 `
  --device 0 `
  --name yolo_nangang
```

| 参数 | 说明 |
|------|------|
| `--model` | `yolo11n.pt`（快）/ `yolo11s.pt`（更准）/ `yolov8n.pt` |
| `--device` | `0`=第一块 GPU，`cpu`=仅 CPU；默认自动选 GPU |
| `--batch` | 4060 笔记本可先试 `8`～`16`；OOM 再降 |
| `--export-onnx` | 训完额外导出 ONNX |

GPU 版 PyTorch（本机已验证 RTX 4060）：

```powershell
# 官方 CUDA 12.1 wheel（约 2.3GB，可用国内镜像）
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 输出

默认写到 `mxd_tools/models/<name>/`：

```text
models/yolo_nangang/
  weights/best.pt      # 验证集最优
  weights/last.pt
  results.png          # 曲线
  ...
models/yolo_nangang_best.pt   # 方便拷贝的别名
```

## 简单推理试一下

小字号 + 半透明标签底（推荐看误检）：

```powershell
python .\scripts\predict_yolo.py `
  --model models/yolo_nangang_e5000_best.pt `
  --source screen_caps/彩虹岛-南港西郊平原/ScreenShot_2026-08-20_095130_246.png `
  --font-size 13 --bg-alpha 0.4
```

也可用 ultralytics 自带（字偏大、底不透明）：

```powershell
yolo predict model=models/yolo_nangang_best.pt source=dataset/彩虹岛-南港西郊平原/generated/yolo/images/val imgsz=640
```

或在 Python 里：

```python
from ultralytics import YOLO
model = YOLO("models/yolo_nangang_best.pt")
model.predict("path/to/window.png", save=True)
```
