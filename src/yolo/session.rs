use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

use crate::yolo::postprocess::decode_yolo_output;
use crate::yolo::{Detection, LetterboxMeta, YoloDevice};

pub struct YoloDetector {
    session: Session,
    imgsz: u32,
    conf: f32,
    iou: f32,
    pub device_label: String,
}

impl YoloDetector {
    pub fn load(onnx: &Path, device: YoloDevice) -> Result<Self> {
        // iou 默认 0.7，与 Ultralytics predict 一致
        Self::load_with_thresholds(onnx, device, 0.25, 0.7, 640)
    }

    pub fn load_with_thresholds(
        onnx: &Path,
        device: YoloDevice,
        conf: f32,
        iou: f32,
        imgsz: u32,
    ) -> Result<Self> {
        if !onnx.is_file() {
            bail!("找不到 ONNX: {}", onnx.display());
        }

        let mut device_label = "cpu".to_string();

        let session = match device {
            YoloDevice::Cuda(id) => {
                #[cfg(feature = "cuda")]
                {
                    use ort::ep::CUDA;
                    let try_cuda = Session::builder()
                        .context("创建 ORT SessionBuilder 失败")?
                        .with_execution_providers([
                            CUDA::default().with_device_id(id as i32).build()
                        ]);
                    match try_cuda {
                        Ok(b) => match b.commit_from_file(onnx) {
                            Ok(s) => {
                                device_label = format!("cuda:{id}");
                                eprintln!("YOLO: 使用 CUDA EP (device={id})");
                                s
                            }
                            Err(e) => {
                                eprintln!("YOLO: CUDA session 失败，回退 CPU: {e}");
                                device_label = "cpu(fallback)".to_string();
                                Session::builder()
                                    .context("创建 ORT SessionBuilder 失败")?
                                    .commit_from_file(onnx)
                                    .with_context(|| {
                                        format!("加载 ONNX 失败: {}", onnx.display())
                                    })?
                            }
                        },
                        Err(e) => {
                            eprintln!("YOLO: 注册 CUDA EP 失败，回退 CPU: {e}");
                            device_label = "cpu(fallback)".to_string();
                            Session::builder()
                                .context("创建 ORT SessionBuilder 失败")?
                                .commit_from_file(onnx)
                                .with_context(|| format!("加载 ONNX 失败: {}", onnx.display()))?
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = id;
                    eprintln!(
                        "YOLO: 未启用 cargo feature `cuda`，使用 CPU。\
                         需要 GPU: cargo build --release --features cuda --bin yolo_predict"
                    );
                    device_label = "cpu(no-cuda-feature)".to_string();
                    Session::builder()
                        .context("创建 ORT SessionBuilder 失败")?
                        .commit_from_file(onnx)
                        .with_context(|| format!("加载 ONNX 失败: {}", onnx.display()))?
                }
            }
            YoloDevice::Cpu => Session::builder()
                .context("创建 ORT SessionBuilder 失败")?
                .commit_from_file(onnx)
                .with_context(|| format!("加载 ONNX 失败: {}", onnx.display()))?,
        };

        Ok(Self {
            session,
            imgsz,
            conf,
            iou,
            device_label,
        })
    }

    pub fn set_thresholds(&mut self, conf: f32, iou: f32) {
        self.conf = conf;
        self.iou = iou;
    }

    /// `rgb` 为 packed RGB8，长度 = w*h*3。
    pub fn detect_rgb8(&mut self, w: u32, h: u32, rgb: &[u8]) -> Result<Vec<Detection>> {
        if rgb.len() != (w as usize) * (h as usize) * 3 {
            bail!(
                "RGB 缓冲长度不符: got {} expect {}",
                rgb.len(),
                w as usize * h as usize * 3
            );
        }
        let (input, meta) = letterbox_rgb(rgb, w, h, self.imgsz);
        let shape = input.shape().to_vec();
        let flat = input.into_raw_vec_and_offset().0;

        let outputs = {
            let input_tensor =
                TensorRef::from_array_view((shape, flat.as_slice())).context("构造输入张量失败")?;
            self.session
                .run(ort::inputs![input_tensor])
                .context("ORT 推理失败")?
        };

        let (_name, value) = outputs.iter().next().context("ORT 无输出")?;
        let (out_shape, out_data) = value
            .try_extract_tensor::<f32>()
            .context("解析输出张量失败")?;
        let arr = ndarray::ArrayD::from_shape_vec(
            out_shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
            out_data.to_vec(),
        )
        .context("输出 shape 与数据不匹配")?;

        Ok(decode_yolo_output(&arr, &meta, self.conf, self.iou))
    }
}

fn letterbox_rgb(rgb: &[u8], w: u32, h: u32, imgsz: u32) -> (Array4<f32>, LetterboxMeta) {
    // 对齐 Ultralytics LetterBox(auto=False, center=True, scaleup=True)：
    // ratio = min(new/old)；new_unpad = round；pad = round(d - 0.1)；双线性缩放。
    let wf = w as f32;
    let hf = h as f32;
    let size = imgsz as f32;
    let gain = (size / hf).min(size / wf);
    let nw = (wf * gain).round().max(1.0) as u32;
    let nh = (hf * gain).round().max(1.0) as u32;
    let dw = (size - nw as f32) * 0.5;
    let dh = (size - nh as f32) * 0.5;
    let pad_x = (dw - 0.1).round();
    let pad_y = (dh - 0.1).round();
    let right = ((dw + 0.1).round() as i32).max(0) as u32;
    let bottom = ((dh + 0.1).round() as i32).max(0) as u32;
    let left = pad_x.max(0.0) as u32;
    let top = pad_y.max(0.0) as u32;

    let resized = resize_rgb_bilinear(rgb, w, h, nw, nh);
    let mut canvas = vec![114u8; (imgsz as usize) * (imgsz as usize) * 3];
    for y in 0..nh {
        for x in 0..nw {
            let dx = left + x;
            let dy = top + y;
            if dx >= imgsz || dy >= imgsz {
                continue;
            }
            let si = ((y * nw + x) * 3) as usize;
            let di = ((dy * imgsz + dx) * 3) as usize;
            canvas[di] = resized[si];
            canvas[di + 1] = resized[si + 1];
            canvas[di + 2] = resized[si + 2];
        }
    }
    // 右侧/底边若因 round 对不齐，已用 114 填充；校验尺寸
    let _ = (right, bottom);

    let mut arr = Array4::<f32>::zeros((1, 3, imgsz as usize, imgsz as usize));
    for y in 0..imgsz as usize {
        for x in 0..imgsz as usize {
            let i = (y * imgsz as usize + x) * 3;
            arr[[0, 0, y, x]] = canvas[i] as f32 / 255.0;
            arr[[0, 1, y, x]] = canvas[i + 1] as f32 / 255.0;
            arr[[0, 2, y, x]] = canvas[i + 2] as f32 / 255.0;
        }
    }

    let meta = LetterboxMeta {
        gain,
        pad_x,
        pad_y,
        orig_w: w,
        orig_h: h,
    };
    (arr, meta)
}

fn resize_rgb_bilinear(rgb: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    if w == nw && h == nh {
        return rgb.to_vec();
    }
    let mut out = vec![0u8; (nw as usize) * (nh as usize) * 3];
    let x_scale = w as f32 / nw as f32;
    let y_scale = h as f32 / nh as f32;
    for y in 0..nh {
        let fy = (y as f32 + 0.5) * y_scale - 0.5;
        let y0 = fy.floor().max(0.0) as i32;
        let y1 = (y0 + 1).min(h as i32 - 1);
        let wy = fy - y0 as f32;
        let y0 = y0.clamp(0, h as i32 - 1) as u32;
        let y1 = y1 as u32;
        for x in 0..nw {
            let fx = (x as f32 + 0.5) * x_scale - 0.5;
            let x0 = fx.floor().max(0.0) as i32;
            let x1 = (x0 + 1).min(w as i32 - 1);
            let wx = fx - x0 as f32;
            let x0 = x0.clamp(0, w as i32 - 1) as u32;
            let x1 = x1 as u32;
            let i00 = ((y0 * w + x0) * 3) as usize;
            let i01 = ((y0 * w + x1) * 3) as usize;
            let i10 = ((y1 * w + x0) * 3) as usize;
            let i11 = ((y1 * w + x1) * 3) as usize;
            let o = ((y * nw + x) * 3) as usize;
            for c in 0..3 {
                let v = (1.0 - wy) * ((1.0 - wx) * rgb[i00 + c] as f32 + wx * rgb[i01 + c] as f32)
                    + wy * ((1.0 - wx) * rgb[i10 + c] as f32 + wx * rgb[i11 + c] as f32);
                out[o + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}
