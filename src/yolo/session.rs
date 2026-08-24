use std::path::Path;

use anyhow::{bail, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use crate::yolo::postprocess::decode_yolo_output_flat;
use crate::yolo::preprocess::{letterbox_rgb_into, LetterboxBuffers};
use crate::yolo::{Detection, YoloDevice};

pub struct YoloDetector {
    session: Session,
    imgsz: u32,
    conf: f32,
    iou: f32,
    pub device_label: String,
    input_buf: Vec<f32>,
    letterbox_bufs: LetterboxBuffers,
}

impl YoloDetector {
    pub fn load(onnx: &Path, device: YoloDevice) -> Result<Self> {
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
        let mk_builder = || {
            Session::builder()
                .map_err(|e| anyhow::anyhow!("创建 ORT SessionBuilder 失败: {e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("设置图优化失败: {e}"))?
                .with_intra_threads(4)
                .map_err(|e| anyhow::anyhow!("设置 intra_threads 失败: {e}"))
        };
        let session = match device {
            YoloDevice::Cuda(id) => {
                #[cfg(feature = "cuda")]
                {
                    use ort::ep::CUDA;
                    let try_cuda = mk_builder()?
                        .with_execution_providers([CUDA::default().with_device_id(id as i32).build()]);
                    match try_cuda {
                        Ok(mut b) => match b.commit_from_file(onnx) {
                            Ok(s) => {
                                device_label = format!("cuda:{id}");
                                eprintln!("YOLO: 使用 CUDA EP (device={id})");
                                s
                            }
                            Err(e) => {
                                eprintln!("YOLO: CUDA session 失败，回退 CPU: {e}");
                                device_label = "cpu(fallback)".to_string();
                                mk_builder()?
                                    .commit_from_file(onnx)
                                    .with_context(|| {
                                        format!("加载 ONNX 失败: {}", onnx.display())
                                    })?
                            }
                        },
                        Err(e) => {
                            eprintln!("YOLO: 注册 CUDA EP 失败，回退 CPU: {e}");
                            device_label = "cpu(fallback)".to_string();
                            mk_builder()?.commit_from_file(onnx).with_context(|| {
                                format!("加载 ONNX 失败: {}", onnx.display())
                            })?
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
                    mk_builder()?.commit_from_file(onnx).with_context(|| {
                        format!("加载 ONNX 失败: {}", onnx.display())
                    })?
                }
            }
            YoloDevice::Cpu => mk_builder()?
                .commit_from_file(onnx)
                .with_context(|| format!("加载 ONNX 失败: {}", onnx.display()))?,
        };

        let plane = (imgsz as usize) * (imgsz as usize) * 3;
        Ok(Self {
            session,
            imgsz,
            conf,
            iou,
            device_label,
            input_buf: Vec::with_capacity(plane),
            letterbox_bufs: LetterboxBuffers::new().context("初始化 letterbox 缓冲失败")?,
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

        let meta = letterbox_rgb_into(
            rgb,
            w,
            h,
            self.imgsz,
            &mut self.letterbox_bufs,
            &mut self.input_buf,
        )?;

        let shape = [1_i64, 3, self.imgsz as i64, self.imgsz as i64];
        let input_tensor = TensorRef::from_array_view((shape, self.input_buf.as_slice()))
            .context("构造输入张量失败")?;
        let outputs = self
            .session
            .run(ort::inputs![input_tensor])
            .context("ORT 推理失败")?;

        let (_name, value) = outputs.iter().next().context("ORT 无输出")?;
        let (out_shape, out_data) = value
            .try_extract_tensor::<f32>()
            .context("解析输出张量失败")?;

        Ok(decode_yolo_output_flat(
            &out_shape,
            out_data,
            &meta,
            self.conf,
            self.iou,
        ))
    }
}
