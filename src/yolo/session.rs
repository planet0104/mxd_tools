use std::path::Path;

use anyhow::{bail, Context, Result};
use ort::session::Session;
use ort::value::TensorRef;

use crate::ort_util::{build_session, build_session_from_memory, OrtDevice};
use crate::yolo::postprocess::{decode_yolo_batch_output, decode_yolo_output_flat};
use crate::yolo::preprocess::{letterbox_rgb_into, LetterboxBuffers};
use crate::yolo::{Detection, LetterboxMeta, YoloDevice};

/// 默认 YOLO 权重（编译期嵌入，发布 exe 无需旁路 `.onnx`）。
pub const EMBEDDED_YOLO_ONNX: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/onnx/best.onnx"
));

pub const EMBEDDED_YOLO_ONNX_NAME: &str = "best.onnx (embedded)";

fn yolo_to_ort(device: YoloDevice) -> OrtDevice {
    match device {
        YoloDevice::Cpu => OrtDevice::Cpu,
        YoloDevice::Cuda(id) => OrtDevice::Cuda(id),
    }
}

pub struct YoloDetector {
    session: Session,
    imgsz: u32,
    conf: f32,
    iou: f32,
    pub device_label: String,
    input_buf: Vec<f32>,
    letterbox_bufs: LetterboxBuffers,
    /// `None` = 未探测；固定 batch=1 模型为 `false`。
    batch_tensor_ok: Option<bool>,
}

impl YoloDetector {
    pub fn load(onnx: &Path, device: YoloDevice) -> Result<Self> {
        Self::load_with_thresholds(onnx, device, 0.25, 0.7, 640)
    }

    /// 使用编译期嵌入的默认 ONNX。
    pub fn load_embedded(device: YoloDevice) -> Result<Self> {
        Self::load_from_bytes_with_thresholds(EMBEDDED_YOLO_ONNX, device, 0.25, 0.7, 640)
    }

    pub fn load_from_bytes(onnx: &[u8], device: YoloDevice) -> Result<Self> {
        Self::load_from_bytes_with_thresholds(onnx, device, 0.25, 0.7, 640)
    }

    pub fn load_with_thresholds(
        onnx: &Path,
        device: YoloDevice,
        conf: f32,
        iou: f32,
        imgsz: u32,
    ) -> Result<Self> {
        let (session, device_label) = build_session(onnx, yolo_to_ort(device), 4)?;
        Self::from_session(session, device_label, conf, iou, imgsz)
    }

    pub fn load_from_bytes_with_thresholds(
        onnx: &[u8],
        device: YoloDevice,
        conf: f32,
        iou: f32,
        imgsz: u32,
    ) -> Result<Self> {
        let (session, device_label) = build_session_from_memory(onnx, yolo_to_ort(device), 4)?;
        Self::from_session(session, device_label, conf, iou, imgsz)
    }

    fn from_session(
        session: Session,
        device_label: String,
        conf: f32,
        iou: f32,
        imgsz: u32,
    ) -> Result<Self> {
        eprintln!("YOLO: {device_label}");
        let plane = (imgsz as usize) * (imgsz as usize) * 3;
        Ok(Self {
            session,
            imgsz,
            conf,
            iou,
            device_label,
            input_buf: Vec::with_capacity(plane),
            letterbox_bufs: LetterboxBuffers::new(),
            batch_tensor_ok: None,
        })
    }

    pub fn set_thresholds(&mut self, conf: f32, iou: f32) {
        self.conf = conf;
        self.iou = iou;
    }

    /// `rgb` 为 packed RGB8，长度 = w*h*3。
    pub fn detect_rgb8(&mut self, w: u32, h: u32, rgb: &[u8]) -> Result<Vec<Detection>> {
        let mut metas = Vec::with_capacity(1);
        self.preprocess_one(w, h, rgb, &mut metas)?;
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
            &out_shape, out_data, &metas[0], self.conf, self.iou,
        ))
    }

    /// 批量 YOLO 推理：`frames` 为 `(w, h, rgb)` 列表。
    /// 若 ONNX 固定 batch=1，自动回退为同 session 逐帧 GPU 推理。
    pub fn detect_rgb8_batch(
        &mut self,
        frames: &[(u32, u32, &[u8])],
    ) -> Result<Vec<Vec<Detection>>> {
        if frames.is_empty() {
            return Ok(vec![]);
        }
        if self.batch_tensor_ok != Some(false) && frames.len() > 1 {
            match self.detect_rgb8_batch_tensor(frames) {
                Ok(v) => {
                    self.batch_tensor_ok = Some(true);
                    return Ok(v);
                }
                Err(e) => {
                    self.batch_tensor_ok = Some(false);
                    eprintln!("YOLO: batch 维度不可用 ({e})，后续逐帧推理");
                }
            }
        }
        let mut out = Vec::with_capacity(frames.len());
        for &(w, h, rgb) in frames {
            out.push(self.detect_rgb8(w, h, rgb)?);
        }
        Ok(out)
    }

    fn detect_rgb8_batch_tensor(
        &mut self,
        frames: &[(u32, u32, &[u8])],
    ) -> Result<Vec<Vec<Detection>>> {
        let n = frames.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let plane = (self.imgsz as usize) * (self.imgsz as usize) * 3;
        self.input_buf.resize(n * plane, 0.0);
        let mut metas = Vec::with_capacity(n);
        for (i, (w, h, rgb)) in frames.iter().enumerate() {
            if rgb.len() != (*w as usize) * (*h as usize) * 3 {
                bail!(
                    "batch[{i}] RGB 长度不符: got {} expect {}",
                    rgb.len(),
                    *w as usize * *h as usize * 3
                );
            }
            let offset = i * plane;
            let mut slot = Vec::with_capacity(plane);
            let meta =
                letterbox_rgb_into(rgb, *w, *h, self.imgsz, &mut self.letterbox_bufs, &mut slot)?;
            self.input_buf[offset..offset + plane].copy_from_slice(&slot);
            metas.push(meta);
        }
        let shape = [n as i64, 3, self.imgsz as i64, self.imgsz as i64];
        let input_tensor = TensorRef::from_array_view((shape, self.input_buf.as_slice()))
            .context("构造 batch 输入张量失败")?;
        let outputs = self
            .session
            .run(ort::inputs![input_tensor])
            .context("ORT batch 推理失败")?;
        let (_name, value) = outputs.iter().next().context("ORT 无输出")?;
        let (out_shape, out_data) = value
            .try_extract_tensor::<f32>()
            .context("解析 batch 输出张量失败")?;
        Ok(decode_yolo_batch_output(
            &out_shape, out_data, &metas, self.conf, self.iou,
        ))
    }

    fn preprocess_one(
        &mut self,
        w: u32,
        h: u32,
        rgb: &[u8],
        metas: &mut Vec<LetterboxMeta>,
    ) -> Result<()> {
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
        metas.push(meta);
        Ok(())
    }
}
