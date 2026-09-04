//! YOLO11 ONNX 推理（ort / ONNX Runtime）。
//!
//! ONNX YOLO 检测（CPU 默认可用；可选 `--features cuda` 用于工具链）。

mod labels;
mod postprocess;
mod preprocess;
mod session;

pub use labels::{
    class_name, hp_ratio_from_class_id, hp_ratio_from_label, CLASS_NAMES, HP_BAR_CLASS_FIRST,
    HP_BAR_CLASS_LAST,
};
pub use session::{YoloDetector, EMBEDDED_YOLO_ONNX, EMBEDDED_YOLO_ONNX_NAME};

#[derive(Debug, Clone, Copy)]
pub enum YoloDevice {
    Cpu,
    /// NVIDIA 设备号（通常 0）
    Cuda(u32),
}

impl YoloDevice {
    pub fn parse(s: &str) -> Self {
        let t = s.trim().to_ascii_lowercase();
        if t == "cpu" {
            return Self::Cpu;
        }
        if let Some(rest) = t.strip_prefix("cuda") {
            let id = rest
                .trim_start_matches(':')
                .trim()
                .parse::<u32>()
                .unwrap_or(0);
            return Self::Cuda(id);
        }
        if t == "0" || t == "gpu" {
            return Self::Cuda(0);
        }
        Self::Cpu
    }
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub class_id: usize,
    pub label: &'static str,
    pub conf: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct LetterboxMeta {
    pub gain: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub orig_w: u32,
    pub orig_h: u32,
}

/// 在检测列表中取置信度最高的血条框，返回 (比例, conf)。
pub fn best_hp_from_detections(dets: &[Detection]) -> Option<(f32, f32)> {
    let mut best: Option<(f32, f32)> = None;
    for d in dets {
        let Some(ratio) =
            hp_ratio_from_class_id(d.class_id).or_else(|| hp_ratio_from_label(d.label))
        else {
            continue;
        };
        if best.map(|(_, c)| d.conf > c).unwrap_or(true) {
            best = Some((ratio, d.conf));
        }
    }
    best
}
