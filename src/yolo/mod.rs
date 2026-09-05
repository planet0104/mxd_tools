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

/// 血条检测结果（含类别，便于对照 YOLO 是否认错档）。
#[derive(Debug, Clone)]
pub struct HpBarHit {
    pub ratio: f32,
    pub conf: f32,
    pub class_id: usize,
    pub label: String,
}

/// 在检测列表中取置信度最高的血条框。
pub fn best_hp_from_detections(dets: &[Detection]) -> Option<HpBarHit> {
    let mut best: Option<HpBarHit> = None;
    for d in dets {
        let Some(ratio) =
            hp_ratio_from_class_id(d.class_id).or_else(|| hp_ratio_from_label(d.label))
        else {
            continue;
        };
        if best.as_ref().map(|b| d.conf > b.conf).unwrap_or(true) {
            best = Some(HpBarHit {
                ratio,
                conf: d.conf,
                class_id: d.class_id,
                label: d.label.to_string(),
            });
        }
    }
    best
}

/// 列出全部血条检测（按 conf 降序），用于调试日志。
pub fn list_hp_detections(dets: &[Detection]) -> Vec<HpBarHit> {
    let mut out = Vec::new();
    for d in dets {
        let Some(ratio) =
            hp_ratio_from_class_id(d.class_id).or_else(|| hp_ratio_from_label(d.label))
        else {
            continue;
        };
        out.push(HpBarHit {
            ratio,
            conf: d.conf,
            class_id: d.class_id,
            label: d.label.to_string(),
        });
    }
    out.sort_by(|a, b| {
        b.conf
            .partial_cmp(&a.conf)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}
