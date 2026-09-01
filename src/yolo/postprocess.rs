use crate::yolo::labels::class_name;
use crate::yolo::{Detection, LetterboxMeta};

/// 从 ORT 输出切片解码，避免整份 `to_vec()` 拷贝。
pub fn decode_yolo_output_flat(
    shape: &[i64],
    data: &[f32],
    meta: &LetterboxMeta,
    conf_thres: f32,
    iou_thres: f32,
) -> Vec<Detection> {
    let (channels, num) = match shape {
        [_, c, n] => (*c as usize, *n as usize),
        [c, n] => (*c as usize, *n as usize),
        _ => return Vec::new(),
    };
    if channels < 5 || num == 0 || data.len() < channels * num {
        return Vec::new();
    }
    let nc = channels - 4;

    let mut candidates: Vec<Detection> = Vec::new();
    for i in 0..num {
        let cx = data[i];
        let cy = data[num + i];
        let w = data[2 * num + i];
        let h = data[3 * num + i];

        let mut best_cls = 0usize;
        let mut best_score = data[4 * num + i];
        for c in 1..nc {
            let s = data[(4 + c) * num + i];
            if s > best_score {
                best_score = s;
                best_cls = c;
            }
        }
        if best_score < conf_thres {
            continue;
        }

        let x1 = (cx - w * 0.5 - meta.pad_x) / meta.gain;
        let y1 = (cy - h * 0.5 - meta.pad_y) / meta.gain;
        let x2 = (cx + w * 0.5 - meta.pad_x) / meta.gain;
        let y2 = (cy + h * 0.5 - meta.pad_y) / meta.gain;

        let (x1, y1, x2, y2) = clip_xyxy(x1, y1, x2, y2, meta.orig_w, meta.orig_h);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        candidates.push(Detection {
            class_id: best_cls,
            label: class_name(best_cls),
            conf: best_score,
            x1,
            y1,
            x2,
            y2,
        });
    }

    nms(candidates, iou_thres)
}

/// 批量 YOLO 输出解码：shape `[N, 4+nc, num_anchors]`。
pub fn decode_yolo_batch_output(
    shape: &[i64],
    data: &[f32],
    metas: &[LetterboxMeta],
    conf_thres: f32,
    iou_thres: f32,
) -> Vec<Vec<Detection>> {
    let (batch, channels, num) = match shape {
        [b, c, n] => (*b as usize, *c as usize, *n as usize),
        [b, _, c, n] if shape.len() == 4 => (*b as usize, *c as usize, *n as usize),
        _ => return vec![],
    };
    if batch == 0 || channels < 5 || num == 0 {
        return vec![];
    }
    let plane = channels * num;
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let start = b * plane;
        if start + plane > data.len() {
            out.push(Vec::new());
            continue;
        }
        let slice = &data[start..start + plane];
        let meta = metas.get(b).copied().unwrap_or(LetterboxMeta {
            gain: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            orig_w: 0,
            orig_h: 0,
        });
        out.push(decode_yolo_output_flat(
            &[1, channels as i64, num as i64],
            slice,
            &meta,
            conf_thres,
            iou_thres,
        ));
    }
    out
}

/// Ultralytics YOLO 输出: shape [1, 4+nc, num_anchors]，如 [1, 23, 8400]。
pub fn decode_yolo_output(
    output: &ndarray::ArrayD<f32>,
    meta: &LetterboxMeta,
    conf_thres: f32,
    iou_thres: f32,
) -> Vec<Detection> {
    let shape_i64: Vec<i64> = output.shape().iter().map(|&d| d as i64).collect();
    decode_yolo_output_flat(
        &shape_i64,
        output.as_slice().unwrap_or(&[]),
        meta,
        conf_thres,
        iou_thres,
    )
}

fn clip_xyxy(x1: f32, y1: f32, x2: f32, y2: f32, w: u32, h: u32) -> (f32, f32, f32, f32) {
    let wf = w as f32;
    let hf = h as f32;
    (
        x1.clamp(0.0, wf),
        y1.clamp(0.0, hf),
        x2.clamp(0.0, wf),
        y2.clamp(0.0, hf),
    )
}

fn box_iou(a: &Detection, b: &Detection) -> f32 {
    let xx1 = a.x1.max(b.x1);
    let yy1 = a.y1.max(b.y1);
    let xx2 = a.x2.min(b.x2);
    let yy2 = a.y2.min(b.y2);
    let w = (xx2 - xx1).max(0.0);
    let h = (yy2 - yy1).max(0.0);
    let inter = w * h;
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let uni = area_a + area_b - inter;
    if uni <= 0.0 {
        0.0
    } else {
        inter / uni
    }
}

fn nms(mut dets: Vec<Detection>, iou_thres: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| {
        b.conf
            .partial_cmp(&a.conf)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = Vec::new();
    let mut suppressed = vec![false; dets.len()];
    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if suppressed[j] {
                continue;
            }
            if dets[i].class_id != dets[j].class_id {
                continue;
            }
            if box_iou(&dets[i], &dets[j]) > iou_thres {
                suppressed[j] = true;
            }
        }
    }
    keep
}
