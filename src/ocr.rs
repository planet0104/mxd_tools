use std::path::{Path, PathBuf};
use std::sync::Mutex;

use image::{imageops::FilterType, RgbImage};
use paddle_ocr_rs::ocr_lite::OcrLite;

/// 与 save_from_minimap.py 相同：NAME_BOX = (46, 24, 46+129, 24+36)
const NAME_BOX: (u32, u32, u32, u32) = (46, 24, 129, 36);

const DET_NAME: &str = "ch_PP-OCRv4_det_infer.onnx";
const CLS_NAME: &str = "ch_ppocr_mobile_v2.0_cls_infer.onnx";
const REC_NAME: &str = "ch_PP-OCRv4_rec_infer.onnx";

static OCR: Mutex<Option<OcrLite>> = Mutex::new(None);

fn models_dir() -> Result<PathBuf, String> {
    let mut dirs = Vec::new();
    if let Ok(p) = std::env::var("MXD_OCR_MODELS") {
        dirs.push(PathBuf::from(p));
    }
    dirs.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models"));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            dirs.push(dir.join("models"));
        }
    }
    if let Some(p) = python_rapidocr_models() {
        dirs.push(p);
    }

    for dir in dirs {
        if dir.join(DET_NAME).is_file()
            && dir.join(CLS_NAME).is_file()
            && dir.join(REC_NAME).is_file()
        {
            return Ok(dir);
        }
    }
    Err(format!(
        "找不到 RapidOCR 同款 ONNX 模型（{DET_NAME} / {CLS_NAME} / {REC_NAME}）。\
         可把 rapidocr_onnxruntime 的 models 目录拷到 mxd_tools/models，或设置 MXD_OCR_MODELS。"
    ))
}

fn python_rapidocr_models() -> Option<PathBuf> {
    let output = std::process::Command::new("python")
        .args([
            "-c",
            "import rapidocr_onnxruntime as m, pathlib; print(pathlib.Path(m.__file__).parent / 'models')",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let dir = PathBuf::from(path);
    if dir.is_dir() {
        Some(dir)
    } else {
        None
    }
}

fn engine() -> Result<std::sync::MutexGuard<'static, Option<OcrLite>>, String> {
    let mut guard = OCR.lock().map_err(|_| "OCR 锁失败".to_string())?;
    if guard.is_none() {
        let dir = models_dir()?;
        let mut ocr = OcrLite::new();
        ocr.init_models(
            dir.join(DET_NAME).to_str().ok_or("模型路径无效")?,
            dir.join(CLS_NAME).to_str().ok_or("模型路径无效")?,
            dir.join(REC_NAME).to_str().ok_or("模型路径无效")?,
            2,
        )
        .map_err(|e| format!("加载 RapidOCR ONNX 失败：{e}"))?;
        *guard = Some(ocr);
    }
    Ok(guard)
}

fn chinese_len(s: &str) -> usize {
    s.chars()
        .filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c))
        .count()
}

fn looks_like_map_name(s: &str) -> bool {
    let n = chinese_len(s);
    n >= 2 && n <= 12 && s.chars().count() <= 16
}

fn run_detect(
    ocr: &mut OcrLite,
    img: &RgbImage,
    padding: u32,
    do_angle: bool,
) -> Result<Vec<(f64, String)>, String> {
    // 参数对齐 RapidOCR / RapidOcrOnnx：boxScoreThresh=0.5, boxThresh=0.3, unClipRatio=1.6
    let result = ocr
        .detect(img, padding, 1024, 0.5, 0.3, 1.6, do_angle, false)
        .map_err(|e| e.to_string())?;
    let mut items = Vec::new();
    for block in result.text_blocks {
        let text = block.text.trim().to_string();
        if text.is_empty() {
            continue;
        }
        let y = block
            .box_points
            .iter()
            .map(|p| p.y as f64)
            .fold(f64::INFINITY, f64::min);
        items.push((if y.is_finite() { y } else { 0.0 }, text));
    }
    items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(items)
}

fn score_names(names: &[(f64, String)]) -> i32 {
    let good: Vec<&String> = names
        .iter()
        .map(|(_, t)| t)
        .filter(|t| looks_like_map_name(t))
        .collect();
    let mut score = good.len() as i32 * 10;
    for t in &good {
        score += chinese_len(t) as i32;
    }
    score
}

/// 纯 Rust 推理 RapidOCR 同款 ONNX（paddle-ocr-rs + ort），流程对齐 Python save_from_minimap.read_map_names。
pub fn read_map_names(image: &RgbImage) -> Result<(String, String), String> {
    let (x, y, w, h) = NAME_BOX;
    if image.width() < x + w || image.height() < y + h {
        return Err(format!(
            "截图太小：{}x{}，需要至少 {}x{}",
            image.width(),
            image.height(),
            x + w,
            y + h
        ));
    }
    let crop = image::imageops::crop_imm(image, x, y, w, h).to_image();
    let scaled = image::imageops::resize(&crop, w * 3, h * 3, FilterType::Nearest);

    let mut guard = engine()?;
    let ocr = guard.as_mut().ok_or("OCR 未初始化")?;

    // 小地图名称框通常正向，优先不做角度分类（避免误旋转导致乱码）
    let attempts: [(u32, bool); 4] = [(0, false), (10, false), (50, false), (50, true)];
    let mut best: Option<(i32, Vec<(f64, String)>)> = None;
    let mut last_err = String::new();
    for (padding, do_angle) in attempts {
        match run_detect(ocr, &scaled, padding, do_angle) {
            Ok(items) => {
                let s = score_names(&items);
                if best.as_ref().map(|(bs, _)| s > *bs).unwrap_or(true) {
                    best = Some((s, items));
                }
            }
            Err(e) => last_err = e,
        }
    }

    let Some((_score, items)) = best else {
        return Err(if last_err.is_empty() {
            "OCR 无结果".into()
        } else {
            last_err
        });
    };
    let names: Vec<String> = items
        .into_iter()
        .map(|(_, t)| t)
        .filter(|t| looks_like_map_name(t))
        .collect();
    if names.len() < 2 {
        return Err(format!("未能识别两行地图名：{}", names.join(" / ")));
    }
    Ok((names[0].clone(), names[1].clone()))
}

pub fn read_map_names_from_path(path: &Path) -> Result<(String, String), String> {
    let img = image::open(path).map_err(|e| e.to_string())?.to_rgb8();
    read_map_names(&img)
}
