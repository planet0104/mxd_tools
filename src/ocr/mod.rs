//! 文字识别：PP-OCRv5 det + PP-OCRv4 rec（ONNX Runtime）。

mod det;
mod paddle;
mod runtime;

use anyhow::Result;
use image::RgbImage;

pub use det::TextBox;
pub use runtime::OcrRuntime;

/// 在图像中检测文本行区域。
pub fn detect_text_boxes(img: &RgbImage) -> Result<Vec<TextBox>> {
    det::detect_text_boxes(img)
}

/// 对 RGB 图像做 OCR，返回识别到的全文。
pub fn recognize_rgb(img: &RgbImage) -> Result<String> {
    paddle::recognize_rgb(img)
}

/// 批量 OCR，多张名牌 ROI 一次推理。
pub fn recognize_rgb_batch(imgs: &[&RgbImage]) -> Result<Vec<String>> {
    paddle::recognize_rgb_batch(imgs)
}
