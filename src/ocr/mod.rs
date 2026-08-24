//! 文字识别：Windows 上使用 WinRT `Media.Ocr`（系统内置中文 OCR）。

#[cfg(windows)]
mod win;

use anyhow::Result;
use image::RgbImage;

/// 对 RGB 图像做 OCR，返回识别到的全文（可能含换行）。
pub fn recognize_rgb(img: &RgbImage) -> Result<String> {
    #[cfg(windows)]
    {
        return win::recognize_rgb(img);
    }
    #[cfg(not(windows))]
    {
        let _ = img;
        anyhow::bail!("OCR 当前仅实现 Windows（Media.Ocr）")
    }
}
