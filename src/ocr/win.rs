use anyhow::{Context, Result};
use image::RgbImage;
use windows::core::HSTRING;
use windows::Globalization::Language;
use windows::Graphics::Imaging::{BitmapPixelFormat, SoftwareBitmap};
use windows::Media::Ocr::OcrEngine;
use windows::Storage::Streams::DataWriter;

fn rgb_to_bgra(img: &RgbImage) -> Vec<u8> {
    let (w, h) = img.dimensions();
    let mut bgra = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let [r, g, b] = img.get_pixel(x, y).0;
            let i = ((y * w + x) * 4) as usize;
            bgra[i] = b;
            bgra[i + 1] = g;
            bgra[i + 2] = r;
            bgra[i + 3] = 255;
        }
    }
    bgra
}

fn create_software_bitmap(img: &RgbImage) -> Result<SoftwareBitmap> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        anyhow::bail!("OCR 输入尺寸为 0");
    }
    let bgra = rgb_to_bgra(img);
    let writer = DataWriter::new().context("DataWriter::new")?;
    writer
        .WriteBytes(&bgra)
        .context("WriteBytes")?;
    let buffer = writer
        .DetachBuffer()
        .context("DetachBuffer")?;
    let bitmap = SoftwareBitmap::CreateCopyFromBuffer(
        &buffer,
        BitmapPixelFormat::Bgra8,
        w as i32,
        h as i32,
    )
    .context("CreateCopyFromBuffer")?;
    Ok(bitmap)
}

fn create_ocr_engine() -> Result<OcrEngine> {
    if let Ok(engine) = OcrEngine::TryCreateFromLanguage(&Language::CreateLanguage(&HSTRING::from(
        "zh-CN",
    ))?) {
        return Ok(engine);
    }
    OcrEngine::TryCreateFromUserProfileLanguages().context("创建 OCR 引擎失败（需 Windows 10+ 中文语言包）")
}

pub fn recognize_rgb(img: &RgbImage) -> Result<String> {
    let engine = create_ocr_engine()?;
    let bitmap = create_software_bitmap(img)?;
    let text = engine
        .RecognizeAsync(&bitmap)
        .context("RecognizeAsync")?
        .get()
        .context("等待 OCR 结果")?
        .Text()
        .context("读取 OCR 文本")?
        .to_string();
    Ok(text)
}
