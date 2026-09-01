//! 用最新 ONNX 跑截图目录，并与 Python(.pt) 推理 JSON 比对。
//!
//! 用法：
//!   cargo run --release --bin yolo_compare -- \
//!     --source screen_caps/彩虹岛-南港西郊平原 \
//!     --onnx models/yolo_nangang_e1000.onnx \
//!     --pt models/yolo_nangang_e1000_best.pt

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use mxd_tools::yolo::{Detection, YoloDetector, YoloDevice};
use serde::Deserialize;

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
}

fn collect_images(source: &Path) -> Result<Vec<PathBuf>> {
    if source.is_file() {
        return Ok(vec![source.to_path_buf()]);
    }
    if !source.is_dir() {
        bail!("--source 不是文件或目录: {}", source.display());
    }
    let mut files: Vec<PathBuf> = fs::read_dir(source)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_ascii_lowercase())
                    .as_deref(),
                Some("png" | "jpg" | "jpeg" | "bmp" | "webp")
            )
        })
        .collect();
    files.sort();
    Ok(files)
}

#[derive(Debug, Deserialize)]
struct DumpFile {
    images: Vec<DumpImage>,
}

#[derive(Debug, Deserialize)]
struct DumpImage {
    file: String,
    dets: Vec<DumpDet>,
}

#[derive(Debug, Deserialize, Clone)]
struct DumpDet {
    class_id: usize,
    label: String,
    conf: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

fn iou(a: &DumpDet, b: &Detection) -> f32 {
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

struct MatchStats {
    py_n: usize,
    rust_n: usize,
    matched: usize,
    iou_sum: f32,
    conf_abs_sum: f32,
    class_mismatch: usize,
}

fn match_dets(py: &[DumpDet], rust: &[Detection], iou_thr: f32) -> MatchStats {
    let mut used = vec![false; rust.len()];
    let mut matched = 0usize;
    let mut iou_sum = 0.0f32;
    let mut conf_abs_sum = 0.0f32;
    let mut class_mismatch = 0usize;

    let mut py_sorted = py.to_vec();
    py_sorted.sort_by(|a, b| {
        b.conf
            .partial_cmp(&a.conf)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for p in &py_sorted {
        let mut best_j = None;
        let mut best_iou = 0.0f32;
        for (j, r) in rust.iter().enumerate() {
            if used[j] || r.class_id != p.class_id {
                continue;
            }
            let v = iou(p, r);
            if v > best_iou {
                best_iou = v;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            if best_iou >= iou_thr {
                used[j] = true;
                matched += 1;
                iou_sum += best_iou;
                conf_abs_sum += (p.conf - rust[j].conf).abs();
                if rust[j].class_id != p.class_id {
                    class_mismatch += 1;
                }
                continue;
            }
        }
        // 再试：跨类最高 IoU（统计类别错配）
        let mut best_j = None;
        let mut best_iou = 0.0f32;
        for (j, r) in rust.iter().enumerate() {
            if used[j] {
                continue;
            }
            let v = iou(p, r);
            if v > best_iou {
                best_iou = v;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            if best_iou >= iou_thr {
                used[j] = true;
                matched += 1;
                iou_sum += best_iou;
                conf_abs_sum += (p.conf - rust[j].conf).abs();
                class_mismatch += 1;
            }
        }
    }

    MatchStats {
        py_n: py.len(),
        rust_n: rust.len(),
        matched,
        iou_sum,
        conf_abs_sum,
        class_mismatch,
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") || args.is_empty() {
        eprintln!(
            "用法: yolo_compare --source <截图目录> --onnx <onnx> --pt <best.pt> \\\n\
             \t[--conf 0.25] [--iou 0.7] [--match-iou 0.5] [--min-recall 0.85] [--device cpu]"
        );
        std::process::exit(1);
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = PathBuf::from(arg_value(&args, "--source").context("需要 --source")?);
    let onnx =
        PathBuf::from(arg_value(&args, "--onnx").unwrap_or("models/yolo_nangang_e1000.onnx"));
    let pt = PathBuf::from(arg_value(&args, "--pt").unwrap_or("models/yolo_nangang_e1000_best.pt"));
    let conf: f32 = arg_value(&args, "--conf").unwrap_or("0.25").parse()?;
    let iou: f32 = arg_value(&args, "--iou").unwrap_or("0.7").parse()?;
    let match_iou: f32 = arg_value(&args, "--match-iou").unwrap_or("0.5").parse()?;
    let min_recall: f32 = arg_value(&args, "--min-recall").unwrap_or("0.85").parse()?;
    let min_precision: f32 = arg_value(&args, "--min-precision")
        .unwrap_or("0.85")
        .parse()?;
    let device = YoloDevice::parse(arg_value(&args, "--device").unwrap_or("cpu"));
    let py_device = arg_value(&args, "--py-device").unwrap_or("cpu");

    let source = if source.is_absolute() {
        source
    } else {
        root.join(&source)
    };
    let onnx = if onnx.is_absolute() {
        onnx
    } else {
        root.join(&onnx)
    };
    let pt = if pt.is_absolute() { pt } else { root.join(&pt) };

    let dump_path = root.join("tmp/yolo_compare_py.json");
    fs::create_dir_all(dump_path.parent().unwrap())?;

    let dump_script = root.join("scripts/dump_yolo_dets.py");
    eprintln!("==> Python 参考推理 → {}", dump_path.display());
    let st = Command::new("python")
        .arg(&dump_script)
        .arg("--model")
        .arg(&pt)
        .arg("--source")
        .arg(&source)
        .arg("--out")
        .arg(&dump_path)
        .arg("--conf")
        .arg(conf.to_string())
        .arg("--iou")
        .arg(iou.to_string())
        .arg("--device")
        .arg(py_device)
        .status()
        .context("启动 python dump_yolo_dets.py 失败")?;
    if !st.success() {
        bail!("python dump 失败: {st}");
    }

    let dump: DumpFile = serde_json::from_str(
        &fs::read_to_string(&dump_path).with_context(|| format!("读 {}", dump_path.display()))?,
    )?;
    let by_name: std::collections::HashMap<_, _> =
        dump.images.iter().map(|i| (i.file.as_str(), i)).collect();

    let mut det = YoloDetector::load_with_thresholds(&onnx, device, conf, iou, 640)?;
    eprintln!("==> Rust ONNX device={}", det.device_label);

    let images = collect_images(&source)?;
    if images.is_empty() {
        bail!("无图片: {}", source.display());
    }

    let mut tot_py = 0usize;
    let mut tot_rust = 0usize;
    let mut tot_matched = 0usize;
    let mut tot_iou = 0.0f32;
    let mut tot_conf = 0.0f32;
    let mut tot_cls_bad = 0usize;
    let mut worst: Option<(String, f32)> = None;

    for path in &images {
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("?");
        let Some(py_img) = by_name.get(name) else {
            eprintln!("跳过（Python dump 无此文件）: {name}");
            continue;
        };
        let img = image::open(path)
            .with_context(|| format!("读图 {}", path.display()))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        let rust_dets = det.detect_rgb8(w, h, img.as_raw())?;
        let st = match_dets(&py_img.dets, &rust_dets, match_iou);
        let recall = if st.py_n == 0 {
            1.0
        } else {
            st.matched as f32 / st.py_n as f32
        };
        let precision = if st.rust_n == 0 {
            1.0
        } else {
            st.matched as f32 / st.rust_n as f32
        };
        let mean_iou = if st.matched == 0 {
            0.0
        } else {
            st.iou_sum / st.matched as f32
        };
        println!(
            "{name}: py={} rust={} matched={} recall={:.3} prec={:.3} mean_iou={:.3} conf_mae={:.3}",
            st.py_n,
            st.rust_n,
            st.matched,
            recall,
            precision,
            mean_iou,
            if st.matched == 0 {
                0.0
            } else {
                st.conf_abs_sum / st.matched as f32
            }
        );
        if let Some(ref mut w) = worst {
            if recall < w.1 {
                *w = (name.to_string(), recall);
            }
        } else {
            worst = Some((name.to_string(), recall));
        }

        tot_py += st.py_n;
        tot_rust += st.rust_n;
        tot_matched += st.matched;
        tot_iou += st.iou_sum;
        tot_conf += st.conf_abs_sum;
        tot_cls_bad += st.class_mismatch;
    }

    let recall = if tot_py == 0 {
        1.0
    } else {
        tot_matched as f32 / tot_py as f32
    };
    let precision = if tot_rust == 0 {
        1.0
    } else {
        tot_matched as f32 / tot_rust as f32
    };
    let mean_iou = if tot_matched == 0 {
        0.0
    } else {
        tot_iou / tot_matched as f32
    };
    let conf_mae = if tot_matched == 0 {
        0.0
    } else {
        tot_conf / tot_matched as f32
    };

    println!("----");
    println!(
        "合计: py={tot_py} rust={tot_rust} matched={tot_matched} \
         recall={recall:.3} prec={precision:.3} mean_iou={mean_iou:.3} \
         conf_mae={conf_mae:.3} class_mismatch={tot_cls_bad}"
    );
    if let Some((name, r)) = worst {
        println!("最差 recall: {name} ({r:.3})");
    }

    if recall < min_recall || precision < min_precision || mean_iou < 0.70 {
        eprintln!(
            "FAIL: Rust 与 Python 差距过大 (需要 recall>={min_recall}, prec>={min_precision}, mean_iou>=0.70)"
        );
        std::process::exit(2);
    }
    println!("PASS: Rust 与 Python 足够接近");
    Ok(())
}
