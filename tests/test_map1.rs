//! test_map1
//!
//! - 断言可用用例中的标注文件做校验
//! - 导出图：只用小地图截图作输入，全程 Rust OCR/定位/下载/标注，结果写入项目 `tmp/<用例名>/`

use std::fs;
use std::path::{Path, PathBuf};

use image::RgbImage;
use mxd_tools::image_util::{crop_rgb, find_player_yellow, mark_cross};
use mxd_tools::locate::{locate_from_images, LocateResult, VIEW_H, VIEW_W, VIEW_X, VIEW_Y};
use mxd_tools::map_api::{fetch_canvas, fetch_full_map, resolve_map_id};
use mxd_tools::ocr::read_map_names;
use mxd_tools::paths::safe_filename;

const CASE_NAME: &str = "test_map1";

fn fixtures_src() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_cases")
        .join(CASE_NAME)
}

/// 持久观察目录：`mxd_tools/tmp/<用例名>/`（已 gitignore）
fn observe_out_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tmp")
        .join(CASE_NAME);
    fs::create_dir_all(&dir).expect("创建观察输出目录失败");
    dir
}

fn find_named(dir: &Path, needle: &str) -> PathBuf {
    for entry in fs::read_dir(dir).expect("读取用例目录") {
        let path = entry.expect("dir entry").path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.contains(needle) {
            return path;
        }
    }
    panic!("在 {} 中找不到包含「{needle}」的文件", dir.display());
}

fn find_shot_png(dir: &Path) -> PathBuf {
    for entry in fs::read_dir(dir).expect("读取用例目录") {
        let path = entry.expect("dir entry").path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if path.extension().and_then(|s| s.to_str()) == Some("png") && name.contains("222") {
            return path;
        }
    }
    panic!("在 {} 中找不到 222 小地图 png", dir.display());
}

fn load_shot(dir: &Path) -> RgbImage {
    let path = find_shot_png(dir);
    let img = image::open(&path)
        .unwrap_or_else(|e| panic!("打开小地图失败 {}: {e}", path.display()))
        .to_rgb8();
    assert_eq!((img.width(), img.height()), (222, 222));
    img
}

fn load_expected_names(dir: &Path) -> (String, String) {
    let path = find_named(dir, "地图名称");
    let content = fs::read_to_string(&path).expect("读地图名称");
    let mut street = None;
    let mut name = None;
    for line in content.lines() {
        if let Some(v) = line.strip_prefix("一级地图名字:") {
            street = Some(v.trim().to_string());
        }
        if let Some(v) = line.strip_prefix("二级地图名字:") {
            name = Some(v.trim().to_string());
        }
    }
    (
        street.expect("缺少一级地图名字"),
        name.expect("缺少二级地图名字"),
    )
}

fn load_expected_player_top_left(dir: &Path) -> (u32, u32, u32, u32) {
    let path = find_named(dir, "玩家真实位置");
    let content = fs::read_to_string(&path).expect("读坐标文件");
    let parts: Vec<&str> = content.split_whitespace().collect();
    assert!(parts.len() >= 2, "坐标文件格式错误：{content}");
    let xy: Vec<u32> = parts[0]
        .split(',')
        .map(|s| s.trim().parse().expect("坐标数字"))
        .collect();
    let wh: Vec<u32> = parts[1]
        .split(['x', 'X', '×'])
        .map(|s| s.trim().parse().expect("尺寸数字"))
        .collect();
    (xy[0], xy[1], wh[0], wh[1])
}

fn load_expected_full_map_ratio(dir: &Path) -> (f64, f64) {
    let path = find_named(dir, "x_y百分比");
    let content = fs::read_to_string(&path).expect("读完整地图百分比坐标");
    let xy: Vec<f64> = content
        .trim()
        .split(',')
        .map(|s| s.trim().parse().expect("百分比数字"))
        .collect();
    assert_eq!(xy.len(), 2, "百分比格式应为 x,y：{content}");
    (xy[0], xy[1])
}

/// 仅用小地图截图：Rust OCR → 解析 ID → 网络拉画布/完整图 → 定位 → 标注。
/// 不读用例里的地图名/坐标/完整地图文件。
fn rust_locate_from_shot_only(shot: &RgbImage) -> (LocateResult, RgbImage, RgbImage) {
    let (street, name) = read_map_names(shot).expect("Rust OCR 失败");
    let query = format!("{street}-{name}");
    let map_id = resolve_map_id(&query).expect("解析地图 ID 失败");
    let canvas = fetch_canvas(map_id).expect("下载小地图画布失败");
    let full = fetch_full_map(map_id).expect("下载完整地图失败");
    let result = locate_from_images(shot, &canvas, &full, &street, &name, Some(map_id))
        .expect("Rust 定位失败");
    (result, full, canvas)
}

/// 导出到 `tmp/<用例名>/`，文件名前缀带用例名，避免多用例互相覆盖。
fn export_observe_artifacts(
    case_name: &str,
    shot: &RgbImage,
    full: &RgbImage,
    result: &LocateResult,
) -> PathBuf {
    let out = observe_out_dir();
    let tag = safe_filename(&format!(
        "{case_name}_{}-{}{}",
        result.street,
        result.name,
        result
            .map_id
            .map(|id| format!("_{id}"))
            .unwrap_or_default()
    ));

    let shot_path = out.join(format!("{tag}__01_输入_小地图截图.png"));
    let full_path = out.join(format!("{tag}__02_原始_完整地图.png"));
    let marked_path = out.join(format!("{tag}__03_标注_玩家位置.png"));
    let meta_path = out.join(format!("{tag}__04_识别结果.txt"));

    shot.save(&shot_path).expect("写小地图截图失败");
    full.save(&full_path).expect("写原始完整地图失败");

    let mut marked = full.clone();
    mark_cross(&mut marked, result.full_x, result.full_y);
    marked.save(&marked_path).expect("写标注图失败");

    let meta = format!(
        "用例 {case_name}\n一级地图 {}\n二级地图 {}\n地图ID {:?}\n对齐 {} 分数 {:.3} 原点 {:?}\n玩家截图 {:.1},{:.1}\n画布 {:.1},{:.1} ({}x{})\n完整地图 {:.1},{:.1} ({}x{})\n相对位置 {:.4},{:.4}\n",
        result.street,
        result.name,
        result.map_id,
        result.align.mode,
        result.align.score,
        result.align.loc,
        result.shot_x,
        result.shot_y,
        result.canvas_x,
        result.canvas_y,
        result.canvas_w,
        result.canvas_h,
        result.full_x,
        result.full_y,
        result.full_w,
        result.full_h,
        result.full_x / result.full_w as f64,
        result.full_y / result.full_h as f64,
    );
    fs::write(&meta_path, meta).expect("写识别结果失败");

    println!("已导出观察结果到 {}", out.display());
    println!("  {}", shot_path.file_name().unwrap().to_string_lossy());
    println!("  {}", full_path.file_name().unwrap().to_string_lossy());
    println!("  {}", marked_path.file_name().unwrap().to_string_lossy());
    println!("  {}", meta_path.file_name().unwrap().to_string_lossy());
    out
}

#[test]
fn test_map1_player_yellow_matches_annotation() {
    let dir = fixtures_src();
    let shot = load_shot(&dir);
    let (tx, ty, tw, th) = load_expected_player_top_left(&dir);
    let expected_cx = tx as f64 + tw as f64 / 2.0;
    let expected_cy = ty as f64 + th as f64 / 2.0;

    let view = crop_rgb(&shot, VIEW_X, VIEW_Y, VIEW_W, VIEW_H);
    let (vx, vy) = find_player_yellow(&view).expect("应能找到玩家黄点");
    let shot_x = VIEW_X as f64 + vx;
    let shot_y = VIEW_Y as f64 + vy;

    assert!(
        (shot_x - expected_cx).abs() <= 2.0 && (shot_y - expected_cy).abs() <= 2.0,
        "黄点中心 ({shot_x:.1},{shot_y:.1}) 应接近标注中心 ({expected_cx:.1},{expected_cy:.1})"
    );
}

#[test]
fn test_map1_ocr_map_names() {
    let dir = fixtures_src();
    let (want_street, want_name) = load_expected_names(&dir);
    let shot = load_shot(&dir);
    let (street, name) = read_map_names(&shot).expect("OCR 应识别两行地图名");
    assert_eq!(street, want_street, "一级地图");
    assert_eq!(name, want_name, "二级地图");
}

#[test]
fn test_map1_locate_and_export_to_tmp() {
    let dir = fixtures_src();
    let shot = load_shot(&dir);

    // 识别/下载/标注：只用截图 + Rust，不用用例标注文本/本地完整图
    let (result, full, _canvas) = rust_locate_from_shot_only(&shot);
    let out = export_observe_artifacts(CASE_NAME, &shot, &full, &result);
    assert!(out.is_dir(), "观察目录应存在：{}", out.display());

    // 以下仅用标注文件做断言，不参与出图
    let (want_street, want_name) = load_expected_names(&dir);
    assert_eq!(result.street, want_street);
    assert_eq!(result.name, want_name);
    assert_eq!(result.map_id, Some(2_000_000));
    assert!(result.align.score >= 0.5);
    assert_eq!(result.align.mode, "partial");
    assert_eq!(result.align.loc, (44, 0));

    let (tx, ty, tw, th) = load_expected_player_top_left(&dir);
    let expected_cx = tx as f64 + tw as f64 / 2.0;
    let expected_cy = ty as f64 + th as f64 / 2.0;
    assert!(
        (result.shot_x - expected_cx).abs() <= 2.0 && (result.shot_y - expected_cy).abs() <= 2.0
    );

    let (rx, ry) = load_expected_full_map_ratio(&dir);
    let got_rx = result.full_x / result.full_w as f64;
    let got_ry = result.full_y / result.full_h as f64;
    assert!(
        (got_rx - rx).abs() <= 0.03 && (got_ry - ry).abs() <= 0.03,
        "完整地图相对位置 ({got_rx:.3},{got_ry:.3}) 应接近标注百分比 ({rx},{ry})"
    );

    // 确认导出文件都在项目 tmp/用例名 下且文件名互不冲突
    let entries: Vec<_> = fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(entries.iter().any(|n| n.contains("__01_输入_小地图截图")));
    assert!(entries.iter().any(|n| n.contains("__02_原始_完整地图")));
    assert!(entries.iter().any(|n| n.contains("__03_标注_玩家位置")));
    assert!(entries.iter().any(|n| n.starts_with("test_map1_")));
}
