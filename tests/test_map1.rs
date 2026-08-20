//! 遍历 `test_cases/test_map*`：
//! - 用标注文件中的地图名解析 ID（不再 OCR）
//! - 导出图只用小地图截图作输入，定位/下载/标注写入 `tmp/<用例名>/`

use std::fs;
use std::path::{Path, PathBuf};

use image::RgbImage;
use mxd_tools::image_util::{crop_rgb, find_player_yellow, mark_cross};
use mxd_tools::locate::{locate_from_images, LocateResult, VIEW_H, VIEW_W, VIEW_X, VIEW_Y};
use mxd_tools::map_api::{fetch_canvas, fetch_full_map, resolve_map_id};
use mxd_tools::paths::safe_filename;

fn cases_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_cases")
}

fn fixtures_src(case_name: &str) -> PathBuf {
    cases_root().join(case_name)
}

fn observe_out_dir(case_name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tmp")
        .join(case_name);
    fs::create_dir_all(&dir).expect("创建观察输出目录失败");
    dir
}

fn case_names() -> Vec<String> {
    let mut names = Vec::new();
    for entry in fs::read_dir(cases_root()).expect("读取 test_cases") {
        let path = entry.expect("dir entry").path();
        if !path.is_dir() {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if name.starts_with("test_map") {
            names.push(name);
        }
    }
    names.sort();
    assert!(!names.is_empty(), "test_cases 下没有 test_map* 目录");
    names
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

fn expected_player_center(dir: &Path) -> (f64, f64) {
    let (tx, ty, tw, th) = load_expected_player_top_left(dir);
    (tx as f64 + tw as f64 / 2.0, ty as f64 + th as f64 / 2.0)
}

fn load_expected_full_map_ratio(dir: &Path) -> (f64, f64) {
    let path = find_named(dir, "x_y百分比");
    let content = fs::read_to_string(&path).expect("读完整地图百分比坐标");
    let line = content
        .lines()
        .find(|l| !l.trim().is_empty())
        .expect("百分比文件为空");
    let xy: Vec<f64> = line
        .trim()
        .split(',')
        .map(|s| s.trim().parse().expect("百分比数字"))
        .collect();
    assert_eq!(xy.len(), 2, "百分比格式应为 x,y：{line}");
    (xy[0], xy[1])
}

fn rust_locate_from_shot_and_names(
    shot: &RgbImage,
    street: &str,
    name: &str,
) -> (LocateResult, RgbImage) {
    let query = format!("{street}-{name}");
    let map_id = resolve_map_id(&query)
        .unwrap_or_else(|| panic!("解析地图 ID 失败 query={query} street={street} name={name}"));
    let canvas = fetch_canvas(map_id).expect("下载小地图画布失败");
    let full = fetch_full_map(map_id).expect("下载完整地图失败");
    let result = locate_from_images(shot, &canvas, &full, street, name, Some(map_id))
        .expect("Rust 定位失败");
    (result, full)
}

fn export_observe_artifacts(
    case_name: &str,
    shot: &RgbImage,
    full: &RgbImage,
    result: &LocateResult,
) -> PathBuf {
    let out = observe_out_dir(case_name);
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

fn assert_yellow(case_name: &str) {
    let dir = fixtures_src(case_name);
    let shot = load_shot(&dir);
    let (cx, cy) = expected_player_center(&dir);
    let view = crop_rgb(&shot, VIEW_X, VIEW_Y, VIEW_W, VIEW_H);
    let (vx, vy) = find_player_yellow(&view).expect("应能找到玩家黄点");
    let shot_x = VIEW_X as f64 + vx;
    let shot_y = VIEW_Y as f64 + vy;
    assert!(
        (shot_x - cx).abs() <= 2.0 && (shot_y - cy).abs() <= 2.0,
        "{case_name} 黄点中心 ({shot_x:.1},{shot_y:.1}) 应接近标注左上角换算中心 ({cx:.1},{cy:.1})"
    );
}

fn assert_resolve_names(case_name: &str) {
    let dir = fixtures_src(case_name);
    let (street, name) = load_expected_names(&dir);
    let query = format!("{street}-{name}");
    let map_id = resolve_map_id(&query).unwrap_or_else(|| panic!("{case_name} 应能解析 {query}"));
    assert!(map_id > 0, "{case_name} 地图 ID 无效");
}

fn assert_locate_and_export(case_name: &str) {
    let dir = fixtures_src(case_name);
    let shot = load_shot(&dir);
    let (want_street, want_name) = load_expected_names(&dir);
    let (result, full) = rust_locate_from_shot_and_names(&shot, &want_street, &want_name);
    let out = export_observe_artifacts(case_name, &shot, &full, &result);
    assert!(out.is_dir());

    assert_eq!(result.street, want_street, "{case_name} 一级地图");
    assert_eq!(result.name, want_name, "{case_name} 二级地图");
    assert!(
        result.align.score >= 0.35,
        "{case_name} 对齐分数过低：{:.3} mode={}",
        result.align.score,
        result.align.mode
    );

    let (cx, cy) = expected_player_center(&dir);
    assert!(
        (result.shot_x - cx).abs() <= 2.0 && (result.shot_y - cy).abs() <= 2.0,
        "{case_name} 截图坐标 ({:.1},{:.1}) 应接近标注中心 ({cx:.1},{cy:.1})",
        result.shot_x,
        result.shot_y
    );

    let (rx, ry) = load_expected_full_map_ratio(&dir);
    let got_rx = result.full_x / result.full_w as f64;
    let got_ry = result.full_y / result.full_h as f64;
    assert!(
        (got_rx - rx).abs() <= 0.12 && (got_ry - ry).abs() <= 0.12,
        "{case_name} 完整地图相对位置 ({got_rx:.3},{got_ry:.3}) 应接近标注百分比 ({rx},{ry})"
    );

    let prefix = format!("{case_name}_");
    let entries: Vec<_> = fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(entries.iter().any(|n| n.contains("__01_输入_小地图截图")));
    assert!(entries.iter().any(|n| n.contains("__02_原始_完整地图")));
    assert!(entries.iter().any(|n| n.contains("__03_标注_玩家位置")));
    assert!(entries.iter().any(|n| n.starts_with(&prefix)));
}

#[test]
fn test_map1_player_yellow_matches_annotation() {
    assert_yellow("test_map1");
}

#[test]
fn test_map1_resolve_map_names() {
    assert_resolve_names("test_map1");
}

#[test]
fn test_map1_locate_and_export_to_tmp() {
    assert_locate_and_export("test_map1");
}

#[test]
fn test_map2_player_yellow_matches_annotation() {
    assert_yellow("test_map2");
}

#[test]
fn test_map2_resolve_map_names() {
    assert_resolve_names("test_map2");
}

#[test]
fn test_map2_locate_and_export_to_tmp() {
    assert_locate_and_export("test_map2");
}

#[test]
fn test_all_case_dirs_are_covered() {
    let names = case_names();
    assert!(names.iter().any(|n| n == "test_map1"));
    assert!(names.iter().any(|n| n == "test_map2"), "未发现 test_map2");
}
