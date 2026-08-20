//! 批量验证：完整窗口截图 → 匹配小地图 → 标注完整大地图。
//!
//! 用法：
//!   cargo run --release --bin validate_screen_caps -- \
//!     --caps screen_caps/彩虹岛-南港西郊平原 \
//!     --minimap assets/maps/50001/map_50001_minimap.png \
//!     --full assets/maps/50001/map_50001_render_cn.png

use std::env;
use std::path::PathBuf;

use mxd_tools::minimap_match::validate_screen_caps_dir;

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") || args.is_empty() {
        eprintln!(
            "用法: validate_screen_caps --caps <截图目录> --minimap <png> --full <png> [--out <目录>]"
        );
        std::process::exit(1);
    }

    let caps_dir = PathBuf::from(arg_value(&args, "--caps").expect("需要 --caps"));
    let minimap_path = PathBuf::from(arg_value(&args, "--minimap").expect("需要 --minimap"));
    let full_path = PathBuf::from(arg_value(&args, "--full").expect("需要 --full"));
    let out_dir = arg_value(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tmp/screen_cap_locate")
        });

    match validate_screen_caps_dir(&caps_dir, &minimap_path, &full_path, &out_dir, None) {
        Ok(sum) => {
            for line in &sum.lines {
                println!("{line}");
            }
            if sum.ok != sum.total {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("失败：{e}");
            std::process::exit(2);
        }
    }
}
