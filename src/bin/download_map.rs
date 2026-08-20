//! 按地图名或地图 ID 下载官方小地图 + 完整渲染图。
//!
//! 用法：
//!   cargo run --bin download_map -- 50001
//!   cargo run --bin download_map -- 彩虹岛-南港西郊平原
//!   cargo run --bin download_map -- 南港 --out tmp

use std::env;
use std::path::PathBuf;

use mxd_tools::map_api::{download_map_pngs, fetch_map_name, resolve_map_id};

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "用法: download_map <地图名|地图ID> [--out <目录>]\n\
             例:   download_map 50001\n\
             例:   download_map 彩虹岛-南港西郊平原 --out tmp"
        );
        std::process::exit(1);
    }

    let mut out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tmp");
    if let Some(i) = args.iter().position(|a| a == "--out") {
        if i + 1 >= args.len() {
            eprintln!("--out 需要目录参数");
            std::process::exit(1);
        }
        out = PathBuf::from(&args[i + 1]);
        args.drain(i..=i + 1);
    }
    let query = args.join(" ");

    let map_id = match resolve_map_id(&query) {
        Some(id) => id,
        None => {
            eprintln!("无法解析地图：{query}");
            eprintln!("可直接传数字 ID，例如 50001；或到 mxd.dvg.cn 用 NPC/地图名查 ID。");
            std::process::exit(2);
        }
    };

    println!("地图 ID {map_id}");
    if let Some((street, name)) = fetch_map_name(map_id) {
        println!("maplestory.io: {street} / {name}");
    }

    match download_map_pngs(map_id, &out) {
        Ok((mini, full)) => {
            println!("小地图 {}", mini.display());
            println!("完整图 {}", full.display());
        }
        Err(e) => {
            eprintln!("下载失败：{e}");
            std::process::exit(3);
        }
    }
}
