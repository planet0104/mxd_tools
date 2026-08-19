use std::io::Cursor;
use std::path::{Path, PathBuf};

use image::RgbImage;
use regex::Regex;
use serde_json::Value;

use crate::paths::safe_filename;

const WIKI_API: &str = "https://wiki.biligame.com/maplestory/api.php";
const RENDER_URL: &str = "https://maplestory.io/api/GMS/83/map/{map_id}/render";
const NAME_URL: &str = "https://maplestory.io/api/GMS/83/map/{map_id}/name";
const MINIMAP_URL: &str = "https://maplestory.io/api/GMS/83/map/{map_id}/minimap";
const UA: &str = "Mozilla/5.0";

fn client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .user_agent(UA)
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())
}

pub fn http_bytes(url: &str) -> Result<(String, Vec<u8>), String> {
    let resp = client()?.get(url).send().map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let ctype = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let data = resp.bytes().map_err(|e| e.to_string())?.to_vec();
    Ok((ctype, data))
}

fn http_json(url: &str) -> Result<Value, String> {
    let resp = client()?.get(url).send().map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json().map_err(|e| e.to_string())
}

fn parse_map_id(text: &str) -> Option<u64> {
    let re = Regex::new(r"(?:Map/)?(\d{7,9})").ok()?;
    let cleaned = text.replace(',', "");
    re.captures(&cleaned)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse().ok())
}

fn wiki_title_id(title: &str) -> Option<u64> {
    let url = format!(
        "{WIKI_API}?action=query&format=json&redirects=1&titles={}",
        urlencoding::encode(title)
    );
    let data = http_json(&url).ok()?;
    if let Some(arr) = data.pointer("/query/redirects").and_then(|v| v.as_array()) {
        for item in arr {
            if let Some(to) = item.get("to").and_then(|v| v.as_str()) {
                if let Some(id) = parse_map_id(to) {
                    return Some(id);
                }
            }
        }
    }
    if let Some(pages) = data.pointer("/query/pages").and_then(|v| v.as_object()) {
        for page in pages.values() {
            if let Some(title) = page.get("title").and_then(|v| v.as_str()) {
                if let Some(id) = parse_map_id(title) {
                    return Some(id);
                }
            }
        }
    }
    None
}

fn wiki_search_id(keyword: &str) -> Option<u64> {
    let url = format!(
        "{WIKI_API}?action=query&format=json&list=search&srlimit=10&srsearch={}",
        urlencoding::encode(keyword)
    );
    let data = http_json(&url).ok()?;
    let arr = data.pointer("/query/search")?.as_array()?;
    for item in arr {
        for key in ["title", "snippet"] {
            if let Some(text) = item.get(key).and_then(|v| v.as_str()) {
                if let Some(id) = parse_map_id(text) {
                    return Some(id);
                }
            }
        }
    }
    None
}

fn candidates(name: &str) -> Vec<String> {
    let text = name.trim().to_string();
    let mut out = vec![text.clone()];
    out.push(
        text.replace('-', ":")
            .replace('：', ":")
            .replace('/', ":"),
    );
    let parts: Vec<&str> = text
        .split(|c| matches!(c, '-' | ':' | '：' | '/' | '｜' | '|'))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if let Some(last) = parts.last() {
        out.push((*last).to_string());
        if parts.len() >= 2 {
            out.push(format!("{}:{}", parts[0], last));
        }
    }
    out
}

pub fn resolve_map_id(name: &str) -> Option<u64> {
    let text = name.trim();
    if Regex::new(r"^\d{1,9}$").ok()?.is_match(text) {
        return text.parse().ok();
    }
    let mut seen = std::collections::HashSet::new();
    for cand in candidates(text) {
        if !seen.insert(cand.clone()) {
            continue;
        }
        if let Some(id) = wiki_title_id(&cand) {
            return Some(id);
        }
    }
    wiki_search_id(text)
}

fn map_label(map_id: u64, fallback: &str) -> String {
    let url = NAME_URL.replace("{map_id}", &map_id.to_string());
    let Ok(info) = http_json(&url) else {
        return fallback.to_string();
    };
    let street = info
        .get("streetName")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let name = info.get("name").and_then(|v| v.as_str()).unwrap_or("").trim();
    if !street.is_empty() && !name.is_empty() {
        format!("{street}_{name}")
    } else if !name.is_empty() {
        name.to_string()
    } else if !street.is_empty() {
        street.to_string()
    } else {
        fallback.to_string()
    }
}

fn decode_png(data: &[u8], ctype: &str) -> Result<RgbImage, String> {
    if !ctype.to_lowercase().contains("png") && !data.starts_with(b"\x89PNG") {
        return Err(format!("接口没有返回图片：{ctype}"));
    }
    image::load(Cursor::new(data), image::ImageFormat::Png)
        .map_err(|e| e.to_string())
        .map(|img| img.to_rgb8())
}

/// 从网络拉取完整地图 PNG（内存，不读本地文件）。
pub fn fetch_full_map(map_id: u64) -> Result<RgbImage, String> {
    let url = RENDER_URL.replace("{map_id}", &map_id.to_string());
    let (ctype, data) = http_bytes(&url)?;
    decode_png(&data, &ctype)
}

/// 从网络拉取官方小地图画布（内存，不读本地缓存）。
pub fn fetch_canvas(map_id: u64) -> Result<RgbImage, String> {
    let url = MINIMAP_URL.replace("{map_id}", &map_id.to_string());
    let (ctype, data) = http_bytes(&url)?;
    decode_png(&data, &ctype)
}

pub fn save_map(name: &str, out_dir: &Path) -> Result<(u64, PathBuf, String), String> {
    let map_id = resolve_map_id(name).ok_or_else(|| format!("找不到地图：{name}"))?;
    let url = RENDER_URL.replace("{map_id}", &map_id.to_string());
    let (ctype, data) = http_bytes(&url)?;
    if !ctype.to_lowercase().contains("png") && !data.starts_with(b"\x89PNG") {
        return Err(format!("接口没有返回图片：{ctype}"));
    }
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let label = map_label(map_id, name.trim());
    let path = out_dir.join(format!("{}_{map_id}.png", safe_filename(name.trim())));
    std::fs::write(&path, &data).map_err(|e| e.to_string())?;
    Ok((map_id, path, label))
}
