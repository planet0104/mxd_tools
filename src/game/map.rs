use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct PlatformSeg {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    #[serde(default)]
    pub id: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeSeg {
    pub x: f32,
    pub y1: f32,
    pub y2: f32,
    pub kind: String,
    #[serde(default = "default_rope_w")]
    pub width: f32,
}

fn default_rope_w() -> f32 {
    16.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct MobSpawn {
    pub mob_id: u32,
    pub x: f32,
    pub y: f32,
    pub walk_x1: f32,
    pub walk_x2: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SlopePoly {
    pub points: Vec<[f32; 2]>,
    #[serde(default)]
    pub source: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Portal {
    pub id: u32,
    pub name: String,
    pub x: f32,
    pub y: f32,
    pub to_map: u32,
    pub kind: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MapPlatformsFile {
    pub map_id: u32,
    pub image: String,
    pub image_size: [u32; 2],
    pub platforms: Vec<PlatformSeg>,
    #[serde(default)]
    pub ropes: Vec<RopeSeg>,
    #[serde(default)]
    pub spawns: Vec<MobSpawn>,
    #[serde(default)]
    pub slopes: Vec<SlopePoly>,
    #[serde(default)]
    pub portals: Vec<Portal>,
}

#[derive(Debug, Clone)]
pub struct GameMap {
    pub map_id: u32,
    pub image_path: String,
    pub width: f32,
    pub height: f32,
    pub platforms: Vec<PlatformSeg>,
    pub ropes: Vec<RopeSeg>,
    pub spawns: Vec<MobSpawn>,
    pub slopes: Vec<SlopePoly>,
    pub portals: Vec<Portal>,
}

impl GameMap {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("读取 {path:?}"))?;
        let file: MapPlatformsFile = serde_json::from_str(&text)?;
        let base = path.parent().context("地图目录")?;
        Ok(Self {
            map_id: file.map_id,
            image_path: base.join(&file.image).to_string_lossy().into(),
            width: file.image_size[0] as f32,
            height: file.image_size[1] as f32,
            platforms: file.platforms,
            ropes: file.ropes,
            spawns: file.spawns,
            slopes: file.slopes,
            portals: file.portals,
        })
    }

    /// 脚下最近可站立平台 y（图像坐标，向下为正）。
    pub fn ground_at(&self, x: f32, feet_y: f32, max_drop: f32) -> Option<f32> {
        let mut best: Option<f32> = None;
        for p in &self.platforms {
            let py = platform_y_at_x(p, x)?;
            if py < feet_y - 8.0 {
                continue;
            }
            if py > feet_y + max_drop {
                continue;
            }
            if best.map(|b| py < b).unwrap_or(true) {
                best = Some(py);
            }
        }
        for slope in &self.slopes {
            if let Some(py) = slope_ground_y(slope, x) {
                if py < feet_y - 8.0 {
                    continue;
                }
                if py > feet_y + max_drop {
                    continue;
                }
                if best.map(|b| py < b).unwrap_or(true) {
                    best = Some(py);
                }
            }
        }
        best
    }

    pub fn rope_at<'a>(&'a self, x: f32, y: f32) -> Option<&'a RopeSeg> {
        let mut best: Option<(&RopeSeg, f32)> = None;
        for r in &self.ropes {
            let half = r.width * 0.5 + crate::game::types::ROPE_GRAB_X;
            if x < r.x - half || x > r.x + half {
                continue;
            }
            let ymin = r.y1.min(r.y2);
            let ymax = r.y1.max(r.y2);
            if y < ymin - 32.0 || y > ymax + 24.0 {
                continue;
            }
            let d = (x - r.x).abs();
            if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                best = Some((r, d));
            }
        }
        best.map(|(r, _)| r)
    }

    pub fn portal_near(&self, x: f32, y: f32) -> Option<&Portal> {
        let mut best: Option<(&Portal, f32)> = None;
        for p in &self.portals {
            let dx = x - p.x;
            let dy = y - p.y;
            let d = (dx * dx + dy * dy).sqrt();
            if d > 64.0 {
                continue;
            }
            if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                best = Some((p, d));
            }
        }
        best.map(|(p, _)| p)
    }

    pub fn wall_at(&self, x: f32, feet_y: f32) -> bool {
        self.ground_at(x, feet_y - 40.0, 80.0).is_none()
    }

    pub fn max_stand_y(&self) -> f32 {
        let mut max_y = 0.0f32;
        for p in &self.platforms {
            if (p.y1 - p.y2).abs() < 3.0 {
                max_y = max_y.max((p.y1 + p.y2) * 0.5);
            }
        }
        for slope in &self.slopes {
            if let Some((a, b)) = slope_top_edge(&slope.points) {
                max_y = max_y.max(a[1].max(b[1]));
            }
        }
        max_y
    }

    /// 跌出最低平台后触发重生的 y 阈值。
    pub fn death_y(&self) -> f32 {
        self.max_stand_y() + 40.0
    }

    pub fn default_spawn(&self) -> (f32, f32) {
        if let Some(sp) = self.spawns.first() {
            return (sp.x, sp.y);
        }
        (400.0, self.max_stand_y())
    }
}

fn platform_y_at_x(p: &PlatformSeg, x: f32) -> Option<f32> {
    let (xmin, xmax) = if p.x1 <= p.x2 {
        (p.x1, p.x2)
    } else {
        (p.x2, p.x1)
    };
    if x < xmin - 4.0 || x > xmax + 4.0 {
        return None;
    }
    if (p.y1 - p.y2).abs() < 2.0 {
        return Some((p.y1 + p.y2) * 0.5);
    }
    let t = if (xmax - xmin).abs() < 0.01 {
        0.5
    } else {
        (x - xmin) / (xmax - xmin)
    };
    Some(p.y1 + (p.y2 - p.y1) * t)
}

fn slope_top_edge(points: &[[f32; 2]]) -> Option<([f32; 2], [f32; 2])> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len();
    let mut best: Option<([f32; 2], [f32; 2], f32)> = None;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        let mid_y = (a[1] + b[1]) * 0.5;
        if best.map(|(_, _, by)| mid_y < by).unwrap_or(true) {
            best = Some((a, b, mid_y));
        }
    }
    best.map(|(a, b, _)| (a, b))
}

fn slope_ground_y(slope: &SlopePoly, x: f32) -> Option<f32> {
    let (a, b) = slope_top_edge(&slope.points)?;
    let xmin = a[0].min(b[0]);
    let xmax = a[0].max(b[0]);
    if x < xmin - 4.0 || x > xmax + 4.0 {
        return None;
    }
    let t = if (xmax - xmin).abs() < 0.01 {
        0.5
    } else {
        (x - xmin) / (xmax - xmin)
    };
    let ax = if a[0] <= b[0] { a } else { b };
    let bx = if a[0] <= b[0] { b } else { a };
    Some(ax[1] + (bx[1] - ax[1]) * t)
}
