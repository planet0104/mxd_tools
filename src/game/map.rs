use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::game::types::SAME_LEVEL_TOL;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WalkAhead {
    /// 可走到同高度平台（含微台阶），值为站立 y
    SameLevel(f32),
    /// 前方无同高地面，但下方有更低平台 → 走出后下落
    Fall,
    /// 墙/高台侧面/虚空边缘 → 挡住
    Blocked,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlatformSeg {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    #[serde(default)]
    pub id: u32,
    /// WZ foothold layerId（地图层级）
    #[serde(default)]
    pub layer: i32,
    /// WZ foothold groupId（同层内连通组，对应 zM）
    #[serde(default)]
    pub group: i32,
    #[serde(default)]
    pub prev: u32,
    #[serde(default)]
    pub next: u32,
}

/// 脚底站立信息（含层级，用于侧墙只挡同组）
#[derive(Debug, Clone, Copy)]
pub struct StandInfo {
    pub y: f32,
    pub id: u32,
    pub layer: i32,
    pub group: i32,
}

/// 相对当前脚点可用的绳/梯方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClimbDir {
    /// 绳/梯底端紧邻当前层上方 → 上爬
    Up,
    /// 绳/梯顶端紧邻当前层且向下延伸 → 下爬
    Down,
}

/// 仅「紧邻当前层」的绳梯目标（不含远处上层绳）。
#[derive(Debug, Clone, Copy)]
pub struct ClimbHint {
    pub dx: f32,
    pub dir: ClimbDir,
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

    /// 脚下最近可站立平台（含 layer/group）。
    pub fn stand_at(&self, x: f32, feet_y: f32, max_drop: f32) -> Option<StandInfo> {
        let mut best: Option<StandInfo> = None;
        for p in &self.platforms {
            let Some(py) = platform_y_at_x(p, x) else {
                continue;
            };
            if py <= feet_y + SAME_LEVEL_TOL {
                continue;
            }
            if py > feet_y + max_drop {
                continue;
            }
            let info = StandInfo {
                y: py,
                id: p.id,
                layer: p.layer,
                group: p.group,
            };
            if best.map(|b| py < b.y).unwrap_or(true) {
                best = Some(info);
            }
        }
        for slope in &self.slopes {
            let Some(py) = slope_ground_y(slope, x) else {
                continue;
            };
            if py <= feet_y + SAME_LEVEL_TOL {
                continue;
            }
            if py > feet_y + max_drop {
                continue;
            }
            let info = StandInfo {
                y: py,
                id: 0,
                layer: -1,
                group: -1,
            };
            if best.map(|b| py < b.y).unwrap_or(true) {
                best = Some(info);
            }
        }
        best
    }

    /// 脚下最近可站立平台 y（图像坐标，向下为正）。
    pub fn ground_at(&self, x: f32, feet_y: f32, max_drop: f32) -> Option<f32> {
        self.stand_at(x, feet_y, max_drop).map(|s| s.y)
    }

    /// 正下方更低平台（严格 x，无 platform 边距），用于判 walk-off 坠落。
    pub fn ground_below_at(&self, x: f32, feet_y: f32, max_drop: f32) -> Option<f32> {
        let mut best: Option<f32> = None;
        for p in &self.platforms {
            let Some(py) = platform_y_at_x_strict(p, x) else {
                continue;
            };
            if py <= feet_y + SAME_LEVEL_TOL {
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
            let Some(py) = slope_ground_y(slope, x) else {
                continue;
            };
            if py <= feet_y + SAME_LEVEL_TOL || py > feet_y + max_drop {
                continue;
            }
            if best.map(|b| py < b).unwrap_or(true) {
                best = Some(py);
            }
        }
        best
    }

    /// 从 y_from 下落到 y_to 时穿过的平台顶（单向平台落地，任意层级）。
    pub fn land_at(&self, x: f32, y_from: f32, y_to: f32) -> Option<StandInfo> {
        if y_to < y_from - 0.5 {
            return None;
        }
        let mut best: Option<StandInfo> = None;
        let lo = y_from - 2.0;
        let hi = y_to + 2.0;
        for p in &self.platforms {
            let Some(py) = platform_y_at_x(p, x) else {
                continue;
            };
            if py < lo || py > hi {
                continue;
            }
            let info = StandInfo {
                y: py,
                id: p.id,
                layer: p.layer,
                group: p.group,
            };
            if best.map(|b| py < b.y).unwrap_or(true) {
                best = Some(info);
            }
        }
        for slope in &self.slopes {
            let Some(py) = slope_ground_y(slope, x) else {
                continue;
            };
            if py < lo || py > hi {
                continue;
            }
            let info = StandInfo {
                y: py,
                id: 0,
                layer: -1,
                group: -1,
            };
            if best.map(|b| py < b.y).unwrap_or(true) {
                best = Some(info);
            }
        }
        best
    }

    pub fn land_y(&self, x: f32, y_from: f32, y_to: f32) -> Option<f32> {
        self.land_at(x, y_from, y_to).map(|s| s.y)
    }

    /// 竖直墙 foothold：仅当 `fh_layer/fh_group` 与墙同组时阻挡（空中传 None=不挡侧墙）。
    pub fn resolve_wall_x(
        &self,
        x_from: f32,
        x_to: f32,
        y_top: f32,
        y_bottom: f32,
        fh: Option<(i32, i32)>,
    ) -> f32 {
        if (x_to - x_from).abs() < 0.01 {
            return x_to;
        }
        let Some((fl, fg)) = fh else {
            return x_to;
        };
        if fl < 0 {
            return x_to;
        }
        let mut x = x_to;
        let going_right = x_to > x_from;
        for p in &self.platforms {
            if p.layer != fl || p.group != fg {
                continue;
            }
            let (xmin, xmax) = if p.x1 <= p.x2 {
                (p.x1, p.x2)
            } else {
                (p.x2, p.x1)
            };
            let dx = xmax - xmin;
            let ymin = p.y1.min(p.y2);
            let ymax = p.y1.max(p.y2);
            if dx >= 8.0 || (ymax - ymin) < 8.0 {
                continue;
            }
            let wx = (p.x1 + p.x2) * 0.5;
            if ymax < y_top + 2.0 || ymin > y_bottom - 2.0 {
                continue;
            }
            if going_right && x_from <= wx && x >= wx {
                x = wx - 0.5;
            } else if !going_right && x_from >= wx && x <= wx {
                x = wx + 0.5;
            }
        }
        x
    }

    /// 前方水平移动；侧墙/高台阻挡只对当前 foothold 同 layer+group。
    pub fn walk_ahead(
        &self,
        x: f32,
        feet_y: f32,
        to_x: f32,
        fh: Option<(i32, i32)>,
    ) -> WalkAhead {
        use crate::game::types::{FALL_PROBE, SAME_LEVEL_TOL, WALL_HIT_H};

        let y_hi = feet_y - WALL_HIT_H;
        let blocked_x = self.resolve_wall_x(x, to_x, y_hi, feet_y - 2.0, fh);
        if (blocked_x - to_x).abs() > 0.1 {
            return WalkAhead::Blocked;
        }

        // 同层脚点必须用 strict_stand_at：stand_at 会跳过当前高度平台，
        // 平地会永远到不了 SameLevel，落地稳定后整图无法走路。
        if let Some(st) = self.strict_stand_at(to_x, feet_y) {
            return WalkAhead::SameLevel(st.y);
        }
        if let Some(st) = self.stand_at(to_x, feet_y, SAME_LEVEL_TOL) {
            if (st.y - feet_y).abs() <= SAME_LEVEL_TOL {
                return WalkAhead::SameLevel(st.y);
            }
        }

        // 同组内略高台面才挡；其它层级平台可从旁穿过，靠跳跃落上
        if let Some((fl, fg)) = fh {
            if fl >= 0 {
                if let Some(gy) =
                    self.surface_above_in_group(to_x, feet_y, WALL_HIT_H + SAME_LEVEL_TOL, fl, fg)
                {
                    if gy < feet_y - SAME_LEVEL_TOL {
                        return WalkAhead::Blocked;
                    }
                }
            }
        }

        // 前方无同层脚点：下方有可接住的平台则自然下落；否则视为地图/虚空边缘挡死。
        if self.ground_below_at(to_x, feet_y + 2.0, FALL_PROBE).is_some() {
            return WalkAhead::Fall;
        }

        WalkAhead::Blocked
    }

    /// 严格平台边（无 platform_y_at_x 的 ±4px 容差），用于地面行走判边。
    pub fn strict_stand_at(&self, x: f32, feet_y: f32) -> Option<StandInfo> {
        let mut best: Option<StandInfo> = None;
        for p in &self.platforms {
            let Some(py) = platform_y_at_x_strict(p, x) else {
                continue;
            };
            if py < feet_y - SAME_LEVEL_TOL || py > feet_y + SAME_LEVEL_TOL {
                continue;
            }
            let info = StandInfo {
                y: py,
                id: p.id,
                layer: p.layer,
                group: p.group,
            };
            if best.map(|b| py < b.y).unwrap_or(true) {
                best = Some(info);
            }
        }
        best
    }

    fn surface_above_in_group(
        &self,
        x: f32,
        feet_y: f32,
        max_up: f32,
        layer: i32,
        group: i32,
    ) -> Option<f32> {
        let mut best: Option<f32> = None;
        for p in &self.platforms {
            if p.layer != layer || p.group != group {
                continue;
            }
            let Some(py) = platform_y_at_x(p, x) else {
                continue;
            };
            if py >= feet_y - 2.0 || py < feet_y - max_up {
                continue;
            }
            if best.map(|b| py > b).unwrap_or(true) {
                best = Some(py);
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

    /// 攀爬到顶落脚：顶端平台通常略高于绳顶（y 更小），`stand_at` 只向下找会漏掉。
    pub fn stand_at_climb_exit(&self, x: f32, rope_top_y: f32) -> Option<StandInfo> {
        const TOL_UP: f32 = 40.0;
        const TOL_DOWN: f32 = 16.0;
        let mut best: Option<(f32, StandInfo)> = None;
        for p in &self.platforms {
            let Some(py) = platform_y_at_x(p, x) else {
                continue;
            };
            if py < rope_top_y - TOL_UP || py > rope_top_y + TOL_DOWN {
                continue;
            }
            let dist = (py - rope_top_y).abs();
            let info = StandInfo {
                y: py,
                id: p.id,
                layer: p.layer,
                group: p.group,
            };
            if best.map(|(bd, _)| dist < bd).unwrap_or(true) {
                best = Some((dist, info));
            }
        }
        best.map(|(_, info)| info)
    }

    /// 相对当前脚点：绳/梯底部紧邻上方 → 可上爬；顶部紧邻当前层且向下延伸 → 可下爬。
    /// 远处上层绳梯不算（下层跳不上去）。
    pub fn nearest_adjacent_climb(&self, feet_x: f32, feet_y: f32) -> Option<ClimbHint> {
        // 上爬：绳底在脚点上方可跳跃抓取范围内。
        const UP_REACH: f32 = 80.0;
        const UP_BELOW_SLACK: f32 = 28.0;
        // 下爬：绳顶贴在当前脚点附近，且绳身继续向下。
        const DOWN_TOP_SLACK: f32 = 40.0;
        const DOWN_MIN_LEN: f32 = 36.0;
        // 同距时优先上爬；下爬额外惩罚，避免近处下绳盖过稍远上绳（换层死循环主因）。
        const DOWN_DIST_PENALTY: f32 = 400.0;

        let mut best: Option<(f32, ClimbHint)> = None;
        for r in &self.ropes {
            let top = r.y1.min(r.y2);
            let bot = r.y1.max(r.y2);
            let dx = r.x - feet_x;
            let dist = dx.abs();

            let up_ok = bot <= feet_y + UP_BELOW_SLACK && feet_y - bot <= UP_REACH;
            if up_ok {
                let hint = ClimbHint {
                    dx,
                    dir: ClimbDir::Up,
                };
                let score = dist;
                if best.map(|(bd, _)| score < bd).unwrap_or(true) {
                    best = Some((score, hint));
                }
            }

            let down_ok = (top - feet_y).abs() <= DOWN_TOP_SLACK
                && bot >= feet_y + DOWN_MIN_LEN
                && top <= feet_y + DOWN_TOP_SLACK;
            if down_ok {
                let hint = ClimbHint {
                    dx,
                    dir: ClimbDir::Down,
                };
                let score = dist + DOWN_DIST_PENALTY;
                if best.map(|(bd, _)| score < bd).unwrap_or(true) {
                    best = Some((score, hint));
                }
            }
        }
        best.map(|(_, h)| h)
    }

    /// 当前层可跳上的最近更高平台（一层台阶，高度在跳跃可达内）。返回目标 x 相对脚点的 dx。
    pub fn nearest_step_up_dx(&self, feet_x: f32, feet_y: f32) -> Option<f32> {
        const MAX_UP: f32 = 80.0;
        const MIN_UP: f32 = 16.0;
        const MAX_APPROACH: f32 = 280.0;

        let mut best: Option<(f32, f32)> = None;
        for p in &self.platforms {
            // 近似水平平台：取两端 y 均作为站立高度。
            if (p.y1 - p.y2).abs() > 6.0 {
                continue;
            }
            let py = (p.y1 + p.y2) * 0.5;
            let rise = feet_y - py;
            if rise < MIN_UP || rise > MAX_UP {
                continue;
            }
            let x0 = p.x1.min(p.x2);
            let x1 = p.x1.max(p.x2);
            if x1 - x0 < 4.0 {
                continue;
            }
            // 目标点：已在平台水平范围内则原地跳；否则走向最近端。
            let target_x = if feet_x < x0 {
                x0 + 6.0
            } else if feet_x > x1 {
                x1 - 6.0
            } else {
                feet_x
            };
            let dx = target_x - feet_x;
            if dx.abs() > MAX_APPROACH {
                continue;
            }
            let score = dx.abs() + rise * 0.15;
            if best.map(|(bs, _)| score < bs).unwrap_or(true) {
                best = Some((score, dx));
            }
        }
        best.map(|(_, dx)| dx)
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

    pub fn wall_at(&self, x: f32, feet_y: f32, fh: Option<(i32, i32)>) -> bool {
        use crate::game::types::WALL_HIT_H;
        let y_hi = feet_y - WALL_HIT_H;
        let probed = self.resolve_wall_x(x, x + 1.0, y_hi, feet_y - 2.0, fh);
        (probed - (x + 1.0)).abs() > 0.1
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

    /// 跌出最低平台后触发救援的 y 阈值。
    pub fn death_y(&self) -> f32 {
        self.max_stand_y() + 40.0
    }

    /// 该竖直列是否仍有可站/可接住的脚点（含略高于脚底的平台，用于腾空判虚空）。
    /// 无落点则禁止水平飞入，避免跳上二台前掉进空隙。
    pub fn has_support_column(&self, x: f32, feet_y: f32) -> bool {
        let death = self.death_y();
        // 已远高于脚的平台视为错过；仅保留脚下可落 + 脚上方擦边吸附带。
        const SNAP_ABOVE: f32 = 28.0;
        for p in &self.platforms {
            let Some(py) = platform_y_at_x(p, x) else {
                continue;
            };
            if py < feet_y - SNAP_ABOVE || py > death {
                continue;
            }
            return true;
        }
        for slope in &self.slopes {
            let Some(py) = slope_ground_y(slope, x) else {
                continue;
            };
            if py < feet_y - SNAP_ABOVE || py > death {
                continue;
            }
            return true;
        }
        false
    }

    /// 脚略低于平台顶时吸附上去（擦边未 land_at 时防掉虚空）。
    pub fn ledge_snap_at(&self, x: f32, feet_y: f32) -> Option<StandInfo> {
        const SNAP_UP: f32 = 28.0;
        const SNAP_DOWN: f32 = 6.0;
        let mut best: Option<StandInfo> = None;
        for p in &self.platforms {
            let Some(py) = platform_y_at_x(p, x) else {
                continue;
            };
            // 平台顶在脚上方不远，或刚没过脚底一点点
            if py < feet_y - SNAP_UP || py > feet_y + SNAP_DOWN {
                continue;
            }
            let info = StandInfo {
                y: py,
                id: p.id,
                layer: p.layer,
                group: p.group,
            };
            // 取最接近脚底的平台顶
            if best.map(|b| (py - feet_y).abs() < (b.y - feet_y).abs()).unwrap_or(true) {
                best = Some(info);
            }
        }
        best
    }

    /// 虚空救援：找离 (x,y) 最近的可站立脚点。
    pub fn nearest_stand(&self, x: f32, y: f32) -> Option<(f32, StandInfo)> {
        let mut best: Option<(f32, f32, StandInfo)> = None;
        for p in &self.platforms {
            let (xmin, xmax) = if p.x1 <= p.x2 {
                (p.x1, p.x2)
            } else {
                (p.x2, p.x1)
            };
            if xmax - xmin < 8.0 || (p.y1 - p.y2).abs() >= 2.0 {
                continue;
            }
            let sx = x.clamp(xmin + 4.0, xmax - 4.0);
            let Some(py) = platform_y_at_x(p, sx) else {
                continue;
            };
            let dist = (sx - x).abs() + (py - y).abs() * 0.5;
            let info = StandInfo {
                y: py,
                id: p.id,
                layer: p.layer,
                group: p.group,
            };
            if best.map(|(bd, _, _)| dist < bd).unwrap_or(true) {
                best = Some((dist, sx, info));
            }
        }
        best.map(|(_, sx, info)| (sx, info))
    }

    /// 可站立平台（含斜坡顶边）在 x 方向上的范围；用于空中贴图边界，不外扩 padding。
    pub fn playable_x_bounds(&self) -> (f32, f32) {
        const BODY_INSET: f32 = 4.0;
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;

        for p in &self.platforms {
            let (xmin, xmax) = if p.x1 <= p.x2 {
                (p.x1, p.x2)
            } else {
                (p.x2, p.x1)
            };
            if xmax - xmin < 8.0 {
                continue;
            }
            if (p.y1 - p.y2).abs() >= 2.0 {
                continue;
            }
            min_x = min_x.min(xmin);
            max_x = max_x.max(xmax);
        }

        for slope in &self.slopes {
            let Some((a, b)) = slope_top_edge(&slope.points) else {
                continue;
            };
            let xmin = a[0].min(b[0]);
            let xmax = a[0].max(b[0]);
            if xmax - xmin < 8.0 {
                continue;
            }
            min_x = min_x.min(xmin);
            max_x = max_x.max(xmax);
        }

        if min_x > max_x {
            return (16.0, self.width - 16.0);
        }
        (min_x + BODY_INSET, max_x - BODY_INSET)
    }

    /// 当前高度上、包含 x 的连续水平平台区间（不含怪物巡逻 inset）。
    pub fn platform_span_at(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        for (lo, hi) in self.horizontal_spans_at(y) {
            if x >= lo - 4.0 && x <= hi + 4.0 {
                return Some((lo, hi));
            }
        }
        None
    }

    fn horizontal_spans_at(&self, y: f32) -> Vec<(f32, f32)> {
        const Y_TOL: f32 = 4.0;
        const GAP: f32 = 8.0;

        let mut spans: Vec<(f32, f32)> = Vec::new();
        for p in &self.platforms {
            let (xmin, xmax) = if p.x1 <= p.x2 {
                (p.x1, p.x2)
            } else {
                (p.x2, p.x1)
            };
            if xmax - xmin < 8.0 {
                continue;
            }
            if (p.y1 - p.y2).abs() >= 2.0 {
                continue;
            }
            let py = (p.y1 + p.y2) * 0.5;
            if (py - y).abs() > Y_TOL {
                continue;
            }
            spans.push((xmin, xmax));
        }
        spans.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut merged: Vec<(f32, f32)> = Vec::new();
        for (lo, hi) in spans {
            if let Some(last) = merged.last_mut() {
                if lo <= last.1 + GAP {
                    last.1 = last.1.max(hi);
                    continue;
                }
            }
            merged.push((lo, hi));
        }
        merged
    }

    pub fn default_spawn(&self) -> (f32, f32) {
        if let Some(sp) = self.spawns.first() {
            return (sp.x, sp.y);
        }
        (400.0, self.max_stand_y())
    }

    /// 刷怪点所在高度上、包含 x 的连续水平平台巡逻区间。
    pub fn walk_range_at(&self, x: f32, y: f32) -> (f32, f32) {
        const EDGE_PAD: f32 = 8.0;
        const MIN_HALF: f32 = 20.0;

        for (lo, hi) in self.horizontal_spans_at(y) {
            if x >= lo - 4.0 && x <= hi + 4.0 {
                let mut w1 = lo + EDGE_PAD;
                let mut w2 = hi - EDGE_PAD;
                if w2 - w1 < MIN_HALF * 2.0 {
                    let mid = (lo + hi) * 0.5;
                    w1 = mid - MIN_HALF;
                    w2 = mid + MIN_HALF;
                }
                return (w1, w2);
            }
        }
        (x - MIN_HALF, x + MIN_HALF)
    }
}

fn platform_y_at_x_strict(p: &PlatformSeg, x: f32) -> Option<f32> {
    let (xmin, xmax) = if p.x1 <= p.x2 {
        (p.x1, p.x2)
    } else {
        (p.x2, p.x1)
    };
    if (xmax - xmin) < 8.0 {
        return None;
    }
    if x < xmin || x > xmax {
        return None;
    }
    if (p.y1 - p.y2).abs() < 2.0 {
        return Some((p.y1 + p.y2) * 0.5);
    }
    let t = (x - xmin) / (xmax - xmin);
    let (ya, yb) = if p.x1 <= p.x2 {
        (p.y1, p.y2)
    } else {
        (p.y2, p.y1)
    };
    Some(ya + (yb - ya) * t)
}

fn platform_y_at_x(p: &PlatformSeg, x: f32) -> Option<f32> {
    let (xmin, xmax) = if p.x1 <= p.x2 {
        (p.x1, p.x2)
    } else {
        (p.x2, p.x1)
    };
    // 过短/近乎竖直的线段不当作站立面（避免墙体中点被当成地面）
    if (xmax - xmin) < 8.0 {
        return None;
    }
    if x < xmin - 4.0 || x > xmax + 4.0 {
        return None;
    }
    if (p.y1 - p.y2).abs() < 2.0 {
        return Some((p.y1 + p.y2) * 0.5);
    }
    let t = (x - xmin) / (xmax - xmin);
    let (ya, yb) = if p.x1 <= p.x2 {
        (p.y1, p.y2)
    } else {
        (p.y2, p.y1)
    };
    Some(ya + (yb - ya) * t)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::load_default_map;

    #[test]
    fn adjacent_climb_ignores_upper_rope_from_spawn_ground() {
        let map = load_default_map().expect("default map");
        // 出生地 y=1225：绳 488 底在 1068，非紧邻，不应给出上爬目标。
        let hint = map.nearest_adjacent_climb(416.0, 1225.0);
        assert!(
            hint.is_none()
                || hint.map(|h| (h.dx + 416.0 - 488.0).abs() > 1.0).unwrap_or(true),
            "spawn ground must not chase rope@488"
        );
        // 右岛梯子底 1191，脚点 1225 → 可上爬。
        let ladder = map.nearest_adjacent_climb(1477.0, 1225.0).expect("ladder");
        assert_eq!(ladder.dir, ClimbDir::Up);
        assert!(ladder.dx.abs() < 2.0);
    }

    #[test]
    fn adjacent_climb_prefers_up_rope_over_nearer_down() {
        let map = load_default_map().expect("default map");
        // y=865：近处绳 1020 可下爬，稍远绳 1092 可上爬 → 必须优先上爬。
        let hint = map
            .nearest_adjacent_climb(961.0, 865.0)
            .expect("should see climb");
        assert_eq!(hint.dir, ClimbDir::Up, "must prefer Up over nearer Down");
        assert!(
            hint.dx > 50.0,
            "should walk right toward rope@1092, dx={}",
            hint.dx
        );
    }

    #[test]
    fn step_up_from_spawn_finds_mid_ledge() {
        let map = load_default_map().expect("default map");
        let dx = map
            .nearest_step_up_dx(416.0, 1225.0)
            .expect("spawn should see 1165 ledges");
        // 左侧 218–308 或右侧 617–656 的 1165 台阶。
        assert!(dx.abs() > 10.0, "should walk toward a ledge, dx={dx}");
    }

    #[test]
    fn playable_x_bounds_inside_image() {
        let map = load_default_map().expect("default map");
        let (lo, hi) = map.playable_x_bounds();
        assert!(lo < hi);
        assert!((lo - 81.0).abs() < 1.0, "left bound should be ~81, got {lo}");
        assert!(hi < map.width);
        assert!(hi < map.width - 16.0);
    }

    #[test]
    fn left_ground_edge_blocks_walk_off() {
        let map = load_default_map().expect("default map");
        let y = 1225.0;
        let span = map.platform_span_at(100.0, y).expect("ground span");
        assert!((span.0 - 77.0).abs() < 1.0, "leftmost ground x=77, got {}", span.0);
        let ahead = map.walk_ahead(78.0, y, 76.0, Some((2, 0)));
        assert!(
            matches!(ahead, WalkAhead::Blocked),
            "walking past left map edge should block, got {ahead:?}"
        );
    }

    #[test]
    fn upper_platform_edge_allows_fall_when_ground_below() {
        let map = load_default_map().expect("default map");
        // 上层小平台右缘：下方有更低地面时应允许走出去下落
        let y = 565.0;
        let x = 1826.0;
        let ahead = map.walk_ahead(x - 2.0, y, x + 6.0, Some((0, 33)));
        assert!(
            matches!(ahead, WalkAhead::Fall),
            "walk off upper ledge with ground below should Fall, got {ahead:?}"
        );
        assert!(
            map.ground_below_at(x + 6.0, y + 2.0, 720.0).is_some(),
            "fixture expects catchable ground below"
        );
    }

    #[test]
    fn ground_right_edge_blocks_when_no_ground_below() {
        let map = load_default_map().expect("default map");
        let y = 1225.0;
        let span = map.platform_span_at(400.0, y).expect("ground span");
        let ahead = map.walk_ahead(span.1 - 5.0, y, span.1 + 5.0, Some((2, 0)));
        assert!(
            matches!(ahead, WalkAhead::Blocked),
            "ground floor void edge should block, got {ahead:?}"
        );
    }

    #[test]
    fn void_gap_between_floor_and_upper_has_no_support() {
        let map = load_default_map().expect("default map");
        // 一层右缘 758，二台从 758@1165 起；脚已落到 1200 且 x>758 → 虚空列
        assert!(
            !map.has_support_column(780.0, 1200.0),
            "past first-floor end and below upper deck must be unsupported"
        );
        assert!(
            map.has_support_column(740.0, 1180.0),
            "over first floor should still have catchable ground"
        );
        assert!(
            map.has_support_column(780.0, 1160.0),
            "at upper deck height should have support"
        );
    }

    #[test]
    fn ledge_snap_pulls_feet_slightly_below_deck() {
        let map = load_default_map().expect("default map");
        let snap = map
            .ledge_snap_at(780.0, 1175.0)
            .expect("should snap onto y=1165 deck");
        assert!(
            (snap.y - 1165.0).abs() < 1.0,
            "snap y={}, want ~1165",
            snap.y
        );
    }
}
