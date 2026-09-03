use std::path::Path;

use anyhow::{Context, Result};
use rand::Rng;
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
    pub fn walk_ahead(&self, x: f32, feet_y: f32, to_x: f32, fh: Option<(i32, i32)>) -> WalkAhead {
        use crate::game::types::{FALL_PROBE, SAME_LEVEL_TOL, WALL_HIT_H};

        let y_hi = feet_y - WALL_HIT_H;
        let blocked_x = self.resolve_wall_x(x, to_x, y_hi, feet_y - 2.0, fh);
        if (blocked_x - to_x).abs() > 0.1 {
            return WalkAhead::Blocked;
        }

        // 同层脚点必须用 strict_stand_at：stand_at 会跳过当前高度平台，
        // 平地会永远到不了 SameLevel，落地稳定后整图无法走路。
        // 离散邻台：下行最多约 4px（避免粘到应下落的更低台）。
        // 斜坡 foothold：允许 SAME_LEVEL_TOL 内下行，否则每帧走坡会被判 Fall。
        if let Some(st) = self.strict_stand_at(to_x, feet_y) {
            let max_down = if self.platform_is_slope(st.id) {
                SAME_LEVEL_TOL
            } else {
                4.0
            };
            if st.y <= feet_y + max_down && feet_y - st.y <= SAME_LEVEL_TOL {
                return WalkAhead::SameLevel(st.y);
            }
        }
        if let Some(st) = self.stand_at(to_x, feet_y, SAME_LEVEL_TOL) {
            let max_down = if self.platform_is_slope(st.id) {
                SAME_LEVEL_TOL
            } else {
                4.0
            };
            if st.y <= feet_y + max_down && (feet_y - st.y) <= SAME_LEVEL_TOL {
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
        if self
            .ground_below_at(to_x, feet_y + 2.0, FALL_PROBE)
            .is_some()
        {
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

    fn platform_is_slope(&self, id: u32) -> bool {
        // 只认「可走坡道」（足够宽）；WZ 里 10px 级微斜连段不能放宽下行，否则会粘到应下落的邻台。
        self.platforms.iter().find(|p| p.id == id).is_some_and(|p| {
            (p.y1 - p.y2).abs() >= 2.0 && (p.x1 - p.x2).abs() >= 40.0
        })
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
            if best
                .map(|b| (py - feet_y).abs() < (b.y - feet_y).abs())
                .unwrap_or(true)
            {
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
            if (p.y1 - p.y2).abs() < 2.0 {
                let py = (p.y1 + p.y2) * 0.5;
                if (py - y).abs() > Y_TOL {
                    continue;
                }
                spans.push((xmin, xmax));
            } else {
                // 可走斜坡（≥40px 宽）：脚点 y 落在坡段高度带内时并入水平 span，
                // 否则平地右缘会把 can_exit 判死，必须跳才能走上/走下坡。
                // 过短微斜连段不并入，避免把应下落的缝粘成平地。
                if (p.x1 - p.x2).abs() < 40.0 {
                    continue;
                }
                let ymin = p.y1.min(p.y2);
                let ymax = p.y1.max(p.y2);
                if y < ymin - Y_TOL || y > ymax + Y_TOL {
                    continue;
                }
                spans.push((xmin, xmax));
            }
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

    /// 多平台出生候选：按高度带合并水平脚点，抽样若干站立位置。
    pub fn player_spawn_candidates(&self) -> Vec<(f32, f32)> {
        use crate::game::types::{
            PLAYER_SPAWN_EDGE_PAD, PLAYER_SPAWN_MIN_PLATFORM_W, PLAYER_SPAWN_Y_BAND,
        };
        use std::collections::BTreeMap;

        let mut bands: BTreeMap<i32, Vec<(f32, f32, f32)>> = BTreeMap::new();
        for p in &self.platforms {
            let (xmin, xmax) = if p.x1 <= p.x2 {
                (p.x1, p.x2)
            } else {
                (p.x2, p.x1)
            };
            if xmax - xmin < PLAYER_SPAWN_MIN_PLATFORM_W {
                continue;
            }
            if (p.y1 - p.y2).abs() >= 2.0 {
                continue;
            }
            let py = (p.y1 + p.y2) * 0.5;
            let band = (py / PLAYER_SPAWN_Y_BAND).round() as i32;
            bands.entry(band).or_default().push((xmin, xmax, py));
        }

        let mut out: Vec<(f32, f32)> = Vec::new();
        for spans in bands.values() {
            let mut merged: Vec<(f32, f32, f32)> = Vec::new();
            let mut sorted = spans.clone();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for (lo, hi, py) in sorted {
                if let Some(last) = merged.last_mut() {
                    if lo <= last.1 + 8.0 && (py - last.2).abs() <= 4.0 {
                        last.1 = last.1.max(hi);
                        continue;
                    }
                }
                merged.push((lo, hi, py));
            }
            for (lo, hi, py) in merged {
                let w = hi - lo;
                if w < PLAYER_SPAWN_MIN_PLATFORM_W {
                    continue;
                }
                let pad = PLAYER_SPAWN_EDGE_PAD.min(w * 0.25);
                let a = lo + pad;
                let b = hi - pad;
                if b <= a {
                    continue;
                }
                let mid = (a + b) * 0.5;
                out.push((mid, py));
                if w >= PLAYER_SPAWN_MIN_PLATFORM_W * 2.0 {
                    out.push((a + (b - a) * 0.25, py));
                    out.push((a + (b - a) * 0.75, py));
                }
            }
        }

        for sp in &self.spawns {
            let (w1, w2) = self.walk_range_at(sp.x, sp.y);
            let x = sp.x.clamp(w1, w2);
            out.push((x, sp.y));
        }

        out.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        });
        out.dedup_by(|a, b| (a.0 - b.0).abs() < 12.0 && (a.1 - b.1).abs() < 8.0);
        if out.is_empty() {
            out.push(self.default_spawn());
        }
        out
    }

    /// 按 RNG 从候选点随机选出生位置（训练泛化用）。
    pub fn random_player_spawn<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> (f32, f32) {
        let cands = self.player_spawn_candidates();
        let i = rng.gen_range(0..cands.len());
        let (x, y) = cands[i];
        if let Some(st) = self.stand_at(x, y + 40.0, 120.0) {
            (x, st.y)
        } else if let Some((sx, st)) = self.nearest_stand(x, y) {
            (sx, st.y)
        } else {
            (x, y)
        }
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
    use rand::SeedableRng;

    #[test]
    fn adjacent_climb_ignores_upper_rope_from_spawn_ground() {
        let map = load_default_map().expect("default map");
        // 出生地 y=1225：绳 488 底在 1068，非紧邻，不应给出上爬目标。
        let hint = map.nearest_adjacent_climb(416.0, 1225.0);
        assert!(
            hint.is_none()
                || hint
                    .map(|h| (h.dx + 416.0 - 488.0).abs() > 1.0)
                    .unwrap_or(true),
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
    fn player_spawn_candidates_cover_multiple_altitudes() {
        let map = load_default_map().expect("default map");
        let cands = map.player_spawn_candidates();
        assert!(cands.len() >= 4, "expected several spawn candidates, got {}", cands.len());
        let mut ys: Vec<i32> = cands
            .iter()
            .map(|(_, y)| (*y / 40.0).round() as i32)
            .collect();
        ys.sort_unstable();
        ys.dedup();
        assert!(
            ys.len() >= 3,
            "candidates should span multiple platform heights, bands={ys:?}"
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let a = map.random_player_spawn(&mut rng);
        assert!(
            map.strict_stand_at(a.0, a.1).is_some()
                || map.stand_at(a.0, a.1 + 40.0, 120.0).is_some(),
            "spawn ({}, {}) should be standable",
            a.0,
            a.1
        );
    }

    #[test]
    fn playable_x_bounds_inside_image() {
        let map = load_default_map().expect("default map");
        let (lo, hi) = map.playable_x_bounds();
        assert!(lo < hi);
        assert!(
            (lo - 81.0).abs() < 1.0,
            "left bound should be ~81, got {lo}"
        );
        assert!(hi < map.width);
        assert!(hi < map.width - 16.0);
    }

    #[test]
    fn left_upper_small_plats_15_12_16_walkable() {
        let map = load_default_map().expect("default map");
        let span = map.platform_span_at(250.0, 1165.0);
        eprintln!("span at 15: {span:?}");
        assert!(
            span.is_some_and(|(lo, hi)| lo <= 220.0 && hi >= 300.0),
            "15/12/16 should merge into one span, got {span:?}"
        );
        for (x, tx) in [(250.0_f32, 270.0), (270.0, 300.0), (240.0, 255.0)] {
            let ahead = map.walk_ahead(x, 1165.0, tx, Some((0, 1)));
            eprintln!("{x}->{tx}: {ahead:?}");
            assert!(
                matches!(ahead, WalkAhead::SameLevel(_)),
                "same-level walk {x}->{tx} got {ahead:?}"
            );
        }
    }

    #[test]
    fn flat_to_slope_22_span_merges_and_walkable() {
        let map = load_default_map().expect("default map");
        // 19/21/20 @1105 + 坡 22 → 水平 span 应连到坡上，否则右缘 clamp 卡死。
        let span = map.platform_span_at(500.0, 1105.0);
        eprintln!("span at 20: {span:?}");
        assert!(
            span.is_some_and(|(lo, hi)| lo <= 320.0 && hi >= 600.0),
            "flat+slope should merge past 527, got {span:?}"
        );
        for (x, tx) in [(510.0_f32, 528.0), (528.0, 535.0), (550.0, 560.0)] {
            let y = if x <= 527.0 {
                1105.0
            } else {
                1105.0 + (x - 527.0) / 90.0 * 60.0
            };
            let ahead = map.walk_ahead(x, y, tx, Some((0, 1)));
            eprintln!("{x}/{y:.1}->{tx}: {ahead:?}");
            assert!(
                matches!(ahead, WalkAhead::SameLevel(_)),
                "walk on/onto slope 22 {x}->{tx} got {ahead:?}"
            );
        }
    }

    #[test]
    fn sim_walks_down_slope_22_without_jump() {
        use crate::game::input::InputFrame;
        use crate::game::sim::GameSim;
        let map = load_default_map().expect("default map");
        let mut sim = GameSim::new_preview(map, 1);
        {
            let p = &mut sim.state.player;
            p.x = 500.0;
            p.y = 1105.0;
            p.on_ground = true;
            p.climbing = false;
            p.fh_id = 20;
            p.fh_layer = 0;
            p.fh_group = 1;
            p.vx = 0.0;
            p.vy = 0.0;
        }
        let right = InputFrame {
            right: true,
            ..InputFrame::default()
        };
        for _ in 0..180 {
            sim.tick(&right);
        }
        let p = &sim.state.player;
        eprintln!("after walk slope: x={:.1} y={:.1} gnd={}", p.x, p.y, p.on_ground);
        assert!(p.x > 560.0, "should walk onto slope past 527, x={}", p.x);
        assert!(p.y > 1115.0, "should descend slope, y={}", p.y);
        assert!(p.on_ground, "should stay grounded walking slope");
    }

    #[test]
    fn upper_slope_102_merges_with_adjacent_flat() {
        let map = load_default_map().expect("default map");
        // 102: 527..617 y 865→805；左端应与左侧平地在 865 合并
        let span = map.platform_span_at(540.0, 860.0);
        eprintln!("span near slope 102: {span:?}");
        assert!(span.is_some(), "should stand on slope 102 band");
        let ahead = map.walk_ahead(530.0, 863.0, 540.0, Some((2, 22)));
        eprintln!("on 102: {ahead:?}");
        assert!(
            matches!(ahead, WalkAhead::SameLevel(_)),
            "walk along slope 102 got {ahead:?}"
        );
    }

    #[test]
    fn small_plat_15_can_fall_left_to_ground() {
        let map = load_default_map().expect("default map");
        // 15 左缘外：下方大台 1225，应 Fall 而非 Blocked
        let ahead = map.walk_ahead(220.0, 1165.0, 210.0, Some((0, 1)));
        let below = map.ground_below_at(210.0, 1167.0, 720.0);
        eprintln!("15 left exit: {ahead:?} below={below:?}");
        assert!(
            matches!(ahead, WalkAhead::Fall),
            "left off 15 should Fall onto ground, got {ahead:?}"
        );
    }

    #[test]
    fn left_ground_edge_blocks_walk_off() {
        let map = load_default_map().expect("default map");
        let y = 1225.0;
        let span = map.platform_span_at(100.0, y).expect("ground span");
        assert!(
            (span.0 - 77.0).abs() < 1.0,
            "leftmost ground x=77, got {}",
            span.0
        );
        let ahead = map.walk_ahead(78.0, y, 76.0, Some((2, 0)));
        assert!(
            matches!(ahead, WalkAhead::Blocked),
            "walking past left map edge should block, got {ahead:?}"
        );
    }

    #[test]
    fn small_ledge_133_left_allows_fall_to_big_platform() {
        let map = load_default_map().expect("default map");
        // 小平台 133 @ y=876；左侧无同组更高挡板，下方有大台 y=925 → 应 Fall。
        let cases = [
            (1648.0_f32, 876.0, 1636.0),
            (1636.0, 879.0, 1624.0),
            (1634.0, 882.0, 1620.0),
            (1660.0, 876.0, 1628.0),
        ];
        for (x, y, tx) in cases {
            let ahead = map.walk_ahead(x, y, tx, Some((1, 19)));
            let below = map.ground_below_at(tx, y + 2.0, 720.0);
            eprintln!("x={x} -> {tx}: ahead={ahead:?} below={below:?}");
            assert!(
                matches!(ahead, WalkAhead::Fall | WalkAhead::SameLevel(_)),
                "left off small ledge should Fall/SameLevel, got {ahead:?} at x={x}->{tx}"
            );
        }
    }

    #[test]
    fn isolated_ledge_135_left_allows_fall() {
        let map = load_default_map().expect("default map");
        // 135 @ 1695-1732 y=868，左侧空隙后下方是大台；不应粘到更低的 133/134。
        for (x, tx) in [(1700.0_f32, 1688.0), (1697.0, 1680.0), (1696.0, 1690.0)] {
            let ahead = map.walk_ahead(x, 868.0, tx, Some((1, 19)));
            eprintln!("135 {x}->{tx}: {ahead:?}");
            assert!(
                matches!(ahead, WalkAhead::Fall),
                "135 left should Fall, got {ahead:?}"
            );
        }
    }

    #[test]
    fn sim_can_walk_off_133_left_onto_big_floor() {
        use crate::game::input::InputFrame;
        use crate::game::sim::GameSim;
        let map = load_default_map().expect("default map");
        let mut sim = GameSim::new_preview(map, 1);
        {
            let p = &mut sim.state.player;
            p.x = 1640.0;
            p.y = 879.0;
            p.on_ground = true;
            p.climbing = false;
            p.fh_id = 132;
            p.fh_layer = 1;
            p.fh_group = 19;
            p.vx = 0.0;
            p.vy = 0.0;
        }
        let left = InputFrame {
            left: true,
            ..InputFrame::default()
        };
        let y0 = sim.state.player.y;
        let x0 = sim.state.player.x;
        for _ in 0..90 {
            sim.tick(&left);
        }
        eprintln!(
            "after walk: x={} y={} gnd={} fh={}",
            sim.state.player.x,
            sim.state.player.y,
            sim.state.player.on_ground,
            sim.state.player.fh_id
        );
        assert!(
            sim.state.player.x < x0 - 8.0 || sim.state.player.y > y0 + 20.0,
            "should walk left off ledge or fall down, x0={x0} x={} y0={y0} y={}",
            sim.state.player.x,
            sim.state.player.y
        );
        assert!(
            sim.state.player.y > 900.0,
            "should land on big platform ~925, y={}",
            sim.state.player.y
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
