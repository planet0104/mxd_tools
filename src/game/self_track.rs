//! 运动确认 + 持续跟踪自身玩家（YOLO「玩家」框残差锁定）。
//!
//! `residual_x = player_screen_dx - floor_dx`：
//! - 镜头跟随（地图中央）：自身屏坐标几乎不动，地板反滚 → residual ≈ 世界位移
//! - 镜头锁边（地图边缘）：地板不动，自身屏移 → residual ≈ 世界位移

use crate::game::input::InputFrame;
use crate::yolo::Detection;

const PLAYER_LABEL: &str = "玩家";
const FLOOR_LABEL: &str = "地板";

const RESIDUAL_DEADZONE: f32 = 2.5;
const ASSOC_MAX_DIST: f32 = 120.0;
const MISS_REUSE_MAX: u8 = 4;
const CONTRADICTION_LIMIT: u8 = 3;
const PROBE_HALF_FRAMES: u32 = 4;
const PROBE_ROUNDS: u32 = 2;
const LOCK_SCORE_MIN: f32 = 2.5;
const LOCK_SCORE_MARGIN: f32 = 1.0;

/// 自身脚点与检测框（预览叠层用）。
#[derive(Debug, Clone)]
pub struct SelfPlayerHit {
    pub x: f32,
    pub y: f32,
    pub conf: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl SelfPlayerHit {
    pub fn from_det(d: &Detection) -> Self {
        Self {
            x: (d.x1 + d.x2) * 0.5,
            y: d.y2,
            conf: d.conf,
            x1: d.x1,
            y1: d.y1,
            x2: d.x2,
            y2: d.y2,
        }
    }

    pub fn foot(&self) -> (f32, f32) {
        (self.x, self.y)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackMode {
    Unlocked,
    Probing,
    Locked,
}

impl TrackMode {
    pub fn label(self) -> &'static str {
        match self {
            TrackMode::Unlocked => "UNLOCKED",
            TrackMode::Probing => "PROBE",
            TrackMode::Locked => "LOCKED",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraRegime {
    Unknown,
    Follow,
    Clamp,
}

impl CameraRegime {
    pub fn label(self) -> &'static str {
        match self {
            CameraRegime::Unknown => "unknown",
            CameraRegime::Follow => "follow",
            CameraRegime::Clamp => "clamp",
        }
    }
}

#[derive(Debug, Clone)]
struct CandPrev {
    x: f32,
    y: f32,
    score: f32,
}

#[derive(Debug)]
pub struct SelfTracker {
    mode: TrackMode,
    hit: Option<SelfPlayerHit>,
    miss_frames: u8,
    contradict_frames: u8,
    prev_players: Vec<(f32, f32)>,
    prev_floors: Vec<(f32, f32)>,
    floor_dx: f32,
    regime: CameraRegime,
    probe_frame: u32,
    probe_scores: Vec<CandPrev>,
}

impl Default for SelfTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfTracker {
    pub fn new() -> Self {
        Self {
            mode: TrackMode::Unlocked,
            hit: None,
            miss_frames: 0,
            contradict_frames: 0,
            prev_players: Vec::new(),
            prev_floors: Vec::new(),
            floor_dx: 0.0,
            regime: CameraRegime::Unknown,
            probe_frame: 0,
            probe_scores: Vec::new(),
        }
    }

    pub fn mode(&self) -> TrackMode {
        self.mode
    }

    pub fn regime(&self) -> CameraRegime {
        self.regime
    }

    pub fn floor_dx(&self) -> f32 {
        self.floor_dx
    }

    pub fn hit(&self) -> Option<&SelfPlayerHit> {
        self.hit.as_ref()
    }

    pub fn needs_probe(&self) -> bool {
        matches!(self.mode, TrackMode::Unlocked | TrackMode::Probing)
    }

    /// 探测阶段应下发的左右键（覆盖 NavBot）。
    pub fn probe_input(&self) -> InputFrame {
        let mut input = InputFrame::default();
        let half = PROBE_HALF_FRAMES.max(1);
        let phase = (self.probe_frame / half) % 2;
        if phase == 0 {
            input.right = true;
        } else {
            input.left = true;
        }
        input
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// 用本帧 YOLO 与已下发水平指令更新跟踪；返回用于编 obs 的自身（含短暂漏检复用）。
    pub fn update(
        &mut self,
        detections: &[Detection],
        commanded_dx: f32,
        min_player_conf: f32,
    ) -> Option<SelfPlayerHit> {
        let players: Vec<&Detection> = detections
            .iter()
            .filter(|d| d.label == PLAYER_LABEL && d.conf >= min_player_conf)
            .collect();
        let floors: Vec<(f32, f32)> = detections
            .iter()
            .filter(|d| d.label == FLOOR_LABEL)
            .map(|d| ((d.x1 + d.x2) * 0.5, (d.y1 + d.y2) * 0.5))
            .collect();

        self.floor_dx = estimate_shift_dx(&self.prev_floors, &floors).unwrap_or(0.0);
        if self.floor_dx.abs() < 1.0 {
            self.floor_dx = 0.0;
        }

        let cmd = sign_with_deadzone(commanded_dx, 0.1);

        match self.mode {
            TrackMode::Unlocked => {
                self.mode = TrackMode::Probing;
                self.probe_frame = 0;
                self.probe_scores.clear();
                self.run_probe_frame(&players, cmd);
            }
            TrackMode::Probing => {
                self.run_probe_frame(&players, cmd);
                let total = PROBE_HALF_FRAMES * 2 * PROBE_ROUNDS;
                if self.probe_frame >= total {
                    self.finish_probe(&players);
                }
            }
            TrackMode::Locked => {
                self.run_locked_frame(&players, cmd);
            }
        }

        self.prev_floors = floors;
        self.prev_players = players
            .iter()
            .map(|d| ((d.x1 + d.x2) * 0.5, d.y2))
            .collect();

        self.hit.clone()
    }

    fn run_probe_frame(&mut self, players: &[&Detection], cmd: f32) {
        self.probe_frame = self.probe_frame.saturating_add(1);
        if players.is_empty() {
            return;
        }

        let mut next_scores: Vec<CandPrev> = players
            .iter()
            .map(|d| {
                let (x, y) = ((d.x1 + d.x2) * 0.5, d.y2);
                CandPrev { x, y, score: 0.0 }
            })
            .collect();

        for (i, d) in players.iter().enumerate() {
            let (x, y) = ((d.x1 + d.x2) * 0.5, d.y2);
            let screen_dx = nearest_dx(&self.prev_players, x, y).unwrap_or(0.0);
            let residual = screen_dx - self.floor_dx;
            self.update_regime(screen_dx);
            if cmd != 0.0 {
                let r = sign_with_deadzone(residual, RESIDUAL_DEADZONE);
                if r == cmd {
                    next_scores[i].score += residual.abs().max(1.0);
                } else if r == -cmd {
                    next_scores[i].score -= 0.5;
                }
            }
            if let Some((j, _)) = nearest_index(
                &self.probe_scores.iter().map(|c| (c.x, c.y)).collect::<Vec<_>>(),
                x,
                y,
                ASSOC_MAX_DIST,
            ) {
                next_scores[i].score += self.probe_scores[j].score;
            }
        }

        self.probe_scores = next_scores;
    }

    fn finish_probe(&mut self, players: &[&Detection]) {
        let Some((best_i, best)) = self
            .probe_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
        else {
            self.mode = TrackMode::Unlocked;
            self.hit = None;
            return;
        };

        let mut second = f32::NEG_INFINITY;
        for (i, c) in self.probe_scores.iter().enumerate() {
            if i != best_i {
                second = second.max(c.score);
            }
        }

        if best.score < LOCK_SCORE_MIN || best.score < second + LOCK_SCORE_MARGIN {
            self.mode = TrackMode::Probing;
            self.probe_frame = 0;
            self.probe_scores.clear();
            return;
        }

        if let Some(d) = players.get(best_i).copied().or_else(|| {
            players.iter().copied().min_by(|a, b| {
                let da = ((a.x1 + a.x2) * 0.5 - best.x).hypot(a.y2 - best.y);
                let db = ((b.x1 + b.x2) * 0.5 - best.x).hypot(b.y2 - best.y);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
        }) {
            self.hit = Some(SelfPlayerHit::from_det(d));
            self.mode = TrackMode::Locked;
            self.miss_frames = 0;
            self.contradict_frames = 0;
            self.probe_scores.clear();
        } else {
            self.mode = TrackMode::Unlocked;
            self.hit = None;
        }
    }

    fn run_locked_frame(&mut self, players: &[&Detection], cmd: f32) {
        let Some(prev) = self.hit.clone() else {
            self.mode = TrackMode::Unlocked;
            return;
        };

        let Some((_idx, d)) = nearest_player(players, prev.x, prev.y, ASSOC_MAX_DIST) else {
            self.miss_frames = self.miss_frames.saturating_add(1);
            if self.miss_frames > MISS_REUSE_MAX {
                self.mode = TrackMode::Probing;
                self.probe_frame = 0;
                self.probe_scores.clear();
                self.hit = None;
            }
            return;
        };

        let (x, _y) = ((d.x1 + d.x2) * 0.5, d.y2);
        let screen_dx = x - prev.x;
        let residual = screen_dx - self.floor_dx;
        self.update_regime(screen_dx);

        if cmd != 0.0 {
            let r = sign_with_deadzone(residual, RESIDUAL_DEADZONE);
            if r != 0.0 && r != cmd {
                self.contradict_frames = self.contradict_frames.saturating_add(1);
            } else {
                self.contradict_frames = 0;
            }
        }

        if self.contradict_frames >= CONTRADICTION_LIMIT {
            self.mode = TrackMode::Probing;
            self.probe_frame = 0;
            self.probe_scores.clear();
            self.hit = None;
            self.contradict_frames = 0;
            return;
        }

        self.miss_frames = 0;
        self.hit = Some(SelfPlayerHit::from_det(d));
    }

    fn update_regime(&mut self, player_screen_dx: f32) {
        let fd = self.floor_dx.abs();
        let pd = player_screen_dx.abs();
        if fd >= 4.0 && pd <= 3.0 {
            self.regime = CameraRegime::Follow;
        } else if fd <= 2.0 && pd >= 4.0 {
            self.regime = CameraRegime::Clamp;
        }
    }
}

fn sign_with_deadzone(v: f32, dead: f32) -> f32 {
    if v > dead {
        1.0
    } else if v < -dead {
        -1.0
    } else {
        0.0
    }
}

fn nearest_dx(prev: &[(f32, f32)], x: f32, y: f32) -> Option<f32> {
    let (i, _) = nearest_index(prev, x, y, ASSOC_MAX_DIST)?;
    Some(x - prev[i].0)
}

fn nearest_index(prev: &[(f32, f32)], x: f32, y: f32, max_dist: f32) -> Option<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &(px, py)) in prev.iter().enumerate() {
        let dist = (px - x).hypot(py - y);
        if dist > max_dist {
            continue;
        }
        if best.map(|(_, d)| dist < d).unwrap_or(true) {
            best = Some((i, dist));
        }
    }
    best
}

fn nearest_player<'a>(
    players: &[&'a Detection],
    x: f32,
    y: f32,
    max_dist: f32,
) -> Option<(usize, &'a Detection)> {
    let pts: Vec<(f32, f32)> = players
        .iter()
        .map(|d| ((d.x1 + d.x2) * 0.5, d.y2))
        .collect();
    let (i, _) = nearest_index(&pts, x, y, max_dist)?;
    Some((i, players[i]))
}

fn estimate_shift_dx(old: &[(f32, f32)], new: &[(f32, f32)]) -> Option<f32> {
    if old.is_empty() || new.is_empty() {
        return None;
    }
    let mut dxs = Vec::new();
    for &(ox, oy) in old.iter().take(8) {
        let mut best: Option<(f32, f32)> = None;
        for &(nx, ny) in new {
            let dist = (ox - nx).hypot(oy - ny);
            if dist > 160.0 {
                continue;
            }
            if best.map(|(_, d)| dist < d).unwrap_or(true) {
                best = Some((nx - ox, dist));
            }
        }
        if let Some((dx, _)) = best {
            dxs.push(dx);
        }
    }
    if dxs.len() < 2 {
        return dxs.first().copied();
    }
    dxs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(dxs[dxs.len() / 2])
}

/// 残差：`player_screen_dx - floor_dx`（单测/诊断用）。
pub fn residual_x(player_screen_dx: f32, floor_dx: f32) -> f32 {
    player_screen_dx - floor_dx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yolo::CLASS_NAMES;

    fn det_player(x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection {
            class_id: 10,
            label: CLASS_NAMES[10],
            conf: 0.9,
            x1,
            y1,
            x2,
            y2,
        }
    }

    fn det_floor(x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection {
            class_id: 0,
            label: CLASS_NAMES[0],
            conf: 0.9,
            x1,
            y1,
            x2,
            y2,
        }
    }

    #[test]
    fn residual_follow_and_clamp_same_sign() {
        // 中央跟随：人不动，地板 -10 → residual +10
        assert!((residual_x(0.0, -10.0) - 10.0).abs() < 1e-3);
        // 边缘锁镜头：地板不动，人 +10 → residual +10
        assert!((residual_x(10.0, 0.0) - 10.0).abs() < 1e-3);
    }

    #[test]
    fn probe_locks_moving_self_in_follow_regime() {
        let mut tr = SelfTracker::new();
        // 他人固定世界坐标：跟随镜头时与地板同滚
        // 自身钉屏：screen_dx=0，floor_dx=-8，residual=+8
        let self_x = 400.0;
        let mut other_x = 200.0;
        let mut floor_x = 100.0;
        for i in 0..20 {
            let cmd = if (i / PROBE_HALF_FRAMES) % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            let floor_dx = -cmd * 8.0;
            floor_x += floor_dx;
            other_x += floor_dx;
            // self stays
            let dets = vec![
                det_player(other_x - 20.0, 300.0, other_x + 20.0, 400.0),
                det_player(self_x - 20.0, 300.0, self_x + 20.0, 400.0),
                det_floor(floor_x, 420.0, floor_x + 200.0, 460.0),
                det_floor(floor_x + 250.0, 420.0, floor_x + 450.0, 460.0),
            ];
            let _ = self_x;
            let _ = tr.update(&dets, cmd, 0.25);
        }
        assert_eq!(tr.mode(), TrackMode::Locked);
        let hit = tr.hit().expect("locked hit");
        assert!((hit.x - 400.0).abs() < 30.0, "should lock screen-stable self");
    }

    #[test]
    fn probe_locks_moving_self_in_clamp_regime() {
        let mut tr = SelfTracker::new();
        let mut self_x = 500.0;
        let other_x = 200.0;
        let floor_x = 100.0;
        for i in 0..20 {
            let cmd = if (i / PROBE_HALF_FRAMES) % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            self_x += cmd * 8.0;
            let dets = vec![
                det_player(other_x - 20.0, 300.0, other_x + 20.0, 400.0),
                det_player(self_x - 20.0, 300.0, self_x + 20.0, 400.0),
                det_floor(floor_x, 420.0, floor_x + 200.0, 460.0),
                det_floor(floor_x + 250.0, 420.0, floor_x + 450.0, 460.0),
            ];
            let _ = tr.update(&dets, cmd, 0.25);
        }
        assert_eq!(tr.mode(), TrackMode::Locked);
        let hit = tr.hit().expect("locked hit");
        assert!(hit.x > 450.0, "should lock the moving self near edge");
    }
}
