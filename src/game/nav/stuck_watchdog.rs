//! 全局卡住看门狗：位置不动、决策循环、爬绳来回空转 → 硬重置。

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::super::input::InputFrame;

/// 墙钟超过该时长仍无进展则硬重置。
pub const GLOBAL_STUCK_SECS: u64 = 10;
/// 爬绳允许更长：慢速上爬时单帧位移常 <12px，靠净垂直进展清停滞。
pub const GLOBAL_STUCK_CLIMB_SECS: u64 = 22;
/// 硬重置后的宽限，避免立刻再次触发。
const GRACE_SECS: u64 = 2;
/// 判定「位置未动」的像素阈值。
const MOVE_EPS_PX: f32 = 12.0;
/// 爬绳单帧：忽略 OCR 微抖（勿用过大，否则慢爬每帧都算不动）。
const CLIMB_MOVE_EPS_PX: f32 = 20.0;
/// 爬绳相对停滞起点的净垂直进展，超过则清停滞（正常上爬不会被 10s 打断）。
const CLIMB_NET_PROGRESS_PX: f32 = 36.0;
/// 用于循环检测的指纹窗口。
const FP_WINDOW: usize = 24;
/// 同一根绳在绳顶卡住恢复超过该次数 → 强制弃绳离开。
pub const ROPE_YOYO_LIMIT: u32 = 3;
/// 弃绳后封边时长（导航 tick）；过长会逼 bot 在右侧台阶空转十几分钟。
pub const ROPE_BLOCK_TICKS: u32 = 180;

#[derive(Debug, Clone)]
pub struct GlobalStuckWatchdog {
    last_x: f32,
    last_y: f32,
    has_pos: bool,
    stagnant_since: Option<Instant>,
    /// 进入停滞时的坐标，用于爬绳净进展判定。
    progress_anchor_x: f32,
    progress_anchor_y: f32,
    fingerprints: VecDeque<u32>,
    loop_since: Option<Instant>,
    grace_until: Option<Instant>,
    /// 最近一次硬重置原因（供日志）。
    pub last_fire: Option<&'static str>,
    /// 同一绳 x 上的恢复攀爬次数（来回空转计数）。
    rope_resume_x: Option<i32>,
    rope_resume_count: u32,
    /// 近段时间到达的绳顶/绳底标记，用于检测 yo-yo。
    rope_end_visits: VecDeque<(i32, bool)>,
}

impl Default for GlobalStuckWatchdog {
    fn default() -> Self {
        Self {
            last_x: 0.0,
            last_y: 0.0,
            has_pos: false,
            stagnant_since: None,
            progress_anchor_x: 0.0,
            progress_anchor_y: 0.0,
            fingerprints: VecDeque::with_capacity(FP_WINDOW),
            loop_since: None,
            grace_until: None,
            last_fire: None,
            rope_resume_x: None,
            rope_resume_count: 0,
            rope_end_visits: VecDeque::with_capacity(8),
        }
    }
}

impl GlobalStuckWatchdog {
    pub fn reset_tracking(&mut self, x: f32, y: f32) {
        self.last_x = x;
        self.last_y = y;
        self.has_pos = true;
        self.stagnant_since = None;
        self.progress_anchor_x = x;
        self.progress_anchor_y = y;
        self.fingerprints.clear();
        self.loop_since = None;
        self.grace_until = Some(Instant::now() + Duration::from_secs(GRACE_SECS));
    }

    /// 弃绳离开后清掉该绳的 yo-yo 计数。
    pub fn clear_rope_yoyo(&mut self) {
        self.rope_resume_x = None;
        self.rope_resume_count = 0;
        self.rope_end_visits.clear();
    }

    /// 记录一次「在某绳上恢复攀爬」。若已来回太多次，返回 true 表示应弃绳。
    pub fn note_rope_resume(&mut self, rope_x: f32) -> bool {
        let key = rope_x.round() as i32;
        if self.rope_resume_x == Some(key) {
            self.rope_resume_count = self.rope_resume_count.saturating_add(1);
        } else {
            self.rope_resume_x = Some(key);
            self.rope_resume_count = 1;
            self.rope_end_visits.clear();
        }
        self.rope_resume_count >= ROPE_YOYO_LIMIT
    }

    /// 记录到达绳顶/底。短时间顶↔底交替则视为 yo-yo。
    pub fn note_rope_end(&mut self, rope_x: f32, at_top: bool) -> bool {
        let key = rope_x.round() as i32;
        self.rope_end_visits.push_back((key, at_top));
        while self.rope_end_visits.len() > 6 {
            self.rope_end_visits.pop_front();
        }
        let same: Vec<bool> = self
            .rope_end_visits
            .iter()
            .filter(|(x, _)| *x == key)
            .map(|(_, t)| *t)
            .collect();
        if same.len() < 4 {
            return false;
        }
        // 顶底顶底… 至少 4 次交替。
        let mut alt = 0u32;
        for w in same.windows(2) {
            if w[0] != w[1] {
                alt += 1;
            }
        }
        alt >= 3
    }

    pub fn should_abandon_rope(&self, rope_x: f32) -> bool {
        let key = rope_x.round() as i32;
        self.rope_resume_x == Some(key) && self.rope_resume_count >= ROPE_YOYO_LIMIT
    }

    /// 观测一帧决策。若应硬重置，返回原因字符串。
    pub fn observe(
        &mut self,
        x: f32,
        y: f32,
        reason: &'static str,
        intent: &InputFrame,
    ) -> Option<&'static str> {
        self.observe_at(Instant::now(), x, y, reason, intent)
    }

    pub fn observe_at(
        &mut self,
        now: Instant,
        x: f32,
        y: f32,
        reason: &'static str,
        intent: &InputFrame,
    ) -> Option<&'static str> {
        if self.grace_until.is_some_and(|t| now < t) {
            self.last_x = x;
            self.last_y = y;
            self.has_pos = true;
            return None;
        }

        let climb_mode = reason.contains("climb");
        let move_eps = if climb_mode {
            CLIMB_MOVE_EPS_PX
        } else {
            MOVE_EPS_PX
        };
        let frame_moved = if self.has_pos {
            (x - self.last_x).abs() > move_eps || (y - self.last_y).abs() > move_eps
        } else {
            true
        };
        // 爬绳：单帧常只有数像素，用相对停滞起点的净垂直位移认进展。
        let net_climb_progress = climb_mode
            && self.stagnant_since.is_some()
            && (y - self.progress_anchor_y).abs() > CLIMB_NET_PROGRESS_PX;
        let moved = frame_moved || net_climb_progress;
        self.last_x = x;
        self.last_y = y;
        self.has_pos = true;

        if moved {
            self.stagnant_since = None;
            self.progress_anchor_x = x;
            self.progress_anchor_y = y;
            self.loop_since = None;
            self.fingerprints.clear();
            return None;
        }

        if self.stagnant_since.is_none() {
            self.progress_anchor_x = x;
            self.progress_anchor_y = y;
        }
        let stagnant_since = *self.stagnant_since.get_or_insert(now);
        let fp = fingerprint(reason, intent);
        self.fingerprints.push_back(fp);
        if self.fingerprints.len() > FP_WINDOW {
            self.fingerprints.pop_front();
        }

        let looping = is_repeating_loop(&self.fingerprints);
        if looping {
            self.loop_since.get_or_insert(now);
        } else {
            self.loop_since = None;
        }

        let limit = Duration::from_secs(if climb_mode {
            GLOBAL_STUCK_CLIMB_SECS
        } else {
            GLOBAL_STUCK_SECS
        });
        let pos_stuck = now.duration_since(stagnant_since) >= limit;
        let loop_stuck = self
            .loop_since
            .is_some_and(|t| now.duration_since(t) >= limit);
        if pos_stuck || loop_stuck {
            let why = if reason.contains("climb") {
                "global_stuck_climb"
            } else if looping || self.loop_since.is_some() {
                "global_stuck_loop"
            } else {
                "global_stuck_pos"
            };
            self.last_fire = Some(why);
            return Some(why);
        }
        None
    }

    pub fn note_fired(&mut self, x: f32, y: f32) {
        self.reset_tracking(x, y);
    }
}

fn fingerprint(reason: &'static str, intent: &InputFrame) -> u32 {
    let mut h = 2166136261u32;
    for b in reason.bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(16777619);
    }
    let mut bits = 0u32;
    if intent.left {
        bits |= 1;
    }
    if intent.right {
        bits |= 2;
    }
    if intent.up {
        bits |= 4;
    }
    if intent.down {
        bits |= 8;
    }
    if intent.jump {
        bits |= 16;
    }
    if intent.attack {
        bits |= 32;
    }
    h ^ bits
}

/// 窗口内是否存在周期 2..=6 的严格重复（输出来回抖）。
fn is_repeating_loop(fps: &VecDeque<u32>) -> bool {
    let n = fps.len();
    if n < 8 {
        return false;
    }
    let slice: Vec<u32> = fps.iter().copied().collect();
    for period in 2..=6 {
        if n < period * 3 {
            continue;
        }
        let start = n - period * 3;
        let mut ok = true;
        for i in 0..period * 2 {
            if slice[start + i] != slice[start + period + i] {
                ok = false;
                break;
            }
        }
        let uniq: std::collections::HashSet<u32> =
            slice[start..start + period].iter().copied().collect();
        if ok && uniq.len() >= 2 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inp(up: bool, left: bool) -> InputFrame {
        InputFrame {
            up,
            left,
            ..InputFrame::default()
        }
    }

    #[test]
    fn position_stall_fires_after_10s() {
        let mut w = GlobalStuckWatchdog::default();
        let t0 = Instant::now();
        assert!(w
            .observe_at(t0, 100.0, 200.0, "patrol", &inp(false, true))
            .is_none());
        assert!(w
            .observe_at(
                t0 + Duration::from_secs(1),
                101.0,
                200.0,
                "patrol",
                &inp(false, true)
            )
            .is_none());
        assert!(w
            .observe_at(
                t0 + Duration::from_secs(10),
                101.0,
                201.0,
                "patrol",
                &inp(false, true)
            )
            .is_none());
        assert_eq!(
            w.observe_at(
                t0 + Duration::from_secs(12),
                101.0,
                201.0,
                "patrol",
                &inp(false, true)
            ),
            Some("global_stuck_pos")
        );
    }

    #[test]
    fn climb_top_stall_still_fires() {
        let mut w = GlobalStuckWatchdog::default();
        let t0 = Instant::now();
        assert!(w
            .observe_at(t0, 488.0, 860.0, "climb_orphan_up", &inp(true, false))
            .is_none());
        assert!(w
            .observe_at(
                t0 + Duration::from_secs(1),
                489.0,
                860.0,
                "climb_orphan_up",
                &inp(true, false)
            )
            .is_none());
        assert!(w
            .observe_at(
                t0 + Duration::from_secs(10),
                489.0,
                861.0,
                "climb_orphan_up",
                &inp(true, false)
            )
            .is_none());
        assert_eq!(
            w.observe_at(
                t0 + Duration::from_secs(24),
                489.0,
                861.0,
                "climb_orphan_up",
                &inp(true, false)
            ),
            Some("global_stuck_climb")
        );
    }

    #[test]
    fn climb_micro_jitter_does_not_clear_stall() {
        let mut w = GlobalStuckWatchdog::default();
        let t0 = Instant::now();
        assert!(w
            .observe_at(t0, 488.0, 860.0, "climb_up_active", &inp(true, false))
            .is_none());
        // 20px OCR 抖：达不到净进展阈值，应继续累计。
        assert!(w
            .observe_at(
                t0 + Duration::from_secs(5),
                488.0,
                880.0,
                "climb_up_active",
                &inp(true, false)
            )
            .is_none());
        assert_eq!(
            w.observe_at(
                t0 + Duration::from_secs(28),
                490.0,
                875.0,
                "climb_up_finish_hold",
                &inp(true, false)
            ),
            Some("global_stuck_climb")
        );
    }

    #[test]
    fn climb_progress_clears_stall() {
        let mut w = GlobalStuckWatchdog::default();
        let t0 = Instant::now();
        let _ = w.observe_at(t0, 488.0, 1100.0, "climb_up_active", &inp(true, false));
        // 明显上移：应清停滞，不能中途打断正常爬绳。
        assert!(w
            .observe_at(
                t0 + Duration::from_secs(11),
                488.0,
                900.0,
                "climb_up_active",
                &inp(true, false)
            )
            .is_none());
    }

    #[test]
    fn climb_gradual_progress_clears_stall() {
        // 复现 preview：慢爬每帧 <20px，旧逻辑 10s 必触发 stuck→abandon。
        let mut w = GlobalStuckWatchdog::default();
        let t0 = Instant::now();
        let _ = w.observe_at(t0, 1477.0, 1200.0, "climb_up_active", &inp(true, false));
        for i in 1..=15 {
            let y = 1200.0 - (i as f32) * 8.0;
            let r = w.observe_at(
                t0 + Duration::from_secs(i as u64),
                1477.0,
                y,
                "climb_up_active",
                &inp(true, false),
            );
            assert!(r.is_none(), "gradual climb must not stuck at t={i}s y={y} got {r:?}");
        }
    }

    #[test]
    fn rope_resume_yoyo_trips() {
        let mut w = GlobalStuckWatchdog::default();
        assert!(!w.note_rope_resume(488.0));
        assert!(!w.note_rope_resume(488.0));
        assert!(w.note_rope_resume(488.0));
        assert!(w.should_abandon_rope(488.0));
    }

    #[test]
    fn movement_clears_non_climb_stall() {
        let mut w = GlobalStuckWatchdog::default();
        let t0 = Instant::now();
        let _ = w.observe_at(t0, 0.0, 0.0, "a", &inp(true, false));
        let _ = w.observe_at(t0 + Duration::from_secs(5), 1.0, 0.0, "a", &inp(true, false));
        assert!(w
            .observe_at(
                t0 + Duration::from_secs(6),
                40.0,
                0.0,
                "patrol",
                &inp(true, false)
            )
            .is_none());
        assert!(w
            .observe_at(
                t0 + Duration::from_secs(15),
                41.0,
                0.0,
                "patrol",
                &inp(true, false)
            )
            .is_none());
    }

    #[test]
    fn alternating_output_loop_labeled() {
        let mut w = GlobalStuckWatchdog::default();
        let t0 = Instant::now();
        w.reset_tracking(50.0, 50.0);
        w.grace_until = None;
        let mut last = None;
        // 爬绳停滞阈值更长，需跑够墙钟时间。
        for i in 0..90 {
            let reason = if i % 2 == 0 {
                "climb_orphan_up"
            } else {
                "climb_up_air"
            };
            let intent = if i % 2 == 0 {
                inp(true, false)
            } else {
                inp(false, true)
            };
            let t = t0 + Duration::from_millis(300 * i as u64);
            last = w.observe_at(t, 50.0, 50.0, reason, &intent);
        }
        assert!(
            matches!(last, Some("global_stuck_loop") | Some("global_stuck_climb")),
            "got {last:?}"
        );
    }
}
