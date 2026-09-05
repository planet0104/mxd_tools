//! Live Nav 诊断日志：区分 YOLO 偏差 vs 导航/执行问题（跳台、边缘空砍、有梯不爬）。

use mxd_tools::game::nav::{NavBot, SubGoal, SurvivalMode};
use mxd_tools::game::observation::{obs_climb_hint, obs_platform_edge};
use mxd_tools::game::{
    obs_enemy_in_attack_range, obs_floor_ahead, obs_floor_ahead_connected, obs_floor_underfoot,
    obs_jump_allowed, obs_jump_target_ahead, obs_nearest_same_level_enemy_px, obs_step_up_dx,
    InputFrame, VisionSenseState, VisionStep, WINDOW_H, WINDOW_W, OBS_DIM,
};
use mxd_tools::yolo::Detection;

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub struct NavDiagLogger {
    last_reason: &'static str,
    last_goal: String,
    last_full: Instant,
    last_interesting: Instant,
    file_path: PathBuf,
}

impl Default for NavDiagLogger {
    fn default() -> Self {
        let file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tmp/nav_diag.log");
        if let Some(parent) = file_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // 新开一轮覆盖，避免混读旧日志。
        let _ = std::fs::write(&file_path, "");
        Self {
            last_reason: "",
            last_goal: String::new(),
            last_full: Instant::now() - Duration::from_secs(10),
            last_interesting: Instant::now() - Duration::from_secs(10),
            file_path,
        }
    }
}

impl NavDiagLogger {
    fn emit(&self, lines: &mut Vec<String>, line: String) {
        append_utf8(&self.file_path, &line);
        lines.push(line);
    }

    /// 每帧视觉结果后调用；返回需要打印的日志行（可能多条）。
    pub fn on_vision(
        &mut self,
        vtick: u32,
        step: &VisionStep,
        obs: &[f32; OBS_DIM],
        bot: &NavBot,
        sense: &VisionSenseState,
        intent: &InputFrame,
        hp_ratio: f32,
        survival: SurvivalMode,
    ) -> Vec<String> {
        let goal = bot.active_goal();
        let goal_s = goal.label();
        let reason = bot.last_reason;
        let diag = bot.diag();
        let face = if sense.facing >= 0.0 { 1.0 } else { -1.0 };

        let under = obs_floor_underfoot(obs);
        let floor_r = obs_floor_ahead(obs, 1.0);
        let floor_l = obs_floor_ahead(obs, -1.0);
        let conn_r = obs_floor_ahead_connected(obs, 1.0);
        let conn_l = obs_floor_ahead_connected(obs, -1.0);
        let edge_r = obs_platform_edge(obs, 1.0);
        let edge_l = obs_platform_edge(obs, -1.0);
        let edge_face = obs_platform_edge(obs, face);
        let step_dx = obs_step_up_dx(obs, WINDOW_W, WINDOW_H);
        let jump_ok = obs_jump_allowed(obs, face, sense.climbing);
        let jump_tgt = obs_jump_target_ahead(obs, face, WINDOW_W, WINDOW_H);
        let nearest_e = obs_nearest_same_level_enemy_px(obs, WINDOW_W, WINDOW_H);
        let atk_range = obs_enemy_in_attack_range(obs, face);
        let edge_whiff = intent.attack && edge_face && !atk_range;
        let climb = obs_climb_hint(obs, WINDOW_W, WINDOW_H);
        let exits = bot.exit_summary(diag.nav_node);

        let at_graph_ledge = bot.at_node_ledge(diag.nav_node, diag.nav_x);

        let is_step = matches!(goal, SubGoal::StepUp { .. })
            || reason.starts_with("step_up")
            || reason.contains("step_up");
        let reason_changed = reason != self.last_reason;
        let goal_changed = goal_s != self.last_goal;
        let jump_now = intent.jump;
        let has_climb_exit = exits.contains("climb_");
        let survive_hot = matches!(
            survival,
            SurvivalMode::FleeClimb | SurvivalMode::HealWait
        );
        let interesting = reason_changed
            || goal_changed
            || is_step
            || jump_now
            || edge_whiff
            || edge_l
            || edge_r
            || step.self_player.is_none()
            || diag.visual_conf < 3
            || survive_hot
            || (has_climb_exit
                && !matches!(goal, SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. }));

        let mut out = Vec::new();
        let min_gap = if is_step || edge_whiff || edge_l || edge_r {
            Duration::from_millis(350)
        } else {
            Duration::from_millis(800)
        };
        let force_full = self.last_full.elapsed() >= Duration::from_secs(2);
        let allow_interest = interesting && self.last_interesting.elapsed() >= min_gap;

        if !force_full && !allow_interest {
            return out;
        }

        let hyp = hypothesize(
            is_step,
            edge_whiff,
            step_dx,
            under,
            floor_r,
            floor_l,
            jump_ok,
            jump_tgt,
            atk_range,
            step.self_player.is_some(),
            diag.visual_conf,
            diag.step_stall,
            reason,
            has_climb_exit,
            &goal,
            climb.is_some(),
            edge_l || edge_r,
            at_graph_ledge,
            intent.jump,
        );

        let climb_s = match climb {
            Some(h) => format!(
                "{}@{:.0}",
                match h.dir {
                    mxd_tools::game::ClimbDir::Up => "up",
                    mxd_tools::game::ClimbDir::Down => "down",
                },
                h.dx
            ),
            None => "-".into(),
        };

        let face_mismatch = nearest_e.map(|(edx, _)| {
            let enemy_r = edx > 8.0;
            let enemy_l = edx < -8.0;
            (enemy_r && face < 0.0) || (enemy_l && face > 0.0)
        });
        let atk_wrong = intent.attack && face_mismatch == Some(true);

        self.emit(
            &mut out,
            format!(
                "[诊断] t={vtick} reason={reason} goal={goal_s} exec={} | \
loc=node{} ({:.0},{:.0}) conf={} climb={} face={} | \
hp_pct={:.0} mode={:?} | \
yolo: {} | \
obs: under={} floorL/R={}/{} connL/R={}/{} edgeL/R={}/{} step_dx={} jump_ok={} jump_tgt={} \
enemy={} atk_rng={} face_mis={} atk_wrong={} climb_hint={climb_s} | \
keys={} | \
step: stall={} jumped={} jdir={:.0} cd={} walkL/R={:?}/{:?} dropL/R={:?}/{:?} graph_ledge={} | \
exits=[{exits}] | hyp={hyp}",
                diag.exec.label(),
                diag.nav_node,
                diag.nav_x,
                diag.nav_y,
                diag.visual_conf,
                sense.climbing as u8,
                if face > 0.0 { "R" } else { "L" },
                hp_ratio * 100.0,
                survival,
                summarize_yolo(step),
                under as u8,
                floor_l as u8,
                floor_r as u8,
                conn_l as u8,
                conn_r as u8,
                edge_l as u8,
                edge_r as u8,
                step_dx
                    .map(|d| format!("{d:.0}"))
                    .unwrap_or_else(|| "-".into()),
                jump_ok as u8,
                jump_tgt as u8,
                nearest_e
                    .map(|(dx, dy)| format!("({dx:.0},{dy:.0})"))
                    .unwrap_or_else(|| "-".into()),
                atk_range as u8,
                face_mismatch.map(|m| m as u8).unwrap_or(0),
                atk_wrong as u8,
                keys_tag(intent),
                diag.step_stall,
                diag.step_jumped as u8,
                diag.step_jump_dir,
                diag.step_jump_cd,
                diag.walk_left,
                diag.walk_right,
                diag.drop_left,
                diag.drop_right,
                at_graph_ledge as u8,
            ),
        );

        if atk_wrong {
            self.emit(
                &mut out,
                format!(
                    "[诊断] 反方向攻击: face={} enemy={} keys={} hp_pct={:.0} mode={:?}",
                    if face > 0.0 { "R" } else { "L" },
                    nearest_e
                        .map(|(dx, dy)| format!("({dx:.0},{dy:.0})"))
                        .unwrap_or_else(|| "-".into()),
                    keys_tag(intent),
                    hp_ratio * 100.0,
                    survival,
                ),
            );
        }

        if survive_hot && matches!(goal, SubGoal::ClimbUp { .. }) && under && sense.climbing {
            self.emit(
                &mut out,
                format!(
                    "[诊断] 假攀爬卡死风险: mode={:?} under=1 climb_sticky=1 keys={} \
(应跳+对准挂梯，不应只按U)",
                    survival,
                    keys_tag(intent),
                ),
            );
        }

        if edge_whiff {
            self.emit(
                &mut out,
                format!(
                    "[诊断] 边缘空砍: face={} edge={} atk_rng=0 nearest_enemy={} combat={} farm={}",
                    if face > 0.0 { "R" } else { "L" },
                    edge_face as u8,
                    nearest_e
                        .map(|(dx, dy)| format!("({dx:.0},{dy:.0})"))
                        .unwrap_or_else(|| "none".into()),
                    diag.combat_active as u8,
                    diag.farm_local as u8,
                ),
            );
        }

        if is_step {
            let pending = match (diag.pending_from, diag.pending_kind, diag.pending_to) {
                (Some(f), Some(k), Some(t)) => format!("{}-{}->{}", f, k.label(), t),
                _ => "-".into(),
            };
            self.emit(
                &mut out,
                format!(
                    "[诊断] 跳台: target_goal={goal_s} obs_step_dx={} pending={pending} \
fail={} ticks={} grounded={} graph_ledge={} edge_face={} jump_key={} | \
注: step_dx=- 表示 YOLO 未给出 16~80px 抬升台阶",
                    step_dx
                        .map(|d| format!("{d:.0}"))
                        .unwrap_or_else(|| "无YOLO台阶".into()),
                    diag.subgoal_failures,
                    diag.subgoal_ticks,
                    diag.grounded_est as u8,
                    at_graph_ledge as u8,
                    edge_face as u8,
                    intent.jump as u8,
                ),
            );
        }

        if has_climb_exit
            && !matches!(goal, SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. })
            && (edge_l || edge_r || at_graph_ledge || is_step)
        {
            self.emit(
                &mut out,
                format!(
                    "[诊断] 有梯不爬: climb_hint={climb_s} goal={goal_s} reason={reason} \
exits有climb但当前目标不是爬绳 — 偏规划/封边，不是YOLO没看到梯子"
                ),
            );
        }

        if (edge_l || edge_r) && (conn_l || conn_r) {
            self.emit(
                &mut out,
                format!(
                    "[诊断] 边缘信号矛盾: edgeL/R={}/{} 但 connL/R={}/{} \
(connected仍为true会让 walk_ok=true → step_up 继续 approach 不跳)",
                    edge_l as u8, edge_r as u8, conn_l as u8, conn_r as u8
                ),
            );
        }

        self.last_reason = reason;
        self.last_goal = goal_s;
        if force_full {
            self.last_full = Instant::now();
        }
        if allow_interest || force_full {
            self.last_interesting = Instant::now();
        }
        out
    }
}

fn hypothesize(
    is_step: bool,
    edge_whiff: bool,
    step_dx: Option<f32>,
    under: bool,
    floor_r: bool,
    floor_l: bool,
    jump_ok: bool,
    jump_tgt: bool,
    atk_range: bool,
    has_self: bool,
    conf: u8,
    stall: u32,
    reason: &str,
    has_climb_exit: bool,
    goal: &SubGoal,
    climb_hint: bool,
    at_obs_edge: bool,
    at_graph_ledge: bool,
    jumping: bool,
) -> &'static str {
    if !has_self {
        return "self_miss_player";
    }
    if conf < 3 {
        return "low_visual_conf_loc_drift";
    }
    if edge_whiff {
        return "edge_whiff_yolo_or_combat_sticky";
    }
    if has_climb_exit
        && at_obs_edge
        && !matches!(goal, SubGoal::ClimbUp { .. } | SubGoal::ClimbDown { .. })
        && !jumping
    {
        return "has_ladder_exit_but_not_climbing";
    }
    if is_step {
        if step_dx.is_none() && reason.contains("wait") {
            return "yolo_no_higher_floor_box";
        }
        if step_dx.is_none() && reason.contains("jump") {
            return "jump_by_graph_no_obs_step";
        }
        if step_dx.is_none() && reason.contains("approach") && at_obs_edge {
            return "approach_at_edge_no_yolo_step_should_jump_or_climb";
        }
        if !under {
            return "airborne_or_floor_miss";
        }
        if stall >= 3 {
            return "step_stall_cant_approach";
        }
        if reason.contains("approach") && !floor_r && !floor_l {
            return "approach_but_no_floor_ahead";
        }
        if reason.contains("wait") && jump_ok && !jump_tgt {
            return "ledge_no_step_target_yolo";
        }
        if reason.contains("wait") && (at_obs_edge || at_graph_ledge) && !jumping {
            return "ledge_wait_no_jump_key";
        }
        if reason.contains("fell") {
            return "fell_after_jump_timing_or_loc";
        }
        if climb_hint && reason.contains("unreachable") {
            return "step_failed_climb_hint_available";
        }
        return "step_up_in_progress";
    }
    if atk_range {
        return "ok";
    }
    "ok"
}

fn keys_tag(i: &InputFrame) -> String {
    let mut s = String::new();
    if i.left {
        s.push('L');
    }
    if i.right {
        s.push('R');
    }
    if i.up {
        s.push('U');
    }
    if i.down {
        s.push('D');
    }
    if i.jump {
        s.push('J');
    }
    if i.attack {
        s.push('A');
    }
    if i.pick_up {
        s.push('Z');
    }
    if s.is_empty() {
        s.push('-');
    }
    s
}

fn summarize_yolo(step: &VisionStep) -> String {
    let mut floor = 0u32;
    let mut rope = 0u32;
    let mut ladder = 0u32;
    let mut enemy = 0u32;
    let mut player = 0u32;
    let mut drop = 0u32;
    let mut best_floor: Option<&Detection> = None;
    for d in &step.detections {
        match d.label {
            "地板" => {
                floor += 1;
                if best_floor.map(|b| d.conf > b.conf).unwrap_or(true) {
                    best_floor = Some(d);
                }
            }
            "绳子" => rope += 1,
            "梯子" => ladder += 1,
            "花蘑菇" | "蓝蜗牛" | "绿蜗牛" | "红蜗牛" | "树怪" => enemy += 1,
            "玩家" => player += 1,
            "金币" | "药水" => drop += 1,
            _ => {}
        }
    }
    let self_s = match &step.self_player {
        Some(h) => format!("self=track@({:.0},{:.0})c={:.2}", h.x, h.y, h.conf),
        None => "self=MISS".into(),
    };
    let floor_s = best_floor
        .map(|d| {
            format!(
                "floor0=({:.0},{:.0})-{:.0}x{:.0}c={:.2}",
                d.x1,
                d.y1,
                d.x2 - d.x1,
                d.y2 - d.y1,
                d.conf
            )
        })
        .unwrap_or_else(|| "floor0=-".into());
    format!("{self_s} nF={floor} nR={rope} nL={ladder} nE={enemy} nP={player} nD={drop} {floor_s}")
}

fn append_utf8(path: &PathBuf, line: &str) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
}
