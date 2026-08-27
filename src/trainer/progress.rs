//! 训练评估心跳：worker 写状态文件，主进程定期汇总输出。

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::game::GameSim;
use crate::trainer::log::log_line;

pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);

/// 单局评估进度（写入 status 文件供主进程读取）。
#[derive(Debug, Clone)]
pub struct EvalStatus {
    pub label: String,
    pub tick: usize,
    pub max_ticks: usize,
    pub hp: i32,
    pub max_hp: i32,
    pub potions: u32,
    pub meso: u32,
    pub mobs_alive: usize,
    pub npc_count: usize,
    pub fitness: f32,
    pub pickup_score: f32,
    pub vision_shaping_score: f32,
    pub memory_shaping_weighted: f32,
    pub stagnation_penalty: f32,
    pub idle_forfeit: bool,
    pub player_alive: bool,
    /// eval 循环已结束（可能在等视觉线程退出）。
    pub eval_done: bool,
}

#[derive(Debug, Clone, Default)]
pub struct EvalProgressConfig {
    /// 如 `gen2#9`
    pub label: String,
    pub status_file: Option<PathBuf>,
    /// 单进程评估时直接打印到 stderr
    pub console: bool,
}

impl EvalStatus {
    pub fn from_sim(label: &str, sim: &GameSim, tick: usize, max_ticks: usize) -> Self {
        let gt = sim.ground_truth();
        Self {
            label: label.to_string(),
            tick,
            max_ticks,
            hp: gt.hp,
            max_hp: gt.max_hp,
            potions: gt.potions,
            meso: gt.meso,
            mobs_alive: gt.mob_count,
            npc_count: sim.npc_players.len(),
            fitness: sim.fitness.score,
            pickup_score: sim.fitness.pickup_score,
            vision_shaping_score: sim.fitness.vision_shaping_score,
            memory_shaping_weighted: sim.fitness.shaping_config().memory_weight
                * sim.fitness.memory_shaping_score,
            stagnation_penalty: sim.fitness.stagnation_penalty,
            idle_forfeit: sim.fitness.idle_forfeit,
            player_alive: !sim.is_episode_over(),
            eval_done: false,
        }
    }

    pub fn life_tag(&self) -> &'static str {
        if self.eval_done {
            "完"
        } else if self.idle_forfeit {
            "早停"
        } else if self.tick >= self.max_ticks {
            "跑满"
        } else if self.player_alive {
            "活"
        } else {
            "亡"
        }
    }

    pub fn format_line(&self) -> String {
        format!(
            "label={} tick={}/{} hp={}/{} potions={} meso={} mobs={} npcs={} fitness={:.1} alive={} done={}",
            self.label,
            self.tick,
            self.max_ticks,
            self.hp,
            self.max_hp,
            self.potions,
            self.meso,
            self.mobs_alive,
            self.npc_count,
            self.fitness,
            u8::from(self.player_alive),
            u8::from(self.eval_done)
        )
    }

    pub fn write_file(&self, path: &Path) {
        let _ = fs::write(path, format!("{}\n", self.format_line()));
    }

    pub fn log_console(&self) {
        let life = if self.player_alive {
            "存活"
        } else {
            "已死亡"
        };
        log_line(format!(
            "[评估 {}] tick {}/{} | 主角 {life} HP {}/{} | 药 {} 币 {} | 怪 {} | 装饰 {} | 适应度 {:.1} (拾取{:.1}+视觉{:.1}+内存{:.1}−停滞{:.1})",
            self.label,
            self.tick,
            self.max_ticks,
            self.hp,
            self.max_hp,
            self.potions,
            self.meso,
            self.mobs_alive,
            self.npc_count,
            self.fitness,
            self.pickup_score,
            self.vision_shaping_score,
            self.memory_shaping_weighted,
            self.stagnation_penalty,
        ));
    }

    pub fn read_file(path: &Path) -> Option<Self> {
        let text = fs::read_to_string(path).ok()?;
        parse_status_line(text.lines().next()?.trim())
    }
}

pub fn maybe_emit_eval_heartbeat(
    last: &mut Instant,
    sim: &GameSim,
    tick: usize,
    max_ticks: usize,
    progress: Option<&EvalProgressConfig>,
) {
    let Some(cfg) = progress else {
        return;
    };
    if last.elapsed() < HEARTBEAT_INTERVAL {
        return;
    }
    *last = Instant::now();
    let status = EvalStatus::from_sim(&cfg.label, sim, tick, max_ticks);
    if let Some(path) = &cfg.status_file {
        status.write_file(path);
    }
    if cfg.console {
        status.log_console();
    }
}

fn parse_status_line(line: &str) -> Option<EvalStatus> {
    let mut fields: std::collections::HashMap<&str, String> = std::collections::HashMap::new();
    for part in line.split_whitespace() {
        let (k, v) = part.split_once('=')?;
        fields.insert(k, v.to_string());
    }
    let tick_parts: Vec<_> = fields.get("tick")?.split('/').collect();
    if tick_parts.len() != 2 {
        return None;
    }
    Some(EvalStatus {
        label: fields.get("label")?.clone(),
        tick: tick_parts[0].parse().ok()?,
        max_ticks: tick_parts[1].parse().ok()?,
        hp: fields.get("hp")?.split('/').next()?.parse().ok()?,
        max_hp: fields.get("hp")?.split('/').nth(1)?.parse().ok()?,
        potions: fields.get("potions")?.parse().ok()?,
        meso: fields.get("meso")?.parse().ok()?,
        mobs_alive: fields.get("mobs")?.parse().ok()?,
        npc_count: fields.get("npcs")?.parse().ok()?,
        fitness: fields.get("fitness")?.parse().ok()?,
        pickup_score: 0.0,
        vision_shaping_score: 0.0,
        memory_shaping_weighted: 0.0,
        stagnation_penalty: 0.0,
        idle_forfeit: false,
        player_alive: fields.get("alive")?.parse::<u8>().ok()? != 0,
        eval_done: fields
            .get("done")
            .and_then(|s| s.parse::<u8>().ok())
            .map(|v| v != 0)
            .unwrap_or(false),
    })
}

pub fn log_steady_heartbeat(
    spawns_completed: u32,
    total_spawns: u32,
    in_flight: &[(usize, PathBuf)],
    run_started: Instant,
    session_best: f32,
    global_best: f32,
) {
    let mut alive_workers = 0usize;
    let mut running: Vec<(usize, EvalStatus)> = Vec::new();
    for (spawn_id, path) in in_flight {
        if let Some(s) = EvalStatus::read_file(path) {
            if s.player_alive && !s.eval_done {
                alive_workers += 1;
            }
            running.push((*spawn_id, s));
        }
    }
    let pending = in_flight.len().saturating_sub(running.len());

    log_line(format!(
        "[训练心跳] 出生 {}/{} 完成 并行{}(活{}) 待读{} 已跑{:.0}s 最佳{:.1}/{:.1}",
        spawns_completed + 1,
        total_spawns,
        in_flight.len(),
        alive_workers,
        pending,
        run_started.elapsed().as_secs_f64(),
        session_best,
        global_best.max(session_best),
    ));

    let all_dead = running.iter().all(|(_, s)| !s.player_alive && !s.eval_done);
    let mut leaders: Vec<_> = running
        .iter()
        .filter(|(_, s)| s.player_alive && !s.eval_done)
        .collect();
    if leaders.is_empty() {
        leaders = running.iter().collect();
    }
    leaders.sort_by(|a, b| {
        b.1.fitness
            .partial_cmp(&a.1.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<String> = leaders
        .iter()
        .take(3)
        .map(|(spawn_id, s)| {
            format!(
                "s{spawn_id} fit{:.0}@{}/{} {}",
                s.fitness,
                s.tick,
                s.max_ticks,
                s.life_tag()
            )
        })
        .collect();
    if !top.is_empty() {
        let hint = if all_dead && !running.is_empty() {
            " (均已死亡，等待 worker 退出)"
        } else {
            ""
        };
        log_line(format!("  领先: {}{}", top.join(" | "), hint));
    }
}

pub fn log_pool_heartbeat(
    pop_generation: u32,
    generations_total: u32,
    completed: usize,
    population: usize,
    in_flight: &[(usize, PathBuf)],
    gen_started: Instant,
    done_best: f32,
    global_best: f32,
) {
    let mut alive_workers = 0usize;
    let mut running: Vec<(usize, EvalStatus)> = Vec::new();
    for (idx, path) in in_flight {
        if let Some(s) = EvalStatus::read_file(path) {
            if s.player_alive && !s.eval_done {
                alive_workers += 1;
            }
            running.push((*idx, s));
        }
    }
    let pending = in_flight.len().saturating_sub(running.len());
    let alive: Vec<_> = running
        .iter()
        .filter(|(_, s)| s.player_alive && !s.eval_done)
        .collect();

    log_line(format!(
        "[训练心跳] 代{}/{} {}/{}完成 并行{}(活{}) 待启动{} 本代{:.0}s 最佳{:.1}/{:.1}",
        pop_generation + 1,
        generations_total,
        completed,
        population,
        in_flight.len(),
        alive_workers,
        pending,
        gen_started.elapsed().as_secs_f64(),
        done_best,
        global_best.max(done_best),
    ));

    let all_dead = alive.is_empty()
        && !running.is_empty()
        && running.iter().all(|(_, s)| !s.eval_done);
    let mut leaders: Vec<_> = if !alive.is_empty() {
        alive
    } else {
        running.iter().collect()
    };
    leaders.sort_by(|a, b| {
        b.1.fitness
            .partial_cmp(&a.1.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<String> = leaders
        .iter()
        .take(3)
        .map(|(idx, s)| {
            format!("#{idx} fit{:.0}@{}/{} {}", s.fitness, s.tick, s.max_ticks, s.life_tag())
        })
        .collect();
    if !top.is_empty() {
        let hint = if all_dead {
            " (均已死亡，等待 worker 退出)"
        } else {
            ""
        };
        log_line(format!("  领先: {}{}", top.join(" | "), hint));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_line_roundtrip() {
        let s = EvalStatus {
            label: "gen1#3".into(),
            tick: 800,
            max_ticks: 3000,
            hp: 72,
            max_hp: 100,
            potions: 1,
            meso: 50,
            mobs_alive: 4,
            npc_count: 4,
            fitness: 120.5,
            pickup_score: 100.0,
            vision_shaping_score: 15.0,
            memory_shaping_weighted: 5.5,
            stagnation_penalty: 0.0,
            idle_forfeit: false,
            player_alive: true,
            eval_done: false,
        };
        let parsed = parse_status_line(&s.format_line()).unwrap();
        assert_eq!(parsed.label, "gen1#3");
        assert_eq!(parsed.tick, 800);
        assert_eq!(parsed.hp, 72);
        assert!(parsed.player_alive);
    }
}
