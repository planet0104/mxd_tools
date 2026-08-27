//! `neat_preview` 专用：砍怪、掉落、拾取、NEAT 决策等 stderr 日志。

use std::collections::HashSet;

use mxd_tools::game::observation::{obs_has_drop, obs_has_enemy};
use mxd_tools::game::types::DropKind;
use mxd_tools::game::vision::VisionStep;
use mxd_tools::game::{GameSim, InputFrame, TrainingFitness};
use mxd_tools::yolo::Detection;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DropKey {
    kind: u8,
    x10: i32,
    y10: i32,
}

impl DropKey {
    fn from_drop(kind: DropKind, x: f32, y: f32) -> Self {
        Self {
            kind: match kind {
                DropKind::Meso => 0,
                DropKind::RedPotion => 1,
            },
            x10: (x * 10.0).round() as i32,
            y10: (y * 10.0).round() as i32,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct FitnessSnapshot {
    score: f32,
    pickup_score: f32,
    vision_shaping_score: f32,
    memory_shaping_score: f32,
    stagnation_penalty: f32,
    mob_hit_events: u32,
    mob_kill_events: u32,
    meso_events: u32,
    potion_events: u32,
    meso_units: u32,
    attack_align_events: u32,
    pickup_align_events: u32,
    stagnation_penalty_events: u32,
}

impl FitnessSnapshot {
    fn capture(f: &TrainingFitness) -> Self {
        Self {
            score: f.score,
            pickup_score: f.pickup_score,
            vision_shaping_score: f.vision_shaping_score,
            memory_shaping_score: f.memory_shaping_score,
            stagnation_penalty: f.stagnation_penalty,
            mob_hit_events: f.mob_hit_events,
            mob_kill_events: f.mob_kill_events,
            meso_events: f.meso_events,
            potion_events: f.potion_events,
            meso_units: f.meso_units,
            attack_align_events: f.attack_align_events,
            pickup_align_events: f.pickup_align_events,
            stagnation_penalty_events: f.stagnation_penalty_events,
        }
    }
}

pub struct PreviewEventLog {
    enabled: bool,
    episode: u32,
    drops_on_ground: HashSet<DropKey>,
    fitness: FitnessSnapshot,
    meso_bag: u32,
    potions_bag: u32,
    kills: u32,
    hp: i32,
}

impl PreviewEventLog {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            episode: 0,
            drops_on_ground: HashSet::new(),
            fitness: FitnessSnapshot::default(),
            meso_bag: 0,
            potions_bag: 0,
            kills: 0,
            hp: 0,
        }
    }

    pub fn begin_episode(&mut self, sim: &GameSim, seed: u64, target_fitness: f32) {
        if !self.enabled {
            return;
        }
        self.episode += 1;
        self.sync_snapshots(sim);
        self.line(format!(
            "=== 第 {} 局开始 seed={} 目标适应度={:.1} HP={}/{} ===",
            self.episode,
            seed,
            target_fitness,
            sim.state.player.hp,
            sim.state.player.max_hp
        ));
    }

    pub fn on_neat_decision(
        &mut self,
        tick: u32,
        input: &InputFrame,
        vision: &VisionStep,
        neat_outputs: Option<&[f32]>,
    ) {
        if !self.enabled {
            return;
        }
        let obs = &vision.observation.values;
        let pick_raw = neat_outputs
            .and_then(|o| o.get(4))
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "?".into());
        self.line(format!(
            "[NEAT t={tick}] 输入={} | 原始输出: 捡={pick_raw} | YOLO: {} | obs敌人={} obs掉落={}",
            format_input(input),
            summarize_detections(&vision.detections),
            u8::from(obs_has_enemy(obs)),
            u8::from(obs_has_drop(obs)),
        ));
        if let Some(p) = &vision.self_player {
            self.detail(format!(
                "  OCR自身 ({:.0},{:.0}) conf={:.2} text=\"{}\"",
                p.x, p.y, p.player_conf, p.ocr_text
            ));
        }
    }

    pub fn after_tick(&mut self, tick: u32, input: &InputFrame, sim: &GameSim) {
        if !self.enabled {
            return;
        }
        self.log_combat_and_drops(tick, sim);
        self.log_pickups(tick, input, sim);
        self.log_shaping_and_penalty(tick, sim);
        self.log_player_damage(tick, sim);
        self.sync_snapshots(sim);
    }

    pub fn end_episode(&mut self, sim: &GameSim, reason: &str) {
        if !self.enabled {
            return;
        }
        let f = &sim.fitness;
        let mem_w = f.shaping_config().memory_weight;
        let mem_weighted = mem_w * f.memory_shaping_score;
        self.line(format!(
            "=== 第 {} 局结束 ({}) tick={} 适应度={:.1} ===",
            self.episode, reason, sim.state.tick, f.score
        ));
        self.detail(format!(
            "  击杀={} 命中={} YOLO拾取={}次(币{}单位+药{}) 背包币={} 药={} 局末惩罚={:.1}",
            sim.state.kills,
            f.mob_hit_events,
            f.meso_events + f.potion_events,
            f.meso_units,
            f.potion_events,
            sim.state.meso,
            sim.state.potions,
            f.episode_penalty
        ));
        self.detail(format!(
            "  适应度分解: 拾取{:.1} +视觉{:.1} +内存{:.1}×{:.2}={:.1} −停滞{:.1} −局末{:.1}",
            f.pickup_score,
            f.vision_shaping_score,
            f.memory_shaping_score,
            mem_w,
            mem_weighted,
            f.stagnation_penalty,
            f.episode_penalty
        ));
        self.detail(format!(
            "  shaping事件: 攻击对齐={} 拾取对齐={} 停滞罚次={}",
            f.attack_align_events, f.pickup_align_events, f.stagnation_penalty_events
        ));
    }

    fn log_combat_and_drops(&mut self, tick: u32, sim: &GameSim) {
        let prev = self.fitness;
        let cur = FitnessSnapshot::capture(&sim.fitness);

        let hits = cur.mob_hit_events.saturating_sub(prev.mob_hit_events);
        if hits > 0 {
            self.line(format!(
                "[战斗 t={tick}] 命中怪物 ×{hits} (累计命中 {})",
                cur.mob_hit_events
            ));
        }

        let kills = cur.mob_kill_events.saturating_sub(prev.mob_kill_events);
        if kills > 0 {
            self.line(format!(
                "[战斗 t={tick}] 击杀怪物 ×{kills} (累计击杀 {})",
                sim.state.kills
            ));
        }

        let mut on_ground = HashSet::new();
        for drop in &sim.state.drops {
            if !drop.alive {
                continue;
            }
            let key = DropKey::from_drop(drop.kind, drop.x, drop.y);
            on_ground.insert(key);
            if !self.drops_on_ground.contains(&key) {
                self.line(format!(
                    "[掉落 t={tick}] {} @ ({:.0},{:.0})",
                    drop_kind_label(drop.kind),
                    drop.x,
                    drop.y
                ));
            }
        }
        self.drops_on_ground = on_ground;
    }

    fn log_pickups(&mut self, tick: u32, input: &InputFrame, sim: &GameSim) {
        let prev = self.fitness;
        let cur = FitnessSnapshot::capture(&sim.fitness);

        let meso_scored = cur.meso_events.saturating_sub(prev.meso_events);
        if meso_scored > 0 {
            let units = cur.meso_units.saturating_sub(prev.meso_units);
            self.line(format!(
                "[拾取 t={tick}] 金币 +{units} (YOLO计分 +{:.1}) 背包={}",
                units as f32,
                sim.state.meso
            ));
        } else if input.pick_up && sim.state.meso > self.meso_bag {
            let gained = sim.state.meso - self.meso_bag;
            self.line(format!(
                "[拾取 t={tick}] 金币 +{gained} (未YOLO计分, 框外或未识别) 背包={}",
                sim.state.meso
            ));
        }

        let potion_scored = cur.potion_events.saturating_sub(prev.potion_events);
        if potion_scored > 0 {
            self.line(format!(
                "[拾取 t={tick}] 红药水 ×{potion_scored} (YOLO计分 +50/个) 背包={}",
                sim.state.potions
            ));
        } else if input.pick_up && sim.state.potions > self.potions_bag {
            let gained = sim.state.potions - self.potions_bag;
            self.line(format!(
                "[拾取 t={tick}] 红药水 ×{gained} (未YOLO计分) 背包={}",
                sim.state.potions
            ));
        }
    }

    fn log_shaping_and_penalty(&mut self, tick: u32, sim: &GameSim) {
        let prev = self.fitness;
        let cur = FitnessSnapshot::capture(&sim.fitness);

        let atk_align = cur.attack_align_events.saturating_sub(prev.attack_align_events);
        if atk_align > 0 {
            self.detail(format!(
                "[shaping t={tick}] 攻击命中对齐 +0.5 ×{atk_align}"
            ));
        }

        let pick_align = cur.pickup_align_events.saturating_sub(prev.pickup_align_events);
        if pick_align > 0 {
            self.detail(format!(
                "[shaping t={tick}] 拾取键+obs有掉落 +1.0 ×{pick_align}"
            ));
        }

        let pen_events = cur
            .stagnation_penalty_events
            .saturating_sub(prev.stagnation_penalty_events);
        if pen_events > 0 {
            let pen = cur.stagnation_penalty - prev.stagnation_penalty;
            self.line(format!(
                "[停滞 t={tick}] 惩罚 −{pen:.1} (累计 −{:.1}, 当前适应度 {:.1})",
                cur.stagnation_penalty, cur.score
            ));
        }

        if sim.fitness.idle_forfeit && prev.score == cur.score {
            self.line(format!("[早停 t={tick}] 无产出停滞，本局结束"));
        }
    }

    fn log_player_damage(&mut self, tick: u32, sim: &GameSim) {
        if sim.state.player.hp < self.hp {
            self.line(format!(
                "[受伤 t={tick}] HP {} → {}/{}",
                self.hp,
                sim.state.player.hp,
                sim.state.player.max_hp
            ));
        }
    }

    fn sync_snapshots(&mut self, sim: &GameSim) {
        self.fitness = FitnessSnapshot::capture(&sim.fitness);
        self.meso_bag = sim.state.meso;
        self.potions_bag = sim.state.potions;
        self.kills = sim.state.kills;
        self.hp = sim.state.player.hp;
        self.drops_on_ground = sim
            .state
            .drops
            .iter()
            .filter(|d| d.alive)
            .map(|d| DropKey::from_drop(d.kind, d.x, d.y))
            .collect();
    }

    fn line(&self, msg: String) {
        eprintln!("{msg}");
    }

    fn detail(&self, msg: String) {
        eprintln!("{msg}");
    }
}

fn drop_kind_label(kind: DropKind) -> &'static str {
    match kind {
        DropKind::Meso => "金币",
        DropKind::RedPotion => "红药水",
    }
}

fn format_input(input: &InputFrame) -> String {
    let mut parts = Vec::new();
    if input.left {
        parts.push("左");
    }
    if input.right {
        parts.push("右");
    }
    if input.up {
        parts.push("上");
    }
    if input.down {
        parts.push("下");
    }
    if input.jump {
        parts.push("跳");
    }
    if input.attack {
        parts.push("攻");
    }
    if input.pick_up {
        parts.push("捡");
    }
    if input.use_potion {
        parts.push("喝药");
    }
    if parts.is_empty() {
        "—".to_string()
    } else {
        parts.join("+")
    }
}

fn summarize_detections(dets: &[Detection]) -> String {
    if dets.is_empty() {
        return "无检测".to_string();
    }
    let mut counts: std::collections::BTreeMap<&str, u32> = std::collections::BTreeMap::new();
    for d in dets {
        *counts.entry(d.label).or_insert(0) += 1;
    }
    counts
        .iter()
        .map(|(label, n)| format!("{label}×{n}"))
        .collect::<Vec<_>>()
        .join(" ")
}
