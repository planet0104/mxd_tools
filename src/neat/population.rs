use rand::Rng;
use serde::{Deserialize, Serialize};

use super::genome::{crossover, mutate, Compatibility, Genome};
use super::genome::reset_innovations;

const TARGET_SPECIES: usize = 8;
const STALE_SPECIES: u32 = 12;
const FRESH_FRACTION: f32 = 0.10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Species {
    pub representative: Genome,
    pub members: Vec<Genome>,
    pub stale: u32,
    pub best: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationConfig {
    pub size: usize,
    pub compat: Compatibility,
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            size: 150,
            compat: Compatibility::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    pub generation: u32,
    pub species: Vec<Species>,
    pub best_ever: Genome,
    pub config: PopulationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub population: Population,
    pub seed: u64,
    /// 创新号映射；旧检查点缺省时从 `population` 重建。
    #[serde(default)]
    pub innovation: super::genome::InnovationState,
    /// 已完成评估的个体数（持续进化模式进度）。
    #[serde(default)]
    pub spawns_completed: u32,
    /// 下一个待分配的出生编号（`0..total_spawn_target`）。
    #[serde(default)]
    pub next_spawn_id: u32,
    /// 训练目标个体总数（CLI `--generations`）。
    #[serde(default)]
    pub total_spawn_target: u32,
}

impl Population {
    pub fn new<R: Rng + ?Sized>(cfg: PopulationConfig, rng: &mut R) -> Self {
        reset_innovations();
        let genomes: Vec<Genome> = (0..cfg.size)
            .map(|_| Genome::random_minimal(rng))
            .collect();
        let best_ever = genomes.first().cloned().unwrap_or_else(|| Genome::random_minimal(rng));
        let species = speciate(genomes, &cfg.compat);
        Self {
            generation: 0,
            species,
            best_ever,
            config: cfg,
        }
    }

    pub fn genomes_mut(&mut self) -> impl Iterator<Item = &mut Genome> {
        self.species.iter_mut().flat_map(|s| s.members.iter_mut())
    }

    pub fn genomes(&self) -> impl Iterator<Item = &Genome> {
        self.species.iter().flat_map(|s| s.members.iter())
    }

    pub fn update_best(&mut self) {
        self.update_best_with(None);
    }

    /// 用指定基因组或当前种群成员更新 `best_ever`。
    pub fn update_best_with(&mut self, candidate: Option<&Genome>) {
        if let Some(g) = candidate {
            if g.fitness > self.best_ever.fitness {
                self.best_ever = g.clone();
            }
            return;
        }
        let best = self
            .species
            .iter()
            .flat_map(|s| s.members.iter())
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        if let Some(g) = best {
            if g.fitness > self.best_ever.fitness {
                self.best_ever = g.clone();
            }
        }
    }

    /// 单个体评估结束后写入种群（物种归类 + 裁剪规模 + 虚拟代计数）。
    pub fn on_eval_complete(&mut self, genome: Genome) {
        self.update_best_with(Some(&genome));
        let compat = self.config.compat;
        if let Some(sp) = self
            .species
            .iter_mut()
            .find(|sp| sp.representative.compatibility_distance(&genome, &compat) < compat.threshold)
        {
            sp.members.push(genome);
            if let Some(rep) = sp
                .members
                .iter()
                .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            {
                sp.representative = rep.clone();
            }
        } else {
            let fitness = genome.fitness;
            self.species.push(Species {
                representative: genome.clone(),
                members: vec![genome],
                stale: 0,
                best: fitness,
            });
        }
        self.trim_to_size();
        self.cull_stale();
    }

    /// 从当前已评估个体中繁殖一个后代，供下一槽位立即开跑。
    pub fn spawn_offspring<R: Rng + ?Sized>(&mut self, rng: &mut R, spawn_index: u32) -> Genome {
        let pool: Vec<Genome> = self.genomes().cloned().collect();
        if pool.is_empty() || pool.iter().all(|g| g.fitness == 0.0) {
            return Genome::random_minimal(rng);
        }
        if spawn_index > 0 && rng.gen::<f32>() < FRESH_FRACTION {
            return Genome::random_minimal(rng);
        }
        let parent_a = tournament_pick(&pool, rng);
        let mut child = if rng.gen_bool(0.75) && pool.len() > 1 {
            let parent_b = tournament_pick(&pool, rng);
            crossover(parent_a, parent_b, rng)
        } else {
            parent_a.clone()
        };
        mutate(&mut child, rng);
        child.fitness = 0.0;
        child.adjusted_fitness = 0.0;
        child
    }

    /// 每完成 `config.size` 次评估，虚拟代 +1（日志/检查点用）。
    pub fn bump_virtual_generation(&mut self, spawns_completed: u32) {
        let epoch = spawns_completed / self.config.size.max(1) as u32;
        self.generation = epoch;
    }

    fn trim_to_size(&mut self) {
        let mut all: Vec<Genome> = self.genomes().cloned().collect();
        if all.len() <= self.config.size {
            return;
        }
        all.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all.truncate(self.config.size);
        self.species = speciate(all, &self.config.compat);
    }

    /// 续训时调整种群规模：保留现有基因组，不足则交叉变异补齐。
    pub fn resize_to<R: Rng + ?Sized>(&mut self, new_size: usize, rng: &mut R) {
        if new_size == self.config.size {
            return;
        }
        self.config.size = new_size;
        let mut flat: Vec<Genome> = self.genomes().cloned().collect();
        if flat.is_empty() {
            flat.push(self.best_ever.clone());
        }
        flat.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        if new_size < flat.len() {
            flat.truncate(new_size);
        } else {
            while flat.len() < new_size {
                let parent_a = tournament_pick(&flat, rng);
                let child = if rng.gen_bool(0.75) {
                    let parent_b = tournament_pick(&flat, rng);
                    crossover(parent_a, parent_b, rng)
                } else {
                    parent_a.clone()
                };
                let mut child = child;
                mutate(&mut child, rng);
                child.fitness = 0.0;
                child.adjusted_fitness = 0.0;
                flat.push(child);
            }
        }
        self.species = speciate(flat, &self.config.compat);
    }

    pub fn evolve<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.generation += 1;
        self.update_best();
        self.cull_stale();

        let total: f32 = self
            .species
            .iter()
            .map(|s| s.members.iter().map(|g| g.fitness).sum::<f32>())
            .sum();
        let _ = total;

        for sp in &mut self.species {
            let sum: f32 = sp.members.iter().map(|g| g.fitness.max(0.0)).sum();
            let denom = (sp.members.len() as f32).max(1.0);
            for m in &mut sp.members {
                m.adjusted_fitness = m.fitness.max(0.0) / denom * sum.max(1.0);
            }
            sp.representative = sp
                .members
                .iter()
                .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
                .cloned()
                .unwrap_or_else(|| sp.representative.clone());
        }

        let mut children: Vec<Genome> = Vec::new();
        let elite_n = (self.config.size / 20).max(2);
        let mut flat: Vec<Genome> = self.genomes().cloned().collect();
        flat.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        for g in flat.iter().take(elite_n) {
            children.push(g.clone());
        }

        while children.len() < self.config.size {
            let child = self.spawn_offspring(rng, children.len() as u32);
            children.push(child);
        }

        let fresh_n = ((self.config.size as f32) * FRESH_FRACTION).round() as usize;
        for g in children.iter_mut().take(fresh_n) {
            *g = Genome::random_minimal(rng);
        }

        self.species = speciate(children, &self.config.compat);
    }
}

fn speciate(genomes: Vec<Genome>, compat: &Compatibility) -> Vec<Species> {
    let mut species: Vec<Species> = Vec::new();
    for g in genomes {
        let fitness = g.fitness;
        if let Some(sp) = species
            .iter_mut()
            .find(|sp| sp.representative.compatibility_distance(&g, compat) < compat.threshold)
        {
            sp.members.push(g);
        } else {
            species.push(Species {
                representative: g.clone(),
                members: vec![g],
                stale: 0,
                best: fitness,
            });
        }
    }
    while species.len() > TARGET_SPECIES {
        species.sort_by(|a, b| a.members.len().cmp(&b.members.len()));
        let merged = species.remove(0);
        if let Some(first) = species.first_mut() {
            first.members.extend(merged.members);
        }
    }
    species
}

fn tournament_pick<'a, R: Rng + ?Sized>(pool: &'a [Genome], rng: &mut R) -> &'a Genome {
    let k = 3.min(pool.len());
    let mut best = &pool[rng.gen_range(0..pool.len())];
    for _ in 1..k {
        let g = &pool[rng.gen_range(0..pool.len())];
        if g.fitness > best.fitness {
            best = g;
        }
    }
    best
}

impl Population {
    fn cull_stale(&mut self) {
        for sp in &mut self.species {
            let best = sp.members.iter().map(|g| g.fitness).fold(0.0_f32, f32::max);
            if best > sp.best + 1e-3 {
                sp.best = best;
                sp.stale = 0;
            } else {
                sp.stale += 1;
            }
        }
        self.species.retain(|s| s.stale < STALE_SPECIES || s.members.len() > 2);
    }
}
