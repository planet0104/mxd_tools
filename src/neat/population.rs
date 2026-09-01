use rand::Rng;
use serde::{Deserialize, Serialize};

use super::genome::{crossover, mutate, rank_fitness, Compatibility, Genome};
use super::genome::reset_innovations;

const TARGET_SPECIES: usize = 8;
const STALE_SPECIES: u32 = 12;
/// 常规随机注入比例（提高多样性，抑制早熟）。
const FRESH_FRACTION: f32 = 0.22;
/// 最优 peak 连续多少次评估未提升，则提高随机注入。
const BEST_STAGNATION_EVALS: u32 = 80;
const STAGNANT_FRESH_FRACTION: f32 = 0.55;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Species {
    pub representative: Genome,
    pub members: Vec<Genome>,
    pub stale: u32,
    pub best: f32,
}

fn default_elite_breed_count() -> usize {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationConfig {
    pub size: usize,
    /// 选亲时只从前 N 名历史 peak 个体中锦标赛（持续进化补位）。
    #[serde(default = "default_elite_breed_count")]
    pub elite_breed_count: usize,
    pub compat: Compatibility,
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            size: 150,
            elite_breed_count: 5,
            compat: Compatibility::default(),
        }
    }
}

impl PopulationConfig {
    pub fn with_size(size: usize) -> Self {
        Self {
            size,
            elite_breed_count: (size / 2).max(2),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    pub generation: u32,
    pub species: Vec<Species>,
    pub best_ever: Genome,
    pub config: PopulationConfig,
    /// 已完成评估的个体数（持续进化进度）。
    #[serde(default)]
    pub evaluations_completed: u32,
    /// 自上次刷新 best_ever 以来的评估次数。
    #[serde(default)]
    pub evals_since_best_improve: u32,
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
        let mut best_ever = genomes.first().cloned().unwrap_or_else(|| Genome::random_minimal(rng));
        // 未评估个体 fitness=0；若保留 0，终局大负分的个体永远无法刷新最优（日志也不出）。
        best_ever.fitness = f32::NEG_INFINITY;
        best_ever.peak_fitness = f32::NEG_INFINITY;
        let species = speciate(genomes, &cfg.compat);
        Self {
            generation: 0,
            species,
            best_ever,
            config: cfg,
            evaluations_completed: 0,
            evals_since_best_improve: 0,
        }
    }

    pub fn genomes_mut(&mut self) -> impl Iterator<Item = &mut Genome> {
        self.species.iter_mut().flat_map(|s| s.members.iter_mut())
    }

    pub fn genomes(&self) -> impl Iterator<Item = &Genome> {
        self.species.iter().flat_map(|s| s.members.iter())
    }

    pub fn archive(&self) -> Vec<Genome> {
        let mut all: Vec<Genome> = self.genomes().cloned().collect();
        all.sort_by(|a, b| {
            rank_fitness(b)
                .partial_cmp(&rank_fitness(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all
    }

    pub fn update_best(&mut self) {
        self.update_best_with(None);
    }

    /// 用指定基因组或当前种群成员更新 `best_ever`（按 peak 排名）。
    pub fn update_best_with(&mut self, candidate: Option<&Genome>) {
        if let Some(g) = candidate {
            if rank_fitness(g) > rank_fitness(&self.best_ever) {
                self.best_ever = g.clone();
            }
            return;
        }
        let best = self.genomes().max_by(|a, b| {
            rank_fitness(a)
                .partial_cmp(&rank_fitness(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(g) = best {
            if rank_fitness(g) > rank_fitness(&self.best_ever) {
                self.best_ever = g.clone();
            }
        }
    }

    /// 单个体死亡/局结束后写入基因库（按 peak 保留 top-N，立即供下一槽位选亲）。
    pub fn on_eval_complete(&mut self, mut genome: Genome) {
        self.evaluations_completed += 1;
        let before = rank_fitness(&self.best_ever);
        self.update_best_with(Some(&genome));
        if rank_fitness(&self.best_ever) > before + 1e-3 {
            self.evals_since_best_improve = 0;
        } else {
            self.evals_since_best_improve = self.evals_since_best_improve.saturating_add(1);
        }
        let compat = self.config.compat;
        if let Some(sp) = self
            .species
            .iter_mut()
            .find(|sp| sp.representative.compatibility_distance(&genome, &compat) < compat.threshold)
        {
            sp.members.push(genome);
            if let Some(rep) = sp.members.iter().max_by(|a, b| {
                rank_fitness(a)
                    .partial_cmp(&rank_fitness(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                sp.representative = rep.clone();
            }
        } else {
            let peak = rank_fitness(&genome);
            self.species.push(Species {
                representative: genome.clone(),
                members: vec![genome],
                stale: 0,
                best: peak,
            });
        }
        self.trim_to_size();
        self.cull_stale();
    }

    /// 从 peak 排名前列的已评估个体中繁殖后代，补位下一槽位。
    pub fn spawn_offspring<R: Rng + ?Sized>(&mut self, rng: &mut R, spawn_index: u32) -> Genome {
        let mut pool = self.archive();
        if pool.is_empty() || pool.iter().all(|g| rank_fitness(g) <= 0.0) {
            return Genome::random_minimal(rng);
        }
        let stagnant = self.evals_since_best_improve >= BEST_STAGNATION_EVALS;
        let fresh_p = if stagnant {
            STAGNANT_FRESH_FRACTION
        } else {
            FRESH_FRACTION
        };
        if spawn_index > 0 && rng.gen::<f32>() < fresh_p {
            if stagnant
                && self.evals_since_best_improve > 0
                && self.evals_since_best_improve % 100 == 0
            {
                eprintln!(
                    "最优停滞 {} 局，注入随机个体 (fresh={:.0}%)",
                    self.evals_since_best_improve,
                    fresh_p * 100.0
                );
            }
            return Genome::random_minimal(rng);
        }
        let elite_n = self
            .config
            .elite_breed_count
            .max(1)
            .min(pool.len());
        // 停滞时放宽选亲面，避免只在 top-k 近亲繁殖。
        let elite_n = if stagnant {
            elite_n.max((pool.len() * 3 / 4).max(1))
        } else {
            elite_n
        };
        pool.truncate(elite_n);

        let parent_a = tournament_pick(&pool, rng);
        let mut child = if rng.gen_bool(0.75) && pool.len() > 1 {
            let parent_b = tournament_pick(&pool, rng);
            crossover(parent_a, parent_b, rng)
        } else {
            parent_a.clone()
        };
        mutate(&mut child, rng);
        if stagnant {
            // 额外一轮突变，跳出局部最优。
            mutate(&mut child, rng);
        }
        child.fitness = 0.0;
        child.adjusted_fitness = 0.0;
        child.peak_fitness = 0.0;
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
            rank_fitness(b)
                .partial_cmp(&rank_fitness(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all.truncate(self.config.size);
        self.species = speciate(all, &self.config.compat);
    }

    /// 续训时调整种群规模：保留 peak 排名前列，不足则交叉变异补齐。
    pub fn resize_to<R: Rng + ?Sized>(&mut self, new_size: usize, rng: &mut R) {
        if new_size == self.config.size {
            return;
        }
        self.config.size = new_size;
        self.config.elite_breed_count = (new_size / 2).max(2);
        let mut flat: Vec<Genome> = self.genomes().cloned().collect();
        if flat.is_empty() {
            flat.push(self.best_ever.clone());
        }
        flat.sort_by(|a, b| {
            rank_fitness(b)
                .partial_cmp(&rank_fitness(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
                child.peak_fitness = 0.0;
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
            .map(|s| s.members.iter().map(|g| rank_fitness(g)).sum::<f32>())
            .sum();
        let _ = total;

        for sp in &mut self.species {
            let sum: f32 = sp.members.iter().map(|g| rank_fitness(g).max(0.0)).sum();
            let denom = (sp.members.len() as f32).max(1.0);
            for m in &mut sp.members {
                m.adjusted_fitness = rank_fitness(m).max(0.0) / denom * sum.max(1.0);
            }
            sp.representative = sp
                .members
                .iter()
                .max_by(|a, b| {
                    rank_fitness(a)
                        .partial_cmp(&rank_fitness(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned()
                .unwrap_or_else(|| sp.representative.clone());
        }

        let mut children: Vec<Genome> = Vec::new();
        let elite_n = (self.config.size / 20).max(2);
        let mut flat = self.archive();
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
        let fitness = rank_fitness(&g);
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
        if rank_fitness(g) > rank_fitness(best) {
            best = g;
        }
    }
    best
}

impl Population {
    fn cull_stale(&mut self) {
        for sp in &mut self.species {
            let best = sp
                .members
                .iter()
                .map(|g| rank_fitness(g))
                .fold(0.0_f32, f32::max);
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn scored(id_weight: f32, final_fit: f32, peak: f32) -> Genome {
        let mut g = Genome::random_minimal(&mut rand::rngs::StdRng::seed_from_u64(1));
        g.connections[0].weight = id_weight;
        g.fitness = final_fit;
        g.peak_fitness = peak;
        g
    }

    #[test]
    fn trim_keeps_highest_final_fitness() {
        let mut pop =
            Population::new(PopulationConfig::with_size(2), &mut rand::rngs::StdRng::seed_from_u64(7));
        pop.on_eval_complete(scored(1.0, 95.0, 100.0));
        pop.on_eval_complete(scored(2.0, 80.0, 20.0));
        pop.on_eval_complete(scored(3.0, 48.0, 50.0));
        let archive = pop.archive();
        assert_eq!(archive.len(), 2);
        assert!(rank_fitness(&archive[0]) > rank_fitness(&archive[1]));
        assert_eq!(rank_fitness(&archive[0]), 95.0);
    }

    #[test]
    fn rank_ignores_peak_entirely() {
        // 高 peak + 低终局不再有任何加成，也不再额外重罚：排名只看终局。
        let collapsed = scored(1.0, 40.0, 200.0);
        let steady = scored(2.0, 60.0, 60.0);
        assert_eq!(rank_fitness(&collapsed), 40.0);
        assert!(rank_fitness(&steady) > rank_fitness(&collapsed));
    }

    #[test]
    fn rank_uses_final_including_negatives() {
        let lucky = scored(1.0, 10.0, 0.0);
        let walker = scored(2.0, -44.0, 0.0);
        let worse = scored(3.0, -80.0, 0.0);
        assert_eq!(rank_fitness(&lucky), 10.0);
        assert_eq!(rank_fitness(&walker), -44.0);
        assert!(rank_fitness(&walker) > rank_fitness(&worse));
    }

    #[test]
    fn spawn_offspring_uses_elite_pool_only() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let mut pop = Population::new(
            PopulationConfig {
                size: 4,
                elite_breed_count: 1,
                ..PopulationConfig::default()
            },
            &mut rng,
        );
        pop.on_eval_complete(scored(10.0, 0.0, 200.0));
        pop.on_eval_complete(scored(20.0, 0.0, 10.0));
        let child = pop.spawn_offspring(&mut rng, 1);
        assert!(!child.connections.is_empty());
        assert_eq!(child.peak_fitness, 0.0);
    }
}
