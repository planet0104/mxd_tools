use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use super::{HIDDEN_NODE_START, INPUT_SIZE, OUTPUT_NODE_START, OUTPUT_SIZE};

static INNOVATION_COUNTER: AtomicUsize = AtomicUsize::new(1);
static INNOVATION_MAP: Mutex<Option<std::collections::HashMap<(usize, usize), usize>>> =
    Mutex::new(None);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionGene {
    pub in_node: usize,
    pub out_node: usize,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub connections: Vec<ConnectionGene>,
    /// 局末适应度（含终局惩罚，可能远低于峰值）。
    pub fitness: f32,
    pub adjusted_fitness: f32,
    /// 本局历史最高适应度（演化选亲依据）。
    #[serde(default)]
    pub peak_fitness: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Compatibility {
    pub c1: f32,
    pub c2: f32,
    pub c3: f32,
    pub threshold: f32,
}

impl Default for Compatibility {
    fn default() -> Self {
        Self {
            c1: 1.0,
            c2: 1.0,
            c3: 0.4,
            threshold: 3.0,
        }
    }
}

/// 选亲/排名用：优先 peak，旧检查点无 peak 时回退 final。
pub fn rank_fitness(g: &Genome) -> f32 {
    if g.peak_fitness > 0.0 {
        g.peak_fitness
    } else {
        g.fitness
    }
}

impl Genome {
    pub fn random_minimal<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let mut g = Self {
            connections: Vec::new(),
            fitness: 0.0,
            adjusted_fitness: 0.0,
            peak_fitness: 0.0,
        };
        let n = rng.gen_range(4..12);
        for _ in 0..n {
            let (i, o) = random_io_pair(rng);
            g.add_connection(rng, i, o);
        }
        g
    }

    pub fn compatibility_distance(&self, other: &Self, cfg: &Compatibility) -> f32 {
        let mut i = 0;
        let mut j = 0;
        let mut disjoint = 0;
        let mut excess = 0;
        let mut weight_diff = 0.0;
        let mut matching = 0;

        while i < self.connections.len() && j < other.connections.len() {
            let a = self.connections[i].innovation;
            let b = other.connections[j].innovation;
            if a == b {
                if self.connections[i].enabled && other.connections[j].enabled {
                    weight_diff += (self.connections[i].weight - other.connections[j].weight).abs();
                    matching += 1;
                }
                i += 1;
                j += 1;
            } else if a < b {
                disjoint += 1;
                i += 1;
            } else {
                disjoint += 1;
                j += 1;
            }
        }
        excess += self.connections.len().saturating_sub(i);
        excess += other.connections.len().saturating_sub(j);
        let n = self.connections.len().max(other.connections.len()).max(1);
        let avg_w = if matching > 0 {
            weight_diff / matching as f32
        } else {
            0.0
        };
        (cfg.c1 * excess as f32 + cfg.c2 * disjoint as f32) / n as f32 + cfg.c3 * avg_w
    }

    fn add_connection<R: Rng + ?Sized>(&mut self, rng: &mut R, in_node: usize, out_node: usize) {
        if in_node == out_node {
            return;
        }
        if self.connections.iter().any(|c| c.in_node == in_node && c.out_node == out_node) {
            return;
        }
        if creates_cycle(self, in_node, out_node) {
            return;
        }
        let innovation = innovation_for(in_node, out_node);
        self.connections.push(ConnectionGene {
            in_node,
            out_node,
            weight: rng.gen_range(-1.0..1.0),
            enabled: true,
            innovation,
        });
        self.connections.sort_by_key(|c| c.innovation);
    }

    fn next_hidden_id(&self) -> usize {
        self.connections
            .iter()
            .map(|c| c.in_node.max(c.out_node))
            .max()
            .unwrap_or(HIDDEN_NODE_START)
            + 1
    }
}

pub fn mutate<R: Rng + ?Sized>(genome: &mut Genome, rng: &mut R) {
    for c in &mut genome.connections {
        if rng.gen_bool(0.8) {
            c.weight += rng.gen_range(-0.3..0.3);
            c.weight = c.weight.clamp(-3.0, 3.0);
        }
    }
    if rng.gen_bool(0.25) {
        let (i, o) = random_io_pair(rng);
        genome.add_connection(rng, i, o);
    }
    if rng.gen_bool(0.08) {
        mutate_add_node(genome, rng);
    }
    if rng.gen_bool(0.01) {
        let enabled: Vec<usize> = genome
            .connections
            .iter()
            .enumerate()
            .filter(|(_, c)| c.enabled)
            .map(|(i, _)| i)
            .collect();
        if let Some(&idx) = enabled.choose(rng) {
            genome.connections[idx].enabled = false;
        }
    }
}

fn mutate_add_node<R: Rng + ?Sized>(genome: &mut Genome, rng: &mut R) {
    let enabled: Vec<usize> = genome
        .connections
        .iter()
        .enumerate()
        .filter(|(_, c)| c.enabled)
        .map(|(i, _)| i)
        .collect();
    let Some(&idx) = enabled.choose(rng) else {
        return;
    };
    let old = genome.connections[idx].clone();
    genome.connections[idx].enabled = false;
    let hidden = genome.next_hidden_id();
    genome.add_connection(rng, old.in_node, hidden);
    genome.add_connection(rng, hidden, old.out_node);
}

pub fn crossover<R: Rng + ?Sized>(a: &Genome, b: &Genome, rng: &mut R) -> Genome {
    let (fitter, other) = if rank_fitness(a) >= rank_fitness(b) { (a, b) } else { (b, a) };
    let mut child = Genome {
        connections: Vec::new(),
        fitness: 0.0,
        adjusted_fitness: 0.0,
        peak_fitness: 0.0,
    };
    let mut j = 0;
    for gene in &fitter.connections {
        while j < other.connections.len() && other.connections[j].innovation < gene.innovation {
            j += 1;
        }
        let matched = other.connections.get(j).filter(|g| g.innovation == gene.innovation);
        let picked = if let Some(g) = matched {
            if rng.gen_bool(0.5) {
                gene.clone()
            } else {
                g.clone()
            }
        } else if rng.gen_bool(0.75) {
            gene.clone()
        } else {
            continue;
        };
        child.connections.push(picked);
    }
    child.connections.sort_by_key(|c| c.innovation);
    child
}

fn random_io_pair<R: Rng + ?Sized>(rng: &mut R) -> (usize, usize) {
    let inputs: Vec<usize> = (0..INPUT_SIZE).collect();
    let outputs: Vec<usize> = (OUTPUT_NODE_START..OUTPUT_NODE_START + OUTPUT_SIZE).collect();
    let hidden: Vec<usize> = (HIDDEN_NODE_START..HIDDEN_NODE_START + 8).collect();
    let in_node = if rng.gen_bool(0.85) {
        *inputs.choose(rng).unwrap()
    } else {
        *hidden.choose(rng).unwrap()
    };
    let out_node = if rng.gen_bool(0.7) {
        *outputs.choose(rng).unwrap()
    } else {
        *hidden.choose(rng).unwrap()
    };
    (in_node, out_node)
}

fn innovation_for(in_node: usize, out_node: usize) -> usize {
    let mut guard = INNOVATION_MAP.lock().unwrap();
    if guard.is_none() {
        *guard = Some(std::collections::HashMap::new());
    }
    let map = guard.as_mut().unwrap();
    let key = (in_node, out_node);
    if let Some(&id) = map.get(&key) {
        return id;
    }
    let id = INNOVATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    map.insert(key, id);
    id
}

pub fn reset_innovations() {
    INNOVATION_COUNTER.store(1, Ordering::Relaxed);
    let mut guard = INNOVATION_MAP.lock().unwrap();
    *guard = Some(std::collections::HashMap::new());
}

/// 检查点保存的创新号状态（`(in_node, out_node) → innovation` + 下一编号）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InnovationState {
    pub next_id: usize,
    pub entries: Vec<(usize, usize, usize)>,
}

pub fn export_innovation_state() -> InnovationState {
    let guard = INNOVATION_MAP.lock().unwrap();
    let map = guard.as_ref().cloned().unwrap_or_default();
    InnovationState {
        next_id: INNOVATION_COUNTER.load(Ordering::Relaxed),
        entries: map
            .into_iter()
            .map(|((i, o), id)| (i, o, id))
            .collect(),
    }
}

pub fn restore_innovation_state(state: &InnovationState) {
    let mut map = std::collections::HashMap::new();
    let mut max_id = 0usize;
    for (i, o, id) in &state.entries {
        max_id = max_id.max(*id);
        map.insert((*i, *o), *id);
    }
    let next = state.next_id.max(max_id.saturating_add(1)).max(1);
    INNOVATION_COUNTER.store(next, Ordering::Relaxed);
    let mut guard = INNOVATION_MAP.lock().unwrap();
    *guard = Some(map);
}

/// 从种群内全部连接重建创新号表（兼容无 `InnovationState` 的旧检查点）。
pub fn restore_innovations_from_population(
    population: &super::population::Population,
) {
    let mut map = std::collections::HashMap::new();
    let mut max_id = 0usize;
    for g in population.genomes().chain(std::iter::once(&population.best_ever)) {
        for c in &g.connections {
            max_id = max_id.max(c.innovation);
            map.entry((c.in_node, c.out_node))
                .and_modify(|id: &mut usize| *id = (*id).max(c.innovation))
                .or_insert(c.innovation);
        }
    }
    INNOVATION_COUNTER.store(max_id.saturating_add(1).max(1), Ordering::Relaxed);
    let mut guard = INNOVATION_MAP.lock().unwrap();
    *guard = Some(map);
}

fn creates_cycle(genome: &Genome, from: usize, to: usize) -> bool {
    if from == to {
        return true;
    }
    let mut stack = vec![to];
    let mut visited = vec![false; HIDDEN_NODE_START + 64];
    while let Some(node) = stack.pop() {
        if node == from {
            return true;
        }
        if node >= visited.len() {
            continue;
        }
        if visited[node] {
            continue;
        }
        visited[node] = true;
        for c in &genome.connections {
            if c.enabled && c.in_node == node {
                stack.push(c.out_node);
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn crossover_keeps_innovation_order() {
        reset_innovations();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let a = Genome::random_minimal(&mut rng);
        let b = Genome::random_minimal(&mut rng);
        let child = crossover(&a, &b, &mut rng);
        for w in child.connections.windows(2) {
            assert!(w[0].innovation <= w[1].innovation);
        }
    }

    #[test]
    fn restore_innovations_from_population_rebuilds_counter() {
        reset_innovations();
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let g = Genome::random_minimal(&mut rng);
        let max_innov = g.connections.iter().map(|c| c.innovation).max().unwrap_or(0);
        let pop = super::super::population::Population {
            generation: 3,
            species: vec![super::super::population::Species {
                representative: g.clone(),
                members: vec![g],
                stale: 0,
                best: 0.0,
            }],
            best_ever: Genome {
                connections: vec![],
                fitness: 0.0,
                adjusted_fitness: 0.0,
                peak_fitness: 0.0,
            },
            config: Default::default(),
            evaluations_completed: 0,
        };
        reset_innovations();
        restore_innovations_from_population(&pop);
        let state = export_innovation_state();
        assert!(state.next_id > max_innov);
        assert!(!state.entries.is_empty());
    }
}
