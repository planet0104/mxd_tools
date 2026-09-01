//! 训练过程中导出的最优基因组快照（供 `neat_preview` 热加载）。

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};

use super::genome::Genome;
use super::population::Population;

pub const DEFAULT_BEST_GENOME_FILE: &str = "tmp/neat_best_genome.json";
pub const DEFAULT_SESSION_BEST_FILE: &str = "tmp/neat_session_best.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestGenomeSnapshot {
    pub generation: u32,
    pub fitness: f32,
    pub training_seed: u64,
    pub genome: Genome,
    pub updated_at: String,
}

impl BestGenomeSnapshot {
    pub fn from_population(population: &Population, training_seed: u64) -> Self {
        Self {
            generation: population.generation,
            fitness: crate::neat::rank_fitness(&population.best_ever),
            training_seed,
            genome: population.best_ever.clone(),
            updated_at: Utc::now().to_rfc3339(),
        }
    }

    pub fn from_genome(genome: Genome, generation: u32, training_seed: u64) -> Self {
        Self {
            generation,
            fitness: crate::neat::rank_fitness(&genome),
            training_seed,
            genome,
            updated_at: Utc::now().to_rfc3339(),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| format!("读取最优基因组 {path:?}"))?;
        serde_json::from_str(&text).context("解析最优基因组 JSON")
    }

    pub fn save_atomic(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, json)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }
}

pub fn save_best_if_improved(
    path: &Path,
    population: &Population,
    training_seed: u64,
    last_saved_fitness: &mut f32,
) -> Result<bool> {
    let new_fit = crate::neat::rank_fitness(&population.best_ever);
    if new_fit <= *last_saved_fitness {
        return Ok(false);
    }
    let prev = *last_saved_fitness;
    let snap = BestGenomeSnapshot::from_population(population, training_seed);
    snap.save_atomic(path)?;
    *last_saved_fitness = snap.fitness;
    eprintln!(
        "最优个体已更新: rank_fit {:.2} → {:.2} (gen={}, connections={}) → {}",
        prev,
        snap.fitness,
        snap.generation,
        snap.genome.connections.len(),
        path.display()
    );
    Ok(true)
}

pub fn save_session_best(
    path: &Path,
    genome: &Genome,
    generation: u32,
    training_seed: u64,
) -> Result<()> {
    let snap = BestGenomeSnapshot::from_genome(genome.clone(), generation, training_seed);
    snap.save_atomic(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::genome::Genome;
    use rand::SeedableRng;

    #[test]
    fn snapshot_roundtrip() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let g = Genome::random_minimal(&mut rng);
        let snap = BestGenomeSnapshot::from_genome(g, 3, 42);
        let json = serde_json::to_string(&snap).unwrap();
        let back: BestGenomeSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.generation, 3);
        assert_eq!(back.training_seed, 42);
    }
}
