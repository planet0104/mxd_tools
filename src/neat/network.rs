use crate::game::Action;

use super::genome::Genome;
use super::{INPUT_SIZE, OUTPUT_NODE_START, OUTPUT_SIZE};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 拓扑分层前向传播，返回 OUTPUT_SIZE 个 sigmoid 输出。
pub fn evaluate(genome: &Genome, inputs: &[f32]) -> Vec<f32> {
    let mut values = vec![0.0_f32; INPUT_SIZE + OUTPUT_SIZE + 64];
    for i in 0..INPUT_SIZE.min(inputs.len()) {
        values[i] = inputs[i];
    }

    let layers = topological_layers(genome);
    for layer in layers {
        for node in layer {
            if node < INPUT_SIZE {
                continue;
            }
            let mut sum = 0.0_f32;
            for c in &genome.connections {
                if !c.enabled || c.out_node != node {
                    continue;
                }
                if c.in_node < values.len() {
                    sum += values[c.in_node] * c.weight;
                }
            }
            if node < values.len() {
                values[node] = sigmoid(sum);
            }
        }
    }

    (0..OUTPUT_SIZE)
        .map(|i| values.get(OUTPUT_NODE_START + i).copied().unwrap_or(0.0))
        .collect()
}

pub fn action_from_outputs(outputs: &[f32]) -> Action {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, v) in outputs.iter().enumerate().take(OUTPUT_SIZE) {
        if *v > best_v {
            best_v = *v;
            best = i;
        }
    }
    Action::from_index(best)
}

fn topological_layers(genome: &Genome) -> Vec<Vec<usize>> {
    let mut nodes = std::collections::BTreeSet::new();
    for c in &genome.connections {
        if c.enabled {
            nodes.insert(c.in_node);
            nodes.insert(c.out_node);
        }
    }
    let mut layers: Vec<Vec<usize>> = Vec::new();
    let mut placed = std::collections::BTreeSet::new();
    for &n in &nodes {
        if n < INPUT_SIZE {
            placed.insert(n);
        }
    }
    loop {
        let mut layer = Vec::new();
        'next: for &node in &nodes {
            if placed.contains(&node) || node < INPUT_SIZE {
                continue;
            }
            for c in &genome.connections {
                if c.enabled
                    && c.out_node == node
                    && !placed.contains(&c.in_node)
                    && c.in_node >= INPUT_SIZE
                {
                    continue 'next;
                }
            }
            layer.push(node);
        }
        if layer.is_empty() {
            break;
        }
        for n in &layer {
            placed.insert(*n);
        }
        layers.push(layer);
        if placed.len() >= nodes.len() {
            break;
        }
    }
    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::genome::Genome;
    use rand::SeedableRng;

    #[test]
    fn outputs_len_matches_actions() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let g = Genome::random_minimal(&mut rng);
        let inputs = vec![0.5; INPUT_SIZE];
        let out = evaluate(&g, &inputs);
        assert_eq!(out.len(), OUTPUT_SIZE);
        let _ = action_from_outputs(&out);
    }
}
