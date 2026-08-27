use crate::game::action::actions_from_bits;
use crate::game::InputFrame;

use super::genome::Genome;
use super::{INPUT_SIZE, OUTPUT_NODE_START, OUTPUT_SIZE};

/// sigmoid ≥ 此阈值视为该键按下（可多键同时按下）。
pub const OUTPUT_THRESHOLD: f32 = 0.5;

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

pub fn input_from_outputs(outputs: &[f32]) -> InputFrame {
    let bits: Vec<bool> = outputs
        .iter()
        .take(OUTPUT_SIZE)
        .map(|v| *v >= OUTPUT_THRESHOLD)
        .collect();
    actions_from_bits(&bits)
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
        let inp = input_from_outputs(&out);
        assert!(!inp.attack || out[3] >= OUTPUT_THRESHOLD);
    }

    #[test]
    fn combo_left_jump() {
        let outputs = vec![0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1];
        let inp = input_from_outputs(&outputs);
        assert!(inp.left);
        assert!(inp.jump);
        assert!(!inp.right);
    }
}
