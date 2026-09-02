use crate::game::macro_action::{MacroAction, MACRO_ACTION_COUNT};

use super::genome::Genome;
use super::{INPUT_SIZE, OUTPUT_NODE_START, OUTPUT_SIZE};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

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

/// 动作互斥：只在 `allowed` 为 true 的输出里取 argmax。
///
/// 未连线的输出恒为 0，已连线的输出 sigmoid 后 > 0，因此「有连接」仍然胜出。
/// 边缘撞墙等场景由 `MacroRunner::allowed` 屏蔽无效方向，这里不再回退到被禁动作。
pub fn action_from_outputs(outputs: &[f32], allowed: &[bool; MACRO_ACTION_COUNT]) -> MacroAction {
    let mut best_idx = None;
    let mut best_v = f32::NEG_INFINITY;
    for i in 0..MACRO_ACTION_COUNT {
        if !allowed[i] {
            continue;
        }
        let v = outputs.get(i).copied().unwrap_or(0.0);
        if v > best_v {
            best_v = v;
            best_idx = Some(i);
        }
    }
    if let Some(i) = best_idx {
        return MacroAction::from_index(i);
    }
    // 全部被屏蔽：仍按原始 argmax，由 begin() 当场判失败。
    let mut raw_best = 0;
    let mut raw_v = f32::NEG_INFINITY;
    for i in 0..MACRO_ACTION_COUNT {
        let v = outputs.get(i).copied().unwrap_or(0.0);
        if v > raw_v {
            raw_v = v;
            raw_best = i;
        }
    }
    MacroAction::from_index(raw_best)
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

    const ALL: [bool; MACRO_ACTION_COUNT] = [true; MACRO_ACTION_COUNT];

    #[test]
    fn outputs_match_action_count() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(64);
        let g = Genome::random_minimal(&mut rng);
        let inputs = vec![0.5_f32; INPUT_SIZE];
        let out = evaluate(&g, &inputs);
        assert_eq!(out.len(), MACRO_ACTION_COUNT);
    }

    #[test]
    fn action_picks_strongest_output() {
        let outputs = vec![0.1, 0.9, 0.3, 0.4, 0.5];
        assert_eq!(action_from_outputs(&outputs, &ALL), MacroAction::WalkRight);
        let outputs = vec![0.1, 0.2, 0.3, 0.4, 0.95];
        assert_eq!(action_from_outputs(&outputs, &ALL), MacroAction::Climb);
    }

    #[test]
    fn unconnected_outputs_lose_to_connected_ones() {
        let outputs = vec![0.0, 0.0, 0.5, 0.0, 0.0];
        assert_eq!(action_from_outputs(&outputs, &ALL), MacroAction::JumpLeft);
    }

    #[test]
    fn masked_action_falls_through_to_next_best() {
        let outputs = vec![0.2, 0.9, 0.6, 0.1, 0.0];
        let mut allowed = ALL;
        allowed[MacroAction::WalkRight.index()] = false;
        assert_eq!(action_from_outputs(&outputs, &allowed), MacroAction::JumpLeft);
    }

    #[test]
    fn all_masked_falls_back_to_raw_argmax() {
        let outputs = vec![0.2, 0.9, 0.6, 0.1, 0.0];
        let none = [false; MACRO_ACTION_COUNT];
        assert_eq!(action_from_outputs(&outputs, &none), MacroAction::WalkRight);
    }
}
