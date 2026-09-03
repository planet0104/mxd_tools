use super::types::{PlatformNodeId, ProgressState};

pub struct ProgressMonitor {
    pub state: ProgressState,
}

impl Default for ProgressMonitor {
    fn default() -> Self {
        Self {
            state: ProgressState::default(),
        }
    }
}

impl ProgressMonitor {
    pub fn reset(&mut self, x: f32, y: f32, node: PlatformNodeId) {
        self.state = ProgressState {
            last_x: x,
            last_y: y,
            last_node: node,
            ..Default::default()
        };
    }

    pub fn tick(
        &mut self,
        x: f32,
        y: f32,
        node: PlatformNodeId,
        visited_count: usize,
        subgoal_stagnant: bool,
    ) -> bool {
        // 仅认真实位移；98↔99 接缝 OCR 抖节点不重置，否则永远触发不了脱困。
        let moved = (x - self.state.last_x).abs() > 20.0 || (y - self.state.last_y).abs() > 16.0;
        if moved {
            self.state.stagnant_ticks = 0;
        } else {
            self.state.stagnant_ticks = self.state.stagnant_ticks.saturating_add(1);
        }

        if visited_count > self.state.last_visited_count {
            self.state.global_stall_ticks = 0;
            self.state.last_visited_count = visited_count;
        } else {
            self.state.global_stall_ticks = self.state.global_stall_ticks.saturating_add(1);
        }

        self.state.last_x = x;
        self.state.last_y = y;
        self.state.last_node = node;

        // ~4s 原地 / ~20s 无新访问（决策帧）→ 脱困。
        // 触发后必须清零计数，否则之后每一帧都会 fire，把 bot 锁死在 escape patrol。
        let fire = subgoal_stagnant
            || self.state.stagnant_ticks > 90
            || self.state.global_stall_ticks > 300;
        if fire {
            self.state.stagnant_ticks = 0;
            self.state.global_stall_ticks = 0;
        }
        fire
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_fire_resets_counters_so_not_every_frame() {
        let mut p = ProgressMonitor::default();
        p.reset(100.0, 200.0, 1);
        let mut fired = false;
        for _ in 0..400 {
            if p.tick(100.0, 200.0, 1, 0, false) {
                fired = true;
                break;
            }
        }
        assert!(fired, "should fire after prolonged no-visit stall");
        // 刚 fire：计数已清零，下一帧不应立刻再 fire
        assert!(!p.tick(100.0, 200.0, 1, 0, false));
        assert_eq!(p.state.global_stall_ticks, 1);
        assert_eq!(p.state.stagnant_ticks, 1);
    }
}
