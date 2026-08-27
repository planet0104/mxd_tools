//! 单基因组 eval 逐步耗时统计（`--profile`）。

use std::time::Instant;

pub use crate::trainer::agent::VisionWorkerTiming;

#[derive(Debug, Clone, Default)]
pub struct RenderStepTiming {
    pub draw_ms: f64,
    pub present_ms: f64,
    pub readback_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub struct TickProfile {
    pub tick: usize,
    pub vision_tick: bool,
    pub poll_ms: f64,
    pub vision_results: Vec<VisionWorkerTiming>,
    pub render: Option<RenderStepTiming>,
    pub submit_ok: Option<bool>,
    pub submit_ms: f64,
    pub sim_tick_ms: f64,
    pub loop_total_ms: f64,
}

#[derive(Debug, Default)]
pub struct EvalProfileReport {
    pub ticks: Vec<TickProfile>,
    pub setup_ms: f64,
    pub eval_loop_ms: f64,
    pub drain_ms: f64,
    pub teardown_ms: f64,
    pub wall_total_ms: f64,
}

impl EvalProfileReport {
    /// 将 drain 阶段收到的 worker 耗时合并到对应 tick。
    pub fn merge_worker_timings(&mut self, timings: &[VisionWorkerTiming]) {
        for wt in timings {
            if let Some(tp) = self.ticks.iter_mut().find(|t| t.tick == wt.tick as usize) {
                tp.vision_results.push(wt.clone());
            }
        }
    }
    pub fn print_summary(&self, pace: u32, max_ticks: usize) {
        eprintln!("\n========== 训练 eval 耗时剖析 ==========");
        eprintln!(
            "setup={:.2}ms  eval_loop={:.2}ms  drain={:.2}ms  teardown={:.2}ms  wall={:.2}ms  ticks={}/{}  pace={}",
            self.setup_ms,
            self.eval_loop_ms,
            self.drain_ms,
            self.teardown_ms,
            self.wall_total_ms,
            self.ticks.len(),
            max_ticks,
            pace
        );

        let vision_ticks: Vec<_> = self.ticks.iter().filter(|t| t.vision_tick).collect();
        let n = vision_ticks.len().max(1) as f64;

        if !vision_ticks.is_empty() {
            let avg = |f: fn(&TickProfile) -> f64| -> f64 {
                vision_ticks.iter().map(|t| f(t)).sum::<f64>() / n
            };
            eprintln!("\n--- 感知 tick 均值（共 {} 次）---", vision_ticks.len());
            eprintln!(
                "  GL draw={:.2}ms  readback={:.2}ms  render合计={:.2}ms（headless 无 next_frame）",
                avg(|t| t.render.as_ref().map(|r| r.draw_ms).unwrap_or(0.0)),
                avg(|t| t.render.as_ref().map(|r| r.readback_ms).unwrap_or(0.0)),
                avg(|t| t.render.as_ref().map(|r| r.total_ms).unwrap_or(0.0)),
            );
            eprintln!("  submit(try_send)={:.2}ms", avg(|t| t.submit_ms));

            let all_worker: Vec<&VisionWorkerTiming> = vision_ticks
                .iter()
                .flat_map(|t| t.vision_results.iter())
                .collect();
            if !all_worker.is_empty() {
                let wn = all_worker.len() as f64;
                let wavg = |f: fn(&VisionWorkerTiming) -> f64| -> f64 {
                    all_worker.iter().map(|w| f(w)).sum::<f64>() / wn
                };
                eprintln!("\n--- 视觉线程（YOLO+OCR+NEAT）---");
                eprintln!("  队列等待={:.2}ms", wavg(|w| w.queue_wait_ms));
                eprintln!("  perceive(YOLO+OCR)={:.2}ms", wavg(|w| w.perceive_ms));
                eprintln!("  NEAT前向={:.2}ms", wavg(|w| w.neat_ms));
                eprintln!("  worker合计={:.2}ms", wavg(|w| w.worker_total_ms));
            }
        }

        let sim_avg = self.ticks.iter().map(|t| t.sim_tick_ms).sum::<f64>()
            / self.ticks.len().max(1) as f64;
        let poll_avg = self.ticks.iter().map(|t| t.poll_ms).sum::<f64>()
            / self.ticks.len().max(1) as f64;
        let loop_avg = self.ticks.iter().map(|t| t.loop_total_ms).sum::<f64>()
            / self.ticks.len().max(1) as f64;

        eprintln!("\n--- 每 logic tick 均值 ---");
        eprintln!("  poll={:.3}ms  sim.tick={:.3}ms  主循环合计={:.3}ms", poll_avg, sim_avg, loop_avg);

        eprintln!("\n--- 逐 tick 明细 ---");
        for t in &self.ticks {
            let mut line = format!(
                "tick {:4} | loop {:6.2}ms | poll {:5.2}ms | sim {:5.3}ms",
                t.tick, t.loop_total_ms, t.poll_ms, t.sim_tick_ms
            );
            if t.vision_tick {
                if let Some(r) = &t.render {
                    line.push_str(&format!(
                        " | render {:.2}(d{:.1}+p{:.1}+r{:.1})",
                        r.total_ms, r.draw_ms, r.present_ms, r.readback_ms
                    ));
                }
                line.push_str(&format!(
                    " | submit {} {:.2}ms",
                    if t.submit_ok.unwrap_or(false) { "OK" } else { "DROP" },
                    t.submit_ms
                ));
            }
            for w in &t.vision_results {
                line.push_str(&format!(
                    " | worker[t{}] q{:.1}+yolo{:.1}+neat{:.1}={:.1}ms",
                    w.tick, w.queue_wait_ms, w.perceive_ms, w.neat_ms, w.worker_total_ms
                ));
            }
            eprintln!("{line}");
        }
        eprintln!("==========================================\n");
    }
}

pub fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

pub fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}
