//! 常驻 worker 子进程 stdin/stdout 协议（一行一个 JSON）。

use serde::{Deserialize, Serialize};

use crate::neat::Genome;

/// 主进程 → worker：评估任务。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerJobRequest {
    pub job_idx: usize,
    pub genome: Genome,
    pub episode_seed: u64,
    pub label: String,
    pub status_file: String,
}

/// worker → 主进程：就绪握手。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerReady {
    pub ready: bool,
    pub worker_id: usize,
}

/// worker → 主进程：任务结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerJobResponse {
    pub job_idx: usize,
    pub fitness: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// 主进程 → worker：退出 daemon。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerQuit {
    pub quit: bool,
}

pub fn write_json_line(w: &mut impl std::io::Write, msg: &impl Serialize) -> anyhow::Result<()> {
    let json = serde_json::to_string(msg)?;
    use std::io::Write;
    writeln!(w, "{json}")?;
    w.flush()?;
    Ok(())
}

pub fn read_json_line<R: std::io::BufRead>(r: &mut R) -> anyhow::Result<Option<String>> {
    let mut line = String::new();
    let n = r.read_line(&mut line)?;
    if n == 0 {
        return Ok(None);
    }
    Ok(Some(line))
}
