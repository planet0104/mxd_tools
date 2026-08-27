//! 常驻 worker 子进程池：训练期间固定 N 个 GL 窗 + YOLO 进程，不再每局 spawn。

use std::collections::VecDeque;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use crate::neat::Genome;
use crate::trainer::worker_protocol::{
    read_json_line, write_json_line, WorkerJobRequest, WorkerJobResponse, WorkerQuit, WorkerReady,
};

const READY_TIMEOUT: Duration = Duration::from_secs(120);

struct WorkerSlot {
    id: usize,
    child: Child,
    stdin: ChildStdin,
    busy_job: Option<usize>,
    status_path: Option<PathBuf>,
}

enum PoolEvent {
    Ready { worker_id: usize },
    Done(WorkerJobResponse),
    WorkerExit { worker_id: usize },
}

pub struct WorkerPool {
    workers: Vec<WorkerSlot>,
    idle: VecDeque<usize>,
    event_rx: Receiver<PoolEvent>,
    job_dir: PathBuf,
}

pub struct WorkerPoolConfig {
    pub workers: usize,
    pub exe: PathBuf,
    pub model: PathBuf,
    pub max_ticks: usize,
    pub pace: u32,
    pub fitness_shaping: f32,
    pub no_ocr: bool,
    pub anchor_offset: f32,
}

impl WorkerPool {
    pub fn spawn(cfg: &WorkerPoolConfig) -> anyhow::Result<Self> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let job_dir = manifest.join("tmp/neat_worker_jobs");
        std::fs::create_dir_all(&job_dir)?;

        let (event_tx, event_rx) = mpsc::channel();
        let mut workers = Vec::with_capacity(cfg.workers);
        let mut idle = VecDeque::with_capacity(cfg.workers);

        for id in 0..cfg.workers {
            let mut cmd = Command::new(&cfg.exe);
            cmd.args([
                "--worker-daemon",
                "--worker-id",
                &id.to_string(),
                "--model",
                cfg.model.to_str().unwrap(),
                "--max-ticks",
                &cfg.max_ticks.to_string(),
                "--pace",
                &cfg.pace.to_string(),
                "--fitness-shaping",
                &cfg.fitness_shaping.to_string(),
            ]);
            if cfg.no_ocr {
                cmd.arg("--no-ocr");
                cmd.arg("--anchor-offset");
                cmd.arg(cfg.anchor_offset.to_string());
            }
            let worker_log = manifest.join(format!("tmp/neat_worker_{id}.log"));
            let stderr_file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&worker_log);
            cmd.stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(match stderr_file {
                    Ok(f) => Stdio::from(f),
                    Err(_) => Stdio::null(),
                });

            let mut child = cmd.spawn()?;
            let stdin = child.stdin.take().expect("worker stdin");
            let stdout = child.stdout.take().expect("worker stdout");

            let tx = event_tx.clone();
            thread::spawn(move || worker_stdout_reader(id, stdout, tx));

            workers.push(WorkerSlot {
                id,
                child,
                stdin,
                busy_job: None,
                status_path: None,
            });
            idle.push_back(id);
        }

        drop(event_tx);

        let pool = Self {
            workers,
            idle,
            event_rx,
            job_dir,
        };
        pool.wait_all_ready()?;
        Ok(pool)
    }

    fn wait_all_ready(&self) -> anyhow::Result<()> {
        let deadline = Instant::now() + READY_TIMEOUT;
        let mut ready = vec![false; self.workers.len()];
        let mut ready_count = 0usize;

        while ready_count < self.workers.len() {
            if Instant::now() >= deadline {
                anyhow::bail!("worker 池就绪超时（{ready_count}/{}）", self.workers.len());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            match self.event_rx.recv_timeout(remaining.min(Duration::from_millis(200))) {
                Ok(PoolEvent::Ready { worker_id }) => {
                    if worker_id < ready.len() && !ready[worker_id] {
                        ready[worker_id] = true;
                        ready_count += 1;
                    }
                }
                Ok(PoolEvent::WorkerExit { worker_id }) => {
                    anyhow::bail!("worker {worker_id} 在就绪前退出");
                }
                Ok(PoolEvent::Done(_)) => {}
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("worker 事件通道断开");
                }
            }
        }
        Ok(())
    }

    pub fn job_dir(&self) -> &Path {
        &self.job_dir
    }

    pub fn idle_count(&self) -> usize {
        self.idle.len()
    }

    pub fn busy_count(&self) -> usize {
        self.workers.iter().filter(|w| w.busy_job.is_some()).count()
    }

    /// 若有空闲 worker 则派发单基因组评估任务。
    pub fn try_submit(
        &mut self,
        job_idx: usize,
        genome: &Genome,
        episode_seed: u64,
        label: &str,
        status_path: &Path,
    ) -> anyhow::Result<bool> {
        let worker_id = match self.idle.pop_front() {
            Some(id) => id,
            None => return Ok(false),
        };
        let _ = std::fs::remove_file(status_path);
        let req = WorkerJobRequest {
            job_idx,
            genome: genome.clone(),
            episode_seed,
            label: label.to_string(),
            status_file: status_path.to_string_lossy().into_owned(),
        };
        write_json_line(&mut self.workers[worker_id].stdin, &req)?;
        self.workers[worker_id].busy_job = Some(job_idx);
        self.workers[worker_id].status_path = Some(status_path.to_path_buf());
        Ok(true)
    }

    /// 收取已完成任务；`(job_idx, fitness)`。
    pub fn try_recv_done(&mut self) -> Option<anyhow::Result<(usize, f32)>> {
        match self.event_rx.try_recv() {
            Ok(PoolEvent::Done(resp)) => Some(self.finish_job(resp)),
            Ok(PoolEvent::Ready { .. }) => None,
            Ok(PoolEvent::WorkerExit { worker_id }) => Some(Err(anyhow::anyhow!(
                "worker {worker_id} 意外退出"
            ))),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err(anyhow::anyhow!("worker 事件通道断开"))),
        }
    }

    fn finish_job(&mut self, resp: WorkerJobResponse) -> anyhow::Result<(usize, f32)> {
        if let Some(err) = resp.error {
            anyhow::bail!("worker job {} 失败: {err}", resp.job_idx);
        }
        let worker_id = self
            .workers
            .iter()
            .position(|w| w.busy_job == Some(resp.job_idx))
            .ok_or_else(|| anyhow::anyhow!("收到未知 job {} 的结果", resp.job_idx))?;
        self.workers[worker_id].busy_job = None;
        self.workers[worker_id].status_path = None;
        self.idle.push_back(worker_id);
        Ok((resp.job_idx, resp.fitness))
    }

    pub fn in_flight_status(&self) -> Vec<(usize, PathBuf)> {
        self.workers
            .iter()
            .filter_map(|w| Some((w.busy_job?, w.status_path.clone()?)))
            .collect()
    }

    pub fn shutdown(mut self) -> anyhow::Result<()> {
        for w in &mut self.workers {
            let _ = write_json_line(&mut w.stdin, &WorkerQuit { quit: true });
        }
        let deadline = Instant::now() + Duration::from_secs(30);
        for w in &mut self.workers {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                let _ = w.child.kill();
                continue;
            }
            let _ = wait_child_timeout(w, left);
        }
        Ok(())
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        for w in &mut self.workers {
            let _ = write_json_line(&mut w.stdin, &WorkerQuit { quit: true });
            let _ = w.child.kill();
        }
    }
}

fn worker_stdout_reader(
    worker_id: usize,
    stdout: impl std::io::Read + Send + 'static,
    tx: mpsc::Sender<PoolEvent>,
) {
    let mut reader = BufReader::new(stdout);
    loop {
        let line = match read_json_line(&mut reader) {
            Ok(Some(l)) => l,
            Ok(None) => break,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(ready) = serde_json::from_str::<WorkerReady>(line) {
            if ready.ready {
                let _ = tx.send(PoolEvent::Ready {
                    worker_id: ready.worker_id,
                });
                continue;
            }
        }
        if let Ok(resp) = serde_json::from_str::<WorkerJobResponse>(line) {
            let _ = tx.send(PoolEvent::Done(resp));
            continue;
        }
    }
    let _ = tx.send(PoolEvent::WorkerExit { worker_id });
}

fn wait_child_timeout(w: &mut WorkerSlot, timeout: Duration) -> anyhow::Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = w.child.try_wait()? {
            if !status.success() {
                anyhow::bail!("worker {} 退出码 {:?}", w.id, status.code());
            }
            return Ok(());
        }
        if Instant::now() >= deadline {
            let _ = w.child.kill();
            anyhow::bail!("worker {} 退出超时", w.id);
        }
        thread::sleep(Duration::from_millis(50));
    }
}
