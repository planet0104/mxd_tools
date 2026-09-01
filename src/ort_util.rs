//! 共享 ONNX Runtime Session 构建（CPU / CUDA）。

use std::path::Path;

use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

#[derive(Debug, Clone, Copy)]
pub enum OrtDevice {
    Cpu,
    Cuda(u32),
}

impl OrtDevice {
    pub fn parse(s: &str) -> Self {
        let t = s.trim().to_ascii_lowercase();
        if t == "cpu" {
            return Self::Cpu;
        }
        if let Some(rest) = t.strip_prefix("cuda") {
            let id = rest
                .trim_start_matches(':')
                .trim()
                .parse::<u32>()
                .unwrap_or(0);
            return Self::Cuda(id);
        }
        if t == "0" || t == "gpu" {
            return Self::Cuda(0);
        }
        Self::Cpu
    }
}

pub fn build_session(
    onnx: &Path,
    device: OrtDevice,
    intra_threads: usize,
) -> Result<(Session, String)> {
    if !onnx.is_file() {
        anyhow::bail!("找不到 ONNX: {}", onnx.display());
    }

    let mut label = "cpu".to_string();
    let mk_builder = || {
        Session::builder()
            .map_err(|e| anyhow::anyhow!("创建 ORT SessionBuilder 失败: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("设置图优化失败: {e}"))?
            .with_intra_threads(intra_threads)
            .map_err(|e| anyhow::anyhow!("设置 intra_threads 失败: {e}"))
    };

    let session = match device {
        OrtDevice::Cuda(id) => {
            #[cfg(feature = "cuda")]
            {
                use ort::ep::CUDA;
                let try_cuda = mk_builder()?
                    .with_execution_providers([CUDA::default().with_device_id(id as i32).build()]);
                match try_cuda {
                    Ok(mut b) => match b.commit_from_file(onnx) {
                        Ok(s) => {
                            label = format!("cuda:{id}");
                            s
                        }
                        Err(e) => {
                            eprintln!("ORT CUDA session 失败，回退 CPU: {e}");
                            label = "cpu(fallback)".to_string();
                            mk_builder()?
                                .commit_from_file(onnx)
                                .with_context(|| format!("加载 ONNX 失败: {}", onnx.display()))?
                        }
                    },
                    Err(e) => {
                        eprintln!("ORT 注册 CUDA EP 失败，回退 CPU: {e}");
                        label = "cpu(fallback)".to_string();
                        mk_builder()?
                            .commit_from_file(onnx)
                            .with_context(|| format!("加载 ONNX 失败: {}", onnx.display()))?
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = id;
                label = "cpu(no-cuda-feature)".to_string();
                mk_builder()?
                    .commit_from_file(onnx)
                    .with_context(|| format!("加载 ONNX 失败: {}", onnx.display()))?
            }
        }
        OrtDevice::Cpu => mk_builder()?
            .commit_from_file(onnx)
            .with_context(|| format!("加载 ONNX 失败: {}", onnx.display()))?,
    };

    Ok((session, label))
}
