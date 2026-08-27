/// 自身玩家锚点来源（NEAT 观测原点）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisionAnchorMode {
    /// YOLO 玩家框 + 名牌 OCR（部署一致）。
    #[default]
    Ocr,
    /// 训练加速：用 sim 投影脚点匹配最近 YOLO「玩家」框，跳过 OCR。
    SimMatch,
}

/// 自身锚点配置（仅影响 NEAT 观测原点，不改变 YOLO 检测与拾取计分）。
#[derive(Debug, Clone, Copy)]
pub struct VisionAnchorConfig {
    pub mode: VisionAnchorMode,
    /// SimMatch 模式下锚点 ± 像素随机偏移，模拟 OCR/检测偏差（0=关闭）。
    pub sim_offset_px: f32,
}

impl Default for VisionAnchorConfig {
    fn default() -> Self {
        Self {
            mode: VisionAnchorMode::Ocr,
            sim_offset_px: 0.0,
        }
    }
}

impl VisionAnchorConfig {
    pub fn ocr() -> Self {
        Self::default()
    }

    pub fn sim_match(offset_px: f32) -> Self {
        Self {
            mode: VisionAnchorMode::SimMatch,
            sim_offset_px: offset_px.max(0.0),
        }
    }

    pub fn uses_sim_match(&self) -> bool {
        self.mode == VisionAnchorMode::SimMatch
    }
}

/// 游戏 / NEAT 训练运行配置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameSimConfig {
    /// NEAT 训练模式：装饰玩家、波次刷怪、视觉计分、零初始药水等。
    pub training: bool,
}

impl Default for GameSimConfig {
    fn default() -> Self {
        Self { training: false }
    }
}

impl GameSimConfig {
    pub fn training() -> Self {
        Self { training: true }
    }
}

/// 训练节奏（**不改变**逻辑物理 60Hz；仅改变多常做一次 YOLO+OCR）。
///
/// 部署 / 正式推理时必须 `vision_interval_ticks = 1`（每逻辑帧感知一次）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingPaceConfig {
    /// 每 N 个逻辑帧执行一次 `render + perceive`（中间帧复用上一帧观测）。
    pub vision_interval_ticks: u32,
}

impl Default for TrainingPaceConfig {
    fn default() -> Self {
        Self {
            vision_interval_ticks: 1,
        }
    }
}

impl TrainingPaceConfig {
    /// 与正式 60Hz 推理一致（最慢、无分布偏移）。
    pub fn realtime() -> Self {
        Self::default()
    }

    /// 推荐训练默认：约 5Hz 感知（每 12 tick），物理仍 60Hz。
    pub fn fast() -> Self {
        Self {
            vision_interval_ticks: 12,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.vision_interval_ticks == 0 {
            return Err("vision_interval_ticks 不能为 0".into());
        }
        Ok(())
    }
}
