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

    /// 推荐训练默认：约 4× 少跑视觉，物理仍 60Hz。
    pub fn fast() -> Self {
        Self {
            vision_interval_ticks: 4,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.vision_interval_ticks == 0 {
            return Err("vision_interval_ticks 不能为 0".into());
        }
        Ok(())
    }
}
