/// 自身锚点配置（YOLO+OCR 模式：OCR 脚点 ± 抖动，模拟定位偏差）。
#[derive(Debug, Clone, Copy)]
pub struct VisionAnchorConfig {
    /// 脚点 ± 像素随机偏移（按 `episode_seed` 整局固定）；0=关闭。
    pub anchor_jitter_px: f32,
}

impl Default for VisionAnchorConfig {
    fn default() -> Self {
        Self {
            anchor_jitter_px: 0.0,
        }
    }
}

impl VisionAnchorConfig {
    pub fn ocr() -> Self {
        Self::default()
    }

    pub fn ocr_with_jitter(jitter_px: f32) -> Self {
        Self {
            anchor_jitter_px: jitter_px.max(0.0),
        }
    }

    pub fn uses_anchor_jitter(&self) -> bool {
        self.anchor_jitter_px > 0.0
    }
}

/// 游戏 / 规则 bot / NEAT 训练运行配置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameSimConfig {
    /// 自动玩模式：装饰 NPC、波次刷怪、零初始药水。
    pub bot_play: bool,
    /// 预览模式：受击 HP 最低保留 1，便于持续观察 bot。
    pub preview: bool,
    /// NEAT 训练模式：启用视觉适应度计分、HP 归零结束。
    pub training: bool,
    /// 在多平台候选点中按 episode_seed 随机出生。
    pub random_player_spawn: bool,
    /// 规则 bot 输入门控；NEAT 训练/预览必须关闭（部署无此通道）。
    pub movement_gate: bool,
    /// 挥砍时自动转向最近同层怪。NEAT 训练/预览必须关闭，否则单向狂砍也能打中。
    pub attack_auto_face: bool,
    /// 怪物碰触是否掉血。NEAT 只练寻路时关闭，避免个体因被怪撞死而拿不到探索分。
    pub mob_damage: bool,
}

impl Default for GameSimConfig {
    fn default() -> Self {
        Self {
            bot_play: false,
            preview: false,
            training: false,
            random_player_spawn: false,
            movement_gate: false,
            attack_auto_face: true,
            mob_damage: true,
        }
    }
}

impl GameSimConfig {
    pub fn bot_play() -> Self {
        Self {
            bot_play: true,
            preview: false,
            training: false,
            random_player_spawn: super::types::TRAINING_RANDOM_PLAYER_SPAWN,
            movement_gate: true,
            attack_auto_face: true,
            mob_damage: true,
        }
    }

    /// 与自动玩相同环境，但不因 HP 归零结束。
    pub fn preview() -> Self {
        Self {
            bot_play: true,
            preview: true,
            training: false,
            random_player_spawn: super::types::TRAINING_RANDOM_PLAYER_SPAWN,
            movement_gate: true,
            attack_auto_face: true,
            mob_damage: true,
        }
    }

    /// NEAT 训练：自动玩环境 + 视觉适应度；无 MovementGate；只练寻路，不受怪物伤害。
    pub fn training() -> Self {
        Self {
            bot_play: true,
            preview: false,
            training: true,
            random_player_spawn: super::types::TRAINING_RANDOM_PLAYER_SPAWN,
            movement_gate: false,
            attack_auto_face: false,
            mob_damage: false,
        }
    }

    /// NEAT 预览：与训练同感知约束；开 training 计分便于 diag 对照。默认寻路预览（无伤害）。
    pub fn neat_preview() -> Self {
        Self {
            bot_play: true,
            preview: true,
            training: true,
            random_player_spawn: super::types::TRAINING_RANDOM_PLAYER_SPAWN,
            movement_gate: false,
            attack_auto_face: false,
            mob_damage: false,
        }
    }

    pub fn with_mob_damage(mut self, on: bool) -> Self {
        self.mob_damage = on;
        self
    }
}

/// 视觉感知节奏（**不改变**逻辑物理 60Hz；仅改变每秒 YOLO 次数）。
///
/// 内部仍用 `vision_interval_ticks`；CLI 对外使用 `--detect-hz`（次/秒）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionPaceConfig {
    /// 每 N 个逻辑帧执行一次感知（中间帧复用上一帧观测/动作）。
    pub vision_interval_ticks: u32,
}

impl Default for VisionPaceConfig {
    fn default() -> Self {
        Self {
            vision_interval_ticks: 1,
        }
    }
}

impl VisionPaceConfig {
    /// 与正式 60Hz 推理一致。
    pub fn realtime() -> Self {
        Self::default()
    }

    /// 推荐默认：约 10 次/秒感知（截图+推理约 100ms），物理仍 60Hz。
    pub fn fast() -> Self {
        Self::from_detect_hz(10.0)
    }

    /// 由检测频率（Hz）换算：`interval = round(60 / hz)`，钳制到 `[1, 60]`。
    pub fn from_detect_hz(hz: f32) -> Self {
        let hz = hz.clamp(1.0, super::types::LOGIC_HZ);
        let ticks = (super::types::LOGIC_HZ / hz)
            .round()
            .clamp(1.0, super::types::LOGIC_HZ) as u32;
        Self {
            vision_interval_ticks: ticks,
        }
    }

    /// 等效检测频率（次/秒）。
    pub fn detect_hz(self) -> f32 {
        super::types::LOGIC_HZ / self.vision_interval_ticks.max(1) as f32
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.vision_interval_ticks == 0 {
            return Err("vision_interval_ticks 不能为 0".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod pace_tests {
    use super::VisionPaceConfig;

    #[test]
    fn detect_hz_roundtrips_common_values() {
        assert_eq!(
            VisionPaceConfig::from_detect_hz(5.0).vision_interval_ticks,
            12
        );
        assert_eq!(
            VisionPaceConfig::from_detect_hz(10.0).vision_interval_ticks,
            6
        );
        assert_eq!(
            VisionPaceConfig::from_detect_hz(60.0).vision_interval_ticks,
            1
        );
        assert!((VisionPaceConfig::from_detect_hz(5.0).detect_hz() - 5.0).abs() < 0.01);
        assert!((VisionPaceConfig::from_detect_hz(10.0).detect_hz() - 10.0).abs() < 0.01);
    }
}
