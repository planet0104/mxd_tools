# NEAT 游戏 AI 训练说明

本文档描述「复刻版冒险岛 + YOLO 视觉 + OCR 自身定位 + NEAT」的训练管线。  
代码：`src/bin/neat_trainer.rs`、`src/bin/neat_preview.rs`、`src/bin/training_capture.rs`、`src/neat/`、`src/trainer/`、`GameSim::new_training`。

### 近期改动摘要

| 改动 | 说明 |
|------|------|
| **单人 headless eval** | 每个基因组独立一局 `GameSim::new_training`，相机跟随自身；`--workers` 默认 = `--population` 并行 |
| **headless / visible** | 默认 headless（1×1 隐藏窗）；`--visible --workers 1` 可实时看单人训练 |
| **适应度澄清** | 停滞/早停仅看 **水平位移**；命中怪 30s 豁免；无产出早停看 **近期** 拾取/击杀 |
| **组合键输出** | NEAT 8 路 sigmoid ≥0.5 多比特 → `InputFrame`（可 left+jump 等） |
| **受击击退** | 怪物碰撞仅安全水平击退，角落不会被顶出平台坠亡 |
| **持续进化** | worker 结束即补种；`--generations` = 总出生个体数，标签 `s1234` |
| **训练加速** | headless 截帧无 `next_frame`；eval 按 pace **批处理**全速 sim |

---

## 1. 总体架构

```
┌──────────────────────── 主线程（GL + GameSim） ────────────────────────┐
│  neat_trainer：离屏渲染 → submit_and_wait RGB 帧（生产 eval 同步等待）   │
│  sim.tick_with_action @ 60Hz（单人，相机跟随自身居中）                   │
│  最优个体 → tmp/neat_best_genome.json                                  │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ submit_and_wait（阻塞至本 tick 视觉完成）
┌───────────────────────────────▼───────────────────────────────────────┐
│  vision-neat-agent 后台线程（`src/trainer/agent.rs`）                   │
│  YOLO + OCR 锚定自身 → obs 编码 → NEAT evaluate → Action              │
└───────────────────────────────────────────────────────────────────────┘

   GameSim::new_training — 单人训练
   感知默认每 12 tick 一次（`--pace 12`，约 5Hz），中间 tick 复用上一 Action
   `--workers N`：N 个子进程各跑独立单人 eval（headless GL 占位窗）
```

**创建训练实例：**

```rust
let sim = GameSim::new_training(map, seed);
```

与普通 `GameSim::new` 的区别见 §3。

---

## 2. 视觉感知（部署每帧 1 次；训练可降频）

**部署 / `neat_preview --pace 1`**：每个逻辑帧（60Hz）执行一轮 YOLO+OCR。  
**训练默认 `--pace 12`**：逻辑仍 60Hz `tick`（**游戏内时间**），但每 12 tick 才感知一次（约 **5Hz**），中间帧复用上一帧观测；YOLO+OCR 等待**不计入** `--max-ticks`（见 §9）。

### 2.1 训练 eval（`neat_trainer`）

每个基因组独立一局，相机始终跟随自身居中，与 `neat_preview` / 真实游戏一致：

| 步骤 | 次数 | 说明 |
|------|------|------|
| 离屏渲染 | 1 | headless draw + readback（无 `next_frame`） |
| YOLO | **1** | 全部类别 |
| OCR | **1** | 匹配自身名牌「光头强加强版」 |
| 观测编码 | 1 | `VisionObservation::from_detections` |
| NEAT 前向 | 1 | `evaluate` → `Action` |

实现：`AgentController` + `run_paced_eval_batches()`（`src/trainer/eval.rs`）。

### 2.2 预览 / training_capture

`neat_preview`、`training_capture` 同样用 `AgentController` + `VisionPipeline::perceive()`（训练 eval 用 `submit_and_wait` 同步，preview 用 `try_submit` 异步）：

| 步骤 | 次数 | 说明 |
|------|------|------|
| 离屏渲染 | 1 | `render_target_to_rgb` |
| YOLO | **1** | 全部类别 |
| OCR | **1 轮** | 仅「玩家」框，匹配「光头强加强版」 |
| 观测编码 | 1 | 不再调用模型 |
| 计分提示 | 1 | `step.apply_fitness_hints(&mut sim)` |

```rust
use mxd_tools::neat::{evaluate, input_from_outputs};
use mxd_tools::game::{action_to_input, TrainingPaceConfig, OBS_DIM};

let pace = TrainingPaceConfig::fast(); // 默认 pace=12；预览/部署用 --pace 1
let mut last_obs = vec![0.0; OBS_DIM];

for tick in 0..max_ticks {
    if tick % pace.vision_interval_ticks == 0 {
        let step = pipeline.perceive(&rgb)?;
        step.apply_fitness_hints(&mut sim);
        last_obs.copy_from_slice(&step.observation.values);
    }
    let outputs = evaluate(genome, &last_obs);
    let input = input_from_outputs(&outputs); // 8 路多比特，可组合
    sim.tick(&action_to_input(action));
    if sim.is_episode_over() { break; }
}
let fitness = sim.fitness.score;
```

---

## 3. 训练环境规则（`GameSim::new_training`）

### 3.1 单人训练

- 每名基因组 **独立一局**，相机跟随自身居中。
- 地图上生成 **4 个装饰玩家**（`TRAINING_NPC_NAMES`）巡逻；YOLO 会检出「玩家」框，OCR 靠名牌排除自身。
- 实现：`evaluate_genome()` + `AgentController`。

### 3.2 怪物波次刷新（鼓励巡逻）

- 单个平台上的怪被击杀后 **不会立刻重生**。
- 当**全图所有怪物**都被杀光（死亡动画结束）后，**整图一波同时重生**。
- 迫使个体主动寻找仍有怪的平台，而不是守着一个点刷。
- 实现：`sim.rs` 中 `tick_mobs` 末尾，`mobs.is_empty()` 时 `spawn_mobs()`。

### 3.3 个体死亡

- **HP 归零 → GameOver**，本局 eval 结束。
- NEAT 评估用 `sim.is_episode_over()` 判断终止。
- 喝药（`UsePotion`）是**盲喝**：训练初始 **0 瓶红药**，须先 YOLO 看到药水并 `PickUp`。

### 3.4 掉落物

- 每只怪**必掉金币**（频率高，是主要得分来源）。
- 训练模式红药掉落率 **18%**（`TRAINING_POTION_DROP_CHANCE`）。
- 拾取需主动按 **Z**（`PickUp`），走近不会自动捡。

---

## 4. 适应度（Fitness）

**部署目标**仍是 YOLO 可见拾取；训练额外加 **shaping** 打破全 0 适应度，便于 NEAT 早期进化。

```text
总分 = 拾取分 + 视觉shaping + memory_weight × 内存shaping − 停滞惩罚
```

| 类别 | 条件 | 默认分值 |
|------|------|----------|
| **拾取（主分）** | YOLO 见金币/药水 + `PickUp` + 实际捡到 | 金币=面值，红药 +50 |
| **视觉 shaping** | obs 有敌人槽 + `Attack`，且 **30 tick 内实际命中** | +0.5（每局上限 60） |
| **视觉 shaping** | obs 有掉落槽 + `PickUp` | +2.5（每局上限 60） |
| **内存 shaping** | 命中怪物（×`memory_weight`） | +5 |
| **内存 shaping** | 击杀怪物（×`memory_weight`） | +25 |
| **击杀捡取链** | 击杀后 3s 内 YOLO 可见拾取 | +18/次 |
| **停滞惩罚** | 连续 5s **水平**位移 &lt; 48px，且停滞窗口内无新拾取/击杀 | −15/次，单局上限 90 |
| **无产出早停** | 开局 10s 后，连续 5s **水平**位移 &lt;48px，且停滞窗口内无新拾取/击杀，且距上次命中怪 &gt;15s（**地上有可见金币时不豁免**） | 立即结束本局 |
| **局末惩罚** | ≥2 次击杀但整局未 YOLO 捡到金币 | −25 |
| **局末惩罚** | YOLO 多次看到地上金币却未捡到 | −15 |

> 「站桩」按**相对锚点水平位移 &lt;48px** 判定（**跳跃不计移动**）；正常左右寻路会重置计时。命中怪仅 **15s 内**且**地上无可见金币**时暂缓早停。

CLI：`--fitness-shaping 0.25`（默认 0.25；设 `0` 关闭内存 shaping，保留视觉 shaping）。

**常见局部最优（plateau）**：只砍怪、不按 Z，`memory×命中/击杀 + 视觉攻击 shaping` 可稳定在 ~40–70 分；124.5 这类高分通常需要实际 YOLO 捡币。改 shaping 后请**续训**（勿 `--fresh`），从 checkpoint 继续进化捡币行为。

**不计分**：纯走动/跳跃、空砍（未见怪或攻击后未命中）、未出现在 YOLO 框内的拾取。装饰玩家不入 obs（靠 OCR 名牌排除）。

实现：`src/game/fitness.rs`。

---

## 5. NEAT 观测向量（94 维）

每槽 4 维：`(Δx/W, Δy/H, w/W, h/H)`，**不含类别**。

| 槽位 | 数量 | YOLO 筛选 | 用途 |
|------|------|-----------|------|
| 自身 | 2 | OCR 脚点 | 参考原点 |
| 地板 | 8 | `地板` | 跳跃、落脚（**含大小**） |
| 敌人 | 6 | 五种怪合并 | 攻击/逃跑（**含大小**，不区分类别） |
| 掉落 | 4 | 金币、药水 | 拾取方向 |
| 梯子 | 2 | `梯子` | 攀爬（**含大小**） |
| 绳子 | 3 | `绳子` | 抓绳、升降（**含大小**） |

装饰玩家**不进入**观测向量（YOLO 可能检出「玩家」框，靠 OCR 名牌排除，只保留自身锚点）。

---

## 6. 动作空间（8 路多比特，支持组合键）

网络输出 **8 个 sigmoid**（每位 ≥0.5 视为按下，**可同时为真**），映射为 `InputFrame`：

| 输出位 | 按键 | 训练意图 |
|--------|------|----------|
| 0 | Left (A) | 左移 |
| 1 | Right (D) | 右移 |
| 2 | Jump (Space) | 跳台 |
| 3 | Attack (J) | 攻击 |
| 4 | PickUp (Z) | 拾取 |
| 5 | UsePotion (1) | 喝药 |
| 6 | Up (W) | 抓绳/爬梯 |
| 7 | Down (S) | 沿绳下降 |

示例：`left+jump` = 输出位 0 与 2 同时 ≥0.5，用于跳上邻接平台。全关 = noop。

实现：`input_from_outputs` → `actions_from_bits` → `InputFrame` → `tick_with_action`。

> **破坏性变更**：旧版 9 选 1 argmax 基因组与当前 8 路输出不兼容，需 `--fresh` 重训。

---

## 7. 单局评估伪代码（`src/trainer/eval.rs` + `agent.rs`）

主线程渲染与模拟；视觉线程：YOLO+OCR+NEAT。按 `--pace` **批处理**：块首感知，块内全速 sim。

```rust
let mut agent = AgentController::spawn(pipeline, genome.clone());
let mut sim = GameSim::new_training(map, seed);
let interval = pace.vision_interval_ticks as usize;

let mut tick = 0;
while tick < max_ticks {
    agent.poll(&mut sim);

    let rgb = capture_render_rgb_headless(&assets, &sim, &rt);
    agent.submit_and_wait(&mut sim, tick as u32, rgb, TIMEOUT)?;

    let input = agent.input();
    let batch_end = (tick + interval).min(max_ticks);
    for t in tick..batch_end {
        sim.tick_with_action(&input);
        if sim.is_episode_over() { break; }
    }
    if sim.is_episode_over() { break; }
    tick = batch_end;
}
return sim.fitness.score;
```

**控制链路**：`evaluate` → `input_from_outputs`（多比特阈值）→ `InputFrame` → `tick_with_action`。8 个按键位均已接通 `GameSim` 物理，支持组合（如 left+jump）。

---

## 8. 运行与验证

```powershell
# ── NEAT 持续进化（默认 headless 单人 eval）──
# --generations=3000 表示共出生 3000 个个体；--population=40 为并行槽位
# --max-ticks=18000 = 游戏内约 5 分钟（60Hz×300s，不含 YOLO 等待）
cargo run --release --bin neat_trainer -- --generations 3000 --population 40 --pace 12 --max-ticks 18000 --fresh

# 可视化调试（须 --workers 1）
cargo run --release --bin neat_trainer -- `
  --visible --workers 1 --population 1 --pace 12 --max-ticks 720

# ── 预览 / 截图 / 单测 ──
cargo run --release --bin mini_game_headless -- --training --screenshot out.png

cargo run --release --bin mini_game -- --vision-preview --model models/yolo_nangang_e3000_best.onnx

cargo run --release --bin neat_preview

cargo run --release --bin training_capture -- --seed 42 --capture-tick 800 --pace 12

cargo test --lib
```

### 8.1 训练路径截图验证（`training_capture`）

`training_capture` 与 `neat_trainer` 共用单人 eval 路径（`AgentController` + `GameSim::new_training` + 「光头强加强版」OCR）。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--seed` | 42 | 本局 `episode_seed`（与训练 worker 一致） |
| `--capture-tick` | 800 | 在哪一逻辑 tick 的感知帧截图 |
| `--max-ticks` | `capture-tick + 400` | 最多模拟 tick（游戏内时间） |
| `--pace` | 12 | 感知间隔，应与训练一致 |
| `--model` | `models/yolo_nangang_e3000_best.onnx` | YOLO 模型 |
| `--genome-file` | （无） | 基因组 JSON；省略则用随机最小基因组或 `tmp/neat_best_genome.json` |
| `--out` | `tmp/training_capture/YYYYMMDD_HHMMSS/` | 输出目录 |

```powershell
cargo run --release --bin training_capture -- --seed 42 --capture-tick 800 --pace 12

cargo run --release --bin training_capture -- --capture-tick 40 --max-ticks 80 --pace 12
```

**输出**：`frame_raw.png`（1368×768）、`frame_yolo_ocr.jpg`（YOLO 框 + OCR 标注）。

---

## 9. 训练加速（headless 提速，部署仍 60Hz）

### 9.1 `--max-ticks` 与游戏时间

逻辑物理固定 **60Hz**（`LOGIC_HZ=60`）。`--max-ticks` 统计的是 **游戏内逻辑 tick 数**，**不包含** YOLO+OCR+NEAT 的墙钟等待。

| `--max-ticks` | 游戏内时长 | 说明 |
|---------------|------------|------|
| 3600 | 1 分钟 | 短跑 / 冒烟 |
| 7200 | 2 分钟 | 可视化调试 |
| **18000（默认）** | **5 分钟** | 常规训练推荐 |
| 99900 | ~28 分钟 | 超长局，一般不推荐 |

换算：`游戏秒数 = max_ticks / 60`；`pace=N` 时每局 YOLO 次数 ≈ `max_ticks / N`。

### 9.2 当前单步耗时（CPU，1368×768）

| 环节 | 约耗时 | 说明 |
|------|--------|------|
| 离屏 draw + readback | 3～10ms | headless，无 `next_frame` |
| YOLO | **~25～28ms** | ORT CPU |
| OCR 名牌 | **~37～40ms** | 当前最大瓶颈 |
| **合计（1×YOLO+OCR）** | **~65～100ms/次感知** | worker 线程 |

若每逻辑帧都感知（60Hz，pace=1）：墙钟约 **16 步/秒/线程**，一局 18000 tick 仅 YOLO 就要 **~30 分钟**。

### 9.3 原则

| 项目 | 训练 | 部署 | 说明 |
|------|------|------|------|
| 逻辑物理 60Hz | **保持** | **保持** | 跳跃、冷却与真实一致 |
| YOLO+OCR 频率 | **可降低** | **每帧 1 次** | 默认 `--pace 12`（5Hz） |
| 分辨率 1368×768 | 保持 | 保持 | 与模型一致 |

**不要**通过降低 `LOGIC_HZ` 来加速。

### 9.4 推荐策略

#### ① 持续进化 + 并行（收益最大）

- **持续进化**：任一 worker 结束即从当前优秀个体繁殖并补种，worker 池始终满负荷，不再等「整代」收工。
- `--workers N` / 省略时 **N = population**：N 路 headless 单人 eval 同时进行。
- `--generations 3000` = 共出生 **3000** 个个体（日志标签 `s0`…`s2999`），不是 3000 批。

```powershell
cargo run --release --bin neat_trainer -- --population 40 --generations 3000 --pace 12 --max-ticks 18000 --fresh
```

#### ② 视觉降频（默认 `--pace 12`，约 5Hz）

每 12 tick 才 YOLO+OCR 一次（`60÷12=5` 次/秒），中间帧复用上一观测；块内 sim **全速**批处理。部署前用 `neat_preview --pace 1` 验证上限。

| pace | 感知频率 | 间隔 |
|------|----------|------|
| 1 | 60Hz | 部署 |
| 4 | 15Hz | 较密，训练偏慢 |
| **12（默认）** | **5Hz** | **推荐训练** |
| 20 | 3Hz | 更快，梯度更粗 |

#### ③ Headless + 批处理截帧

默认 1×1 隐藏窗；感知 tick headless 截帧无 `next_frame`；`--visible` 略慢，仅调试用。

### 9.5 组合估算（单局 max_ticks=18000）

| 配置 | 游戏内时长 | 感知次数 | 约单局墙钟（1 worker，仅视觉） |
|------|------------|----------|--------------------------------|
| pace=1 | 5 min | 18000 | ~30 min |
| pace=4 | 5 min | 4500 | ~8 min |
| **pace=12（默认）** | **5 min** | **1500** | **~2.5 min** |
| pace=12 + 40 worker | 5 min | — | 吞吐 ≈ 40 × 单 worker |

> 差个体常因早停/死亡提前结束，实际墙钟更短。完整跑满 18000 tick 时，sim 批处理耗时相对 YOLO 可忽略。

### 9.6 部署验证

```powershell
cargo run --release --bin neat_preview -- --pace 1
```

---

## 10. 训练运行

### 10.1 `neat_trainer`

| 模式 | 触发 | 说明 |
|------|------|------|
| **headless**（默认） | 不加参数 | 1×1 隐藏窗 |
| **visible** | `--visible` | 缩放窗口实时显示单人训练；须 `--workers 1` |

| 参数 | 默认 | 说明 |
|------|------|------|
| `--generations` | 50 | **总出生个体数**（非批次数）；任一 worker 结束即补种下一个 |
| `--population` | 10 | 并行槽位 + 选择库规模；未写 `--workers` 时 worker 数等于该值 |
| `--workers` | 自动=population | 评估子进程数 |
| `--visible` | 关 | 可视化；须 `--workers 1` |
| `--pace` | **12** | 每 N tick 感知一次（默认 5Hz） |
| `--fitness-shaping` | 0.5 | 内存 shaping 权重 |
| `--max-ticks` | **18000** | 单局最大逻辑 tick（**游戏内约 5 分钟**，不含 YOLO 等待） |
| `--seed` | 42 | 随机种子 |
| `--model` | `models/yolo_nangang_e3000_best.onnx` | YOLO ONNX |
| `--checkpoint` | `tmp/neat_checkpoint.json` | 检查点 |
| `--best-genome` | `tmp/neat_best_genome.json` | 最优个体快照 |
| `--fresh` | — | 忽略检查点，重新开训 |

```powershell
# 常规 headless 满并行（3000 个个体持续补种，每局游戏内 5 分钟）
cargo run --release --bin neat_trainer -- --generations 3000 --population 40 --pace 12 --max-ticks 18000 --fresh

# 可视化调试（约 12 秒游戏时间）
cargo run --release --bin neat_trainer -- `
  --visible --workers 1 --population 1 --pace 12 --max-ticks 720

# 短跑验证（出生 8 个个体）
cargo run --release --bin neat_trainer -- --fresh --generations 8 --population 4 --max-ticks 200 --pace 12
```

**长跑监控**（`scripts/train_monitor.ps1`）：

```powershell
cargo build --release --bin neat_trainer
powershell -File scripts/train_monitor.ps1
```

### 10.2 `neat_preview`

| 参数 | 默认 | 说明 |
|------|------|------|
| `--genome` | `tmp/neat_best_genome.json` | 最优基因组 |
| `--pace` | 12 | 感知间隔；与训练一致；部署验证用 `1` |
| `--seed` | 0 | 局种子；0=用快照内 seed |

```powershell
cargo run --release --bin neat_preview
cargo run --release --bin neat_preview -- --pace 1
```

训练与 preview 共用 `AgentController` + 单人 `GameSim::new_training`。

### 10.3 `--profile`

```powershell
cargo run --release --bin neat_trainer -- --profile --profile-ticks 180 --pace 12 --seed 42
```

单基因组 eval 逐步耗时；典型 YOLO+OCR ~65～100ms/次（pace=12 时约 1500 次/局）。

完整训练慢的主因：**population × 每局感知次数 ÷ worker 吞吐**。

---

## 11. 参考

### 11.1 可执行文件

| 文件 | 用途 |
|------|------|
| `src/bin/neat_trainer.rs` | NEAT 训练 CLI（单人 headless/visible + worker 池） |
| `src/bin/neat_preview.rs` | 最优个体可视化回放 |
| `src/bin/training_capture.rs` | 训练 eval 路径截帧 + YOLO/OCR 标注 |
| `src/trainer/agent.rs` | 后台视觉+NEAT（`AgentController`） |
| `src/trainer/eval.rs` | 单人基因组评估（`evaluate_genome`） |
| `src/trainer/render.rs` | 离屏渲染 + `present_training_frame` |
| `src/game/sim.rs` | `new_training`、`tick_with_action` |
| `src/game/vision.rs` | `VisionPipeline::perceive` |
| `scripts/train_monitor.ps1` | 长跑监控与自动重启 |

### 11.2 外部参考

- NEAT 流程：`MarioRS/src/bin/mario_trainer.rs`
