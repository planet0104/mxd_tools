# NEAT 游戏 AI 训练说明

本文档描述「复刻版冒险岛 + YOLO 视觉 + OCR 自身定位 + NEAT」的训练管线。  
代码：`src/bin/neat_trainer.rs`、`src/bin/neat_preview.rs`、`src/bin/training_capture.rs`、`src/neat/`、`src/trainer/`、`GameSim::new_training`。

---

## 1. 总体架构

```
┌──────────────────────── 主线程（GL + GameSim） ────────────────────────┐
│  neat_trainer / neat_preview：离屏渲染 → 非阻塞提交 RGB 帧              │
│  sim.tick(action) @ 60Hz（使用后台最新 Action）                         │
│  fitness = TrainingFitness.score                                      │
│  最优个体 → tmp/neat_best_genome.json                                  │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ try_submit_frame（队列满则丢帧）
┌───────────────────────────────▼───────────────────────────────────────┐
│  vision-neat-agent 后台线程（`src/trainer/agent.rs`）                   │
│  YOLO + OCR → obs → NEAT evaluate → Action + VisionStep               │
└───────────────────────────────────────────────────────────────────────┘

   GameSim::new_training + 4 装饰玩家 + 波次刷怪
   感知默认每 4 tick 一次（`--pace 4`），中间 tick 复用上一 Action
```

**创建训练实例：**

```rust
let sim = GameSim::new_training(map, seed);
```

与普通 `GameSim::new` 的区别见 §3。

---

## 2. 视觉感知（部署每帧 1 次；训练可降频）

**部署 / `neat_preview --pace 1`**：每个逻辑帧（60Hz）执行一轮 YOLO+OCR。  
**训练默认 `--pace 4`**：逻辑仍 60Hz `tick`，但每 4 tick 才 `perceive` 一次，中间帧复用上一帧观测（见 §9.3②）。

每轮感知一次性得到：全部检测框、自身 OCR 位置、NEAT 观测向量、本帧可见掉落（供计分）。

| 步骤 | 次数 | 说明 |
|------|------|------|
| 离屏渲染 | 1 | `render_target_to_rgb` |
| YOLO | **1** | 全部类别 |
| OCR | **1 轮** | 仅「玩家」框，匹配「光头强加强版」 |
| 观测编码 | 1 | 不再调用模型 |
| 计分提示 | 1 | `step.apply_fitness_hints(&mut sim)` 记录可见金币/药水框 |

```rust
use mxd_tools::neat::{evaluate, action_from_outputs};
use mxd_tools::game::{action_to_input, TrainingPaceConfig, OBS_DIM};

let pace = TrainingPaceConfig::fast(); // 训练；预览/部署用 realtime() 或 --pace 1
let mut last_obs = vec![0.0; OBS_DIM];

for tick in 0..max_ticks {
    if tick % pace.vision_interval_ticks == 0 {
        let step = pipeline.perceive(&rgb)?;
        step.apply_fitness_hints(&mut sim);
        last_obs.copy_from_slice(&step.observation.values);
    }
    let outputs = evaluate(genome, &last_obs);
    let action = action_from_outputs(&outputs); // 9 选 1 argmax
    sim.tick(&action_to_input(action));
    if sim.is_episode_over() { break; }
}
let fitness = sim.fitness.score;
```

---

## 3. 训练环境规则（`GameSim::new_training`）

### 3.1 装饰玩家（排除干扰 + 不影响训练节奏）

- 地图上固定 **4 个其他玩家**，在不同高度平台巡逻；精灵来自 `assets/player/` 固定池，名牌从 `TRAINING_NPC_NAMES` 洗牌分配（与「光头强加强版」无相近字，如「南港商人」「冒险萌新」等）。
- YOLO 会检出为「**玩家**」框 → 网络必须靠 **OCR 排除自身**。
- **免疫怪物**：装饰玩家无 HP，不参与 `check_mob_touch`，**不会被怪扣血或打死**。
- **偶发打怪（仅表现）**：约每秒 3% 概率对身前怪普攻一次；伤害**不可击杀**（怪物 HP 最低保留 1），**不掉落、不触发波次清空**，避免 NPC 抢怪影响 NEAT 个体练躲避与拾取。
- 实现：`src/game/npc.rs`（`NPC_ATTACK_CHANCE_PER_SEC` 等常量可调）。

### 3.2 怪物波次刷新（鼓励巡逻）

- 单个平台上的怪被击杀后 **不会立刻重生**。
- 当**全图所有怪物**都被杀光（死亡动画结束）后，**整图一波同时重生**。
- 迫使个体主动寻找仍有怪的平台，而不是守着一个点刷。
- 实现：`sim.rs` 中 `tick_mobs` 末尾，`mobs.is_empty()` 时 `spawn_mobs()`。

### 3.3 个体死亡

- **HP 归零 → 本局结束**（`GameModal::GameOver`）。
- NEAT 评估用 `sim.is_episode_over()` 判断终止；死亡后不再 `tick`。
- 喝药（`UsePotion`）是**盲喝**：训练初始 **0 瓶红药**，必须先 YOLO 看到地上药水并 `PickUp` 拾取，再按 `1` 使用。

### 3.4 掉落物

- 每只怪**必掉金币**（频率高，是主要得分来源）。
- 训练模式红药掉落率 **18%**（`TRAINING_POTION_DROP_CHANCE`）：够偶尔捡到学盲喝，又不过量，避免掩盖「躲避怪物」的训练重点。
- 拾取需主动按 **Z**（`PickUp`），走近不会自动捡。

---

## 4. 适应度（Fitness）

**部署目标**仍是 YOLO 可见拾取；训练额外加 **shaping** 打破全 0 适应度，便于 NEAT 早期进化。

```text
总分 = 拾取分 + 视觉shaping + memory_weight × 内存shaping
```

| 类别 | 条件 | 默认分值 |
|------|------|----------|
| **拾取（主分）** | YOLO 见金币/药水 + `PickUp` + 实际捡到 | 金币=面值，红药 +50 |
| **视觉 shaping** | obs 有敌人槽 + `Attack` | +1 |
| **视觉 shaping** | obs 有掉落槽 + `PickUp` | +1 |
| **内存 shaping** | 命中怪物（×`memory_weight`） | +3 |
| **内存 shaping** | 击杀怪物（×`memory_weight`） | +15 |

CLI：`--fitness-shaping 0.2`（默认 0.2；设 `0` 关闭内存 shaping，保留视觉 shaping）。

**不计分**：纯走动/跳跃、未出现在 YOLO 框内的拾取、装饰 NPC。

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

装饰玩家**不进入**观测向量（仅作为 YOLO「玩家」干扰项，靠 OCR 排除）。

---

## 6. 动作空间（9 选 1，每逻辑帧一个离散动作）

| Action | 按键 | 训练意图 |
|--------|------|----------|
| Noop | — | 等待 |
| Left / Right | A/D | 移动、躲避、靠近掉落 |
| Jump | Space | 跳台 |
| Attack | J | 盲打（靠 YOLO 敌人框） |
| PickUp | Z | 盲捡金币/药水 |
| UsePotion | 1 | 盲喝（维持 HP 以继续捡） |
| Up / Down | W/S | 抓绳/爬梯、沿绳升降 |

网络输出 9 个 sigmoid，取 **argmax** 映射为单一 `Action`（见 `action_from_outputs`）。

---

## 7. 单局评估伪代码（`src/trainer/eval.rs` + `agent.rs`）

主线程只做渲染与模拟；YOLO+OCR+NEAT 在 `vision-neat-agent` 线程。

```rust
let mut agent = AgentController::spawn(pipeline, genome.clone());
let mut sim = GameSim::new_training(map, seed);

for tick in 0..max_ticks {
    agent.poll(&mut sim);  // 非阻塞收取后台 Action / 计分提示

    if tick % pace.vision_interval_ticks == 0 {
        let rgb = capture_render_rgb(&assets, &sim, &rt).await; // 主线程 GL
        agent.try_submit_frame(tick, rgb);  // 非阻塞；工人忙则丢帧
    }

    let action = agent.action();
    sim.tick_with_action(&action_to_input(action), Some(action));
    if sim.is_episode_over() { break; }
}
return sim.fitness.score;
```

---

## 8. 运行与验证

```powershell
# 训练环境截图（含 4 装饰玩家，与 YOLO 训练画面一致）
cargo run --release --bin mini_game_headless -- --training --screenshot out.png

# 手动操作 + YOLO 叠加（非 NEAT）
cargo run --release --bin mini_game -- --vision-preview --model models/yolo_nangang_e3000_best.onnx

# NEAT 最优个体可视化回放（60Hz 流畅；YOLO 在后台）
cargo run --release --bin neat_preview

# 从训练 eval 同路径截取一帧（YOLO+OCR 标注），见 §8.1
cargo run --release --bin training_capture -- --seed 42 --capture-tick 800 --pace 4

# 单测
cargo test --lib
```

### 8.1 训练路径截图验证（`training_capture`）

与 `neat_trainer` / `evaluate_genome` **同一条 eval 路径**：`GameSim::new_training` → 离屏渲染 → YOLO+OCR → 在指定 tick 保存原图与标注图。用于人工核对 YOLO 框、OCR 自身定位、装饰 NPC 是否干扰，**不需要先跑完整训练**（无基因组时用 `random_minimal` 或 `tmp/neat_best_genome.json`）。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--seed` | 42 | 本局 `episode_seed`（与训练 worker 一致） |
| `--capture-tick` | 800 | 在哪一逻辑 tick 的感知帧截图 |
| `--max-ticks` | `capture-tick + 400` | 最多模拟 tick（早死会提前结束） |
| `--pace` | 4 | 感知间隔，应与训练一致 |
| `--model` | `models/yolo_nangang_e3000_best.onnx` | YOLO 模型 |
| `--genome-file` | （无） | 基因组 JSON 或 `BestGenomeSnapshot`；省略则用 `--seed` 随机最小基因组，或回退 `tmp/neat_best_genome.json` |
| `--out` | `tmp/training_capture/YYYYMMDD_HHMMSS/` | 输出目录 |

```powershell
# 基础：随机基因组，第 800 tick 截帧（训练环境含 4 装饰 NPC）
cargo run --release --bin training_capture -- --seed 42 --capture-tick 800 --pace 4

# 用已训练最优个体 + 与训练相同的 seed/pace
cargo run --release --bin training_capture -- ^
  --genome-file tmp/neat_best_genome.json ^
  --seed 42 --capture-tick 1200 --pace 4

# 指定输出目录
cargo run --release --bin training_capture -- --capture-tick 500 --out tmp/my_capture
```

**输出文件**（在 `--out` 目录下）：

| 文件 | 内容 |
|------|------|
| `frame_raw.png` | 1368×768 原图（与训练离屏渲染一致） |
| `frame_yolo_ocr.jpg` | YOLO 框 + OCR 自身名牌标注 |

终端会打印：tick、fitness、YOLO 各类计数、OCR 是否找到「光头强加强版」及坐标。

**与 `mini_game_headless --training --screenshot` 的区别**：后者是**静态开局**一帧；`training_capture` 会先按 NEAT 驱动模拟到 `capture-tick`，再截该时刻的感知帧，更接近真实训练画面。

**注意**：

- `capture-tick` 必须是 `--pace` 的整数倍（或在感知帧上），否则该 tick 不会触发 YOLO，截取会失败。
- **耗时**：每次感知约 **100～200ms**（CPU YOLO+OCR，首帧含 OCR 加载更慢）。`capture-tick 800`、`pace 4` 约需 **200 次感知**。快速验证请用小 tick：

```powershell
# 快速验证（约 10 次感知，十几秒～半分钟）
cargo run --release --bin training_capture -- --seed 42 --capture-tick 40 --max-ticks 80 --pace 4
```

运行中会打印 `[当前/总数] 渲染+感知 tick …` 进度；无输出时是在等待 YOLO+OCR，属正常。

---

## 9. 训练加速（headless 提速，部署仍 60Hz）

### 9.1 当前单步耗时（CPU，1368×768）

| 环节 | 约耗时 | 说明 |
|------|--------|------|
| 离屏渲染 + `next_frame` | 5～15ms | OpenGL；headless 可尽量去掉 vsync 等待 |
| YOLO (`detect_rgb8`) | **~25～28ms** | ORT CPU/CUDA 差距 <10%（见 §9.3③） |
| OCR 名牌 | **~37～40ms** | CPU EP，当前**最大瓶颈** |
| 观测编码 | <1ms | 纯 CPU |
| NEAT 前向 | <1ms | 94 维小网络可忽略 |
| `GameSim::tick` | <0.1ms | 可忽略 |
| **合计** | **~62～67ms/次感知** | 本机实测（`find_player --bench`） |

若 **每逻辑帧都感知**（60Hz）：约 **16 步/秒/线程**，一局 3600 tick（60 秒游戏时间）≈ **3.8 分钟/基因组**（仅视觉，不含种群规模）。

NEAT 种群 300 × 50 代 ≈ 15000 次评估 → 单线程约 **950 小时**量级，**必须并行 + 降感知频率**。

### 9.2 原则：什么能加速、什么不能动

| 项目 | 训练 | 部署/预览 | 说明 |
|------|------|-----------|------|
| 逻辑物理 `LOGIC_HZ=60` | **保持** | **保持** | 跳跃、击退、攻击冷却与真实一致 |
| YOLO+OCR 频率 | **可降低** | **每帧 1 次** | 训练少跑视觉；上线仍 60Hz 全感知 |
| 输入分辨率 1368×768 | 保持 | 保持 | 与 `screen_caps` / 模型一致 |
| 观测向量定义 | 保持 | 保持 | 同一套 94 维 |

**不要**通过把 `LOGIC_DT` 改成 30Hz 来「加速」——会破坏物理，部署时对不上。

### 9.3 推荐策略（按收益排序）

#### ① 种群并行评估（收益最大）

- `--workers 1`（默认）：单进程顺序评估，一个 **1×1 隐藏 GL 窗** + 离屏渲染（见 `src/headless_gl.rs`）。
- `--workers N`（N>1）：训练开始时启动 **N 个常驻 worker 子进程**（`--worker-daemon`），各持 1 隐藏窗 + 独立 YOLO+OCR；每局 eval 经 stdin/stdout 派任务，**进程与窗口全程复用**，不再每局 spawn/销毁。
- 例：`--workers 16 --population 50` → 固定 16 个 worker 窗，完成一个基因组立即接下一个。
- N 路并行约 **N×** 吞吐（YOLO+OCR 各一份，内存 ×N）。

#### ② 视觉降频（`TrainingPaceConfig`，约 3～4×）

逻辑仍 **每帧 `tick` 60 次/秒**，但 **每 N 帧才 `perceive` 一次**，中间帧复用上一帧观测决策：

```rust
use mxd_tools::game::TrainingPaceConfig;

let pace = TrainingPaceConfig::fast(); // vision_interval_ticks = 4
let mut last_obs = vec![0.0; OBS_DIM];

for tick in 0..max_ticks {
    if tick % pace.vision_interval_ticks == 0 {
        // render → perceive（~63ms）
        let step = pipeline.perceive(&rgb)?;
        step.apply_fitness_hints(&mut sim);
        last_obs.copy_from_slice(&step.observation.values);
    }
    let outputs = evaluate(genome, &last_obs);
    let action = action_from_outputs(&outputs);
    sim.tick(&action_to_input(action));
}
```

- `vision_interval_ticks = 4` → 视觉调用减 **75%**，同一段游戏时间（tick 数）墙钟约 **快 3～4 倍**。
- 部署时 `vision_interval_ticks = 1`，网络每帧拿到新观测（通常 **≥ 训练刷新率**，可接受）。
- 配置类型：`src/game/config.rs` → `TrainingPaceConfig::fast()`。

#### ③ YOLO GPU（本机实测：**收益极小，不推荐优先**）

**2026-08-26 复测**（`yolo_nangang_e3000_best.onnx`，1368×768，`find_player --bench --bench-iters 30`）：

| 环节 | CPU | CUDA (`--features cuda`) |
|------|-----|--------------------------|
| YOLO 仅推理 | **27.0ms** | 24.9ms |
| OCR | **40.0ms** | 36.8ms（仍走 CPU EP） |
| **YOLO+OCR 合计** | **67.1ms** | 61.7ms（约 **8%** 快） |

Rust `yolo_infer --bench` 11 张图：CPU avg **27.9ms**，CUDA avg **26.1ms**。

对比 **Python Ultralytics**（同 `.pt` 模型）：CPU **55ms**，CUDA **27ms**（PyTorch GPU 约 2×）。  
Rust 走 **ONNX Runtime**，CPU 已高度优化（~28ms），CUDA EP 对小 batch、单张 640 letterbox 几乎带不动，且 OCR 仍在 CPU，全链路瓶颈不在 YOLO。

**结论**：训练加速请优先 **视觉降频（§9.3②）+ 多 worker 并行**；不必为 ORT 强上 CUDA。若未来 OCR 也上 GPU 或 batch 推理，再复测。

#### ④ Headless 去掉帧等待

- 训练循环中 `next_frame().await` 仅用于 GL 上传；conf 设 `vsync: false`，不要人为 `sleep`。
- 收益相对视觉推理较小（~10%），但实现简单。

#### ⑤ OCR 优化（次要）

- 仅对 YOLO「玩家」框做 OCR（已实现）。
- 若本帧无玩家框可跳过 OCR（训练早期可能失败，慎用）。
- 长期：GPU OCR 或与 YOLO 批处理；当前瓶颈在 CPU det+rec。

### 9.4 组合估算（单局 3600 tick）

| 配置 | 约墙钟/基因组 | 相对实时单线程 |
|------|----------------|----------------|
| 60Hz 感知，CPU，1 线程 | ~230s | 1× |
| 15Hz 感知（N=4），CPU，1 线程 | ~65s | ~3.5× |
| 15Hz 感知，CUDA YOLO，1 线程 | ~62s | ~3.7×（CUDA 增益可忽略） |
| 15Hz 感知，CPU，8 worker | ~8s | ~28× |
| 15Hz 感知，CPU，16 worker | ~4s | ~55×（约线性，受 RAM/IO 限制） |
| 15Hz 感知（N=6），CPU，16 worker | ~3s | ~70×（部署前需 `neat_preview --pace 1` 验证） |

种群 300 × 50 代，用 **16 worker + pace 4** 可把单线程数百小时压到 **数小时～一夜** 量级（20 逻辑线程机器；仍取决于 `max_ticks` 与早停）。

### 9.5 部署验证（训练加速后必做）

训练可用 `TrainingPaceConfig::fast()`（`--pace 4`），但 **最优个体上线前** 必须用全帧感知回放：

```powershell
cargo run --release --bin neat_preview -- --pace 1
```

确认盲打、盲捡、爬绳/爬梯、OCR 自身定位仍正常。`mini_game --vision-preview` 仅用于手动操作 + YOLO 调试，**不跑 NEAT 网络**。

---

## 10. 训练运行

### 10.1 `neat_trainer`（训练）

macroquad/miniquad 的 WGL **必须绑定 HWND**，无法做到完全无窗；训练进程会创建 **1×1 占位窗** 后立即 `ShowWindow(SW_HIDE)` 并从任务栏移除（`src/headless_gl.rs`）。实际 YOLO 输入仍来自 **1368×768 离屏 `render_target`**。预览请用 §10.2 的 `neat_preview`。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--generations` | 50 | 进化代数 |
| `--population` | 50 | 种群大小（续训时以检查点为准） |
| `--workers` | 1 | `1`=顺序；`N>1`=最多 N 个子进程并行评估 |
| `--pace` | 4 | 每 N tick 感知一次（`TrainingPaceConfig::fast()`） |
| `--fitness-shaping` | 0.2 | 内存 shaping 权重（命中/击杀）；0=仅拾取+视觉 shaping |
| `--max-ticks` | 3600 | 单局最大逻辑帧（约 60 秒） |
| `--seed` | 42 | 随机种子 |
| `--model` | `models/yolo_nangang_e3000_best.onnx` | YOLO 模型 |
| `--checkpoint` | `tmp/neat_checkpoint.json` | 种群 + 创新号检查点 |
| `--best-genome` | `tmp/neat_best_genome.json` | 最优个体快照（供 preview 热加载） |
| `--fresh` | — | 忽略已有检查点，重新开训 |

内部子进程（勿手动调用）：常驻池 `--worker-daemon --worker-id N`；单次调试 `--worker-eval --genome-file ...`。

```powershell
# 短跑验证（建议首次，动作空间 9 维需 --fresh 开新训）
cargo run --release --bin neat_trainer -- --fresh --generations 2 --population 4 --max-ticks 200 --pace 4

# worker / 内存压测（开训前建议跑一轮，观察任务管理器）
cargo run --release --bin neat_trainer -- --fresh --generations 2 --population 32 --workers 16 --pace 4 --max-ticks 500

# 正式训练 — 多核 CPU 推荐（约 16～20 逻辑线程；续训去掉 --fresh）
cargo run --release --bin neat_trainer -- --generations 100 --population 50 --workers 16 --pace 4 --max-ticks 3000

# 更激进（内存充足时）：pace 6 + 略小种群，单局更短
cargo run --release --bin neat_trainer -- --generations 100 --population 40 --workers 18 --pace 6 --max-ticks 2400

# 自定义最优基因组输出路径
cargo run --release --bin neat_trainer -- --best-genome tmp/my_best.json --generations 10
```

**多核 `--workers` 怎么选（瓶颈在 YOLO+OCR，不在 NEAT）：**

| 逻辑线程 | 建议 `--workers` | 说明 |
|----------|------------------|------|
| 4～8 | 4～6 | 留线程给系统 |
| 16～20 | **16～18** | 每 worker 一子进程，各占一份 YOLO+OCR（内存约 0.5～1GB/路） |
| 32+ | 18～24 | 再增 worker 收益递减，注意 RAM 与磁盘 IO |

优先调 **`--workers`**（近似线性加速），其次 **`--pace 4→6`**（少跑视觉，部署前用 `neat_preview --pace 1` 验证），再考虑 **`--max-ticks 3000→2400`**（多数局会早死时可缩短上限）。**不要**为加速改 `LOGIC_HZ` 或强上 ORT CUDA（见 §9.3③）。

**检查点续训**：去掉 `--fresh` 即从 `tmp/neat_checkpoint.json` 恢复种群与创新号（`InnovationState`）。旧版无创新号字段的检查点会自动从种群连接重建。

**最优快照**：适应度刷新时原子写入 `--best-genome`；`generation` 字段统一为 `population.generation`。

### 10.2 `neat_preview`（可视化回放）

与 `neat_trainer` **独立进程**，边训练边预览最优个体。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--genome` | `tmp/neat_best_genome.json` | 最优基因组 JSON |
| `--model` | `models/yolo_nangang_e3000_best.onnx` | YOLO 模型 |
| `--pace` | 4 | 感知间隔 tick（与训练默认一致）；部署验证用 `1`（CPU 上很慢） |
| `--seed` | 0 | 局种子；`0` 表示用快照内 `training_seed` |
| （默认） | watch 开 | 文件更新后热加载；`--no-watch` 关闭 |

```powershell
# 边训练边热加载（另开终端）
cargo run --release --bin neat_preview

# 全帧感知部署验证
cargo run --release --bin neat_preview -- --pace 1

# 指定基因组、关闭热加载
cargo run --release --bin neat_preview -- --genome tmp/my_best.json --no-watch
```

窗口约为逻辑分辨率 1/3（456×256），含 YOLO 框叠加与 HUD（适应度、动作、本局得分）。`R` 重开一局；死亡后自动重开。

**线程模型**（与训练 eval 共用 `AgentController`）：

- **主线程**：60Hz `sim.tick` + 绘制；每 `pace` tick 离屏渲染并 `try_submit_frame`（不等待 YOLO）。
- **后台 `vision-neat-agent` 线程**：YOLO+OCR → NEAT → 回传 `Action`；主线程 `poll()` 非阻塞收取。
- YOLO 框/HUD 动作可能略滞后于画面（视觉线程慢时），但游戏不再因推理卡住。
- `--pace 1` 时视觉线程仍按全帧频率工作，框更新慢属正常，模拟与动画保持流畅。

适应度仅来自 `sim.fitness.score`（YOLO 可见拾取），`ground_truth()` 不参与 NEAT 计分。

### 10.3 `--profile`（单个体 eval 耗时剖析）

不跑完整 NEAT 进化，只跑 **一个基因组** 的 eval 循环并打印逐步耗时，用于定位瓶颈（GL 读回 vs YOLO+OCR vs sim.tick）。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--profile` | 关 | 开启剖析模式（与 `--fresh` / 正常训练互斥） |
| `--profile-ticks` | 32 | 剖析 tick 数（建议 16～64，含若干次感知帧） |
| `--seed` | 0 | 局种子 |
| `--pace` | 4 | 感知间隔，与训练一致 |
| `--genome-file` | 无 | 指定基因组；默认 `tmp/neat_best_genome.json`，不存在则用 `random_minimal` |

```powershell
cargo run --release --bin neat_trainer -- --profile --profile-ticks 32 --pace 4 --seed 42
```

输出分四段：

| 阶段 | 含义 |
|------|------|
| `eval_loop` | 主线程 32 tick 循环（渲染 + submit + sim.tick） |
| `drain` | 等待视觉线程处理完已提交帧（含 YOLO+OCR 冷启动） |
| 感知 tick 均值 | GL draw / present / readback、submit 耗时 |
| 视觉线程 | 队列等待、perceive(YOLO+OCR)、NEAT 前向 |

**典型结果（pace=4，CPU YOLO+OCR，32 tick）**：

- 主循环 ~80ms（sim.tick 可忽略；感知 tick 上 GL readback ~7ms）
- YOLO+OCR 暖机后 ~100～120ms/帧；首帧 ~190ms（含 OCR 模型加载）
- 视觉线程慢于主线程时，后续 `submit` 会 **DROP**（队列深度 2），属预期行为
- NEAT 前向 ~0.01ms，可忽略

完整训练慢的主因是 **感知次数 × worker 吞吐**（如 3000 tick、pace 4 → 750 次 YOLO），而非单次推理 1～3 秒。

---

## 11. 参考

### 11.1 可执行文件

| 文件 | 用途 |
|------|------|
| `src/headless_gl.rs` | 1×1 隐藏 GL 窗 + `swap_interval=0`（WGL 占位，非游戏画面） |
| `src/bin/neat_trainer.rs` | NEAT 训练 CLI（隐藏 GL 窗 + 离屏渲染） |
| `src/bin/neat_preview.rs` | 最优个体可视化回放 |
| `src/bin/training_capture.rs` | 训练 eval 路径截取一帧 + YOLO/OCR 标注（§8.1） |
| `src/bin/mini_game_headless.rs` | 截图/帧导出（加 `--training` 含装饰 NPC） |
| `src/bin/mini_game.rs` | 手动游玩 + YOLO 调试预览 |
| `src/bin/find_player.rs` | YOLO+OCR 找玩家 benchmark |
| `src/bin/yolo_infer.rs` | YOLO 批量推理 |

### 11.2 训练 / NEAT 核心

| 路径 | 内容 |
|------|------|
| `src/trainer/agent.rs` | 后台视觉+NEAT 线程（`AgentController`） |
| `src/trainer/render.rs` | 主线程离屏渲染读回 |
| `src/trainer/eval.rs` | 单基因组评估循环 |
| `src/trainer/mod.rs` | 训练模块入口 |
| `src/neat/genome.rs` | 基因组、变异、创新号 |
| `src/neat/network.rs` | 前向传播、`action_from_outputs` |
| `src/neat/population.rs` | 物种、进化、检查点 |
| `src/neat/snapshot.rs` | 最优基因组快照 |

### 11.3 游戏 / 视觉

| 路径 | 内容 |
|------|------|
| `src/game/sim.rs` | `new_training`、`is_episode_over`、波次刷怪 |
| `src/game/npc.rs` | 装饰玩家 |
| `src/game/fitness.rs` | 视觉计分 |
| `src/game/vision.rs` | YOLO+OCR 管线 |
| `src/game/observation.rs` | 94 维观测编码 |
| `src/game/action.rs` | 9 维离散动作 |
| `src/game/config.rs` | `TrainingPaceConfig` |
| `src/game/view.rs` | 离屏渲染、绘制 |

### 11.4 外部参考

- NEAT 流程：`MarioRS/src/bin/mario_trainer.rs`
