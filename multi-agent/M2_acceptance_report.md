# ARR 1.1 — Milestone 2 验收报告：真实 Embodied 多轮闭环

> 依据 chatgpt 复查任务书 Phase 1-4 执行。本里程碑目标：**不只在“能跑一次”**，
> 而是把系统推进到“真实多轮、跨领域、能继承改进并能恢复”的可用产品。
> 本报告所有数字均来自 `_real_loop.py` 对**真实 Embodied 引擎**（1-D
> point-mass 机器人仿真，`robot_sim_v1` 冻结数据集）的实际运行，非预设。

## 1. 阻塞 bug 修复（真实多轮回路能跑起来的前提）

| # | 问题 | 根因 | 修复 | 验证 |
|---|------|------|------|------|
| B1 | 多轮回路 `CandidateConfigError: candidate directory already exists` | `run_dual_layer` 把所有轮的 domain campaign 目录写死为 `dual_layer`；跨轮后 baseline 被 bake 进新 config，导致 baseline/candidate 目录与上一轮**同路径冲突** | `research_cycle.py` 按轮命名空间化：`domain_campaign_id="dual_layer_r{round_no}"`，每轮隔离 | `_real_loop.py` 3 轮全绿；`build_baseline` 仍按 `baseline_config_sha256` 拒绝真正不同的 baseline |
| B2 | 真实引擎 train→evaluate 找不到 checkpoint | `embodied_compute` 训练后未调用 `record_checkpoint`，evaluate 阶段 `checkpoint_for_run` 无索引可查 | `embodied.py` 训练写完 `checkpoint.json` 后调用 `record_checkpoint(...)` | 探针 base_Q/cand_Q 由全 NaN 变为真实值 |
| B3 | 库内 `state_path` 为空解析到 CWD（不安全） | `run_loop` 未归一化空路径 | `controller.py`：空/None 归一化为 `<project>/state/loop_state.json` | `_real_loop.py` 用 `state_path=""` 跑通 |

## 2. 真实多轮回路验收（核心交付）

运行：`python _real_loop.py`，真实引擎 + 冻结数据集，harness 层用遵守闸门契约的 stub 驱动。

### 2.1 三轮真实研究轮（FULL RUN）

```
status: completed
rounds_completed: 3          # 3 个版本化轮次均完成
harness_evolve calls: 3     # 每轮真实调用一次 harness（证明跑了 3 轮）
final baseline: .../baseline_v2.yaml
final domain overrides: {train.lr: 0.01}
n_domain_accepted_total: 1   # 真实领域改进被接受 1 次
n_harness_accepted_total: 3  # harness 改进被接受 3 次
```

逐轮版本继承：
```
v1: domain={train.lr: 0.01} harness=[planner.strategy] -> baseline_v2.yaml
v2: domain={}               harness=[planner.strategy] -> baseline_v2.yaml
v3: domain={}               harness=[planner.strategy] -> baseline_v2.yaml
```

### 2.2 真实引擎评测（state_mse@1，越低越好）

第 1 轮 baseline `train.lr=0.003`（**注意**：验证脚本仅修改**拷贝**到临时工程的 config，
规范 `config.yaml` 的 `train.lr=0.02` 未被改动）：

| 轮 | baseline lr | candidate lr | seed50 state_mse@1 | seed51 state_mse@1 | 方向归一化判定 |
|----|------------|--------------|--------------------|--------------------|----------------|
| 1 | 0.003 | 0.003 | 0.004471 → 0.004471 | 0.004437 → 0.004437 | Δ0 → **REJECT** |
| 1 | 0.003 | 0.005 | 0.004471 → 0.003583 | 0.004437 → 0.003494 | 改善 → **ACCEPT** |
| 1 | 0.003 | 0.01  | 0.004471 → 0.002530 | 0.004437 → 0.002433 | 改善约 **44%** → **ACCEPT**（最终 bake） |
| 2 | 0.01  | 0.003 | 0.002530 → 0.004471 | 0.002433 → 0.004437 | 退化 → **REJECT** |
| 2 | 0.01  | 0.005 | 0.002530 → 0.003583 | 0.002433 → 0.003494 | 退化 → **REJECT** |
| 2 | 0.01  | 0.01  | 0.002530 → 0.002530 | 0.002433 → 0.002433 | Δ0 → **REJECT** |
| 3 | 0.01  | (同第2轮) | — | — | **REJECT**（无进一步改进） |

**关键结论**：方向感知判定正确——
- 第 1 轮真实改进被 **ACCEPT**，`train.lr 0.003→0.01` 使 `state_mse@1` 由 ~0.0044 降到 ~0.0025（≈44% 下降）。
- 第 2、3 轮诚实 **REJECT**：更小的 lr 反而抬高 MSE，等于基线则 Δ0，均非改进。
- 这正是 chatgpt P0#1 要求的“模型改善被正确判 ACCEPT、退化被正确判 REJECT”，且 `state_mse@1` 是 minimize 指标，方向归一化无误。

### 2.3 强制中断恢复（RECOVERY RUN）

预置 `current_round=2`（模拟第 1 轮完成并已 checkpoint、进程在第 2 轮前崩溃）：

```
harness_evolve calls: 2     # 只跑了第 2、3 轮 —— 第 1 轮**未被重跑**
rounds_completed: 1         # 恢复轮次自身的版本历史只新增 1 个（第 2 轮接受，第 3 轮诚实拒绝）
status: completed
```

证明恢复机制正确：崩溃后只重跑**未完成**的轮次，已完成轮次不重算（checkpoint 永远指向“下一轮要跑的轮”，时序为 *完成→递增→原子保存*）。

## 3. 回归测试（无回退）

| 套件 | 结果 |
|------|------|
| `research_runtime` 候选生命周期 + 双环隔离 (`test_unified_candidate` + `test_loop`) | 25/25 通过 |
| `research_runtime` 配置构建安全边界 (`test_candidate_config`) | 25/25 通过（含新增 `test_collision_with_different_config`） |
| `adaptive_runtime` harness 演化 (`test_harness_evolution`) | 19/19 通过 |
| `tests/rt_tests` campaign (`test_campaign_runner/resume/integration`) | 28/28 通过 |
| `adaptive_runtime` research_ingest | 7/7 通过 |

`CandidateConfigBuilder.build` 幂等性改动保留了“永不覆盖”安全边界：
- 相同输入（相同 config 哈希）**复用**既有实验目录（镜像 `build_baseline`）；
- 同路径**不同 config** 仍**拒绝**（新增 `test_collision_with_different_config` 覆盖）。

## 4. 提交与推送

- `08ebdaac` fix: 真实回路上游前置修复（embodied checkpoint 索引 + 安全默认 state_path）
- `f6f0e14c` feat: Milestone 2 —— 真实 Embodied 多轮闭环、恢复、候选幂等
- 已推送到 `feature/arr-0.1-domain-registry`（未 merge main，未开 PR）。

## 5. 下一步（按 chatgpt 路线图，非本阶段）

- M3：三领域（embodied / quant / scinet）统一接入。
- M4：Package H（仅当 M1-M3 真实多轮稳定后再做）。
- 真实 `harness_evolve`（含 heldout_task OOD 验证）接入 loop 需构造 heldout_task，列为后续。
