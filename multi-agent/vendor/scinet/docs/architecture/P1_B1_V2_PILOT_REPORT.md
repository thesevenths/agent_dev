# P1-B1 — Controlled Pendulum V2 Pilot 报告

**结论先行：按 Q-A 恢复原规格生成的 20 条 V2 pilot，独立验收 10/10 PASS；原 v1 pilot 完整保留、零改动。**
新 pilot 不再是"极端强驱动主导"，初态/物理参数/动作持续时间均具多样性。
**STOP：未生成全量数据集、未训练 JEPA、未实现模型或 MPC。**

---

## 1. Q-A / Q-B / Q-C 决策落地

| 决策 | 批准内容 | 本轮落地 |
|------|----------|----------|
| **Q-A 恢复原规格** | `torque_bounds=[-1,1]`、`episode_steps=1000`、θ0∈U(-π,π)、ω0∈U(-1,1)、m/L/b 在 train range 采样、g=9.81、dt=0.02 | ✅ `controlled_pendulum_v2_pilot.yaml` |
| **Q-B 主训练只有 bounded oscillation** | transitional 独立评估、continuous rotation 为 eval-only stress-test | ✅ v2 pilot 20/20 全为 bounded；v1 历史 pilot 保留为强驱动 stress 证据，不混入 |
| **Q-C short_rod 降 dt** | stress-test 用 dt=0.005，保持物理时长与力矩波形 | ✅ 200×0.02s = 800×0.005s = 4.0s；误差见 §5 |

**原 20 条 v1 pilot 未删除、未覆盖**（`datasets/controlled_pendulum_pilot/` 的 `npz_sha256`
仍为 `8d769ff5...`，`config_hash` 仍为 `41bb7d66...`）。

---

## 2. 完整显式配置（resolved，写入 manifest）

`configs/dynamics/controlled_pendulum_v2_pilot.yaml` → `manifest.json["resolved_config"]`：

| 键 | 值 | 来源 |
|----|----|------|
| `integrator` | `rk4` | 规格 §4.4 |
| `theta_representation` | `unwrapped` | 规格 §4.4 |
| `dt` | `0.02` | 规格（主分布，先验证） |
| `episode_steps` | `1000` | 规格 §4.4 |
| `torque_bounds` | `[-1.0, 1.0]` N·m | 规格 §4.4 |
| `theta0_range` | `U(-π, π)` | 规格 §4.4 |
| `omega0_range` | `U(-1, 1)` rad/s | 规格 §4.4 |
| `mass_range` | `[0.8, 1.2]` kg | 规格 §5.1 train range |
| `length_range` | `[0.8, 1.2]` m | 规格 §5.1 train range |
| `damping_rate_range` | `[0.05, 0.15]` 1/s | 规格 §5.1 train range（**归一化阻尼率**） |
| `gravity_g` | `9.81` m/s² | 主分布固定 |
| `action_profiles` | random / step / sinusoid / alternating / **zero** | 4 类 + 真正零力矩 |
| `action_hold_steps_options` | `[1, 2, 5, 10]` | 新增多样 hold 时长 |
| `per_step_finite_guard` / `parameter_validation` | `true` | P4 加固 |
| `collapse_thresholds` | `0.1 / 0.25 / 0.1 / true` | 规格 §4.4 预注册，冻结 |
| `parameter_ood_groups` | mass / length / damping / **joint**，各 75 | 规格 §5.1，**未擅自新增第五组** |

**四组 OOD 原样保留**：没有因为 v1 出现 `low_gravity` 就新增第五组。

---

## 3. 数据身份与 Manifest 完整性

| 项 | 值 |
|----|----|
| 输出目录 | `datasets/controlled_pendulum_v2_pilot/`（**新建，独立**） |
| `dataset_version` | `cpv2-ff53b15f8996` |
| `config_hash` | 由 `resolved_config` 内容寻址 |
| `git_commit` | `3143772cf8ede786254a01ee6d0013a4d4c243d1` |
| `npz_sha256` | `0f48914d8b1449adc923cdb474c453e114fb663d72f1ddf8bf758f65b4110810` |

> **`git_commit` 口径说明**：manifest 记录的是**数据采集时刻**的 HEAD
> (`3143772`)，即 P1-A 四份审查报告提交之后、本轮 P1-A'/P1-B1 提交之前。
> 这是准确的不可变历史记录，**不会**为了"对齐当前 HEAD"而重新生成数据
> （重生成会改变 `npz_sha256`，反而破坏可追溯性）。本轮修复提交在
> `3143772` 之后，其 SHA 见 `P1_A_REVIEW_FIX_REPORT.md` 与 git log。
| observations | `[20, 1001, 2]` **float64** |
| conditions | `[20, 1000, 1]` **float64** |
| valid_mask | `[20, 1000]` bool，全 True |
| `DynamicsBatch` float64 | ✅ 校验通过，20 000 个有效 transition |

**每条 episode 的 manifest 记录**（满足"拿到 manifest + 代码版本 + 数据文件即可完整复现每条轨迹"）：

`episode_id` / `seed` / `initial_state` / `physical_parameters`(g,L,m,b,I,b归一化语义) /
`action_profile`(全参数) / `action_hold_steps` / `action_stats` /
`regime` + `regime_detail`(预注册判据全部数值) / `split` / `episode_sha256` /
`final_theta_rad` / `max_abs_omega_rad_s`。

---

## 4. 数据多样性实测（对比 v1）

| 维度 | v1（历史强驱动） | **v2（本轮）** |
|------|------------------|----------------|
| 初始角度 θ0 | 固定 `2.0`（1 个值） | **20 个互不相同**，范围 `[-3.087, +3.133]`，std `2.1699` |
| 初始角速度 ω0 | 固定 `0.0` | **20 个互不相同**，范围 `[-0.937, +0.964]` |
| 质量 m | 5 个固定场景（含 3.0） | `U(0.8,1.2)` 采样，全部在 train range |
| 长度 L | 5 个固定场景（含 0.3） | `U(0.8,1.2)` 采样 |
| 阻尼率 b | 5 个固定场景（含 1.0） | `U(0.05,0.15)` 采样 |
| 重力 g | 含 `1.62`（low_gravity） | 主分布固定 `9.81` |
| 力矩 | `±5.0 N·m` | `[-1,1] N·m`，实测 `[-0.9981, 0.9998]`，**零越界** |
| 动作 hold | 每步独立 | `[1,2,5,10]` 多样 |
| 零力矩轨迹 | 无 | **有**（`zero` profile，实测 `zero_fraction=1.0`） |
| θ 全局 max / std | `82.115` / `12.685` | **`8.819`** / **`2.213`** |
| 单条 episode 占 \|θ\| | **53.5%** | **24.0%** |
| \|θ_final\| > 2π 的条数 | 3 | 1（见下方说明） |
| regime 分布 | 16 / 3 / 1 | **20 / 0 / 0**（全 bounded） |

> **需要说明的一个诚实细节**：v2 中有 1 条（`E02_sinusoid`）终态 `|θ_f|=6.352 > 2π`，
> 但它的 regime 仍是 **bounded_oscillation**，因为 θ_range < 2π 且 ω 多次换向——
> 这是"从接近 -π 的初态摆到 -2π 附近"，**不是连续旋转**（winding = -0.52）。
> 按预注册判据分类正确，不应与 v1 的 `S1_step`（winding +12.27）混为一谈。

---

## 5. Q-C：short_rod stress-test（dt=0.005）

对历史 v1 的 `S1_step`（只读），保持**相同物理时长 4.0 s**与**相同力矩波形**
（`np.repeat` 4 倍，200 步 × 0.02s → 800 步 × 0.005s），与 scipy DOP853 参考比较：

| 配置 | 步数 | dt | 最大状态误差 | 结论 |
|------|------|----|--------------|------|
| v1 原始 | 200 | 0.02 | **7.628e-03** | 强驱动下单步标签噪声偏大 |
| **Q-C stress** | 800 | **0.005** | **1.095e-05** | **误差降低 ×696.8** |

主训练集仍用 `dt=0.02`（实测 DOP853 误差仅 `1.65e-05`，见 §6），
因此**没有**在未记录数值精度证据的情况下改动主数据集 dt。

---

## 6. Phase 5 独立验收（10/10 PASS）

`datasets/controlled_pendulum_v2_pilot/audit_report.json`

| # | 检查 | 结果 | 关键实测 |
|---|------|------|----------|
| 1 | 用**持久化 conditions 数组**直接 exact replay | ✅ | 20/20 逐位一致，mismatches=[] |
| 2 | 独立 DOP853 交叉验证 | ✅ | 最大状态误差 **1.65e-05** < 1e-4（v1 为 7.63e-03，**改善 ~462×**） |
| 3 | 能量平衡 `dE/dt=τω-bIω²` | ✅ | 最大相对残差 **7.45e-04** < 1e-2 |
| 4 | 参数与动作边界 | ✅ | 0 违反；全部在 train range / `[-1,1]` |
| 5 | 逐步 finite-state | ✅ | 全有限；max \|Δθ\|/step = **0.1390** ≪ π |
| 6 | 预注册 regime 分类 | ✅ | 20/20 bounded；强驱动样本未混入 |
| 7 | 初态多样性 | ✅ | θ0 / ω0 各 **20 个**互不相同 |
| 8 | episode 级划分隔离 | ✅ | 14/3/3，零重叠，窗口源全在本 split |
| 9 | SHA-256 完整性 | ✅ | npz + 20 个 episode hash 全部匹配 |
| 10 | 报告数字可追溯性 | ✅ | `verify_p1a_reports.py` v2 **48/48 PASS**，全部数值比较 |

**产出文件**：

```
datasets/controlled_pendulum_v2_pilot/
├── pilot_episodes.npz
├── manifest.json
├── audit_report.json
└── analysis/
    ├── independent_verification.json
    ├── trajectory_distribution.json
    ├── action_information.json
    ├── split_proposal.json
    └── figures/{theta_trajectories,initial_state_diversity,
                action_and_delta_theta}.png
```

---

## 7. 限制与 STOP

- v2 pilot 是**验收产物**，不是可训练数据集（20 条、3 条 test 覆盖有限）。
- **未生成全量数据集**（train/validation/frozen_test/parameter_ood_test/stress_test 五个 split 尚未生成）。
- 未实现 `dynamics/trajectory_windows.py`、`models/action_conditioned_world_model.py`。
- 未实现 JEPA 训练 / E0–E5 / Linear Probe / MPC；Q6（checkpoint 命名）与 Q4（latent_dim）仍未关闭。
- 未修改 Pusher-v1、legacy checkpoint、受限 Core/Adapter 文件；未合并 main、未开 PR。
- **STOP，等待人工批准**后才会进入 P1-B2 全量数据与 P2-03A。

## 8. 复现

```bash
python scripts/collect_controlled_pendulum_v2.py \
    --config configs/dynamics/controlled_pendulum_v2_pilot.yaml
python scripts/audit_controlled_pendulum_v2.py
python scripts/analyze_p1b1_v2.py
python scripts/verify_p1a_reports.py
```
