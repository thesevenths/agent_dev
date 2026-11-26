# P2-03A — Action-conditioned JEPA 就绪计划 (Readiness Plan)

**结论先行：P2-03A 的模型/训练/评估架构设计已完整且可实施，但【当前不满足开训条件】。**
阻塞项不在模型层，而在**数据层与规格符合性**：实际 P1-A pilot 配置系统性偏离了已批准的
`P2_03_ACTION_JEPA.md` §4.4，且 pilot 数据存在单一初态、极端轨迹主导、数值精度边界三处限制。

本计划**保留 E0–E5 全部既有实验定义**（§3 原样引用），不实现任何训练代码、不生成全量数据、
不修改任何既有规格或配置。所有改动止于"计划 + 可测量验收标准"。

---

## 1. 开训条件判定（Gate 0 / Gate 1 复核结果）

| 前置条件 | 状态 | 证据 |
|----------|------|------|
| Gate 0：Git 与历史实验核对 | ✅ 完成 | `P1_A_BRANCH_INTEGRITY_REPORT.md`：feature 含 P2-02、0 冲突、无重复实现 |
| Gate 1：P1-A 科学验收 | ⚠️ **未通过** | `P1_A_SCIENTIFIC_REVIEW.md` S1–S5；`P1_A_TRAJECTORY_DISTRIBUTION.md` |
| **规格符合性（本计划新增发现）** | ❌ **未通过** | §2，实际配置偏离 `P2_03_ACTION_JEPA.md` §4.4 |
| 全量数据生成 | ⛔ 未开始（正确） | — |
| JEPA / E0–E5 训练 | ⛔ 未开始（正确） | — |

> **判定：暂缓 P1-B（全量数据）与 P2（E0–E4 训练）。** 需先完成 §2 的规格对齐与 §6 的数据前置整改。

---

## 2. ⚠️ 核心发现：实际 pilot 配置 vs 已批准规格（逐项核对）

`P2_03_ACTION_JEPA.md` §4.4 明确写道："All of the following MUST be explicit config keys and
**recorded in the dataset manifest**"，且 §4.4 的 AMENDED 段落要求 collapse 阈值
"MUST live in `configs/dynamics/controlled_pendulum_jepa.yaml` and MUST be frozen BEFORE any
`frozen_test` / `parameter_ood_test` result is inspected"。

对 `configs/dynamics/controlled_pendulum_jepa.yaml` 逐键实测：

> **AMENDED (P1-A' review fix P4).** 原表格把 `integrator` 与 `theta unwrapped`
> 直接标为 ✅，混淆了三个不同层次。独立 GitHub 审查指出：规格要求的是
> **「所有物理配置必须是显式配置项，并记录到 manifest」**，而当时 YAML 里
> 根本没有 `integrator` / `theta_representation` 两个键。现按三层重新判定：
>
> | 层次 | 当时实际情况 | 判定 |
> |------|--------------|------|
> | 代码实现行为（确为 RK4、确实不 mod 2π） | 源码可确认 | ✅ 通过 |
> | YAML 显式声明 | 两键均缺失 | ❌ 不通过 |
> | Manifest 记录完整配置 | 无完整 resolved config | ❌ 不通过 |
>
> **本轮修复**：v1 config 补齐 `integrator` / `theta_representation` 等显式键并加
> `resolved_pilot_config`（P3）；v2 config 补齐全部规格键且 collector 写入 manifest
> 完整 `resolved_config` + `git_commit`（见 `P1_B1_V2_PILOT_REPORT.md` §2–§3）。
> 下表「判定」列保留的是**修复前**的审计结论，作为历史记录不予篡改。

| 规格 §4.4 要求的键 / 值 | 规格 pilot 值 | 修复前实际 pilot 值 | 判定（修复前） |
|--------------------------|---------------|---------------|------|
| `gravity_g` | `9.81` | 9.81 / **1.62** | ⚠️ 部分偏差 |
| `dt` | `0.02` | `0.02` | ✅ |
| `integrator` | deterministic RK4 | 代码为 RK4，**但 YAML 无显式键** | ❌ 显式声明缺失（代码 ✅） |
| **`episode_steps`** | **`1000`** | **`horizon_steps: 200`** | ❌ **1/5，且键名不同** |
| `theta` unwrapped | required | 代码 unwrapped，**但 YAML 无显式键** | ❌ 显式声明缺失（代码 ✅） |
| **`theta0_range` / `omega0_range`** | **`[-π,π]` / `[-1,1]`** | **固定 `θ0=2.0, ω0=0.0`** | ❌ **退化为单点** |
| **`torque_bounds`** | **`[-1.0, 1.0]`** | **`action_max_torque: 5.0` → `[-5,5]`** | ❌ **5 倍，且键名不同** |
| `action_hold_steps` | required（piecewise-constant hold） | **无**（每步一个独立力矩） | ❌ 缺失 |
| `action_profiles` | 4 类，sign/duration 多样 | 4 类 | ✅ |
| **`mass` / `length` / `damping` 分布** | **uniform over train ranges §5.1**（m 0.8–1.2 / L 0.8–1.2 / b 0.05–0.15） | **5 个固定场景**，含 `m=3.0`、`L=0.3`、`b=1.0` | ❌ **远超 train range，无分布采样** |
| **`parameter_ood_ranges`** | 四组 × 75 episodes | **无** | ❌ 缺失 |
| `collision / NaN / Inf guards` | **assert finite at every RK4 step** | 仅 audit 事后检查，仿真器内**逐步断言缺失** | ⚠️ 部分（已由 P4 `ValidatedPendulum` 补齐） |
| `collapse.min_latent_std` | `0.1` | **无** | ❌ 缺失 |
| `collapse.max_collapsed_fraction` | `0.25` | **无** | ❌ 缺失 |
| `collapse.max_cov_offdiag` | `0.1` | **无** | ❌ 缺失 |
| `collapse.require_finite` | `true` | **无** | ❌ 缺失 |
| `latent_dim` / `block_size` / `batch_size` / `seeds`（§4.4 首表） | 8/16, 10, 128, 5 seeds | pilot config 中无（仅 `experiment.seed`） | ⚠️ 尚未到训练阶段，但需在 P1-B/P2 补齐 |

> **Q3 状态更新**：规格 GAP_ANALYSIS.md:171 要求「P1-A 用这些起始值跑 20 pilot 并证明稳定性」。
> 由于既有 8/8 PASS 是在**偏离规格的配置**上完成的，Q3 **不能沿用该结论关闭**；
> 本轮按 Q-A 恢复规格后，已在新的 20 条 V2 pilot 上重新完成 10/10 独立验收
> （`P1_B1_V2_SCIENTIFIC_REVIEW.md` §8），Q3 的数值稳定性部分据此可复核关闭。

### 2.1 根因链（why → why-not → what removes it）

```
规格: torque_bounds=[-1,1], 初态为分布, m/L/b 在 train range 内
  ↓ 实际实现改为 torque_max=5.0、单一初态、5 个极端离散场景
  ↓ short_rod: I=mL²=0.09 → τ_max/I=55.56 rad/s²（规格按 nominal I≈1 时仅 1 rad/s²）
  ↓ E_barrier=5.89 J 而 E0=4.17 J（71%），极易越顶
  ↓ S1_step 连续旋转 12.59 圈 / 79.11 rad
  ↓ 单条轨迹占全部 |θ| 的 53.5%，数据分布被强驱动 regime 主导
```

**因此 79 rad 不只是"物理参数组合"，更是规格漂移的结果。** 若按规格的 `torque_bounds=[-1,1]`
与 train ranges 采样，越顶旋转不会出现（`τ_max/I ≈ 1 rad/s²`，`E_barrier≈19.62 J`）。
这解释了 ChatGPT 提出的"不凭空规定最大力矩，应按惯量/势垒确定"——规格原本已经给出了标定值。

> **本轮不擅自改回规格值**（任务书明确"Do not arbitrarily change the simulator parameters"、
> "Review ... and preserve its E0-E5 experimental definitions"）。
> 规格对齐是**人工决策项**，列于 §7 待批问题 Q-A。

### 2.2 其余规格一致性

- §5.1 四组 OOD（mass/length/damping/joint，各 75 episodes）依赖 train ranges
  `m 0.8–1.2 / L 0.8–1.2 / b 0.05–0.15`。当前 pilot 的 `L=0.3`、`m=3.0`、`b=1.0`
  **不在任何 train/OOD 组定义内**，与 §5.1 的参数空间不兼容 → P1-B 必须重建参数采样。
- §5.2（已由 `0e305d6` 修订）的 E1/E2 容量差异说明、E2/E3/E4 等参数量约束，**保留不变**。
- §7 的 G0–G6 门控定义，**保留不变**（§4 原样引用）。

---

## 3. E0–E5 实验定义（原样保留，来自 `P2_03_ACTION_JEPA.md` §5）

| ID | Encoder | Predictor | Action | Role |
|----|---------|-----------|--------|------|
| E0 | — | Persistence | — | lowest baseline（locked） |
| E1 | MLP | unconditioned（legacy `Predictor(z)`） | none | ordinary JEPA |
| E2 | MLP | conditional | correct | **target model** |
| E3 | MLP | conditional | shuffled | negative control |
| E4 | MLP | conditional | zero tensor | action-contribution check |
| E5 | MLP | conditional | correct | multi-step rollout |

**必须同时保留的约束（不得在实现中放宽）：**

1. **E0 锁定**（§5.0）：`x_hat[t+h] = x[t]` raw state，无 encoder/predictor/probe/training；
   与所有模型在**相同 episode 与相同 timestep**上比较；`Skill = 1 - MSE_model/MSE_persistence`，
   **显式报告分母**；分母近零时返回 `skill=None` + degenerate flag，不除零、不静默裁剪。
2. **E1 用 legacy `Predictor(z)`**（实例化，不修改）；E4（零动作）是**不同实验**，不可用
   `condition=None` 代替（`ConditionalPredictor` 会拒绝 None）。
3. **E2/E3/E4 架构与参数量完全相同**，仅动作张量不同；E3 用固定种子 `shuffle_conditions`
   （已核实 `models/state_dynamics.py:429`，通用 helper，无领域常量）。
4. **E1/E2 训练预算与评估协议相同但参数量按结构不同**，容量差必须显式报告；
   主动作信息证据是 **E2 vs E3 / E2 vs E4**，E2 vs E1 是次要比较。
5. **G2/G3 主指标是 train-only Linear-Probe raw-state RMSE/MAE**（§6.1），
   latent loss 仅作诊断。每个变体独立在 TRAIN 上拟合 probe，用 VALIDATION 选超参，
   **绝不**在 `frozen_test` / `parameter_ood_test` 上拟合或调参。
6. 统计：5 seeds，episode-level paired bootstrap（10,000 重采样，95% CI），均值与中位数，
   相关时间步不得当独立样本。

---

## 4. 可测量验收标准（G0–G6 原样 + 逐实验增补）

### 4.1 既有门控（§7，保留）

| Gate | Criterion |
|------|-----------|
| G0 | RK4 用精确持久化的 action 重放得到 `x[t+1]`；manifest SHA-256 + 参数范围 + 四组 OOD 范围/计数 + split 隔离全部验证 |
| G1 | 按**预注册**阈值无 latent collapse（per-dim std / collapsed fraction / cov offdiag / finite） |
| G2 | E2 train-only Linear-Probe raw-state RMSE/MAE < E1，paired 95% CI 排除 0；报告容量差 |
| G3 | E2 < E3(shuffled)，paired 95% CI 排除 0 |
| G4 | E2 Linear-Probe raw-state RMSE < E0 Persistence，skill > 0，报告分母 |
| G5 | horizons 1/2/5/10/20 有限、不发散、open-loop 反馈 predicted latents |
| G6 | 四组 Parameter OOD 分别测量并如实报告退化（不设幅度 pass/fail） |

**Stage gate：G0 失败 ⇒ 不开 P2；P2 的 G1–G3 失败 ⇒ 不开 P4。**（§7 / §8 保留）

### 4.2 逐实验增补的可测量标准（本计划新增，不改原定义）

| 实验 | 必测指标 | 数值验收标准（预注册建议，待人工确认） |
|------|----------|----------------------------------------|
| **E0** | 分母 `MSE_persistence` 逐 horizon [1,2,5,10,20] | 必须 > 0 且显式列出；若 < `min_denominator` → `skill=None` |
| **E1** | `sum(p.numel())` encoder/predictor/total；collapse 指标；probe RMSE | 参数量记录完整；G1 通过；作为 E2 的次要对照 |
| **E2** | probe RMSE/MAE、latent MSE、per-dim std、cov offdiag、5 seeds 均值±CI | G1/G2/G3/G4 全部通过；`Δ_action = L_E1 - L_E2 > 0` 且 CI 排除 0 |
| **E3** | shuffle 后 probe RMSE；**E2 vs E3 的 paired CI** | 必须**回退**（E3 显著差于 E2），否则动作信息结论不成立 |
| **E4** | 零动作 probe RMSE；E2 vs E4 paired CI | 必须回退；E4 ≠ E3（二者是不同负对照，需分别报告） |
| **E5** | horizons 1/2/5/10/20 的 open-loop 误差曲线、发散标志 | h=1 与单步 target **逐元素相等**；h=20 有限无 NaN；误差单调不爆炸 |
| 全实验 | `experiment_id` 唯一性（`config_hash+variant+seed`）、git commit、dataset_version、config_hash | 每次运行唯一、不覆盖历史 |

### 4.3 ChatGPT 建议的三个评价维度 → 映射到既有门控（不新造实验）

| 建议维度 | 落点 | 说明 |
|----------|------|------|
| Raw-state Probe（R²/RMSE） | **已是 G2/G3/G4 主指标**（§6.1） | 无需新增，按规格执行即可 |
| Multi-step Rollout（1/5/10/20 误差是否发散） | **已是 G5**（§7） | 用 `ConditionalPredictor.rollout` 的 `[B,H,L]`（已核实 `conditional_predictor.py:148`） |
| Action Intervention（固定初态、改变未来动作序列） | **新增诊断项 D-A** | 见 §5.6；不替代 E0–E5，作为 E2 训练后的**定性+定量**附加检查 |

---

## 5. 实现计划（详细设计，本阶段不落地代码）

全部为 **NET-NEW** 文件（沿用规格 §9 清单），禁止修改
`models/world_model.py` / `predictor.py` / `conditional_predictor.py` / `utils/losses.py` /
`train.py` / `data/pendulum.py` / `pusher_v1` / 历史 checkpoint。

### 5.1 `dynamics/trajectory_windows.py`（薄适配层）

- **绝不重复实现窗口逻辑**：调用 `dynamics_batch_from_windows(window_len=K+1, stride=1)`
  （`dynamics/contracts.py:241-312`），它已保证 `obs[s:s+K+1]` / `cond[s:s+K]` 不跨 episode。
- 输出三元组：`context_block = flatten(obs[s:s+K])` `[B,K*D]`、
  `action_t = cond[s+K-1]` `[B,U]`、`target_block = flatten(obs[s+1:s+K+1])` `[B,K*D]`。
- **必须单测的恒等式**（规格 §2.2）：
  ```
  context_block[..., -D:] == x_t
  target_block[...,  -D:] == x_{t+1}
  target_block[..., :-D]  == context_block[..., D:]
  action_t                == u_t   where (x_t, u_t) -> x_{t+1}
  ```
- **多步**：`target_k_block = flatten(obs[s+k : s+k+K])`（**正好 K 帧**，不是 K+1），
  `action_seq = cond[s+K-1 : s+K-1+k]`，`z_pred_k = rollout(...)[:, k-1]`，
  `z_target_k = stop_gradient(target_encoder(target_k_block))`。
- **边界（规格 §2.2 强调，必须测）**：`0 <= s <= L - K - k_max`。
  `K=10, L=60, k_max=20` → `s <= 30`，不是 49；否则 horizon-20 被 NumPy 静默截断。
- **h=1 必须与单步 `target_block` 逐元素相等**（规格已给出数值验证表）。
- 测试文件：`tests/dynamics/test_trajectory_windows.py`。

### 5.2 `models/action_conditioned_world_model.py`

- 显式 `PREDICTOR_MODES = ("conditional", "unconditioned")`，**config-driven**
  （规格 §3 AMENDED item 8；不用 `isinstance`、不用 `predictor or ...` 默认）。
- `encoder` 复用 `models/encoder.py:76 Encoder(block_size, state_dim, hidden_dims, latent_dim, kind)`
  —— 它吃 `[B, K*D]` 展平块（已核实 `encoder.py:79-100`）。
- `predictor`：`conditional` → 复用 `ConditionalPredictor(latent_dim, condition_dim, hidden_dims)`
  （`conditional_predictor.py:79`）；`unconditioned` → 复用 legacy `Predictor(z)`。二者**只实例化，不修改**。
- `target_encoder = deepcopy(encoder)` + `requires_grad_(False)`，沿用 legacy EMA 模式
  （`world_model.py:67-69, 76-79, 128-134`），不改动 legacy 文件。
- `update_target_encoder(momentum)`；`encode` / `encode_target`（`@torch.no_grad()`）。
- **Q6 未决**：是否用 `predictor` 作 child 名（与 legacy state_dict 前缀冲突）。
  计划要求 `model_class` 字段 + 双向 checkpoint 加载测试；**Q6 关闭前不定案**。

### 5.3 训练 `scripts/train_action_jepa.py`（先单步 P2，再多步 P4）

单步（规格 §4.1，原样）：
```
z_context = model.encode(context_block)
z_target  = model.encode_target(target_block).detach()
z_pred    = model.predictor(z_context, action_t)
loss = pred_weight * prediction_loss({1:z_pred},{1:z_target},[1])
     + vicreg_loss(z_context, var_weight, cov_weight)     # utils/losses.py:83
optimizer.step(); model.update_target_encoder(ema_momentum(step,...))
```
- 已核实 API 名称：`utils/losses.py:27 prediction_loss`、`:83 vicreg_loss`、
  `state_dynamics.py:429 shuffle_conditions`、`conditional_predictor.py:148 rollout`。
- VICReg 只作用于 `z_context`（与 legacy `train.py:174-178` 一致）。
- 只有 `requires_grad=True` 的参数进 optimizer（`train.py:141-142`）。
- **每 epoch 记录**：EMA momentum、VICReg 系数、per-dim latent std、cov offdiag、
  逐 horizon loss、梯度范数、seed/device/dataset_version/config_hash/git commit/时长；
  `experiment_id` 按 `(config_hash, variant, seed)` 唯一，不覆盖。
- 多步**只在 G1–G3 通过后**开启：`horizons=[1,2,5,10]`，`prediction_loss(preds, targets, horizons)`，
  权重 `w_k` 来自 config，VICReg 不变；**不引入 Transformer 或额外正则**。

### 5.4 训练配置（沿用 §4.4，标注为未验证设计值）

`latent_dim` 8/16（**Q4 open**）、`encoder_hidden=(64,32)`、`predictor_hidden=(64,64)`、
`block_size=10`、`batch_size=128`、`lr=1e-3`、`weight_decay=1e-4`、
**5 个模型 seed 与 dataset_seed 严格分离**、CPU smoke 优先。
collapse 阈值（`0.1 / 0.25 / 0.1 / true`）**必须在 P1-B 前写入 config 并冻结**，
且阈值修订只能基于 train/validation 证据并逐次记录理由。

### 5.5 评估与消融 `scripts/evaluate_action_jepa.py`

- 按 §6.1 执行 train-only probe → validation 选超参 → frozen_test/OOD 不重拟合。
- 四组 Parameter OOD（§5.1）分别报告 G6；`shuffle_conditions` 固定种子。
- 5 seeds + episode-level paired bootstrap（10,000，95% CI），报均值与中位数。
- 输出 `P2_03A_FINAL_REPORT.md`（P6，结论先行 + 全表 + 数字可追溯性核验，沿用 P2-02 做法）。

### 5.6 新增诊断 D-A：Action Intervention（ChatGPT 建议，不改 E0–E5）

> **AMENDED (P1-A' review fix P1).** 原验收条件写「正/负力矩导致相反方向的角加速度」，
> 在 nominal 初态 `θ0=2.0, ω0=0` 下**不成立**，属于验收条件设计错误（非模拟器错误）。
> 角加速度 `α = −(g/L)sinθ − bω + τ/(mL²)`；代入 `g=9.81, L=1, m=1, b=0.1, θ=2, ω=0`
> 得基项 `−9.81·sin(2) = −8.920208 rad/s²`，于是 `τ=+2.5 → −6.420208`、
> `τ=−2.5 → −11.420208`，**两者同为负**。正力矩只是削弱重力项，未改变方向。
> 已改用 `θ0=0, ω0=0`（重力矩为零）使 `α = τ/(mL²)`，方向性才可直接检验。

固定**同一 context block**（同一 episode、同一起点 `s`，因此 `z_context` 完全相同）
与同一训练好的 E2，输入 4 组未来动作序列（全零 / 全正 / 全负 / 交替），
在 latent 与 probe raw-state 两个层面比较预测轨迹：

| 必测 | 修正后的验收条件 |
|------|------------------|
| **解析级前置断言**（新增，必做） | 取 `θ0=0, ω0=0`：正力矩 `α>0`、负力矩 `α<0`、零力矩 `α≈0`；解析值 `α=τ/(mL²)` 与 `ControlledPendulum.derivatives` 数值逐元素相等（单测 `test_action_intervention_analytic_signs`） |
| 相同 context block 下的 4 组预测终态 θ 不全相同 | 至少 3 个互不相同；**必须固定 `s` 与 `episode_id`**，只改变动作序列，避免把不同状态误当成动作效应 |
| 动作打乱后预测随之改变 | 与 E3 结论一致 |
| 预测差异方向与能量/做功一致 | 在**消除重力偏置的初态**（`θ0=0`）下，正/负力矩产生相反方向角加速度；在 `θ0=2` 等一般初态下只要求"差异方向与 `Δα=Δτ/I` 一致"，**不**要求符号相反 |
| **不宣称** | 不把此项当作"模型学会动力学"的替代证据；G2/G3 仍是主判据 |

D-A 仍是**附加诊断**，不替代、不改变 E0–E5 与 G0–G6 定义。

---

## 6. 数据层前置整改（P1-B 之前必须完成，均未执行）

按优先级排列，全部来自前两份审查报告的实证发现：

| # | 整改项 | 来源 | 为什么阻塞 |
|---|--------|------|------------|
| D1 | **规格对齐**：把 §4.4 全部强制键补进 config 并写入 manifest（`torque_bounds`、`episode_steps`、`theta0_range`/`omega0_range`、`m/L/b` 分布、四组 `parameter_ood_ranges`、`action_hold_steps`、4 个 `collapse.*` 阈值、逐步 finite guard） | 本报告 §2 | 否则 G0/G1 无法按规格执行，且配置漂移不可追溯 |
| D2 | **初态分布采样**：`θ0~U(-π,π)`、`ω0~U(-1,1)`，逐 episode 记录 | S1 / 规格 §4.4 | 单一初态使泛化结论无效 |
| D3 | **按 `τ_max/I` 与 `E_barrier` 标定力矩**，避免 5 场景控制强度相差 33 倍 | S4 / 分布报告 §6 | 极端 regime 主导 |
| D4 | **regime 预注册标签 + 分层**：`bounded/transitional/rotation`，stress-test 独立 `eval_only` 组 | 分布报告 §6.3 | 防止归一化与 loss 被单条轨迹绑架 |
| D5 | **short_rod 数值精度**：更小 `dt`（0.005 实测误差降至 1.1e-5）或降力矩 | S3 / `step_convergence.json` | 单步标签含 ~0.8% 积分器噪声 |
| D6 | **动作多样性**：增加 hold duration、连续幅值 profile、含 τ=0 的 episode | S5 | 防止模型靠识别 profile 类型取巧 |
| D7 | **episode 级分层划分**（scenario × profile × regime × IC bin），每 split 覆盖全部 regime；独立 `dataset_version` + manifest | Phase 4 | pilot 的 3 条 test 只覆盖 2 场景 |

---

## 7. 待人工决策的问题（本计划不擅自决定）

| ID | 问题 | 关联 |
|----|------|------|
| **Q-A** | 是否把 pilot 配置**改回规格 §4.4 设计值**（`torque_bounds=[-1,1]`、`episode_steps=1000`、train-range 参数采样）？还是保留当前更强驱动配置并**修订规格**？二者只能选一，不能默默并存 | §2 / D1 |
| **Q-B** | 第一阶段训练目标是否仅 bounded oscillation？strong-rotation 是否统一划为 eval-only stress-test？ | 分布报告 §6.2 |
| **Q-C** | `short_rod` 采用更小 `dt` 还是降低 `τ_max`？ | S3 / D5 |
| Q3（既有 open） | 物理参数范围的数值稳定性需在**修正后的配置**上重跑 20 pilot 才关闭 | GAP_ANALYSIS.md:171 |
| Q4（既有 open） | `latent_dim` 8 vs 16、`block_size` 10、first-round horizons | GAP_ANALYSIS.md:172 |
| Q6（既有 open） | 新模型 child 命名 + `model_class` + 双向 checkpoint 加载 | GAP_ANALYSIS.md:174 |

> **注意**：Q3 原本要求"P1-A 跑 20 pilot 并证明稳定性"。既有 8/8 PASS 是在**偏离规格的配置**上完成的，
> 因此若 Q-A 选择"改回规格值"，Q3 **必须重新验证**才能关闭，不能直接沿用本次结论。

---

## 8. 阶段顺序（沿用规格 §8，保留 stage gate）

| Stage | Work | Gate | 当前状态 |
|-------|------|------|----------|
| P0 / P0.1 / P0-fix | 审计 + 规格修订 + 校验加固 | — | ✅ 完成 |
| P1-A | ODE + RK4 + 20 pilot episodes | 数值稳定性 | ✅ 工程完成 / ⚠️ 科学待补 |
| **P1-A'（新增）** | 本轮 4 份审查报告 + 规格对齐 + D1–D7 整改 | **人工验收** | ◀ **当前位置** |
| P1-B | 全量数据（train/val/frozen/四组 OOD）+ manifest | manifest complete | ⛔ 暂缓 |
| P1-C | 数据验收：exact-action replay / 物理 sanity / split isolation | **G0 STOP GATE** | ⛔ 暂缓 |
| P2 | 单步 `ActionConditionedWorldModel` + E0–E4，5 seeds | G1–G3 | ⛔ 暂缓 |
| P3 | EMA + VICReg 诊断 | no collapse | ⛔ 暂缓 |
| P4 | 多步 JEPA + E5，horizons 1/2/5/10/20 | G4–G5 | ⛔ 暂缓 |
| P5 | train-only Linear Probe + 四组 Parameter OOD | G6 | ⛔ 暂缓 |
| P6 | `P2_03A_FINAL_REPORT.md` + feature 分支提交 | 独立复现 | ⛔ 暂缓 |

**Stage gate 保留**：G0 失败 ⇒ 不开 P2；P2 的 G1–G3 失败 ⇒ 不开 P4。

---

## 9. 限制与合规声明

- ✅ **保留** `P2_03_ACTION_JEPA.md` 的 E0–E5 全部定义、§5.0/5.1/5.2 AMENDED 约束、
  §6.1 主指标、§7 G0–G6、§8 阶段顺序；本计划**未修改该文档**（仅引用与审计）。
- 未修改任何模拟器参数、pilot 配置、pilot 数据、manifest、生产代码、受限 Core/Adapter 文件。
- 未生成全量数据集、未实现 `trajectory_windows.py` / `ActionConditionedWorldModel`、
  未写训练或评估脚本、未训练 JEPA、未实现 MPC、未合并 main、未开 PR。
- 新增内容仅：只读分析脚本 `scripts/analyze_p1a_pilot.py`、
  数字核验脚本 `scripts/verify_p1a_reports.py`、`analysis/` 下 JSON+PNG、4 份报告。

### 验证结果（本轮实测）

| 检查 | 命令 | 结果 |
|------|------|------|
| 全量回归 | `pytest tests/ --ignore=tests/test_pysr.py -q` | **255 passed + 101 subtests** |
| 既有 pilot audit | `python scripts/audit_controlled_pendulum.py --config ...` | **8/8 PASS**（未改动数据/管线） |
| 独立分析 | `python scripts/analyze_p1a_pilot.py --config ...` | 5 JSON + 6 PNG 全部生成 |
| 数字可追溯性 | `python scripts/verify_p1a_reports.py` | **PASS 28/28，0 虚构** |
| 受限文件 | `git diff --name-only HEAD -- <受限路径>` | **空**（零改动） |

## 10. 停止条件

**STOP，等待人工批准。** 在进入 P1-B（全量数据生成）或 P2（JEPA 训练）之前，
必须先就 §7 的 Q-A / Q-B / Q-C 与 §6 的 D1–D7 整改方案给出明确裁决。
