# P1-B1 — V2 Pilot 独立科学复核报告

**结论先行：V2 pilot 在物理正确性、数值精度、数据完整性、初态多样性与划分隔离上均通过独立验证；
但它仍只是验收产物，不能作为"已证明 JEPA 可学"的证据。**

本报告不依赖 collector 自身的输出结论，采用外部基准与独立重算。所有数字可回溯到
`datasets/controlled_pendulum_v2_pilot/analysis/*.json` 与 `audit_report.json`。

---

## 1. 独立验证方法

| 项目 | 方法（独立于被审对象） |
|------|------------------------|
| ODE 参考解 | `scipy.integrate.solve_ivp(method="DOP853", rtol=1e-12, atol=1e-14)`，**重新实现** `f(t,y)`，每个控制步恒定力矩 |
| 重放 | 直接读取**持久化的 `conditions` 数组**重积分，不用重新生成动作 |
| 能量 | 解析恒等式 `dE/dt = τω - b·I·ω²`（`I=mL²`），梯形近似比对 `ΔE` |
| 完整性 | `hashlib.sha256` 直接重算 npz 字节 + 逐 episode 数组字节 |
| 契约 | `dynamics.contracts.DynamicsBatch(float64)` 构造并触发 `validate_dynamics_batch` |
| 配置一致性 | `scripts/check_config_impl_consistency.py` 校验声明 vs 实现 |

> **表述边界（采纳 ChatGPT 技术判断）**：DOP853 交叉验证支持的是
> **「方程实现与数值积分的一致性」**。它不能仅凭一次数值比较就证明所有物理建模假设
> （点质量、无摩擦轴承、刚性无质量杆、恒定阻尼率）都与真实实验装置一致。
> 本报告不声称完成了真实系统的物理标定。

---

## 2. 物理方程与单位（全部显式）

```
theta_dot = omega
omega_dot = -(g/L) sin(theta) - b*omega + torque/(m*L^2)
I = m*L^2
```

| 量 | 单位 | 备注 |
|----|------|------|
| θ | rad，**unwrapped** | 从不 mod 2π |
| ω | rad/s | |
| τ | N·m | 被 I 归一化 |
| g | m/s² | 9.81（主分布固定） |
| L | m | U(0.8, 1.2) |
| m | kg | U(0.8, 1.2) |
| **b** | **1/s，归一化阻尼率 = c/(mL²)** | **不是**物理粘性系数 c [N·m·s]；已在 config 与 manifest 注明 |
| dt | s | 0.02（主）/ 0.005（stress） |

`alpha = -(g/L)sinθ - bω + τ/(mL²)`；参数校验 `m>0, L>0, dt>0, g>0, b>=0`、全部 finite。

---

## 3. 数值精度

| 检查 | v1（历史） | **v2（本轮）** |
|------|-----------|----------------|
| 独立 DOP853 最大状态误差 | 7.628e-03（short_rod 主导） | **1.6515e-05** |
| 能量平衡最大相对残差 | 1.163e-03 | **7.4512e-04** |
| max \|Δθ\| / step | 1.0255 rad | **0.1390 rad** |
| 是否需缩小主 dt | 是（short_rod） | **否**（主分布 dt=0.02 已足够） |

改善来自 Q-A 恢复规格（力矩从 ±5 降到 ±1、`τ_max/I` 从 55.56 回到 ~1），
而不是更换积分器。**主数据集 dt 未改动**，符合任务书要求。

Q-C stress-test（dt=0.005，保持 4.0 s 物理时长与相同力矩波形）：
误差 7.628e-03 → **1.0946e-05**，降低 **×696.8**。

---

## 4. 数据分布（独立重算）

| 量 | min | p50 | p95 | max | std |
|----|-----|-----|-----|-----|-----|
| θ (rad) | -9.0752 | -0.0065 | 2.9852 | **8.8188** | **2.2127** |
| ω (rad/s) | -6.6593 | -0.0031 | 4.2894 | 6.9502 | 2.4951 |
| τ (N·m) | -0.9981 | 0.0000 | 0.6657 | 0.9998 | 0.4890 |
| \|Δθ\| / step | 0.0000 | 0.0317 | 0.0982 | 0.1390 | 0.0302 |

**与 v1 对照（极端主导问题是否解决）**：

| 指标 | v1 | v2 | 判定 |
|------|----|----|------|
| θ 全局 max | 82.115 | 8.819 | ✅ 大幅收敛 |
| θ 全局 std | 12.685 | 2.213 | ✅ 降低 5.7× |
| 单条 episode 占 \|θ\| | 53.5% | 24.0% | ✅ 不再被单条绑架 |
| \|θ_f\|>2π 条数 | 3 | 1（且为 bounded，非旋转） | ✅ |
| continuous_rotation | 1 | 0 | ✅ 主训练不含旋转 |

**regime 分层（预注册判据）**：v2 = 20 bounded / 0 transitional / 0 rotation；
v1 历史 pilot = 16 / 3 / 1 保留为 **eval-only stress-test**。两者物理样本**未混合**。

---

## 5. 初态与动作多样性

| 维度 | v1 | v2 |
|------|----|----|
| 不同 θ0 | 1 | **20** |
| 不同 ω0 | 1 | **20** |
| θ0 实测范围 | {2.0} | `[-3.087, +3.133]`，std 2.1699 |
| ω0 实测范围 | {0.0} | `[-0.937, +0.964]` |
| m / L / b | 5 个固定场景 | train range 内逐 episode 采样 |
| 动作取值 | step/alt 各仅 2 个 | 含 `zero`、`[1,2,5,10]` hold、随机/正弦连续值 |
| 真正 τ=0 轨迹 | 无 | 有（4 条 zero profile） |

图表：`analysis/figures/theta_trajectories.png`、`initial_state_diversity.png`、
`action_and_delta_theta.png`。

---

## 6. Phase 3 动作信息（修正后的受控对照）

采用**同一初态 θ0=0, ω0=0**（重力矩为零）与同一物理参数，仅改变力矩序列：

| 臂 | 初始角加速度 α₀ (rad/s²) | 终态 θ (rad) |
|----|--------------------------|--------------|
| zero | **0.000000** | 0.0000 |
| positive_constant (+1) | **+1.179313** | +0.1145 |
| negative_constant (-1) | **-1.179313** | -0.1145 |
| random | -1.088144 | -0.0692 |
| shuffled_random | -1.175325 | -0.0102 |
| reversed_random | +0.933391 | -0.0180 |

解析值 `α = ±1/(mL²) = ±1.179313` 与数值 `derivatives()` **逐元素一致**。
正/负力矩终态对称（±0.1145），直接验证了 D-A 修正后的方向性验收条件。
shuffled / reversed 复用与 `random` **完全相同的动作多重集**，仅排列不同，却产生不同终态
→ 动作**顺序**携带信息。

> **仍然必须区分**：以上证明的是**数据生成过程**的动作敏感性，
> **不是**任何模型已学会利用动作。真正的判据仍是 P2-03A 的 E2 vs E3 / E2 vs E4 + paired bootstrap CI。

---

## 7. 数据划分与泄漏

全局 episode 级 **14/3/3**（70/15/15），确定性 `RandomState(master_seed)`：
train∩validation / train∩test / validation∩test **全空**。
用 `dynamics_batch_from_windows(window_len=11, stride=5)` 按 split 分别构建窗口，
实测每个窗口的 `episode_id` 均属于该 split，且 `valid_mask` 全 True。

正式全量数据集仍需按 `scenario × profile × regime × IC bin` 分层，
并使用独立 `dataset_version` 与 manifest（本 pilot 仅作协议演示）。

---

## 8. 已解决 / 未解决

| 前轮发现 | 状态 |
|----------|------|
| S1 初态退化为单点 | ✅ 已解决（20 个不同 θ0 / ω0） |
| S2 极端轨迹主导 53.5% | ✅ 已解决（降至 24.0%，θ std 降 5.7×） |
| S3 short_rod dt 精度 | ✅ 已隔离（stress-test dt=0.005；主分布无需改） |
| S4 `b` 语义未文档化 | ✅ 已解决（config + manifest 明确为归一化阻尼率） |
| S5 动作多样性不足 / 无 τ=0 | ✅ 已解决（hold steps + zero profile） |
| 规格漂移（P2_03A_READINESS_PLAN §2） | ✅ 主分布已恢复原规格；四组 OOD 原样保留 |
| Q3 物理范围稳定性 | ✅ 在**修正后的配置**上重新验证（10/10 PASS） |
| Q4 latent_dim 8/16 | ⏸ 仍未关闭（属模型阶段） |
| Q6 checkpoint 命名 / `model_class` | ⏸ 仍未关闭（属模型阶段） |
| **JEPA 是否学到动作条件动力学** | ⏸ **完全未证明**（尚未训练） |

---

## 9. STOP

本报告的所有结论限于**数据与模拟器层**。未生成全量数据集、未实现窗口适配层或
`ActionConditionedWorldModel`、未训练 E0–E5、未实现 MPC。
**等待人工批准**后才进入 P1-B2 / P2-03A。
