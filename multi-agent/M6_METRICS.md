# M6 Independent Evaluation & Metrics

## 1. 独立性契约

- 指标合同与 **primary metric 在看任何候选结果之前冻结**到 `preregistration.json`，
  带 evaluator version SHA；晋级时核对 evaluator SHA，不一致即拒。
- 主指标由**失败诊断得到的瓶颈组件**按固定映射确定：
  objective_generator→objective_quality；hypothesis_critic→falsifiability_rate；
  experiment_designer→experiment_validity；failure_diagnoser→diagnosis_accuracy。
- 评估函数只接收已记录的任务结果；候选代码**不能传入 evaluator callable**、不能控制裁决。
- 8 个指标分别报告，**不折叠成无约束加权分**。
- 所有结果（失败、无效、中断、基础设施错误）全部保留；基础设施失败存在即不得晋级。

## 2. 八个指标

| 指标 | 含义 | 备注 |
|---|---|---|
| objective_quality | 研究问题质量（证据、可测准则、干预、预算） | 由真实 `scientific_judgment` 的 research_value 归一化 |
| falsifiability_rate | 可证伪假设比例 | 行为探针：对无 falsifying observation 的假设必须拒绝 |
| experiment_validity | 实验设计有效性 | 对照组 / confounder / paired seed 探针 |
| confounder_coverage | 混杂变量覆盖 | designer 配置的真实行为检查 |
| diagnosis_accuracy | 根因诊断准确率 | 有/无证据链与反事实探针 |
| evidence_completeness | 证据完整性 | 真实 evidence_refs 数量 |
| budget_efficiency | 预算效率 | 由任务真实 budget 派生，非自报 |
| transfer_performance | 迁移表现 | 跨任务家族的失效诊断迁移 |

## 3. 预注册与效应量

- `minimum_meaningful_effect` 来自配置（默认 falsifiability_rate = 0.05），**先看后改禁止**。
- 主指标改进 ≥ MME 且 Holm 校正后显著，且无 regressed primary / infra failure，才 ACCEPTED。
- 否则 REJECTED（明确变差）或 INCONCLUSIVE（证据不足）。

## 4. 统计方法

`harness_rsi/evaluation.py::analyse_paired`：
- paired t 检验（复用 `evolution.statistics.paired_p_value`，非误用 1.96）
- paired bootstrap CI（按固定 seed 重采样）
- **clustered bootstrap by task family**（整族重采样，不在族内按行重采）
- paired permutation test（固定 seed，符号翻转）
- McNemar（二元指标，0.5 阈值，正态近似）
- Holm step-down 多重比较校正（主指标 + 7 个 secondary 构成预注册族，
  缺失统计量按 p=1 处理，不缩小分母）
- Cohen's dz 效应量、Student-t CI、power analysis（`m4_3.sample_size`）

## 5. 评估器版本

`EVALUATOR_VERSION = m6_eval_v1`；`gate_version_sha()` 记录正式晋级门合同 SHA。
二者随证据包落盘，任何统计规则变化都会使旧证据失效。

## 6. 诚实边界

- pilot 统计是描述性的，不作确认性结论；
- 本基准的 novelty 仅做**快照内重复检测**，`scientific_judgment` 明确
  `literature_check=not_performed`，不声称全局新颖；
- Quant 数据为合成，只验证工程闭环，不声称 alpha。
