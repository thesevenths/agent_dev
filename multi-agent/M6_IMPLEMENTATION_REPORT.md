# M6 Implementation Report

## 1. 交付概览

M6 在 M5 Acceptance Hardening HEAD `4db264ab` 上新建分支
`feature/arr-m6-harness-rsi` 实施，未 merge、未建 PR。全部代码位于
`multi-agent/research_runtime/harness_rsi/`（13 个模块）与统一 driver
`multi-agent/run_m6_harness_rsi.py`。

关键原则：**M6 是 M5 Runtime 的延伸，不是新框架**。它复用 RuntimeState 的 hash-chain
journal / checkpoint / reservation / classify，复用 RRSICoordinator / EvolutionRegistry，
复用 M5 结构化 holdout verifier，复用 M4.3 冻结统计与盲评。唯一新增的"注册"是 M6 自己的
组件版本模型，不新增第二个 Scheduler / Store / Gate。

## 2. 文件清单

| 模块 | 职责 |
|---|---|
| `identity.py` | 内容 SHA、原子冻结、磁盘重读复核、Git 指纹 |
| `benchmark.py` | 60 任务三域基准、family 隔离、sealed gold、contamination ledger、power analysis |
| `harness.py` | 4 个独立版本化研究方法组件、不可变 bundle、完整 SHA、单组件变更约束 |
| `candidate_generation.py` | 失败分类、瓶颈检测、经验检索、提案、静态校验、冻结；硬证据边界 |
| `method_runner.py` | 真实研究方法模块的确定性执行与行为探针评分 |
| `real_engine.py` | 包装 SciNet++/Embodied/Quant 真实 adapter；NaN/缺失 fail-closed |
| `execution.py` | Base/Candidate 配对、家族随机顺序、完整轨迹、infra/research 失败分离 |
| `evaluation.py` | 预注册、8 指标、paired bootstrap/permutation/McNemar/cluster/Holm/CI/效应量 |
| `promotion.py` | 正式晋级门：禁 legacy bool、磁盘重读、task/family 不重叠、statistical gate、gate SHA |
| `adoption.py` | 真实 Harness Loader / Adoption Manager，loaded SHA journal、campaign 绑定 |
| `rollback.py` | 真实版本切换：停调度、防混版、校验 parent、smoke、恢复 |
| `experience.py` | Success/Failure/Improvement/Transfer 四类经验 + 检索消融 |
| `__init__.py` | 统一导出 |

配套：`configs/m6_harness_rsi.yaml`、`configs/m6_harness_rsi_real.yaml`、
`run_m6_harness_rsi.py`、`research_runtime/tests/test_m6_harness_rsi.py`（29 项，含 20+ 对抗用例）、
CI workflow 接入。

## 3. 生命周期（单一 driver）

```
initialize → budget reserve → benchmark freeze → dev failure analysis →
preregistration (主指标由瓶颈组件固定规则确定) → failure-driven candidate →
paired selection → independent evaluation → hidden holdout → formal promotion →
[EXPLICIT HUMAN APPROVAL] → adoption (loaded SHA) → new-round verification →
real rollback → experience + ablation → checkpoint → finalize
```

每个阶段调用 `state.append(run_id, round, f"m6_{stage}", payload)` 写入同一 hash-chain
journal；任何未通过阶段都写 BLOCKED/INCONCLUSIVE 到 `final_result.json`。

## 4. 真跑发现并修复的真实缺陷

1. **预注册主指标与瓶颈组件不匹配**：首次真跑 selection/holdout 上 objective_quality
   无改善（mean_diff=0），但 falsifiability_rate +0.5 显著。根因是主指标在诊断之前写死。
   修复：driver 改为先真实 dev 失败诊断，再按固定 `COMPONENT_PRIMARY_METRIC` 规则确定
   主指标并冻结预注册；规则先于候选结果，符合预注册原则。
2. **主指标重复进入比较族**：`PreRegistration` 的 `secondary_metrics` 默认值包含
   falsifiability_rate；当它被选为主指标时同一指标进入两次比较。修复：`__post_init__`
   从 secondary 中剔除主指标；修复后对比数从 8 降为 7，主指标唯一。
3. **统计门字段不齐**：`analyse_paired` 的显著性在 per-contrast 中，而正式晋级门需要
   顶层 `significant_after_correction`。修复：driver 增加 `_statistical_gate()` 显式展平。
4. **resume 语义与 immutability 冲突**：EvolutionRegistry 按 experiment_id 幂等，重跑会
   因证据时间戳冲突。修复：晋级已验证且批准到来时，复用原 verdict、把批准作为**新事件**
   记录在 `<rid>_approval_recorded`，不改写历史评估。
5. **批准未绑定精确 SHA**：原逻辑只看 `approved:true`。修复：必须同时满足
   `reviewed_candidate_sha == 当前候选 SHA`。

## 5. 验证

- engine-free 全量回归：**729 passed / 12 skipped / 37 subtests**（345.69s）；
- M6 专项 29 项：`29 passed`（含 20 项强制对抗用例）；
- engine-free 完整闭环：`M6_EVIDENCE/run_final2/final_result.json` overall COMPLETE；
- 真实引擎完整闭环：`M6_EVIDENCE/run_real_v2/final_result.json`（SciNet++/Embodied/Quant
  真 adapter，见验收报告）；
- AST 门禁通过；CI 已加入 M6 测试模块。
