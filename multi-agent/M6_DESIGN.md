# M6 — Research Harness Recursive Self-Improvement：设计与源码映射

## 1. 目标与边界

M6 优化的是**研究方法本身**，不是研究对象（模型配置）。它回答的唯一核心问题：

> Base Harness / Candidate Harness 在相同任务、相同预算、相同 seed 下配对对照，
> 在独立 holdout 上更好，被正式晋级并被下一轮 Runtime 真正加载，在新任务上再次证明改善。

三个明确不做的事：
1. **不做 Prompt 优化器**：四个可版本化组件是 Objective Generator / Hypothesis Critic /
   Experiment Designer / Failure Diagnoser。Prompt 只是方法表达，不作为优化目标。
2. **不做第二个 Agent Framework**：复用 M5 的 RuntimeState / RuntimeCoordinator /
   RRSICoordinator / EvolutionRegistry / Campaign / Experience Store / budget / journal。
3. **不伪造任何批准、holdout、统计或真跑结果**：缺证据即 fail-closed（INCONCLUSIVE/BLOCKED）。

## 2. 起点与分支

- 起点 SHAs：`agent_dev @ 4db264abf902b53e6ef117eaba2e2188e0914b01`（M5 Acceptance Hardening HEAD）
- 外部引擎：`SciNet++ @ 95e65d0a187db33a98ff3b278b53c22d2a432d1d`（`feature/p2-03a-action-jepa`）
- 分支：`feature/arr-m6-harness-rsi`（不 merge main，不建 PR）

## 3. source-to-requirement 映射（实际读源码后的结论）

| 需求 | 复用资产（源码位置） | M6 新增 |
|---|---|---|
| 统一 Runtime / journal / checkpoint / 恢复 | `research_runtime/runtime_state.py`：`append` (hash chain)、`checkpoint`、`verify_journal`、`reserve_execution`、`classify` | 无重写；driver 仅追加 `m6_*` 事件与最终 checkpoint |
| Evolution Registry / parent 链 | `research_runtime/rrsi/evolution_registry.py`：`record`/`parent_for` | `RRSICoordinator.run_formal_harness_promotion` thin delegate，仍用同一 registry |
| 结构化 holdout 验证 | `research_runtime/rrsi/holdout_evidence.py`：`build/verify/gate_allows_promotion_verified` | `harness_rsi/promotion.py` 加严：禁 legacy bool、task family 不重叠、per-task result SHA 磁盘复核、statistical gate、gate version SHA |
| 冻结统计与盲评 | `research_runtime/evolution/statistics.py`、`m4_3/statistical_analysis.py`、`m4_3/blind_review.py`、`m4_3/sample_size.py` | `harness_rsi/evaluation.py` 新增 paired bootstrap、clustered bootstrap、permutation、McNemar、Holm、8 指标、预注册 |
| 现有研究组件 | `objective_discovery.py`、`objective_candidate_generator.py`、`objective_critic.py`、`scientific_judgment.py` | `harness_rsi/harness.py` 把它们作为四个独立版本化组件封装进不可变 bundle |
| 真实引擎调用 | `research_runtime/cli.py::_build_real_domain_runner`、`settings.py`、`domains/{scinet,embodied,quant}.py` | `harness_rsi/real_engine.py` 包装真实三域 runner；NaN/缺失即 fail-closed |
| Experience Store | `adaptive_runtime/experience.py::Store` | `harness_rsi/experience.py` 四类 Experience + 检索消融（链接原 run/task/SHA） |
| 现有 harness 候选空间 | `adaptive_runtime/harness_evolution.py`、`evolution_spaces.HARNESS_SEARCH_SPACE`（train.lr / weight_decay / vicreg_var_weight） | **关键缺口**：这些只改研究对象训练参数，不含 Critic/Designer/Diagnoser。M6 新增四个研究方法组件，不复用该搜索空间冒充 RSI |

## 4. 组件与数据流

```
BenchmarkSpec(冻结 manifest + sealed gold)
        │  generator_view (仅 dev，无答案)
        ▼
Failure Classification → Bottleneck Detection → Experience Retrieval
        ▼
Improvement Proposal → Candidate（只改一个方法组件）→ Static Validation → Freeze
        ▼
Base/Candidate paired（随机顺序 / 同 seed / 全留存 / infra 分离）
        ▼
Independent Evaluation（预注册 / 8 指标 / 配对统计 / evaluator SHA）
        ▼
Hidden Holdout（独立任务家族 / per-task result SHA 落盘）
        ▼
Formal Promotion（禁 legacy bool + artifact_reader 磁盘重读 + statistical gate）
        ▼
Human Approval（唯一人工关卡；必须绑定精确 SHA）
        ▼
Adoption（加载精确 SHA + journal loaded SHA + campaign 绑定）→ 新轮验证
        ▼
Real Rollback（停调度 → 校验 parent → 切换 → smoke → 恢复）
```

## 5. 关键不变量

- **Candidate SHA = 完整冻结产物的内容 SHA**（`harness.identity.sha256_json`），不含名字、
  不含部分 dict；freeze 后文件字节 SHA 单独绑定，加载时两者都复核。
- **一次只改一个研究方法组件**（`assert_single_method_change`），model/tool/environment
  变更不得伪装成研究方法改进。
- **gold/rubric 与 manifest 物理分离**：sealed 文件 SHA 写入 manifest，但 sealed 内容
  不出现在 generator_view 中（对抗测试验证）。
- **holdout 与 selection 同时按 task id 和 task family 隔离**；家族重叠即拒绝。
- **主指标由失败诊断的瓶颈组件按固定规则确定**（在候选生成之前冻结），不允许看结果后改指标。
- **正式晋级禁 legacy boolean 路径**；`holdout_confirmed=True` 单独出现即拒。
- **批准绑定精确 SHA**；通用 approved 标志不等于对另一个候选的同意。
- **回滚不是返回 SHA**：实际停调度、防混版、校验 parent、切换、smoke、恢复，全程 journal。
