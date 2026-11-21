# M5-02 Dual-layer RRSI Architecture

**结论先行**：M5-02 新增薄 RRSI 骨架，严格区分 Domain Model 与 Research Harness 两层。每次实验只演化一层；两层各自有版本号、parent SHA、实验记录与 gate；held-out 不参与候选生成，只有 `ACCEPTED + holdout_confirmed` 才能晋级。失败/拒绝候选全部保留。本阶段不实现 Meta-RL，不自动修改生产代码。

## 1. 新增模块

| 文件 | 职责 |
|---|---|
| `research_runtime/rrsi/evolution_experiment.py` | 统一实验契约、层白名单、single-layer guard、promotion gate |
| `research_runtime/rrsi/evolution_registry.py` | append-only 版本/实验 registry：layer、candidate_sha、parent_sha、status、gate_passed、evidence |
| `research_runtime/rrsi/domain_model_evolver.py` | 复用既有 `generate_domain_candidates` / `evaluate_domain_candidate`；需要真实 injected runner，无 runner 即无证据 |
| `research_runtime/rrsi/harness_evolver.py` | 复用既有 `adaptive_runtime.harness_evolution`；候选只含 harness knobs |
| `research_runtime/rrsi/coordinator.py` | 瓶颈判层、单层执行、版本继承、gate 后晋级 |

未重写：现有 `research_runtime/evolution/*`、`adaptive_runtime.harness_evolution`、CandidateConfigBuilder、12-step Gate、ResearchService/Director。

## 2. 双层隔离

| 规则 | 实现 |
|---|---|
| 不同实验同时改两层 | `assert_single_layer()` 拒绝；coordinator 每次只调用一层 generator/evaluator |
| 独立版本 | registry 按 layer 记录，`parent_for(layer)` 只取该层最近 PROMOTED 版本 SHA |
| 独立证据 | evidence 带 `source_layer`，experiment_id 不可跨层复用（`ResearchStoreConflict`） |
| 共享 Experience 不混淆来源 | diagnosis 可来自统一 Experience，但实验记录保留 `source_domains` / `source_layer` |
| held-out 保护 | `generate()` 没有 holdout 参数；`EvolutionExperiment` 禁止 `holdout_used_for_selection=True` |
| 不降统计门槛 | `gate_allows_promotion` 要求 decision=ACCEPTED、holdout_confirmed=True、frozen_test_used=False、training_only=False |
| 保留失败 | REJECTED / INCONCLUSIVE / 失败候选均写入 registry |
| 版本继承 | 每次实验带 parent_sha；只有 PROMOTED 版本成为下一轮 parent |

## 3. 生命周期

```
Diagnosis (shared Experience, source preserved)
  -> Coordinator.select_layer()
  -> layer-specific generate(dominant_failure)        # NO holdout metrics
  -> layer-specific experiment (real runner; paired + held-out)
  -> gate_allows_promotion(...)
  -> registry.record(PROMOTED | REJECTED | INCONCLUSIVE)
```

Domain Model 候选走 Hypothesis → Plan → Approval → Real Experiment → Evaluation → Evidence（由现有 Director/Service/Gate 执行）；Harness 候选走 Diagnosis → Candidate Harness → Experiment → Paired Evaluation → Held-out → Accept/Reject（由现有 harness evolution 执行）。

## 4. 验证

`test_m5_02_rrsi.py` engine-free 证明：
- 三域/两层的路由与 single-layer guard；
- 生成函数不接收 held-out 指标；
- 只有 gate 通过的候选 PROMOTED；
- REJECTED 候选仍落库，另一层 registry 不被污染；
- parent_sha 继承最近 PROMOTED 版本；
- cross-layer experiment_id 复用被拒绝。

## 5. 未完成 / 风险

- 本阶段没有执行真实 Domain Model / Harness 候选晋级 E2E；只完成真实引擎的跨域 Smoke 与 RRSI 契约测试。真实 RRSI 晋级应在下一步以 held-out 真实验证单独执行。
- 第一版不实现自动修改任意生产代码、Meta-RL 或跨层联合优化；这是有意边界，不是遗漏。
