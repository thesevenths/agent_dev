# ARR M7 — Final Acceptance Report

- Generated: 2026-11-17T15:02:19
- Branch: `feature/arr-m7-scientific-discovery`
- Repository HEAD: `4f647fcb7662f9bd736ca69f556f2ff1c2700a42` (dirty=True)
- Harness SHA: `4f647fcb7662f9bd736ca69f556f2ff1c2700a42`
- Evidence: `M7_EVIDENCE/run_v4`
- E2E result SHA: `7b4c1d3d114c8e2bc034a62c065444f30a434d2536dddc8665e46ae1ab16036e`

## 0. Verdict

**M7_SYSTEM_COMPLETE / NO_CONFIRMED_NEW_FINDING**

完整研究闭环在真实 API、真实 LLM、真实引擎上执行并留下可复核证据；但预注册的 MME=0.01 未达到（selection mean|effect|=0.000282，replication mean|effect|=0.000354），因此科学结论保持 INCONCLUSIVE。未伪造 Novelty，也未把非显著结果提升为发现。

## 1. Gate 判定

| Gate | Status | Evidence |
|---|---|---|
| G0 | PASS | HEAD=4f647fcb7662f9bd736ca69f556f2ff1c2700a42; harness=4f647fcb7662f9bd736ca69f556f2ff1c2700a42; journal_events=21 |
| G1 | PASS | providers=['openalex']; unique_papers=19; success_rate=0.666667; full_text=0 |
| G2 | PASS | statuses=POTENTIALLY_NOVEL,POTENTIALLY_NOVEL |
| G3 | PASS | n_questions=4 |
| G4 | PASS | priorities=[0.6941, 0.6941, 0.6941, 0.6941] |
| G5 | PASS | accepted=4; llm_output_used=False |
| G6 | PASS | experiment=exp_70b079cfb0cf71; mme=0.01 |
| G7 | PASS | real_llm_calls=2; engine_results_checked=12 |
| G8 | PASS | graph_sha=bb5b105bf206b82b...; claim=claim_227a5d9d8a385a29 |
| G9 | PASS | holdout_chain=21dd5a2ce0fe9620...; replication_verdict=INCONCLUSIVE |
| G10 | PASS | tier=candidate; memory_sha=334a9b2eeeaf8036...; unverified findings never promoted |
| G11 | PASS | Reused M6 Research Harness RSI for shared Hypothesis Critic component (commit 4f647fcb); not re-implemented in M7 |

## 2. 真实证据链

- Runtime journal: ok=True, events=21, head=19533294e1054e94...
- Budget reservations: 2; states=['bound', 'bound']
- LLM trajectories: 2 real calls; tokens=[3828, 1519]
- Engine evidence: 12 result files re-read from disk, ok=True
- Holdout ledger: 21dd5a2ce0fe9620c16aaa4bd76ad0d99552f65cd0f611c5c3858ce0f4dbb216; purpose=final_holdout_evaluation
- Claim graph: bb5b105bf206b82b83c1170fea6b2c084183a6fc0323a56a0707b8484686e4f7; nodes_checked=9
- Memory: 334a9b2eeeaf803696bd594e50f6eb85d4320613861094db0cccfeaffadc0bbe; tier=candidate
- Full research_runtime suite: 781 passed, 12 skipped, 37 subtests passed (final clean-basetemp run).

## 3. 研究能力指标（不合并为单一 ARR Score）

| Category | Metric | Value | Interpretation |
|---|---|---|---|
| Question Discovery | High-value Question Rate | 4/4 | value decision |
| Question Discovery | Novelty Precision | NOT_MEASURED | no external ground truth labels |
| Question Discovery | Research Budget Efficiency | 0.6429 | from value artifact |
| Scientific Judgment | Falsifiability Rate | 4/4 | independent critic accepted |
| Scientific Judgment | Correct Hypothesis Rejection | 4/4 invalid LLM drafts rejected | validity only; scientific correctness NOT_MEASURED |
| Scientific Judgment | Research Value Calibration | NOT_MEASURED | no ex-post utility labels |
| Experiment Quality | Experiment Validity | PASS | paired arms, frozen config, guardrails ok |
| Experiment Quality | Hypothesis Discrimination | INCONCLUSIVE | favorable direction but below MME |
| Experiment Quality | Expected Information Gain | 0.531 | computed from competing hypotheses |
| Experiment Quality | Actual Information Gain | NOT_MEASURED | binary inconclusive verdict cannot quantify posterior shift |
| Experiment Quality | Reproducibility | 12/12 SHA verified | replication executed |
| Evidence Quality | Evidence Completeness | 12 results | all re-read from disk |
| Evidence Quality | Claim Provenance | PASS | graph nodes=9, edges=7 |
| Evidence Quality | Contradiction Detection | PASS (tests), 0 in real run | engine-free adversarial suite |
| Evidence Quality | Independent Replication Rate | 1/1 executed, 0/1 confirmed | below MME |
| Recursive Improvement | Base/Candidate Research Quality | SEE M6.1 EVIDENCE | M6 harness RSI reused |
| Recursive Improvement | Paired Effect / Family CI / Held-out Generalization | SEE M6.1 EVIDENCE | not re-run in M7 E2E |

## 4. 最终问题逐条回答

1. **是否可以从零开始自主发现研究问题？** 是。questions artifact 包含 4 个由真实文献 gap 生成的问题；见 artifacts/questions.json。
2. **文献检索是否真正接入学术数据库？** 是。provider=['openalex']; unique_papers=19; 见 artifacts/literature.json。
3. **是否能够判断已有研究是否已经解决问题？** 是（限检索语料）。novelty verdicts=POTENTIALLY_NOVEL,POTENTIALLY_NOVEL; 见 artifacts/novelty.json。
4. **是否能够发现潜在 Research Gap？** 是（受限）。literature_check=performed; verdicts 来自真实 prior-art 匹配，不宣称全局新颖。
5. **是否能够评估研究价值？** 是。7 维评分 + information gain + priority；见 artifacts/value.json。
6. **是否能够提出互相竞争的科学假设？** 是。H1/H2/H3/H0 accepted=4; 独立 Critic 已拒绝不可证伪的 LLM 输出；见 artifacts/hypotheses.json。
7. **是否能够设计区分假设的实验？** 是。protocol=exp_70b079cfb0cf71; primary_metric=state_mse@1; MME=0.01; 见 artifacts/experiment_protocol.json。
8. **是否能够真正使用 LLM Agent 和 Scientific Engine？** 是。real LLM calls=2, provider=vllm-openai-compatible, model=/models/Qwen3.8-27B-FP8; engine result SHA verified=12。
9. **是否能够通过 Holdout 和 Replication？** 流程通过，统计确认未通过。holdout access=True; replication verdict=INCONCLUSIVE; 方向与 baseline 一致但效应低于 MME。
10. **是否能够积累可信的科学知识？** 是（分层）。KG integrity=True; memory tier=candidate; 未确认结果只进 candidate memory。
11. **是否能够通过 M6 RSI 改善自己的研究方法？** 复用既有能力。M7 复用 M6.1 final fixes 的 HarnessLoader/Critic 策略与正式晋级链；本次 E2E 未新晋级 M7 组件。
12. **是否已经产生真正经过验证的新科学发现？** 否。selection=INCONCLUSIVE, replication=INCONCLUSIVE; mean|effect|=0.000282/0.000354 < MME 0.01。

## 5. 已知限制

- 文献层只读取 metadata/abstract，full_text_accessible=0；Novelty 只代表检索语料内判断。
- Semantic Scholar 在先前真实探测中返回 429；最终 E2E 使用 OpenAlex + Crossref。
- LLM Agent 真实调用成功，但其自由格式假设未通过独立 Critic，最终假设采用证据约束的确定性生成；该拒绝是设计内的 fail-closed 行为。
- Embodied 引擎是确定性仿真，不是真实机器人硬件，也未声称训练 SciNet++ checkpoint。
- 效应方向与 baseline 一致但小于预注册 MME，因此不能升级为科学发现。

## 6. 复现命令

```bash
py -m pytest -q research_runtime/tests/test_m7_*.py
py -m research_runtime.run_m7_e2e --out-dir M7_EVIDENCE/run_v4 --run-id m7_real_discovery_v4 --max-requests 6
py -m research_runtime.build_m7_report --run-dir M7_EVIDENCE/run_v4
```
