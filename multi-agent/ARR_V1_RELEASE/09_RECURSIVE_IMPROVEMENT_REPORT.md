# ARR v1.0 — 09 Recursive Improvement Report

## 1. 机制（WP-07）

- `MethodRegistry`：版本化方法注册表，薄封装 M5 EvolutionRegistry（append-only、SHA 绑定）。
- Candidate 注册要求 `source_failure_refs` 非空且 SHA 程序化校验（不信任 LLM 生成的 SHA）。
- 保留 M6 promotion/adoption/rollback 晋级链；历史方法（含 REJECTED）不删除。

## 2. 本轮真实实验

- 真实 3 轮 program 中每轮 verdict INCONCLUSIVE，无"被实验证伪的具体失败模式"足以支撑
  一个可诊断的方法论瓶颈 → 按 ADR-005 的 NO_CLEAR_BOTTLENECK 规则**不强制生成 Candidate**。
- M7.4 pilot 的 base 在 6/6 development cases 全对，同样得到 NO_CLEAR_BOTTLENECK。
- 因此：RSI 机制存在且可测（7 个 registry/benchmark 测试），但**未证明改进成功**。

## 3. 结论
L3 Recursive Improvement = INCONCLUSIVE（未达 PASS 要求：真实瓶颈→Candidate→paired→holdout→
replication→adoption→post-adoption 全链未触发）。这是合法结论，不是工程失败。
