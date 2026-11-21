# ARR v1.0 — 06 Runtime E2E Report

> ARR v1.0 不重复 M7.6 的单轮生产 E2E；本验收引用其**真实原始证据**并核查。

## 1. L1 生产链路真实证据（`M7_6_EVIDENCE/run2`）

| 项 | 实测 |
|---|---|
| runtime_integration | `FORMAL_PRODUCTION_RUNTIME` |
| 学术 API | OpenAlex 19 篇真实论文 |
| LLM | vLLM `/models/Qwen3.8-27B-FP8`，2 次真实调用（5547 tokens） |
| 审批 | 带外 approval_cli，`source=human`，SHA 绑定，driver 零自批 |
| Harness | 真实 M6 `6c197910...` 经 HarnessLoader 解析，非 git HEAD |
| journal | 28 事件，verify ok |
| campaign | 2 个真实 `camp_*`，lifecycle 四事件齐全 |
| reservation | 2 个 consumed/succeeded |
| result | 12/12 result SHA 校验通过 |
| verdict | INCONCLUSIVE（诚实） |

## 2. v1.0 WP-01 加固后的行为

- 重复 dispatch 命中缓存结果，backend 物理执行恰好 1 次（新测试断言）。
- 不同 executor 并发 claim 被拒；crash 后 `RECOVERY_REQUIRED`，不盲重。
- result 结构化持久化并记 result_sha；cache hit 返回同 SHA。
- claim graph 三态：INCONCLUSIVE → `INCONCLUSIVE_EVIDENCE` 中性边，不再伪造成 REFUTES。
- 复合身份绑定：protocol+harness+dataset+method+git SHA 记入 approval journal。

## 3. 结论
L1 生产执行链路成立；v1.0 加固修复了 3 个 P0 真缺陷（见 01_SOURCE_AUDIT）。
