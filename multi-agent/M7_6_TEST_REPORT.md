# M7.6 Test Report

## 1. Commands

```
py -m pytest -q research_runtime/tests/test_m7_6_approval_gate.py       # 18 tests
py -m pytest -q research_runtime/tests/test_m7_6_production_runtime.py  # 9 tests
py -m pytest -q research_runtime/tests/test_m7_6_p1_regressions.py      # 11 tests
py -m pytest -q research_runtime/tests/test_m7_*.py                     # 82 tests
py -m pytest -q research_runtime/tests                                  # full suite
```

## 2. 结果

| 范围 | 结果 |
|---|---|
| P0-A 审批负向矩阵 | 18 passed |
| P0-B 生产 Runtime 负向矩阵 | 9 passed |
| P1 回归 | 11 passed |
| 全部 M7 测试 | 82 passed |
| 完整 research_runtime 套件 | **833 passed / 12 skipped / 37 subtests** |

Baseline（M7.5 末）为 797 passed；净增 36 个测试，全部为 M7.6 新增，零回归。

## 3. Engine-free vs Real 区分

### Engine-free（夹具，明确标记）
- `test_m7_6_approval_gate.py` / `test_m7_6_production_runtime.py`：使用 FakeProvider/FakeAgent/fake_executor，
  所有生产审批夹具必须 `fixture_approval=True`；生产模式的夹具审批被明确断言拒绝。
- 这些测试证明**门控逻辑与失败闭合**，不证明科学能力。

### Real（真实依赖）
- `M7_6_EVIDENCE/run2`：真实 OpenAlex（19 篇）、真实 vLLM LLM（2 次调用，5547 tokens）、
  真实 M6 HarnessLoader（SHA 6c197910）、真实 DomainPack 引擎、真实带外审批、真实 M5 RuntimeState journal。
- 真实 Pilot：M7.2（14 cases）、M7.3（40 results）、M7.4（base 诊断）、M7.5（32 results）均用真实 LLM。

## 4. 关键负向测试（P0 绕过必须封死）

| 测试 | 验证 |
|---|---|
| `test_missing_approval_blocks` | 无审批 → BLOCKED |
| `test_agent_cannot_approve_itself` | agent 自批 → 拒绝 |
| `test_legacy_human_operator_string_rejected` | 伪造 `human-operator` → 拒绝 |
| `test_system_policy_source_rejected` | system_policy → 拒绝 |
| `test_fixture_cannot_be_used_in_production` | 测试夹具不能进生产 |
| `test_auto_approval_bypass_count_is_zero` | 生产 driver 无自动审批代码 |
| `test_production_never_calls_direct_executor` | 生产不调 direct executor |
| `test_missing_dispatcher/loader/harness_sha_blocks_production` | 缺任一正式契约 → BLOCKED |
| `test_harness_sha_mismatch_blocks_dispatch` | SHA 不匹配 → BLOCKED |
| `test_reservation_consumed_on_success_and_failure` | 预算结算 |
| `test_journal_integrity_survives_repeat_dispatch` | 重启/重复不重复消费 |

## 5. 失败与跳过

- 无失败测试。
- 12 skipped 为既有测试中依赖外部条件的用例（与 baseline 一致）。

## 6. CI

CI job 在最终提交后触发；本地完整套件已绿。CI 结果见最终答复。
