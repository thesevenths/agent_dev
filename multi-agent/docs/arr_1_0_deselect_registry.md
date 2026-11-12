# ARR 1.0 — 被 deselect 的 CI 测试逐项根因登记（Package D4）

登记原则：**先实测，再下结论**。下面每一条都在「与 GitHub Actions 完全相同的
环境变量 + 干净 venv」下单独执行过，错误行是真实输出，不是从注释推测。

复现命令（Windows 本地 / clean venv）：

```bash
cd E:/agent_dev/multi-agent
RESEARCH_SKIP_E2E=1 \
PG_CONN_STR="postgresql+psycopg://ci:ci@127.0.0.1:5432/ci" \
TAVILY_API_KEY="ci-placeholder-no-network" \
LLM_API_KEY="ci-placeholder-no-network" \
OPENAI_API_KEY="ci-placeholder-no-network" \
python -m pytest <node_id> -q -p no:cacheprovider
```

| # | 测试 node id | 实测结果 | 分类 | 根因（证据） | 处置 |
|---|---|---|---|---|---|
| 1 | `adaptive_runtime/tests/test_selftest_e2e.py::TestSelftest::test_all_checks_pass` | ERROR | **环境 / 磁盘状态依赖**（非 ARR 缺陷） | `selftest.py:289 positive_path()` → `autopromote.promote_and_land()` 抛 `RuntimeError: policy 未通过，拒绝自动 promote：['calibration']`；`Policy.checks[0]` = `Check(name='calibration', passed=False, blocking=True, detail={'exists': True, ...})`。该自检要求仓库内 `experiments/calibration.json` 处于特定状态，属于**有状态的历史自检**，与 R1–R4 / ARR 1.0 无关。单次运行耗时 312s。 | 保持 deselect；单独跟踪。已登记，未用 skip 掩盖 |
| 2 | `…::TestSelftest::test_cases_have_expected_reasons` | ERROR | 同上 | 同一 `positive_path()` 前置失败 | 同上 |
| 3 | `…::TestSelftest::test_no_sandbox_leak` | ERROR | 同上 | 同一 `positive_path()` 前置失败 | 同上 |
| 4 | `adaptive_runtime/tests/test_benchmark.py::TestFakeDoesNotPollute::test_real_mode_does_write` | FAILED | **真实缺陷**（历史 legacy benchmark 路径，非 ARR 主闭环） | `test_benchmark.py:77` `AssertionError: [] is not true : 真实模式应当写经验库`。`ledger_calls` 非空但 `store_calls` 为空 → 非 fake 路径没有调用 Experience store。属于旧 `benchmark` 子系统的行为缺陷。 | 保持 deselect；**不在 ARR 1.0 范围无差别重构**（reviewer D4 明确要求「只修复影响实验正确性/可信度/后续扩展的问题」）。已登记为待办 |
| 5 | `adaptive_runtime/tests/test_evolution_version.py::TestRC3StallForcedL3::test_dominant_l1_exhausted_forces_l3` | FAILED | **真实行为差异**（历史 evolution/RC3 规则） | `test_evolution_version.py:186` `AssertionError: None != ('PLANNING_FAILURE', 3)`。当 dominant 的唯一 L1 旋钮为 `inconclusive/within_noise` 时，`evolution.evolve_plan()` 返回 `forced_pair=None`，而测试期望强制升 L3。属于 RC3 停滞规则与测试期望不一致。 | 保持 deselect；单独跟踪 |
| 6 | `adaptive_runtime/tests/test_landing_roundtrip.py::TestLiveValue::test_falls_back_to_user_area_when_absent` | 单独跑 PASS；**完整 suite 内 FAIL**：`AssertionError: '/models/Qwen3.8-27B-FP8' != 'qwen'` | **测试环境污染（真实缺陷，已修复）** | 根因：`config.py` 在 **模块导入期**调用 `load_dotenv()`；`tests/rt_tests/test_agent_integration.py` 会导入 `agents → llm → tools → config`，把 `.env` 的 `LLM_MODEL` 写进 `os.environ`，污染随后所有测试。单独跑该文件时不导入这条链，所以「碰巧通过」。这正是 reviewer D4 点名的「环境变量污染」问题。 | **移除 deselect + 修复测试隔离**：在 `mock.patch.dict(os.environ)` 内 `pop("LLM_MODEL")`，让「when absent」前提真正成立；单独跑 27/27、与 `test_agent_integration.py` 串跑 36/36 均通过。CI 现在真实运行该测试 |

## 结论

- 原有 4 条 `--deselect` 共覆盖 **6 个测试**（`test_selftest_e2e.py` 整个文件 3 个 + 3 个单点）。
- ARR 1.0 逐项实测后，**移除了 1 条 deselect 并修复了它背后的真实环境污染缺陷**（#6），
  剩余 5 个 deselected 全部有可复现的根因与错误行。
- **没有新增任何 deselect / skip 来掩盖本轮改动**；新增的 27 条 ARR 1.0 测试全部在 CI 默认 scope（`tests/rt_tests/` + `adaptive_runtime/tests/`）内真实运行。
- 8 个 SciNet++ checkout 依赖用例采用仓库既有的 `RESEARCH_SKIP_E2E` 约定做**条件 self-skip**（CI 可见 `skipped`，本地有真实 checkout 时 10/10 全跑），没有用 deselect 隐藏。

## 复现汇总（本机）

```
# 6 个 deselected 全跑（CI env）：2 failed, 1 passed, 3 errors in 312.18s
# 完整 CI 命令（移除 #6 后）：见 docs/arr_1_0_architecture.md「回归」
```
