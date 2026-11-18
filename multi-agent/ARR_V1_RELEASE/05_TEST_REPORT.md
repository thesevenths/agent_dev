# ARR v1.0 — 05 Test Report

## 1. 命令与结果

```
py -m pytest -q research_runtime/tests
=> 869 passed / 12 skipped / 37 subtests passed (398s)
```

| 基线 | 当前 | 净增 |
|---|---|---|
| M7.6: 833 passed / 12 skipped | 869 passed / 12 skipped | +36（全部 v1.0 新增） |

## 2. 新增测试（均为负向/行为测试，非模块存在性）

| 文件 | 覆盖 |
|---|---|
| `test_arr_v1_wp01_runtime_integrity.py` | 重复 dispatch 不重跑 backend（旧测试盲点已补）、并发 claim 拒绝、状态转换校验、crash→RECOVERY_REQUIRED 不盲重、失败结算、INCONCLUSIVE 不产生 REFUTES 边、exactly-once 诚实声明 |
| `test_arr_v1_wp02_program.py` | program 状态机、重复 round 拒绝、3 轮由真实 artifacts 规划（family 不重复、parent 链、memory 继承）、planning digest 逐轮变化、预算超限阻断、resume 幂等、adapter 映射真实 state |
| `test_arr_v1_wp03_06_judgment_memory.py` | 区分度矩阵检测不可区分假设、六类记忆、七状态归因、retrieval≠improvement、负迁移 GT 测量 |
| `test_arr_v1_wp07_08_rsi_benchmark.py` | method registry 需失败证据、SHA 校验/漂移、外部 benchmark 无数据 BLOCKED、有真实数据才评估、coverage matrix 六项 |
| `test_arr_v1_wp09_10_ops.py` | CLI 全部子命令有真实 handler、无装饰占位、可复现包导出/校验/缺档 fail-closed、无任意 shell |

## 3. Real vs Fixture

| 类型 | 证据 |
|---|---|
| Engine-free（明确标记） | 所有 `test_arr_v1_*` 单测：注入 fake backend/coordinator/agent，验证门控与状态机 |
| Real（真实依赖） | `M7_6_EVIDENCE/run2`（真实 OpenAlex/vLLM/HarnessLoader/审批/DomainPack/M5 journal）；`ARR_V1_EVIDENCE/program_e2e`（真实 vLLM 3 轮 + 真实 OpenAlex + 跨轮继承） |

## 4. 失败与跳过

- 失败：0。
- 跳过：12（与 baseline 一致，依赖外部条件）。
- 无删除/弱化任何既有测试。

## 5. 开发期发现并修复的真实缺陷（源码证据）

1. WP-01 schema 注释含分号导致 `SCHEMA_SQL.split(";")` 语法错误 → 移除注释分号。
2. cache-hit 返回字段 `result_sha` 与首次 `result_sha256` 不一致 → 统一。
3. `_stage("APPROVAL_BINDING", {...})` 位置参数误用 → 改 `**payload`。
4. 真实 CLI 运行时发现 `program inspect` 的 integrity=false（manifest SHA 未持久化）→ `_save` 写完整对象，重启可校验。

这些均由真实执行发现，而非推测。
