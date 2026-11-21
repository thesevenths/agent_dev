# ARR v1.0 — 04 Implementation Report

## 1. 已实现（复用为主）

| WP | 改动 | 复用资产 | 根因/决策 |
|---|---|---|---|
| WP-01 | `runtime_state.py`: `rt_campaign_exec` 表 + `claim_campaign`/`campaign_state`/`set_campaign_state`/`recover_interrupted_campaigns`；`formal_runtime.py`: dispatch 加原子 claim + 状态机 + 结果缓存 + result_sha；`discovery_coordinator.py`: 审批复合 SHA + claim 三态映射；`knowledge/graph.py`: `INCONCLUSIVE_EVIDENCE` 边 | M5 RuntimeState `BEGIN IMMEDIATE`、`verify_journal`、`consume_reservation` | 原 dispatch 重复执行 backend（P0-1）、INCONCLUSIVE 被写成 REFUTES（P0-2）、无原子 claim（P0-3） |
| WP-02 | 新增 `research_runtime/program/`（state/history_index/driver/discovery_round） | DiscoveryCoordinator 单轮、RuntimeState journal/checkpoint | 无 program 层；loop/controller 是 evolution 语义不适配 |
| WP-03/04 | `discovery/experiment/designer.py`: `discrimination_matrix` + protocol.discrimination 字段 | 现有 hypotheses.predictions、Critic | 区分度只在文本层，无量化 H×arm 矩阵 |
| WP-05 | `discovery/memory/verified.py`: 六类 `MEMORY_CLASSES` + 七态生命周期 + 扩展字段（含默认值兼容） | 现有两级 tier + SHA 链 + promote 验证 | methodology/contradiction 类缺失 |
| WP-06 | `capability_bench/memory_attribution.py`: 七状态归因账本 + 负迁移报告 | memory_bench 四臂 | retrieval 被当成 usage/outcome |
| WP-07 | `harness_rsi/method_registry.py`: 版本化方法注册表（薄封装 EvolutionRegistry） | EvolutionRegistry、M6 promotion/adoption/rollback | 无 method registry；Candidate 无持久化 |
| WP-08 | `capability_bench/external_benchmarks.py`: adapter + coverage matrix | 现有 family/GT 字段 | 无外部 adapter；不假设可访问、不伪造 |
| WP-09/10 | `run_arr_v1_program_e2e.py`、`arr_cli.py`、`reproducibility.py` | 真实 vLLM、OpenAlex、DomainPack、M7.6 生产 E2E 证据 | 无 program CLI / 可复现包 |

## 2. 新代码清单

- `research_runtime/program/{__init__,program_state,history_index,program_driver,discovery_round}.py`
- `research_runtime/capability_bench/{memory_attribution,external_benchmarks}.py`
- `research_runtime/harness_rsi/method_registry.py`
- `research_runtime/{arr_cli,reproducibility,run_arr_v1_program_e2e}.py`
- 测试：`test_arr_v1_wp01..wp10` 共 6 个新测试文件（36 个新测试）

## 3. 兼容性/迁移

- 所有新字段带默认值：M7/M7.6 存量 artifact 与 SQLite 可继续读取；campaign 表为新增表（不改旧表）。
- `discrimination` 字段进入 protocol → preregistration SHA 变化（属于新的科学协议版本，非重写历史）。
- 未修改用户的 `config.py`/`llm.py`（已备份保护）。

## 4. 架构约束（诚实）

- M5 CampaignRunner（ML ExperimentPlan）与 M7 DomainPack（arms/seeds/splits）契约不同，保留 FormalRuntimeAdapter 边界；WP-01 只加固该适配层，不合并两套抽象。
- "exactly-once" 不做保证：仅保证 at-most-once dispatch claim；backend 物理执行的幂等性由后端自身契约决定，crash 后标 RECOVERY_REQUIRED/UNKNOWN 不盲重。
- 外部审批仍是 M4.3 的未认证 operator 字符串；不声称企业身份。

## 5. 未实现/阻塞

见 `15_BLOCKED_REGISTER.md`：独立专家盲评、≥8 family、外部 benchmark 访问/许可、认证身份、SciNet++ 大规模算力。均不伪造。
