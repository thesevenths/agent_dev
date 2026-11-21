# M4.0.1 P1-04 — Evidence Reader 使用契约 + Discovery Evidence Adapter

> 结论先行：ARR 1.1 现存**两套** Evidence 读取入口，职责必须严格分离，M4.1
> Discovery 只能依赖统一的只读 `DiscoveryEvidenceAdapter`（底层 = M4.0 统一读模型），
> **不得**触碰 12 步 fail-closed 统计门的 legacy 证据源 `default_evidence_provider`。

---

## 1. 两套 Reader：职责边界（不可混淆）

| 维度 | ① Legacy 门控证据源 | ② 统一只读读模型（M4.0-04） |
|------|--------------------|------------------------------|
| 入口 | `research_service.default_evidence_provider(experiment_id, evidence)` | `evidence_read_model.query_run_evidence(run_id, workspace=)` → `UnifiedEvidenceReader` |
| 读取载体 | 仅 append-only `evidence.jsonl`（`event_type="result"`） | 6 种既有载体：`evidence_index` / `domain_summary` / `candidate_registry` / `engine_metrics` / `evidence_jsonl` / `evidence_bundle` |
| 消费方 | **12 步 fail-closed 统计门**（research_service.py:193,327；arr_runtime.py:559） | `ResearchService.query_run_evidence`、`UnifiedResearchScheduler.query_run_evidence`、**M4.1 Discovery** |
| 是否改过 | **M4.0 起刻意不替换**（控制改动范围） | M4.0 新建 |
| 是否生成 verdict | 否（只重建 `ExperimentResult`） | 否（raw verdict 原样保留，从不重算） |
| 是否写 | 否 | 否（只读） |

**关键事实**：M3.x 真实证据写在 `evidence_index(v2)` / `candidate_registry` / `domain_summary` /
engine `*metrics*.json` 等载体，`evidence.jsonl` 只承载 #3/#4 类事件。若 M4.1 误用 ①，
将**看不到** M3.x 的任何真实证据。因此 M4.1 必须走 ②。

> 验收条件 A（当前 run 不读前缀相似其他 run）、B（重建关系不冒充原生）由 P1-01/P1-02 已落地；
> 本任务只负责把"用哪个 Reader"的契约显式化。

---

## 2. `DiscoveryEvidenceAdapter`（M4.1 唯一允许的依赖）

位置：`research_runtime/discovery_evidence_adapter.py`。薄门面（facade），**不引入第二套系统**，
完全委托给 `query_run_evidence`（入口 ②），仅新增 discovery 导向的访问器，无新证据语义。

```python
adapter = DiscoveryEvidenceAdapter(workspace=...)   # workspace=None → 默认 multi-agent/
adapter.query_run(run_id)            -> dict   # 完整 EvidenceView，与 Service/Scheduler 完全一致
adapter.discover_entities(run_id)    -> list   # 去重后的证据实体（身份 = 真实字节 SHA-256）
adapter.carriers(run_id)             -> list   # 各载体读取情况
adapter.raw_verdict_census(run_id)   -> dict   # 对 raw verdict 字符串的"统计报告"，非重算门
adapter.unresolved(run_id)           -> list   # 诚实的显式缺口（如 engine-metric 碰撞）
```

### 不变量（由 `test_m4_0_discovery_adapter.py` 强制）

1. **只读**：从不写文件、从不调用 `default_evidence_provider`、从不参与统计门。
   静态校验：`DiscoveryEvidenceAdapter` 源码中不得出现 `default_evidence_provider`（见 `_self_source_uses_legacy_gate_provider`）。
2. **不生成 verdict**：返回 `EvidenceView` 原样，包含 `raw_verdict_census`（报告），
   但 view 顶层**无** `verdict/decision/accept/reject/gate/significant/p_value` 等门控键。
3. **不伪造实体**：每个返回实体都至少有 1 个真实 carrier（`carriers` 非空）；
   适配器从不凭空合成实体（例如 M3.4 从未写入的 **Experience**）。缺失 Experience → 保持 `unresolved`。
4. **稳定契约**：M4.1 只能依赖本门面的公共方法，不得耦合 `UnifiedEvidenceReader` 内部 carrier 读取细节。
5. **幂等 / 确定性**：同 run 多次 `query_run` 结果一致（sorted、无时间戳）。

---

## 3. M4.1 Discovery 必须遵守的规则

- ✅ 通过 `DiscoveryEvidenceAdapter` 读取证据；只读。
- ✅ 把 `raw_verdict_census` 当作"已记录 verdict 字符串"的报告来读，不把它当成 accept/reject 决策。
- ✅ 对 `unresolved` 列表里的诚实缺口（缺失 Experience 等）保持 unresolved，不编造。
- ❌ 不得调用 `default_evidence_provider`（12 步门的 legacy 源）。
- ❌ 不得重算/制造统计 verdict、不得改动 12 步 fail-closed 门链。
- ❌ 不得写入任何历史载体 / 证据文件。

---

## 4. 验收对照（M4.0.1 验收条件）

| 验收项 | 落点 |
|--------|------|
| D. Discovery 边界明确 | §1 表 + §2 适配器 + 本契约文档 |
| M4.1 必须明确调用统一 Evidence Reader | §2 / §3（契约测试 `_self_source_uses_legacy_gate_provider`） |
| E. 历史 Evidence/metrics/verdict/records 不变 | 适配器只读；`default_evidence_provider` 未改 |
| 不降低统计门槛 / 不动 12 步门 | §3 ❌ 规则；适配器不触碰门链 |
| I. 不开发 M4.1 / 不建 Director | 本任务只交付契约 + 只读适配器 + 测试，未实现 M4.1 |

---

## 5. 交付物

- `research_runtime/discovery_evidence_adapter.py` — 只读适配器（facade）。
- `research_runtime/tests/test_m4_0_discovery_adapter.py` — 5 项契约测试（全绿）。
- 本契约文档。
