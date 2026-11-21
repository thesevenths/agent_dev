# M4 架构审计：Research Director 与统一研究知识模型

> **本阶段只做架构审计，不写任何 `director/` 实现代码。**
> 审计对象：`E:/agent_dev/multi-agent/research_runtime/` 真实源码（逐文件读取 + Grep 行号证据）。
> 基线 commit：`ed56e723`（P0-4 CI 修复之后）。
> 工程纪律：不 merge main、不开 PR、不删历史证据、不重建已有 Runtime、不降低统计门槛、不用 FakeRunner。

---

## 0. 结论先行

| 判断 | 结论 |
|---|---|
| 是否要新建第二套 Memory / Evidence 系统？ | **不要。** 结构化经验记忆 `research_experience` 表、版本化假设/计划、证据事件流均已存在，且被生产链调用。 |
| 6 个规划模块里真正缺失的是什么？ | **只有 Objective 层**（自主选题 + 排序 + 资源分配）确实缺失；`research_planner` / `research_memory` / `decision_engine` 均已有等价实现。 |
| 最大风险是什么？ | **ID 空间割裂**：`H_*` / `dom_*` / `rev_*` / `plan_*` / `campaign_*` / `session_*` / `exp_*` 七套 ID 无任何映射表；evidence 有 5 处载体但 `ResearchService` 只认其中 1 处。 |
| 建议的 M4 形态 | **一个薄协调层 + 一张 ID 映射表 + 复用现有 SQLite**，而不是 6 个新模块。 |

**回答任务书最终原则**：M4 若只做「6 个新模块脚手架」，并不能让 ARR 更接近自主研究——因为选题之后仍无统一的证据回溯。真正提升自主性的是：让 Director 能回答「过去哪些假设被证伪、为什么、用的是哪份冻结数据、哪次运行的证据」，而这需要先统一知识模型，而非先写调度器。

---

## 1. 当前代码模块映射（带行号证据）

### 1.1 活跃生产路径

| 模块 | 职责 | 持久化 | 关键证据 |
|---|---|---|---|
| `research_runtime/evidence_store.py` | append-only JSONL 事件审计流（submitted/result），明确「No dedup」 | `evidence.jsonl` | `EvidenceStore` :23；`append` :29；被 `experiment_manager.py:39,52` 写入 |
| `research_runtime/evidence_verifier.py` | 纯函数 SHA-256 / manifest / bundle 完整性校验 | 无 | `EvidenceVerifier.verify_report` :89；`verify_bundle_integrity` :207；被 `research_quality.py:276,291`、`cli.py:283-294` 调用 |
| `research_runtime/evolution/candidate_registry.py` | 候选状态机 + append-only 审计 | `candidate_registry.jsonl` | `register` :41、`transition` :66、`_append` :86-90；被 `evolution_controller.py:323` 调用 |
| `research_runtime/evolution/candidate.py` | 8 态状态机、`hyp_fingerprint` | 无 | `CandidateStatus` :33、`can_transition_to` :49、`DomainModelCandidate` :82 |
| `research_runtime/evolution/evolution_controller.py` | 双层 RRSI 真实验收入口 `run_dual_layer`；M3.0 Student-t + Holm 门控 | 由 scheduler 落盘 | `_decide_domain` :175-250、`_combine_dual_layer_decisions` :267-291、`run_dual_layer` :300 |
| `research_runtime/loop/loop_state.py` | 单域 loop 的原子 JSON 状态 + 版本继承 | `loop_state.json` | `LoopState` :60、`save` :129、`new_state` :150 |
| `research_runtime/loop/controller.py` | 调用 `run_dual_layer` 的内层循环 | — | :70-71、:127 |
| `research_runtime/research_store/db.py` | **已有 SQLite 基础设施**（WAL / BEGIN IMMEDIATE / RLock / 幂等迁移） | `research.sqlite` | `ResearchDatabase` :49、`session()` :81-94、`_connect()` :75-79 |
| `research_runtime/research_store/{hypothesis,experiment_plan,research_session,approval}_store.py` | 版本化假设/计划/会话/审批 | 同上 1 个 DB | `hypothesis_versions` `hypothesis_store.py:19-29`；`research_experience` `research_session_store.py:53-60` |
| `research_runtime/research_service.py` | 12 步 fail-closed 研究服务 | 委托上述 stores | :149；**被 `research_environment.py:195,218` 构造**（非幽灵） |
| `research_runtime/hypothesis_manager.py` | 规则/LLM 假设生成 + 实验计划 | 不持久化 | `RuleBasedProposer` :117、`build_experiment_plan` :347；被 `research_service.py:195,339,427` 调用 |
| `research_runtime/scheduler.py` | M3.4 跨域调度器 + 证据索引 + 幂等恢复 + P0-2 provenance | `scheduler_state.json` / `evidence_index.json` / `run_history.jsonl` | `UnifiedResearchScheduler` :392 |

### 1.2 半活跃 / 幽灵模块（避免误当基础设施）

| 模块 | 事实 | 证据 |
|---|---|---|
| `UnifiedResearchScheduler` | **只被 M3_4_runs 跑批脚本与测试引用**，`research_runtime/` 包内除自身零 import，`cli.py` 亦无 | `run_unified.py:36,116`、`reaggregate.py:20`、`rederive_evidence_index.py:37`、`verify_recovery.py:23` |
| `evidence_bundle.build_bundle` | **仅被自身 `main()` 调用**；`cli.py` 只消费「已生成的产物路径」，不 import 生成器 | `evidence_bundle.py:307`；`cli.py:218,260,290-301` |
| `research_runtime/orchestrator.py` / `run_real_research_loop.py` | 属 ARR 1.0 旧链，与 M3.x `run_dual_layer` 真实验收路径不同 | 历史遗留，未删除 |
| `longterm.py` | **不在 `research_runtime/`**：1 个在 `multi-agent/` 根 + 56 个在 `experiments/**/sandbox-*/` | 向量记忆表 `memories` `longterm.py:71-88`；被 `supervisor.py:43`、`graph.py:85` import（LangGraph Agent Runtime）；`research_runtime/` 内仅 `evidence_store.py:3` 注释提及，**零 import** |
| `director/` 及 6 个 M4 模块名 | **全仓零命中**，尚不存在 | Grep `objective_discovery|objective_ranker|...` 仅误命中 `directory` |

---

## 2. 现有模块可复用程度

评级：**可复用**=已有等价物且被生产链调用；**部分**=同类但语义/粒度不同；**缺失**=全仓无等价实现。

| M4 规划模块 | 已有等价物（路径:行） | 评级 | M4 应如何处置 |
|---|---|---|---|
| `objective_discovery.py` | `hypothesis_manager.RuleBasedProposer` :117 / `LLMProposer` :194；`research_service._propose` :312-380；`schemas.ResearchObjective` :70 | **部分** | 现有代码从实验结果发现 **hypothesis**；`objective` 仅是 `start_research_session` 的**入参**（`research_service.py:269-302`）。需新增「从历史经验生成 objective」的能力，但复用 `ResearchObjective` 数据结构 |
| `objective_ranker.py` | `scheduler._select_promotion` :206-248（确定性排序 + tie-break）；`domain_family_significance` `evolution_controller.py:144-172` | **部分** | 排序对象现为 candidate/target，非 objective。复用其「确定性、非执行顺序覆盖」原则，不新建评分框架 |
| `resource_allocator.py` | `research_policy.ResearchDomainPolicy` 预算字段 :48-53；`research_service._authorize` :615-630；`scheduler.DomainSpec.max_minutes/rounds` | **部分** | 现为**静态上限 + 校验**，无动态分配。M4 可做轻量预算分配，但必须复用 `ApprovalPolicy` / `_authorize` 门，不得绕过审批 |
| `research_planner.py` | `hypothesis_manager.build_experiment_plan` :347；`research_service._create_plan` :425-476；`ExperimentPlanStore` :49-101；`schemas.ExperimentPlan` :380 | **可复用（等价物已存在）** | **不新建**。plan 创建→SHA 版本化→审批→执行链已完整 |
| `research_memory.py` | `research_session_store.record_experience` :257-268 + `research_experience` 表 :53-60 + `experience()` :270；`ResearchSession.evidence_refs/experience_refs` `schemas.py:132-165`；`EvidenceStore` JSONL 审计流 | **可复用（等价物已存在）** | **不新建第二套 Memory**。`research_runtime/` 无向量记忆，但结构化经验记忆已足够起步；`longterm.py` 属另一子系统，不引入 |
| `decision_engine.py` | `_decide_domain` `evolution_controller.py:175-250`、`_combine_dual_layer_decisions` :267-291、`comparator._decide` `comparator.py:359-380`、`scheduler._select_promotion`、`candidate.py` 状态机 | **可复用（已有 4 份等价物，且偏多）** | **不新建**。真正需要的是「统一门面 + 去重」，而非第五个决策实现 |

**净结论**：6 个规划模块中，`research_planner` / `research_memory` / `decision_engine` **三个不应新建**；`objective_discovery` / `objective_ranker` / `resource_allocator` 是薄适配层。M4 的工程量应集中在「统一知识模型 + ID 映射 + 证据归一」，不是 6 个文件。

---

## 3. 重复职责与数据冲突风险（事实，非推测）

### 3.1 Hypothesis 三套 schema，ID 空间互不相交

1. 业务对象 `schemas.py:301-341` `Hypothesis`（15 字段，`status: proposed|approved|rejected|tested`）
2. 持久化 `research_store/schemas.py:57-66` `StoredHypothesis`（`content_sha256`/`version`/`hypothesis:dict`），键 `(hypothesis_id, content_sha256)` `hypothesis_store.py:28`
3. 演进候选 `evolution/candidate.py:82-128` `DomainModelCandidate`，`hypothesis` 字段**仅是 str** :100，键 `hyp_fingerprint` :104-107

七套 ID：`H_HORIZON_*`（`hypothesis_manager.py:164`）、`dom_*`（`candidate.py:111`）、`rev_*`（`hypothesis_store.py:135`）、`plan_*`（`research_service.py:451`）、`campaign_*`（:146）、`session_*`（`research_session_store.py:78`）、`exp_*`（:260）。**跨系统无任何 ID 映射表** → 无法回答「这条 ACCEPT 来自哪个 objective」。

### 3.2 Evidence 五处载体，但只有一个被消费

| # | 载体 | 位置 |
|---|---|---|
| 1 | `evidence.jsonl` 事件流 | `evidence_store.py:25`；`experiment_manager.py:39,52` 写入 |
| 2 | 发布目录 JSON + `raw/` + `evidence_hashes.json` | `evidence_bundle.py:204-260` |
| 3 | `evidence_index.json`（按 `hyp_fingerprint` 聚合） | `scheduler.py:_build_evidence_index` |
| 4 | `candidate_registry.jsonl` | `candidate_registry.py:86-90` |
| 5 | SQLite `research_session_events` / `research_experience` / `campaign_events` | `research_session_store.py:29-60`、`campaign_store.py:52-59` |

`research_service.default_evidence_provider` :99-116 **只认** `evidence.jsonl` 的 `event_type="result"`；M3.x 真实运行写的是 #3/#4。→ **M4 若直接读 `ResearchService`，会看不到 M3.x 的任何真实证据**。这是必须先解决的冲突。

### 3.3 状态持久化三套并存、互不引用

- `loop_state.json`（`loop_state.py:129-147`，`arr-1.1-loop-v1`）
- `scheduler_state.json` + `domain_summary.json`（`scheduler.py`，`m3.4-scheduler-v1`→`v2`）
- SQLite `campaigns`（`campaign_store.py:27-36`）与 `research.sqlite`（`research_store/__init__.py:63`）

Grep 未发现三者间交叉 import；`ResearchService` :741,:781,:897 只读 `campaign_store`，从不读 `LoopState` / `scheduler_state`。

### 3.4 决策逻辑四份等价物

`_decide_domain`（统计门）、`comparator._decide`（R1-R8 规则门）、`scheduler._select_promotion`（确定性晋升）、`research_service._finalize_report`（报告 outcome）。**风险**：M4 若再加一个 `decision_engine.py`，同一候选可能出现两套不同 verdict，且统计门槛有被绕过的风险。

---

## 4. 统一研究知识模型 Schema（提议，非实现）

**原则**：不新建表替代现有表；只补「关系」与「缺失的 objective 主体」。所有字段可追溯到现有 schema。

| 实体 | 复用来源 | 主键 | 关键关系 |
|---|---|---|---|
| `Objective` | `schemas.ResearchObjective` :70（`objective_id/domain/description/constraints/success_criteria/created_at`） | `objective_id` | 新增 `parent_objective_id`（可空，支持分解）、`status`、`origin_experience_id` |
| `Hypothesis` | `schemas.Hypothesis` :301 + `StoredHypothesis` 版本化 | `(hypothesis_id, content_sha256)` | `objective_id` 已有；新增 `candidate_fingerprint` 指向 `hyp_fingerprint` |
| `Experiment` | `schemas.ExperimentPlan` :380 + `candidate_registry.jsonl` | `plan_id` + `plan_sha256`（已有） | 新增 `run_id` / `campaign_id` / `evaluation_run_id`（P0-2 已能追溯） |
| `Evidence` | `scheduler.evidence_index.json`（v2）+ `ood_provenance` + `evidence.jsonl` | `evidence_id` = SHA256(content) | `artifact_sha256` / `manifest_sha256` / `checkpoint_sha256` / `dataset_version` / `evaluation_run_id`（P0-2 字段直接复用） |
| `Experience` | `research_experience` 表（已存在） | `experience_id` | `objective_id` 已存在；`kind` 建议固定枚举：`accepted` / `rejected` / `inconclusive` / `regime_shift` / `failure_diagnosis` |

**唯一新增的结构**：`entity_links`（ID 映射表），把 `objective_id ↔ hypothesis_id ↔ hyp_fingerprint ↔ plan_id ↔ campaign_id ↔ session_id ↔ evidence_id ↔ experience_id` 串起来。这是让「自主积累经验」可查询的最小必要结构，不是第二套 Memory。

---

## 5. Research Director 数据流（提议）

```
[Experience/Evidence 查询]  ← 复用 research_experience + evidence_index(v2) + evidence.jsonl
        │  (只读，归一经 entity_links)
        ▼
[Objective 发现]   ← 薄层：从失败/未决经验生成候选 objective，复用 ResearchObjective
        ▼
[排序]             ← 复用 _select_promotion 的确定性规则（最大期望改善 + tie-break）
        ▼
[资源分配]         ← 复用 ResearchDomainPolicy 预算 + _authorize 审批门（不绕过）
        ▼
[计划]             ← 已存在：build_experiment_plan → ExperimentPlanStore(版本化/SHA) → Approval
        ▼
[执行]             ← 已存在：run_dual_layer（M3.x 真实验收路径）/ UnifiedResearchScheduler（跨域）
        ▼
[判定]             ← 已存在：_decide_domain (Student-t + Holm + holdout) + ID/OOD 组合
        ▼
[证据落盘]         ← 已存在：candidate_registry / evidence_index(v2) / ood_provenance
        ▼
[经验回写]         ← 已存在：record_experience + entity_links 补齐关联 → 闭环
```

**关键约束**：Director 不产生 verdict、不改统计门槛、不写伪造证据；它只做「选什么 → 分配多少 → 复用哪个已有执行链 → 把结果关联回知识模型」。

---

## 6. SQLite 持久化设计（复用，不新建基础设施）

**直接复用** `research_store/db.py` 的 `ResearchDatabase`：
- `sqlite3.connect(timeout=30)`、`row_factory=sqlite3.Row`、`PRAGMA busy_timeout=30000` / `foreign_keys=ON`（`db.py:75-79`）
- WAL 在独立连接一次性设置（`db.py:59-64`）
- `threading.RLock` + `session()` 的 `BEGIN IMMEDIATE` 原子事务，每次事务新建/关闭连接（`db.py:55,81-94`）
- 迁移约定：`schema_sql.split(";")` 逐条 `execute`（避免 `executescript()` 隐式 COMMIT），全部 `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` 幂等（`db.py:65-72`；`research_store/__init__.py:56-58,70-71`）
- DB 路径：`default_db_path()` → `DATA_DIR/research.sqlite`（`research_store/__init__.py:61-63`）

**M4 只需追加 2 张表到 `ALL_SCHEMA_SQL`**（不新建 DB、不新建连接层）：

```sql
-- 1) objective 主体（现有 schema 只有作为入参的 ResearchObjective，无持久表）
CREATE TABLE IF NOT EXISTS objectives (
    objective_id          TEXT PRIMARY KEY,
    domain                TEXT NOT NULL,
    description           TEXT NOT NULL,
    constraints_json      TEXT NOT NULL,
    success_criteria_json TEXT NOT NULL,
    parent_objective_id   TEXT,
    status                TEXT NOT NULL,           -- proposed|active|closed|abandoned
    origin_experience_id  TEXT,
    content_sha256        TEXT NOT NULL,
    created_at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_objectives_domain ON objectives(domain, status);

-- 2) 跨系统 ID 映射（解决 §3.1 的七套 ID 割裂）
CREATE TABLE IF NOT EXISTS entity_links (
    link_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    objective_id   TEXT NOT NULL,
    entity_type    TEXT NOT NULL,   -- hypothesis|candidate|plan|campaign|session|evidence|experience
    entity_id      TEXT NOT NULL,
    run_id         TEXT,
    created_at     TEXT NOT NULL,
    UNIQUE(objective_id, entity_type, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_links_entity ON entity_links(entity_type, entity_id);
```

**明确不做**：不引入向量库、不引入 ORM、不复制 `longterm.py` 的 `memories` 表、不新建第二套 `campaign_store.py` 风格的独立连接实现（它无 WAL/RLock，是已知的较差范式）。

---

## 7. M4 分阶段实施计划（本阶段不开发）

| 阶段 | 内容 | 依赖 | 是否新建模块 |
|---|---|---|---|
| **M4.0 知识模型归一** | 追加 `objectives` + `entity_links` 两表；写一次性回填脚本，把 M3.x 已有 `evidence_index(v2)` / `candidate_registry` / `research_experience` 关联到既有 `objective_id`；修 `default_evidence_provider` 使其能读 #3/#4 载体（或提供统一只读视图） | P0-2 provenance 已就绪 | 否（改现有 store + 迁移） |
| **M4.1 最小自主选题** | 只读查询近 N 次经验 → 生成 objective 候选 → 确定性排序 → 落 `objectives` | M4.0 | 薄层 `objective_discovery` + `objective_ranker`（2 个文件，非 6 个） |
| **M4.2 复用式执行闭环** | Director 把选中的 objective 交给**已有** `build_experiment_plan` → `run_dual_layer` → `record_experience`，全程不新建执行链 | M4.1 | 否；`resource_allocator` 仅作为 `ResearchDomainPolicy` 预算适配器 |
| **M4.3 持续改进证明** | 冻结评测任务，多轮运行对比「有 Director 选题」vs「固定人工选题」的 ACCEPT 率 / 证据完整率 / FP 率 | M4.2 | 否 |

**明确不做的部分**：不建 `research_memory.py`、不建 `decision_engine.py`、不建 `research_planner.py`、不引入向量记忆、不做大规模重构、不把 `longterm.py` 搬进 `research_runtime/`。

---

## 8. 每阶段验收标准（可验证，不靠自述报告宣布通过）

| 阶段 | 验收标准 | 证据形式 |
|---|---|---|
| **M4.0** | ① `research.sqlite` 迁移幂等（连续跑 2 次无报错、表结构一致）；② 对 `arr_m34_001` 三域每条 candidate，都能通过 `entity_links` 反查到 objective / campaign / evidence / experience；③ `ResearchService` 与 `scheduler` 读到的 evidence 集合一致；④ 既有 M3.0/M3.2/M3.3/M3.4 全量测试零回归 | 迁移前后 schema dump、SQL 查询结果、CI 绿色运行链接 |
| **M4.1** | ① 相同输入 → 相同 objective 排序（确定性，非执行顺序）；② 生成的 objective 满足 `ResearchObjective` schema 且 `success_criteria` 可判定；③ 不产生任何新的 verdict（判定仍由 `_decide_domain` 负责）；④ 每个 objective 可追溯到触发它的 experience_id | 单元测试（含 tie-break 回归）+ 落库记录 |
| **M4.2** | ① 一次 Director 驱动的真实运行（至少 1 域真实引擎，非 FakeRunner）跑通 plan→execute→evidence→experience；② `entity_links` 全链路无断链；③ 统计门槛与 M3.0 完全一致（Student-t + Holm + holdout），不得为通过而放宽；④ 崩溃恢复后不重复执行已完成域 | 真实运行日志 + `run_history.jsonl` + 测试 |
| **M4.3** | ① 冻结评测任务上多轮对照，报告 ACCEPT 数 / 证据完整率 / FP-FN 变化；② 与 M3.4 基线（`arr_m34_001`：0 ACCEPTED，scinet/quant INCONCLUSIVE、embodied REJECTED）对比且**如实报告退化**；③ 所有数字可从磁盘证据复算，禁止手写结论 | 对照表 + 可复算脚本 + CI 链接 |

---

## 9. 遗留问题与 M4 前置条件

1. **`evidence_bundle.build_bundle` 是幽灵生成器**：`cli.py` 只消费产物路径，不调用生成器（§1.2）。M4.0 需决定是接线还是明确弃用，避免「发布证据」无人真正生成。
2. **`UnifiedResearchScheduler` 未接入 `cli.py`**：目前只有 `M3_4_runs/run_unified.py` 能驱动跨域运行（§1.2）。M4.2 的闭环要么接线、要么明确标注为跑批工具。
3. **两套 SQLite 连接范式不一致**：`research_store/db.py`（WAL+RLock）vs `campaign/campaign_store.py`（`executescript`，无 WAL/RLock，:75-89）。M4 不应再引入第三种。
4. **ID 割裂是 M4 最大前置阻塞**：不先做 `entity_links`，任何「长期记忆」都无法回答「这条经验来自哪个目标、哪次实验」。
5. **P0-4 CI 远程结果待确认**：本次审计时 `ed56e723` 的 3 个 workflow 仍在运行；最终结论以 `M3_4_HARDENING_ACCEPTANCE_REPORT.md` 记录的远程 run 链接与 conclusion 为准。

---

## 10. 审计边界

本次仅做只读源码审计（Read / Grep / git ls-files），**未创建 `director/`、未修改任何生产模块、未执行任何重构**。`experiments/` 下 56 份 `longterm.py` 按历史 sandbox 副本计，未逐个打开；`campaign/schemas.py`、`comparison/report_store.py`、`model_registry.py` 仅通过 Grep 确认存在与行号，未对其内部字段作断言。
