# M4.0 Phase 0 — 只读源码与历史证据审计

> 本阶段**未修改任何生产代码或历史文件**。以下结论均来自实际读文件 / 查询
> SQLite / 复现脚本，不依据历史报告推断。
> 基线：`2641e1c7`。审计时间：2026-11-14。

## 1. 复用的现有基础设施（不重建）

| 模块 | 事实 | 证据 |
|---|---|---|
| `research_runtime/research_store/db.py` | `ResearchDatabase`：`sqlite3.connect(timeout=30)`、`row_factory=Row`、`PRAGMA busy_timeout/foreign_keys/WAL`；`threading.RLock` + `session()` 的 `BEGIN IMMEDIATE` 事务；`schema_sql.split(";")` 逐条执行（`executescript()` 会隐式 COMMIT） | `db.py:49-94` |
| `research_runtime/research_store/__init__.py` | `ALL_SCHEMA_SQL` = 4 个 `SCHEMA_SQL` 拼接；`ResearchStores(db_path=None)` 单点构造；`default_db_path()` → `DATA_DIR/research.sqlite` | `__init__.py:56-75` |
| `research_runtime/research_store/schemas.py` | 已有 `StoredHypothesis` / `StoredExperimentPlan` / `ApprovalRecord` / `ResearchSession` / `CampaignBinding` / `ResearchReport` Pydantic 契约；`safe_id()` 校验 | `schemas.py:57-204` |
| `research_runtime/schemas.py` | 已有 `ResearchObjective`（`objective_id/domain/description/success_criteria/constraints/created_at`） | `schemas.py:70-77` |
| `research_runtime/research_store/research_session_store.py` | 已有 `record_experience()` / `experience()` + `research_experience` 表（append-only 结构化经验） | `:257-276`、`:53-60` |

**决策**：新增表以 `SCHEMA_SQL` 常量加入 `ALL_SCHEMA_SQL`，随 `ResearchStores()` 构造幂等迁移；**不新建独立 SQLite**、不引入 ORM。

## 2. 现有 `research.sqlite` 实况（只读查询）

路径 `multi-agent/research_runtime/_data/research.sqlite`（122,880 bytes），**被 `multi-agent/.gitignore:33` 忽略，不入库** → 迁移前后 schema / 统计必须以 JSON 快照落盘提交。

现有 12 张表 / 行数：
`approval_events=2, approvals=1, campaign_bindings=1, critic_reviews=2,
hypothesis_versions=2, plan_current=1, plan_versions=1,
research_experience=1, research_reports=1, research_session_events=10,
research_sessions=1, sqlite_sequence=3`

唯一完整历史链路（P1-06 会话，全部有明确 ID）：
```
objective_id  obj_pendulum_multistep
session       session_34cfebd14364
hypotheses    H_HORIZON_COSINE / H_HORIZON_LATENT_MSE
plan          plan_216e2d97c74d  (plan_sha256 939320db…9ec)
approval      approval_65a52ff0a6a5
campaign      campaign_bd7e4ec558b05655
experience    exp_d1e49b4a2b40
report        report_c327147de7cc
```

**关键事实**：对 `arr_m34_001` 的全文检索，上述 7 张表命中 **0 行**。
M3.4 真实运行走 `UnifiedResearchScheduler → run_dual_layer`，只写 JSON 证据文件，
**从未进入 P1-06 研究状态层**。因此：
- Campaign / Candidate / Evidence / run_id 可从 JSON 用明确 ID / SHA 确认；
- Objective / Hypothesis / Plan / Session / Experience 在历史中**无原生记录**，
  按任务书 §5「无法确认的历史关系标记 unresolved」与「不允许根据文件名或相似
  文本猜测实体身份」，不得把它们伪装成历史原生实体。

## 3. M3.4 历史证据的明确 ID（可用于 backfill）

`M3_4_runs/arr_m34_001/evidence_index.json`（`m3.4-evidence-index-v2`）：
- `run_id = arr_m34_001`
- 三域 `campaign_id`：`arr_m34_001_{scinet,embodied,quant}`
- 每域 3 个 `decisions`，键为 `hyp_fingerprint`
  （`domain_model::train.lr|None->0.003|DECISION_UNCERTAINTY` 等）
- `ood_provenance`：manifest SHA / config SHA / 40 checkpoint SHA / 40 run ID / 40 OOD metrics

`candidate_registry.jsonl` 每行含 `event / fingerprint / candidate_type / domain / target / status`，
`fingerprint` 与 evidence_index 的 decision 键**完全一致**（可直接做明确 ID 关联）。
`domain_summary.json` 含 `campaign_id / registry_path / records`（records 内含同一
`hyp_fingerprint`、原始 verdict 与 OOD 统计量）。

`research_runtime/_data/evidence.jsonl`（38 MB）：仅 P0/P1-06 时代的
`submitted` / `result` 事件，含 `experiment_id / run_id / checkpoint.sha256`，
`grep -c arr_m34_001` = **0** → 与 M3.4 运行无关，但仍是统一读模型必须支持的载体。

## 4. Evidence 载体与 `default_evidence_provider` 的缺口

`research_service.py:99-116` 只 `EvidenceStore().query(experiment_id=..., event_type="result")`
并 `ExperimentResult.model_validate`；M3.x 的 `evidence_index.json` /
`candidate_registry.jsonl` / `ood_provenance` / `*_metrics.json` 它**完全看不到**。

`evidence_bundle.py` 的发布流程审计结论：
- `build_bundle()` 仅在**自身 `main()`** 被调用（`evidence_bundle.py:307`）；
- `cli.py:218,260,290-301` 只消费 `base_state.json` 里的 `evidence_bundle_files` /
  `evidence_bundle` 路径并调用 `EvidenceVerifier.verify_bundle_integrity()`；
- docstring 声称 "wired into research_runtime.cli"，实际**未自动接线**。
→ 判定为**手动离线发布工具**：不删除、不自动接线（自动发布会写文件，与「统一
只读查询层」职责冲突）；统一读模型把它作为**可选只读载体**（若 release bundle
存在则读 `evidence_hashes.json`）。

## 5. 复现出的两个真实缺陷（先复现，再改）

### 5.1 P0-2 provenance axis 误分类（任务书 §8 点名）

构造反例：文件名 `ood_metrics.json`，但 `config_path` 指向 ID config、
`dataset_version = id-v1`；另一个文件名 `engine_metrics.json`（无 ood 字样）
但内容确为 OOD。实测：

```
engine_metrics.json -> ood  basis=config_path      (正确)
ood_metrics.json    -> ood  basis=filename         (错误!)
ood artifact count  = 2  (应为 1)
```

根因：`scheduler.py` 的 axis 判定用
`cfg_match or ds_match or "ood" in name` —— 文件名的弱信号能**覆盖**内容里
明确的 ID 信号。修复：内容优先，且当内容明确指向 ID 时**否决**文件名弱信号；
无任何内容信号时才用文件名，并标 `axis_basis="filename_unverified"`。

### 5.2 CI 与路径

仓库根为 `E:/agent_dev`，`multi-agent/` 是子目录；实际生效 workflow 是
`.github/workflows/arr-research-runtime.yml`（P0-4 已修复）。新测试必须加入该
根 workflow；测试中的相对路径必须 `Path.as_posix()` 化以兼容 Windows runner。

## 6. 本阶段设计边界

- 只做 M4.0；不开发 M4.1（objective discovery / ranker）、M4.2、M5。
- 不新建第二套 Memory / Evidence / Decision 系统；不引入向量库 / ORM。
- 不改历史 verdict、不动原始 evidence 文件、不降低统计门槛。
- 每个逻辑包独立 commit 并推送 `feature/arr-0.1-domain-registry`。
