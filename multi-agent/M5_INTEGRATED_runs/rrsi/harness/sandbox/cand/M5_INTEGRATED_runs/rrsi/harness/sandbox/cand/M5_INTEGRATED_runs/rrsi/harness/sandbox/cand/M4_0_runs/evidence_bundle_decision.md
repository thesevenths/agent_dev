# M4.0-04 — `evidence_bundle.build_bundle` 发布流程审计与接线决策

> 任务书 §7 要求「先明确其现有发布流程和调用关系，再决定接线或明确弃用。
> 不得直接删除历史工具」。本文件是**只读审计结论 + 决策记录**。

## 1. 实际调用关系（源码证据，非依据 docstring 推断）

| 事实 | 源码位置 |
|---|---|
| `build_bundle()` 定义 | `research_runtime/evidence_bundle.py:199` |
| `build_bundle()` **唯一调用点**：模块自身 `main()`（CLI 入口） | `evidence_bundle.py:307` |
| `cli.py` 从不 import `evidence_bundle` / `build_bundle` | Grep `cli.py`：`evidence_bundle` 仅作为字符串键出现 |
| `cli.py` 消费的是 `base_state.json` 里已发布的 `evidence_bundle_files` 路径列表 | `cli.py:218`、`cli.py:260` |
| `cli.py` 校验已发布 bundle 的自洽性（`EvidenceVerifier.verify_bundle_integrity`） | `cli.py:290-301` |
| `EvidenceVerifier.verify_bundle_integrity` 读 `evidence_hashes.json` 逐文件复核 | `evidence_verifier.py:207-259` |
| `evidence_bundle.py` docstring 声称 "wired into research_runtime.cli" | `evidence_bundle.py:22` — **与事实不符** |

全仓 `build_bundle` 调用点 Grep（排除历史 `experiments/` sandbox）：

```
research_runtime/evidence_bundle.py:199  def build_bundle(...)
research_runtime/evidence_bundle.py:307  result = build_bundle(...)   # 仅 main()
```

## 2. 真实发布流程

```
[人工] python -m research_runtime.evidence_bundle \
           --benchmark <benchmark json> --evolution <evolution json> \
           --name arr_1_0_r1 --out experiments/releases
   -> 生成 experiments/releases/<name>/{manifest.json, *_summary.json,
        verification.json, evidence_hashes.json, raw/}
   -> 产物提交入 git（该模块 P0-3 的初衷是修复被 .gitignore 忽略的 404）
[人工] 在 base_state.json 中登记 evidence_bundle / evidence_bundle_files
[自动] research_runtime.cli 读取登记的路径 -> 做 SHA 复核 -> 纳入验收
```

仓库中现存 1 个真实发布产物：`experiments/releases/arr_1_0_r1/evidence_hashes.json`。

**结论**：`build_bundle` 是**手动离线发布工具**，不在任何自动执行链上；
`cli.py` 是它的消费者，不是调用者。

## 3. 决策：不自动接线，保留工具；纳入统一读模型作为可选只读载体

理由：

1. **职责冲突**：`build_bundle` 会**写**文件（复制 raw artifacts、生成
   `evidence_hashes.json`）。M4.0-04 交付的是「统一**只读** Evidence 查询层」，
   自动接线会引入写副作用，违反任务书 §7「不覆盖原始证据」与 §3「只读」。
2. **语义风险**：自动重新发布会刷新 release 目录内容并改变既有
   `evidence_hashes.json` 的 SHA，直接影响 `cli.py:290-301` 对历史发布包的
   完整性判定，属于改动历史发布语义，超出 M4.0 范围。
3. **不删除历史工具**：`evidence_bundle.py` 原样保留，未修改一行；其 CLI
   仍可人工调用。docstring 与事实不符的问题在本文件记录，不改历史代码。
4. **纳入读模型**：`UnifiedEvidenceReader._read_bundle_releases()` 把
   `experiments/releases/*/evidence_hashes.json` 作为第 6 种载体**只读**枚举。
   bundle 里的哈希清单是「发布包文件清单」，不是数据集 manifest，
   因此**不**误标为 `manifest_sha256`，只在 carrier 记录中保留其数量。

## 4. 统一读模型支持的 6 种载体（实测 `arr_m34_001`）

| # | 载体 | 真实来源 | 识别方式 |
|---|---|---|---|
| 1 | `evidence_index` | `M3_4_runs/arr_m34_001/evidence_index.json` (v2) | 文件 SHA-256；OOD artifact 用其记录的 `file_sha256` |
| 2 | `domain_summary` | `<run>/<domain>/domain_summary.json` | 文件 SHA-256；保留原始 verdict 字符串 |
| 3 | `candidate_registry` | `<run>/<domain>/candidate_registry.jsonl` | 每行内容 SHA-256；保留原始 `status`/`to` |
| 4 | `engine_metrics` | `M2_7_runs/**/outputs/research_campaigns/<campaign_id>/*/*metrics*.json` | 文件 SHA-256；严格按 `campaign_id` 前缀限定 run |
| 5 | `evidence_jsonl` | `research_runtime/_data/evidence.jsonl` (38 MB) | 流式读取 + 结构化 `run_id` 校验，绝不按文件名推断 |
| 6 | `evidence_bundle` | `experiments/releases/*/evidence_hashes.json` | 文件 SHA-256（可选只读载体） |

实测结果（`M4_0_runs/evidence/unified_evidence_view.json`）：
`n_evidence=118`、`n_carriers_read=6`、**40 个 OOD artifact 被
`evidence_index` 与 `engine_metrics` 两个载体识别为同一实体**（重复识别生效）、
81 个实体带 checkpoint SHA / evaluation run ID、原始 verdict census 保留为
`inconclusive / rejected / proposed / ...` 原字符串（未重算）。

## 5. ResearchService 与 Scheduler 一致性

- 两者各自新增一个薄只读方法 `query_run_evidence(run_id, workspace=None)`，
  都委托 `research_runtime.evidence_read_model.query_run_evidence` 同一入口：
  - `research_service.py:ResearchService.query_run_evidence`
  - `scheduler.py:UnifiedResearchScheduler.query_run_evidence`
- 12 步 fail-closed 门链与 `default_evidence_provider` **未改动**；新方法
  不参与任何 verdict，不降低 Student-t / Holm / holdout 门槛。
- 实测 `ResearchService.query_run_evidence(...) == Scheduler.query_run_evidence(...)`
  → `True`，且重复查询结果确定一致。
