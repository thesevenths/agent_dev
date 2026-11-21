# ARR v1.0 — 02 Architecture

## 1. 分层（复用优先）

```
Research Program Layer  (NEW, thin — WP-02)
  ResearchProgramDriver / ProgramState / RoundRecord / BudgetManager / HistoryIndex
        |                         ^  (next-round planning from prior artifacts)
        v                         |
DiscoveryCoordinator (M7, single-round engine — unchanged core)
  literature -> novelty -> RQ -> value -> hypotheses -> experiment
  -> preregistration -> HUMAN_APPROVAL -> execution -> evidence -> claim -> memory
        |
        v  (EXPERIMENT_EXECUTION / REPLICATION via FormalRuntimeAdapter)
FormalRuntimeAdapter (M7.6 + v1.0 hardened — WP-01)
  M5 RuntimeState (journal/reservation/claim) + M6 HarnessLoader (SHA pin) + DomainPack engine
        |
        v
Scientific Engines: DomainPack(embodied) / SciNet++ (external assets)
        |
        v
ClaimEvidenceGraph + VerifiedResearchMemory (cross-round inheritance)
        |
        v
capability_bench (four arms / four groups / family-clustered stats) + MethodRegistry (WP-07/08)
```

## 2. 身份（独立，不可互替）

| 身份 | 来源 | 用途 |
|---|---|---|
| git commit SHA | `git rev-parse HEAD` | 运行源码版本 |
| harness SHA | M6 HarnessLoader.resolve_for_campaign | 研究方法版本（已批准） |
| preregistration SHA | freeze_preregistration | 实验协议版本 |
| dataset SHA | DomainPack dataset_identity | 数据版本 |
| method SHA | MethodRegistry config_sha | 方法论配置版本 |
| result SHA | 引擎结果 freeze_json | 结果完整性 |

审批（M4.3 带外 CLI）绑定 protocol SHA（v1.0 扩展为复合：protocol+dataset+method+harness）。

## 3. 关键契约

- **Runtime/Campaign**：FormalRuntimeAdapter 是唯一生产 seam；M5 CampaignRunner 保持 ML ExperimentPlan 语义，不合并。
- **Program/Round**：Program 管多轮、预算、跨轮继承；Round = 一次 DiscoveryCoordinator.run()，独立于单轮语义。
- **Claim/Evidence**：三态 verdict 显式映射；INCONCLUSIVE 不产生 support/refute 边。
- **Memory**：六类 + 生命周期；INCONCLUSIVE 不晋升 Verified finding；检索返回来源+限制。
- **RSI**：MethodRegistry + 真实失败驱动 Candidate + M6 晋级链；holdout 隔离；无瓶颈不生成。
- **Evaluation**：family 为聚类单元；外部 benchmark adapter 可选接入；专家盲评不可由 LLM 替代。

## 4. 不做什么

- 不新建 Runtime/Campaign/Discovery 框架。
- 不自动改生产源码、审批策略、证据校验规则、评价阈值、冻结 holdout、历史 Base 结果。
- 不把 API 成功/LLM 自评/工程 Gate PASS 当科学能力 PASS。
- 不伪造审批、Ground Truth、专家评审、benchmark 结果。

详见 `03_ADR/*.md`。
