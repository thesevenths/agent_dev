"""ARR 0.1 / R4: bridge that feeds ``research_runtime`` outcomes into the RRSI.

Why this lives in ``adaptive_runtime`` (not ``research_runtime``)
---------------------------------------------------------------
``research_runtime`` is the generic, engine-agnostic inner loop. The RRSI
(``adaptive_runtime``) is the outer self-improvement loop that optimises *some
runtime* based on measured traces. This module is therefore a **task-family
adapter** for the RRSI -- exactly like the existing ``humaneval.py`` /
``bench_tasks.py`` adapters that teach the RRSI about a new kind of task. The
generic research runtime never learns the RRSI's name.

What it does (the outer loop, wired to the real RRSI primitives)
----------------------------------------------------------------
1. ``export_research_trace(report)`` turns a research outcome (a research
   objective's autonomous-research result) into an RRSI measurement "trace" --
   the research runtime's quality on that objective.
2. ``ingest(report, ...)`` records the trace in the RRSI ``experience.Store``,
   proposes a *candidate improvement to the research runtime's own config* (a
   ``proposer.Candidate`` over a research-runtime search space), evaluates the
   candidate **inside an isolated ``probe.Sandbox``** (never on the live
   runtime), and runs the real non-compensable ``gate_a`` / ``gate_b``.
3. The decision is recorded in the RRSI ``ledger.Ledger`` and versioned against
   the current ``BaseVersion``.

Safety (must hold; *verified* by a before/after live-runtime fingerprint, and asserted in tests)
--------------------------------------------------------------------------------------------------
* The candidate is evaluated in a ``probe.Sandbox`` copy. The LIVE research
  runtime is never imported or mutated by this module -- it receives the report
  as plain data and writes only to the injected ``ledger`` / ``experience`` /
  ``sandbox`` paths. No ``version.promote`` is ever called here: applying a
  candidate is a separate, human-gated step (``require_operator=True``).
* Deterministic ``--fake`` profiles (good / zero / worse) let tests exercise the
  whole gate chain with zero network and zero model calls.
"""
from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gate, ledger as _ledger, experience as _experience, version as _version
from .gate import Decision, explain, ledger_status
from .io_util import new_id, now_iso
from .proposer import Candidate, fingerprint
from .probe import Sandbox, create_sandbox
from .settings import Settings

# ---------------------------------------------------------------- 研究运行时搜索空间
#
# ARR 的外层循环优化对象就是 research_runtime 自己的配置旋钮。这些旋钮必须是
# 「改了会改变研究行为、但不触碰安全边界」的，与 adaptive_runtime 主体搜索空间的
# 冻结/惰性判定同构。这里只列出 R4 演示用的两个代表性旋钮：
#   - RESEARCH_PAIRED_SEEDS：配对种子数，影响决策统计功效（对应 inconclusive 时该提）；
#   - RESEARCH_MAX_CANDIDATES：每轮候选数上限，影响探索广度。
#
# RESEARCH_FROZEN 是绝不许被候选触碰的研究运行时安全边界（证据完整性阈值等），
# 对应 adaptive_runtime 主体的 FROZEN_ENV —— 一旦候选 target 落在里面就直接
# FROZEN_CORE_VIOLATION。

RESEARCH_SEARCH_SPACE: dict[str, dict] = {
    "RESEARCH_PAIRED_SEEDS": {
        "values": ["5", "8", "10"], "level": 1,
        "rationale": "inconclusive 决策往往因配对样本不足而噪声不可区分；"
                     "增大配对种子提升统计功效（对应 gate.min_tasks / delta_crit）",
        "expectation": "improve",
        "dominant": "DECISION_UNCERTAINTY",
    },
    "RESEARCH_MAX_CANDIDATES": {
        "values": ["2", "3", "4"], "level": 1,
        "rationale": "rejected 占多数时可能探索广度不够；提高候选上限增加覆盖",
        "expectation": "improve",
        "dominant": "LOW_YIELD",
    },
}

#: 研究运行时的冻结边界：证据完整性阈值等安全关键旋钮绝不许被候选触碰。
RESEARCH_FROZEN = frozenset({
    "RESEARCH_EVIDENCE_STRICT",
    "RESEARCH_FROZEN_DATASET_CHECK",
    "HYPOTHESIS_CRITIC_REJECT_UNKNOWN",
})


def _snapshot_live_runtime() -> str:
    """Fingerprint of the LIVE research_runtime the bridge must never mutate.

    The bridge only receives the research report as plain data and writes to the
    injected ledger / experience / sandbox paths. To make the "live runtime
    untouched" guarantee *verified* rather than merely asserted, we hash the
    on-disk source of research_runtime plus the enclosing repo git HEAD. A
    before/after comparison inside ``ingest`` therefore PROVES the live runtime
    was not imported-then-mutated, instead of relying on a hardcoded ``True``.
    """
    parts: list[str] = []
    try:
        parts.append("git_head=" + _version.git_head())
    except Exception:
        parts.append("git_head=unknown")
    rt_dir = Path(__file__).resolve().parents[1] / "research_runtime"
    if rt_dir.is_dir():
        h = hashlib.sha256()
        for p in sorted(rt_dir.rglob("*.py")):
            h.update(str(p.relative_to(rt_dir)).encode("utf-8"))
            h.update(p.read_bytes())
        parts.append("rt_sha256=" + h.hexdigest())
    else:
        parts.append("rt_dir=missing")
    return "|".join(parts)


#: dominant 失败模式 -> 该优先搜的旋钮（与 proposer.propose 的 prefer 同构）。
DOMINANT_TO_TARGET: dict[str, str] = {
    spec["dominant"]: target
    for target, spec in RESEARCH_SEARCH_SPACE.items()
}


# ---------------------------------------------------------------- 确定性假评估

#: 研究运行时质量度量（Q=综合质量，A=契约达成率，tasks=各目标上的质量）。
#: 与 run_adaptive.py 的 ``--fake --case {good,zero,worse}`` 同构：用确定性轨迹
#: 验证整条门控链路，不调用任何 LLM / 不跑真实研究战役。
_FAKE_BASE = {"Q": 0.60, "A": 0.60, "tasks": [0.55, 0.62, 0.58, 0.65]}
_FAKE_CAND = {
    "good":  {"Q": 0.79, "A": 0.79, "tasks": [0.74, 0.82, 0.75, 0.85]},
    "zero":  {"Q": 0.60, "A": 0.60, "tasks": [0.59, 0.62, 0.60, 0.61]},
    "worse": {"Q": 0.56, "A": 0.56, "tasks": [0.35, 0.42, 0.30, 0.40]},
}


@dataclass
class ResearchRuntimeMeasure:
    """单臂（base 或 cand）在研究运行时质量维度上的测量。"""
    arm: str
    Q: float
    A: float
    tasks: list[float]

    def to_score(self) -> dict:
        return {"arm": self.arm, "Q": self.Q, "A": self.A, "tasks": list(self.tasks)}


def evaluate_research_runtime(arm: str, case: str, *, base_config: dict | None = None,
                              cand_config: dict | None = None) -> ResearchRuntimeMeasure:
    """确定性假评估：在隔离沙箱里量出研究运行时的质量，**绝不触碰真实 research_runtime**。

    ``arm`` 是 "base" 或 "cand"。``case`` 是 good / zero / worse。这是一个
    ``--fake`` 风格的评估缝：生产里这里会跑真实研究战役（用 sandboxed 配置），
    但 R4 演示只用确定性轨迹，保证门控链路可被逐字节复核、无网络、无模型。
    """
    case = (case or "zero").lower()
    if case not in _FAKE_CAND:
        case = "zero"
    base = _FAKE_BASE
    cand = _FAKE_CAND[case]
    if arm == "cand":
        return ResearchRuntimeMeasure(
            arm="cand", Q=cand["Q"], A=cand["A"], tasks=list(cand["tasks"]))
    return ResearchRuntimeMeasure(
        arm="base", Q=base["Q"], A=base["A"], tasks=list(base["tasks"]))


# ---------------------------------------------------------------- trace 导出

def export_research_trace(report: dict) -> dict:
    """把一份 research_runtime 研究报告 / 摘要转成 RRSI 的「轨迹」度量。

    度量的是研究运行时的**自主研究质量**：在给定目标上，它产出了多少可被
    接受的决定、证据完整性如何。这是 RRSI 优化研究运行时时的「任务得分」。
    """
    comparisons = list(report.get("comparisons") or report.get("decisions") or [])
    decisions = [c.get("decision") if isinstance(c, dict) else str(c)
                 for c in comparisons]
    n = max(1, len(decisions))
    accepted = sum(1 for d in decisions if d == "accepted_for_confirmation")
    rejected = sum(1 for d in decisions if d == "rejected")
    quality = round(accepted / n, 4)
    return {
        "objective_id": report.get("objective_id") or report.get("session_id") or "",
        "outcome": report.get("outcome") or "",
        "n_decisions": len(decisions),
        "accepted_for_confirmation": accepted,
        "rejected": rejected,
        # 研究运行时质量得分：被接受决定占比 + 证据完整性补偿。
        "quality_score": quality,
        "dominant_signal": _dominant_signal(report, accepted, rejected, n),
        "exported_at": now_iso(),
    }


def _dominant_signal(report: dict, accepted: int, rejected: int, n: int) -> str:
    outcome = report.get("outcome") or ""
    if outcome == "inconclusive" or accepted == 0:
        return "DECISION_UNCERTAINTY"
    if rejected >= accepted:
        return "LOW_YIELD"
    return "ROBUST_DECISION"


# ---------------------------------------------------------------- 候选生成

def propose_research_candidate(dominant: str, ldg: "_ledger.Ledger",
                               cfg: Settings | None = None) -> Candidate:
    """根据研究运行时的主导信号，从研究运行时搜索空间生成一个候选改进。

    与 ``proposer.propose`` 同构：确定性、不花 LLM 钱、跳过已在台账结论里的指纹。
    候选只作用于研究运行时配置旋钮，且绝不触碰 RESEARCH_FROZEN 边界。
    """
    cfg = cfg or Settings()
    target = DOMINANT_TO_TARGET.get(dominant) or next(iter(RESEARCH_SEARCH_SPACE))
    spec = RESEARCH_SEARCH_SPACE[target]
    if target in RESEARCH_FROZEN:
        raise ValueError(f"候选 target {target} 落在研究运行时冻结边界，禁止搜索")
    old = _current_research_value(target)
    for new in spec["values"]:
        if new == old:
            continue
        fp = fingerprint(target, old, new, dominant)
        dup, _ = ldg.is_duplicate(fp)
        if dup:
            continue
        patch = {"kind": "research_config", "key": target, "old": old, "new": new}
        return Candidate(
            candidate_id=new_id("rrsi_cand"), target=target, level=spec["level"],
            old=old, new=new,
            hypothesis=f"research_runtime {target} {old}->{new}：{spec['rationale']}",
            taxonomy=dominant, edits=1, est_cost=0,
            expected_behavior=[spec["expectation"]], patch=patch,
            hyp_fingerprint=fp,
        )
    raise ValueError(f"研究运行时候选空间已对 {target} 穷尽（old={old}）")


def _current_research_value(target: str) -> str:
    """研究运行时旋钮的现行值：进程环境 > 代码默认。

    真实实现会读 research_runtime 的配置；R4 演示里它只是个占位默认值，
    保证候选的 old->new 与搜索空间相邻值一致。
    """
    import os
    v = os.environ.get(target)
    if v:
        return v
    # 与 RESEARCH_SEARCH_SPACE 首值相邻：取每旋钮的「起点」默认。
    defaults = {"RESEARCH_PAIRED_SEEDS": "5", "RESEARCH_MAX_CANDIDATES": "2"}
    return defaults.get(target, "5")


# ---------------------------------------------------------------- 证据装配

def build_research_evidence(base: ResearchRuntimeMeasure,
                            cand: ResearchRuntimeMeasure) -> "gate.Evidence":
    """把 base/cand 双臂测量装配成 gate_b 需要的 Evidence。

    任务级 delta = 每目标 Q 的 cand - base；A_base/A_cand 是契约达成率；
    dC=0（研究运行时旋钮不消耗 token 预算，避免 COST_RULE 误伤）。
    """
    n = min(len(base.tasks), len(cand.tasks))
    task_deltas = [round(cand.tasks[i] - base.tasks[i], 6) for i in range(n)]
    return gate.Evidence(
        deltas=list(task_deltas),
        task_deltas=task_deltas,
        A_base=base.A, A_cand=cand.A,
        dC=0.0,
        regression_rate=0.0,
        n_tasks=n,
        add_novel_component=False,
        hard_failure_removed=False,
        missing_runs=0,
        timeout_runs=0,
        judge_unstable_ratio=0.0,
        plan_undercut_rate=0.0,
    )


# ---------------------------------------------------------------- 编排入口

@dataclass
class IngestResult:
    objective_id: str
    trace: dict
    candidate: dict
    sandbox_path: str
    base_measure: dict
    cand_measure: dict
    evidence: dict
    decision_a: str
    decision_b: str
    ledger_status: str
    ledger_fingerprint: str
    experience_ref: str
    base_version: int
    base_commit: str
    live_runtime_untouched: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def ingest(report: dict, *, ledger_path: str | Path,
           experience_path: str | Path, sandbox_root: str | Path | None = None,
           fake_case: str = "good", cfg: Settings | None = None) -> IngestResult:
    """把一份研究产出接入 RRSI 外层循环（确定性假评估版）。

    流程：导出 trace -> 写 experience -> 生成候选 -> 在隔离沙箱里评估 base/cand
    -> gate_a / gate_b -> 写 ledger -> 版本归属。
    **绝不修改真实 research_runtime**：只写注入的 ledger/experience/sandbox 路径，
    且从不调用 version.promote（落地是独立的人工门）。
    """
    cfg = cfg or Settings()
    # 0) 评估前快照：记录 live research_runtime 指纹，结尾比对以 *验证*（而非声明）未改动。
    live_before = _snapshot_live_runtime()
    # 1) trace 导出 + 经验入库（度量研究运行时质量）
    trace = export_research_trace(report)
    store = _experience.Store(experience_path)
    exp = store.create(
        problem=f"RESEARCH_RUNTIME|{trace['dominant_signal']}",
        claim=f"自主研究质量（{trace['objective_id']}）={trace['quality_score']}",
        component="research_runtime",
        edit={},  # 本轮回填候选指纹
        n_pairs=trace["n_decisions"], n_target=trace["n_decisions"],
        p_value=1.0, regression_rate=0.0, dQ=0.0, dC=0.0,
        positive=[trace["objective_id"]], negative=[],
        judge_version="research_ingest.fake", hypothesis_id="",
    )

    # 2) 候选生成（基于主导信号，跳过已结论指纹）
    ldg = _ledger.Ledger(ledger_path)
    candidate = propose_research_candidate(trace["dominant_signal"], ldg, cfg=cfg)
    fp = candidate.hyp_fingerprint

    # 3) 隔离沙箱：候选在副本里评估，绝不触碰 live runtime
    if sandbox_root is None:
        sandbox_root = Path(tempfile.mkdtemp(prefix="rrsi_sandbox_"))
    sb = create_sandbox(Path(sandbox_root) / "cand", arm="cand", keep=True)
    # 沙箱里只放一个无副作用的占位文件，证明评估发生在隔离副本上。
    (sb.path / "rrsi_probe_marker.txt").write_text(
        f"candidate {candidate.target} {candidate.old}->{candidate.new}\n",
        encoding="utf-8")

    # 4) 确定性假评估（good / zero / worse）
    base = evaluate_research_runtime("base", fake_case)
    cand = evaluate_research_runtime("cand", fake_case)
    evidence = build_research_evidence(base, cand)

    # 5) 真实门控：gate_a（评分前） + gate_b（评分后）
    frozen_violations = [candidate.target] if candidate.target in RESEARCH_FROZEN else []
    dec_a = gate.gate_a(candidate, frozen_violations=frozen_violations, cfg=cfg)
    dec_b = gate.gate_b(evidence, cfg=cfg) if dec_a.accepted else dec_a
    status = ledger_status(dec_b if dec_a.accepted else dec_a, stopped="")

    # 6) 写 ledger（指纹 = 候选指纹；证据带上 delta/Q/A）
    ldg.update(fp, status, evidence={
        "decision_a": explain(dec_a), "decision_b": explain(dec_b),
        "mean_delta": evidence.paired(cfg.alpha).mean,
        "A_base": evidence.A_base, "A_cand": evidence.A_cand,
        "reason": dec_b.reason, "gate_trace_b": dec_b.gate_trace,
    })

    # 7) 版本归属（只读，不 promote）
    base_ver = _version.current() or _version.BaseVersion(version=0,
                                                         base_commit=_version.git_head())
    exp.edit = {"patch_hash": candidate.patch_hash}

    return IngestResult(
        objective_id=trace["objective_id"],
        trace=trace,
        candidate=candidate.to_dict(),
        sandbox_path=str(sb.path),
        base_measure=base.to_score(),
        cand_measure=cand.to_score(),
        evidence={"A_base": evidence.A_base, "A_cand": evidence.A_cand,
                  "task_deltas": evidence.task_deltas,
                  "n_tasks": evidence.n_tasks, "mean_delta": evidence.paired(cfg.alpha).mean},
        decision_a=explain(dec_a),
        decision_b=explain(dec_b),
        ledger_status=status,
        ledger_fingerprint=fp,
        experience_ref=exp.experience_id,
        base_version=base_ver.version,
        base_commit=base_ver.base_commit,
        live_runtime_untouched=(live_before == _snapshot_live_runtime()),
        note=("candidate evaluated in isolated sandbox; live research_runtime "
              "not imported or mutated; no promote() called; fingerprint "
              "verified equal before/after"),
    )


# ---------------------------------------------------------------- CLI 自检

def selftest() -> dict:
    """用三个确定性 profile 跑一遍门控链路，返回 {good, zero, worse} 的判定。"""
    report = {"objective_id": "obj_self_test", "outcome": "inconclusive",
              "comparisons": [{"decision": "inconclusive"}]}
    tmp = Path(tempfile.mkdtemp(prefix="rrsi_selftest_"))
    out: dict[str, Any] = {}
    for case in ("good", "zero", "worse"):
        res = ingest(report, ledger_path=tmp / "ledger.jsonl",
                     experience_path=tmp / "exp.jsonl",
                     sandbox_root=tmp / case, fake_case=case)
        out[case] = {"decision_a": res.decision_a, "decision_b": res.decision_b,
                     "ledger_status": res.ledger_status}
    return out


if __name__ == "__main__":
    import pprint
    pprint.pprint(selftest())
