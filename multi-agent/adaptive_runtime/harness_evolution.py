"""ARR 1.0 / Package F -- REAL RRSI Harness self-improvement.

This is the system's differentiator: the OUTER loop improves the RESEARCH
SYSTEM, not the scientific model. Package C optimised SciNet++ training knobs;
here the candidate IS a change to the Planner / Experiment Designer / Reviewer
/ Workflow / Memory configuration.

Closed loop (F2/F3/F4)::

    Collect    real TaskResults (report, trajectories, metrics, costs, holdout)
      -> Diagnose  research_failure_analyzer (structured, evidence-backed)
      -> Generate  single-knob Harness candidates (name-spaced ledger keys)
      -> Sandbox   each candidate gets an isolated workspace + harness.json
      -> Experiment SAME tasks, SAME paired seeds, Base vs Candidate,
                    via the SAME ResearchService and the SAME real engines
      -> Evaluate  FIXED domain ResearchQualityEvaluator (outside candidate reach)
                   + independent held-out / OOD tasks + negative control
      -> Gate      real gate_a / gate_b -> ACCEPT / REJECT / INCONCLUSIVE
      -> Record    Ledger + Experience + BaseVersion; NO automatic promotion
      -> Promote   explicit human operator only (version.promote / restore)

Safety invariants (F4), all verified with hashes rather than self-declarations:
  * ``QUALITY_FROZEN_FIELDS`` / ``is_frozen`` reject any candidate that tries
    to edit the evaluator, weights, gate, splits or evidence rules.
  * Before/after SHA-256 of the evaluation substrate proves the ruler did not
    move while the candidate was being scored.
  * The reserved test/OOD splits are never used for candidate selection.
  * Max rounds / no-gain stop / cost and time budget / negative control.
  * No ``version.promote`` is called by the loop; rollback is available.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import gate
from .evolution_spaces import (
    HARNESS_FROZEN,
    HARNESS_SEARCH_SPACE,
    HARNESS_TO_TARGET,
    CandidateType,
    is_frozen,
    namespaced_fingerprint,
    space_for,
)
from .experience import Store as ExperienceStore
from .ledger import Ledger
from .probe import create_sandbox
from .research_failure_analyzer import analyze_many
from .settings import Settings

#: Files a Harness candidate must NEVER be able to change while being scored.
EVALUATION_SUBSTRATE = (
    "research_runtime/research_quality.py",
    "research_runtime/research_task.py",
    "adaptive_runtime/gate.py",
    "adaptive_runtime/evolution_spaces.py",
)

#: Which ResearchHarness field each whitelisted knob maps to, and how to coerce
#: the string value from the search space into that field's type.
KNOB_FIELD: dict[str, tuple[str, Any]] = {
    "planner.strategy": ("planner_strategy", str),
    "planner.hypothesis_prompt": ("hypothesis_prompt", str),
    "memory.experience_top_k": ("experience_top_k", int),
    "workflow.max_candidates": ("max_candidates", int),
    "workflow.campaign_rounds": ("campaign_rounds", int),
    "reviewer.require_falsification": ("require_falsification",
                                       lambda v: bool(int(v))),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation_substrate_snapshot(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in EVALUATION_SUBSTRATE:
        path = root / rel
        out[rel] = _sha256(path) if path.is_file() else "MISSING"
    return out


@dataclass
class HarnessCandidate:
    """One CONTROLLED, single-knob change to the Research Harness.

    Field names mirror ``adaptive_runtime.proposer.Candidate`` so the EXISTING
    ``gate.gate_a`` accepts it unchanged (level / edits / est_cost / target).
    """

    candidate_id: str
    target: str
    old: Any
    new: Any
    taxonomy: str
    hypothesis: str = ""
    level: int = 1
    edits: int = 1
    est_cost: int = 0
    patch: dict[str, Any] = field(default_factory=dict)
    hyp_fingerprint: str = ""
    expected_behavior: list[str] = field(default_factory=list)
    candidate_type: str = CandidateType.RESEARCH_HARNESS.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "target": self.target,
            "old": self.old, "new": self.new, "taxonomy": self.taxonomy,
            "hypothesis": self.hypothesis, "level": self.level,
            "edits": self.edits, "est_cost": self.est_cost,
            "patch": dict(self.patch), "hyp_fingerprint": self.hyp_fingerprint,
            "candidate_type": self.candidate_type,
        }


@dataclass
class EvolutionRecord:
    round_no: int
    dominant: str
    candidate: dict[str, Any]
    base_Q: list[float] = field(default_factory=list)
    cand_Q: list[float] = field(default_factory=list)
    task_deltas: list[float] = field(default_factory=list)
    holdout: dict[str, Any] = field(default_factory=dict)
    negative_control: dict[str, Any] = field(default_factory=dict)
    decision: str = "INCONCLUSIVE"
    decision_a: str = ""
    decision_b: str = ""
    reason: str = ""
    ledger_fingerprint: str = ""
    substrate_unchanged: bool = True
    substrate_sha256: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def generate_harness_candidates(*, dominant: str,
                                base_harness: Any,
                                ledger: Ledger,
                                limit: int = 3) -> list[HarnessCandidate]:
    """Diagnosis -> controlled, single-knob Harness patches (F2).

    Reuses ``evolution_spaces.target_for`` for the dominant failure and adds
    the next-highest-value whitelisted knobs. Every candidate is checked
    against the frozen boundary and against already-decided ledger entries.
    """
    from .io_util import new_id

    space = space_for(CandidateType.RESEARCH_HARNESS)
    ordered = [HARNESS_TO_TARGET.get(dominant, "planner.strategy")]
    ordered += [k for k in HARNESS_SEARCH_SPACE if k not in ordered]
    out: list[HarnessCandidate] = []
    for target in ordered:
        if len(out) >= limit:
            break
        if target not in KNOB_FIELD or is_frozen(CandidateType.RESEARCH_HARNESS,
                                                 target):
            continue
        field_name, coerce = KNOB_FIELD[target]
        old = getattr(base_harness, field_name)
        spec = HARNESS_SEARCH_SPACE[target]
        for raw_new in spec["values"]:
            new = coerce(raw_new)
            if new == old:
                continue
            fp = namespaced_fingerprint(
                CandidateType.RESEARCH_HARNESS, target=target, old=old,
                new=new, taxonomy=dominant)
            if ledger.is_duplicate(fp)[0]:
                continue
            out.append(HarnessCandidate(
                candidate_id=new_id("harness_cand"), target=target,
                old=old, new=new, taxonomy=dominant,
                hypothesis=f"Research Harness {target} {old}->{new}: "
                           f"{spec['rationale']}",
                level=1, edits=1, est_cost=0,
                patch={"kind": "research_harness", "target": target,
                       "old": old, "new": new},
                hyp_fingerprint=fp,
                expected_behavior=[spec["expectation"]],
            ))
            break
    return out


def degenerate_control_candidate(*, base_harness: Any,
                                 dominant: str) -> HarnessCandidate:
    """A deliberately DEGENERATE Harness change that MUST be rejected.

    Acceptance criterion #11: the SAME machinery that accepts a genuine
    improvement must be able to reject a bad one using REAL paired evidence.
    ``repeat_refuted`` makes the planner re-propose an intervention already
    known to be inert -- a valid Harness value, but a known-bad research
    policy. This control goes through the FULL protocol (same tasks, same
    paired seeds, same gate, same ledger), never a side-channel shortcut.
    Its ``taxonomy`` labels it explicitly so it cannot be mistaken for a
    genuine self-improvement attempt in the artifact.
    """
    from .io_util import new_id

    target = "planner.strategy"
    old = base_harness.planner_strategy
    new = "repeat_refuted"
    fp = namespaced_fingerprint(
        CandidateType.RESEARCH_HARNESS, target=target, old=old, new=new,
        taxonomy="NEGATIVE_CONTROL")
    return HarnessCandidate(
        candidate_id=new_id("harness_control"), target=target,
        old=old, new=new, taxonomy="NEGATIVE_CONTROL",
        hypothesis="NEGATIVE CONTROL: planner repeats an inert, already-"
                   "refuted intervention; must be REJECTed by paired evidence",
        level=1, edits=1, est_cost=0,
        patch={"kind": "research_harness", "target": target,
               "old": old, "new": new},
        hyp_fingerprint=fp,
        expected_behavior=["degrade"],
    )


def apply_harness_patch(base_harness: Any, candidate: HarnessCandidate,
                        *, sandbox_root: Path, repo_root: Path
                        ) -> tuple[Any, dict[str, Any]]:
    """Materialise a REAL Harness patch in an isolated sandbox (F3 step 4).

    Unlike Package C's marker file, this writes the actual ``harness.json`` the
    research loop reads, verifies its SHA-256 and verifies that NO frozen field
    was introduced. The returned object is what actually gets executed.
    """
    target = candidate.target
    if target not in KNOB_FIELD:
        raise ValueError(f"unknown Harness knob {target!r}")
    if is_frozen(CandidateType.RESEARCH_HARNESS, target):
        raise ValueError(f"candidate attempts to modify frozen target {target!r}")
    field_name, coerce = KNOB_FIELD[target]
    new_value = coerce(candidate.new)
    forbidden = {k.lower() for k in HARNESS_FROZEN}
    if any(k.lower() in forbidden for k in candidate.patch):
        raise ValueError("candidate patch contains frozen evaluation fields")

    # Sandbox SOURCE is intentionally MINIMAL: only the Harness declaration is
    # copied into the isolated workspace. Copying the whole repo root would drag
    # in thousands of engine job/result files (runtime artefacts), which is both
    # slow and a provenance hazard. Real execution isolation is provided by the
    # per-arm `data_dir` in run_research_task (its own sqlite / evidence /
    # campaign stores); the substrate hash below protects the FIXED evaluator.
    source_root = sandbox_root / f"_source_{candidate.candidate_id}"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "harness_source.json").write_text(
        json.dumps({"base_harness": base_harness.to_dict(),
                    "candidate_type": CandidateType.RESEARCH_HARNESS.value},
                   indent=2, sort_keys=True), encoding="utf-8")
    sandbox = create_sandbox(sandbox_root / candidate.candidate_id,
                             arm="cand", root=source_root, keep=True)
    patched = replace(base_harness, **{field_name: new_value})
    patched.validate()
    payload = {
        "candidate_id": candidate.candidate_id,
        "candidate_type": CandidateType.RESEARCH_HARNESS.value,
        "base_harness_sha256": base_harness.fingerprint(),
        "candidate_harness_sha256": patched.fingerprint(),
        "patch": candidate.to_dict(),
        "harness": patched.to_dict(),
        "frozen_fields": sorted(HARNESS_FROZEN),
    }
    path = sandbox.path / "harness.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               ensure_ascii=False), encoding="utf-8")
    (sandbox.path / "harness_sha256.txt").write_text(
        _sha256(path), encoding="utf-8")
    return patched, {
        "sandbox_path": str(sandbox.path),
        "harness_path": str(path),
        "harness_sha256": _sha256(path),
        "base_sha256": base_harness.fingerprint(),
        "candidate_sha256": patched.fingerprint(),
        "config_changed": base_harness.fingerprint() != patched.fingerprint(),
        "frozen_introduced": [k for k in patched.to_dict()
                              if k.lower() in forbidden],
    }


def _lessons(experience_path: Path, problem: str, top_k: int) -> list[str]:
    if top_k <= 0:
        return []
    try:
        return ExperienceStore(experience_path).lessons_for(problem, top=top_k)
    except Exception:
        return []


def _run_arm(task: Any, harness: Any, *, run_root: Path,
             experience_path: Path) -> dict[str, Any]:
    from research_runtime.arr_runtime import run_research_task
    lessons = _lessons(experience_path, f"RESEARCH_HARNESS|{task.domain}",
                       harness.experience_top_k)
    result = run_research_task(task, harness=harness, run_root=run_root,
                               experiences=lessons)
    return result.model_dump(mode="json")


def evaluate_candidate(*, base_harness: Any, cand_harness: Any,
                       candidate: HarnessCandidate,
                       tasks: list[Any], heldout_task: Any,
                       run_root: Path, experience_path: Path,
                       repo_root: Path, cfg: Settings | None = None,
                       negative_harness: Any | None = None) -> EvolutionRecord:
    """Base/Candidate paired experiments on the SAME tasks + fixed evaluator."""
    cfg = cfg or Settings()
    record = EvolutionRecord(round_no=0, dominant=candidate.taxonomy,
                             candidate=candidate.to_dict())
    before = evaluation_substrate_snapshot(repo_root)
    record.substrate_sha256 = dict(before)
    # EVERY candidate gets its OWN isolated store. Otherwise the second
    # candidate's Base arm would replay a bound campaign and the service would
    # return `already_submitted` (no campaign_id) instead of real evidence.
    arm_root = run_root / candidate.candidate_id
    arm_root.mkdir(parents=True, exist_ok=True)

    base_results: list[dict[str, Any]] = []
    cand_results: list[dict[str, Any]] = []
    for task in tasks:
        base_results.append(_run_arm(task, base_harness, run_root=arm_root,
                                     experience_path=experience_path))
        cand_results.append(_run_arm(task, cand_harness, run_root=arm_root,
                                     experience_path=experience_path))
    def _q(run: dict[str, Any]) -> float | None:
        value = (run.get("quality") or {}).get("Q")
        return float(value) if isinstance(value, (int, float)) else None

    # A FAILED arm is incomplete evidence, never a silently imputed zero.
    incomplete_arms = sum(
        1 for r in (base_results + cand_results) if _q(r) is None)
    base_Q = [_q(r) if _q(r) is not None else 0.0 for r in base_results]
    cand_Q = [_q(r) if _q(r) is not None else 0.0 for r in cand_results]
    deltas = [round(c - b, 6) for b, c in zip(base_Q, cand_Q)]
    record.base_Q, record.cand_Q, record.task_deltas = base_Q, cand_Q, deltas

    # Independent held-out task: NEVER part of gate_b's task_deltas.
    hold_base = _run_arm(heldout_task, base_harness, run_root=arm_root,
                         experience_path=experience_path)
    hold_cand = _run_arm(heldout_task, cand_harness, run_root=arm_root,
                         experience_path=experience_path)
    hold_base_q, hold_cand_q = _q(hold_base), _q(hold_cand)
    incomplete_arms += sum(1 for q in (hold_base_q, hold_cand_q) if q is None)
    h_delta = round((hold_cand_q or 0.0) - (hold_base_q or 0.0), 6)
    record.holdout = {
        "task_id": heldout_task.task_id, "domain": heldout_task.domain,
        "base_Q": hold_base_q,
        "cand_Q": hold_cand_q,
        "delta": h_delta,
        "confirmed": h_delta > 0.0,
        "holdout_used_for_selection": False,
        "base_result_path": hold_base["provenance"].get("result_path"),
        "cand_result_path": hold_cand["provenance"].get("result_path"),
    }

    # NEGATIVE CONTROL: the SAME machinery must reject a degenerate Harness.
    if negative_harness is not None:
        neg = _run_arm(heldout_task, negative_harness, run_root=arm_root,
                       experience_path=experience_path)
        neg_q = _q(neg)
        incomplete_arms += 1 if neg_q is None else 0
        neg_delta = round((neg_q or 0.0) - (hold_base_q or 0.0), 6)
        record.negative_control = {
            "harness_sha256": negative_harness.fingerprint(),
            "Q": neg_q, "delta_vs_base": neg_delta,
            "not_accepted": neg_delta <= 0.0,
            "outcome": neg["outcome"],
        }

    missing = incomplete_arms + sum(
        1 for r in (base_results + cand_results)
        if int((r.get("telemetry") or {}).get("missing_runs") or 0) > 0)
    timeouts = sum(1 for r in (base_results + cand_results)
                   if int((r.get("telemetry") or {}).get("timeout_runs") or 0) > 0)
    A_base = sum(1 for r in base_results if r.get("ok")) / max(1, len(base_results))
    A_cand = sum(1 for r in cand_results if r.get("ok")) / max(1, len(cand_results))
    regression_rate = sum(1 for d in deltas if d < 0) / max(1, len(deltas))
    evidence = gate.Evidence(
        deltas=list(deltas), task_deltas=deltas,
        A_base=A_base, A_cand=A_cand, dC=0.0,
        regression_rate=regression_rate, n_tasks=len(deltas),
        add_novel_component=False, hard_failure_removed=False,
        missing_runs=missing, timeout_runs=timeouts,
        judge_unstable_ratio=0.0, plan_undercut_rate=0.0,
    )
    frozen_violations = [candidate.target] if is_frozen(
        CandidateType.RESEARCH_HARNESS, candidate.target) else []
    dec_a = gate.gate_a(candidate, frozen_violations=frozen_violations, cfg=cfg)
    dec_b = gate.gate_b(evidence, cfg=cfg) if dec_a.accepted else dec_a
    record.decision_a = gate.explain(dec_a)
    record.decision_b = gate.explain(dec_b)
    # A candidate that wins the paired test but FAILS the independent held-out
    # task (or where the negative control was accepted) is NOT accepted.
    holdout_ok = bool(record.holdout.get("confirmed"))
    negative_ok = (not record.negative_control
                   or bool(record.negative_control.get("not_accepted")))
    if dec_b.accepted and holdout_ok and negative_ok:
        record.decision = "ACCEPT"
        record.reason = "paired improvement confirmed on independent task"
    elif dec_b.accepted and (not holdout_ok or not negative_ok):
        record.decision = "INCONCLUSIVE"
        record.reason = ("paired improvement did not survive holdout / "
                         "negative control")
    else:
        record.decision = "REJECT" if any(
            d < 0 and abs(d) > 0.01 for d in deltas) else "INCONCLUSIVE"
        record.reason = gate.explain(dec_b)
    record.ledger_fingerprint = candidate.hyp_fingerprint
    record.evidence = {
        "A_base": A_base, "A_cand": A_cand,
        "regression_rate": regression_rate,
        "mean_delta": round(sum(deltas) / max(1, len(deltas)), 6),
        "missing_runs": missing, "timeout_runs": timeouts,
        "gate_trace_b": dec_b.gate_trace,
        "base_result_paths": [r["provenance"].get("result_path")
                              for r in base_results],
        "cand_result_paths": [r["provenance"].get("result_path")
                              for r in cand_results],
    }
    after = evaluation_substrate_snapshot(repo_root)
    record.substrate_unchanged = (before == after)
    if not record.substrate_unchanged:
        record.decision = "REJECT"
        record.reason = "evaluation substrate changed during scoring"
    return record


def evolve(*, base_harness: Any, tasks: list[Any], heldout_task: Any,
           run_root: str | Path, repo_root: str | Path,
           rounds: int = 2, max_minutes: float = 30.0,
           negative_harness: Any | None = None,
           include_degenerate_control: bool = True,
           cfg: Settings | None = None,
           operator: str = "") -> dict[str, Any]:
    """Full F loop. NEVER promotes automatically (operator-gated)."""
    cfg = cfg or Settings()
    run_root = Path(run_root).resolve()
    repo_root = Path(repo_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(run_root / "harness_ledger.jsonl")
    experience_path = run_root / "harness_experience.jsonl"
    experiences = ExperienceStore(experience_path)
    started = time.time()
    substrate_before = evaluation_substrate_snapshot(repo_root)

    # 1) Collect REAL TaskResults with the baseline Harness.
    baseline_results = [
        _run_arm(t, base_harness, run_root=run_root / "collect",
                 experience_path=experience_path) for t in tasks]
    # 2) Diagnose the RESEARCH PROCESS (not the model).
    analysis = analyze_many(baseline_results)
    dominant = analysis["dominant"]
    records: list[EvolutionRecord] = []
    accepted: list[EvolutionRecord] = []
    no_gain_streak = 0

    for round_no in range(1, rounds + 1):
        if (time.time() - started) / 60.0 > max_minutes:
            break
        candidates = generate_harness_candidates(
            dominant=dominant, base_harness=base_harness, ledger=ledger,
            limit=4)
        if not candidates:
            break
        # Append a DEGENERATE control in the first round. It is evaluated by
        # the exact same paired protocol and must produce a real REJECT, so the
        # artifact proves the loop is not rigged to accept. It is deliberately
        # NOT counted toward the no-gain stop (it is not an improvement
        # attempt), and is never confused with a genuine candidate.
        if round_no == 1 and include_degenerate_control:
            candidates = list(candidates) + [
                degenerate_control_candidate(base_harness=base_harness,
                                             dominant=dominant)]
        for candidate in candidates:
            if (time.time() - started) / 60.0 > max_minutes:
                break
            is_control = candidate.taxonomy == "NEGATIVE_CONTROL"
            cand_harness, patch_report = apply_harness_patch(
                base_harness, candidate,
                sandbox_root=run_root / f"sandboxes_r{round_no}",
                repo_root=repo_root)
            rec = evaluate_candidate(
                base_harness=base_harness, cand_harness=cand_harness,
                candidate=candidate, tasks=tasks, heldout_task=heldout_task,
                run_root=run_root / f"r{round_no}",
                experience_path=experience_path, repo_root=repo_root,
                cfg=cfg, negative_harness=negative_harness)
            rec.round_no = round_no
            rec.evidence["patch"] = patch_report
            rec.evidence["is_negative_control"] = is_control
            records.append(rec)
            status = {"ACCEPT": "accepted", "REJECT": "rejected",
                      "INCONCLUSIVE": "inconclusive"}[rec.decision]
            ledger.update(rec.ledger_fingerprint, status,
                          evidence={"decision_a": rec.decision_a,
                                    "decision_b": rec.decision_b,
                                    "mean_delta": rec.evidence["mean_delta"],
                                    "holdout": rec.holdout})
            experiences.create(
                problem=f"RESEARCH_HARNESS|{dominant}",
                claim=f"{candidate.target} {candidate.old}->{candidate.new}: "
                      f"{rec.decision}",
                component="research_harness",
                edit={"patch_hash": candidate.hyp_fingerprint},
                n_pairs=len(rec.task_deltas), n_target=len(rec.task_deltas),
                p_value=1.0, regression_rate=rec.evidence["regression_rate"],
                dQ=rec.evidence["mean_delta"], dC=0.0,
                positive=[t.task_id for t in tasks] if rec.decision == "ACCEPT" else [],
                negative=[t.task_id for t in tasks] if rec.decision == "REJECT" else [],
                judge_version="arr-1.0-quality-v1",
                hypothesis_id=candidate.hyp_fingerprint,
            )
            if rec.decision == "ACCEPT":
                accepted.append(rec)
                no_gain_streak = 0
            elif not is_control:
                no_gain_streak += 1
            # F4: stop on sustained no-gain, but allow the full candidate
            # shortlist in the first round -- a single-knob search must be
            # allowed to try more than one mechanism before giving up.
            if no_gain_streak >= 4:
                break
        if no_gain_streak >= 4:
            break

    substrate_after = evaluation_substrate_snapshot(repo_root)
    payload: dict[str, Any] = {
        "schema_version": "arr-1.0-harness-evolution-v1",
        "candidate_type": CandidateType.RESEARCH_HARNESS.value,
        "dominant_failure": dominant,
        "analysis": analysis,
        "base_harness_sha256": base_harness.fingerprint(),
        "base_harness": base_harness.to_dict(),
        "rounds_requested": rounds,
        "rounds_executed": max([r.round_no for r in records], default=0),
        "max_minutes": max_minutes,
        "elapsed_seconds": round(time.time() - started, 3),
        "records": [r.to_dict() for r in records],
        "accepted": [r.to_dict() for r in accepted],
        "n_accepted": len(accepted),
        "n_rejected": sum(1 for r in records if r.decision == "REJECT"),
        "n_inconclusive": sum(1 for r in records if r.decision == "INCONCLUSIVE"),
        "n_negative_controls": sum(
            1 for r in records if r.evidence.get("is_negative_control")),
        "negative_controls_rejected": sum(
            1 for r in records if r.evidence.get("is_negative_control")
            and r.decision == "REJECT"),
        "evaluation_substrate_unchanged": substrate_before == substrate_after,
        "evaluation_substrate_sha256": substrate_after,
        "promotion": {
            "status": "not_requested",
            "operator": operator or "",
            "note": "No version.promote() is called by the loop. Applying an "
                    "accepted Harness is a separate, human-gated step.",
        },
        "rollback_available": True,
        "live_runtime_untouched": substrate_before == substrate_after,
    }
    blob = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                      default=str).encode("utf-8")
    payload["artifact_sha256"] = hashlib.sha256(blob).hexdigest()
    name = (f"harness_evolution_{dominant}_"
            f"{payload['artifact_sha256'][:12]}.json")
    out = run_root / name
    out.write_bytes(blob)
    payload["artifact_path"] = str(out)
    return payload


def promote_harness(*, artifact_path: str | Path, operator: str,
                    note: str = "") -> dict[str, Any]:
    """EXPLICIT human-gated promotion. Raises for system/auto operators."""
    from . import version
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if payload.get("n_accepted", 0) < 1:
        raise ValueError("no ACCEPTED Harness candidate; refusing promotion")
    if not payload.get("evaluation_substrate_unchanged"):
        raise ValueError("evaluation substrate changed; refusing promotion")
    before = version.current() or version.ensure_initial()
    new = version.promote(
        evidence=[{"artifact": str(artifact_path),
                   "sha256": payload.get("artifact_sha256"),
                   "accepted": payload["accepted"]}],
        operator=operator, note=note or "ARR 1.0 Harness evolution",
        require_operator=True)
    return {"promoted": True, "from": before.version, "to": new.version,
            "operator": operator, "base_commit": new.base_commit,
            "rollback_hint": "version.restore(prev, operator=<human>)"}
