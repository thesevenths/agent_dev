"""M4.3-02 -- Pilot analysis: descriptive, paired, and honest about limits.

Reads ONLY persisted data (the ExperimentRegistry tables, the append-only
review rows and the controller's event log) and recomputes everything from
scratch -- the analysis is reproducible from the database alone.

Deliberately NOT done here
--------------------------
  * NO significance claim: a pilot with a handful of blocks cannot support one;
  * NO deletion of failed / interrupted / invalid runs: they stay in every
    denominator and are reported explicitly;
  * NO imputation of a missing human review: if the objective-quality reviews
    are still pending, that metric is reported as MISSING and the paired
    contrast on it is not computed;
  * NO effect-size assumption taken from the observed pilot maximum.
"""
from __future__ import annotations

import datetime
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean, variance
from typing import Any

from .experiment_registry import ARMS, RunRecord
from .metrics import (
    ALL_METRICS, METRIC_DEFINITIONS, PRIMARY_METRICS, compute_metric_set,
    frozen_metric_contract,
)
from .paired_eligibility import select_paired_blocks
from .registry_store import RegistryStore
from .sample_size import ContrastSpec, estimate_sample_size, summarise

#: Primary contrasts are FIXED before the analysis (spec section 7).
PRIMARY_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("B-A", "B", "A"),
    ("C-B", "C", "B"),
    ("C-A", "C", "A"),
)

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261116


# --------------------------------------------------------------------- loading
def load_runs(db: Any) -> list[RunRecord]:
    store = RegistryStore(db=db)
    known = set(RunRecord.__dataclass_fields__)
    return [RunRecord(**{k: v for k, v in d.items() if k in known})
            for d in store.load_runs()]


def _load_blocks(db: Any) -> list[dict[str, Any]]:
    return RegistryStore(db=db).load_blocks()


def _human_reviews(db: Any) -> list[dict[str, Any]]:
    rows = db.rows("SELECT record_json FROM m4_3_reviews ORDER BY seq")
    out = []
    for r in rows:
        try:
            rec = json.loads(r["record_json"])
        except Exception:  # noqa: BLE001
            continue
        if rec.get("review_kind") == "human_blind":
            out.append(rec)
    return out


def _automated_scores(db: Any) -> list[dict[str, Any]]:
    rows = db.rows("SELECT record_json FROM m4_3_reviews ORDER BY seq")
    out = []
    for r in rows:
        try:
            rec = json.loads(r["record_json"])
        except Exception:  # noqa: BLE001
            continue
        if rec.get("review_kind") == "automated":
            out.append(rec)
    return out


# ------------------------------------------------------------------- analysis
def _bootstrap_ci(values: list[float], *, resamples: int = BOOTSTRAP_RESAMPLES,
                  seed: int = BOOTSTRAP_SEED,
                  alpha: float = 0.05) -> list[float] | None:
    """Percentile bootstrap CI of the MEAN of paired differences."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(math.floor((alpha / 2) * (len(means) - 1)))]
    hi = means[int(math.ceil((1 - alpha / 2) * (len(means) - 1)))]
    return [round(lo, 6), round(hi, 6)]


def _per_run_primary(run: RunRecord) -> dict[str, float]:
    return {name: compute_metric_set([run])[name] for name in PRIMARY_METRICS}


def analyze_pilot(*, db: Any, out_dir: str | Path,
                  experiment_id: str = "m4_3_pilot",
                  alpha: float = 0.05, target_power: float = 0.8,
                  seoi_quality: float = 0.5,
                  seoi_valid_findings: float = 0.01,
                  bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
                  protocol: dict[str, Any] | None = None,
                  write: bool = True) -> dict[str, Any]:
    """Recompute the full pilot analysis from the persisted registry."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(db)
    blocks = _load_blocks(db)
    store = RegistryStore(db=db)
    integrity = store.verify_integrity()
    human = _human_reviews(db)
    automated = _automated_scores(db)

    by_arm: dict[str, list[RunRecord]] = {a: [r for r in runs if r.arm == a]
                                          for a in ARMS}
    metrics_by_arm = {a: {"n_runs": len(v), **compute_metric_set(v)}
                      for a, v in by_arm.items()}
    overall = {"n_runs": len(runs), **compute_metric_set(runs)}

    # ---------------------------------------------------------- block level
    # P0-01: ONE shared eligibility rule with the value report. ``paired`` is
    # the single source of truth for which blocks may enter a paired contrast.
    expected_protocol_sha256 = str(
        (protocol or {}).get("protocol_sha256")
        or (blocks[0].get("protocol_sha256") if blocks else "")
        or "")
    paired = select_paired_blocks(
        runs=runs, blocks=blocks,
        expected_protocol_sha256=expected_protocol_sha256)
    eligible_ids = {b["block_id"] for b in paired["eligible"]}
    excluded_reasons = {e["block_id"]: e["reasons"] for e in paired["excluded"]}

    runs_by_block: dict[str, dict[str, RunRecord]] = paired["runs_by_block"]
    block_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for block in blocks:
        bid = block["block_id"]
        arms = runs_by_block.get(bid, {})
        row: dict[str, Any] = {"block_id": bid,
                               "snapshot_id": block["snapshot_id"],
                               "seed": block["seed"],
                               "snapshot_sha256": block.get("snapshot_sha256", ""),
                               "arms": {}}
        for arm in ARMS:
            rec = arms.get(arm)
            if rec is None:
                continue
            quality = rec.quality_review.get("total") if rec.quality_review else None
            row["arms"][arm] = {
                "run_id": rec.run_id, "outcome": rec.outcome,
                "finalized": rec.finalized,
                "objective_id": rec.objective_id,
                "campaign_id": rec.campaign_id,
                "plan_id": rec.plan_id, "report_id": rec.report_id,
                "experience_id": rec.experience_id,
                "gate_steps_passed": rec.gate_steps_passed,
                "gate_passed": bool(rec.gate_evidence.get("gate_passed")),
                "external_approval": rec.external_approval,
                "approval_bypass": rec.approval_bypass,
                "compute_units": rec.compute_units,
                "wall_time_s": rec.wall_time_s,
                "objective_quality": quality,
                "has_human_review": bool(rec.independent_reviews),
                "paired_seeds": list(rec.paired_seeds),
                "primary": _per_run_primary(rec),
                "notes": rec.notes,
            }
        complete = len(row["arms"]) == len(ARMS)
        finalized = bid in eligible_ids
        row["complete"] = complete
        row["finalized"] = finalized
        row["exclusion_reasons"] = excluded_reasons.get(bid, [])
        block_rows.append(row)
        if not finalized:
            excluded.append({"block_id": bid,
                             "reasons": excluded_reasons.get(bid)
                             or ["not all arms finalized"]})

    eligible = [b for b in block_rows if b["finalized"]]

    # ------------------------------------------------------- completeness
    def _rate(pred) -> float:
        return round(sum(1 for r in runs if pred(r)) / len(runs), 6) if runs else 0.0

    completeness = {
        "n_runs": len(runs),
        "gate_pass_rate": _rate(lambda r: r.gate_evidence.get("gate_passed") is True),
        "external_approval_rate": _rate(lambda r: r.external_approval),
        "approval_bypass_count": sum(1 for r in runs if r.approval_bypass),
        "report_rate": _rate(lambda r: bool(r.report_id)),
        "experience_rate": _rate(lambda r: bool(r.experience_id)),
        "campaign_rate": _rate(lambda r: bool(r.campaign_id)),
        "plan_rate": _rate(lambda r: bool(r.plan_id)),
        "finalized_rate": _rate(lambda r: r.finalized),
        "failure_rate": _rate(lambda r: r.outcome == "failed"),
        "interruption_rate": _rate(lambda r: r.outcome == "interrupted"),
        "invalid_rate": _rate(lambda r: r.outcome == "invalid"),
        "inconclusive_rate": _rate(lambda r: r.outcome == "inconclusive"),
        "valid_finding_rate": _rate(lambda r: r.outcome == "valid_finding"),
        "negative_finding_rate": _rate(lambda r: r.outcome == "negative_finding"),
        "outcome_counts": {o: sum(1 for r in runs if r.outcome == o)
                           for o in ("valid_finding", "negative_finding",
                                     "inconclusive", "invalid", "failed",
                                     "interrupted")},
        "per_arm": {a: {
            "n_runs": len(v),
            "gate_pass_rate": (round(sum(1 for r in v
                                         if r.gate_evidence.get("gate_passed"))
                                     / len(v), 6) if v else 0.0),
            "valid_findings": sum(1 for r in v if r.outcome == "valid_finding"),
            "failures": sum(1 for r in v if r.outcome in ("failed", "invalid")),
            "interruptions": sum(1 for r in v if r.outcome == "interrupted"),
        } for a, v in by_arm.items()},
    }

    # --------------------------------------------------- objective quality
    quality: dict[str, Any] = {}
    for arm in ARMS:
        totals = [r.quality_review["total"] for r in by_arm[arm]
                  if r.quality_review]
        quality[arm] = {**summarise(totals),
                        "distribution": sorted(totals),
                        "n_with_review": len(totals)}
    quality["n_human_blind_reviews"] = len(human)
    quality["n_automated_scores"] = len(automated)
    quality["review_status"] = (
        "complete" if human and all(q[arm]["n_with_review"] == len(by_arm[arm])
                                    for arm in ARMS) else "pending_or_partial")
    quality["note"] = (
        "objective quality is a HUMAN-BLIND metric; automated scores are stored "
        "separately and never enter it")

    # ----------------------------------------------------- paired contrasts
    contrasts: dict[str, Any] = {}
    diffs_by_contrast: dict[str, list[float]] = {}
    for label, hi, lo in PRIMARY_CONTRASTS:
        per_metric: dict[str, Any] = {}
        for metric in PRIMARY_METRICS:
            diffs: list[float] = []
            for block in eligible:
                a_hi = block["arms"].get(hi, {}).get("primary", {}).get(metric)
                a_lo = block["arms"].get(lo, {}).get("primary", {}).get(metric)
                if a_hi is None or a_lo is None:
                    continue
                if metric == "independently_reviewed_objective_quality" and \
                        block["arms"][hi].get("objective_quality") is None:
                    continue      # missing human review is never imputed
                diffs.append(round(float(a_hi) - float(a_lo), 6))
            stats = summarise(diffs)
            per_metric[metric] = {
                **stats,
                "differences": diffs,
                "bootstrap_ci95": _bootstrap_ci(
                    diffs, resamples=bootstrap_resamples),
                "direction": f"{hi} minus {lo}",
            }
            if metric == PRIMARY_METRICS[0]:
                diffs_by_contrast[label] = diffs
        contrasts[label] = {"arms": f"{hi} vs {lo}", "metrics": per_metric}

    # ------------------------------------------------------------ economics
    economics = {
        "compute_units_total": round(sum(r.compute_units for r in runs), 6),
        "compute_units_per_arm": {a: round(sum(r.compute_units for r in v), 6)
                                  for a, v in by_arm.items()},
        "wall_time_s_total": round(sum(r.wall_time_s for r in runs), 3),
        "wall_time_s_per_arm": {a: round(sum(r.wall_time_s for r in v), 3)
                                for a, v in by_arm.items()},
        "wall_time_s_mean_per_run": (round(mean([r.wall_time_s for r in runs]), 3)
                                     if runs else None),
        "compute_per_valid_finding": overall.get("compute_per_valid_finding"),
        "n_valid_findings": sum(1 for r in runs if r.outcome == "valid_finding"),
    }

    # -------------------------------------------------- data integrity
    campaign_owners: dict[str, set[str]] = {}
    for r in runs:
        if r.campaign_id:
            campaign_owners.setdefault(r.campaign_id, set()).add(r.run_id)
    duplicates = sorted(cid for cid, owners in campaign_owners.items()
                        if len(owners) > 1)
    data_integrity = {
        "registry_event_chain_valid": integrity,
        "duplicate_campaign_ids": duplicates,
        "n_blocks_registered": len(blocks),
        "n_blocks_eligible": len(eligible),
        "n_blocks_excluded": len(excluded),
        "excluded_blocks": excluded,
        "paired_eligibility_rules": paired["rules"],
        "protocol_sha256": (blocks[0].get("protocol_sha256") if blocks else ""),
        "protocol_shas": sorted({b.get("protocol_sha256", "") for b in blocks}),
    }

    # ----------------------------------------------------- sample size
    per_campaign = (mean([r.wall_time_s for r in runs if r.wall_time_s > 0])
                    if any(r.wall_time_s > 0 for r in runs) else None)
    sample = estimate_sample_size(
        paired_differences=diffs_by_contrast,
        contrasts=[ContrastSpec("B-A", seoi_quality),
                   ContrastSpec("C-B", seoi_quality),
                   ContrastSpec("C-A", seoi_quality)],
        alpha=alpha, target_power=target_power,
        cost_per_campaign_s=per_campaign)

    analysis = {
        "schema": "m4_3_pilot_analysis_v1",
        "stage": "pilot",
        "experiment_id": experiment_id,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "protocol": dict(protocol or {}),
        "metric_contract": frozen_metric_contract(),
        "metric_definitions": {k: asdict(v)
                               for k, v in METRIC_DEFINITIONS.items()},
        "n_blocks": len(blocks),
        "n_runs": len(runs),
        "metrics_by_arm": metrics_by_arm,
        "metrics_overall": overall,
        "completeness": completeness,
        "objective_quality": quality,
        "blocks": block_rows,
        "paired_contrasts": contrasts,
        "hierarchy": ["snapshot", "block", "seed"],
        "economics": economics,
        "data_integrity": data_integrity,
        "sample_size": sample,
        "significance_conclusion": None,
        "limitations": _limitations(quality, len(eligible)),
        "conclusion": _conclusion(quality, completeness, len(eligible)),
    }
    if write:
        (out / "pilot_analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True, default=str),
            encoding="utf-8")
        (out / "sample_size_recommendation.json").write_text(
            json.dumps(sample, indent=2, sort_keys=True, default=str),
            encoding="utf-8")
        (out / "pilot_report.md").write_text(
            render_pilot_report(analysis), encoding="utf-8")
    return analysis


def _limitations(quality: dict[str, Any], n_blocks: int) -> list[str]:
    out = [f"only {n_blocks} eligible block(s): descriptive, not inferential"]
    if quality["review_status"] != "complete":
        out.append(
            "objective-quality reviews are not complete: at least two "
            "independent HUMAN reviewers must submit before that metric (and "
            "any contrast on it) can be reported")
    out.append(
        "A/B objectives were selected by the recorded, pre-registered "
        "delegated-operator protocol, not by independent human researchers; "
        "this weakens any claim about human-vs-automation selection")
    return out


def _conclusion(quality: dict[str, Any], completeness: dict[str, Any],
                n_blocks: int) -> dict[str, Any]:
    """One of the four allowed outcomes -- never a forced win for C."""
    if n_blocks < 3:
        verdict = "evidence_insufficient"
        reason = (f"{n_blocks} eligible block(s) cannot support any value "
                  "conclusion")
    elif quality["review_status"] == "complete":
        verdict = "inconclusive_pilot"
        reason = ("the pilot produced a complete paired dataset; the "
                  "confirmatory analysis (not the pilot) decides the value "
                  "conclusion")
    else:
        verdict = "evidence_insufficient"
        reason = ("the primary objective-quality metric has no complete human "
                  "blind review, so no value claim can be made")
    return {
        "verdict": verdict,
        "allowed_outcomes": [
            "ARR has clear incremental value",
            "efficiency improved without a scientific discovery improvement",
            "no clear incremental value",
            "evidence insufficient"],
        "reason": reason,
        "significance_claimed": False,
        "note": ("a pilot NEVER declares a winner; all four outcomes remain "
                 "open until the frozen confirmatory analysis"),
    }


# ------------------------------------------------------------------ render
def render_pilot_report(a: dict[str, Any]) -> str:
    """Markdown pilot report (tables generated from the analysis dict)."""
    lines: list[str] = []
    lines.append(f"# M4.3 Pilot Report -- {a['experiment_id']}")
    lines.append("")
    lines.append(f"Generated: {a['generated_at']}  ")
    lines.append(f"Protocol SHA-256: `{a['data_integrity']['protocol_sha256']}`  ")
    lines.append(f"Metric contract: `{a['metric_contract']['contract_sha256']}`  ")
    lines.append(f"Blocks: **{a['n_blocks']}** (eligible for pairing: "
                 f"{a['data_integrity']['n_blocks_eligible']}) / Runs: "
                 f"**{a['n_runs']}**")
    lines.append("")
    lines.append("> This is a PILOT report. It computes descriptive statistics "
                 "and does not claim significance. The confirmatory value "
                 "conclusion belongs to `M4_3_VALUE_REPORT.md`.")
    lines.append("")

    lines.append("## 1. Completeness and real-environment evidence")
    lines.append("")
    c = a["completeness"]
    lines.append("| check | value |")
    lines.append("|---|---|")
    for key in ("gate_pass_rate", "external_approval_rate", "report_rate",
                "experience_rate", "campaign_rate", "finalized_rate",
                "failure_rate", "interruption_rate", "invalid_rate",
                "inconclusive_rate"):
        lines.append(f"| {key} | {c[key]} |")
    lines.append(f"| approval_bypass_count | {c['approval_bypass_count']} |")
    lines.append(f"| outcome_counts | {c['outcome_counts']} |")
    lines.append("")

    lines.append("## 2. All 16 metrics per arm")
    lines.append("")
    header = "| metric | " + " | ".join(ARMS) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(ARMS))
    for metric in ALL_METRICS:
        cells = []
        for arm in ARMS:
            value = a["metrics_by_arm"][arm].get(metric)
            cells.append("n/a" if value is None else str(value))
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Runs per arm: " + ", ".join(
        f"{arm}={a['metrics_by_arm'][arm]['n_runs']}" for arm in ARMS))
    lines.append("")

    lines.append("## 3. Paired contrasts (block-level, descriptive)")
    lines.append("")
    for label, payload in a["paired_contrasts"].items():
        lines.append(f"### {label} ({payload['arms']})")
        lines.append("")
        lines.append("| metric | n | mean diff | sd | variance | bootstrap 95% CI |")
        lines.append("|---|---|---|---|---|---|")
        for metric, m in payload["metrics"].items():
            ci = m["bootstrap_ci95"]
            lines.append(
                f"| {metric} | {m['n']} | {m['mean']} | {m['sd']} | "
                f"{m['variance']} | {ci if ci else 'n/a (n<2)'} |")
        lines.append("")

    lines.append("## 4. Objective quality (human blind review)")
    lines.append("")
    q = a["objective_quality"]
    lines.append(f"Review status: **{q['review_status']}** -- human blind "
                 f"reviews recorded: {q['n_human_blind_reviews']}, automated "
                 f"scores (kept separate): {q['n_automated_scores']}")
    lines.append("")
    lines.append("| arm | n reviewed | mean | sd | distribution |")
    lines.append("|---|---|---|---|---|")
    for arm in ARMS:
        lines.append(f"| {arm} | {q[arm]['n_with_review']} | {q[arm]['mean']} | "
                     f"{q[arm]['sd']} | {q[arm]['distribution']} |")
    lines.append("")

    lines.append("## 5. Blocks")
    lines.append("")
    lines.append("| block | snapshot | seed | A | B | C | gate ok |")
    lines.append("|---|---|---|---|---|---|---|")
    for b in a["blocks"]:
        cells = []
        for arm in ARMS:
            info = b["arms"].get(arm)
            cells.append(info["outcome"] if info else "MISSING")
        gate = all(b["arms"].get(arm, {}).get("gate_passed") for arm in ARMS
                   if b["arms"].get(arm))
        lines.append(f"| {b['block_id']} | {b['snapshot_id']} | {b['seed']} | "
                     + " | ".join(cells) + f" | {gate} |")
    lines.append("")
    if a["data_integrity"]["excluded_blocks"]:
        lines.append("Excluded blocks: " +
                     json.dumps(a["data_integrity"]["excluded_blocks"]))

    lines.append("")
    lines.append("## 6. Economics")
    lines.append("")
    e = a["economics"]
    lines.append(f"- compute units: {e['compute_units_total']} "
                 f"({e['compute_units_per_arm']})")
    lines.append(f"- wall time (s): {e['wall_time_s_total']} "
                 f"({e['wall_time_s_per_arm']})")
    lines.append(f"- valid findings: {e['n_valid_findings']}")
    lines.append("")

    lines.append("## 7. Sample-size recommendation (confirmatory planning)")
    lines.append("")
    s = a["sample_size"]
    lines.append(f"- status: **{s['status']}**")
    lines.append(f"- blocks recommended: {s['n_blocks_recommended']}")
    lines.append(f"- alpha per contrast: {s['alpha_per_contrast']} "
                 f"(power {s['target_power']}, inflation {s['variance_inflation']})")
    for name, v in s["per_contrast"].items():
        lines.append(f"- {name}: {v['n_blocks']} blocks "
                     f"(variance {v['variance_used']})")
    if s["contrasts_without_variance"]:
        lines.append(f"- NOT estimated (insufficient pilot data): "
                     f"{s['contrasts_without_variance']}")
    if s["budget_estimate"]:
        lines.append(f"- budget: {s['budget_estimate']}")
    lines.append("")

    lines.append("## 8. Limitations")
    lines.append("")
    for item in a["limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 9. Conclusion")
    lines.append("")
    con = a["conclusion"]
    lines.append(f"**{con['verdict']}** -- {con['reason']}")
    lines.append("")
    lines.append(f"Significance claimed: **{con['significance_claimed']}**")
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "PRIMARY_CONTRASTS", "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED", "load_runs",
    "analyze_pilot", "render_pilot_report",
]
