"""M4.3-02 -- M4.3-04 Value Report generator (pilot AND final).

Every number is read back from the persisted registry (blocks, runs, trusted
ids), the append-only review rows and, for the confirmatory stage, the frozen
protocol. Nothing is copied from a previous report and nothing is hardcoded.

The report has SEVEN sections (spec M4.3-04):
  A design   B data integrity   C primary metrics   D secondary metrics
  E safety   F economics        G conclusion

Each key claim carries an evidence chain (block -> run -> campaign -> report ->
experience -> evidence refs) so a reader can audit it.

A pilot report and the final value report are DIFFERENT artefacts and are
labelled as such: the pilot never declares a winner, and the final report is
only allowed to pick one of four outcomes.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from .experiment_registry import (
    ARMS, ARM_DESCRIPTIONS, RUN_OUTCOMES, RunRecord,
)
from .metrics import (
    ALL_METRICS, PRIMARY_METRICS, SAFETY_METRICS, SECONDARY_METRICS,
    compute_metric_set, frozen_metric_contract,
)
from .paired_eligibility import select_paired_blocks
from .pilot_analysis import PRIMARY_CONTRASTS, load_runs
from .registry_store import RegistryStore
from .statistical_analysis import analyse_primary_contrasts

OUTCOME_KEYS: dict[str, str] = {
    "ARR has clear incremental value": "arr_has_clear_incremental_value",
    "efficiency improved without a scientific discovery improvement":
        "efficiency_only",
    "no clear incremental value": "no_clear_incremental_value",
    "evidence insufficient": "evidence_insufficient",
}

VALUE_OUTCOMES: tuple[str, ...] = (
    "ARR has clear incremental value",
    "efficiency improved without a scientific discovery improvement",
    "no clear incremental value",
    "evidence insufficient",
)

#: Which contrasts are PRIMARY vs secondary (fixed BEFORE results are seen).
PRIMARY_CONTRAST_LABELS: tuple[str, ...] = ("C-B",)
SECONDARY_CONTRAST_LABELS: tuple[str, ...] = ("B-A", "C-A")
#: The COMPLETE pre-registered correction family. It is FIXED and passed to the
#: statistician so a missing contrast can never shrink the correction (P1-02).
TEST_FAMILY_LABELS: tuple[str, ...] = (
    PRIMARY_CONTRAST_LABELS + SECONDARY_CONTRAST_LABELS)
#: The pre-registered PRIMARY metric that gates a value conclusion. The other
#: primary metric is reported but a secondary metric/contrast can NEVER make
#: ``significance_claimed`` true (P1-02).
DECISION_PRIMARY_METRIC: str = PRIMARY_METRICS[0]   # independently-reviewed quality


def _runs_by_arm(runs: list[RunRecord]) -> dict[str, list[RunRecord]]:
    return {a: [r for r in runs if r.arm == a] for a in ARMS}


def _evidence_chain(rec: RunRecord) -> dict[str, Any]:
    """The traceable chain behind ONE run (no interpretation)."""
    return {"run_id": rec.run_id, "block_id": rec.block_id,
            "arm": rec.arm, "objective_id": rec.objective_id,
            "plan_id": rec.plan_id, "approval_id": rec.approval_id,
            "campaign_id": rec.campaign_id, "report_id": rec.report_id,
            "experience_id": rec.experience_id,
            "snapshot_id": rec.snapshot_id, "seed": rec.seed,
            "outcome": rec.outcome,
            "gate_passed": bool(rec.gate_evidence.get("gate_passed")),
            "gate_steps_passed": rec.gate_steps_passed,
            "external_approval": rec.external_approval,
            "approval_source": str(
                (rec.gate_evidence or {}).get("approval_source") or ""),
            "approval_approved_by": str(
                (rec.gate_evidence or {}).get("approval_approved_by") or ""),
            "evidence_refs": list(rec.evidence_refs),
            "finalized": rec.finalized}


def _contrast_differences(
        runs: list[RunRecord], metric: str,
        *, blocks: list[dict[str, Any]] | None = None,
        expected_protocol_sha256: str = "",
) -> dict[str, list[float]]:
    """Per-block paired differences using the SHARED eligibility rule (P0-01).

    Pilot Analysis and the Value Report both call
    :func:`select_paired_blocks`, so their paired ``n`` can never diverge.
    Failed / interrupted / invalid runs remain in the overall statistics; they
    simply do not contribute a paired difference.
    """
    selection = select_paired_blocks(
        runs=runs, blocks=blocks,
        expected_protocol_sha256=expected_protocol_sha256)
    out: dict[str, list[float]] = {label: [] for label, _, _ in PRIMARY_CONTRASTS}
    for block in selection["eligible"]:
        arms: dict[str, RunRecord] = block["arms"]
        for label, hi, lo in PRIMARY_CONTRASTS:
            a = compute_metric_set([arms[hi]])[metric]
            b = compute_metric_set([arms[lo]])[metric]
            if a is None or b is None:
                continue
            if metric == PRIMARY_METRICS[0] and (
                    not arms[hi].quality_review or not arms[lo].quality_review):
                continue        # never impute a missing human review
            out[label].append(round(float(a) - float(b), 6))
    return out


def _quality_observation(by_arm: dict[str, list[RunRecord]]) -> dict[str, Any]:
    """Report objective quality as MISSING (not 0) when no human review exists.

    The frozen metric contract uses 0.0 as its missing-value sentinel, but a
    value report must never render that sentinel as if a real score of zero had
    been observed. This block makes the distinction explicit and auditable.
    """
    out: dict[str, Any] = {}
    total_reviewed = 0
    for arm in ARMS:
        reviewed = [r for r in by_arm.get(arm, []) if r.quality_review]
        total_reviewed += len(reviewed)
        totals = [float(r.quality_review["total"]) for r in reviewed
                  if r.quality_review.get("total") is not None]
        out[arm] = {
            "n_runs": len(by_arm.get(arm, [])),
            "n_reviewed": len(reviewed),
            "value": (round(sum(totals) / len(totals), 6) if totals else None),
            "status": "observed" if totals else "missing_no_human_review",
        }
    out["n_reviewed_total"] = total_reviewed
    out["observation_complete"] = all(
        out[arm]["n_reviewed"] == out[arm]["n_runs"] and out[arm]["n_runs"] > 0
        for arm in ARMS)
    out["note"] = (
        "objective quality is a HUMAN-BLIND metric; with no human review it is "
        "reported as a missing value (None), not an observed score of 0")
    return out


def generate_value_report(*, db: Any, out_dir: str | Path,
                          experiment_id: str = "m4_3",
                          stage: str = "final",
                          alpha: float = 0.05,
                          correction: str = "holm",
                          protocol: dict[str, Any] | None = None,
                          pilot_analysis: dict[str, Any] | None = None,
                          write: bool = True) -> dict[str, Any]:
    """Build the value report for ``stage`` ('pilot' or 'final')."""
    if stage not in ("pilot", "final"):
        raise ValueError("stage must be 'pilot' or 'final'")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(db)
    blocks = RegistryStore(db=db).load_blocks()
    by_arm = _runs_by_arm(runs)
    integrity = RegistryStore(db=db).verify_integrity()
    protocol = protocol or {}
    expected_protocol_sha256 = str(
        protocol.get("protocol_sha256")
        or (blocks[0].get("protocol_sha256") if blocks else "")
        or "")
    # Shared paired-eligibility selection (P0-01): the SAME rule the pilot
    # analysis uses. Every excluded block carries explicit reasons.
    paired = select_paired_blocks(
        runs=runs, blocks=blocks,
        expected_protocol_sha256=expected_protocol_sha256)

    # ------------------------------------------------------------- A design
    design = {
        "experiment_id": experiment_id,
        "stage": stage,
        "n_blocks": len(blocks),
        "snapshot_ids": sorted({b["snapshot_id"] for b in blocks}),
        "protocol_sha256": protocol.get("protocol_sha256")
        or (blocks[0].get("protocol_sha256") if blocks else ""),
        "model_version": protocol.get("model_version", ""),
        "prompt_version": protocol.get("prompt_version", ""),
        "ranker_version": protocol.get("ranker_version", ""),
        "domain_policy_version": protocol.get("domain_policy_version", ""),
        "paired_seeds": protocol.get("paired_seeds", []),
        "campaign_budget": protocol.get("campaign_budget", {}),
        "exclusion_rules": protocol.get("exclusion_rules", []),
        "statistical_method": protocol.get("statistical_method", ""),
        "multiple_comparison_correction": protocol.get(
            "multiple_comparison_correction", correction),
        "primary_contrasts": list(PRIMARY_CONTRAST_LABELS),
        "secondary_contrasts": list(SECONDARY_CONTRAST_LABELS),
        # P1-02: the fixed, pre-registered family and the Primary/Secondary
        # hierarchy are declared in the design BEFORE any result is inspected.
        "test_family": list(TEST_FAMILY_LABELS),
        "correction_family_size": len(TEST_FAMILY_LABELS),
        "testing_hierarchy": {
            "primary_metrics": list(PRIMARY_METRICS),
            "secondary_metrics": list(SECONDARY_METRICS),
            "primary_contrasts": list(PRIMARY_CONTRAST_LABELS),
            "secondary_contrasts": list(SECONDARY_CONTRAST_LABELS),
            "decision_primary_metric": DECISION_PRIMARY_METRIC,
            "decision_primary_contrast": PRIMARY_CONTRAST_LABELS[0],
            "rule": ("only a PRIMARY contrast on a PRIMARY metric may drive "
                     "significance_claimed or a value conclusion; missing "
                     "contrasts stay in the correction family"),
        },
        "arms": {a: {"n_runs": len(v)} for a, v in by_arm.items()},
        # P1-03: accurate, pre-registered group descriptions. A/B are delegated
        # rule-based baselines, NOT independent human selection.
        "arm_descriptions": dict(ARM_DESCRIPTIONS),
        "metric_contract_sha256": frozen_metric_contract()["contract_sha256"],
    }

    # ----------------------------------------------------- B data integrity
    def _rate(pred) -> float:
        return round(sum(1 for r in runs if pred(r)) / len(runs), 6) if runs else 0.0

    campaign_owners: dict[str, set[str]] = {}
    for r in runs:
        if r.campaign_id:
            campaign_owners.setdefault(r.campaign_id, set()).add(r.run_id)
    duplicates = sorted(cid for cid, o in campaign_owners.items() if len(o) > 1)
    missing = []
    for r in runs:
        gaps = [name for name, value in (("plan", r.plan_id),
                                         ("approval", r.approval_id),
                                         ("campaign", r.campaign_id),
                                         ("report", r.report_id),
                                         ("experience", r.experience_id))
                if not value]
        if gaps:
            missing.append({"run_id": r.run_id, "arm": r.arm,
                            "missing": gaps, "outcome": r.outcome})
    # ---- approval boundary (P1-03) -------------------------------------
    # The gate checks that an approval RECORD exists, is approved and bound to
    # the plan hash. It does NOT (and cannot) prove WHO pressed the key or that
    # a distinct, authenticated human was present. Report BOTH facts honestly.
    approval_sources: dict[str, int] = {}
    approved_by_values: set[str] = set()
    for r in runs:
        ev = r.gate_evidence or {}
        src = str(ev.get("approval_source") or "unknown")
        approval_sources[src] = approval_sources.get(src, 0) + 1
        if ev.get("approval_approved_by"):
            approved_by_values.add(str(ev["approval_approved_by"]))
    cfg_approver = str(protocol.get("approver_id", "") or "")
    # The pilot runner self-grants via a CLI SUBPROCESS
    # (``approval_cli grant --approved-by <config> --yes``). That is a real,
    # out-of-band command-line authorisation bound to the plan SHA, but it is
    # NOT an independent authenticated human approver.
    cli_subprocess_grants = sum(
        1 for r in runs
        if r.gate_evidence.get("approval_approved_by")
        and (bool(cfg_approver)
             and str(r.gate_evidence["approval_approved_by"]) == cfg_approver))
    approval_boundary = {
        "record_level_check": (
            "the 12-step gate verified an approval RECORD that is approved and "
            "SHA-bound to the current plan (approval_source label, "
            "approval_approved_by identity string)"),
        "approval_sources_labelled": approval_sources,
        "approved_by_identities": sorted(approved_by_values),
        "cli_subprocess_grants_detected": cli_subprocess_grants,
        "config_approver_id": cfg_approver,
        "independent_human_approval_confirmed": False,
        "identity_authentication_confirmed": False,
        "distinction": {
            "cli_subprocess_authorisation": (
                "a separate command-line process (approval_cli grant --yes) "
                "created the approved record; this is what the pilot performed"),
            "real_human_approval": (
                "a distinct human reviewing the plan and personally granting "
                "it -- NOT established by a --yes subprocess with a configured "
                "identity string"),
            "identity_authentication": (
                "verification that the approved_by string is an authenticated "
                "person -- NOT established by the approval record alone"),
        },
        "claim": ("approval evidence proves a SHA-bound CLI subprocess grant "
                  "under the identity label(s) listed above; it does NOT prove "
                  "an independent, authenticated human approver"),
    }

    integrity_section = {
        "n_runs": len(runs),
        "n_runs_per_arm": {a: len(v) for a, v in by_arm.items()},
        "outcome_counts": {o: sum(1 for r in runs if r.outcome == o)
                           for o in RUN_OUTCOMES},
        "gate_completeness": _rate(lambda r: r.gate_steps_passed == 12),
        "approval_completeness": _rate(lambda r: r.external_approval),
        "approval_boundary": approval_boundary,
        "report_completeness": _rate(lambda r: bool(r.report_id)),
        "experience_completeness": _rate(lambda r: bool(r.experience_id)),
        "registry_event_chain_valid": integrity,
        "duplicate_campaign_ids": duplicates,
        "missing_records": missing,
        "excluded_runs": [],
        # P0-01: paired-eligibility is shared with the pilot analysis.
        "n_blocks_eligible_for_pairing": len(paired["eligible"]),
        "excluded_blocks": paired["excluded"],
        "paired_eligibility_rules": paired["rules"],
        "note": ("failed / interrupted / invalid runs are RETAINED and stay in "
                 "every denominator; nothing is deleted"),
    }

    # -------------------------------------------------------- C primary
    contrast_runs: dict[str, dict[str, Any]] = {}
    for metric in PRIMARY_METRICS:
        diffs = _contrast_differences(
            runs, metric, blocks=blocks,
            expected_protocol_sha256=expected_protocol_sha256)
        # Pass the FIXED pre-registered family: a missing contrast can never
        # shrink the correction (P1-02).
        analysed = analyse_primary_contrasts(
            differences=diffs, alpha=alpha, correction=correction,
            family=TEST_FAMILY_LABELS)
        contrast_runs[metric] = analysed

    # P1-02: significance is claimed ONLY from a PRIMARY contrast (C-B) on a
    # PRIMARY metric. A secondary metric or a secondary contrast, however
    # significant, can NEVER set this flag.
    decision_sig = any(
        bool(contrast_runs.get(metric, {}).get("contrasts", {})
             .get(label, {}).get("significant"))
        for metric in PRIMARY_METRICS
        for label in PRIMARY_CONTRAST_LABELS)
    primary = {
        "metrics": list(PRIMARY_METRICS),
        "by_arm": {a: {m: compute_metric_set(v)[m] for m in PRIMARY_METRICS}
                   for a, v in by_arm.items()},
        # P1-03: a MISSING human-blind metric is shown as a missing value with
        # n_reviewed=0, NEVER as an actual score of 0.0.
        "objective_quality_observation": _quality_observation(by_arm),
        "contrasts": contrast_runs,
        "primary_contrast": PRIMARY_CONTRAST_LABELS[0],
        "decision_primary_metric": DECISION_PRIMARY_METRIC,
        "significance_claimed": bool(stage == "final" and decision_sig),
        "significance_basis": {
            "allowed_metrics": list(PRIMARY_METRICS),
            "allowed_contrasts": list(PRIMARY_CONTRAST_LABELS),
            "excluded_contrasts": list(SECONDARY_CONTRAST_LABELS),
            "excluded_metrics": list(SECONDARY_METRICS),
            "rule": ("significant only if a PRIMARY contrast on a PRIMARY "
                     "metric survives the fixed-family correction"),
        },
    }

    # ------------------------------------------------------ D secondary
    secondary = {
        "metrics": list(SECONDARY_METRICS),
        "by_arm": {a: {m: compute_metric_set(v)[m] for m in SECONDARY_METRICS}
                   for a, v in by_arm.items()},
    }

    # --------------------------------------------------------- E safety
    safety = {
        "metrics": list(SAFETY_METRICS),
        "by_arm": {a: {m: compute_metric_set(v)[m] for m in SAFETY_METRICS}
                   for a, v in by_arm.items()},
        "counts": {
            "unsupported_evidence_claim": sum(
                1 for r in runs if r.unsupported_evidence_claim),
            "false_objective": sum(1 for r in runs if r.false_objective),
            "frozen_split_violation": sum(
                1 for r in runs if r.frozen_split_violation),
            "approval_bypass": sum(1 for r in runs if r.approval_bypass),
            "duplicate_campaign": sum(1 for r in runs if r.duplicate_campaign),
        },
    }

    # ------------------------------------------------------- F economics
    economics = {
        "compute_units_total": round(sum(r.compute_units for r in runs), 6),
        "compute_units_by_arm": {a: round(sum(r.compute_units for r in v), 6)
                                 for a, v in by_arm.items()},
        "wall_time_s_total": round(sum(r.wall_time_s for r in runs), 3),
        "wall_time_s_by_arm": {a: round(sum(r.wall_time_s for r in v), 3)
                               for a, v in by_arm.items()},
        "n_valid_findings": sum(1 for r in runs if r.outcome == "valid_finding"),
        "compute_per_valid_finding": compute_metric_set(runs)[
            "compute_per_valid_finding"],
        "compute_per_objective": {
            a: (round(sum(r.compute_units for r in v) / len(v), 6) if v else None)
            for a, v in by_arm.items()},
        "selection_cost": {
            "automated_discovery_runs": sum(
                1 for r in runs if r.notes.get("selection_context")),
        },
    }

    # ------------------------------------------------------- G conclusion
    conclusion = _conclusion(stage=stage, primary=primary,
                             integrity_section=integrity_section,
                             runs=runs, economics=economics)
    conclusion["outcome_key"] = OUTCOME_KEYS.get(conclusion["outcome"],
                                                 "unknown")
    # One source of truth: the conclusion's claim is the report's claim. The
    # primary section can never advertise significance the conclusion refuses.
    primary["significance_claimed"] = bool(conclusion["significance_claimed"])

    chains = [_evidence_chain(r) for r in runs]
    key_links = {
        label: [c for c in chains if c["arm"] in (label.split("-")[0],
                                                  label.split("-")[1])]
        for label in PRIMARY_CONTRAST_LABELS + SECONDARY_CONTRAST_LABELS}

    report = {
        "schema": "m4_3_value_report_v1",
        "stage": stage,
        "experiment_id": experiment_id,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "is_pilot": stage == "pilot",
        "design": design,
        "data_integrity": integrity_section,
        "primary": primary,
        "secondary": secondary,
        "safety": safety,
        "economics": economics,
        "conclusion": conclusion,
        "evidence_chains": chains,
        "contrast_evidence": key_links,
        "pilot_analysis_ref": ("pilot_analysis.json" if pilot_analysis else ""),
        "metric_contract": frozen_metric_contract(),
    }
    if write:
        name = "value_report.json" if stage == "final" else "pilot_value_report.json"
        (out / name).write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str),
            encoding="utf-8")
        md_name = ("M4_3_VALUE_REPORT.md" if stage == "final"
                   else "M4_3_PILOT_VALUE_REPORT.md")
        (out / md_name).write_text(render_value_report(report), encoding="utf-8")
    return report


def _conclusion(*, stage: str, primary: dict[str, Any],
                integrity_section: dict[str, Any], runs: list[RunRecord],
                economics: dict[str, Any]) -> dict[str, Any]:
    """Choose ONE of the four allowed outcomes -- never a forced C win.

    P1-02: the value conclusion is driven ONLY by the PRE-REGISTERED primary
    contrast (C-B) on the PRE-REGISTERED primary metrics. A secondary contrast
    (B-A / C-A) or a secondary metric is context, never a decision driver.
    """
    if not runs:
        return {"outcome": VALUE_OUTCOMES[3], "significance_claimed": False,
                "reason": "no persisted runs: nothing to report",
                "evidence": [], "decision_contrast": PRIMARY_CONTRAST_LABELS[0]}
    primary_contrast = PRIMARY_CONTRAST_LABELS[0]
    # Snapshot across metrics for the PRIMARY contrast only.
    cb_by_metric = {
        m: primary["contrasts"][m]["contrasts"].get(primary_contrast, {})
        for m in PRIMARY_METRICS}
    cb = cb_by_metric[DECISION_PRIMARY_METRIC]
    n_pairs = cb.get("n") or 0
    sig_by_metric = {
        m: bool(v.get("significant")) for m, v in cb_by_metric.items()}
    decision_sig = primary.get("significance_claimed", False)
    decision_basis = {
        "decision_contrast": primary_contrast,
        "decision_primary_metric": DECISION_PRIMARY_METRIC,
        "primary_metrics": list(PRIMARY_METRICS),
        "primary_contrast_significant_by_metric": sig_by_metric,
        "secondary_contrasts_excluded": list(SECONDARY_CONTRAST_LABELS),
    }
    if stage != "final":
        return {
            "outcome": VALUE_OUTCOMES[3],
            "significance_claimed": False,
            "reason": ("PILOT: this report is a pilot artefact. The pilot never "
                       "declares a winner; the four-way value conclusion is "
                       "reserved for the frozen confirmatory analysis."),
            "pilot_contrast_snapshot": {
                "primary_contrast": primary_contrast,
                "by_primary_metric": cb_by_metric, "n_pairs": n_pairs},
            "decision_basis": decision_basis,
            "evidence": [],
        }
    if n_pairs < 3 or integrity_section["duplicate_campaign_ids"]:
        return {"outcome": VALUE_OUTCOMES[3], "significance_claimed": False,
                "reason": (f"{n_pairs} paired block(s) on the primary contrast "
                           "and/or an integrity problem cannot support a value "
                           "conclusion"),
                "decision_basis": decision_basis, "evidence": []}
    if any(v.get("p_value") is None for v in cb_by_metric.values()):
        # Missing primary data (e.g. pending human review) is NOT evidence of
        # "no value": it is evidence that we cannot decide yet.
        return {"outcome": VALUE_OUTCOMES[3], "significance_claimed": False,
                "reason": ("the pre-registered primary contrast has no "
                           "computable statistic on a primary metric "
                           f"({ {m: v.get('n') for m, v in cb_by_metric.items()} }); "
                           "the human blind review and/or more blocks are "
                           "needed first"),
                "decision_basis": decision_basis, "evidence": []}
    if not decision_sig:
        # No primary-contrast/primary-metric result survived the correction.
        # Secondary metrics/contrasts never change this into a positive claim.
        return {"outcome": VALUE_OUTCOMES[2], "significance_claimed": False,
                "reason": ("the pre-registered primary contrast (C-B) on the "
                           "primary metrics does not survive the fixed-family "
                           "correction; secondary contrasts are not decision "
                           "drivers"),
                "decision_basis": decision_basis,
                "evidence": [primary_contrast]}
    # significance_claimed is True: C-B survived on >=1 primary metric.
    if sig_by_metric.get(DECISION_PRIMARY_METRIC):
        return {"outcome": VALUE_OUTCOMES[0], "significance_claimed": True,
                "reason": ("the pre-registered primary contrast (C-B) is "
                           "significant on the decision primary metric after "
                           "the fixed-family correction"),
                "decision_basis": decision_basis,
                "evidence": [primary_contrast, DECISION_PRIMARY_METRIC]}
    return {"outcome": VALUE_OUTCOMES[1], "significance_claimed": True,
            "reason": ("C-B is significant on a primary metric other than the "
                       "decision metric: efficiency improved without a "
                       "scientific-discovery improvement"),
            "decision_basis": decision_basis,
            "evidence": [primary_contrast]}


# ------------------------------------------------------------------ render
def render_value_report(r: dict[str, Any]) -> str:
    lines: list[str] = []
    title = ("M4.3-04 PILOT Value Report" if r["is_pilot"]
             else "M4.3-04 VALUE REPORT")
    lines.append(f"# {title} -- {r['experiment_id']}")
    lines.append("")
    lines.append(f"Generated: {r['generated_at']}  ")
    lines.append(f"Stage: **{r['stage']}**  ")
    lines.append(f"Protocol SHA-256: `{r['design']['protocol_sha256']}`")
    lines.append("")
    if r["is_pilot"]:
        lines.append("> **PILOT ARTEFACT.** The value conclusion below is "
                     "deliberately `evidence insufficient`; pilots do not "
                     "declare winners. The frozen confirmatory analysis "
                     "produces `M4_3_VALUE_REPORT.md`.")
        lines.append("")

    d = r["design"]
    lines.append("## A. Experiment design")
    lines.append("")
    lines.append(f"- blocks: {d['n_blocks']}; snapshots: {len(d['snapshot_ids'])}")
    for _arm in ARMS:
        lines.append(f"- arm {_arm}: {d.get('arm_descriptions', {}).get(_arm, '')}")
    lines.append(f"- model / prompt / ranker / policy: `{d['model_version']}` / "
                 f"`{d['prompt_version']}` / `{d['ranker_version']}` / "
                 f"`{d['domain_policy_version']}`")
    lines.append(f"- paired seeds: {d['paired_seeds']}; budget: "
                 f"{d['campaign_budget']}")
    lines.append(f"- exclusion rules: {d['exclusion_rules']}")
    lines.append(f"- statistical method: {d['statistical_method']} "
                 f"(correction: {d['multiple_comparison_correction']})")
    lines.append(f"- primary contrast: {d['primary_contrasts']}; secondary: "
                 f"{d['secondary_contrasts']}")
    lines.append("")

    b = r["data_integrity"]
    lines.append("## B. Data integrity")
    lines.append("")
    lines.append("| check | value |")
    lines.append("|---|---|")
    for key in ("n_runs", "gate_completeness", "approval_completeness",
                "report_completeness", "experience_completeness",
                "registry_event_chain_valid", "outcome_counts"):
        lines.append(f"| {key} | {b[key]} |")
    lines.append("")
    ab = b.get("approval_boundary") or {}
    if ab:
        lines.append("### Approval boundary (authorisation vs human identity)")
        lines.append("")
        lines.append(f"- record-level check: {ab['record_level_check']}")
        lines.append(f"- approval sources (labels): {ab['approval_sources_labelled']}")
        lines.append(f"- approved_by identities: {ab['approved_by_identities']}")
        lines.append(f"- CLI subprocess grants detected: "
                     f"{ab['cli_subprocess_grants_detected']}")
        lines.append(f"- independent human approval confirmed: "
                     f"**{ab['independent_human_approval_confirmed']}**")
        lines.append(f"- identity authentication confirmed: "
                     f"**{ab['identity_authentication_confirmed']}**")
        lines.append(f"- claim: {ab['claim']}")
        lines.append("")
    if b["duplicate_campaign_ids"]:
        lines.append(f"**Duplicate campaign ids:** {b['duplicate_campaign_ids']}")
        lines.append("")
    if b["missing_records"]:
        lines.append("Missing trusted ids per run:")
        lines.append("")
        for m in b["missing_records"]:
            lines.append(f"- {m['run_id']} ({m['arm']}, {m['outcome']}): "
                         f"{m['missing']}")
        lines.append("")

    c = r["primary"]
    lines.append("## C. Primary metrics")
    lines.append("")
    qobs = c.get("objective_quality_observation") or {}
    if qobs:
        lines.append("Objective quality (HUMAN-BLIND; missing values shown as "
                     "`missing`, never as a 0 score):")
        lines.append("")
        lines.append("| arm | n reviewed | value | status |")
        lines.append("|---|---|---|---|")
        for arm in ARMS:
            info = qobs.get(arm, {})
            val = info.get("value")
            lines.append(f"| {arm} | {info.get('n_reviewed')} | "
                         f"{'missing' if val is None else val} | "
                         f"{info.get('status')} |")
        lines.append("")
    lines.append("| metric (raw contract values) | " + " | ".join(ARMS) + " |")
    lines.append("|---|" + "---|" * len(ARMS))
    for metric in c["metrics"]:
        lines.append(f"| {metric} | " + " | ".join(
            str(c["by_arm"][a][metric]) for a in ARMS) + " |")
    lines.append("")
    for metric, payload in c["contrasts"].items():
        lines.append(f"### {metric}")
        lines.append("")
        lines.append("| contrast | n | mean diff | 95% CI | effect (dz) | "
                     "p | corrected | significant |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for label, m in payload["contrasts"].items():
            ci = m.get("confidence_interval_95")
            corr = (m.get("correction") or {}).get("effective_alpha")
            lines.append(
                f"| {label} | {m.get('n')} | {m.get('mean')} | {ci} | "
                f"{m.get('effect_size_dz')} | {m.get('p_value')} | "
                f"{corr} | {m.get('significant')} |")
        lines.append("")

    s = r["secondary"]
    lines.append("## D. Secondary metrics")
    lines.append("")
    lines.append("| metric | " + " | ".join(ARMS) + " |")
    lines.append("|---|" + "---|" * len(ARMS))
    for metric in s["metrics"]:
        lines.append(f"| {metric} | " + " | ".join(
            str(s["by_arm"][a][metric]) for a in ARMS) + " |")
    lines.append("")

    sf = r["safety"]
    lines.append("## E. Safety")
    lines.append("")
    lines.append("| metric | " + " | ".join(ARMS) + " |")
    lines.append("|---|" + "---|" * len(ARMS))
    for metric in sf["metrics"]:
        lines.append(f"| {metric} | " + " | ".join(
            str(sf["by_arm"][a][metric]) for a in ARMS) + " |")
    lines.append("")
    lines.append(f"Counts: {sf['counts']}")
    lines.append("")

    e = r["economics"]
    lines.append("## F. Economics")
    lines.append("")
    lines.append(f"- total compute units: {e['compute_units_total']} "
                 f"({e['compute_units_by_arm']})")
    lines.append(f"- total wall time (s): {e['wall_time_s_total']} "
                 f"({e['wall_time_s_by_arm']})")
    lines.append(f"- valid findings: {e['n_valid_findings']}; compute per valid "
                 f"finding: {e['compute_per_valid_finding']}")
    lines.append(f"- compute per objective: {e['compute_per_objective']}")
    lines.append("")

    con = r["conclusion"]
    lines.append("## G. Conclusion")
    lines.append("")
    lines.append(f"**{con['outcome']}**")
    lines.append("")
    lines.append(f"{con['reason']}")
    lines.append("")
    lines.append(f"Significance claimed: **{con['significance_claimed']}**")
    lines.append("")
    if r["contrast_evidence"]:
        lines.append("Evidence chains (block -> campaign -> report -> experience):")
        lines.append("")
        for label, chains in r["contrast_evidence"].items():
            lines.append(f"- **{label}**")
            for ch in chains:
                lines.append(
                    f"  - `{ch['run_id']}` block=`{ch['block_id']}` "
                    f"campaign=`{ch['campaign_id']}` report=`{ch['report_id']}` "
                    f"experience=`{ch['experience_id']}` "
                    f"outcome={ch['outcome']} gate={ch['gate_steps_passed']}/12")
        lines.append("")
    return "\n".join(lines) + "\n"


__all__ = ["VALUE_OUTCOMES", "OUTCOME_KEYS", "PRIMARY_CONTRAST_LABELS",
           "SECONDARY_CONTRAST_LABELS", "generate_value_report",
           "render_value_report"]
