"""M6 -- formal Harness promotion: verify first, then promote.

Formal promotion is STRICTER than the M5 dual-layer gate:

  * the legacy boolean ``holdout_confirmed`` path is FORBIDDEN for formal
    harness promotion -- a bare True without sealed evidence cannot promote;
  * structured holdout evidence must bind candidate SHA, parent SHA, manifest
    SHA, per-task result SHAs, seed/split and a real artifact reader that
    RE-READS the冻结 files from disk;
  * task ids AND task families of holdout must be disjoint from selection;
  * the pre-registered statistical gate (primary MME + corrected significance)
    must be present and passed;
  * the gate implementation version SHA is recorded.

The actual registry write reuses the existing M5 ``EvolutionRegistry`` through
``RRSICoordinator``; no second registry is created.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..rrsi.evolution_registry import EvolutionRegistry
from ..rrsi.holdout_evidence import verify_holdout_evidence
from .harness import HarnessBundle
from .identity import sha256_json, verify_frozen

GATE_VERSION = "m6_formal_harness_promotion_v1"


class PromotionError(ValueError):
    """Raised when the promotion contract itself is violated."""


def gate_version_sha() -> str:
    """Identity of the gate implementation contract (not an assertion)."""
    return sha256_json({
        "gate": GATE_VERSION,
        "required": sorted([
            "structured_holdout", "artifact_reader", "manifest_recheck",
            "task_id_disjoint", "family_disjoint", "per_task_result_sha",
            "statistical_gate", "candidate_sha", "parent_sha",
        ]),
        "legacy_boolean_path_allowed": False,
    })


def make_artifact_reader(*, candidate_sha: str,
                         candidate_artifact_path: str,
                         parent_sha: str = "",
                         parent_artifact_path: str = "") -> Any:
    """Build the REAL disk re-reader used by the strict holdout verifier."""
    def reader(evidence: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        # 1. The frozen candidate artifact must still be exactly the one bound.
        from .harness import load_frozen_bundle
        check = load_frozen_bundle(candidate_artifact_path,
                                   expected_candidate_sha=candidate_sha)
        if not check["ok"]:
            reasons.extend("candidate_" + r for r in check["reasons"])
        # 2. Parent artifact, when one exists, must also be verifiable.
        if parent_sha and parent_artifact_path:
            pc = load_frozen_bundle(parent_artifact_path,
                                    expected_candidate_sha=parent_sha)
            if not pc["ok"]:
                reasons.extend("parent_" + r for r in pc["reasons"])
        # 3. EVERY per-task result file must exist and match its recorded SHA.
        paths = list(evidence.get("result_file_paths") or [])
        shas = list(evidence.get("result_shas") or [])
        if not paths or len(paths) != len(shas):
            reasons.append("per_task_result_files_missing")
        else:
            for p, expected in zip(paths, shas):
                v = verify_frozen(p, expected)
                if not v["ok"]:
                    reasons.append("result_" + "|".join(v["reasons"]))
        return {"ok": not reasons, "reasons": reasons}
    return reader


def verify_formal_promotion(*, candidate: HarnessBundle,
                            parent_sha: str,
                            holdout_evidence: dict[str, Any],
                            manifest_path: str,
                            artifact_reader: Any,
                            preregistration_sha: str,
                            statistical_gate: dict[str, Any],
                            promotable_view: dict[str, Any],
                            legacy_boolean_holdout: Any = None
                            ) -> dict[str, Any]:
    """Fail-closed formal verification. Returns a verdict, never writes state."""
    reasons: list[str] = []
    if legacy_boolean_holdout is not None:
        reasons.append("legacy_boolean_holdout_path_forbidden")
    if artifact_reader is None:
        reasons.append("missing_artifact_reader")
    if not manifest_path or not Path(manifest_path).is_file():
        reasons.append("missing_manifest")
    if not preregistration_sha:
        reasons.append("missing_preregistration_sha")

    # M5 structured verification (candidate/parent/manifest/SHAs/re-read).
    verdict = verify_holdout_evidence(
        holdout_evidence, expected_candidate_sha=candidate.sha256,
        expected_parent_sha=parent_sha,
        artifact_reader=artifact_reader, manifest_path=manifest_path)
    if not verdict["verified"]:
        reasons.extend("m6_" + r for r in verdict["reasons"])

    # M6-specific: task-family disjointness + manifest identity cross-check.
    ev_fams = set(holdout_evidence.get("evaluation_families") or [])
    sel_fams = set(holdout_evidence.get("selection_families") or [])
    if not ev_fams or not sel_fams:
        reasons.append("task_families_missing")
    elif ev_fams & sel_fams:
        reasons.append("task_family_overlap_selection_holdout")
    expected_manifest = str(promotable_view.get("manifest_sha256") or "")
    if expected_manifest and str(holdout_evidence.get("dataset_manifest_sha", "")
                                ).lower() != expected_manifest.lower():
        reasons.append("benchmark_manifest_mismatch")
    ev_ids = set(holdout_evidence.get("evaluation_task_ids") or [])
    sel_ids = set(holdout_evidence.get("selection_task_ids") or [])
    if not sel_ids or not ev_ids:
        reasons.append("selection_or_holdout_empty")
    allowed_ids = set(promotable_view.get("holdout_task_ids") or [])
    if allowed_ids and not ev_ids.issubset(allowed_ids):
        reasons.append("holdout_task_ids_not_from_benchmark")
    allowed_sel = set(promotable_view.get("selection_task_ids") or [])
    if allowed_sel and not sel_ids.issubset(allowed_sel):
        reasons.append("selection_task_ids_not_from_benchmark")

    # Pre-registered statistical gate: no gate -> no promotion.
    required_stat = ("decision", "primary_meets_mme", "significant_after_correction",
                     "infrastructure_failures", "evaluator_sha256")
    if not isinstance(statistical_gate, dict):
        reasons.append("statistical_gate_missing")
    else:
        missing = [k for k in required_stat if k not in statistical_gate]
        if missing:
            reasons.extend("statistical_gate_missing_" + k for k in missing)
        else:
            if (statistical_gate["decision"] != "ACCEPTED"
                    or not statistical_gate["primary_meets_mme"]
                    or not statistical_gate["significant_after_correction"]):
                reasons.append("statistical_gate_not_passed")
            if statistical_gate["infrastructure_failures"]:
                reasons.append("infrastructure_failure_present")
            if statistical_gate["evaluator_sha256"] != preregistration_sha:
                reasons.append("evaluator_version_mismatch")

    return {"verified": not reasons, "reasons": reasons,
            "gate_version_sha": gate_version_sha(),
            "structured_verification": verdict}


def formal_promote(*, registry: EvolutionRegistry, experiment_id: str,
                   candidate: HarnessBundle, parent_sha: str,
                   holdout_evidence: dict[str, Any], manifest_path: str,
                   artifact_reader: Any, preregistration_sha: str,
                   statistical_gate: dict[str, Any],
                   promotable_view: dict[str, Any],
                   human_approval: dict[str, Any] | None = None,
                   legacy_boolean_holdout: Any = None) -> dict[str, Any]:
    """Verify, then write exactly ONE promotion outcome to the M5 registry."""
    verdict = verify_formal_promotion(
        candidate=candidate, parent_sha=parent_sha,
        holdout_evidence=holdout_evidence, manifest_path=manifest_path,
        artifact_reader=artifact_reader,
        preregistration_sha=preregistration_sha,
        statistical_gate=statistical_gate,
        promotable_view=promotable_view,
        legacy_boolean_holdout=legacy_boolean_holdout)
    approved = bool(human_approval and human_approval.get("approved") is True)
    if not verdict["verified"]:
        status = "REJECTED" if statistical_gate.get("decision") == "REJECTED" \
            else "INCONCLUSIVE"
    elif not approved:
        status = "INCONCLUSIVE"   # verified but NOT adopted without human grant
    else:
        status = "PROMOTED"

    evidence = {
        "gate_version_sha": verdict["gate_version_sha"],
        "formal_verification": verdict,
        "holdout_evidence": holdout_evidence,
        "statistical_gate": statistical_gate,
        "preregistration_sha256": preregistration_sha,
        "human_approval": human_approval or {"approved": False,
                                             "reason": "missing_human_approval"},
        "candidate_sha": candidate.sha256,
        "parent_sha": parent_sha,
        "outcome": status,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    registry.record(experiment_id=experiment_id, layer="research_harness",
                    candidate_id=candidate.harness_id,
                    candidate_sha=candidate.sha256, parent_sha=parent_sha,
                    status=status, gate_passed=(status == "PROMOTED"),
                    evidence=evidence)
    return {"status": status, "verification": verdict,
            "candidate_sha": candidate.sha256, "parent_sha": parent_sha,
            "experiment_id": experiment_id}


__all__ = [
    "PromotionError", "GATE_VERSION", "gate_version_sha",
    "make_artifact_reader", "verify_formal_promotion", "formal_promote",
]
