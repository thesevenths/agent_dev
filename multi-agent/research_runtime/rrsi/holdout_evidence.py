"""M5 Integrated -- structured, verifiable Holdout evidence for RRSI promotion.

A boolean ``holdout_confirmed`` is NOT proof. Promotion requires a structured
evidence object whose identities and SHAs are cross-checked against the exact
candidate/parent and whose evaluation task ids are provably disjoint from the
task ids used for candidate selection. This is a framework-agnostic ML rule:
a held-out set that overlaps selection, or whose dataset/split/results cannot
be identified, is not held out at all.

Fail-closed: any missing/invalid/unverifiable field -> promotion denied with an
explicit reason. Nothing here writes state or grants anything.
"""
from __future__ import annotations

import re
from typing import Any

_SHA64 = re.compile(r"^[0-9a-f]{64}$")
_NONEMPTY = re.compile(r"\S")

# Splits that are reserved for FINAL confirmation, never for candidate
# selection or standalone promotion evidence.
_RESERVED_FINAL_SPLITS = frozenset({"test", "final", "reserved", "holdout_final"})


class HoldoutEvidenceError(ValueError):
    """Raised when structured holdout evidence cannot be verified."""


def _sha_ok(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA64.match(value.strip().lower()))


def _id_list_ok(values: Any) -> bool:
    return (isinstance(values, (list, tuple))
            and len(values) > 0
            and all(isinstance(v, str) and bool(_NONEMPTY.search(v))
                    for v in values))


def verify_holdout_evidence(
    evidence: Any,
    *,
    expected_candidate_sha: str,
    expected_parent_sha: str = "",
    min_holdout_tasks: int = 1,
    min_holdout_seeds: int = 2,
) -> dict[str, Any]:
    """Verify a structured holdout evidence object. Returns a verdict dict.

    Required keys (all must be REAL and self-consistent):
      dataset_manifest_sha : sha256 of the frozen dataset manifest actually used
      split_id             : holdout split/session id (not a reserved final split)
      candidate_sha        : must equal expected_candidate_sha
      parent_sha           : must equal expected_parent_sha
      evaluation_task_ids  : non-empty concrete task ids scored on holdout
      selection_task_ids   : task ids used to generate/select the candidate
      result_shas          : sha256 per evaluation task (same count)
      metrics              : dict with at least one finite numeric metric
      statistics           : dict with n_holdout_seeds / direction info
      independent          : True (explicit claim, cross-checked below)
    """
    problems: list[str] = []
    if not isinstance(evidence, dict):
        return {"verified": False, "reasons": ["evidence_not_an_object"]}

    manifest = evidence.get("dataset_manifest_sha")
    if not _sha_ok(manifest):
        problems.append("dataset_manifest_sha_missing_or_invalid")

    split_id = evidence.get("split_id")
    if not isinstance(split_id, str) or not _NONEMPTY.search(split_id):
        problems.append("split_id_missing")
    elif split_id.strip().lower() in _RESERVED_FINAL_SPLITS:
        problems.append("split_is_reserved_final_not_holdout")

    if not _sha_ok(evidence.get("candidate_sha")):
        problems.append("candidate_sha_missing_or_invalid")
    elif evidence["candidate_sha"].strip().lower() != expected_candidate_sha.strip().lower():
        problems.append("candidate_sha_mismatch")

    if expected_parent_sha:
        if not _sha_ok(evidence.get("parent_sha")):
            problems.append("parent_sha_missing_or_invalid")
        elif evidence.get("parent_sha", "").strip().lower() != expected_parent_sha.strip().lower():
            problems.append("parent_sha_mismatch")

    eval_ids = evidence.get("evaluation_task_ids")
    sel_ids = evidence.get("selection_task_ids")
    result_shas = evidence.get("result_shas")
    if not _id_list_ok(eval_ids):
        problems.append("evaluation_task_ids_missing")
    if not isinstance(sel_ids, (list, tuple)):
        problems.append("selection_task_ids_missing")
    if not isinstance(result_shas, (list, tuple)) or not all(
            _sha_ok(x) for x in result_shas):
        problems.append("result_shas_missing_or_invalid")
    elif _id_list_ok(eval_ids) and len(result_shas) != len(eval_ids):
        problems.append("result_sha_count_mismatch")
    if _id_list_ok(eval_ids) and isinstance(sel_ids, (list, tuple)):
        overlap = set(eval_ids) & set(str(x) for x in sel_ids)
        if overlap:
            problems.append("holdout_overlaps_selection")

    metrics = evidence.get("metrics")
    if not isinstance(metrics, dict) or not any(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            and v == v and abs(v) != float("inf") for v in metrics.values()):
        problems.append("metrics_missing_or_non_finite")

    stats = evidence.get("statistics")
    if not isinstance(stats, dict):
        problems.append("statistics_missing")
    else:
        n_seeds = stats.get("n_holdout_seeds")
        if not isinstance(n_seeds, int) or n_seeds < min_holdout_seeds:
            problems.append("n_holdout_seeds_below_floor")
        direction = stats.get("holdout_direction_fraction")
        if direction is not None and not (
                isinstance(direction, (int, float)) and 0.0 <= direction <= 1.0):
            problems.append("holdout_direction_fraction_invalid")

    if evidence.get("independent") is not True:
        problems.append("independence_not_asserted")

    n_tasks = len(eval_ids) if isinstance(eval_ids, (list, tuple)) else 0
    if n_tasks < min_holdout_tasks:
        problems.append("too_few_holdout_tasks")

    return {"verified": not problems, "reasons": problems,
            "n_holdout_tasks": len(eval_ids) if isinstance(eval_ids, (list, tuple)) else 0}


def gate_allows_promotion_verified(
    decision: str,
    *,
    holdout_confirmed: bool,
    frozen_test_used: bool = False,
    training_only: bool = False,
    holdout_evidence: Any = None,
    expected_candidate_sha: str = "",
    expected_parent_sha: str = "",
) -> tuple[bool, list[str]]:
    """Strict promotion gate: boolean + structured evidence must BOTH pass."""
    from .evolution_experiment import gate_allows_promotion  # local: avoid cycle
    reasons: list[str] = []
    if not gate_allows_promotion(decision, holdout_confirmed=holdout_confirmed,
                                 frozen_test_used=frozen_test_used,
                                 training_only=training_only):
        reasons.append("base_gate_denied")
    if not expected_candidate_sha:
        reasons.append("no_candidate_sha_to_bind")
    verdict = verify_holdout_evidence(
        holdout_evidence,
        expected_candidate_sha=expected_candidate_sha,
        expected_parent_sha=expected_parent_sha)
    if not verdict["verified"]:
        reasons.extend(verdict["reasons"])
    return (not reasons, reasons)


__all__ = [
    "HoldoutEvidenceError", "verify_holdout_evidence",
    "gate_allows_promotion_verified",
]
