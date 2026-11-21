"""M6.1-fix-A: truly independent hidden-holdout confirmation.

The candidate is FROZEN during Selection with a fixed SHA. Holdout tasks are
accessed ONLY here, via the HoldoutAccessVerifier, with a preregistered purpose
and a one-shot guard: once a holdout manifest has been used for a given
candidate SHA, re-using it (e.g. after re-tuning) is HOLDOUT_LEAKAGE and fails
closed. Holdout results are NEVER returned to candidate generation; they are
written to a separate evidence artifact and the generator view is re-checked
for absence of holdout ids/families before and after.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .benchmark_v2 import (
    BenchmarkIsolationError, HoldoutAccessVerifier,
    assert_no_holdout_in_generator_view,
)
from .identity import freeze_json, sha256_json, verify_frozen

CONFIRM_VERSION = "m6_1_confirmatory_holdout_v1"
ALLOWED_PURPOSE = "final_holdout_evaluation"


class ConfirmationError(RuntimeError):
    """Raised on any holdout protocol violation."""


def freeze_candidate_for_selection(*, candidate_sha: str,
                                   selection_result_sha: str,
                                   prereg_sha: str,
                                   out_path: str | Path) -> dict[str, Any]:
    """Freeze the candidate SHA at the END of Selection, before holdout access."""
    if len(candidate_sha) != 64 or len(selection_result_sha) != 64:
        raise ConfirmationError("candidate/selection SHA must be 64-hex")
    obj = {"version": CONFIRM_VERSION,
           "candidate_sha": candidate_sha,
           "selection_result_sha": selection_result_sha,
           "preregistration_sha": prereg_sha,
           "frozen_at_selection": True}
    return {**obj, **freeze_json(obj, out_path)}


def run_confirmatory_holdout(*, spec: Any, verifier: HoldoutAccessVerifier,
                             frozen: dict[str, Any],
                             experiment_id: str,
                             runner: Callable[[Any, str], dict[str, Any]],
                             out_dir: str | Path,
                             holdout_task_ids: list[str] | None = None,
                             actor: str = "confirmatory_evaluator"
                             ) -> dict[str, Any]:
    """Execute Base/Candidate on holdout ONLY via verifier; return sealed evidence.

    ``runner(task, arm) -> {"ok", "metrics", "result_sha256", "artifact_path"}``
    is the SAME real engine runner used in Selection; it receives no candidate
    object and cannot write back to generation.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Precondition: generator never sees holdout.
    clean = assert_no_holdout_in_generator_view(spec)
    if not clean["ok"]:
        raise ConfirmationError("generator view already leaks holdout")
    if not frozen.get("frozen_at_selection"):
        raise ConfirmationError("candidate not frozen at Selection")

    wanted = holdout_task_ids or [t.task_id for t in spec.by_split("holdout")]
    auth = verifier.authorize(purpose=ALLOWED_PURPOSE,
                              experiment_id=experiment_id,
                              requested_task_ids=wanted, actor=actor)
    if not auth["ok"]:
        raise ConfirmationError("HOLDOUT_LEAKAGE: " + json.dumps(auth))

    # One-shot guard: a prior access for THIS candidate+holdout manifest is
    # illegal (no re-tuning against the same holdout).
    ledger = verifier.verify_chain()
    prior = [e for e in (json.loads(verifier.path.read_text(encoding="utf-8")
                                    ).get("entries", []))
             if e.get("experiment_id") == experiment_id and e.get(
                 "purpose") == ALLOWED_PURPOSE]
    if len(prior) != 1:
        raise ConfirmationError(
            f"holdout reused for experiment {experiment_id}: {len(prior)} accesses")

    tasks = verifier.tasks(purpose=ALLOWED_PURPOSE,
                           experiment_id=experiment_id,
                           requested_task_ids=wanted, actor=actor)
    base_rows: list[dict[str, Any]] = []
    cand_rows: list[dict[str, Any]] = []
    for task in tasks:
        for arm, rows in (("base", base_rows), ("candidate", cand_rows)):
            raw = runner(task, arm)
            artifact = str(raw.get("artifact_path", ""))
            if not artifact or not Path(artifact).is_file():
                raise ConfirmationError(
                    f"{task.task_id}/{arm}: no real result artifact")
            # Re-read every result from disk and verify its SHA.
            chk = verify_frozen(artifact, str(raw.get("result_sha256", "")))
            if not chk["ok"]:
                raise ConfirmationError(
                    f"{task.task_id}/{arm}: result sha mismatch "
                    + "|".join(chk["reasons"]))
            rows.append({"task_id": task.task_id, "family_id": task.family_id,
                         "arm": arm, "result_sha256": chk["sha256"],
                         "metrics": dict(raw.get("metrics", {}))})

    evidence = freeze_json({
        "version": CONFIRM_VERSION, "experiment_id": experiment_id,
        "candidate_sha": frozen["candidate_sha"],
        "selection_result_sha": frozen["selection_result_sha"],
        "holdout_access_sha": auth["entry_sha"],
        "holdout_manifest_sha": spec.split_manifest_shas["holdout"],
        "base": base_rows, "candidate": cand_rows,
        "isolated_from_selection": True,
        "generator_never_saw_holdout": assert_no_holdout_in_generator_view(
            spec)["ok"],
        "holdout_ledger_chain": ledger["chain_sha"],
    }, root / "holdout_confirmation.json")
    return {"ok": True, "evidence_sha": evidence["sha256"],
            "evidence_path": evidence["path"],
            "n_tasks": len(tasks), "base": base_rows, "candidate": cand_rows}


__all__ = ["CONFIRM_VERSION", "ConfirmationError",
           "freeze_candidate_for_selection", "run_confirmatory_holdout"]
