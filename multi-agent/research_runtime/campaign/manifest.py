"""P1-04: the FINAL, auditable record of what a campaign actually ran.

Deliberate boundary: this module records experiment FACTS only. It never
computes a baseline-vs-candidate delta, a win/loss, a confidence interval or
a promotion decision. Those belong to P1-05 (Comparator) and P1-06 (Agent
loop). A campaign that executed correctly is a success even when every
candidate is worse than the baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .campaign_store import CampaignStore
from .policy import CampaignPolicy
from .schemas import CampaignRecord, CampaignRunSpec, TrialRecord

MANIFEST_VERSION = 1


def _trial_fact(trial: TrialRecord) -> dict[str, Any]:
    return {
        "status": trial.status,
        "arm_type": trial.arm_type,
        "arm_id": trial.arm_id,
        "seed": trial.seed,
        "hypothesis_id": trial.hypothesis_id,
        "config_path": trial.config_relpath or trial.config_path,
        "config_sha256": trial.config_sha256,
        "baseline_config_sha256": trial.baseline_config_sha256,
        "dataset_version": trial.dataset_version,
        "train_job_id": trial.train_job_id,
        "train_run_id": trial.train_run_id,
        "evaluation_job_id": trial.evaluation_job_id,
        "evaluation_split": trial.evaluation_split,
        "primary_metric": trial.primary_metric,
        "train_result_path": trial.train_result_path,
        "evaluation_result_path": trial.evaluation_result_path,
        "log_path": trial.log_path,
        "validation_metrics": dict(trial.validation_metrics),
        "artifacts": list(trial.artifacts),
        "error_type": trial.error_type,
        "error_message": trial.error_message,
        "skip_reason": trial.skip_reason,
        "started_at": trial.started_at,
        "finished_at": trial.finished_at,
    }


def build_campaign_result(record: CampaignRecord, *,
                          plan: dict[str, Any] | None = None,
                          run_spec: CampaignRunSpec | dict | None = None,
                          policy: CampaignPolicy | None = None) -> dict[str, Any]:
    """Pure function: same record in, same JSON out (no clock, no I/O)."""
    arms: dict[str, dict[str, Any]] = {}
    for trial in record.trials:
        arms.setdefault(trial.arm_id, {})[f"seed_{trial.seed}"] = _trial_fact(trial)
    spec_dict = (run_spec.model_dump() if isinstance(run_spec, CampaignRunSpec)
                 else dict(run_spec or {}))
    return {
        "schema_version": MANIFEST_VERSION,
        "campaign_id": record.campaign_id,
        "plan_id": record.plan_id,
        "objective_id": record.objective_id,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "dataset_version": record.dataset_version,
        "evaluation_split": record.evaluation_split,
        "baseline_config_sha256": record.baseline_config_sha256,
        "plan": plan if plan is not None else record.plan,
        "run_spec": spec_dict,
        "policy": policy.snapshot() if policy else record.policy_snapshot,
        "arms": arms,
        "summary": {
            "total_trials": len(record.trials),
            "succeeded_trials": record.completed_trials,
            "failed_trials": record.failed_trials,
            "skipped_trials": record.skipped_trials,
        },
        "scope_note": (
            "P1-04 records execution facts only. Paired comparison, "
            "guardrails and promotion are P1-05/P1-06."
        ),
    }


def write_manifest(record: CampaignRecord, *, campaign_dir: str | Path,
                   store: CampaignStore,
                   plan: dict[str, Any] | None = None,
                   run_spec: CampaignRunSpec | dict | None = None,
                   policy: CampaignPolicy | None = None) -> dict[str, str]:
    """Write campaign.json / campaign_events.jsonl / campaign_result.json.

    Checkpoints and worker logs stay in the engine's own run directories; only
    small, hash-traceable manifests are copied here.
    """
    root = Path(campaign_dir)
    root.mkdir(parents=True, exist_ok=True)

    campaign_json = root / "campaign.json"
    campaign_json.write_text(
        json.dumps(record.model_dump(mode="json"), indent=2,
                   ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    events_jsonl = root / "campaign_events.jsonl"
    events_jsonl.write_text(
        "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                for event in store.list_events(record.campaign_id)),
        encoding="utf-8",
    )

    result_json = root / "campaign_result.json"
    result_json.write_text(
        json.dumps(build_campaign_result(record, plan=plan, run_spec=run_spec,
                                         policy=policy),
                   indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "campaign_json": str(campaign_json),
        "campaign_events": str(events_jsonl),
        "campaign_result": str(result_json),
    }
