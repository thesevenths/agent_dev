"""Shared P1-05 test infrastructure: REAL store / comparator / report store.

Only the experiment EXECUTION is replaced by hand-built TrialRecords. The
SQLite CampaignStore, the immutable ComparisonReportStore and the whole
comparator + statistics + guardrail stack are the real production code, so the
tests exercise the actual persistence, hashing and decision paths.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_runtime.campaign.campaign_store import CampaignStore
from research_runtime.campaign.schemas import CampaignRecord, TrialRecord
from research_runtime.comparison import (
    CampaignComparator,
    ComparisonPolicy,
    ComparisonReportStore,
)
from research_runtime.schemas import CandidateExperiment, ExperimentPlan

DATASET_VERSION = "ds-v1"
PRIMARY_METRIC = "mse@1"
BASE_HASH = "BASE_CONFIG_SHA256"


def make_trial(*, seed: int, arm_type: str, arm_id: str, value: float,
               status: str = "succeeded", campaign_id: str = "cmp_001",
               dataset_version: str = DATASET_VERSION,
               evaluation_split: str = "validation",
               primary_metric: str = PRIMARY_METRIC,
               config_sha256: str | None = None,
               baseline_config_sha256: str | None = BASE_HASH,
               train_run_id: str | None = None,
               evaluation_run_id: str | None = None,
               guardrail_metrics: dict[str, float] | None = None,
               error_message: str | None = None) -> TrialRecord:
    tid = f"{arm_id}_seed_{seed}"
    run = train_run_id or f"run_{tid}"
    eval_run = evaluation_run_id if evaluation_run_id is not None else run
    metrics = {primary_metric: value}
    if guardrail_metrics:
        metrics.update(guardrail_metrics)
    return TrialRecord(
        trial_id=tid, campaign_id=campaign_id, arm_type=arm_type,
        arm_id=arm_id, seed=seed,
        hypothesis_id=("H1" if arm_type == "candidate" else None),
        config_path=f"/tmp/{tid}/config.yaml",
        config_relpath=f"outputs/research_campaigns/cmp_001/{tid}/config.yaml",
        config_sha256=config_sha256 or f"cfg_{tid}",
        baseline_config_sha256=baseline_config_sha256,
        dataset_version=dataset_version,
        train_job_id=f"train_{tid}", train_run_id=run,
        evaluation_job_id=f"eval_{tid}",
        evaluation_split=evaluation_split, primary_metric=primary_metric,
        status=status,
        validation_metrics=metrics,
        evaluation_spec={"run_id": eval_run, "parent_run_id": run,
                         "split": evaluation_split, "stage": "evaluate"},
        error_message=error_message,
    )


def make_plan(seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
              n_candidates: int = 1, *,
              guardrails: dict[str, dict[str, Any]] | None = None,
              metric_direction: str = "minimize",
              primary_metric: str = PRIMARY_METRIC,
              dataset_version: str = DATASET_VERSION) -> ExperimentPlan:
    candidates = [CandidateExperiment(
        candidate_id=f"cand_{index}", hypothesis_id=f"H{index}",
        config_overrides={"generic.knob": index + 1},
        seeds=list(seeds), dataset_version=dataset_version,
    ) for index in range(n_candidates)]
    return ExperimentPlan(
        plan_id="plan_001", objective_id="obj_001",
        baseline_experiment_id="baseline_001",
        dataset_version=dataset_version, candidates=candidates,
        primary_metric=primary_metric, metric_direction=metric_direction,
        paired_seeds=list(seeds), guardrails=guardrails or {},
    )


def make_record(trials: list[TrialRecord], plan: ExperimentPlan, *,
                campaign_id: str = "cmp_001",
                dataset_version: str = DATASET_VERSION,
                evaluation_split: str = "validation",
                baseline_config_sha256: str | None = BASE_HASH) -> CampaignRecord:
    return CampaignRecord(
        campaign_id=campaign_id, plan_id=plan.plan_id,
        objective_id=plan.objective_id, status="completed",
        dataset_version=dataset_version,
        baseline_config_sha256=baseline_config_sha256,
        evaluation_split=evaluation_split,
        trials=list(trials),
        plan=plan.model_dump(mode="json"),
        run_spec={"campaign_id": campaign_id, "plan_id": plan.plan_id,
                  "objective_id": plan.objective_id,
                  "baseline_config_path": "configs/base.yaml",
                  "dataset_version": dataset_version,
                  "evaluation_split": evaluation_split},
    )


def paired_trials(seeds: tuple[int, ...], baseline: list[float],
                  candidate: list[float], *,
                  candidate_id: str = "cand_0",
                  baseline_guard: list[dict[str, float]] | None = None,
                  candidate_guard: list[dict[str, float]] | None = None,
                  **kwargs: Any) -> list[TrialRecord]:
    out: list[TrialRecord] = []
    for index, seed in enumerate(seeds):
        out.append(make_trial(seed=seed, arm_type="baseline",
                              arm_id="baseline", value=baseline[index],
                              guardrail_metrics=(baseline_guard or [{}] * len(seeds))[index],
                              **kwargs))
        out.append(make_trial(seed=seed, arm_type="candidate",
                              arm_id=candidate_id, value=candidate[index],
                              guardrail_metrics=(candidate_guard or [{}] * len(seeds))[index],
                              **kwargs))
    return out


class ComparisonTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "campaigns.sqlite"
        self.campaign_root = self.root / "outputs" / "research_campaigns"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def store(self) -> CampaignStore:
        return CampaignStore(self.db_path)

    def report_store(self) -> ComparisonReportStore:
        return ComparisonReportStore(self.campaign_root)

    def persist(self, record: CampaignRecord,
                store: CampaignStore | None = None) -> CampaignStore:
        store = store or self.store()
        store.create_campaign(record, event_type="campaign_created")
        return store

    def comparator(self, *, store: CampaignStore | None = None,
                   report_store: ComparisonReportStore | None = None,
                   policy: ComparisonPolicy | None = None,
                   **overrides: Any) -> CampaignComparator:
        return CampaignComparator(
            store or self.store(), report_store or self.report_store(),
            policy=policy or ComparisonPolicy(), **overrides)
