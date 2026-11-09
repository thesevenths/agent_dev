"""Shared P1-06 test harness: injected fakes, NO engine and NO LLM.

These doubles deliberately implement only the REAL P1-04/P1-05 surface used by
the service (``run``/``resume``/``get_status``/``load_campaign``, ``compare``,
``list_reports``/``load``). The production modules are exercised unchanged.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from research_runtime.campaign.schemas import CampaignRecord
from research_runtime.research_policy import ResearchDomainPolicy
from research_runtime.research_service import ResearchDependencies, ResearchService
from research_runtime.research_store import ResearchStores
from research_runtime.schemas import ExperimentResult, FailureReport

OBJECTIVE_ID = "obj_p106"
SOURCE_EXPERIMENT_ID = "exp_baseline"
HYPOTHESIS_ID = "H_lr"
PLAN_ID = "plan_p106"
PRIMARY_METRIC = "mse@1"
DATASET_VERSION = "ds_frozen_v1"


def operator_hypothesis(*, intervention: dict[str, Any] | None = None,
                         hypothesis_id: str = HYPOTHESIS_ID,
                         source_experiment_id: str = SOURCE_EXPERIMENT_ID,
                         evidence_refs: list[str] | None = None) -> dict[str, Any]:
    """A structured, human-authored hypothesis (still Critic-reviewed later)."""
    return {
        "hypothesis_id": hypothesis_id,
        "title": "smaller step size",
        "statement": "a smaller optimisation step reduces one-step error",
        "mechanism": "optimisation step size",
        "evidence_refs": evidence_refs or [
            f"experiment:{source_experiment_id}:metric:{PRIMARY_METRIC}"],
        "intervention": intervention if intervention is not None
                         else {"train.lr": 0.0008},
        "predicted_effects": [{"metric": PRIMARY_METRIC,
                               "direction": "decrease"}],
        "falsification_criteria": [
            "no improvement on the held-out validation split over paired seeds"],
        "confidence": 0.5,
    }


class FakeCampaignStore:
    def __init__(self):
        self.records: dict[str, CampaignRecord] = {}

    def load_campaign(self, campaign_id: str):
        return self.records.get(campaign_id)

    def put(self, record: CampaignRecord) -> None:
        self.records[record.campaign_id] = record


class FakeComparisonReportStore:
    def __init__(self):
        self.items: list[dict[str, Any]] = []
        self.loaded: dict[str, Any] = {}

    def list_reports(self, campaign_id: str, candidate_id: str | None = None):
        return [item for item in self.items
                if candidate_id is None or item["candidate_id"] == candidate_id]

    def load(self, campaign_id: str, comparison_id: str):
        return self.loaded.get(comparison_id)


@dataclass
class FakeComparisonReport:
    comparison_id: str = "cmp_1"
    candidate_id: str = f"cand_{HYPOTHESIS_ID}"
    decision: str = "rejected"
    decision_rule: str = "R3_guardrail_hard_violation"
    n_pairs: int = 2
    mean_delta: float = -0.01
    guardrail_violations: list[str] | None = None

    def __post_init__(self):
        if self.guardrail_violations is None:
            self.guardrail_violations = ["cosine@1"]

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {"comparison_id": self.comparison_id,
                "candidate_id": self.candidate_id, "decision": self.decision,
                "decision_rule": self.decision_rule, "n_pairs": self.n_pairs,
                "mean_delta": self.mean_delta,
                "guardrail_violations": self.guardrail_violations}


class FakeRunner:
    def __init__(self, store: FakeCampaignStore):
        self.store = store
        self.run_calls: list[dict[str, Any]] = []
        self.resume_calls: list[str] = []

    def run(self, plan, run_spec) -> CampaignRecord:
        self.run_calls.append({"plan": plan, "run_spec": run_spec})
        record = CampaignRecord(
            campaign_id=run_spec.campaign_id, plan_id=plan.plan_id,
            objective_id=plan.objective_id, status="completed",
            dataset_version=run_spec.dataset_version,
            evaluation_split=run_spec.evaluation_split,
            plan=plan.model_dump(mode="json"),
            run_spec=run_spec.model_dump(mode="json"),
        )
        self.store.put(record)
        return record

    def resume(self, campaign_id: str) -> CampaignRecord:
        self.resume_calls.append(campaign_id)
        record = self.store.load_campaign(campaign_id)
        if record is None:
            raise KeyError(campaign_id)
        record.status = "completed"
        self.store.put(record)
        return record

    def get_status(self, campaign_id: str) -> dict[str, Any]:
        record = self.store.load_campaign(campaign_id)
        return {"campaign_id": campaign_id,
                "status": record.status if record else "unknown",
                "summary": {"total_trials": 4}}


class FakeComparator:
    def __init__(self, store: FakeComparisonReportStore, *,
                 reports: list[FakeComparisonReport] | None = None):
        self.store = store
        self.reports = reports
        self.calls: list[str] = []

    def compare(self, campaign_id: str):
        self.calls.append(campaign_id)
        if self.reports is not None:
            return self.reports
        report = FakeComparisonReport()
        self.store.items.append({"comparison_id": report.comparison_id,
                                 "candidate_id": report.candidate_id,
                                 "decision": report.decision})
        self.store.loaded[report.comparison_id] = report
        return [report]


def default_domain(**overrides: Any) -> ResearchDomainPolicy:
    kwargs: dict[str, Any] = dict(
        domain="unit_test", baseline_config_path="configs/baseline.yaml",
        dataset_version=DATASET_VERSION,
        allowed_dataset_versions=(DATASET_VERSION,),
        allowed_config_keys=("train.lr", "model.width"),
        intervention_policy={"dynamics_error_accumulation": {"train.lr": 0.0008}},
        primary_metric=PRIMARY_METRIC, paired_seeds=(42, 43),
        evaluation_split="validation", max_candidates=2, max_seeds=5,
        max_jobs=30, max_total_seconds=3600.0, job_timeout_s=300.0,
        max_campaign_rounds=2,
    )
    kwargs.update(overrides)
    return ResearchDomainPolicy(**kwargs)


def fake_evidence(experiment_id: str):
    result = ExperimentResult(
        experiment_id=experiment_id, job_id="job_baseline",
        status="succeeded", stage="evaluate",
        metrics={PRIMARY_METRIC: 0.1, "mse@2": 0.2, "mse@3": 0.3},
    )
    report = FailureReport(
        experiment_id=experiment_id, job_id="job_baseline",
        category="dynamics_error_accumulation",
        evidence={"rule": "horizon_degradation"},
        recommended_action="compare on the same validation split",
    )
    return result, [report]


def fake_dataset_verifier(base_config: str, dataset_version: str):
    return {"verified": True, "dataset_version": dataset_version}


class Harness:
    """Build a service with injected fakes and drive it up to human approval."""

    def __init__(self, *, domain: ResearchDomainPolicy | None = None,
                 evidence_provider=None, dataset_verifier=None,
                 comparator_reports=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.stores = ResearchStores(self.root / "research.sqlite")
        self.campaign_store = FakeCampaignStore()
        self.report_store = FakeComparisonReportStore()
        self.runner = FakeRunner(self.campaign_store)
        self.comparator = FakeComparator(
            self.report_store, reports=comparator_reports)
        self.domain = domain or default_domain(
            report_dir=str(self.root / "reports"))
        self.deps = ResearchDependencies(
            stores=self.stores, domain=self.domain,
            campaign_store=self.campaign_store, report_store=self.report_store,
            runner_factory=lambda: self.runner,
            comparator_factory=lambda: self.comparator,
            evidence_provider=evidence_provider or fake_evidence,
            dataset_verifier=dataset_verifier or fake_dataset_verifier,
            campaign_root=str(self.root / "campaigns"),
        )
        self.service = ResearchService(self.deps)

    def close(self) -> None:
        self.tmp.cleanup()

    # ------------------------------------------------------------- chain
    def chain(self, *, hypothesis: dict[str, Any] | None = None,
              plan_id: str = PLAN_ID, grant: bool = True,
              approved_by: str = "alice", source: str = "human"):
        """Objective -> evidence -> hypothesis -> Critic -> plan -> approval."""
        session = self.service.start_research_session(
            objective_id=OBJECTIVE_ID, description="reduce prediction error")
        proposed = self.service.propose_research_hypotheses(
            objective_id=OBJECTIVE_ID,
            source_experiment_id=SOURCE_EXPERIMENT_ID,
            hypotheses_json=json.dumps([hypothesis or operator_hypothesis()]))
        reviewed = self.service.review_research_hypotheses(
            hypothesis_ids_json=json.dumps([HYPOTHESIS_ID]))
        plan = self.service.create_research_plan(
            objective_id=OBJECTIVE_ID,
            hypothesis_ids_json=json.dumps([HYPOTHESIS_ID]), plan_id=plan_id)
        # Some negative tests intentionally stop at the Critic gate.
        request = (self.service.request_plan_approval(plan_id=plan_id)
                  if plan.get("ok") else {"ok": False,
                                         "error_code": "no_plan_created"})
        approval_id = request.get("approval_id")
        if grant and approval_id:
            self.stores.approvals.grant(
                approval_id, approved_by=approved_by, source=source,
                current_resource_sha256=plan["plan_sha256"])
        return {"session": session, "proposed": proposed, "reviewed": reviewed,
                "plan": plan, "request": request, "approval_id": approval_id}

    def run(self, approval_id: str, plan_id: str = PLAN_ID):
        return self.service.run_research_campaign(
            plan_id=plan_id, approval_id=approval_id)

    def replace_domain(self, **overrides: Any) -> None:
        """Rebuild the service with a modified injected policy."""
        self.domain = replace(self.domain, **overrides)
        self.deps = ResearchDependencies(
            stores=self.stores, domain=self.domain,
            campaign_store=self.campaign_store, report_store=self.report_store,
            runner_factory=lambda: self.runner,
            comparator_factory=lambda: self.comparator,
            evidence_provider=self.deps.evidence_provider,
            dataset_verifier=self.deps.dataset_verifier,
            campaign_root=self.deps.campaign_root,
        )
        self.service = ResearchService(self.deps)
