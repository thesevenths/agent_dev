"""P1-05: paired comparator and statistical decision engine.

Strict boundary: this module READS campaign facts and WRITES verdicts. It never
submits a job, never trains a model, never edits a config, never promotes a
ModelVersion and never asks an LLM. ``accepted_for_confirmation`` only means the
candidate earned an independent evaluation on the reserved final split.

Input source (important)
------------------------
The comparator reads ``CampaignStore.load_campaign()`` -> ``CampaignRecord``,
NOT ``campaign_result.json``. Only the SQLite record keeps the full
``TrialRecord.evaluation_spec`` (run_id / parent_run_id / split) and the
per-trial config hashes needed to PROVE that validation reused the trained
checkpoint and that both arms used the same frozen data. The manifest is a
facts-only summary and would make several integrity checks impossible.

Decision rules (FIRST MATCH WINS, exactly as specified for P1-05)
------------------------------------------------------------------
    R1 pairs_incomplete                     -> inconclusive
    R2 insufficient_paired_seeds            -> inconclusive
    R3 guardrail_hard_violation             -> rejected
    R4 guardrail_insufficient_evidence     -> inconclusive
    R5 confidence_interval_regression      -> rejected
    R6 confidence_interval_includes_zero    -> inconclusive
    R7 practical_effect_confirmed           -> accepted_for_confirmation
    R8 no_decisive_rule                     -> inconclusive

Incomplete evidence therefore CAPS the decision at inconclusive rather than
being silently dropped. Guardrail verdicts are still fully recorded in
``guardrail_results`` / ``guardrail_violations`` even when R1/R2 fires first.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable

from ..campaign.campaign_store import CampaignStore
from ..campaign.schemas import CampaignRecord, TrialRecord
from ..schemas import RESERVED_FINAL_SPLITS, ExperimentPlan
from .exceptions import ComparisonInputError, ComparisonIntegrityError
from .guardrails import GuardrailEvaluator
from .report_store import ComparisonReportStore, content_sha256, sha256_bytes
from .schemas import (
    COMPARISON_ALGORITHM_VERSION,
    ComparisonPolicy,
    GuardrailAssessment,
    MissingPair,
    PairedComparisonReport,
    PairedObservation,
)
from .statistics import INTERVAL_METHOD, PairedStatistics, PairedStatisticsEngine

COMPARISON_NAMESPACE = uuid.UUID("b7f3c2d4-9e18-4a6b-8c05-2f6d1a0e7b93")

ConfigHashVerifier = Callable[[TrialRecord], "str | None"]
DatasetEvidence = Callable[[CampaignRecord], "dict[str, str] | None"]


class CampaignComparator:
    def __init__(self, campaign_store: CampaignStore,
                 report_store: ComparisonReportStore | None, *,
                 statistics: Any | None = None,
                 guardrail_evaluator: Any | None = None,
                 policy: ComparisonPolicy | None = None,
                 config_hash_verifier: ConfigHashVerifier | None = None,
                 dataset_evidence: DatasetEvidence | None = None,
                 event_store: Any | None = None,
                 algorithm_version: str = COMPARISON_ALGORITHM_VERSION):
        self.campaign_store = campaign_store
        self.report_store = report_store
        self.statistics = statistics or PairedStatisticsEngine()
        self.guardrail_evaluator = guardrail_evaluator or GuardrailEvaluator()
        self.policy = policy or ComparisonPolicy()
        self.config_hash_verifier = config_hash_verifier
        self.dataset_evidence = dataset_evidence
        self.event_store = event_store if event_store is not None else campaign_store
        self.algorithm_version = algorithm_version
        self.event_errors: list[str] = []

    # ------------------------------------------------------------- public API
    def compare(self, campaign_id: str) -> list[PairedComparisonReport]:
        """Compare EVERY candidate against the SAME baseline over paired seeds."""
        record = self._load(campaign_id)
        plan = self._plan(record)
        prepared: list[dict[str, Any]] = []
        for candidate in plan.candidates:
            try:
                pairs, missing, evidence = self._collect_pairs(record, candidate, plan)
            except ComparisonIntegrityError as exc:
                self._event(record, "comparison_failed",
                            {"candidate_id": candidate.candidate_id,
                             "error": str(exc)})
                continue
            deltas = [p.delta for p in pairs]
            base_stats = self.statistics.summarize(
                deltas, alpha=self.policy.familywise_alpha)
            p_value = (base_stats.p_value_two_sided
                       if base_stats.p_value_two_sided is not None else 1.0)
            prepared.append({
                "candidate": candidate, "pairs": pairs, "missing": missing,
                "evidence": evidence, "deltas": deltas,
                "base_stats": base_stats, "p_value": p_value,
            })
        if not prepared:
            return []

        family = len(prepared)
        holm = None
        if family > 1 and self.policy.multiple_comparison_method == "holm":
            holm = self.statistics.holm_adjust(
                [item["p_value"] for item in prepared],
                familywise_alpha=self.policy.familywise_alpha)

        reports: list[PairedComparisonReport] = []
        for index, item in enumerate(prepared):
            if holm is not None:
                entry = holm.entries[index]
                decision_alpha = entry.adjusted_alpha
                holm_significant = entry.significant
                mc: dict[str, Any] = {
                    "method": "holm", "family_size": family,
                    "familywise_alpha": self.policy.familywise_alpha,
                    "rank": entry.rank, "raw_p_value": entry.p_value,
                    "adjusted_p": entry.adjusted_p,
                    "adjusted_alpha": entry.adjusted_alpha,
                    "holm_significant": holm_significant,
                    "note": ("Holm step-down controls the family-wise error "
                             "rate; the decision CI uses adjusted_alpha, and "
                             "adjusted_p is compared with familywise_alpha"),
                }
            else:
                decision_alpha = self.policy.familywise_alpha
                holm_significant = True   # CI lower bound already implies significance
                mc = {
                    "method": ("holm" if family > 1 else "single_candidate"),
                    "family_size": family,
                    "familywise_alpha": self.policy.familywise_alpha,
                    "adjusted_alpha": decision_alpha,
                    "holm_significant": holm_significant,
                    "note": ("no correction needed for a single candidate"
                             if family == 1 else
                             "correction disabled by injected policy"),
                }
            ci95 = self.statistics.summarize(item["deltas"], alpha=0.05)
            decision_stats = (
                item["base_stats"] if decision_alpha == self.policy.familywise_alpha
                else self.statistics.summarize(item["deltas"], alpha=decision_alpha))
            report = self._build_report(record, plan, item, ci95,
                                        decision_stats, mc, holm_significant)
            reports.append(self._persist(record, report))
        return reports

    def compare_candidate(self, campaign: CampaignRecord | str,
                          candidate_id: str, *,
                          plan: ExperimentPlan | None = None
                          ) -> PairedComparisonReport:
        """Single-candidate comparison (no family-wise correction)."""
        record = self._load(campaign) if isinstance(campaign, str) else campaign
        plan = plan or self._plan(record)
        candidate = next((c for c in plan.candidates
                          if c.candidate_id == candidate_id), None)
        if candidate is None:
            raise ComparisonIntegrityError(
                f"candidate {candidate_id!r} is not in plan {plan.plan_id!r}")
        pairs, missing, evidence = self._collect_pairs(record, candidate, plan)
        deltas = [p.delta for p in pairs]
        alpha = self.policy.familywise_alpha
        base_stats = self.statistics.summarize(deltas, alpha=alpha)
        ci95 = (base_stats if alpha == 0.05
                else self.statistics.summarize(deltas, alpha=0.05))
        mc = {
            "method": "single_candidate", "family_size": 1,
            "familywise_alpha": alpha, "adjusted_alpha": alpha,
            "holm_significant": True,
            "note": "single candidate: no family-wise correction required",
        }
        item = {"candidate": candidate, "pairs": pairs, "missing": missing,
                "evidence": evidence, "deltas": deltas,
                "base_stats": base_stats}
        return self._persist(
            record, self._build_report(record, plan, item, ci95,
                                       base_stats, mc, True))

    # --------------------------------------------------------------- loading
    def _load(self, campaign_id: str) -> CampaignRecord:
        if self.campaign_store is None:
            raise ComparisonIntegrityError("no CampaignStore configured")
        record = self.campaign_store.load_campaign(campaign_id)
        if record is None:
            raise ComparisonIntegrityError(
                f"unknown campaign {campaign_id!r}")
        return record

    @staticmethod
    def _plan(record: CampaignRecord) -> ExperimentPlan:
        if not record.plan:
            raise ComparisonIntegrityError(
                "campaign record has no serialized ExperimentPlan; "
                "pass plan=... to compare_candidate")
        try:
            return ExperimentPlan.model_validate(record.plan)
        except Exception as exc:
            raise ComparisonIntegrityError(
                f"persisted plan is invalid: {exc}") from exc

    # ----------------------------------------------------------- pair collection
    def _collect_pairs(self, record: CampaignRecord, candidate: Any,
                       plan: ExperimentPlan
                       ) -> tuple[list[PairedObservation],
                                  list[MissingPair], dict[str, Any]]:
        pairs: list[PairedObservation] = []
        missing: list[MissingPair] = []
        for seed in plan.paired_seeds:
            baselines = [t for t in record.trials
                        if t.arm_type == "baseline" and t.seed == seed]
            candidates = [t for t in record.trials
                          if t.arm_type == "candidate" and t.seed == seed
                          and t.arm_id == candidate.candidate_id]
            if len(baselines) > 1 or len(candidates) > 1:
                raise ComparisonIntegrityError(
                    f"duplicate trials for seed {seed} "
                    f"(baseline={len(baselines)}, candidate={len(candidates)})")
            b = baselines[0] if baselines else None
            c = candidates[0] if candidates else None
            if b is None or b.status != "succeeded" or \
                    c is None or c.status != "succeeded":
                missing.append(self._missing_pair(seed, b, c))
                continue

            self._validate_trial(record, b, plan, arm="baseline")
            self._validate_trial(record, c, plan, arm="candidate")
            if (b.baseline_config_sha256 and c.baseline_config_sha256
                    and b.baseline_config_sha256 != c.baseline_config_sha256):
                raise ComparisonIntegrityError(
                    f"seed {seed}: baseline_config_sha256 differs between arms; "
                    "the two arms did not start from the same frozen dataset")
            b_value = b.validation_metrics.get(plan.primary_metric)
            c_value = c.validation_metrics.get(plan.primary_metric)
            delta = self._delta(b_value, c_value, plan, seed)
            pairs.append(PairedObservation(
                seed=seed, baseline_value=float(b_value),
                candidate_value=float(c_value), delta=delta,
                baseline_trial_id=b.trial_id, candidate_trial_id=c.trial_id,
                baseline_train_job_id=b.train_job_id,
                baseline_evaluation_job_id=b.evaluation_job_id,
                baseline_train_run_id=b.train_run_id,
                candidate_train_job_id=c.train_job_id,
                candidate_evaluation_job_id=c.evaluation_job_id,
                candidate_train_run_id=c.train_run_id,
                baseline_config_sha256=b.config_sha256,
                candidate_config_sha256=c.config_sha256,
                dataset_version=b.dataset_version,
                evaluation_split=b.evaluation_split or "",
                baseline_guardrail_metrics=dict(b.validation_metrics),
                candidate_guardrail_metrics=dict(c.validation_metrics),
            ))

        evidence = self._evidence(record, pairs)
        return pairs, missing, evidence

    @staticmethod
    def _missing_pair(seed: int, b: TrialRecord | None,
                      c: TrialRecord | None) -> MissingPair:
        def reason(trial: TrialRecord | None, arm: str) -> str:
            if trial is None:
                return f"no {arm} trial recorded for this seed"
            detail = trial.error_message or trial.skip_reason or ""
            return f"{arm} trial status={trial.status}" + (f": {detail}" if detail else "")
        b_ok = b is not None and b.status == "succeeded"
        c_ok = c is not None and c.status == "succeeded"
        if not b_ok and not c_ok:
            detail = "; ".join(filter(None, [reason(b, "baseline"),
                                            reason(c, "candidate")]))
            return MissingPair(seed=seed, arm="both",
                               reason=detail or "neither arm has a trial")
        if not b_ok:
            return MissingPair(seed=seed, arm="baseline",
                               reason=reason(b, "baseline"))
        return MissingPair(seed=seed, arm="candidate",
                           reason=reason(c, "candidate"))

    def _validate_trial(self, record: CampaignRecord, trial: TrialRecord,
                        plan: ExperimentPlan, *, arm: str) -> None:
        def bad(message: str) -> None:
            raise ComparisonIntegrityError(
                f"{arm}/{trial.trial_id}: {message}")
        if trial.campaign_id != record.campaign_id:
            bad("belongs to a different campaign")
        if (trial.dataset_version != record.dataset_version
                or trial.dataset_version != plan.dataset_version):
            bad(f"dataset_version {trial.dataset_version!r} != "
                f"{record.dataset_version!r} / plan {plan.dataset_version!r}")
        if trial.evaluation_split != record.evaluation_split:
            bad(f"evaluation_split {trial.evaluation_split!r} != campaign "
                f"{record.evaluation_split!r}")
        if trial.evaluation_split in RESERVED_FINAL_SPLITS:
            bad(f"used reserved final split {trial.evaluation_split!r}")
        if trial.primary_metric != plan.primary_metric:
            bad(f"primary_metric {trial.primary_metric!r} != "
                f"plan {plan.primary_metric!r}")
        if not (trial.train_job_id and trial.evaluation_job_id
                and trial.train_run_id):
            bad("missing train/evaluation job id or train run_id")
        spec = trial.evaluation_spec
        if not isinstance(spec, dict):
            bad("evaluation_spec missing; checkpoint reuse cannot be proven")
        if spec.get("run_id") != trial.train_run_id:
            bad(f"evaluation run_id {spec.get('run_id')!r} != train "
                f"run_id {trial.train_run_id!r}")
        if spec.get("parent_run_id") != trial.train_run_id:
            bad("evaluation parent_run_id does not match the train run_id")
        if spec.get("split") != record.evaluation_split:
            bad("persisted evaluation spec requested a different split")
        if not trial.config_sha256:
            bad("missing config_sha256")
        if (record.baseline_config_sha256 and trial.baseline_config_sha256
                and trial.baseline_config_sha256
                != record.baseline_config_sha256):
            bad("baseline_config_sha256 differs from the campaign record")
        if self.config_hash_verifier is not None:
            actual = self.config_hash_verifier(trial)
            if actual is not None and actual != trial.config_sha256:
                bad(f"config hash on disk {actual} != recorded "
                    f"{trial.config_sha256}; refusing to compare tampered evidence")

    def _delta(self, b_value: Any, c_value: Any, plan: ExperimentPlan,
               seed: int) -> float:
        if b_value is None or c_value is None:
            raise ComparisonIntegrityError(
                f"seed {seed}: primary metric {plan.primary_metric!r} missing "
                "from a 'succeeded' trial")
        try:
            return self.statistics.normalize(
                [b_value], [c_value], plan.metric_direction)[0]
        except ComparisonInputError as exc:
            raise ComparisonIntegrityError(
                f"seed {seed}: invalid primary metric value ({exc})") from exc

    def _evidence(self, record: CampaignRecord,
                  pairs: list[PairedObservation]) -> dict[str, Any]:
        candidate_hashes = "|".join(p.candidate_config_sha256 for p in pairs)
        split_hashes = None
        if self.dataset_evidence is not None:
            try:
                split_hashes = self.dataset_evidence(record)
            except Exception as exc:
                split_hashes = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "baseline_config_sha256": record.baseline_config_sha256,
            "candidate_config_sha256": (
                sha256_bytes(candidate_hashes.encode("utf-8")) if pairs else None),
            "dataset_version": record.dataset_version,
            "evaluation_split": record.evaluation_split,
            "dataset_split_hashes": split_hashes,
        }

    # ------------------------------------------------------------- decision
    def _decide(self, *, n_pairs: int, n_planned: int,
                missing: list[MissingPair], assessment: GuardrailAssessment,
                ci95: PairedStatistics, decision_stats: PairedStatistics,
                threshold: float, holm_significant: bool
                ) -> tuple[str, str]:
        ci = decision_stats.confidence_interval
        if n_pairs == 0 or n_pairs < n_planned or missing:
            return "inconclusive", "R1_pairs_incomplete"
        if n_pairs < self.policy.min_paired_seeds:
            return "inconclusive", "R2_insufficient_paired_seeds"
        if assessment.has_hard_violation:
            return "rejected", "R3_guardrail_hard_violation"
        if assessment.insufficient:
            return "inconclusive", "R4_guardrail_insufficient_evidence"
        if ci is not None and ci[1] < 0.0:
            return "rejected", "R5_confidence_interval_regression"
        if ci is None or ci[0] <= 0.0 <= ci[1]:
            return "inconclusive", "R6_confidence_interval_includes_zero"
        if (ci[0] > threshold and assessment.all_passed
                and holm_significant):
            return "accepted_for_confirmation", "R7_practical_effect_confirmed"
        return "inconclusive", "R8_no_decisive_rule"

    # ------------------------------------------------------------- report
    def _build_report(self, record: CampaignRecord, plan: ExperimentPlan,
                       item: dict[str, Any], ci95: PairedStatistics,
                       decision_stats: PairedStatistics,
                       mc: dict[str, Any], holm_significant: bool
                       ) -> PairedComparisonReport:
        candidate = item["candidate"]
        pairs: list[PairedObservation] = item["pairs"]
        missing: list[MissingPair] = item["missing"]
        evidence = item["evidence"]
        deltas = item["deltas"]
        baseline_values = [p.baseline_value for p in pairs]
        candidate_values = [p.candidate_value for p in pairs]
        assessment: GuardrailAssessment = self.guardrail_evaluator.evaluate(
            plan.guardrails, pairs)
        threshold, threshold_mode = self.policy.effective_threshold(baseline_values)
        decision, rule = self._decide(
            n_pairs=len(pairs), n_planned=len(plan.paired_seeds),
            missing=missing, assessment=assessment, ci95=ci95,
            decision_stats=decision_stats, threshold=threshold,
            holm_significant=holm_significant)

        notes: list[str] = [
            "delta_convention=positive_is_better; metric_direction="
            f"{plan.metric_direction} comes from ExperimentPlan and is NEVER "
            "guessed from a metric name",
            f"interval_method={INTERVAL_METHOD}; confidence_interval_95 uses "
            "alpha=0.05; decision_confidence_interval uses alpha="
            f"{decision_stats.alpha:.6g} (confidence_level="
            f"{decision_stats.confidence_level:.6g})",
            "Assumptions: paired seeds are independent training runs; deltas "
            "are approximately normal; the interval quantifies TRAINING-SEED "
            "uncertainty, not per-sample prediction uncertainty",
            "NO bootstrap: resampling a handful of seeds cannot create "
            "information; a sample-level paired bootstrap answers a different "
            "question and is deliberately deferred",
            f"practical_effect_threshold={threshold:.6g} "
            f"(resolution={threshold_mode}, mode="
            f"{self.policy.practical_effect_mode}); "
            f"min_paired_seeds={self.policy.min_paired_seeds} is an engineering "
            "gate, not a statistical law",
            "Guardrails are checked per paired seed; a missing/non-finite "
            "metric is never a pass; relative rules must declare direction",
            f"decision_rule={rule}",
            "accepted_for_confirmation is NOT promotion: no ModelRegistry "
            "write occurs in this module",
        ]
        if missing:
            notes.append(f"{len(missing)} planned pair(s) missing; reasons are "
                         "recorded in missing_pairs, none were dropped silently")
        if assessment.violations:
            notes.append("non-passing guardrails: "
                         + "; ".join(assessment.violations))
        if mc.get("family_size", 1) > 1:
            notes.append("Multiple candidates: " + mc["method"]
                         + " family-wise error control applied; "
                         + f"holm_significant={holm_significant}")
        if evidence.get("dataset_split_hashes") is None:
            notes.append("dataset split hashes not supplied by an injected "
                         "dataset_evidence provider")
        notes.extend(ci95.notes)
        if decision_stats is not ci95:
            notes.extend(decision_stats.notes)
        notes = list(dict.fromkeys(notes))

        fingerprint = {
            "campaign_id": record.campaign_id,
            "plan_id": plan.plan_id,
            "candidate_id": candidate.candidate_id,
            "dataset_version": record.dataset_version,
            "evaluation_split": record.evaluation_split,
            "primary_metric": plan.primary_metric,
            "metric_direction": plan.metric_direction,
            "guardrails": plan.guardrails,
            "algorithm_version": self.algorithm_version,
            "policy": self.policy.snapshot(),
            "observations": [
                [p.seed, p.baseline_value, p.candidate_value,
                 p.baseline_config_sha256, p.candidate_config_sha256,
                 p.baseline_train_run_id, p.candidate_train_run_id]
                for p in pairs],
            "missing": [m.model_dump(mode="json") for m in missing],
        }
        comparison_id = uuid.uuid5(
            COMPARISON_NAMESPACE,
            json.dumps(fingerprint, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")),
        ).hex

        refs: list[str] = []
        for p in pairs:
            refs.append(f"campaign:{record.campaign_id}:trial:"
                        f"{p.baseline_trial_id}:evaluation_job:"
                        f"{p.baseline_evaluation_job_id}")
            refs.append(f"campaign:{record.campaign_id}:trial:"
                        f"{p.candidate_trial_id}:evaluation_job:"
                        f"{p.candidate_evaluation_job_id}")

        report = PairedComparisonReport(
            comparison_id=comparison_id,
            campaign_id=record.campaign_id,
            plan_id=plan.plan_id,
            candidate_id=candidate.candidate_id,
            primary_metric=plan.primary_metric,
            baseline_values=baseline_values,
            candidate_values=candidate_values,
            paired_deltas=deltas,
            mean_delta=ci95.mean_delta,
            median_delta=ci95.median_delta,
            confidence_interval_95=ci95.confidence_interval,
            decision_confidence_interval=decision_stats.confidence_interval,
            confidence_level=decision_stats.confidence_level,
            interval_method=INTERVAL_METHOD,
            std_delta=ci95.std_delta,
            n_planned=len(plan.paired_seeds),
            n_pairs=len(pairs),
            metric_direction=plan.metric_direction,
            delta_convention="positive_is_better",
            practical_effect_threshold=threshold,
            practical_effect_mode=self.policy.practical_effect_mode,
            decision_rule=rule,
            multiple_comparison=mc,
            guardrail_violations=assessment.hard_violations,
            guardrail_results=[r.model_dump(mode="json")
                               for r in assessment.results],
            missing_pairs=[m.model_dump(mode="json") for m in missing],
            paired_observations=[p.model_dump(mode="json") for p in pairs],
            evidence_refs=refs,
            comparison_algorithm_version=self.algorithm_version,
            baseline_config_sha256=evidence.get("baseline_config_sha256"),
            candidate_config_sha256=evidence.get("candidate_config_sha256"),
            dataset_version=record.dataset_version,
            evaluation_split=record.evaluation_split,
            dataset_split_hashes=evidence.get("dataset_split_hashes"),
            decision=decision,                 # type: ignore[arg-type]
            notes=notes,
        )
        return report.model_copy(update={"report_sha256": content_sha256(report)})

    # ------------------------------------------------------------- persistence
    def _persist(self, record: CampaignRecord,
                 report: PairedComparisonReport) -> PairedComparisonReport:
        payload: dict[str, Any] = {
            "comparison_id": report.comparison_id,
            "candidate_id": report.candidate_id,
            "decision": report.decision,
            "decision_rule": report.decision_rule,
            "n_pairs": report.n_pairs,
            "n_planned": report.n_planned,
            "algorithm_version": report.comparison_algorithm_version,
            "report_sha256": report.report_sha256,
        }
        if self.report_store is not None:
            artifact = self.report_store.save(report,
                                              campaign_id=record.campaign_id)
            # Do NOT mutate the report after the digest was computed: the
            # persisted bytes, report_sha256 and the returned object must agree.
            payload.update({"report_path": artifact.path,
                            "file_sha256": artifact.file_sha256,
                            "created_new": artifact.created_new})
        self._event(record, "comparison_completed", payload)
        return report

    def _event(self, record: CampaignRecord, event_type: str,
               payload: dict[str, Any]) -> None:
        store = self.event_store
        if store is None or not hasattr(store, "append_event"):
            return
        try:
            store.append_event(record.campaign_id, event_type, payload=payload)
        except Exception as exc:   # an audit-log failure must not erase a report
            self.event_errors.append(f"{event_type}: {type(exc).__name__}: {exc}")
