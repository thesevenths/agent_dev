"""P1-04: campaign lifecycle -- preflight, execute, pause, resume, cancel.

Boundary (explicit)
-------------------
This layer EXECUTES an approved ExperimentPlan. It never proposes a
hypothesis, never edits a prompt, never calls an LLM, never compares
baseline against candidate, never promotes a model and never touches the
ModelRegistry. Those are P1-02, P1-05 and P1-06 respectively. A campaign is
successful when the planned experiments were executed and recorded
trustworthily -- NOT when a candidate scored better.

Serial, seed-outer execution is enforced. Every state transition is written
to SQLite with an append-only event before the corresponding engine call.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..candidate_config import (
    CandidateConfigBuilder,
    CandidateConfigError,
    sha256_file,
)
from ..experiment_manager import ExperimentManager
from ..schemas import ExperimentPlan
from .campaign_store import CampaignStore
from .exceptions import (
    CampaignIntegrityError,
    CampaignPreflightError,
    CampaignStoreError,
    RecoverableCampaignError,
)
from .manifest import write_manifest
from .policy import CampaignPolicy
from .scheduler import ScheduleItem, build_schedule
from .schemas import (
    CAMPAIGN_TERMINAL_STATUSES,
    TRIAL_TERMINAL_STATUSES,
    CampaignRecord,
    CampaignRunSpec,
    TrialRecord,
    _now,
)
from .trial_runner import TrialRunner


class CampaignRunner:
    def __init__(self, manager: ExperimentManager,
                 config_builder: CandidateConfigBuilder,
                 store: CampaignStore, policy: CampaignPolicy, *,
                 trial_runner: TrialRunner | None = None,
                 campaign_root: str | Path | None = None,
                 clock: Any = time.monotonic):
        self.manager = manager
        self.config_builder = config_builder
        self.store = store
        self.policy = policy
        self.trial_runner = trial_runner or TrialRunner(manager, policy, store)
        self.clock = clock
        project = getattr(config_builder, "project_dir", None)
        if campaign_root is not None:
            self.campaign_root = Path(campaign_root)
        elif project is not None:
            self.campaign_root = (Path(project)
                                  / getattr(config_builder, "campaign_root",
                                            "outputs/research_campaigns"))
        else:
            raise CampaignPreflightError(
                "campaign_root is required when the config builder has no "
                "project_dir")

    # ------------------------------------------------------------ preflight
    def _preflight(self, plan: ExperimentPlan,
                   run_spec: CampaignRunSpec) -> list[ScheduleItem]:
        """Reject unsafe plans BEFORE the first job is submitted."""
        if not isinstance(plan, ExperimentPlan):
            raise CampaignPreflightError("plan must be a validated ExperimentPlan")
        if run_spec.plan_id != plan.plan_id:
            raise CampaignPreflightError("run_spec.plan_id != plan.plan_id")
        if run_spec.objective_id != plan.objective_id:
            raise CampaignPreflightError("objective_id mismatch between plan and run_spec")
        if run_spec.dataset_version != plan.dataset_version:
            raise CampaignPreflightError("dataset_version mismatch between plan and run_spec")

        approved = set(run_spec.approved_hypothesis_ids)
        if not approved:
            raise CampaignPreflightError(
                "no approved_hypothesis_ids supplied; P1-04 may only execute "
                "plans whose hypotheses passed the P1-02 critic")
        for candidate in plan.candidates:
            if candidate.hypothesis_id not in approved:
                raise CampaignPreflightError(
                    f"candidate {candidate.candidate_id!r} references "
                    f"unapproved hypothesis {candidate.hypothesis_id!r}")

        if len(plan.candidates) > self.policy.max_candidates:
            raise CampaignPreflightError("too many candidates for policy")
        seeds = list(plan.paired_seeds)
        if not seeds or len(seeds) > self.policy.max_seeds:
            raise CampaignPreflightError("paired_seeds is empty or exceeds policy")
        if len(set(seeds)) != len(seeds):
            raise CampaignPreflightError("paired_seeds must be unique")

        split = run_spec.evaluation_split
        if split != self.policy.evaluation_split:
            raise CampaignPreflightError(
                f"evaluation_split {split!r} is not the policy's allowed "
                f"selection split {self.policy.evaluation_split!r}")
        if split in self.policy.reserved_final_splits:
            raise CampaignPreflightError(
                f"evaluation_split {split!r} is reserved for final confirmation")
        for candidate in plan.candidates:
            if candidate.evaluation_split != split:
                raise CampaignPreflightError(
                    f"candidate {candidate.candidate_id!r} requests split "
                    f"{candidate.evaluation_split!r}, plan uses {split!r}")
            if candidate.evaluation_split in self.policy.reserved_final_splits:
                raise CampaignPreflightError("a candidate may not select on the final split")

        if run_spec.max_parallel != 1 or self.policy.max_parallel != 1:
            raise CampaignPreflightError(
                "P1-04 is serial; max_parallel > 1 is not implemented")
        expected = self.policy.expected_jobs(len(plan.candidates), len(seeds))
        budget = min(run_spec.max_jobs, self.policy.max_jobs)
        if expected > budget:
            raise CampaignPreflightError(
                f"campaign needs {expected} jobs, budget allows {budget}")

        # Baseline config must exist and the frozen dataset must verify on disk
        # (version + every split hash), not merely carry a matching label.
        base = Path(run_spec.baseline_config_path)
        if not base.is_absolute():
            base = Path(self.config_builder.project_dir) / base
        if not base.is_file():
            raise CampaignPreflightError(f"baseline config not found: {base}")
        try:
            frozen = self.config_builder.verify_dataset(
                run_spec.baseline_config_path, run_spec.dataset_version)
        except CandidateConfigError as exc:
            raise CampaignPreflightError(f"frozen dataset check failed: {exc}") from exc
        if not frozen.get("verified"):
            raise CampaignPreflightError("frozen dataset could not be verified")

        adapter = getattr(self.manager, "adapter", None)
        stages = tuple(getattr(adapter, "stages", ()) or ())
        if stages:
            for stage in (self.policy.train_stage, self.policy.evaluate_stage):
                if stage not in stages:
                    raise CampaignPreflightError(
                        f"adapter does not support stage {stage!r}")
        try:
            health = adapter.health_check() if adapter is not None else {"ok": True}
        except Exception as exc:
            raise CampaignPreflightError(
                f"adapter health_check failed ({type(exc).__name__}: {exc})"
            ) from exc
        if health.get("ok") is False:
            raise CampaignPreflightError(f"adapter unhealthy: {health}")
        return build_schedule(seeds, plan.candidates)

    # -------------------------------------------------------------- configs
    def _campaign_dir(self, campaign_id: str) -> Path:
        return self.campaign_root / campaign_id

    def _ensure_config(self, item: ScheduleItem, plan: ExperimentPlan,
                       run_spec: CampaignRunSpec,
                       expected_hash: str | None) -> tuple[str, str, str, str]:
        """Return (config_relpath, config_sha256, baseline_sha256, config_path).

        Existing directories are REUSED (never overwritten) but their config
        hash must match the persisted one -- resume with tampered configs is
        an integrity error, not a silently different experiment.
        """
        directory = (self._campaign_dir(run_spec.campaign_id)
                     / item.trial_id).resolve()
        root = self.campaign_root.resolve()
        try:
            directory.relative_to(root)
        except ValueError as exc:
            raise CampaignIntegrityError(
                f"config directory escapes campaign root: {directory}") from exc
        if directory.is_dir():
            cfg = directory / "config.yaml"
            meta = directory / "metadata.json"
            if not cfg.is_file() or not meta.is_file():
                raise CampaignIntegrityError(
                    f"incomplete config artifacts in {directory}")
            actual = sha256_file(cfg)
            try:
                metadata = json.loads(meta.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise CampaignIntegrityError(
                    f"unreadable config metadata: {exc}") from exc
            if actual != metadata.get("candidate_config_sha256"):
                raise CampaignIntegrityError(
                    f"{item.trial_id}: config.yaml no longer matches its metadata")
            if expected_hash and actual != expected_hash:
                raise CampaignIntegrityError(
                    f"{item.trial_id}: config hash {actual} != persisted "
                    f"{expected_hash}; refusing to resume a different experiment")
            return (
                cfg.relative_to(Path(self.config_builder.project_dir)).as_posix(),
                actual, str(metadata.get("baseline_config_sha256") or ""),
                str(cfg),
            )

        try:
            if item.arm_type == "baseline":
                artifact = self.config_builder.build_baseline(
                    base_config=run_spec.baseline_config_path,
                    campaign_id=run_spec.campaign_id,
                    seed=item.seed,
                    dataset_version=run_spec.dataset_version,
                )
            else:
                artifact = self.config_builder.build(
                    base_config=run_spec.baseline_config_path,
                    campaign_id=run_spec.campaign_id,
                    candidate_id=str(item.candidate_id),
                    hypothesis_id=str(item.hypothesis_id),
                    seed=item.seed,
                    overrides=item.overrides,
                    dataset_version=run_spec.dataset_version,
                )
        except CandidateConfigError as exc:
            raise CampaignPreflightError(
                f"config generation failed for {item.trial_id}: {exc}") from exc
        return (artifact.config_relpath, artifact.candidate_config_sha256,
                artifact.baseline_config_sha256, str(artifact.config_path))

    def _make_trial(self, item: ScheduleItem, plan: ExperimentPlan,
                    run_spec: CampaignRunSpec,
                    config: tuple[str, str, str, str]) -> TrialRecord:
        relpath, digest, base_digest, cfg_path = config
        return TrialRecord(
            trial_id=item.trial_id,
            campaign_id=run_spec.campaign_id,
            arm_type=item.arm_type,
            arm_id=item.arm_id,
            seed=item.seed,
            hypothesis_id=item.hypothesis_id,
            config_path=cfg_path,
            config_relpath=relpath,
            config_sha256=digest,
            baseline_config_sha256=base_digest,
            dataset_version=run_spec.dataset_version,
            evaluation_split=run_spec.evaluation_split,
            primary_metric=plan.primary_metric,
            status="config_ready",
        )

    # ------------------------------------------------------------- lifecycle
    def run(self, plan: ExperimentPlan,
            run_spec: CampaignRunSpec) -> CampaignRecord:
        existing = self.store.load_campaign(run_spec.campaign_id)
        if existing is not None:
            if run_spec.resume:
                return self.resume(run_spec.campaign_id)
            self.store.append_event(run_spec.campaign_id,
                                    "duplicate_run_ignored",
                                    payload={"plan_id": run_spec.plan_id})
            return existing                      # idempotent, no new jobs

        schedule = self._preflight(plan, run_spec)
        record = CampaignRecord(
            campaign_id=run_spec.campaign_id,
            plan_id=plan.plan_id,
            objective_id=plan.objective_id,
            status="preflighting",
            dataset_version=run_spec.dataset_version,
            evaluation_split=run_spec.evaluation_split,
            plan=plan.model_dump(mode="json"),
            run_spec=run_spec.model_dump(mode="json"),
            policy_snapshot=self.policy.snapshot(),
        )
        self.store.create_campaign(record, payload={"plan_id": plan.plan_id})

        try:
            for item in schedule:
                config = self._ensure_config(item, plan, run_spec, None)
                trial = self._make_trial(item, plan, run_spec, config)
                self.store.save_trial(record, trial, event_type="config_ready",
                                      payload={"config_sha256": trial.config_sha256,
                                               "config_path": trial.config_relpath})
                if record.baseline_config_sha256 is None:
                    record.baseline_config_sha256 = trial.baseline_config_sha256
        except CampaignStoreError:
            raise
        except (CampaignPreflightError, CampaignIntegrityError) as exc:
            record.status = "failed"
            self.store.save_campaign(record, event_type="config_generation_failed",
                                     payload={"error": str(exc)})
            return record

        record.status = "running"
        self.store.save_campaign(record, event_type="campaign_running",
                                 payload={"trials": len(record.trials)})
        return self._execute(record, plan, run_spec)

    def resume(self, campaign_id: str) -> CampaignRecord:
        record = self.store.load_campaign(campaign_id)
        if record is None:
            raise CampaignPreflightError(f"unknown campaign {campaign_id!r}")
        if record.status in CAMPAIGN_TERMINAL_STATUSES:
            self.store.append_event(campaign_id, "resume_ignored_terminal",
                                    payload={"status": record.status})
            return record
        plan = ExperimentPlan.model_validate(record.plan)
        run_spec = CampaignRunSpec.model_validate(record.run_spec)
        schedule = build_schedule(plan.paired_seeds, plan.candidates)
        try:
            for item in schedule:
                existing = record.trial(item.trial_id)
                expected = existing.config_sha256 if existing else None
                config = self._ensure_config(item, plan, run_spec, expected)
                if existing is None:
                    trial = self._make_trial(item, plan, run_spec, config)
                    self.store.save_trial(record, trial,
                                          event_type="config_recovered",
                                          payload={"config_sha256":
                                                   trial.config_sha256})
        except (CampaignPreflightError, CampaignIntegrityError) as exc:
            record.status = "failed"
            self.store.save_campaign(record, event_type="resume_config_integrity_error",
                                     payload={"error": str(exc)})
            return record
        record.status = "running"
        self.store.save_campaign(record, event_type="campaign_resumed")
        return self._execute(record, plan, run_spec)

    def get_status(self, campaign_id: str) -> dict[str, Any]:
        """Read-only. Never submits, never mutates."""
        record = self.store.load_campaign(campaign_id)
        if record is None:
            return {"campaign_id": campaign_id, "status": "unknown"}
        return {
            "campaign_id": campaign_id,
            "status": record.status,
            "updated_at": record.updated_at,
            "dataset_version": record.dataset_version,
            "evaluation_split": record.evaluation_split,
            "summary": {
                "total_trials": len(record.trials),
                "succeeded_trials": record.completed_trials,
                "failed_trials": record.failed_trials,
                "skipped_trials": record.skipped_trials,
            },
        }

    def cancel(self, campaign_id: str) -> CampaignRecord:
        record = self.store.load_campaign(campaign_id)
        if record is None:
            raise CampaignPreflightError(f"unknown campaign {campaign_id!r}")
        if record.status in CAMPAIGN_TERMINAL_STATUSES:
            return record
        record.status = "cancelled"
        self.store.save_campaign(record, event_type="campaign_cancelled",
                                 payload={"note": "no new jobs will be submitted; "
                                                 "already submitted jobs remain recorded"})
        self._write_manifest(record)
        return record

    # -------------------------------------------------------------- internal
    def _execute(self, record: CampaignRecord, plan: ExperimentPlan,
                 run_spec: CampaignRunSpec) -> CampaignRecord:
        schedule = build_schedule(plan.paired_seeds, plan.candidates)
        deadline = self.clock() + run_spec.max_total_seconds
        for item in schedule:
            trial = record.trial(item.trial_id)
            if trial is None:
                continue
            if trial.status in TRIAL_TERMINAL_STATUSES:
                continue
            if record.status == "cancelled":
                break

            # A candidate without a successful paired baseline is meaningless.
            if item.arm_type == "candidate":
                baseline = record.trial(f"baseline_seed_{item.seed}")
                if baseline is None or baseline.status != "succeeded":
                    trial.status = "skipped"
                    trial.skip_reason = f"baseline_seed_{item.seed} did not succeed"
                    trial.finished_at = _now()
                    self.store.save_trial(record, trial,
                                          event_type="trial_skipped",
                                          payload={"reason": trial.skip_reason})
                    continue

            if self.clock() > deadline:
                return self._pause(record, plan, run_spec,
                                   "max_total_seconds exhausted")
            try:
                self.trial_runner.execute(trial, record=record,
                                          run_spec=run_spec)
            except RecoverableCampaignError as exc:
                return self._pause(record, plan, run_spec, str(exc))
            except CampaignStoreError as exc:
                # Persistence is broken: STOP. Any further submit would be an
                # unrecoverable, unrecorded experiment.
                record.status = "failed"
                try:
                    self.store.append_event(record.campaign_id,
                                            "campaign_store_failure",
                                            payload={"error": str(exc)})
                except CampaignStoreError:
                    pass
                return record
            except Exception as exc:  # one trial must not erase the others
                trial.status = "failed"
                trial.error_type = type(exc).__name__
                trial.error_message = str(exc)
                trial.finished_at = _now()
                try:
                    self.store.save_trial(record, trial,
                                          event_type="trial_exception",
                                          payload={"error": str(exc)})
                except CampaignStoreError:
                    record.status = "failed"
                    return record
                if not self.policy.continue_on_trial_failure:
                    break
        return self._finalize(record, plan, run_spec)

    def _pause(self, record: CampaignRecord, plan: ExperimentPlan,
               run_spec: CampaignRunSpec, reason: str) -> CampaignRecord:
        record.status = "paused"
        self.store.save_campaign(record, event_type="campaign_paused",
                                 payload={"reason": reason})
        self._write_manifest(record, plan, run_spec)
        return record

    def _finalize(self, record: CampaignRecord, plan: ExperimentPlan,
                  run_spec: CampaignRunSpec) -> CampaignRecord:
        if record.status == "cancelled":
            final = "cancelled"
        elif record.trials and record.completed_trials == len(record.trials):
            final = "completed"
        elif record.failed_trials or record.skipped_trials:
            final = "completed_with_failures"
        else:
            final = "completed_with_failures"
        record.status = final                   # type: ignore[assignment]
        self.store.save_campaign(record, event_type="campaign_finalized",
                                 payload={"status": final,
                                          "summary": {
                                              "total": len(record.trials),
                                              "succeeded": record.completed_trials,
                                              "failed": record.failed_trials,
                                              "skipped": record.skipped_trials}})
        return self._write_manifest(record, plan, run_spec)

    def _write_manifest(self, record: CampaignRecord,
                        plan: ExperimentPlan | None = None,
                        run_spec: CampaignRunSpec | None = None
                        ) -> CampaignRecord:
        paths = write_manifest(
            record,
            campaign_dir=self._campaign_dir(record.campaign_id),
            store=self.store,
            plan=(plan.model_dump(mode="json") if plan else None),
            run_spec=run_spec,
            policy=self.policy,
        )
        record.final_manifest_path = paths["campaign_result"]
        # Persist the manifest location too, so resume() can find the audit
        # trail without re-deriving it. No extra event: this is a cache update.
        try:
            self.store.save_campaign(record)
        except CampaignStoreError:
            pass
        return record
