"""P1-04: execute ONE arm/seed = train -> validation, resumably.

Two rules dominate this module:

1. PERSIST BEFORE EXECUTE. The exact ExperimentSpec and its deterministic
   job_id are written to SQLite before ``submit()``. A crash between submit
   and the next state write is recoverable because the id can be recomputed.
   ``collect_result(job_id, persisted_spec)`` is always called with the saved
   spec -- never with ``ExperimentManager._specs``, which is process memory.

2. NEVER RESUBMIT AN UNDECIDABLE JOB. An engine that cannot be reached is NOT
   evidence that a job is missing. Only an existence-based probe returning
   ``None`` may lead to a submit; any lookup failure pauses the campaign.
   A deterministic job id also means the engine itself rejects a duplicate
   with its own primary-key constraint (defence in depth).

Integrity checks use the ENGINE'S OWN evidence (``evaluation_split``,
``dataset_version``, ``run_id``, ``checkpoint``), not the requested spec.
"""
from __future__ import annotations

import math
from typing import Any, Callable

from ..experiment_manager import ExperimentManager
from ..schemas import (
    TERMINAL_STATUSES,
    ExperimentResult,
    ExperimentSpec,
)
from .campaign_store import CampaignStore
from .exceptions import CampaignIntegrityError, RecoverableCampaignError
from .policy import CampaignPolicy
from .scheduler import make_job_id
from .schemas import CampaignRecord, CampaignRunSpec, TrialRecord, _now

JobProbe = Callable[[str], "str | None"]


class TrialRunner:
    def __init__(self, manager: ExperimentManager, policy: CampaignPolicy,
                 store: CampaignStore, *, job_probe: JobProbe | None = None):
        self.manager = manager
        self.policy = policy
        self.store = store
        self.job_probe = job_probe

    # ------------------------------------------------------------- helpers
    def _probe(self, job_id: str) -> str | None:
        try:
            if self.job_probe is not None:
                return self.job_probe(job_id)
            adapter = getattr(self.manager, "adapter", None)
            probe = getattr(adapter, "probe_job", None)
            if callable(probe):
                return probe(job_id)
            return self.manager.get_status(job_id)
        except Exception as exc:  # transport/CLI failure is NOT "job missing"
            raise RecoverableCampaignError(
                f"job lookup for {job_id!r} is undecidable "
                f"({type(exc).__name__}: {exc}); refusing to resubmit"
            ) from exc

    def _fail(self, record: CampaignRecord, trial: TrialRecord, *,
              error_type: str, message: str, event: str,
              status_after: str = "failed") -> TrialRecord:
        trial.status = status_after            # type: ignore[assignment]
        trial.error_type = error_type
        trial.error_message = message
        trial.finished_at = _now()
        self.store.save_trial(record, trial, event_type=event,
                              payload={"error_type": error_type,
                                       "error": message})
        return trial

    def _train_spec(self, trial: TrialRecord,
                    run_spec: CampaignRunSpec) -> ExperimentSpec:
        job_id = make_job_id(run_spec.campaign_id, trial.trial_id,
                             self.policy.train_stage)
        return ExperimentSpec(
            experiment_id=trial.trial_id,
            objective_id=run_spec.objective_id,
            hypothesis_id=trial.hypothesis_id,
            engine=getattr(getattr(self.manager, "adapter", None),
                           "name", "unknown"),
            stage=self.policy.train_stage,
            config_path=trial.config_relpath or trial.config_path,
            dataset_version=trial.dataset_version,
            seed=trial.seed,
            split=self.policy.train_split,
            budget={"epochs": self.policy.epochs, "job_id": job_id},
        )

    def _evaluation_spec(self, train_spec: ExperimentSpec,
                         train_job_id: str, train_run_id: str,
                         trial: TrialRecord,
                         run_spec: CampaignRunSpec) -> ExperimentSpec:
        spec = train_spec.with_run_id(
            train_run_id, stage=self.policy.evaluate_stage,
            parent_job_id=train_job_id, split=run_spec.evaluation_split,
        )
        budget = {k: v for k, v in (spec.budget or {}).items()
                  if k not in ("epochs", "job_id")}
        budget["job_id"] = make_job_id(run_spec.campaign_id, trial.trial_id,
                                       self.policy.evaluate_stage)
        return spec.model_copy(update={"budget": budget})

    def _recover_or_submit(self, spec: ExperimentSpec, job_id: str,
                           *, stage: str) -> str:
        """Return an existing job id, or submit exactly once.

        Any ambiguity is a RecoverableCampaignError -> the campaign pauses.
        """
        status = self._probe(job_id)
        if status is not None:
            return "existing"
        try:
            record = self.manager.submit(spec)
        except Exception as exc:
            raise RecoverableCampaignError(
                f"{stage} submit for {job_id!r} was rejected "
                f"({type(exc).__name__}: {exc}); pausing instead of retrying"
            ) from exc
        returned = str((record or {}).get("job_id") or "")
        if returned != job_id:
            raise CampaignIntegrityError(
                f"engine returned job_id {returned!r}, expected deterministic "
                f"{job_id!r}; refusing to track an unrecoverable job"
            )
        return "submitted"

    def _wait(self, job_id: str, timeout: float, *, stage: str) -> str:
        try:
            status = self.manager.wait(job_id, timeout=timeout)
        except Exception as exc:
            raise RecoverableCampaignError(
                f"{stage} wait for {job_id!r} failed "
                f"({type(exc).__name__}: {exc}); the worker may still be running"
            ) from exc
        if status not in TERMINAL_STATUSES:
            raise RecoverableCampaignError(
                f"{stage} job {job_id!r} still has status {status!r} after "
                f"{timeout}s; pausing so resume() can keep waiting on the SAME job"
            )
        return str(status)

    # -------------------------------------------------------------- checks
    @staticmethod
    def _finite(value: Any) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)))

    def _train_problems(self, result: ExperimentResult,
                        trial: TrialRecord) -> tuple[list[str], bool]:
        problems: list[str] = []
        integrity = False
        if not result.succeeded:
            problems.append(f"train status={result.status} "
                            f"({result.error_type or 'no error type'}): "
                            f"{result.error or ''}".strip())
        if not result.checkpoint or not result.checkpoint.get("path"):
            problems.append("train produced no checkpoint path")
        if not result.run_id:
            problems.append("train produced no run_id; validation cannot "
                            "reuse the trained checkpoint")
        if result.dataset_version != trial.dataset_version:
            problems.append(
                f"train dataset_version mismatch: expected "
                f"{trial.dataset_version!r}, engine reported "
                f"{result.dataset_version!r}")
            integrity = True
        return problems, integrity

    def _eval_problems(self, result: ExperimentResult, trial: TrialRecord,
                       run_spec: CampaignRunSpec) -> tuple[list[str], bool]:
        problems: list[str] = []
        integrity = False
        if not result.succeeded:
            problems.append(f"evaluation status={result.status} "
                            f"({result.error_type or 'no error type'}): "
                            f"{result.error or ''}".strip())
        actual_split = result.evaluation_split
        if actual_split != run_spec.evaluation_split:
            problems.append(
                f"engine evaluated on {actual_split!r}, requested "
                f"{run_spec.evaluation_split!r}")
            integrity = True
        if actual_split in self.policy.reserved_final_splits:
            problems.append(
                f"evaluation used reserved final split {actual_split!r}; "
                "candidate selection must never touch it")
            integrity = True
        if result.run_id != trial.train_run_id:
            problems.append(
                f"evaluation run_id {result.run_id!r} != train run_id "
                f"{trial.train_run_id!r}; checkpoint reuse is unproven")
            integrity = True
        if result.dataset_version != trial.dataset_version:
            problems.append(
                f"evaluation dataset_version mismatch: expected "
                f"{trial.dataset_version!r}, engine reported "
                f"{result.dataset_version!r}")
            integrity = True
        required = list(self.policy.required_metrics)
        if trial.primary_metric and trial.primary_metric not in required:
            required.append(trial.primary_metric)
        for metric in required:
            if metric not in result.metrics or not self._finite(result.metrics[metric]):
                problems.append(f"required metric {metric!r} missing or non-finite")
        for artifact in self.policy.required_artifacts:
            if artifact not in (result.artifacts or []):
                problems.append(f"required artifact {artifact!r} missing")
        return problems, integrity

    # ------------------------------------------------------------- execute
    def execute(self, trial: TrialRecord, *, record: CampaignRecord,
                run_spec: CampaignRunSpec,
                timeout_s: float | None = None) -> TrialRecord:
        """Run one trial. Returns a COMPLETE record on success OR failure.

        Only ``RecoverableCampaignError`` escapes (campaign -> paused). A
        failed job, a wrong split or a mismatched dataset version is recorded
        as a failed trial so the remaining paired trials still execute.
        """
        if trial.status in ("succeeded", "failed", "skipped"):
            return trial
        timeout = float(timeout_s if timeout_s is not None
                        else min(run_spec.job_timeout_s,
                                 self.policy.timeout_per_job_s))
        trial.started_at = trial.started_at or _now()

        # ---- train -------------------------------------------------------
        train_spec = self._train_spec(trial, run_spec)
        train_job_id = str(train_spec.budget["job_id"])
        trial.train_job_id = train_job_id
        trial.train_spec = train_spec.model_dump()
        trial.status = "train_submitted"
        # Persist BEFORE the engine call. If this write fails, no job exists.
        self.store.save_trial(
            record, trial, event_type="train_spec_persisted",
            payload={"job_id": train_job_id, "spec": train_spec.model_dump()},
        )
        mode = self._recover_or_submit(train_spec, train_job_id,
                                       stage=self.policy.train_stage)
        self.store.save_trial(record, trial,
                              event_type=("train_submitted" if mode == "submitted"
                                          else "train_existing_job_resumed"),
                              payload={"job_id": train_job_id, "mode": mode})
        trial.status = "training"
        self.store.save_trial(record, trial, event_type="train_wait",
                              payload={"job_id": train_job_id, "timeout_s": timeout})
        self._wait(train_job_id, timeout, stage=self.policy.train_stage)

        try:
            train_result = self.manager.collect_result(train_job_id, train_spec)
        except Exception as exc:
            return self._fail(record, trial, error_type="result_unavailable",
                              message=f"{type(exc).__name__}: {exc}",
                              event="train_result_unavailable")

        train_problems, train_integrity = self._train_problems(train_result, trial)
        if train_problems:
            return self._fail(
                record, trial,
                error_type="integrity_error" if train_integrity else "train_failed",
                message="; ".join(train_problems), event="train_rejected",
            )
        trial.train_metrics = dict(train_result.metrics)
        trial.train_run_id = train_result.run_id
        trial.train_result_path = train_result.result_path
        trial.log_path = train_result.log_path
        trial.status = "train_succeeded"
        self.store.save_trial(record, trial, event_type="train_succeeded",
                              payload={"job_id": train_job_id,
                                       "run_id": train_result.run_id,
                                       "checkpoint": train_result.checkpoint})

        # ---- validation (same run_id = same checkpoint) -------------------
        eval_spec = self._evaluation_spec(train_spec, train_job_id,
                                          str(train_result.run_id), trial,
                                          run_spec)
        eval_job_id = str(eval_spec.budget["job_id"])
        trial.evaluation_job_id = eval_job_id
        trial.evaluation_spec = eval_spec.model_dump()
        trial.evaluation_split = run_spec.evaluation_split
        trial.status = "eval_submitted"
        self.store.save_trial(
            record, trial, event_type="evaluation_spec_persisted",
            payload={"job_id": eval_job_id, "spec": eval_spec.model_dump()},
        )
        mode = self._recover_or_submit(eval_spec, eval_job_id,
                                       stage=self.policy.evaluate_stage)
        self.store.save_trial(record, trial,
                              event_type=("evaluation_submitted"
                                          if mode == "submitted"
                                          else "evaluation_existing_job_resumed"),
                              payload={"job_id": eval_job_id, "mode": mode})
        trial.status = "evaluating"
        self.store.save_trial(record, trial, event_type="evaluation_wait",
                              payload={"job_id": eval_job_id,
                                       "timeout_s": timeout})
        self._wait(eval_job_id, timeout, stage=self.policy.evaluate_stage)
        try:
            eval_result = self.manager.collect_result(eval_job_id, eval_spec)
        except Exception as exc:
            return self._fail(record, trial, error_type="result_unavailable",
                              message=f"{type(exc).__name__}: {exc}",
                              event="evaluation_result_unavailable")

        eval_problems, eval_integrity = self._eval_problems(eval_result, trial,
                                                           run_spec)
        if eval_problems:
            return self._fail(
                record, trial,
                error_type="integrity_error" if eval_integrity else "evaluation_failed",
                message="; ".join(eval_problems), event="evaluation_rejected",
            )
        trial.validation_metrics = dict(eval_result.metrics)
        trial.artifacts = list(eval_result.artifacts)
        trial.evaluation_result_path = eval_result.result_path
        trial.status = "succeeded"
        trial.finished_at = _now()
        self.store.save_trial(record, trial, event_type="trial_succeeded",
                              payload={"job_id": eval_job_id,
                                       "run_id": eval_result.run_id,
                                       "split": eval_result.evaluation_split,
                                       "metrics": trial.validation_metrics})
        return trial
