"""Submit / query / collect SciNet++ jobs without blocking the caller.

The submit path returns immediately: a detached ``worker`` process owns the
long-running stage and writes its own state to the JobStore. Critically, the
worker's stdio is redirected to DEVNULL -- if it inherited the CLI's stdout
pipe, ``subprocess.run(capture_output=True)`` on the Agent side would block
until training finished, defeating the whole async design.
"""
from __future__ import annotations

import datetime
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .eval_service import build_evaluate_command
from .job_store import JobStore
from .result_parser import build_result, log_tail, read_result, write_result
from .settings import (
    PROJECT_DIR,
    job_log_path,
    load_config_meta,
    resolve_db_path,
    run_dir_for,
)
from .train_service import (
    build_analyze_command,
    build_symbolic_command,
    build_train_command,
    build_tta_command,
)

SUPPORTED_STAGES = ("train", "evaluate", "analyze", "symbolic", "tta")


def build_command(stage: str, config: str, run_id: str, *,
                  epochs: int | None = None,
                  extra_args: list[str] | None = None,
                  split: str | None = None) -> list[str]:
    """Registry of stage -> command builder. Add a stage here, nowhere else.

    ``split`` is only meaningful for the evaluation stage; other stages get
    their split from their own script logic (train -> train/validation,
    tta -> adapt) so a stray CLI flag can never silently change them.
    """
    if stage not in SUPPORTED_STAGES:
        raise ValueError(f"stage must be one of {SUPPORTED_STAGES}, got {stage!r}")
    if stage == "train":
        return build_train_command(config, run_id, epochs=epochs, extra_args=extra_args)
    if stage == "evaluate":
        return build_evaluate_command(config, run_id, extra_args=extra_args,
                                      split=split)
    if stage == "analyze":
        return build_analyze_command(config, run_id)
    if stage == "symbolic":
        return build_symbolic_command(config, run_id)
    return build_tta_command(config, run_id)


def make_run_id(job_id: str) -> str:
    """Unique per job. Never rely on ``ensure_output_dir``'s 'latest run' fallback.

    ``make_run_id()`` in utils/config.py only has second resolution, so two
    jobs submitted in the same second would collide. The job_id suffix removes
    that risk and makes the run directory traceable back to the job.
    """
    return f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{job_id[:8]}"


class JobRunner:
    def __init__(self, db_path: str | None = None, *,
                 project_dir: str | Path | None = None,
                 stale_seconds: float = 90.0):
        self.db_path = resolve_db_path(db_path)
        self.store = JobStore(self.db_path)
        self.project_dir = Path(project_dir) if project_dir else PROJECT_DIR
        self.stale_seconds = stale_seconds

    # --------------------------------------------------------------- submit
    def submit(self, *, experiment_id: str, stage: str, config: str,
               run_id: str | None = None, epochs: int | None = None,
               job_id: str | None = None, spec: dict | None = None,
               extra_args: list[str] | None = None,
               split: str | None = None) -> dict:
        job_id = job_id or uuid.uuid4().hex
        run_id = run_id or make_run_id(job_id)
        meta = load_config_meta(config)          # fail fast on a bad config path
        command = build_command(stage, config, run_id, epochs=epochs,
                                extra_args=extra_args, split=split)
        out_dir = run_dir_for(meta, run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        log_path = str(job_log_path(job_id))
        # One result file PER JOB. train and evaluate legitimately share the
        # same run_id (evaluate reads that run's checkpoint.pt); a single
        # result.json would let the later stage overwrite the earlier one.
        result_path = str(out_dir / f"result_{job_id}.json")
        spec_payload = dict(spec or {})
        spec_payload.setdefault("config", config)
        spec_payload.setdefault("budget", {})
        if split:
            spec_payload["requested_split"] = split

        record = self.store.create_job(
            job_id=job_id, experiment_id=experiment_id, stage=stage,
            spec=spec_payload, command=command, cwd=str(self.project_dir),
            run_id=run_id, log_path=log_path, result_path=result_path,
        )

        worker_cmd = [
            sys.executable, "-m", "research_api.worker",
            "--job-id", job_id, "--db", str(self.db_path),
        ]
        subprocess.Popen(
            worker_cmd, cwd=str(self.project_dir),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, close_fds=True,
        )
        return {
            "job_id": job_id,
            "experiment_id": experiment_id,
            "stage": stage,
            "status": record["status"],
            "run_id": run_id,
            "result_path": result_path,
            "log_path": log_path,
        }

    # ---------------------------------------------------------------- query
    def get_status(self, job_id: str, *, reconcile: bool = True) -> dict | None:
        if reconcile:
            self.store.reconcile_stale(self.stale_seconds)
        return self.store.get(job_id)

    def collect_result(self, job_id: str) -> dict:
        """Always returns a full ExperimentResult-shaped dict, even on failure."""
        job = self.get_status(job_id)
        if job is None:
            raise KeyError(f"unknown job_id: {job_id}")
        result = read_result(job.get("result_path"))
        if result is not None:
            return result
        # Worker died before writing result.json -> synthesise a complete record.
        return build_result(
            job=job, status=job["status"], exit_code=job.get("exit_code"),
            out_dir=Path(job["result_path"]).parent if job.get("result_path") else self.project_dir,
            error=job.get("error") or log_tail(job.get("log_path")),
            error_type=job.get("error_type") or "result_unavailable",
            finished_at=job.get("finished_at"),
        )

    def wait(self, job_id: str, timeout: float = 3600.0,
             poll_interval: float = 1.0) -> dict:
        """Block until terminal. Used by the CLI and by tests, not by the Agent."""
        deadline = time.monotonic() + timeout
        while True:
            job = self.get_status(job_id)
            if job is None:
                raise KeyError(f"unknown job_id: {job_id}")
            if job["status"] in ("succeeded", "failed", "cancelled"):
                return job
            if time.monotonic() >= deadline:
                return {"job_id": job_id, "status": "timeout",
                        "last_status": job["status"]}
            time.sleep(poll_interval)

    def list_jobs(self, experiment_id: str | None = None,
                  status: str | None = None) -> list[dict]:
        return self.store.list_jobs(experiment_id=experiment_id, status=status)

    # ------------------------------------------------------------- utilities
    def write_failure_result(self, job_id: str) -> dict | None:
        """Backfill a result.json for a job whose worker vanished. Idempotent."""
        job = self.get_status(job_id, reconcile=False)
        if job is None or job["status"] not in ("failed", "cancelled"):
            return None
        existing = read_result(job.get("result_path"))
        if existing is not None:
            return existing
        payload = build_result(
            job=job, status=job["status"], exit_code=job.get("exit_code"),
            out_dir=Path(job["result_path"]).parent,
            error=job.get("error") or log_tail(job.get("log_path")),
            error_type=job.get("error_type") or "worker_interrupted",
            finished_at=job.get("finished_at"),
        )
        write_result(job["result_path"], payload)
        return payload
