"""Detached worker: owns one job from 'queued' to terminal state.

Runs as its own process (launched by ``JobRunner.submit``), so the submitting
CLI/Agent can exit immediately. It writes an atomic ``result.json`` and updates
the SQLite JobStore. A heartbeat thread lets ``JobStore.reconcile_stale``
detect a crashed worker after a process/host restart.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from .job_store import JobStore
from .result_parser import build_result, log_tail, write_result
from .settings import PROJECT_DIR, resolve_config_path


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _heartbeat(store: JobStore, job_id: str, interval: float, stop: threading.Event):
    while not stop.wait(interval):
        try:
            store.heartbeat(job_id)
        except Exception:
            return


def _dataset_version(config: str | None) -> str | None:
    """Best-effort: read the YAML config and reuse the canonical version id."""
    try:
        import yaml
        from data.dataset import dataset_version
        path = resolve_config_path(config) if config else None
        if not path or not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        return dataset_version(cfg)
    except Exception:
        # research_api must still work in a lightweight env without torch.
        return None


def _required_artifact(stage: str) -> str | None:
    return {
        "train": "checkpoint.pt",
        "evaluate": "horizon_metrics.npy",
    }.get(stage)


def _command_option(command: list[str], option: str) -> str | None:
    """Extract ``--flag value`` from the ACTUAL argv the worker executed.

    Reading the command (rather than the caller's spec) means result.json
    records what really ran, even if a human edited the job in the store.
    """
    try:
        index = command.index(option)
        return str(command[index + 1])
    except (ValueError, IndexError):
        return None


def run_job(job_id: str, db_path: str, heartbeat_interval: float = 3.0) -> int:
    store = JobStore(db_path)
    job = store.get(job_id)
    if job is None:
        print(json.dumps({"error": f"unknown job_id {job_id}"}))
        return 2
    if not store.claim_running(job_id, os.getpid()):
        # Another worker already owns it -- exactly-once execution guard.
        print(json.dumps({"job_id": job_id, "skipped": "not in queued state"}))
        return 0

    command = job.get("command") or []
    if not command:
        store.update(job_id, status="failed", error_type="empty_command",
                     error="job has no command", finished_at=_now())
        return 1
    cwd = job.get("cwd") or str(PROJECT_DIR)
    log_path = job.get("log_path")
    result_path = job.get("result_path") or str(PROJECT_DIR / "result.json")
    spec = job.get("spec") or {}
    timeout_s = float((spec.get("budget") or {}).get("timeout_s", 3600) or 3600)
    stage = job.get("stage")
    started_at = _now()

    stop = threading.Event()
    hb = threading.Thread(target=_heartbeat,
                          args=(store, job_id, heartbeat_interval, stop),
                          daemon=True)
    hb.start()

    exit_code: int | None = None
    error: str | None = None
    error_type: str | None = None
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(
                "# ---- research_api worker ----\n"
                f"# job_id   : {job_id}\n"
                f"# stage    : {stage}\n"
                f"# run_id   : {job.get('run_id')}\n"
                f"# started  : {started_at}\n"
                f"# command  : {json.dumps(command, ensure_ascii=False)}\n"
                f"# cwd      : {cwd}\n\n"
            )
            log.flush()
            proc = subprocess.run(
                command, cwd=cwd, stdout=log,
                stderr=subprocess.STDOUT, timeout=timeout_s,
            )
            exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        error_type = "timeout"
        error = f"stage exceeded budget timeout_s={timeout_s}"
    except Exception as exc:  # noqa: BLE001 - worker must always record a result
        error_type = "worker_exception"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()

    status = "succeeded" if exit_code == 0 else "failed"
    if status == "succeeded":
        required = _required_artifact(stage)
        if required and not (Path(result_path).parent / required).is_file():
            status = "failed"
            error_type = "missing_artifact"
            error = f"stage exited 0 but {required} was not produced"
        elif not error:
            error, error_type = None, None
    if status == "failed" and not error:
        error = log_tail(log_path)
        error_type = error_type or "process_failed"

    out_dir = Path(result_path).parent
    finished_at = _now()   # single clock read: DB and result.json must agree
    payload = build_result(
        job=job, status=status, exit_code=exit_code, out_dir=out_dir,
        started_at=started_at, finished_at=finished_at,
        error=error, error_type=error_type,
        dataset_version=_dataset_version(spec.get("config")),
        split=_command_option(command, "--split") if stage == "evaluate" else None,
    )
    write_result(result_path, payload)
    store.update(
        job_id, status=status, exit_code=exit_code, error=error,
        error_type=error_type, result_path=result_path, finished_at=finished_at,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="research_api detached worker")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--heartbeat-interval", type=float, default=3.0)
    args = parser.parse_args()
    sys.exit(run_job(args.job_id, args.db, args.heartbeat_interval))


if __name__ == "__main__":
    main()
