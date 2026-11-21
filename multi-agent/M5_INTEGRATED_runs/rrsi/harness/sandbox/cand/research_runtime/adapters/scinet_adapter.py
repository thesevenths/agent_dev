"""SciNet++ adapter -- subprocess JSON-RPC over ``research_api.cli``.

No PyTorch import, no model classes, no shared filesystem assumptions beyond
paths. Everything goes through structured JSON, matching the P0 contract:

    Agent Runtime (python 3.13 + LangGraph)
        |  subprocess: <scinet_python> -m research_api.cli <cmd>
        v
    SciNet++ (conda scinet_gpu + torch) -- detached worker -> train.py / ...

``submit`` returns immediately because the engine's worker redirects its stdio
to DEVNULL; otherwise the captured pipe would stay open for the whole training
run and the "async" call would secretly block.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any

from ..schemas import ExperimentResult, ExperimentSpec
from ..settings import CLI_TIMEOUT_S, resolve_engine
from .base import ModelAdapter


class AdapterError(RuntimeError):
    """Engine CLI failed or returned an error payload."""


class SciNetAdapter(ModelAdapter):
    name = "scinet++"
    stages = ("train", "evaluate", "analyze", "symbolic", "tta")

    def __init__(self, *, project_dir: str | None = None,
                 python_executable: str | None = None,
                 config_path: str | None = None,
                 db_path: str | None = None,
                 cli_timeout: float | None = None,
                 strict: bool = True):
        self.project_dir, self.python = resolve_engine(
            project_dir=project_dir, python_executable=python_executable,
            strict=strict,
        )
        from ..settings import DEFAULT_CONFIG
        self.default_config = config_path or DEFAULT_CONFIG
        self.db_path = db_path
        self.cli_timeout = cli_timeout if cli_timeout is not None else CLI_TIMEOUT_S

    # ------------------------------------------------------------- transport
    def _cli(self, *args: str, timeout: float | None = None) -> dict[str, Any]:
        cmd = [str(self.python), "-m", "research_api.cli", *args]
        if self.db_path:
            cmd += ["--db", str(self.db_path)]
        try:
            proc = subprocess.run(
                cmd, cwd=str(self.project_dir), capture_output=True,
                text=True, timeout=timeout or self.cli_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdapterError(
                f"engine CLI timed out after {timeout or self.cli_timeout}s: {cmd}"
            ) from exc

        try:
            payload = json.loads(proc.stdout)
        except (ValueError, TypeError) as exc:
            raise AdapterError(
                f"engine CLI returned non-JSON (rc={proc.returncode}): "
                f"stdout={proc.stdout[:500]!r} stderr={proc.stderr[:500]!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise AdapterError(f"engine CLI returned {type(payload).__name__}, expected object")
        # A JobStore record legitimately contains an ``error`` column whose
        # value is null on success. Only a NON-EMPTY error string (or a non-zero
        # exit code) signals a CLI failure -- checking mere key presence would
        # reject every status/wait/list response.
        cli_error = payload.get("error")
        has_error = isinstance(cli_error, str) and bool(cli_error.strip())
        if proc.returncode != 0 or has_error:
            raise AdapterError(
                f"engine CLI error (rc={proc.returncode}): "
                f"{cli_error or payload}"
            )
        return payload

    # ---------------------------------------------------------- ModelAdapter
    def submit(self, spec: ExperimentSpec) -> str:
        if spec.stage not in self.stages:
            raise AdapterError(
                f"stage {spec.stage!r} not supported by {self.name}; "
                f"expected one of {self.stages}"
            )
        args = [
            "submit",
            "--experiment-id", spec.experiment_id,
            "--stage", spec.stage,
            "--config", spec.config_path,
            "--spec-json", json.dumps(spec.to_engine_spec(), ensure_ascii=False),
        ]
        if spec.run_id:
            args += ["--run-id", spec.run_id]
        # Only the evaluation stage consumes an explicit split. Other stages
        # own their split policy (train -> train/validation, tta -> adapt), so
        # forwarding a stray flag could silently change what they train on.
        if spec.stage == "evaluate":
            args += ["--split", spec.split]
        epochs = spec.budget.get("epochs")
        if epochs is not None:
            args += ["--epochs", str(int(epochs))]
        if spec.budget.get("job_id"):
            args += ["--job-id", str(spec.budget["job_id"])]
        payload = self._cli(*args)
        job_id = payload.get("job_id")
        if not job_id:
            raise AdapterError(f"engine submit returned no job_id: {payload}")
        return str(job_id)

    def get_status(self, job_id: str) -> str:
        return str(self._cli("status", "--job-id", job_id).get("status", "incomplete"))

    def collect_result(self, job_id: str,
                       spec: ExperimentSpec | None = None) -> ExperimentResult:
        try:
            raw = self._cli("result", "--job-id", job_id)
        except AdapterError as exc:
            return ExperimentResult.incomplete(
                experiment_id=spec.experiment_id if spec else "unknown",
                job_id=job_id, engine=self.name, error=str(exc),
            )
        return ExperimentResult.from_cli_result(
            raw, experiment_id=spec.experiment_id if spec else None
        )

    def wait(self, job_id: str, timeout: float = 3600.0,
             poll_interval: float = 1.0) -> str:
        """Server-side wait: one CLI process polls instead of N subprocesses."""
        payload = self._cli(
            "wait", "--job-id", job_id,
            "--timeout", str(timeout), "--poll", str(poll_interval),
            timeout=timeout + 60,
        )
        return str(payload.get("status", "incomplete"))

    # ------------------------------------------------------------- extras
    def freeze_dataset(self, *, config: str | None = None, out: str,
                       splits: tuple[str, ...] = ("train", "validation", "adapt", "test")) -> dict:
        """Materialise an immutable dataset; returns its dataset_version."""
        args = [
            "freeze", "--config", config or self.default_config,
            "--out", out, "--splits", *splits,
        ]
        return self._cli(*args, timeout=max(self.cli_timeout, 600.0))

    def list_jobs(self, experiment_id: str | None = None) -> list[dict]:
        args = ["list"]
        if experiment_id:
            args += ["--experiment-id", experiment_id]
        return list(self._cli(*args).get("jobs", []))

    def probe_job(self, job_id: str) -> str | None:
        """Existence-based lookup, safe for campaign recovery.

        The engine's ``status`` command reports an unknown job as an ERROR
        payload, indistinguishable from a broken CLI at the transport layer.
        ``list`` is a successful query whose result is authoritative: a job id
        either appears or it was never submitted. If ``list`` itself fails,
        ``AdapterError`` propagates and the campaign pauses instead of
        launching a duplicate training run.
        """
        for job in self.list_jobs():
            if str(job.get("job_id")) == job_id:
                return str(job.get("status") or "incomplete")
        return None

    def capabilities(self) -> dict:
        return {
            "name": self.name,
            "stages": list(self.stages),
            "project_dir": str(self.project_dir),
            "python": str(self.python),
            "transport": "subprocess/json-cli",
        }

    def health_check(self) -> dict:
        self._cli("list")
        return {"ok": True, **self.capabilities()}
