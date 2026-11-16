"""M6 -- REAL engine adapter for Base/Candidate paired execution.

This wrapper does NOT fake results. For every benchmark task it:

  1. grades the research METHOD with the real deterministic method modules;
  2. executes the task's REAL scientific engine (SciNet++ subprocess JSON-RPC,
     or the existing Embodied / Quant local engines) through the SAME
     ``_build_real_domain_runner`` used by M5;
  3. attaches the real engine Q and its artifact path as EVIDENCE on the result.

A real engine that is unavailable or returns NaN raises: the arm is recorded as
an INFRASTRUCTURE failure, never as a research success. Candidate code has no
access to this evaluator decision.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from ..cli import _build_real_domain_runner
from ..settings import DEFAULT_SCINET_DIR
from .method_runner import deterministic_method_runner

_PROJECTS = {
    "embodied": Path("research_runtime/domains/projects/embodied"),
    "quant": Path("research_runtime/domains/projects/quant"),
}


def build_real_runner(root: Path, *, timeout_s: float = 900.0) -> Any:
    """Build the real-engine M6 runner. Fail closed on engine faults."""
    root = Path(root)
    cache: dict[str, Any] = {}

    def runner(harness_artifact: dict[str, Any], task: Any) -> dict[str, Any]:
        # Method grade always comes from REAL production research modules.
        out = deterministic_method_runner(harness_artifact, task)
        domain = getattr(task, "domain", "")
        source = dict(getattr(task, "source", {}) or {})
        if domain not in ("scinet", "embodied", "quant"):
            raise RuntimeError(f"unknown domain for real engine: {domain}")
        if domain not in cache:
            project = root / _PROJECTS[domain] if domain in _PROJECTS else None
            cache[domain] = _build_real_domain_runner(
                domain, project_dir=project)
        if domain == "scinet":
            cfg_rel = source.get("config", "configs/pendulum_smoke.yaml")
            cfg_path = DEFAULT_SCINET_DIR / cfg_rel
            ood = DEFAULT_SCINET_DIR / "configs" / "pendulum_smoke_ood.yaml"
            art = {"config_path": str(cfg_path),
                   "ood_config_path": str(ood if ood.is_file() else cfg_path)}
        else:
            cfg_path = root / _PROJECTS[domain] / "config.yaml"
            art = {"config_path": str(cfg_path)}
        job_dir = (root / "M6_EVIDENCE" / "real_engine_jobs" /
                   f"{getattr(harness_artifact, 'get', lambda *_: '')('harness_sha', 'x')[:12]}" /
                   f"{getattr(task, 'task_id', 'task')}")
        art["directory"] = str(job_dir)
        q = cache[domain](art)
        if q is None or (isinstance(q, float) and math.isnan(q)):
            raise RuntimeError(
                f"real engine returned NaN for {domain}/{getattr(task, 'task_id', '')}; "
                "infrastructure failure, not a research result")
        out["real_engine"] = {"domain": domain, "engine_Q": float(q),
                              "config_path": str(cfg_path),
                              "job_dir": str(job_dir),
                              "dataset": source.get("dataset", "")}
        out["metrics"]["engine_Q"] = float(q)
        return out

    return runner


__all__ = ["build_real_runner"]
