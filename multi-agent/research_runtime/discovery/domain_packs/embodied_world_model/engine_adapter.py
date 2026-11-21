"""M7 Domain Pack (embodied): REAL M5-compatible scientific engine calls.

This is not a FakeRunner. Each arm/seed/split pair performs a real TRAIN job
(fitting the frozen world model) followed by an EVALUATE job that reuses the
trained checkpoint and rolls it out on the requested frozen split. Every
result.json is re-read from disk and its SHA-256 is bound into the summary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ....domains.embodied import PROJECT_DIR, embodied_compute
from ....domains.local_engine import LocalComputeAdapter
from ....harness_rsi.identity import freeze_json, sha256_file
from ....schemas import ExperimentSpec
from . import experiment_templates
from .dataset_registry import dataset_identity

ENGINE_NAME = "m7_embodied_world_model_local"


def _run_one(adapter: LocalComputeAdapter, spec: ExperimentSpec
             ) -> dict[str, Any]:
    job_id = str(spec.budget["job_id"])
    submitted = adapter.submit(spec)
    if submitted != job_id:
        raise RuntimeError(f"engine ignored deterministic job_id {job_id}")
    adapter.wait(job_id)
    result = adapter.collect_result(job_id, spec)
    if not result.result_path:
        return {"status": "incomplete", "error": "no result_path",
                "job_id": job_id}
    result_path = Path(result.result_path)
    return {
        "job_id": job_id,
        "status": result.status,
        "stage": result.stage,
        "split": result.evaluation_split or spec.split,
        "run_id": result.run_id,
        "experiment_id": result.experiment_id,
        "engine": result.engine,
        "metrics": dict(result.metrics),
        "error": result.error,
        "error_type": result.error_type,
        "exit_code": result.exit_code,
        "dataset_version": result.dataset_version,
        "config_sha256": (result.raw or {}).get("config_sha256", ""),
        "manifest_sha256": (result.raw or {}).get("manifest_sha256", ""),
        "elapsed_seconds": (result.raw or {}).get("elapsed_seconds", 0.0),
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "raw": result.raw,
    }


def run_experiment(*, experiment_id: str, arms: list[str], seeds: list[int],
                   splits: list[str], out_dir: str | Path) -> dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    configs = experiment_templates.make_configs(root)
    adapter = LocalComputeAdapter(
        name=ENGINE_NAME, project_dir=PROJECT_DIR, compute=embodied_compute,
        job_root=root / "jobs")
    ident = dataset_identity()
    results: dict[str, Any] = {}
    for arm in arms:
        if arm not in configs["configs"]:
            raise KeyError(f"unknown arm {arm}")
        cfg = configs["configs"][arm]
        for seed in seeds:
            key = f"{arm}|s{seed}"
            train_job = f"{experiment_id}_{arm}_s{seed}_train"
            run_id = f"{experiment_id}_{arm}_s{seed}"
            train_spec = ExperimentSpec(
                experiment_id=experiment_id, objective_id=experiment_id,
                hypothesis_id=arm, engine=ENGINE_NAME, stage="train",
                config_path=cfg["path"], dataset_version=ident["dataset_version"],
                seed=int(seed), split="train",
                budget={"job_id": train_job, "arm": arm}, run_id=run_id)
            train = _run_one(adapter, train_spec)
            seed_result = {"config_sha256": cfg["sha256"], "train": train,
                           "evaluations": {}}
            if train.get("status") != "succeeded":
                seed_result["status"] = "BLOCKED"
                results[key] = seed_result
                continue
            for split in splits:
                eval_job = f"{experiment_id}_{arm}_s{seed}_eval_{split}"
                eval_spec = train_spec.with_run_id(
                    run_id, stage="evaluate", split=split,
                    parent_job_id=train["job_id"])
                eval_spec = eval_spec.model_copy(update={
                    "budget": {"job_id": eval_job, "arm": arm,
                               "split": split}})
                seed_result["evaluations"][split] = _run_one(adapter, eval_spec)
            seed_result["status"] = ("succeeded" if all(
                e.get("status") == "succeeded"
                for e in seed_result["evaluations"].values()) else "failed")
            results[key] = seed_result
    obj = {
        "version": "m7_embodied_engine_results_v1",
        "experiment_id": experiment_id,
        "engine": adapter.capabilities(),
        "dataset": ident,
        "configs": configs,
        "arms": list(arms), "seeds": list(seeds), "splits": list(splits),
        "results": results,
        "honesty_note": ("A real deterministic simulated robot engine; no "
                         "hardware execution and no SciNet++ checkpoint "
                         "training is claimed."),
    }
    frozen = freeze_json(obj, root / "engine_results.json")
    return {"ok": True, "summary_sha256": frozen["sha256"],
            "path": frozen["path"], "summary": obj}


__all__ = ["ENGINE_NAME", "run_experiment"]
