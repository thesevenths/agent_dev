"""ARR-RSI-01 -- 20 deterministic scientific research tasks + sealed ground truth.

Isolation contract
------------------
``PublicTask`` is the ONLY task object handed to the Research Runtime or to a
Diagnosis implementation. It has no ground-truth field by construction.
``GroundTruth`` lives in a separate store that is imported only by the
independent evaluator and by the artifact writer (sealed file).  ``agent_view``
additionally removes the hidden-test reference, so an agent cannot even learn
which frozen artifact will be used for final scoring.

Reproducibility contract
------------------------
Every task pins: engine config, frozen dataset version, split SHAs, paired
seed, and budget. Two calls to ``build_tasks()`` produce byte-identical output.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BENCHMARK_ID = "arr_rsi_01"
BENCHMARK_VERSION = "arr-rsi-01-tasks-v1"
FAMILIES = (
    "pendulum_dynamics",
    "newtonian_dynamics",
    "symbolic_regression",
    "robot_dynamics",
    "scientific_ml",
)
SEEDS_PER_FAMILY = (42, 43, 44, 45)

# The two engine sources verified end-to-end before P0. See CHANGE_LIST.md.
ENGINE_SOURCES: dict[str, dict[str, str]] = {
    "pendulum_dynamics": {
        "config": "configs/pendulum_smoke.yaml",
        "dataset_dir": "datasets/pendulum_smoke_v1",
    },
    "newtonian_dynamics": {
        "config": "configs/pendulum_smoke.yaml",
        "dataset_dir": "datasets/pendulum_smoke_v1",
    },
    "symbolic_regression": {
        "config": "configs/pendulum_smoke.yaml",
        "dataset_dir": "datasets/pendulum_smoke_v1",
    },
    "robot_dynamics": {
        "config": "configs/pendulum_smoke.yaml",
        "dataset_dir": "datasets/pendulum_smoke_v1",
    },
    "scientific_ml": {
        "config": "configs/pendulum_smoke_ood.yaml",
        "dataset_dir": "datasets/pendulum_smoke_ood_v1",
    },
}

_FAILURE_MODES: dict[str, tuple[str, str, str]] = {
    # family: (failure_mode, root_cause_key, correct_intervention_key)
    "pendulum_dynamics": (
        "dynamics_error_accumulation",
        "latent_dynamics_undertrained",
        "train.lr",
    ),
    "newtonian_dynamics": (
        "parameter_identifiability_loss",
        "insufficient_representation_capacity",
        "model.predictor_hidden",
    ),
    "symbolic_regression": (
        "conservation_signal_weak",
        "weak_prediction_objective",
        "jepa.pred_weight",
    ),
    "robot_dynamics": (
        "long_horizon_state_drift",
        "latent_dynamics_undertrained",
        "jepa.ema_momentum",
    ),
    "scientific_ml": (
        "ood_regime_shift",
        "insufficient_representation_capacity",
        "model.latent_dim",
    ),
}

_OBJECTIVES: dict[str, str] = {
    "pendulum_dynamics": (
        "Reduce multi-step latent dynamics prediction error on the frozen "
        "damped-pendulum world model under matched research budgets."
    ),
    "newtonian_dynamics": (
        "Improve identification of the governing second-order dynamics from "
        "the frozen pendulum trajectories without changing the data."
    ),
    "symbolic_regression": (
        "Strengthen the latent signal that supports compact symbolic recovery "
        "of the governing law from frozen trajectories."
    ),
    "robot_dynamics": (
        "Reduce long-horizon state-drift of the frozen dynamics surrogate "
        "under a control-oriented evaluation."
    ),
    "scientific_ml": (
        "Improve latent-dynamics generalisation on the independently frozen "
        "regime-shift (OOD) split."
    ),
}

_INTERVENTION_VALUES: dict[str, Any] = {
    "train.lr": 0.0008,
    "model.predictor_hidden": [64, 64],
    "jepa.pred_weight": 4.0,
    "jepa.ema_momentum": 0.99,
    "model.latent_dim": 8,
}

# Same-cost but mechanistically irrelevant intervention per failure mode.
_PLACEBO_KEYS: dict[str, str] = {
    "dynamics_error_accumulation": "train.weight_decay",
    "parameter_identifiability_loss": "jepa.ema_momentum",
    "conservation_signal_weak": "train.weight_decay",
    "long_horizon_state_drift": "train.weight_decay",
    "ood_regime_shift": "jepa.pred_weight",
}
_PLACEBO_VALUES: dict[str, Any] = {
    "train.weight_decay": 0.0001,
    "jepa.ema_momentum": 0.999,
    "jepa.pred_weight": 1.01,
}

GT_LEAF_KEYS = frozenset({
    "ground_truth_root_cause", "correct_intervention",
    "distractor_intervention",
})


class GroundTruthLeak(RuntimeError):
    """Raised when a sealed ground-truth field reaches an agent-facing view."""


@dataclass(frozen=True)
class PublicTask:
    """Agent-visible task. No ground-truth field exists on this type."""

    task_id: str
    task_family: str
    task_description: str
    dataset_reference: str
    train_split: str
    validation_split: str
    hidden_test_split: str
    failure_mode: str
    valid_interventions: tuple[dict[str, Any], ...]
    primary_metric: str
    budget: dict[str, Any]
    engine_config: str
    dataset_version: str
    seed: int
    objective: str
    metric_direction: str = "minimize"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["valid_interventions"] = [dict(x) for x in self.valid_interventions]
        return d

    def agent_view(self) -> dict[str, Any]:
        """View for Research/Diagnosis agents: hidden test ref removed."""
        d = self.to_dict()
        d.pop("hidden_test_split", None)
        d["view"] = "agent"
        return d

    def manifest_view(self) -> dict[str, Any]:
        d = self.to_dict()
        d["task_sha256"] = sha256_json(d)
        return d


@dataclass(frozen=True)
class GroundTruth:
    task_id: str
    ground_truth_root_cause: str
    correct_intervention: str
    distractor_intervention: str
    falsification_condition: str
    evidence_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_split_refs(scinet_dir: str | Path, dataset_dir: str) -> dict[str, str]:
    """Content-addressed split references from the REAL frozen manifest."""
    path = Path(scinet_dir) / dataset_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"frozen dataset manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    version = manifest.get("dataset_version")
    if not version:
        raise ValueError(f"manifest has no dataset_version: {path}")
    splits = manifest.get("splits") or {}
    refs: dict[str, str] = {}
    for name in ("train", "validation", "test"):
        if name not in splits:
            raise ValueError(f"manifest {path} missing split {name}")
        entry = splits[name]
        digest = entry.get("sha256") if isinstance(entry, dict) else str(entry)
        refs[name] = f"{dataset_dir}/{name}@{version}:{digest}"
    return {"dataset_version": str(version), **refs}


def build_tasks(scinet_dir: str | Path) -> list[PublicTask]:
    """Deterministically build the 20 tasks. No randomness, no GT."""
    tasks: list[PublicTask] = []
    for family in FAMILIES:
        source = ENGINE_SOURCES[family]
        refs = read_split_refs(scinet_dir, source["dataset_dir"])
        failure_mode, _cause, correct_key = _FAILURE_MODES[family]
        placebo_key = _PLACEBO_KEYS[failure_mode]
        for seed in SEEDS_PER_FAMILY:
            task_id = f"{BENCHMARK_ID}::{family}::s{seed}"
            interventions = (
                {"key": correct_key, "value": _INTERVENTION_VALUES[correct_key],
                 "role": "candidate", "family": family, "seed": seed},
                {"key": placebo_key, "value": _PLACEBO_VALUES[placebo_key],
                 "role": "placebo", "family": family, "seed": seed},
            )
            tasks.append(PublicTask(
                task_id=task_id,
                task_family=family,
                task_description=_OBJECTIVES[family],
                dataset_reference=(
                    f"{source['dataset_dir']}@{refs['dataset_version']}"),
                train_split=refs["train"],
                validation_split=refs["validation"],
                hidden_test_split=refs["test"],
                failure_mode=failure_mode,
                valid_interventions=interventions,
                primary_metric="latent_mse@1",
                budget={"experiments": 2, "max_jobs": 8,
                        "max_total_seconds": 900.0, "epochs": 1,
                        "paired_seeds": [seed]},
                engine_config=source["config"],
                dataset_version=refs["dataset_version"],
                seed=seed,
                objective=_OBJECTIVES[family],
            ))
    assert len(tasks) == 20
    return tasks


def build_ground_truth(tasks: list[PublicTask]) -> dict[str, GroundTruth]:
    """Sealed answers. NEVER called on an agent-facing code path."""
    out: dict[str, GroundTruth] = {}
    for task in tasks:
        _fm, cause_key, correct_key = _FAILURE_MODES[task.task_family]
        placebo_key = _PLACEBO_KEYS[_FAILURE_MODES[task.task_family][0]]
        out[task.task_id] = GroundTruth(
            task_id=task.task_id,
            ground_truth_root_cause=cause_key,
            correct_intervention=correct_key,
            distractor_intervention=placebo_key,
            falsification_condition=(
                f"No significant paired improvement on {task.primary_metric} "
                f"when only {correct_key} is changed under seed {task.seed}."),
            evidence_note=(
                "Root-cause keys are pre-registered by the benchmark author, "
                "not derived from any observed run."),
        )
    return out


class GroundTruthStore:
    """Sealed, hash-verified ground truth. Evaluator-only."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def seal(self, truth: dict[str, GroundTruth]) -> dict[str, Any]:
        obj = {"version": BENCHMARK_VERSION,
               "records": [truth[k].to_dict() for k in sorted(truth)]}
        digest = sha256_json(obj)
        payload = {**obj, "ground_truth_sha256": digest}
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8")
        return {"path": str(self.path), "ground_truth_sha256": digest,
                "n_records": len(truth)}

    def load(self, expected_sha256: str = "") -> dict[str, GroundTruth]:
        obj = json.loads(self.path.read_text(encoding="utf-8"))
        records = obj.get("records", [])
        digest = sha256_json({"version": obj.get("version"),
                              "records": records})
        if expected_sha256 and digest != expected_sha256:
            raise GroundTruthLeak("sealed ground truth SHA mismatch")
        return {str(r["task_id"]): GroundTruth(**r) for r in records}


def assert_no_ground_truth(obj: Any, path: str = "$") -> None:
    """Recursive fail-closed check for agent-facing payloads."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in GT_LEAF_KEYS:
                raise GroundTruthLeak(f"ground truth at {path}.{key}")
            assert_no_ground_truth(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            assert_no_ground_truth(value, f"{path}[{index}]")


def task_manifest(tasks: list[PublicTask], *, ground_truth_sha256: str = "") -> dict[str, Any]:
    body = {
        "benchmark_id": BENCHMARK_ID,
        "version": BENCHMARK_VERSION,
        "n_tasks": len(tasks),
        "families": list(FAMILIES),
        "tasks": [t.manifest_view() for t in tasks],
        "ground_truth_sha256": ground_truth_sha256,
        "ground_truth_separately_sealed": True,
    }
    body["manifest_sha256"] = sha256_json(body)
    return body


__all__ = [
    "BENCHMARK_ID", "BENCHMARK_VERSION", "FAMILIES", "SEEDS_PER_FAMILY",
    "GroundTruthLeak", "PublicTask", "GroundTruth", "GroundTruthStore",
    "build_tasks", "build_ground_truth", "read_split_refs",
    "assert_no_ground_truth", "task_manifest", "sha256_json",
]
