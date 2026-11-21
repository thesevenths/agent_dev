"""ARR-RSI-01 P0.1 -- 20 deterministic tasks with STRICT information isolation.

P0.1 isolation contract
-----------------------
The Research/Diagnosis agents receive ONLY ``PublicTask.agent_view()``. That
view contains no ground-truth root cause, no preset failure mode, no family
label, no candidate/placebo role or value, no hidden-test reference, and no
engine config identity. It exposes only an opaque agent_task_id, a neutral
objective, frozen data identity and TRAIN/VALIDATION split refs, the primary
metric and matched budget, and an ALLOWED CONFIG PARAMETER SCHEMA.

20 executions = 5 independent research units (family x frozen dataset) with
4 paired seed replicates. Seeds are NOT independent research tasks.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BENCHMARK_ID = "arr_rsi_01"
EXPERIMENT_ID = "ARR-RSI-01-P0.1"
BENCHMARK_VERSION = "arr-rsi-01-p0.1-tasks-v1"
FAMILIES = (
    "pendulum_dynamics", "newtonian_dynamics", "symbolic_regression",
    "robot_dynamics", "scientific_ml",
)
SEEDS_PER_FAMILY = (42, 43, 44, 45)
INDEPENDENCE_UNIT = "task_family|dataset_version"

ENGINE_SOURCES: dict[str, dict[str, str]] = {
    "pendulum_dynamics": {"config": "configs/pendulum_smoke.yaml",
                          "dataset_dir": "datasets/pendulum_smoke_v1"},
    "newtonian_dynamics": {"config": "configs/pendulum_smoke.yaml",
                           "dataset_dir": "datasets/pendulum_smoke_v1"},
    "symbolic_regression": {"config": "configs/pendulum_smoke.yaml",
                            "dataset_dir": "datasets/pendulum_smoke_v1"},
    "robot_dynamics": {"config": "configs/pendulum_smoke.yaml",
                       "dataset_dir": "datasets/pendulum_smoke_v1"},
    "scientific_ml": {"config": "configs/pendulum_smoke_ood.yaml",
                      "dataset_dir": "datasets/pendulum_smoke_ood_v1"},
}

_SEALED: dict[str, dict[str, str]] = {
    "pendulum_dynamics": {
        "failure_mode": "dynamics_error_accumulation",
        "root_cause": "latent_dynamics_undertrained",
        "correct_key": "train.lr",
        "placebo_key": "train.weight_decay",
        "objective": "Reduce multi-step latent dynamics prediction error on "
                     "the frozen damped-pendulum world model under matched "
                     "research budgets.",
    },
    "newtonian_dynamics": {
        "failure_mode": "parameter_identifiability_loss",
        "root_cause": "insufficient_representation_capacity",
        "correct_key": "model.predictor_hidden",
        "placebo_key": "jepa.ema_momentum",
        "objective": "Improve identification of governing dynamics from the "
                     "frozen trajectories without changing the data.",
    },
    "symbolic_regression": {
        "failure_mode": "conservation_signal_weak",
        "root_cause": "weak_prediction_objective",
        "correct_key": "jepa.pred_weight",
        "placebo_key": "train.weight_decay",
        "objective": "Strengthen the latent signal supporting compact symbolic "
                     "recovery of the governing law from frozen trajectories.",
    },
    "robot_dynamics": {
        "failure_mode": "long_horizon_state_drift",
        "root_cause": "latent_dynamics_undertrained",
        "correct_key": "jepa.ema_momentum",
        "placebo_key": "train.weight_decay",
        "objective": "Reduce long-horizon state drift of the frozen dynamics "
                     "surrogate under a control-oriented evaluation.",
    },
    "scientific_ml": {
        "failure_mode": "ood_regime_shift",
        "root_cause": "insufficient_representation_capacity",
        "correct_key": "model.latent_dim",
        "placebo_key": "jepa.pred_weight",
        "objective": "Improve latent-dynamics generalisation on an "
                     "independently frozen regime-shift split.",
    },
}

NEUTRAL_OBJECTIVE = (
    "Reduce multi-step latent prediction error on the frozen dynamics dataset "
    "under a fixed research budget. Use the recorded trajectory and evidence "
    "to identify a measurable failure and, if justified, propose one controlled "
    "change from the allowed parameter schema."
)

ALLOWED_CONFIG_SCHEMA: tuple[dict[str, str], ...] = (
    {"key": "train.lr", "type": "float", "range": "(0, 1]"},
    {"key": "train.weight_decay", "type": "float", "range": "[0, 1)"},
    {"key": "model.latent_dim", "type": "int", "range": "[1, 64]"},
    {"key": "model.predictor_hidden", "type": "int-list", "range": "positive widths"},
    {"key": "jepa.pred_weight", "type": "float", "range": "(0, 100]"},
    {"key": "jepa.ema_momentum", "type": "float", "range": "(0, 1)"},
)

LEAK_KEYS = frozenset({
    "ground_truth_root_cause", "correct_intervention",
    "distractor_intervention", "failure_mode", "task_family",
    "hidden_test_split", "valid_interventions", "sealed_interventions",
    "expected_solution", "placebo_intervention", "candidate_intervention",
})


class GroundTruthLeak(RuntimeError):
    """Raised when sealed information reaches an agent-facing payload."""


@dataclass(frozen=True)
class PublicTask:
    """Internal record; agent-facing data comes from agent_view only."""

    task_id: str
    task_family: str
    task_description: str
    dataset_reference: str
    train_split: str
    validation_split: str
    hidden_test_split: str
    failure_mode: str
    sealed_interventions: tuple[dict[str, Any], ...]
    primary_metric: str
    budget: dict[str, Any]
    engine_config: str
    dataset_version: str
    seed: int
    objective: str
    metric_direction: str = "minimize"

    @property
    def agent_task_id(self) -> str:
        return "t_" + hashlib.sha256(self.task_id.encode("utf-8")).hexdigest()[:12]

    @property
    def independence_unit(self) -> str:
        return f"{self.task_family}|{self.dataset_version}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sealed_interventions"] = [dict(x) for x in self.sealed_interventions]
        return d

    def agent_view(self) -> dict[str, Any]:
        """The ONLY representation given to Research/Diagnosis agents."""
        d = {
            "agent_task_id": self.agent_task_id,
            "objective": NEUTRAL_OBJECTIVE,
            "dataset_reference": self.dataset_reference,
            "train_split": self.train_split,
            "validation_split": self.validation_split,
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
            "budget": dict(self.budget),
            "allowed_config_schema": [dict(x) for x in ALLOWED_CONFIG_SCHEMA],
            "view": "agent_p0.1",
        }
        assert_no_ground_truth(d)
        return d

    def manifest_view(self) -> dict[str, Any]:
        d = self.to_dict()
        d["task_sha256"] = sha256_json(d)
        d["independence_unit"] = self.independence_unit
        return d


@dataclass(frozen=True)
class GroundTruth:
    task_id: str
    ground_truth_root_cause: str
    correct_intervention: str
    distractor_intervention: str
    falsification_condition: str
    independence_unit: str = ""
    evidence_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_split_refs(scinet_dir: str | Path, dataset_dir: str) -> dict[str, str]:
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
    tasks: list[PublicTask] = []
    for family in FAMILIES:
        source = ENGINE_SOURCES[family]
        refs = read_split_refs(scinet_dir, source["dataset_dir"])
        sealed = _SEALED[family]
        for seed in SEEDS_PER_FAMILY:
            task_id = f"{BENCHMARK_ID}::{family}::s{seed}"
            interventions = (
                {"key": sealed["correct_key"], "role": "sealed_candidate"},
                {"key": sealed["placebo_key"], "role": "sealed_placebo"},
            )
            tasks.append(PublicTask(
                task_id=task_id, task_family=family,
                task_description=sealed["objective"],
                dataset_reference=f"{source['dataset_dir']}@{refs['dataset_version']}",
                train_split=refs["train"], validation_split=refs["validation"],
                hidden_test_split=refs["test"], failure_mode=sealed["failure_mode"],
                sealed_interventions=interventions, primary_metric="latent_mse@1",
                budget={"experiments": 2, "max_jobs": 8,
                        "max_total_seconds": 900.0, "epochs": 1,
                        "paired_seeds": [seed]},
                engine_config=source["config"],
                dataset_version=refs["dataset_version"], seed=seed,
                objective=sealed["objective"],
            ))
    assert len(tasks) == 20
    return tasks


def build_ground_truth(tasks: list[PublicTask]) -> dict[str, GroundTruth]:
    out: dict[str, GroundTruth] = {}
    for task in tasks:
        sealed = _SEALED[task.task_family]
        out[task.task_id] = GroundTruth(
            task_id=task.task_id,
            ground_truth_root_cause=sealed["root_cause"],
            correct_intervention=sealed["correct_key"],
            distractor_intervention=sealed["placebo_key"],
            falsification_condition=(
                f"No significant paired improvement on {task.primary_metric} "
                f"when only {sealed['correct_key']} is changed under seed "
                f"{task.seed}."),
            independence_unit=task.independence_unit,
            evidence_note="Author-sealed before execution; never shown to agents.",
        )
    return out


class GroundTruthStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def seal(self, truth: dict[str, GroundTruth]) -> dict[str, Any]:
        obj = {"version": BENCHMARK_VERSION, "experiment_id": EXPERIMENT_ID,
               "records": [truth[k].to_dict() for k in sorted(truth)]}
        digest = sha256_json(obj)
        self.path.write_text(json.dumps(
            {**obj, "ground_truth_sha256": digest}, indent=2, sort_keys=True,
            ensure_ascii=False), encoding="utf-8")
        return {"path": str(self.path), "ground_truth_sha256": digest,
                "n_records": len(truth)}

    def load(self, expected_sha256: str = "") -> dict[str, GroundTruth]:
        obj = json.loads(self.path.read_text(encoding="utf-8"))
        records = obj.get("records", [])
        digest = sha256_json({"version": obj.get("version"),
                              "experiment_id": obj.get("experiment_id"),
                              "records": records})
        if expected_sha256 and digest != expected_sha256:
            raise GroundTruthLeak("sealed ground truth SHA mismatch")
        return {str(r["task_id"]): GroundTruth(**r) for r in records}


def assert_no_ground_truth(obj: Any, path: str = "$") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in LEAK_KEYS:
                raise GroundTruthLeak(f"leaked field {path}.{key}")
            assert_no_ground_truth(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            assert_no_ground_truth(value, f"{path}[{index}]")


def task_manifest(tasks: list[PublicTask], *, ground_truth_sha256: str = "",
                  experiment_id: str = EXPERIMENT_ID) -> dict[str, Any]:
    body = {
        "experiment_id": experiment_id, "benchmark_id": BENCHMARK_ID,
        "version": BENCHMARK_VERSION, "n_executions": len(tasks),
        "n_independent_units": len({t.independence_unit for t in tasks}),
        "n_seed_replicates_per_unit": len(SEEDS_PER_FAMILY),
        "independence_unit": INDEPENDENCE_UNIT, "families": list(FAMILIES),
        "tasks": [t.manifest_view() for t in tasks],
        "ground_truth_sha256": ground_truth_sha256,
        "ground_truth_separately_sealed": True,
        "agent_view_fields": sorted(tasks[0].agent_view().keys()),
    }
    body["manifest_sha256"] = sha256_json(body)
    return body


__all__ = [
    "BENCHMARK_ID", "EXPERIMENT_ID", "BENCHMARK_VERSION", "FAMILIES",
    "SEEDS_PER_FAMILY", "INDEPENDENCE_UNIT", "ALLOWED_CONFIG_SCHEMA",
    "GroundTruthLeak", "PublicTask", "GroundTruth", "GroundTruthStore",
    "build_tasks", "build_ground_truth", "read_split_refs",
    "assert_no_ground_truth", "task_manifest", "sha256_json",
]
