"""ARR_EMB_RSI_01 -- benchmark v1: real, controllable world-model failures.

Every failure mechanism is a real engine configuration/data change verified by
`setup_benchmark.py`. The agent never sees failure_id, ground truth or hidden
test outcome.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .integrity import ArtifactClaim, sha256_json

BENCHMARK_ID = "ARR-Embodied-RSI-v1"
EXPERIMENT_ID = "ARR_EMB_RSI_01"
VERSION = "arr-emb-rsi-v1"

# Oracle values are ENGINE-VERIFIED by setup_benchmark.py; no generic multipliers.
FAILURES: dict[str, dict[str, Any]] = {
    "E01": {
        "mechanism": "Insufficient model capacity: predictor hidden widths too small "
                     "to represent multi-step latent dynamics.",
        "generator": "pendulum",
        "dataset": "datasets/pendulum_smoke_v1",
        "failure_override": {"model.predictor_hidden": [8, 8]},
        "oracle_override": {"model.predictor_hidden": [16, 16]},
        "root_cause": "insufficient_model_capacity",
        "alternatives": ["optimisation_budget_too_small",
                         "dataset_coverage_too_narrow"],
        "expected_metric": "latent_mse@1",
        "expected_direction": "minimize",
        "falsification": "Oracle hidden=[16,16] does not improve validation "
                         "latent_mse@1 over hidden=[8,8] under matched budget.",
    },
    "E02": {
        "mechanism": "Insufficient state coverage: training split reduced while "
                     "hidden test is kept byte-identical.",
        "generator": "pendulum",
        "dataset": "datasets/arr_emb_rsi_coverage_16",
        "failure_override": {"data.train_samples": 16,
                             "data.frozen_dir": "datasets/arr_emb_rsi_coverage_16"},
        "oracle_override": {"data.train_samples": 64,
                            "data.frozen_dir": "datasets/arr_emb_rsi_coverage_64"},
        "root_cause": "insufficient_state_action_coverage",
        "alternatives": ["insufficient_model_capacity", "measurement_noise"],
        "expected_metric": "latent_mse@1",
        "expected_direction": "minimize",
        "falsification": "Oracle train_samples=64 does not improve validation "
                         "latent_mse@1 under a fixed hidden test artifact.",
    },
    "E03": {
        "mechanism": "Dynamics distribution shift: ID-trained model evaluated on "
                     "an independently frozen t_max=14 regime.",
        "generator": "pendulum",
        "dataset": "datasets/pendulum_smoke_v1",
        "failure_override": {"model.predictor_hidden": [8, 8]},
        "oracle_override": {"model.predictor_hidden": [16, 16]},
        "ood_frozen_dir": "datasets/pendulum_smoke_ood_v1",
        "root_cause": "dynamics_distribution_shift",
        "alternatives": ["insufficient_model_capacity", "long_horizon_error_accumulation"],
        "expected_metric": "ood_latent_mse@1",
        "expected_direction": "minimize",
        "falsification": "Oracle does not improve OOD latent_mse@1 under the same "
                         "ID-trained checkpoint budget.",
    },
    "E04": {
        "mechanism": "Long-horizon error accumulation: multi-step prediction loss "
                     "removed from training (rollout_steps=[1]).",
        "generator": "pendulum",
        "dataset": "datasets/pendulum_smoke_v1",
        "failure_override": {"jepa.rollout_steps": [1]},
        "oracle_override": {"jepa.rollout_steps": [1, 2, 3]},
        "root_cause": "long_horizon_error_accumulation",
        "alternatives": ["insufficient_model_capacity", "weak_prediction_objective"],
        "expected_metric": "horizon_ratio@3@1",
        "expected_direction": "minimize",
        "falsification": "Oracle rollout_steps=[1,2,3] does not reduce "
                         "latent_mse@3 / latent_mse@1 vs rollout_steps=[1].",
    },
}


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    failure_id: str
    seed: int
    independent_unit: str
    generator: str
    dataset: str
    failure_override: dict[str, Any]
    oracle_override: dict[str, Any]
    ood_frozen_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def agent_view(self) -> dict[str, Any]:
        """Only an opaque case id and neutral objective reach the agent."""
        from .integrity import assert_no_ground_truth
        view = {"case_id": "case_" + sha256_json(
                    {"id": self.case_id, "salt": "arr_emb_agent_view"})[:12],
                "independent_unit": self.independent_unit,
                "objective": "Identify the dominant failure mechanism from the "
                             "provided evidence and propose one justified intervention.",
                "budget": {"max_jobs": 4, "max_seconds": 900}}
        assert_no_ground_truth(view)
        return view


def build_cases(*, seeds: tuple[int, ...] = (42, 43)) -> list[BenchmarkCase]:
    cases = []
    for fid, spec in FAILURES.items():
        for seed in seeds:
            cases.append(BenchmarkCase(
                case_id=f"{fid}__u{seed}", failure_id=fid, seed=seed,
                independent_unit=f"{fid}|{spec['dataset']}",
                generator=spec["generator"], dataset=spec["dataset"],
                failure_override=dict(spec["failure_override"]),
                oracle_override=dict(spec["oracle_override"]),
                ood_frozen_dir=str(spec.get("ood_frozen_dir") or "")))
    return cases


def sealed_ground_truth(cases: list[BenchmarkCase]) -> dict[str, dict[str, Any]]:
    out = {}
    for case in cases:
        spec = FAILURES[case.failure_id]
        out[case.case_id] = {
            "case_id": case.case_id,
            "failure_id": case.failure_id,
            "ground_truth": spec["root_cause"],
            "correct_intervention": spec["oracle_override"],
            "alternatives": list(spec["alternatives"]),
            "expected_metric": spec["expected_metric"],
            "expected_direction": spec["expected_direction"],
            "falsification_condition": spec["falsification"],
            "hidden_test_result": "SEALED — not exposed to agent",
        }
    return out


def evidence_from_run(*, case: BenchmarkCase, run: dict[str, Any],
                      resolver: Any) -> dict[str, Any]:
    """Build an evidence package from REAL engine artifacts only."""
    claims = []
    for name, path in (run.get("artifact_paths") or {}).items():
        claims.append(ArtifactClaim(
            claim_id=f"{case.case_id}:{name}",
            artifact_path=path,
            artifact_sha256=(run.get("artifact_hashes") or {}).get(name, ""),
            claim_text=f"real engine artifact {name} for run {run.get('run_id')}",
            run_id=str(run.get("run_id") or "")))
    metrics = dict(run.get("metrics") or {})
    if run.get("ood_metrics"):
        metrics.update({f"ood_{k}": v for k, v in run["ood_metrics"].items()})
    m1 = metrics.get("latent_mse@1")
    m3 = metrics.get("latent_mse@3")
    if m1 and m3 and m1 > 0:
        metrics["horizon_ratio@3@1"] = m3 / m1
    config_snapshot = {k: {"expected": v,
                           "actual": _dotted_get(run.get("applied_config") or {}, k)}
                       for k, v in case.failure_override.items()}
    return {"case_id": case.case_id, "independent_unit": case.independent_unit,
            "metrics": metrics, "config_snapshot": config_snapshot,
            "budget_usage": run.get("budget") or {},
            "claims": [c.to_dict() for c in claims],
            "task_label": "unlabelled_dynamics_case",
            "artifact_verification": resolver.verify(claims)}


def _dotted_get(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


__all__ = ["BENCHMARK_ID", "EXPERIMENT_ID", "VERSION", "FAILURES",
           "BenchmarkCase", "build_cases", "sealed_ground_truth",
           "evidence_from_run"]
