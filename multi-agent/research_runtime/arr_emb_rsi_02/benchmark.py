"""ARR_EMB_RSI_02 -- blind task regimes, failure registry, intervention registry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BENCHMARK_ID = "ARR-Embodied-RSI-Blind-v2"
EXPERIMENT_ID = "ARR_EMB_RSI_02"
VERSION = "arr-emb-rsi-blind-v2"

GENERATORS = {
    "pendulum": {"config": "configs/pendulum_smoke.yaml",
                 "dataset": "datasets/arr_emb_rsi_02/pendulum_v1",
                 "ood_dataset": "datasets/arr_emb_rsi_02/pendulum_ood_v1",
                 "cov16_dataset": "datasets/arr_emb_rsi_02/pendulum_cov16",
                 "cov4_dataset": "datasets/arr_emb_rsi_02/pendulum_cov4",
                 "cov64_dataset": "datasets/arr_emb_rsi_02/pendulum_cov64",
                 "t_max": 8.0, "ood_t_max": 14.0,
                 "common": {"data.train_samples": 64, "data.test_samples": 16,
                            "data.validation_samples": 16,
                            "data.adapt_samples": 16, "device": "cpu",
                            "train.epochs": 1, "train.batch_size": 16}},
    "newton": {"config": "configs/newton.yaml",
               "dataset": "datasets/arr_emb_rsi_02/newton_v1",
               "ood_dataset": "datasets/arr_emb_rsi_02/newton_ood_v1",
               "cov16_dataset": "datasets/arr_emb_rsi_02/newton_cov16",
                 "cov4_dataset": "datasets/arr_emb_rsi_02/newton_cov4",
               "cov64_dataset": "datasets/arr_emb_rsi_02/newton_cov64",
               "t_max": 3.0, "ood_t_max": 6.0,
               "common": {"data.train_samples": 64, "data.test_samples": 16,
                          "data.validation_samples": 16,
                          "data.adapt_samples": 16, "device": "cpu",
                          "train.epochs": 1, "train.batch_size": 16}},
    "double_pendulum": {
        "config": "configs/double_pendulum.yaml",
        "dataset": "datasets/arr_emb_rsi_02/double_pendulum_v1",
        "ood_dataset": "datasets/arr_emb_rsi_02/double_pendulum_ood_v1",
        "cov16_dataset": "datasets/arr_emb_rsi_02/double_pendulum_cov16",
                 "cov4_dataset": "datasets/arr_emb_rsi_02/double_pendulum_cov4",
        "cov64_dataset": "datasets/arr_emb_rsi_02/double_pendulum_cov64",
        "t_max": 50.0, "ood_t_max": 80.0,
        "common": {"data.train_samples": 64, "data.test_samples": 16,
                   "data.validation_samples": 16,
                   "data.adapt_samples": 16, "device": "cpu",
                   "train.epochs": 1, "train.batch_size": 16}},
}

MECHANISMS = ("E01", "E02", "E03", "E04")

# Sealed truth: never given to the agent. Oracle uses it by design.
FAILURE_TRUTH: dict[str, dict[str, Any]] = {
    "E01": {"cause": "insufficient_model_capacity",
            "oracle_override": {"model.predictor_hidden": [32, 32]}},
    "E02": {"cause": "insufficient_data_coverage",
            "oracle_override": {"data.train_samples": 64}},
    "E03": {"cause": "dynamics_distribution_shift",
            "oracle_override": {"model.predictor_hidden": [32, 32]}},
    "E04": {"cause": "optimization_instability",
            "oracle_override": {"train.lr": 0.0008}},
}

# Registry: every candidate intervention must be a real engine parameter.
INTERVENTIONS: dict[str, dict[str, Any]] = {
    "insufficient_model_capacity": {
        "id": "IV-CAP-02", "parameter": "model.predictor_hidden",
        "allowed": {"type": "int-list", "min_width": 8, "max_width": 64},
        "mechanism": "increase predictor width when latent prediction is weak"},
    "insufficient_data_coverage": {
        "id": "IV-COV-02", "parameter": "data.train_samples",
        "allowed": {"type": "int", "min": 16, "max": 128},
        "mechanism": "increase training state coverage"},
    "dynamics_distribution_shift": {
        "id": "IV-SHIFT-02", "parameter": "model.predictor_hidden",
        "allowed": {"type": "int-list", "min_width": 8, "max_width": 64},
        "mechanism": "increase capacity to transfer across dynamics regimes"},
    "optimization_instability": {
        "id": "IV-OPT-02", "parameter": "train.lr",
        "allowed": {"type": "float", "min": 0.0001, "max": 0.01},
        "mechanism": "reduce learning rate when optimisation is unstable"},
}

ALLOWED_KEYS = tuple(sorted({v["parameter"] for v in INTERVENTIONS.values()}))


@dataclass(frozen=True)
class Regime:
    regime_id: str
    generator: str
    mechanism: str
    config: str
    dataset: str
    ood_dataset: str
    common: dict[str, Any]
    failure_override: dict[str, Any]
    oracle_override: dict[str, Any]
    sealed_truth: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def blind_view(self) -> dict[str, Any]:
        """Opaque, GT-free view; no generator, mechanism or config identity."""
        import hashlib
        opaque = "regime_" + hashlib.sha256(
            self.regime_id.encode()).hexdigest()[:12]
        return {"regime_id": opaque,
                "objective": "Diagnose the dominant failure from normalized "
                             "experimental evidence and propose one registered "
                             "intervention.",
                "budget": {"max_jobs": 4, "max_seconds": 900}}


def build_regimes() -> list[Regime]:
    regimes = []
    for gen, spec in GENERATORS.items():
        for mech in MECHANISMS:
            common = dict(spec["common"])
            common["data.frozen_dir"] = spec["dataset"]
            common["data.t_max"] = spec["t_max"]
            failure = {}
            if mech == "E01":
                failure = {"model.predictor_hidden": [8, 8]}
            elif mech == "E02":
                failure = {"data.train_samples": 4,
                           "data.frozen_dir": spec["cov4_dataset"],
                           "train.batch_size": 4}
            elif mech == "E03":
                failure = {"model.predictor_hidden": [8, 8]}
            else:
                failure = {"train.lr": 0.2}
            truth = FAILURE_TRUTH[mech]
            oracle = dict(truth["oracle_override"])
            if mech == "E02":
                oracle["data.frozen_dir"] = spec["cov64_dataset"]
                oracle["data.train_samples"] = 64
                oracle["train.batch_size"] = 4
            if mech == "E02" and gen != "pendulum":
                # Oracle inversion in real engine: skip, do not fake validity.
                continue
            regimes.append(Regime(
                regime_id=f"{gen}::{mech}", generator=gen, mechanism=mech,
                config=spec["config"], dataset=spec["dataset"],
                ood_dataset=spec["ood_dataset"], common=common,
                failure_override=failure, oracle_override=oracle,
                sealed_truth=dict(truth)))
    return regimes


__all__ = ["BENCHMARK_ID", "EXPERIMENT_ID", "VERSION", "GENERATORS",
           "MECHANISMS", "INTERVENTIONS", "ALLOWED_KEYS", "Regime",
           "build_regimes"]
