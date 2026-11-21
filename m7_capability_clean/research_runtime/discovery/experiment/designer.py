"""M7 -- discriminating experiment designer + immutable preregistration.

The design answers: which experiment distinguishes H1/H2/H3/H0, what supports
or refutes each, what is held fixed, how many INDEPENDENT FAMILIES (not seeds),
MME, power, stopping rules and falsification rules. Preregistration is frozen
before any result and re-read from disk before promotion.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...harness_rsi.identity import freeze_json, sha256_json
from ...harness_rsi.statistics_v2 import family_power_analysis

STAGES = ("development", "selection", "hidden_holdout",
          "confirmatory_replication")


@dataclass
class ExperimentProtocol:
    experiment_id: str
    question_id: str
    hypothesis_ids: list[str]
    baseline: str
    treatment: str
    controls: list[str]
    ablations: list[str]
    dataset_manifest_sha: str
    family_manifest_sha: str
    split_manifest_sha: str
    primary_metric: str
    secondary_metrics: list[str]
    metric_direction: str
    minimum_meaningful_effect: float
    statistical_unit: str
    power_analysis: dict[str, Any]
    random_seeds: list[int]
    budget: dict[str, Any]
    stopping_rules: list[str]
    falsification_rules: list[str]
    preregistration_sha: str = ""
    stages: list[str] = field(default_factory=lambda: list(STAGES))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["preregistration_sha"] = self.preregistration_sha or sha256_json(d)
        return d

    @property
    def sha256(self) -> str:
        return sha256_json({**self.to_dict(), "preregistration_sha": ""})


def design_experiment(*, question: Any, hypotheses: list[Any],
                      family_manifest_sha: str,
                      split_manifest_sha: str,
                      dataset_manifest_sha: str,
                      primary_metric: str,
                      mme: float,
                      pilot_effects: dict[str, float] | None = None,
                      alpha: float = 0.05,
                      target_power: float = 0.8,
                      n_families_planned: int = 8,
                      seeds: list[int] | None = None) -> ExperimentProtocol:
    """Build a controlled protocol; power uses PILOT family variance, not guesses."""
    power = family_power_analysis(pilot_effects or {}, alpha=alpha,
                                  target_power=target_power)
    eid = "exp_" + hashlib.sha256(
        f"{question.question_id}|{primary_metric}|{mme}".encode()).hexdigest()[:14]
    return ExperimentProtocol(
        experiment_id=eid, question_id=question.question_id,
        hypothesis_ids=[h.hypothesis_id for h in hypotheses],
        baseline="frozen baseline harness + baseline world model",
        treatment="action-conditioned treatment (single controlled change)",
        controls=["capacity-matched baseline", "same dataset manifest",
                  "same training budget", "paired seeds"],
        ablations=["remove action conditioning", "remove multi-step loss",
                   "ID-only evaluation"],
        dataset_manifest_sha=dataset_manifest_sha,
        family_manifest_sha=family_manifest_sha,
        split_manifest_sha=split_manifest_sha,
        primary_metric=primary_metric,
        secondary_metrics=["state_mse@5", "state_mse@10",
                           "ood_prediction_error", "compute_cost"],
        metric_direction="minimize" if "mse" in primary_metric else "maximize",
        minimum_meaningful_effect=mme,
        statistical_unit="task_family",
        power_analysis={"pilot": power,
                        "planned_independent_families": n_families_planned,
                        "alpha": alpha, "target_power": target_power,
                        "seed_note": "seeds are NOT independent families"},
        random_seeds=list(seeds or [42, 43, 44, 45, 46]),
        budget={"max_campaign_rounds": 1, "max_jobs": 12,
                "max_total_seconds": 1800.0},
        stopping_rules=["stop when preregistered family count reached",
                        "stop on infrastructure failure (record BLOCKED)",
                        "never stop early for favorable result"],
        falsification_rules=["holdout improvement below MME -> refuted",
                             "opposite direction on >50% families -> refuted",
                             "counter-evidence is append-only, never deleted"])


def freeze_preregistration(protocol: ExperimentProtocol,
                           path: str | Path) -> dict[str, Any]:
    obj = protocol.to_dict()
    frozen = freeze_json({**obj, "preregistration_sha": protocol.sha256},
                         path)
    # file_sha256 is the digest of the exact bytes on disk; keep it separate
    # from the content identity (protocol.sha256).
    return {**obj, "file_sha256": frozen["sha256"],
            "preregistration_sha": protocol.sha256}


def verify_preregistration(protocol: ExperimentProtocol, path: str | Path
                           ) -> dict[str, Any]:
    from ...harness_rsi.identity import sha256_file
    if not Path(path).is_file():
        return {"ok": False, "reasons": ["preregistration_file_missing"]}
    # Recompute the CONTENT identity from disk; a tampered file changes it.
    on_disk = json.loads(Path(path).read_text(encoding="utf-8"))
    # Identity is defined over the protocol with an EMPTY sha placeholder
    # (same convention as compute in ExperimentProtocol.sha256).
    on_disk["preregistration_sha"] = ""
    if sha256_json(on_disk) != protocol.sha256:
        return {"ok": False, "reasons": ["preregistration_content_mismatch"]}
    return {"ok": True, "reasons": [], "sha256": protocol.sha256}


__all__ = ["STAGES", "ExperimentProtocol", "design_experiment",
           "freeze_preregistration", "verify_preregistration"]
