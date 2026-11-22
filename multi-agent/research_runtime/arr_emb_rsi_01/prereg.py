"""ARR_EMB_RSI_01 -- preregistration for the authorised pilot.

Frozen BEFORE pilot execution. Full experiment remains NOT AUTHORIZED.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .integrity import sha256_json

EXPERIMENT_ID = "ARR_EMB_RSI_01"
PREREG_VERSION = "arr-emb-rsi-v1-prereg"

FROZEN = {
    "experiment_id": EXPERIMENT_ID,
    "stage": "pilot",
    "full_experiment_authorized": False,
    "branch": "feature/arr-embodied-rsi-01",
    "engine": {"project": "E:/LLM/SciNet++", "branch": "feature/p2-03a-action-jepa",
               "head": "c6f66c3"},
    "concurrency": 1,
    "seed_policy": "seed is a repeated measurement, never an independent unit",
    "independent_units": ["E01|capacity", "E02|coverage", "E03|shift"],
    "blocked_mechanisms": {
        "E04": "Oracle rollout_steps=[1,2,3] produced no measurable improvement "
               "over [1,2] on horizon ratio; benchmark sensitivity gate failed."
    },
    "pilot_reduction_justification": (
        "Original proposal: 4 mechanisms x 1 variant x 2 seeds x 5 arms = 40 runs. "
        "E04 is blocked by the Oracle STOP rule. Each real engine run costs "
        "2-3 CLI jobs; to stay within the authorised pilot budget we use 1 seed "
        "per remaining independent mechanism x 5 arms = 15 real runs. Seeds are "
        "not independent units, so this reduction does not change the unit of "
        "inference; it reduces replicate precision only. FULL EXPERIMENT IS NOT "
        "AUTHORIZED from this pilot."),
    "arms": ["base", "candidate", "random", "shuffled", "oracle"],
    "primary_metric": "validation_latent_mse@1",
    "ood_metric": "ood_latent_mse@1",
    "direction": "minimize",
    "mme_absolute": 0.02,
    "alpha": 0.05,
    "min_independent_units_for_causal": 6,
    "min_diag_accuracy": 0.75,
    "min_key_accuracy": 0.75,
    "min_evidence_precision": 0.80,
    "max_confidence_ece": 0.15,
    "required_artifact_integrity": True,
    "required_gt_isolation": True,
    "required_config_application_rate": 1.0,
    "retry_rules": "failed arm retried explicitly with archived evidence; "
                   "completed arms never silently rerun",
    "exclusion_criteria": ["infrastructure failure in any arm of a unit",
                           "unverifiable artifact claim",
                           "ground truth leakage"],
    "statistical_tests": ["paired t on unit-level means",
                          "Holm correction across candidate-vs-control contrasts",
                          "bootstrap CI", "effect size dz"],
    "five_verdicts": ["engineering", "benchmark_integrity", "diagnosis",
                      "intervention", "causal_improvement"],
    "stop_rules": ["GT leakage", "invalid independence", "unverifiable evidence",
                   "Oracle no effect", "config not applied", "budget breach",
                   "infra failure", "underpowered statistical design"],
}


def freeze_prereg(path: str | Path) -> dict[str, Any]:
    payload = dict(FROZEN)
    payload["preregistration_sha256"] = sha256_json(payload)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True,
                            ensure_ascii=False), encoding="utf-8")
    return payload


__all__ = ["EXPERIMENT_ID", "PREREG_VERSION", "FROZEN", "freeze_prereg"]
