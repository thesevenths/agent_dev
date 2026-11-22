"""ARR_EMB_RSI_02 -- preregistration for the blind full campaign."""
from __future__ import annotations

import json
from pathlib import Path

from .integrity import sha256_json

EXPERIMENT_ID = "ARR_EMB_RSI_02"

FROZEN = {
    "experiment_id": EXPERIMENT_ID,
    "stage": "full_campaign",
    "full_experiment_authorized": True,
    "branch": "feature/arr-emb-rsi-02-blind-validation",
    "engine": {"project": "E:/LLM/SciNet++", "branch": "feature/p2-03a-action-jepa",
               "head": "c6f66c3"},
    "concurrency": 3,
    "independent_units": 10,
    "seeds": [42, 43, 44],
    "seed_policy": "seeds are repeated measurements, never independent units",
    "arms": ["base", "candidate", "random", "oracle"],
    "excluded_regimes": {
        "newton::E02": "Oracle inversion: 4 samples beat 64 samples in real engine",
        "double_pendulum::E02": "Oracle inversion: 4 samples beat 64 samples",
    },
    "primary_metric": "normalized_improvement",
    "metric_definition": "(base_metric - arm_metric) / max(abs(base_metric),1e-9)",
    "metric_splits": {"default": "validation latent_mse@1",
                      "E03": "OOD latent_mse@1"},
    "mme_normalized": 0.05,
    "alpha": 0.05,
    "confidence_level": 0.95,
    "statistical_tests": ["paired_t_small_sample", "bootstrap_ci",
                          "permutation_p", "holm_correction", "effect_size_dz"],
    "comparisons": ["candidate_vs_base", "candidate_vs_random"],
    "sensitivity": ["oracle_vs_base per mechanism"],
    "exclusion_criteria": ["infrastructure failure", "artifact hash mismatch",
                           "config application mismatch", "GT leak",
                           "invalid agent intervention"],
    "retry_rules": "failed arm retried once with new run id; old evidence archived",
    "budget": {"model_runs": 120, "max_jobs_per_run": 4,
               "max_seconds_per_run": 900},
    "power_analysis": {
        "n_units": 10, "alpha": 0.05, "dz": 1.0,
        "target_power": 0.80,
        "computed_power": 0.8031,
        "note": "noncentral-t power computed with scipy for n=10 paired units, dz=1.0"
    },
    "evidence_gates": {"artifact_integrity": True, "citation_validity_min": 0.8,
                       "evidence_relevance_min": 0.8, "sufficiency": True},
    "confidence_gates": {"max_ece": 0.15, "brier_reported": True,
                         "reliability_diagram": True},
    "diagnosis_gates": {"min_root_cause_accuracy": 0.70,
                        "min_intervention_key_accuracy": 0.60},
    "five_verdicts": ["engineering", "benchmark_integrity", "diagnosis",
                      "intervention", "causal_improvement"],
    "verdict_rule": "scientific claim requires Diagnosis and Causal Improvement PASS",
}


def freeze_prereg(path: str | Path) -> dict:
    payload = dict(FROZEN)
    payload["preregistration_sha256"] = sha256_json(payload)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True,
                            ensure_ascii=False), encoding="utf-8")
    return payload


__all__ = ["EXPERIMENT_ID", "FROZEN", "freeze_prereg"]
