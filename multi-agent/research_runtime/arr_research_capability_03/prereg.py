"""ARR-03 unified preregistration. Frozen before any full campaign run."""
from __future__ import annotations

import json
from pathlib import Path

from .integrity import sha256_json

EXPERIMENT_ID = "ARR_RESEARCH_CAPABILITY_03"
BRANCH = "feature/arr-research-capability-03"

FROZEN = {
    "experiment_id": EXPERIMENT_ID,
    "campaign_id": EXPERIMENT_ID,
    "branch": BRANCH,
    "stage": "full_campaign",
    "full_experiment_authorized": True,
    "tracks": ["M1_diagnosis", "M2_causal", "M3_transfer"],
    "agent": {
        "generator": "ARR.LLM (research_runtime.llm_client / LLMProposer gateway)",
        "model": "qwen3.6-flash",
        "hardcoded_rules_only_as": "separate Rule-Based baseline controller",
        "agent_may_abstain": True,
    },
    "independent_units": {"M2": 10, "M3": 6, "total": 16},
    "seeds": [42, 43, 44],
    "seed_policy": "seeds are repeated measurements; units are task regimes",
    "m2": {
        "regimes": "10 validated ARR_EMB_RSI_02 state-based regimes",
        "excluded": ["newton::E02", "double_pendulum::E02"],
        "arms": ["base", "candidate", "random", "oracle"],
        "model_runs": 120,
    },
    "m3": {
        "regimes": "6 Reacher-v5 action-conditioned regimes",
        "arms": ["base", "arr_fresh", "arr_transfer", "random", "oracle"],
        "model_runs": 90,
        "environment": "gymnasium Reacher-v5 (MuJoCo simulator only)",
        "world_model": "MLP f_theta(s_t,a_t)->s_t+1",
        "no_real_robot": True, "no_visual_jepa": True,
    },
    "budget": {"formal_runs": 210, "preflight_runs_max": 6, "total_max": 216},
    "mme_normalized": 0.05,
    "alpha": 0.05,
    "confidence_level": 0.95,
    "primary_metric_m2": "normalized improvement in latent_mse@1 (OOD for E03)",
    "primary_metric_m3": "normalized improvement in one_step_mse",
    "statistical_tests": ["paired_t_small_sample", "permutation", "cluster_bootstrap",
                          "holm_correction", "effect_size_dz"],
    "diagnosis_gates": {
        "min_accuracy": 0.65, "min_top2_accuracy": 0.80,
        "max_ece": 0.15, "max_brier": 0.30,
    },
    "causal_gates": {"mme": 0.05, "min_units": 6},
    "transfer_gates": {"mme": 0.05, "min_units": 6},
    "power_analysis": {
        "m2_n_units": 10, "m3_n_units": 6,
        "note": "paired design; M2 n=10 dz=1.0 power~0.80; "
                "M3 n=6 requires larger effects (observed oracle sensitivity 0.73-0.94)",
    },
    "exclusion_rules": ["infrastructure_failure", "artifact_hash_mismatch",
                        "config_mismatch", "gt_leak", "oracle_insensitive_regime"],
    "retry_rules": "failed arm retried once with new run id; old evidence archived",
    "six_verdicts": ["engineering", "benchmark_integrity", "m1_diagnosis",
                     "m2_causal_improvement", "m3_embodied_transfer",
                     "overall_research_capability"],
    "verdict_rule": "overall PASS requires M1 and M2 and M3 each PASS; "
                    "engineering PASS cannot compensate scientific FAIL",
}


def freeze_prereg(path: str | Path) -> dict:
    payload = json.loads(json.dumps(FROZEN))
    payload["preregistration_sha256"] = sha256_json(payload)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True,
                            ensure_ascii=False), encoding="utf-8")
    return payload


__all__ = ["EXPERIMENT_ID", "FROZEN", "freeze_prereg"]
