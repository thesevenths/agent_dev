"""M7.3 Research Memory benchmark dataset builder.

Verified-failure memories are curated from the M7.1 audit (not from LLM output).
Ground truth is stored separately. Shuffled and equal-token controls are built
with matched context length so Memory ON cannot win merely by receiving more text.
"""
from __future__ import annotations

import json
from pathlib import Path

from research_runtime.harness_rsi.identity import freeze_json, sha256_json

MEMORIES = [
    {"memory_id": "mem_single_seed", "memory_type": "VERIFIED_FAILURE", "domain": "world_models",
     "claim": "Single-seed validation improvement is not evidence of generalization.",
     "conditions": ["one seed", "validation split only"], "limitations": ["not replicated"],
     "applicability": ["world_model", "model_based_rl"], "confidence": 0.95,
     "evidence_shas": ["m7_1_signal_001"], "verification_status": "VERIFIED"},
    {"memory_id": "mem_capacity_confound", "memory_type": "VERIFIED_FAILURE", "domain": "world_models",
     "claim": "Improvement from extra model capacity cannot be attributed to a method.",
     "conditions": ["capacity changed", "no matched control"],
     "limitations": ["requires capacity-matched arm"],
     "applicability": ["world_model", "ablation"], "confidence": 0.95,
     "evidence_shas": ["m7_1_signal_002"], "verification_status": "VERIFIED"},
    {"memory_id": "mem_validation_leak", "memory_type": "VERIFIED_FAILURE", "domain": "world_models",
     "claim": "Using validation feedback to redesign until it improves contaminates selection.",
     "conditions": ["iterative tuning", "same validation split"],
     "limitations": ["needs new holdout"], "applicability": ["experiment_design"],
     "confidence": 0.95, "evidence_shas": ["m7_1_signal_003"],
     "verification_status": "VERIFIED"},
    {"memory_id": "mem_power_zero", "memory_type": "VERIFIED_FAILURE", "domain": "statistics",
     "claim": "Pilot with zero independent families cannot support power analysis.",
     "conditions": ["n_families=0", "no pilot effects"], "limitations": ["underpowered"],
     "applicability": ["statistics", "experiment_design"], "confidence": 0.95,
     "evidence_shas": ["m7_1_signal_004"], "verification_status": "VERIFIED"},
    {"memory_id": "mem_novelty_jaccard", "memory_type": "VERIFIED_FAILURE", "domain": "literature",
     "claim": "Lexical Jaccard zero-overlap with abstract-only corpus is not novelty.",
     "conditions": ["abstract only", "lexical screening"],
     "limitations": ["no full text"], "applicability": ["novelty", "literature"],
     "confidence": 0.95, "evidence_shas": ["m7_1_signal_005"],
     "verification_status": "VERIFIED"},
    {"memory_id": "mem_llm_self_review", "memory_type": "VERIFIED_FAILURE", "domain": "scientific_judgment",
     "claim": "LLM proposal accepted by the same model is not independent validation.",
     "conditions": ["same model", "self review"], "limitations": ["model identity must be recorded"],
     "applicability": ["critic", "scientific_judgment"], "confidence": 0.9,
     "evidence_shas": ["m7_1_signal_006"], "verification_status": "VERIFIED"},
    {"memory_id": "mem_seed_family", "memory_type": "VERIFIED_FAILURE", "domain": "statistics",
     "claim": "Different random seeds within one family are not independent families.",
     "conditions": ["same task family", "multiple seeds"],
     "limitations": ["clustering remains"], "applicability": ["statistics"],
     "confidence": 0.95, "evidence_shas": ["m7_1_signal_007"],
     "verification_status": "VERIFIED"},
    {"memory_id": "mem_unverified_memory", "memory_type": "VERIFIED_FAILURE", "domain": "memory",
     "claim": "INCONCLUSIVE results must not be promoted to Verified Memory.",
     "conditions": ["inconclusive", "no replication"], "limitations": ["candidate tier only"],
     "applicability": ["memory", "claim_verification"], "confidence": 0.95,
     "evidence_shas": ["m7_1_signal_008"], "verification_status": "VERIFIED"},
    {"memory_id": "mem_holdout", "memory_type": "VERIFIED_FAILURE", "domain": "experiment_design",
     "claim": "Using holdout feedback for candidate tuning invalidates confirmation.",
     "conditions": ["holdout reused"], "limitations": ["needs new independent holdout"],
     "applicability": ["holdout", "rsi"], "confidence": 0.95,
     "evidence_shas": ["m7_1_signal_009"], "verification_status": "VERIFIED"},
    {"memory_id": "mem_value_defaults", "memory_type": "VERIFIED_FAILURE", "domain": "scientific_judgment",
     "claim": "Research Value from field presence and default EIG has no discriminative power.",
     "conditions": ["field presence", "default priors"], "limitations": ["NOT_ASSESSED required"],
     "applicability": ["research_value", "eig"], "confidence": 0.95,
     "evidence_shas": ["m7_1_signal_010"], "verification_status": "VERIFIED"},
]

SCENARIOS = [
    ("s01", "A team reports improvement using one seed on validation only. What experiment should be run next?", ["mem_single_seed"], "NO", "single seed"),
    ("s02", "A candidate adds more epochs and reports lower validation error. What control is required?", ["mem_capacity_confound"], "NO", "capacity"),
    ("s03", "A researcher has repeatedly tuned on the same validation split and now reports a gain. What is the flaw?", ["mem_validation_leak"], "NO", "contamination"),
    ("s04", "A power analysis reports n_families=0 but claims adequate power. How should this be labeled?", ["mem_power_zero"], "NO", "power"),
    ("s05", "A novelty assessor finds no abstract token overlap and claims likely novel. What verdict is justified?", ["mem_novelty_jaccard"], "NO", "novelty"),
    ("s06", "A model generates a hypothesis and the same model rates it acceptable. Is this independent validation?", ["mem_llm_self_review"], "NO", "independence"),
    ("s07", "Three seeds on one task are called three independent families. Is the design valid?", ["mem_seed_family"], "NO", "family"),
    ("s08", "An INCONCLUSIVE result is proposed for Verified Memory. Should it be promoted?", ["mem_unverified_memory"], "NO", "memory"),
    ("s09", "A candidate was selected after inspecting hidden holdout feedback. Is the holdout result confirmatory?", ["mem_holdout"], "NO", "holdout"),
    ("s10", "Four questions all receive identical priority from field-presence scores. Is the value assessment valid?", ["mem_value_defaults"], "NO", "value"),
]

NEUTRAL = ("General background: scientific claims need clear definitions, "
           "measurement procedures, documented assumptions and traceable records. "
           "Reports should state uncertainties and avoid unsupported conclusions. ")


def _pad(text: str, target: int = 900) -> str:
    while len(text) < target:
        text += NEUTRAL
    return text[:target]


def main() -> None:
    root = Path("M7_3_MEMORY")
    (root / "scenarios").mkdir(parents=True, exist_ok=True)
    (root / "ground_truth").mkdir(parents=True, exist_ok=True)
    (root / "memory_snapshots").mkdir(parents=True, exist_ok=True)
    by_id = {m["memory_id"]: m for m in MEMORIES}
    for idx, (sid, question, relevant, expected_repeat, keyword) in enumerate(SCENARIOS):
        chosen = [by_id[mid] for mid in relevant]
        shuffled = [by_id[MEMORIES[(idx + 3) % len(MEMORIES)]["memory_id"]]]
        case = {"scenario_id": sid, "question": question, "domain": "scientific_method"}
        gt = {"scenario_id": sid, "relevant_memory_ids": relevant,
              "expected_repeat_failure": expected_repeat,
              "expected_keyword": keyword,
              "shuffled_memory_ids": [m["memory_id"] for m in shuffled],
              "ground_truth_sha256": ""}
        gt["ground_truth_sha256"] = sha256_json(
            {k: v for k, v in gt.items() if k != "ground_truth_sha256"})
        freeze_json(case, root / "scenarios" / f"{sid}.json")
        freeze_json(gt, root / "ground_truth" / f"{sid}.json")
        freeze_json({
            "arm_on": _pad(json.dumps(chosen, sort_keys=True)),
            "arm_shuffled": _pad(json.dumps(shuffled, sort_keys=True)),
            "arm_equal_token": _pad(NEUTRAL * 6),
            "arm_off": ""}, root / "memory_snapshots" / f"{sid}.json")
    freeze_json({"version": "m7_3_memory_v1", "n_memories": len(MEMORIES),
                 "n_scenarios": len(SCENARIOS),
                 "arms": ["OFF", "ON", "SHUFFLED", "EQUAL_TOKEN"],
                 "expert_review": "NOT_PERFORMED"},
                root / "benchmark_manifest.json")
    print(json.dumps({"n_memories": len(MEMORIES),
                      "n_scenarios": len(SCENARIOS)}))


if __name__ == "__main__":
    main()
