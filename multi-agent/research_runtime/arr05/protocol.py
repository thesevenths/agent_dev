"""ARR-05 preregistration (frozen before formal tracks)."""
from __future__ import annotations

import json
from pathlib import Path

from .core import sha256_json

GATES = {
    "m1_min_accuracy_gain": 0.03, "m1_max_ece": 0.15,
    "m1_min_info_gain": 0.02, "m1_min_abstention_precision": 0.6,
    "m2_mme": 0.05, "m3_mme": 0.05,
    "m3_max_negative_transfer": 0.15, "m3_min_applicability": 0.7,
    "rq_mme": 0.10, "rq_min_falsifiability": 0.7,
}

FROZEN = {
    "campaign_id": "ARR-05-2026-001", "experiment_id": "ARR_RESEARCH_CAPABILITY_05",
    "alpha": 0.05, "confidence": 0.95, "gates": GATES,
    "independent_units": "independent task families; seeds are repeated measurements",
    "statistical_tests": ["paired_t", "permutation", "cluster_bootstrap", "holm",
                          "wilson", "effect_size_dz"],
    "multiple_comparison": "Holm across primary contrasts per track",
    "exclusion": ["infrastructure_failure", "config_mismatch", "artifact_hash_mismatch",
                  "gt_leak", "protocol_violation"],
    "stopping": "budget caps; preflight STOP; no post-hoc sample size changes",
    "calibration": "temperature scaling fit on calibration split ONLY; "
                   "heldout/OOD used exclusively for evaluation",
    "splits": {"calibration": 26, "heldout": 26, "ood": 26},
    "causes": 5,
    "budget": {"max_formal": 1500, "max_llm": 2500, "max_preflight": 60,
               "max_replay": 200},
    "tracks": ["M1", "M2", "M3", "RQ", "RSI"],
    "historical_baseline": "ARR-04-2026-001 (immutable; reproduction verified "
                           "from committed raw records without LLM calls)",
}


def freeze(root: str | Path, manifest: dict[str, Any],
           power: dict[str, Any]) -> dict[str, Any]:
    p = dict(FROZEN)
    p["frozen_at"] = __import__("research_runtime.arr05.core", fromlist=["now_iso"]).now_iso()
    p["task_manifest"] = manifest
    p["power_analysis"] = power
    p["protocol_sha256"] = sha256_json(p)
    root = Path(root)
    (root / "protocol.json").write_text(json.dumps(p, indent=2, sort_keys=True,
                                                    ensure_ascii=False), encoding="utf-8")
    (root / "protocol.sha256").write_text(p["protocol_sha256"] + "\n", encoding="utf-8")
    (root / "frozen_task_manifest.json").write_text(json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return p


__all__ = ["FROZEN", "GATES", "freeze"]
