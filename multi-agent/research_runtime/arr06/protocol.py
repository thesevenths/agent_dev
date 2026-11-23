"""ARR-06 preregistration (frozen before formal tracks)."""
from __future__ import annotations

import json
from pathlib import Path

from .core import sha256_json, now_iso

GATES = {
    "rq1_max_ece": 0.15, "rq1_min_info_gain": 0.02,
    "rq2_mme": 0.05, "rq3_mme": 0.05, "rq3_max_negative": 0.15,
    "rq4_min_promotions": 1, "rq5_mme": 0.10,
}

FROZEN = {
    "campaign_id": "ARR-06-2026-001", "experiment_id": "ARR_RESEARCH_CAPABILITY_06",
    "alpha": 0.05, "confidence": 0.95, "gates": GATES,
    "independent_units": "independent task families; seeds are repeated measurements",
    "tests": ["paired_t", "permutation", "cluster_bootstrap", "holm", "wilson"],
    "multiplicity": "Holm per track primary contrasts",
    "exclusion": ["infrastructure_failure", "config_hash_mismatch", "gt_leak",
                  "protocol_violation"],
    "stopping": "budget caps; no post-hoc sample changes; new protocol version on critical bug",
    "reproducibility_levels": ["R0", "R1", "R2", "R3"],
    "engine_mode": "det (torch threads=1, deterministic algorithms, seed-before-build)",
    "budget": {"max_formal": 700, "max_llm": 600, "max_preflight": 80, "max_replay": 120},
    "tracks": ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5"],
    "historical": "ARR-03/04/05 immutable; source_commit captured at freeze",
}


def freeze(root: Path, source_commit: str) -> dict[str, Any]:
    p = dict(FROZEN)
    p["source_commit"] = source_commit
    p["frozen_at"] = now_iso()
    p["protocol_sha256"] = sha256_json(p)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "protocol.json").write_text(json.dumps(p, indent=2, sort_keys=True,
                                                    ensure_ascii=False), encoding="utf-8")
    (root / "protocol.sha256").write_text(p["protocol_sha256"] + "\n", encoding="utf-8")
    return p


__all__ = ["FROZEN", "GATES", "freeze"]
