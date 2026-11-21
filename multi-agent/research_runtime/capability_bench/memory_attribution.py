"""ARR v1.0 WP-06: memory attribution + negative-transfer tracking.

The M7.3 benchmark measured retrieval and harmful usage but conflated
"memory was retrieved" with "memory improved research". This module tracks the
SEVEN DISTINCT states required by WP-06/§7.3 and measures negative transfer.
It is a thin, append-only ledger; it does NOT change memory_bench's four arms.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..harness_rsi.identity import freeze_json, sha256_json

ATTRIBUTION_STATES = ("memory_retrieved", "memory_injected", "memory_cited",
                      "memory_applied", "memory_changed_decision",
                      "memory_improved_outcome", "memory_caused_harm")
TERMINAL_OUTCOMES = ("memory_improved_outcome", "memory_caused_harm")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class MemoryAttributionLedger:
    def __init__(self, *, root: str | Path, run_id: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.path = self.root / f"attribution_{run_id}.json"
        import json
        if self.path.is_file():
            self.obj = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.obj = {"version": "arr_v1_memory_attribution_v1",
                        "run_id": run_id, "traces": [], "created_at": _now()}
            self._save()

    def _save(self) -> str:
        self.obj["updated_at"] = _now()
        meta = freeze_json(self.obj, self.path)
        self.obj["attribution_sha"] = meta["sha256"]
        return meta["sha256"]

    def record(self, *, scenario_id: str, arm: str, memory_id: str,
               state: str, evidence: dict[str, Any] | None = None) -> None:
        if state not in ATTRIBUTION_STATES:
            raise ValueError(f"unknown attribution state {state!r}")
        self.obj["traces"].append({
            "scenario_id": scenario_id, "arm": arm, "memory_id": memory_id,
            "state": state, "evidence_sha": sha256_json(evidence or {}),
            "at": _now()})
        self._save()

    def trace(self, *, scenario_id: str, memory_id: str) -> list[str]:
        return [t["state"] for t in self.obj["traces"]
                if t["scenario_id"] == scenario_id
                and t["memory_id"] == memory_id]

    def summary(self) -> dict[str, Any]:
        by_arm: dict[str, dict[str, int]] = {}
        for t in self.obj["traces"]:
            arm = t["arm"]
            d = by_arm.setdefault(arm, {s: 0 for s in ATTRIBUTION_STATES})
            d[t["state"]] += 1
        retrieved = sum(d["memory_retrieved"] for d in by_arm.values())
        applied = sum(d["memory_applied"] for d in by_arm.values())
        improved = sum(d["memory_improved_outcome"] for d in by_arm.values())
        harmed = sum(d["memory_caused_harm"] for d in by_arm.values())
        return {
            "by_arm": by_arm,
            "retrieved": retrieved,
            "applied": applied,
            "usage_rate": round(applied / retrieved, 4) if retrieved else None,
            "improved_outcome": improved,
            "caused_harm": harmed,
            "improvement_rate": round(improved / applied, 4) if applied else None,
            "harm_rate": round(harmed / applied, 4) if applied else None,
            "note": "retrieval != usage != outcome; never equate these states",
        }


def negative_transfer_report(*, raw_results_dir: str | Path,
                             ground_truth: dict[str, dict]) -> dict[str, Any]:
    """Measure wrong-domain / stale / contradictory memory usage from GT."""
    import json
    root = Path(raw_results_dir)
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(root.glob("*.json"))]
    wrong_domain = stale = contradictory = 0
    n = 0
    for r in rows:
        if r.get("arm") not in ("ON", "SHUFFLED", "EQUAL_TOKEN"):
            continue
        n += 1
        gt = ground_truth.get(r.get("scenario_id"), {})
        used = set((r.get("prediction") or {}).get("used_memory_ids") or [])
        if used & set(gt.get("shuffled_memory_ids", [])):
            wrong_domain += 1
        if used & set(gt.get("stale_memory_ids", [])):
            stale += 1
        if used & set(gt.get("contradictory_memory_ids", [])):
            contradictory += 1
    return {"n": n, "wrong_domain_use": wrong_domain, "stale_use": stale,
            "contradictory_use": contradictory,
            "wrong_domain_rate": round(wrong_domain / n, 4) if n else None,
            "stale_rate": round(stale / n, 4) if n else None,
            "contradictory_rate": round(contradictory / n, 4) if n else None,
            "note": "rates are descriptive; GT may not declare all categories"}


__all__ = ["ATTRIBUTION_STATES", "TERMINAL_OUTCOMES",
           "MemoryAttributionLedger", "negative_transfer_report"]
