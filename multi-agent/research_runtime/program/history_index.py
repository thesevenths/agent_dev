"""ARR v1.0 WP-02: cross-round research history index.

Read-only over persisted program state + discovery artifacts. Provides the
evidence-backed inputs for next-round planning (never hardcoded scripts):
prior verdicts, failures, claims, memories and untested gaps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..harness_rsi.identity import sha256_json


class ResearchHistoryIndex:
    def __init__(self, *, program_state: Any):
        self.state = program_state

    def rounds(self) -> list[dict[str, Any]]:
        return list(self.state.obj.get("rounds", []))

    def verdict_summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rounds():
            v = str(r.get("scientific_verdict", "NOT_ASSESSED"))
            out[v] = out.get(v, 0) + 1
        return out

    def failures(self) -> list[dict[str, Any]]:
        return [{"round_id": r["round_id"],
                 "reason": (r.get("failure") or {}).get("reason", ""),
                 "evidence_refs": r.get("evidence_refs", [])}
                for r in self.rounds()
                if r.get("state") in ("FAILED", "BLOCKED")
                or (r.get("failure") or {}).get("reason")]

    def memory_refs(self) -> list[str]:
        refs: list[str] = []
        for r in self.rounds():
            refs.extend(r.get("input_memory_refs", []) or [])
            refs.extend(r.get("claim_refs", []) or [])
        # deterministic de-dup
        return sorted(set(refs))

    def untested_gaps(self) -> list[dict[str, Any]]:
        """Questions that were planned but never produced a terminal verdict."""
        return [{"round_id": r["round_id"],
                 "objective": r.get("objective", ""),
                 "family_id": r.get("family_id", "")}
                for r in self.rounds()
                if r.get("scientific_verdict", "NOT_ASSESSED")
                in ("NOT_ASSESSED", "INCONCLUSIVE")]

    def last_round_artifacts(self, *, root: Path) -> dict[str, Any]:
        """Read actual on-disk artifacts of the last completed round."""
        last = self.state.last_round()
        if not last:
            return {}
        rid = last["round_id"]
        base = root / rid
        out: dict[str, Any] = {}
        for name in ("claims", "memory", "artifacts", "experiments"):
            d = base / name
            if d.is_dir():
                out[name] = [str(p) for p in sorted(d.rglob("*.json"))[:50]]
        return out

    def planning_digest(self, *, root: Path) -> dict[str, Any]:
        """A single evidence-backed digest consumed by next-round planning."""
        payload = {
            "verdicts": self.verdict_summary(),
            "failures": self.failures(),
            "memory_refs": self.memory_refs(),
            "untested_gaps": self.untested_gaps(),
            "last_round_artifacts": self.last_round_artifacts(root=root),
            "n_rounds": len(self.rounds()),
        }
        return {"digest": payload, "digest_sha": sha256_json(payload)}


__all__ = ["ResearchHistoryIndex"]
