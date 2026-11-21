"""ARR v1.0 WP-07: versioned ResearchMethod registry.

Thin wrapper over M5 `EvolutionRegistry` (append-only, SHA-bound) — NOT a new
registry engine. Each ResearchMethod has a frozen artifact on disk; the registry
record pins its SHA. MethodCandidates can ONLY be registered with real failure
evidence references and a programmatic SHA verification (never trust an LLM-
generated SHA). Historical methods (including REJECTED) are never deleted.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..harness_rsi.identity import freeze_json, is_sha256, sha256_json
from ..research_store.db import ResearchDatabase
from ..rrsi import EvolutionRegistry

LAYER = "research_method"
PROMOTION_STATUSES = ("PROPOSED", "EVALUATED", "PROMOTED", "REJECTED",
                      "ROLLED_BACK")


class MethodRegistryError(RuntimeError):
    pass


class MethodRegistry:
    def __init__(self, *, db: ResearchDatabase, root: str | Path):
        self.db = db
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.evolution = EvolutionRegistry(db=db)

    def register(self, *, method_id: str, version: str,
                 component_config: dict[str, Any],
                 parent_method_sha: str = "",
                 source_failure_refs: list[str] | None = None,
                 diagnosis: dict[str, Any] | None = None,
                 expected_improvement: str = "",
                 falsification_criteria: list[str] | None = None,
                 side_effect_risks: list[str] | None = None,
                 evaluation_protocol_sha: str = "",
                 status: str = "PROPOSED") -> dict[str, Any]:
        if status not in PROMOTION_STATUSES:
            raise MethodRegistryError(f"unknown method status {status!r}")
        if status in ("PROMOTED", "EVALUATED") and not source_failure_refs:
            raise MethodRegistryError(
                "method_candidate_requires_real_failure_evidence")
        if evaluation_protocol_sha and not is_sha256(evaluation_protocol_sha):
            raise MethodRegistryError("evaluation_protocol_sha must be 64-hex")
        payload = {
            "method_id": method_id, "version": version,
            "parent_method_sha": parent_method_sha,
            "component_config": component_config,
            "source_failure_refs": sorted(set(source_failure_refs or [])),
            "diagnosis": diagnosis or {},
            "expected_improvement": expected_improvement,
            "falsification_criteria": list(falsification_criteria or []),
            "side_effect_risks": list(side_effect_risks or []),
            "evaluation_protocol_sha": evaluation_protocol_sha,
            "promotion_status": status,
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        # Freeze the method artifact; its SHA is the authoritative method_sha.
        frozen = freeze_json(payload, self.root / f"{method_id}_v{version}.json")
        method_sha = frozen["sha256"]
        rec = self.evolution.record(
            experiment_id=f"method_{method_id}_v{version}",
            layer=LAYER, candidate_id=method_id, candidate_sha=method_sha,
            parent_sha=parent_method_sha, status=status,
            gate_passed=status in ("PROMOTED",),
            evidence={"source_failure_refs": payload["source_failure_refs"],
                      "diagnosis": diagnosis or {},
                      "artifact_path": frozen["path"]})
        if rec.get("candidate_sha") != method_sha:
            raise MethodRegistryError("registry_sha_mismatch")
        return {"method_id": method_id, "version": version,
                "method_sha": method_sha, "status": status,
                "artifact_path": frozen["path"], "record": rec}

    def list(self) -> list[dict[str, Any]]:
        return [r for r in self.evolution.all(LAYER)]

    def get(self, method_sha: str) -> dict[str, Any] | None:
        for r in self.list():
            if r.get("candidate_sha") == method_sha:
                return r
        return None

    def verify(self, method_sha: str) -> dict[str, Any]:
        rec = self.get(method_sha)
        if not rec:
            return {"ok": False, "reason": "method_not_found"}
        path = (rec.get("evidence") or {}).get("artifact_path", "")
        if not path or not Path(path).is_file():
            return {"ok": False, "reason": "artifact_missing"}
        import json
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if sha256_json(obj) != method_sha:
            return {"ok": False, "reason": "artifact_sha_drift"}
        return {"ok": True, "method_sha": method_sha,
                "status": rec.get("status")}


__all__ = ["LAYER", "PROMOTION_STATUSES", "MethodRegistry",
           "MethodRegistryError"]
