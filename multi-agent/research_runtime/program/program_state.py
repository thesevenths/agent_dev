"""ARR v1.0 WP-02: persistent multi-round research PROGRAM state.

A ResearchProgram owns multiple ResearchRounds around ONE bounded objective.
This module is a THIN state layer: it does NOT create a Runtime (M5 RuntimeState
still owns journal/reservation/budget claims) and does NOT change the single-round
DiscoveryCoordinator. It persists program/round records with canonical SHA and
validates state transitions explicitly.

Scientific INCONCLUSIVE is NOT an engineering FAILED: a round may end
INCONCLUSIVE and the program still continues to the next planned round.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..harness_rsi.identity import freeze_json, sha256_json

PROGRAM_STATES = (
    "PROGRAM_CREATED", "OBJECTIVE_VALIDATED", "RESEARCH_QUEUED",
    "ROUND_RUNNING", "ROUND_COMPLETED", "MEMORY_COMMIT",
    "NEXT_ROUND_PLANNING", "METHOD_DIAGNOSIS", "CANDIDATE_EVALUATION",
    "PROGRAM_COMPLETED", "BLOCKED", "FAILED", "INCONCLUSIVE", "PAUSED",
    "RECOVERY_REQUIRED",
)
TERMINAL_STATES = frozenset(
    {"PROGRAM_COMPLETED", "FAILED", "BLOCKED", "INCONCLUSIVE"})

ROUND_STATES = ("PLANNED", "RUNNING", "COMPLETED", "INCONCLUSIVE", "FAILED",
                "BLOCKED")

# allowed program-level transitions (explicit, validated)
_TRANSITIONS: dict[str, set[str]] = {
    "PROGRAM_CREATED": {"OBJECTIVE_VALIDATED", "BLOCKED", "FAILED"},
    "OBJECTIVE_VALIDATED": {"RESEARCH_QUEUED", "BLOCKED", "INCONCLUSIVE"},
    "RESEARCH_QUEUED": {"ROUND_RUNNING", "PROGRAM_COMPLETED", "BLOCKED",
                        "INCONCLUSIVE"},
    "ROUND_RUNNING": {"ROUND_COMPLETED", "FAILED", "BLOCKED",
                      "RECOVERY_REQUIRED"},
    "ROUND_COMPLETED": {"MEMORY_COMMIT", "METHOD_DIAGNOSIS", "FAILED",
                        "RECOVERY_REQUIRED"},
    "MEMORY_COMMIT": {"NEXT_ROUND_PLANNING", "CANDIDATE_EVALUATION",
                      "PROGRAM_COMPLETED", "RECOVERY_REQUIRED"},
    "NEXT_ROUND_PLANNING": {"RESEARCH_QUEUED", "PROGRAM_COMPLETED",
                            "INCONCLUSIVE", "RECOVERY_REQUIRED"},
    "METHOD_DIAGNOSIS": {"CANDIDATE_EVALUATION", "NEXT_ROUND_PLANNING",
                         "RECOVERY_REQUIRED"},
    "CANDIDATE_EVALUATION": {"NEXT_ROUND_PLANNING", "PROGRAM_COMPLETED",
                             "RECOVERY_REQUIRED"},
    "PAUSED": {"RESEARCH_QUEUED", "ROUND_RUNNING", "RECOVERY_REQUIRED"},
    "RECOVERY_REQUIRED": {"ROUND_RUNNING", "NEXT_ROUND_PLANNING", "FAILED",
                          "INCONCLUSIVE"},
    "PROGRAM_COMPLETED": set(), "FAILED": set(), "BLOCKED": set(),
    "INCONCLUSIVE": set(),
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class ProgramStateError(RuntimeError):
    pass


class ResearchProgramState:
    """Durable program record + round ledger, with explicit transitions."""

    def __init__(self, *, root: str | Path, program_id: str,
                 objective: dict[str, Any], domain_scope: str,
                 resource_policy: dict[str, Any],
                 approval_policy: dict[str, Any]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.program_id = program_id
        self.path = self.root / "program_state.json"
        if self.path.is_file():
            import json
            self.obj = json.loads(self.path.read_text(encoding="utf-8"))
            if self.obj.get("program_id") != program_id:
                raise ProgramStateError("program_id mismatch on resume")
        else:
            self.obj = {
                "version": "arr_v1_program_v1", "program_id": program_id,
                "objective": dict(objective), "domain_scope": domain_scope,
                "resource_policy": dict(resource_policy),
                "approval_policy": dict(approval_policy),
                "status": "PROGRAM_CREATED", "rounds": [], "events": [],
                "created_at": _now(), "updated_at": _now(),
            }
            self._save()

    # -------------------------------------------------------------- helpers
    def _save(self) -> str:
        import json
        self.obj["updated_at"] = _now()
        # Compute the manifest digest over the content WITHOUT the digest itself
        # (self-referential field), then persist the COMPLETE object including
        # that digest so a restarted process can re-verify it (WP-10).
        content = {k: v for k, v in self.obj.items()
                   if k != "program_manifest_sha"}
        digest = sha256_json(content)
        self.obj["program_manifest_sha"] = digest
        freeze_json(self.obj, self.path)
        return digest

    @property
    def status(self) -> str:
        return str(self.obj["status"])

    def transition(self, new_state: str, *, reason: str = "",
                   evidence: dict[str, Any] | None = None) -> None:
        if new_state not in PROGRAM_STATES:
            raise ProgramStateError(f"unknown program state {new_state!r}")
        current = self.status
        if new_state not in _TRANSITIONS.get(current, set()):
            raise ProgramStateError(f"illegal transition {current}->{new_state}")
        self.obj["status"] = new_state
        self.obj["events"].append({
            "from": current, "to": new_state, "reason": reason,
            "evidence_sha": sha256_json(evidence or {}), "at": _now()})
        self._save()

    def add_round(self, round_record: dict[str, Any]) -> None:
        rid = str(round_record.get("round_id", ""))
        if not rid:
            raise ProgramStateError("round_id required")
        if any(r["round_id"] == rid for r in self.obj["rounds"]):
            raise ProgramStateError(f"duplicate round {rid}")
        self.obj["rounds"].append(dict(round_record))
        self._save()

    def update_round(self, round_id: str, **fields: Any) -> None:
        for rec in self.obj["rounds"]:
            if rec["round_id"] == round_id:
                rec.update(fields)
                self._save()
                return
        raise ProgramStateError(f"unknown round {round_id}")

    def round(self, round_id: str) -> dict[str, Any] | None:
        return next((r for r in self.obj["rounds"]
                     if r["round_id"] == round_id), None)

    def last_round(self) -> dict[str, Any] | None:
        return self.obj["rounds"][-1] if self.obj["rounds"] else None

    def budget_used(self) -> dict[str, int]:
        totals = {"llm_calls": 0, "tokens": 0, "experiments": 0, "rounds": 0}
        for r in self.obj["rounds"]:
            b = r.get("budget_used") or {}
            for k in totals:
                totals[k] += int(b.get(k, 0) or 0)
        totals["rounds"] = len(self.obj["rounds"])
        return totals

    def verify_integrity(self) -> dict[str, Any]:
        import json
        persisted = {k: v for k, v in self.obj.items()
                     if k != "program_manifest_sha"}
        current = sha256_json(persisted)
        recorded = self.obj.get("program_manifest_sha", "")
        return {"ok": current == recorded, "current": current,
                "recorded": recorded}


def new_round_record(*, program_id: str, round_id: str,
                     family_id: str, parent_round_id: str = "",
                     objective: str = "") -> dict[str, Any]:
    """A ResearchRound record with the fields required by WP-02/§12.2."""
    return {
        "round_id": round_id, "program_id": program_id,
        "family_id": family_id, "parent_round_id": parent_round_id,
        "objective": objective,
        "state": "PLANNED",
        "research_question_ref": "", "hypothesis_refs": [],
        "literature_snapshot_ref": "", "preregistration_sha": "",
        "approval_ref": "", "input_memory_refs": [],
        "method_sha": "", "harness_sha": "", "dataset_sha": "",
        "campaign_refs": [], "evidence_refs": [], "claim_refs": [],
        "scientific_verdict": "NOT_ASSESSED", "execution_status": "NOT_RUN",
        "budget_used": {}, "trace_ref": "", "round_manifest_sha": "",
        "created_at": _now(),
    }


__all__ = ["PROGRAM_STATES", "TERMINAL_STATES", "ROUND_STATES",
           "ProgramStateError", "ResearchProgramState", "new_round_record"]
