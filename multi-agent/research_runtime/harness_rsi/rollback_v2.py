"""M6.1 -- real runtime-safe rollback with in-flight campaign isolation.

Hard requirement honoured here: a harness switch affects ONLY NEW campaigns.
An in-flight campaign keeps its ORIGINAL immutable version binding; rollback
pauses new dispatch, drains/isolates in-flight campaigns, loads and verifies
the parent, runs a REAL smoke campaign through the production agent+engine
chain, and only then resumes dispatch.

Crash points are explicit and resumable via a persisted state machine:
  ROLLBACK_REQUESTED -> DISPATCH_PAUSED -> INFLIGHT_DRAINING ->
  PARENT_LOADING -> PARENT_VERIFICATION -> REAL_SMOKE_CAMPAIGN ->
  PARENT_VERIFIED -> DISPATCH_RESUME -> ROLLBACK_COMPLETED
failure -> ROLLBACK_FAILED (dispatch STAYS paused; human intervention).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .adoption import AdoptionManager
from .harness import load_frozen_bundle
from .harness_loader import (
    CampaignVersionBinding, LoaderError, LoadedHarness,
)
from .identity import freeze_json, sha256_json

ROLLBACK_V2_VERSION = "m6_1_rollback_v2"
TERMINAL_INFLIGHT = {"COMPLETED", "CANCELLED", "FAILED"}


class RollbackV2Error(RuntimeError):
    """Raised on any unsafe rollback transition."""


def _journal(state: dict[str, Any], event: str, payload: dict[str, Any]
             ) -> dict[str, Any]:
    entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "event": event, "payload": payload}
    entry["sha"] = sha256_json({**entry, "sha": ""})
    state.setdefault("journal", []).append(entry)
    return entry


def persist_state(state: dict[str, Any], path: str | Path) -> dict[str, Any]:
    frozen = freeze_json(state, path)
    return {"state_sha": frozen["sha256"], "path": frozen["path"]}


def load_state(path: str | Path) -> dict[str, Any]:
    import json
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def real_rollback(*, manager: AdoptionManager,
                  bindings: CampaignVersionBinding,
                  parent_sha: str, parent_artifact_path: str,
                  parent_file_sha: str, reason: str,
                  state_path: str | Path,
                  smoke_campaign: Callable[[str, str], dict[str, Any]],
                  actor: str = "m61_runtime") -> dict[str, Any]:
    """Perform a real, resumable rollback with in-flight version isolation."""
    state = load_state(state_path) or {
        "version": ROLLBACK_V2_VERSION,
        "status": "NOT_STARTED", "from_sha": manager.active_sha(),
        "to_sha": parent_sha, "reason": reason, "actor": actor,
        "journal": [], "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    current = manager.active_sha()
    if current == parent_sha and state["status"] in (
            "NOT_STARTED", "ROLLBACK_REQUESTED"):
        _journal(state, "NOOP_ALREADY_ON_PARENT", {"sha": parent_sha})
        persist_state(state, state_path)
        return {"ok": True, "status": "NOOP", "loaded_sha": parent_sha}

    # 1. Request + pause NEW dispatch.
    if state["status"] == "NOT_STARTED":
        state["status"] = "ROLLBACK_REQUESTED"
        _journal(state, "ROLLBACK_REQUESTED", {"from": current,
                                               "to": parent_sha})
        manager.set_dispatch(stopped=True, reason=reason)
        state["status"] = "DISPATCH_PAUSED"
        _journal(state, "DISPATCH_PAUSED", {"new_campaigns": "blocked"})
        persist_state(state, state_path)

    # 2. Drain / isolate in-flight campaigns: they KEEP their bound version.
    if state["status"] == "DISPATCH_PAUSED":
        inflight = [b for b in bindings.bound_campaigns()
                    if b["state"] in ("BOUND", "RUNNING")]
        state["inflight_at_request"] = [
            {"campaign_id": b["campaign_id"], "harness_sha": b["harness_sha"],
             "state": b["state"]} for b in inflight]
        _journal(state, "INFLIGHT_DRAINING", {
            "n_inflight": len(inflight),
            "policy": "keep_original_version; never_switch_mid_campaign"})
        # Any campaign bound to the OLD version stays on the old version.
        for b in inflight:
            if b["harness_sha"] != current:
                raise RollbackV2Error(
                    f"campaign {b['campaign_id']} has unexpected version "
                    f"{b['harness_sha']} (expected {current})")
        state["status"] = "INFLIGHT_DRAINED"
        persist_state(state, state_path)

    # 3. Load + verify parent artifact (before any switch).
    if state["status"] == "INFLIGHT_DRAINED":
        state["status"] = "PARENT_LOADING"
        check = load_frozen_bundle(
            parent_artifact_path, expected_candidate_sha=parent_sha,
            expected_file_sha=parent_file_sha)
        if not check["ok"]:
            state["status"] = "ROLLBACK_FAILED"
            _journal(state, "ROLLBACK_FAILED", {"reasons": check["reasons"]})
            persist_state(state, state_path)
            raise RollbackV2Error("parent artifact failed verification: "
                                  + "|".join(check["reasons"]))
        state["status"] = "PARENT_VERIFICATION"
        _journal(state, "PARENT_VERIFIED_ARTIFACT", {
            "parent_sha": parent_sha, "path": str(parent_artifact_path)})
        persist_state(state, state_path)

    # 4. REAL smoke campaign through production agent + engine.
    if state["status"] == "PARENT_VERIFICATION":
        state["status"] = "REAL_SMOKE_CAMPAIGN"
        persist_state(state, state_path)
        try:
            smoke = smoke_campaign(parent_sha, str(parent_artifact_path))
        except Exception as exc:  # noqa: BLE001
            smoke = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        state["smoke_result"] = smoke
        if not smoke.get("ok"):
            state["status"] = "ROLLBACK_FAILED"
            _journal(state, "ROLLBACK_FAILED", {"smoke": smoke})
            persist_state(state, state_path)
            # Dispatch REMAINS paused -- never auto-resume an unverified version.
            return {"ok": False, "status": "ROLLBACK_FAILED",
                    "dispatch_paused": True, "smoke": smoke}
        loaded_sha = str(smoke.get("loaded_sha", ""))
        if loaded_sha != parent_sha:
            state["status"] = "ROLLBACK_FAILED"
            _journal(state, "ROLLBACK_FAILED", {
                "reason": "smoke trajectory used wrong harness",
                "loaded_sha": loaded_sha})
            persist_state(state, state_path)
            return {"ok": False, "status": "ROLLBACK_FAILED",
                    "dispatch_paused": True,
                    "reason": "smoke did not run the parent SHA"}
        state["status"] = "PARENT_VERIFIED"
        _journal(state, "PARENT_VERIFIED_BY_REAL_SMOKE", {
            "campaign_id": smoke.get("campaign_id", ""),
            "trajectory_sha": smoke.get("trajectory_sha", ""),
            "loaded_sha": loaded_sha})
        persist_state(state, state_path)

    # 5. Switch runtime version + resume dispatch.
    if state["status"] == "PARENT_VERIFIED":
        switched = manager.switch_to(
            target_sha=parent_sha, artifact_path=parent_artifact_path,
            file_sha256=parent_file_sha, reason=reason)
        _journal(state, "RUNTIME_VERSION_SWITCHED", {
            "from": switched["from_sha"], "to": switched["to_sha"]})
        if manager.active_sha() != parent_sha:
            state["status"] = "ROLLBACK_FAILED"
            manager.set_dispatch(stopped=True, reason="incorrect_loaded_sha")
            persist_state(state, state_path)
            raise RollbackV2Error("loader activated an incorrect SHA")
        state["status"] = "DISPATCH_RESUME"
        manager.set_dispatch(stopped=False, reason="rollback_verified")
        _journal(state, "DISPATCH_RESUMED", {"new_campaigns": "allowed"})
        state["status"] = "ROLLBACK_COMPLETED"
        persist_state(state, state_path)

    return {"ok": True, "status": state["status"],
            "from_sha": state["from_sha"], "to_sha": parent_sha,
            "loaded_sha": manager.active_sha(),
            "inflight": state.get("inflight_at_request", []),
            "smoke": state.get("smoke_result", {}),
            "state_path": str(state_path),
            "state_sha": persist_state(state, state_path)["state_sha"]}


__all__ = ["ROLLBACK_V2_VERSION", "RollbackV2Error", "real_rollback",
           "persist_state", "load_state"]
