"""M6 -- real runtime Harness rollback.

Rollback is an actual version switch, not "return a target SHA":

  1. stop new dispatch and journal it;
  2. refuse to switch while a campaign is bound to a different version (no
     mixed Harness versions inside a campaign);
  3. verify the PARENT artifact on disk against its approved SHA;
  4. activate the parent version and journal the transition;
  5. run a caller-supplied SMOKE VALIDATION against the loaded version;
  6. resume dispatch only after verification succeeds; on smoke failure the
     system stays stopped and reports the rollback as FAILED.
"""
from __future__ import annotations

from typing import Any, Callable

from .adoption import AdoptionError, AdoptionManager
from .harness import load_frozen_bundle
from .identity import sha256_json


class RollbackError(RuntimeError):
    """Raised when a rollback cannot be completed safely."""


def rollback(*, manager: AdoptionManager, target_sha: str,
             target_artifact_path: str, target_file_sha256: str,
             reason: str, smoke: Callable[[str], dict[str, Any]],
             in_flight_campaigns: list[dict[str, Any]] | None = None
             ) -> dict[str, Any]:
    """Perform a real, verifiable rollback to the previous approved version."""
    current = manager.active_sha()
    if current == target_sha:
        return {"ok": False, "status": "NOOP", "reason": "already_on_target"}

    # 3 (before 4): verify the parent/target artifact BEFORE any switch.
    target_check = load_frozen_bundle(
        target_artifact_path, expected_candidate_sha=target_sha,
        expected_file_sha=target_file_sha256)
    if not target_check["ok"]:
        raise RollbackError("rollback target verification failed: "
                            + "|".join(target_check["reasons"]))

    # 1: stop dispatch first.
    manager.set_dispatch(stopped=True, reason=reason)

    # 2: prevent mixed versions inside any in-flight campaign.
    for camp in in_flight_campaigns or []:
        bound = str(camp.get("harness_sha", ""))
        if bound and bound != current:
            manager.set_dispatch(stopped=True,
                                 reason="rollback_blocked_mixed_version")
            raise RollbackError(
                f"campaign {camp.get('campaign_id')} is bound to {bound}, "
                f"not current {current}; refusing mixed-version rollback")

    # 4: switch.
    switched = manager.switch_to(target_sha=target_sha,
                                 artifact_path=target_artifact_path,
                                 file_sha256=target_file_sha256,
                                 reason=reason)

    # 5: smoke validation on the actually-loaded version.
    loaded = manager.active_sha()
    if loaded != target_sha:
        manager.set_dispatch(stopped=True, reason="incorrect_loaded_sha")
        raise RollbackError(f"loader activated {loaded}, expected {target_sha}")
    try:
        smoke_result = smoke(loaded)
    except Exception as exc:  # noqa: BLE001 - fail closed
        smoke_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not smoke_result.get("ok"):
        manager.set_dispatch(stopped=True, reason="rollback_smoke_failed")
        return {"ok": False, "status": "FAILED",
                "reason": "smoke validation failed after switch",
                "smoke": smoke_result, "loaded_sha": loaded}

    # 6: resume only after verification.
    verified = manager.verify_loaded()
    if not verified["ok"]:
        manager.set_dispatch(stopped=True, reason="rollback_verify_failed")
        return {"ok": False, "status": "FAILED",
                "reason": "loaded version verification failed",
                "details": verified}
    manager.set_dispatch(stopped=False, reason="rollback_verified")

    return {"ok": True, "status": "ROLLED_BACK",
            "from_sha": current, "to_sha": target_sha,
            "smoke": smoke_result, "verified": verified,
            "switch": {"from_sha": switched["from_sha"],
                       "to_sha": switched["to_sha"]}}


__all__ = ["RollbackError", "rollback"]
