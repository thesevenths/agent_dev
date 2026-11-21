"""M6 -- real Harness Loader and Adoption Manager.

Registry promotion is NOT adoption. Adoption means: the exact approved SHA is
written to an activation record, re-read from disk, verified byte-for-byte,
logged into the SAME RuntimeState journal, and every subsequent campaign is
bound to that immutable version. A stale loader or an incorrect loaded SHA is
a hard failure, not a warning.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .harness import load_frozen_bundle
from .identity import freeze_json, sha256_json, verify_frozen

ADOPTION_VERSION = "m6_adoption_v1"


class AdoptionError(RuntimeError):
    """Raised when adoption cannot be proven safe."""


class AdoptionManager:
    """Owns the single activation record for one run.

    ``state`` is the EXISTING M5 RuntimeState; we only append journal events to
    it, never reimplement or reset it.
    """

    def __init__(self, *, state: Any, run_id: str, round_no: int,
                 record_path: str | Path):
        self.state = state
        self.run_id = run_id
        self.round_no = round_no
        self.path = Path(record_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, record: dict[str, Any]) -> dict[str, Any]:
        frozen = freeze_json(record, self.path)
        return {"file_sha256": frozen["sha256"], "path": frozen["path"]}

    def _journal(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.state.append(self.run_id, self.round_no, event, payload)

    def active_sha(self) -> str:
        return str(self._read().get("loaded_sha", ""))

    def is_dispatch_stopped(self) -> bool:
        return bool(self._read().get("dispatch_stopped", False))

    def adopt(self, *, approved_record: dict[str, Any],
              candidate_sha: str, artifact_path: str,
              file_sha256: str, human_approval: dict[str, Any],
              parent_sha: str = "") -> dict[str, Any]:
        """Activate an approved candidate after verifying the frozen artifact."""
        if approved_record.get("status") != "PROMOTED":
            raise AdoptionError("registry record is not PROMOTED")
        if not human_approval.get("approved"):
            raise AdoptionError("adoption requires explicit human approval")
        if approved_record.get("candidate_sha") != candidate_sha:
            raise AdoptionError("approved SHA does not match candidate")
        check = load_frozen_bundle(
            artifact_path, expected_candidate_sha=candidate_sha,
            expected_file_sha=file_sha256)
        if not check["ok"]:
            raise AdoptionError("frozen candidate artifact failed re-read: "
                                + "|".join(check["reasons"]))
        self._journal("harness_adoption_requested", {
            "candidate_sha": candidate_sha, "parent_sha": parent_sha,
            "approved_experiment": approved_record.get("experiment_id")})
        record = {
            "adoption_version": ADOPTION_VERSION,
            "run_id": self.run_id,
            "approved_sha": candidate_sha,
            "loaded_sha": candidate_sha,
            "loaded_artifact_path": str(artifact_path),
            "loaded_file_sha256": file_sha256,
            "parent_sha": parent_sha,
            "human_approval": dict(human_approval),
            "registry_experiment": approved_record.get("experiment_id", ""),
            "dispatch_stopped": False,
            "adopted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        written = self._write(record)
        self._journal("harness_loaded", {
            "loaded_sha": candidate_sha, "expected_sha": candidate_sha,
            "activation_file_sha256": written["file_sha256"]})
        verified = self.verify_loaded()
        if not verified["ok"]:
            raise AdoptionError("post-load verification failed: "
                                + "|".join(verified["reasons"]))
        self._journal("harness_version_verified", {
            "loaded_sha": candidate_sha, "verified": True})
        return {"ok": True, "record": record,
                "activation_file_sha256": written["file_sha256"]}

    def switch_to(self, *, target_sha: str, artifact_path: str,
                  file_sha256: str, reason: str) -> dict[str, Any]:
        """Activate a different, verified approved version (used by rollback)."""
        check = load_frozen_bundle(
            artifact_path, expected_candidate_sha=target_sha,
            expected_file_sha=file_sha256)
        if not check["ok"]:
            raise AdoptionError("rollback target artifact failed verification: "
                                + "|".join(check["reasons"]))
        rec = self._read()
        if not rec:
            raise AdoptionError("no activation record to switch")
        previous = rec["loaded_sha"]
        rec.update({"approved_sha": target_sha, "loaded_sha": target_sha,
                    "loaded_artifact_path": str(artifact_path),
                    "loaded_file_sha256": file_sha256,
                    "bound_sha": "", "dispatch_stopped": False,
                    "switch_reason": reason,
                    "previous_sha": previous,
                    "switched_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        written = self._write(rec)
        self._journal("harness_version_switched", {
            "from_sha": previous, "to_sha": target_sha,
            "reason": reason, "activation_file_sha256": written["file_sha256"]})
        return {"ok": True, "from_sha": previous, "to_sha": target_sha,
                "record": rec}

    def set_dispatch(self, *, stopped: bool, reason: str) -> dict[str, Any]:
        """Stop/resume new dispatch as part of a real rollback transition."""
        rec = self._read()
        if not rec:
            raise AdoptionError("no activation record")
        rec["dispatch_stopped"] = bool(stopped)
        rec["dispatch_reason"] = reason
        written = self._write(rec)
        self._journal("harness_dispatch_stopped" if stopped
                      else "harness_dispatch_resumed",
                      {"loaded_sha": rec["loaded_sha"], "reason": reason,
                       "activation_file_sha256": written["file_sha256"]})
        return {"ok": True, "stopped": bool(stopped)}

    def verify_loaded(self) -> dict[str, Any]:
        """Re-read the activation record AND the artifact it points to."""
        rec = self._read()
        if not rec:
            return {"ok": False, "reasons": ["no_activation_record"]}
        if rec.get("loaded_sha") != rec.get("approved_sha"):
            return {"ok": False, "reasons": ["loaded_sha_differs_from_approved"]}
        af = rec.get("loaded_artifact_path", "")
        check = load_frozen_bundle(
            af, expected_candidate_sha=rec["loaded_sha"],
            expected_file_sha=rec.get("loaded_file_sha256", ""))
        if not check["ok"]:
            return {"ok": False, "reasons": check["reasons"]}
        return {"ok": True, "reasons": [], "loaded_sha": rec["loaded_sha"]}

    def bind_campaign(self, campaign_id: str, *, expected_sha: str = ""
                      ) -> dict[str, Any]:
        """Bind one campaign to exactly ONE immutable harness version."""
        rec = self._read()
        if not rec:
            raise AdoptionError("no harness adopted")
        if self.is_dispatch_stopped():
            raise AdoptionError("dispatch stopped; cannot bind campaign")
        bound = str(rec.get("bound_sha", ""))
        if bound and bound != rec["loaded_sha"]:
            raise AdoptionError("mixed harness versions inside a campaign")
        if expected_sha and expected_sha != rec["loaded_sha"]:
            raise AdoptionError("campaign expected a different harness SHA")
        rec.setdefault("bound_campaigns", []).append({
            "campaign_id": campaign_id, "harness_sha": rec["loaded_sha"],
            "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        rec["bound_sha"] = rec["loaded_sha"]
        self._write(rec)
        self._journal("harness_campaign_bound", {
            "campaign_id": campaign_id, "harness_sha": rec["loaded_sha"]})
        return {"ok": True, "harness_sha": rec["loaded_sha"]}

    def post_adoption_performance(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Journal actual post-adoption performance on new research tasks."""
        rec = self._read()
        if not rec:
            raise AdoptionError("no harness adopted")
        payload = {"harness_sha": rec["loaded_sha"], "metrics": metrics,
                   "metrics_sha": sha256_json(metrics)}
        self._journal("harness_post_adoption_performance", payload)
        return {"ok": True, **payload}


__all__ = ["AdoptionError", "ADOPTION_VERSION", "AdoptionManager"]
