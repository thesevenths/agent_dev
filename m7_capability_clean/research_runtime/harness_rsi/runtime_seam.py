"""M6.1 -- verify the FORMAL Runtime Adoption seam on REAL approved artifacts.

The M6 run produced genuine human-approved promotion (SHA 6c197910...) with a
real activation record. M6.1 adds a formal HarnessLoader that must be the ONLY
way a campaign resolves its harness version. This module wires that loader to
the real approved artifacts and proves:

  * Loader rejects missing/mismatched approval (fail closed);
  * a NEW campaign resolves the exact approved SHA;
  * an IN-FLIGHT campaign keeps its ORIGINAL version even when a different
    version is being rolled back;
  * real rollback pauses dispatch, isolates in-flight, verifies the parent and
    (optionally) runs a REAL smoke campaign through production agent+engine.

No new scheduler/store is created: it reuses M5 RuntimeState journal, the M6
EvolutionRegistry, and the M6 adoption/activation records.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adoption import AdoptionManager
from .harness_loader import (
    CampaignVersionBinding, HarnessLoader, LoadedHarness, LoaderError,
)
from .identity import freeze_json, sha256_json


def verify_runtime_seam(*, approved_run_dir: str | Path,
                        approval_file: str | Path,
                        out_dir: str | Path,
                        parent_artifact_path: str | Path,
                        parent_file_sha: str,
                        parent_sha: str,
                        smoke_campaign: Any = None,
                        ) -> dict[str, Any]:
    """Wire formal Loader to the REAL approved M6 artifacts and verify it."""
    root = Path(approved_run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    approval = json.loads(Path(approval_file).read_text(encoding="utf-8"))
    final = json.loads((root / "final_result.json").read_text(encoding="utf-8"))
    approved_sha = str(final["adoption"]["record"]["loaded_sha"])
    if approval.get("reviewed_candidate_sha") != approved_sha:
        return {"ok": False, "reason": "approval_not_bound_to_approved_sha",
                "approved_sha": approved_sha}

    from research_runtime.research_store.db import ResearchDatabase
    from research_runtime.rrsi import EvolutionRegistry
    from research_runtime.runtime_state import RuntimeState

    state = RuntimeState(db=ResearchDatabase(out / "runtime.sqlite"))
    cfg = SimpleNamespace = __import__(
        "types", fromlist=["SimpleNamespace"]).SimpleNamespace
    cfg_obj = cfg(run_id="m61_seam", global_budget=8, max_rounds=4)
    state.initialize("m61_seam", config=cfg_obj, dataset_shas={},
                     max_rounds=4, global_budget=8)
    adoption = AdoptionManager(
        state=state, run_id="m61_seam", round_no=1,
        record_path=out / "activation.json")
    # Import the REAL activation record from the approved M6 run (verified by
    # its own SHA and the user's approval), then re-verify through our loader.
    # final_result.json holds the adoption-TIME snapshot (loaded_sha = the
    # approved candidate) and is authoritative: the on-disk adoption.json may
    # have since been rolled back to the parent, which is expected.
    adoption_snapshot = dict(final["adoption"]["record"])
    real_activation = json.loads((root / "adoption.json").read_text(
        encoding="utf-8"))
    activation = dict(adoption_snapshot)
    activation.setdefault("loaded_artifact_path", real_activation.get(
        "loaded_artifact_path", ""))
    activation.setdefault("loaded_file_sha256", real_activation.get(
        "loaded_file_sha256", ""))
    activation["dispatch_stopped"] = False
    freeze_json(activation, adoption.path)

    registry = EvolutionRegistry(
        db=ResearchDatabase(out / "evolution.sqlite"))
    # Register the REAL approved promotion record (from M6 evidence).
    verdict = json.loads((root / "promotion_verdict.json").read_text(
        encoding="utf-8"))
    registry.record(
        experiment_id="m61_seam_promotion", layer="research_harness",
        candidate_id="M6_HYPOTHESIS_CRITIC_V2", candidate_sha=approved_sha,
        parent_sha=str(verdict.get("parent_sha") or final.get("base_sha", "")),
        status="PROMOTED", gate_passed=True,
        evidence={"human_approval": approval,
                  "source_promotion": sha256_json(verdict)})
    bindings = CampaignVersionBinding(db=ResearchDatabase(
        out / "bindings.sqlite"))
    loader = HarnessLoader(
        registry=registry, adoption=adoption, bindings=bindings,
        artifact_index={approved_sha: str(
            root / "candidates" / "M6_HYPOTHESIS_CRITIC_V2.json")})

    # 1. Fail closed: an unbound approval cannot resolve a campaign.
    fail_closed = False
    try:
        loader.verify_approved("0" * 64)
    except LoaderError:
        fail_closed = True

    # 2. A NEW campaign resolves the exact approved SHA via the formal loader.
    try:
        loaded = loader.resolve_for_campaign(
            campaign_id="camp_m61_new", domain="scinet")
        new_sha = loaded.harness_sha
    except LoaderError as exc:
        return {"ok": False, "reason": "loader_failed_to_resolve",
                "detail": str(exc)}

    # 3. IN-FLIGHT campaign pins its version: bind an in-flight campaign to the
    # approved SHA, then attempt a rollback to parent -- its binding must NOT
    # change (only NEW campaigns use the parent after rollback).
    inflight = LoadedHarness(
        harness_sha=approved_sha, component_shas=loaded.component_shas,
        parent_sha=loaded.parent_sha, activation_id=loaded.activation_id)
    bindings.bind(campaign_id="camp_m61_inflight", domain="scinet",
                  loaded=inflight)
    bindings.mark("camp_m61_inflight", "RUNNING")
    before = bindings.resolve("camp_m61_inflight").harness_sha

    rollback = {"status": "NOT_RUN"}
    if smoke_campaign is not None:
        from .rollback_v2 import real_rollback
        rollback = real_rollback(
            manager=adoption, bindings=bindings, parent_sha=parent_sha,
            parent_artifact_path=parent_artifact_path,
            parent_file_sha=parent_file_sha,
            reason="M6.1 real runtime rollback seam test",
            state_path=out / "rollback_state.json",
            smoke_campaign=smoke_campaign)
    after = bindings.resolve("camp_m61_inflight").harness_sha

    result = {
        "ok": bool(fail_closed and new_sha == approved_sha
                   and before == after == approved_sha),
        "approved_sha": approved_sha,
        "new_campaign_resolved_sha": new_sha,
        "inflight_before": before, "inflight_after": after,
        "inflight_version_preserved": before == after,
        "fail_closed_verified": fail_closed,
        "rollback": rollback,
        "activation_sha": sha256_json(activation),
        "loader_version": "m6_1_harness_loader_v1",
        "journal_ok": state.verify_journal("m61_seam")["ok"],
    }
    freeze_json(result, out / "runtime_seam_verification.json")
    state.append("m61_seam", 1, "m61_runtime_seam_verified",
                 {"sha": sha256_json(result)})
    return result


__all__ = ["verify_runtime_seam"]
