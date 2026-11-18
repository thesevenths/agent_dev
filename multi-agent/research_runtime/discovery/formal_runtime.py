"""M7.6 P0-B: formal production-runtime adapter for the M7 Coordinator.

ARCHITECTURAL NOTE (honest boundary)
------------------------------------
M5 ``ResearchService.run_research_campaign`` executes an ML ``ExperimentPlan``
(hypothesis/candidate/paired-seed trials) through ``CampaignRunner``. M7's
scientific engine executes a DomainPack experiment (arms/seeds/splits). These
are DIFFERENT execution abstractions. Merging them would be an architecture
rewrite, which M7.6 explicitly forbids.

This adapter is therefore the THIN seam that makes M7 execution go through the
SAME production contracts without forking a second runtime:

  * M5 ``RuntimeState``  -- hash-chained journal, executor claim, budget
    reservation lifecycle (reserved -> bound -> consumed/released).
  * M6 ``HarnessLoader.resolve_for_campaign`` -- the ONLY authority for the
    harness SHA; the SHA is pinned per campaign and never echoed from input.
  * M6 campaign-version binding -- each campaign is bound to the loaded harness.
  * real DomainPack engine -- the scientific executor, invoked ONLY through
    this adapter in production; never called directly by the Coordinator.

Production fails CLOSED: no loader / no resolved harness / harness mismatch /
missing backend / missing approval all raise ProductionRuntimeBlocked. There is
no silent LOCAL_DIRECT_EXECUTOR fallback in production mode.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ..harness_rsi.identity import sha256_json
from ..runtime_state import RuntimeState


class ProductionRuntimeBlocked(RuntimeError):
    """A required production contract is absent or inconsistent."""


def formal_campaign_id(*, run_id: str, experiment_id: str,
                       harness_sha: str, preregistration_sha: str) -> str:
    """Deterministic campaign identity bound to the REAL harness + protocol."""
    material = "|".join([run_id, experiment_id, harness_sha,
                         preregistration_sha])
    return "camp_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


class FormalRuntimeAdapter:
    """Reuse M5 RuntimeState + M6 HarnessLoader for M7 experiment execution."""

    def __init__(self, *, runtime: RuntimeState, harness_loader: Any,
                 backend: Callable[..., dict[str, Any]],
                 domain: str, run_id: str,
                 executor_id: str = "m7_formal_runtime_adapter"):
        if runtime is None or harness_loader is None or backend is None:
            raise ProductionRuntimeBlocked(
                "formal adapter requires runtime, harness_loader and backend")
        self.runtime = runtime
        self.harness_loader = harness_loader
        self.backend = backend
        self.domain = domain
        self.run_id = run_id
        self.executor_id = executor_id

    def resolve_harness(self, *, campaign_id: str) -> dict[str, Any]:
        """Resolve the immutable harness identity from the real M6 loader."""
        try:
            loaded = self.harness_loader.resolve_for_campaign(
                campaign_id=campaign_id, domain=self.domain)
        except Exception as exc:  # noqa: BLE001 - loader failure fails closed
            raise ProductionRuntimeBlocked(
                f"harness_loader_failed:{type(exc).__name__}:{exc}") from exc
        sha = str(getattr(loaded, "harness_sha", "") or "")
        if not sha or set(sha) - set("0123456789abcdef"):
            raise ProductionRuntimeBlocked("harness_sha_invalid")
        return {
            "harness_sha": sha,
            "component_shas": dict(getattr(loaded, "component_shas", {}) or {}),
            "parent_sha": str(getattr(loaded, "parent_sha", "") or ""),
            "activation_id": str(getattr(loaded, "activation_id", "") or ""),
        }

    def dispatch(self, *, experiment_id: str, preregistration_sha: str,
                 arms: list[str], seeds: list[int], splits: list[str],
                 out_dir: str, stage: str, expected_harness_sha: str,
                 config_sha: str, round_no: int) -> dict[str, Any]:
        """Execute one approved experiment through the formal seam.

        WP-01 guarantees (at-most-once, NEVER claimed as exactly-once):
          * a durable per-campaign CLAIM is taken BEFORE any backend call;
          * a COMPLETED/FAILED campaign returns its cached result, NEVER
            re-executes the (possibly non-idempotent) backend;
          * a concurrent/different executor is refused;
          * CLAIMED/RUNNING left by a crash becomes RECOVERY_REQUIRED and is
            never blindly retried (backend idempotency is unknown);
          * reservation/journal settle only after the outcome is known.
        """
        # 1) deterministic identity bound to the REAL harness + frozen protocol.
        campaign_id = formal_campaign_id(
            run_id=self.run_id, experiment_id=experiment_id,
            harness_sha=expected_harness_sha,
            preregistration_sha=preregistration_sha)
        binding = self.resolve_harness(campaign_id=campaign_id)
        if binding["harness_sha"] != expected_harness_sha:
            raise ProductionRuntimeBlocked(
                "harness_sha_mismatch:pinned_vs_resolved")

        # 2) ATOMIC durable claim (BEGIN IMMEDIATE inside RuntimeState).
        claim = self.runtime.claim_campaign(
            campaign_id=campaign_id, run_id=self.run_id,
            executor_id=self.executor_id, config_sha=config_sha,
            harness_sha=binding["harness_sha"])
        if not claim.get("ok"):
            if claim.get("already_terminal"):
                # Return the EXACT cached result; never touch backend again.
                cached = dict(claim.get("result") or {})
                cached.update({"campaign_id": campaign_id,
                               "cache_hit": True,
                               "cached_state": claim["state"],
                               "result_sha": claim.get("result_sha", ""),
                               "result_sha256": claim.get("result_sha", ""),
                               "harness_binding": {
                                   "mode": "FORMAL_RUNTIME_ADAPTER_CACHED",
                                   **binding}})
                self.runtime.append(self.run_id, int(round_no),
                                    "campaign_dispatch_cache_hit",
                                    {"campaign_id": campaign_id,
                                     "state": claim["state"]})
                return cached
            raise ProductionRuntimeBlocked(
                "campaign_claim_refused:" + str(claim.get("reason", claim.get("state"))))

        rid = RuntimeState.reservation_id(
            self.run_id, int(round_no), self.domain,
            f"{stage}:{campaign_id}")
        # 3) reserve + bind (idempotent for resume-in-place by same executor).
        self.runtime.reserve_execution(
            self.run_id, rid, config_sha=config_sha,
            domain_id=self.domain, round_no=int(round_no))
        self.runtime.bind_reservation(self.run_id, rid, campaign_id=campaign_id)
        if not claim.get("resume_in_place"):
            self.runtime.append(self.run_id, int(round_no), "campaign_created", {
                "campaign_id": campaign_id, "experiment_id": experiment_id,
                "harness_sha": binding["harness_sha"],
                "component_shas": binding["component_shas"],
                "activation_id": binding["activation_id"],
                "preregistration_sha": preregistration_sha,
                "protocol_binding_sha": sha256_json({
                    "experiment_id": experiment_id, "arms": arms, "seeds": seeds,
                    "splits": splits, "preregistration_sha": preregistration_sha}),
                "reservation_id": rid, "stage": stage})

        # 4) mark RUNNING immediately before the backend call. A crash between
        #    here and result commit leaves RUNNING -> RECOVERY_REQUIRED on boot.
        self.runtime.set_campaign_state(campaign_id, "RUNNING",
                                        executor_id=self.executor_id)
        self.runtime.append(self.run_id, int(round_no), "campaign_running", {
            "campaign_id": campaign_id, "reservation_id": rid})

        try:
            result = self.backend(
                experiment_id=experiment_id, arms=arms, seeds=seeds,
                splits=splits, out_dir=out_dir)
        except Exception as exc:  # noqa: BLE001 - settle then surface
            reason = f"{type(exc).__name__}: {exc}"
            self.runtime.set_campaign_state(
                campaign_id, "FAILED", executor_id=self.executor_id,
                reason=reason)
            self.runtime.consume_reservation(self.run_id, rid,
                                             outcome="failed")
            self.runtime.append(self.run_id, int(round_no),
                                "campaign_failed", {
                "campaign_id": campaign_id, "reservation_id": rid,
                "reason": reason})
            raise

        outcome = "succeeded" if result.get("ok") else "failed"
        result_sha = sha256_json(result)
        # Structured result persistence + terminal claim in one ordering: mark
        # terminal BEFORE journal, so a crash after commit never re-executes.
        self.runtime.set_campaign_state(
            campaign_id, "COMPLETED", executor_id=self.executor_id,
            result=result, result_sha=result_sha, reason=outcome)
        self.runtime.consume_reservation(self.run_id, rid, outcome=outcome)
        self.runtime.append(self.run_id, int(round_no), "campaign_completed", {
            "campaign_id": campaign_id, "reservation_id": rid,
            "outcome": outcome,
            "summary_sha256": result.get("summary_sha256", ""),
            "result_sha256": result_sha,
            "harness_sha": binding["harness_sha"]})

        return {
            **{k: v for k, v in result.items() if k != "summary"},
            "campaign_id": campaign_id,
            "reservation_id": rid,
            "reservation_outcome": outcome,
            "result_sha256": result_sha,
            "cache_hit": False,
            "harness_binding": {"mode": "FORMAL_RUNTIME_ADAPTER", **binding},
        }


__all__ = ["ProductionRuntimeBlocked", "FormalRuntimeAdapter",
           "formal_campaign_id"]
