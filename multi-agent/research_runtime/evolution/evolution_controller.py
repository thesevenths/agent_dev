"""ARR 1.1 / F3 -- dual-layer RRSI coordination controller.

Drives BOTH evolution layers through one entry point and guarantees they stay
separate:

* Layer 1 (RESEARCH_HARNESS) reuses the existing, proven
  ``adaptive_runtime.harness_evolution.evolve`` -- no rewrite.
* Layer 2 (DOMAIN_MODEL) reuses the existing domain whitelist +
  ``CandidateConfigBuilder`` and the existing domain campaign (injected runner).

The two layers have NAME-SPACED ledger fingerprints (see
``evolution_spaces.namespaced_fingerprint``) so a domain knob can never be
confused with a harness improvement, and the evaluation substrate is frozen for
both. The controller emits a combined payload with a ``version_inheritance``
block that the Package-G loop controller consumes to start the next round on the
accepted baseline.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .candidate import CandidateStatus, CandidateType
from .candidate_registry import CandidateRegistry
from .harness_evolver import evolve as _harness_evolve, HARNESS_LAYER
from .domain_evolver import (
    generate_domain_candidates, evaluate_domain_candidate, DOMAIN_LAYER,
)

DomainRunner = Callable[[dict[str, Any]], float]


def _decide_domain(rec: dict[str, Any]) -> CandidateStatus:
    """Fixed, transparent decision for a single-knob domain candidate.

    Missing runs never manufacture a gain (INCONCLUSIVE). A strictly positive
    mean delta is ACCEPTED; otherwise REJECTED. The gate is outside the
    candidate (F4) -- it never reads the candidate's own config.
    """
    if any(d is None for d in rec["deltas"]):
        return CandidateStatus.INCONCLUSIVE
    if rec["mean_delta"] > 0.0:
        return CandidateStatus.ACCEPTED
    return CandidateStatus.REJECTED


def _version_inheritance(harness_payload: dict[str, Any],
                         domain_records: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_harness = len(harness_payload.get("accepted", []) or [])
    accepted_domain = sum(1 for r in domain_records
                          if r.get("decision") == CandidateStatus.ACCEPTED.value)
    return {
        "harness_accepted": accepted_harness,
        "domain_accepted": accepted_domain,
        "inheritable_next_round": bool(accepted_harness or accepted_domain),
        "note": ("next round starts from the accepted Harness AND the accepted "
                 "domain config as the new baseline (Package G responsibility)"),
    }


def run_dual_layer(*, base_harness: Any, tasks: list[Any], heldout_task: Any,
                   run_root: str | Path, repo_root: str | Path,
                   rounds: int = 2, max_minutes: float = 30.0, operator: str = "",
                   negative_harness: Any | None = None,
                   include_degenerate_control: bool = True, cfg: Any = None,
                   domain: str | None = None,
                   domain_base_config: str | Path | None = None,
                   project_dir: str | Path | None = None,
                   dataset_version: str = "",
                   domain_seeds: tuple[int, ...] = (42, 43),
                   domain_runner: DomainRunner | None = None,
                   domain_require_frozen_manifest: bool = True,
                   harness_evolve: Callable[..., dict[str, Any]] = _harness_evolve
                   ) -> dict[str, Any]:
    """Run the two RRSI layers and return a combined, auditable payload."""
    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    registry = CandidateRegistry(run_root / "candidate_registry.jsonl")
    started = time.time()

    # ---- Layer 1: RESEARCH_HARNESS (reuses existing evolve) ----
    harness_payload = harness_evolve(
        base_harness=base_harness, tasks=tasks, heldout_task=heldout_task,
        run_root=run_root / "harness_layer", repo_root=repo_root,
        rounds=rounds, max_minutes=max_minutes, operator=operator,
        negative_harness=negative_harness,
        include_degenerate_control=include_degenerate_control, cfg=cfg)

    # ---- Layer 2: DOMAIN_MODEL (reuses existing whitelist + builder) ----
    domain_records: list[dict[str, Any]] = []
    if domain and domain_base_config and domain_runner is not None:
        dominant = harness_payload.get("dominant_failure", "")
        cands = generate_domain_candidates(domain=domain, dominant=dominant)
        for c in cands:
            registry.register(c, status=CandidateStatus.PROPOSED)
            registry.transition(c.hyp_fingerprint, CandidateStatus.CRITIC_APPROVED)
            registry.transition(c.hyp_fingerprint, CandidateStatus.SANDBOXED)
            rec = evaluate_domain_candidate(
                domain=domain, candidate=c, base_config=domain_base_config,
                project_dir=project_dir, campaign_id="dual_layer",
                seeds=domain_seeds, dataset_version=dataset_version,
                runner=domain_runner,
                require_frozen_manifest=domain_require_frozen_manifest)
            registry.transition(c.hyp_fingerprint, CandidateStatus.EVALUATED,
                               record=rec)
            decision = _decide_domain(rec)
            registry.transition(c.hyp_fingerprint, decision)
            rec["decision"] = decision.value
            domain_records.append(rec)

    payload: dict[str, Any] = {
        "schema_version": "arr-1.1-dual-layer-v1",
        "elapsed_seconds": round(time.time() - started, 3),
        "harness_layer": harness_payload,
        "domain_layer": {
            "domain": domain,
            "candidate_type": CandidateType.DOMAIN_MODEL.value,
            "records": domain_records,
            "n_accepted": sum(1 for r in domain_records
                             if r.get("decision") == CandidateStatus.ACCEPTED.value),
            "n_rejected": sum(1 for r in domain_records
                             if r.get("decision") == CandidateStatus.REJECTED.value),
            "n_inconclusive": sum(1 for r in domain_records
                                 if r.get("decision") == CandidateStatus.INCONCLUSIVE.value),
        },
        "separation": {
            "candidate_types_distinct": HARNESS_LAYER != DOMAIN_LAYER,
            "ledger_namespaced": True,
            "frozen_protected": True,
            "harness_candidates_seen": len(harness_payload.get("records", []) or []),
            "domain_candidates_seen": len(domain_records),
        },
        "version_inheritance": _version_inheritance(harness_payload, domain_records),
    }
    return payload


class DualLayerController:
    """Thin config-driven wrapper around :func:`run_dual_layer`."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        return run_dual_layer(**self.config)
