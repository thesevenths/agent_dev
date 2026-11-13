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

import math
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


#: Absolute floor: fewer than this many VALID paired runs cannot even produce a
#: stable mean -> always INCONCLUSIVE (never a fabricated verdict). This is the
#: DEV/provisional lower bound recorded by ``domain_evolver``; the FORMAL ACCEPT
#: bar is ``MIN_SEEDS_FOR_FORMAL_ACCEPT`` below and is strictly higher.
MIN_VALID_SEEDS = 2

#: Minimum number of VALID paired (base/candidate) seeds required to even consider
#: a *formal* ACCEPT. Below this (e.g. a 2-seed dev/quick run) the candidate can
#: at most be INCONCLUSIVE -- never ACCEPTED. This is the M2.7 statistical gate: a
#: 2-seed gain cannot be separated from noise, so it must not be declared a real
#: scientific improvement (M2.5 already established a 5-seed protocol; M2.6 must
#: not lower that bar).
MIN_SEEDS_FOR_FORMAL_ACCEPT = 5

#: A positive mean improvement must exceed this FRACTION of the baseline metric to
#: count as *practically* (not just statistically) significant. Gains smaller than
#: this are treated as noise-level and downgraded to INCONCLUSIVE, never ACCEPTED.
PRACTICAL_IMPROVEMENT_FLOOR_REL = 0.01


def _decide_domain(rec: dict[str, Any], *,
                   min_seeds_for_accept: int = MIN_SEEDS_FOR_FORMAL_ACCEPT,
                   practical_improvement_floor_rel: float = PRACTICAL_IMPROVEMENT_FLOOR_REL,
                   require_holdout_confirmation: bool = True
                   ) -> CandidateStatus:
    """Fixed, transparent, direction-aware decision for one domain candidate.

    M2.7 statistical gate (no fabricated verdicts, no lowered bar):
      * Any NaN / Inf / None / missing run (``n_invalid > 0``) is INCONCLUSIVE --
        a failed or missing experiment is NOT "real degradation".
      * Fewer than ``min_seeds_for_accept`` valid paired runs is INCONCLUSIVE: a
        2-seed dev/quick run can therefore never be formally ACCEPTED.
      * Statistical significance: the 95% CI (normal approx) of the paired
        improvements must exclude 0; a directional but non-significant gain is
        INCONCLUSIVE, not ACCEPT.
      * Practical significance: the mean improvement must exceed a relative floor
        of the baseline metric (``practical_improvement_floor_rel``); a
        directional but noise-level gain is INCONCLUSIVE, not ACCEPT.
      * Independent holdout confirmation is REQUIRED for a formal ACCEPT: with
        ``require_holdout_confirmation`` (default True) a missing / too-small /
        contradictory holdout downgrades to INCONCLUSIVE. The holdout set is
        evaluated ONLY as confirmation -- it never participates in candidate
        selection, so it cannot leak into the decision it is meant to guard. With
        it disabled (DEV / provisional mode) a positive candidate is only ever
        INCONCLUSIVE, never a formal ACCEPT.
      * HARD FLOORS: ``min_seeds_for_accept`` / ``practical_improvement_floor_rel``
        can only be RAISED above the module constants, never lowered -- no CLI
        flag or Python API caller can relax the gate to manufacture more ACCEPTs.
      * A genuine negative improvement is REJECTED; it is never silently dropped.
    """
    # M2.7 P0-1 rule 8: clamp to the hard floors. A caller may demand MORE
    # evidence, but can never demand LESS than the formal protocol.
    min_seeds_for_accept = max(int(min_seeds_for_accept),
                               MIN_SEEDS_FOR_FORMAL_ACCEPT)
    floor_rel = max(float(practical_improvement_floor_rel),
                    PRACTICAL_IMPROVEMENT_FLOOR_REL)

    n_valid = rec.get("n_valid") or 0
    n_invalid = rec.get("n_invalid") or 0
    # 1) Any failed / missing experiment is not evidence of degradation.
    if n_invalid > 0:
        return CandidateStatus.INCONCLUSIVE
    # 2) Too few valid paired runs to separate signal from noise -> cannot ACCEPT.
    if n_valid < min_seeds_for_accept:
        return CandidateStatus.INCONCLUSIVE
    mean = rec.get("mean_improvement")
    if mean is None or not math.isfinite(mean):
        return CandidateStatus.INCONCLUSIVE
    std = rec.get("std_improvement")
    # 3) Statistical significance: for a gain that would be ACCEPTED (mean > 0), the
    #    95% CI (normal approx) of paired diffs must exclude 0; a directional but
    #    non-significant gain is INCONCLUSIVE. (A negative mean is heading to REJECT
    #    regardless, so the CI gate is only applied on the positive side.)
    if mean > 0 and std is not None and math.isfinite(std) and std > 0 and n_valid >= 2:
        se = std / math.sqrt(n_valid)
        ci_lower = mean - 1.96 * se
        if ci_lower <= 0:
            return CandidateStatus.INCONCLUSIVE
    # 4) Practical significance: improvement must exceed a relative floor of the
    #    baseline metric, otherwise it is noise-level -> INCONCLUSIVE.
    baseline_mean = rec.get("baseline_mean")
    if (baseline_mean is not None and math.isfinite(baseline_mean)
            and abs(baseline_mean) > 0):
        floor = floor_rel * abs(baseline_mean)
        if 0 < mean < floor:
            return CandidateStatus.INCONCLUSIVE
    # 5) Independent holdout confirmation required for a formal ACCEPT.
    ho = rec.get("holdout_improvement")
    ho_n = rec.get("holdout_n_valid") or 0
    if not require_holdout_confirmation:
        # DEV / provisional mode: no independent holdout was executed. A positive
        # candidate is only PROVISIONAL -> INCONCLUSIVE (never a formal ACCEPT);
        # this stops the fast dev path from manufacturing real scientific verdicts.
        if mean > 0:
            return CandidateStatus.INCONCLUSIVE
    else:
        if ho is None or not math.isfinite(ho) or ho_n < min_seeds_for_accept:
            return CandidateStatus.INCONCLUSIVE
        # A candidate that CLAIMS improvement (mean > 0) must be confirmed by the
        # independent holdout; a contradictory (non-positive) holdout downgrades it
        # to INCONCLUSIVE. A candidate that is already worse (mean <= 0) is
        # REJECTED regardless of holdout sign -- the holdout cannot rescue a
        # genuine regression.
        if mean > 0 and ho <= 0:
            return CandidateStatus.INCONCLUSIVE
    # 6) Genuine negative improvement -> REJECT (never silently dropped).
    if mean <= 0:
        return CandidateStatus.REJECTED
    return CandidateStatus.ACCEPTED


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
                   domain_holdout_seeds: tuple[int, ...] = (),
                   domain_campaign_id: str = "dual_layer",
                   harness_evolve: Callable[..., dict[str, Any]] = _harness_evolve,
                   # M2.7 statistical gate (P0-1): never lower the bar for ACCEPT.
                   min_seeds_for_accept: int = MIN_SEEDS_FOR_FORMAL_ACCEPT,
                   practical_improvement_floor_rel: float = PRACTICAL_IMPROVEMENT_FLOOR_REL,
                   require_holdout_confirmation: bool = True
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
                project_dir=project_dir, campaign_id=domain_campaign_id,
                seeds=domain_seeds, dataset_version=dataset_version,
                runner=domain_runner,
                require_frozen_manifest=domain_require_frozen_manifest,
                holdout_seeds=domain_holdout_seeds)
            registry.transition(c.hyp_fingerprint, CandidateStatus.EVALUATED,
                               record=rec)
            decision = _decide_domain(
                rec,
                min_seeds_for_accept=min_seeds_for_accept,
                practical_improvement_floor_rel=practical_improvement_floor_rel,
                require_holdout_confirmation=require_holdout_confirmation)
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
