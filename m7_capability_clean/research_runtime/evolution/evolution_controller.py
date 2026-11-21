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
from .statistics import (
    DEFAULT_ALPHA, paired_p_value, paired_confidence_interval, holm_step_down,
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

#: M3.0: an independent holdout must confirm the paired direction on at least
#: this fraction of its seeds (a strict majority), IN ADDITION to a same-sign
#: holdout mean. A single lucky holdout seed must not be able to confirm a gain.
HOLDOUT_DIRECTION_MIN_FRACTION = 0.6


def _finite_number(x: Any) -> bool:
    """A real, finite number (bools are NOT numbers for evidence purposes)."""
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x))


def _nonneg_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _significance_direction(rec: dict[str, Any], alpha: float,
                           prefix: str = "") -> str | None:
    """Two-sided Student-t decision for ONE paired-improvement sample.

    Returns ``"positive"`` only when the whole ``1-alpha`` CI is above 0,
    ``"negative"`` only when the whole CI is below 0, and ``None`` when the CI
    straddles 0 (or the significance level is unusable). This is the M3.0
    replacement for M2.7's ``mean - 1.96*se`` normal approximation, which
    underestimated uncertainty at n=5 (correct t critical value: 2.776).

    ``std == 0`` is NOT treated as missing: all paired differences are identical,
    so the CI degenerates to the single point ``mean`` and the direction is exact.

    ``prefix`` namespaces the field reads so the SAME logic decides the OOD axis
    (``prefix="ood_"``) as the ID axis (``prefix=""``) under the M3.3 extension.
    A missing / malformed axis fails closed to ``None`` (inconclusive).
    """
    if not _finite_number(alpha) or not (0.0 < alpha < 1.0):
        return None
    n = rec.get(f"{prefix}n_valid")
    mean = rec.get(f"{prefix}mean_improvement")
    std = rec.get(f"{prefix}std_improvement")
    if (not _nonneg_int(n) or n < 2 or not _finite_number(mean)
            or not _finite_number(std) or std < 0):
        return None
    lo, hi = paired_confidence_interval(mean, std, n, alpha)
    if lo > 0.0:
        return "positive"
    if hi < 0.0:
        return "negative"
    return None


def _holdout_confirms(rec: dict[str, Any], direction: str,
                      min_seeds: int, prefix: str = "") -> bool:
    """Independent holdout CONFIRMATION using per-seed raw results.

    M2.7 only checked the holdout *mean*. M3.0 additionally requires:
      * every requested holdout seed produced a valid paired run (no silent
        partial failure -- ``holdout_n_seeds == holdout_n_valid`` and
        ``holdout_n_invalid == 0``);
      * the persisted per-seed raw improvements are complete and finite;
      * a strict majority of holdout seeds point in the SAME direction as the
        paired verdict, and the holdout mean agrees.

    The holdout numbers never enter ``improvements`` / ``mean_improvement``, so
    they cannot leak into candidate selection.
    """
    ho_mean = rec.get(f"{prefix}holdout_improvement")
    ho_n = rec.get(f"{prefix}holdout_n_valid")
    ho_seeds = rec.get(f"{prefix}holdout_n_seeds")
    ho_invalid = rec.get(f"{prefix}holdout_n_invalid")
    ho_imp = rec.get(f"{prefix}holdout_improvements")
    if not _finite_number(ho_mean) or not _nonneg_int(ho_n) or ho_n < min_seeds:
        return False
    if not _nonneg_int(ho_seeds) or ho_seeds != ho_n:
        return False
    if not _nonneg_int(ho_invalid) or ho_invalid != 0:
        return False
    if not isinstance(ho_imp, (list, tuple)) or len(ho_imp) != ho_n:
        return False
    if not all(_finite_number(x) for x in ho_imp):
        return False
    frac_positive = sum(1 for x in ho_imp if x > 0) / ho_n
    if direction == "positive":
        return (ho_mean > 0.0
                and frac_positive >= HOLDOUT_DIRECTION_MIN_FRACTION)
    return (ho_mean < 0.0
            and (1.0 - frac_positive) >= HOLDOUT_DIRECTION_MIN_FRACTION)


def domain_family_significance(records: list[dict[str, Any]],
                               alpha: float = DEFAULT_ALPHA,
                               prefix: str = "") -> list[dict[str, Any]]:
    """Family-wise Holm correction for ONE round's domain candidates.

    A round evaluates several candidates (e.g. three learning rates) and then
    picks the best; using an uncorrected 95% CI for each inflates the chance of a
    false-positive "winner". Holm step-down controls the family-wise error rate.

    ``alpha`` can only be made STRICTER than the frozen protocol constant --
    a caller must never be able to widen it to manufacture ACCEPTs. Records with
    missing / invalid statistics get p=1.0 (least significant), never a free pass.

    ``prefix`` namespaces the axis (``""`` = ID, ``"ood_"`` = OOD) so the SAME
    family correction applies to the OOD axis under the M3.3 extension.
    """
    effective_alpha = min(float(alpha), DEFAULT_ALPHA)
    p_values: list[float] = []
    for rec in records:
        n = rec.get(f"{prefix}n_valid")
        mean = rec.get(f"{prefix}mean_improvement")
        std = rec.get(f"{prefix}std_improvement")
        if (not _nonneg_int(n) or n < 2 or not _finite_number(mean)
                or not _finite_number(std) or std < 0
                or (rec.get(f"{prefix}n_invalid") or 0) > 0):
            p_values.append(1.0)
        else:
            p_values.append(paired_p_value(mean, std, n))
    return holm_step_down(p_values, alpha=effective_alpha)


def _decide_domain(rec: dict[str, Any], *,
                   min_seeds_for_accept: int = MIN_SEEDS_FOR_FORMAL_ACCEPT,
                   practical_improvement_floor_rel: float = PRACTICAL_IMPROVEMENT_FLOOR_REL,
                   require_holdout_confirmation: bool = True,
                   holm_alpha: float | None = None,
                   prefix: str = ""
                   ) -> CandidateStatus:
    """Fixed, transparent, BIDIRECTIONAL statistical decision for one candidate.

    M3.0 protocol (frozen; no fabricated verdicts, no lowered bar):
      * Any NaN / Inf / None / missing experiment (``n_invalid > 0``) is
        INCONCLUSIVE -- a failed run is NOT "real degradation".
      * Fewer than ``min_seeds_for_accept`` valid paired runs is INCONCLUSIVE.
      * MISSING STATISTICS FAIL CLOSED: ``n_valid`` / ``n_invalid`` /
        ``mean_improvement`` / ``std_improvement`` / ``baseline_mean`` must all
        be finite and well-formed. M2.7 silently SKIPPED the CI check when std
        was missing and SKIPPED the practical floor when baseline_mean was
        missing; both are now hard INCONCLUSIVE.
      * BIDIRECTIONAL significance: a two-sided STUDENT-T CI (Holm-adjusted
        ``alpha``) must lie wholly above 0 for ACCEPT, or wholly below 0 for
        REJECT. A CI straddling 0 is INCONCLUSIVE in EITHER direction -- M2.7
        used the normal 1.96 critical value and rejected on ``mean <= 0`` alone.
      * Practical significance: a positive gain must exceed a relative floor of
        the baseline metric; a noise-level gain is INCONCLUSIVE, not ACCEPT.
      * Independent holdout confirmation is REQUIRED: >= min_seeds valid
        holdout seeds, all requested seeds valid, per-seed raw results persisted,
        a same-sign mean AND a strict majority of seeds in the same direction.
        The holdout never participates in candidate selection.
      * DEV / provisional mode (``require_holdout_confirmation=False``): a
        positive candidate is only ever INCONCLUSIVE, never a formal ACCEPT.
      * HARD FLOORS: caller parameters can only RAISE the bar. ``holm_alpha`` is
        clamped to at most ``DEFAULT_ALPHA``.
    """
    min_seeds_for_accept = max(int(min_seeds_for_accept),
                               MIN_SEEDS_FOR_FORMAL_ACCEPT)
    floor_rel = max(float(practical_improvement_floor_rel),
                    PRACTICAL_IMPROVEMENT_FLOOR_REL)
    alpha = (DEFAULT_ALPHA if holm_alpha is None
             else min(float(holm_alpha), DEFAULT_ALPHA))

    # 1) Evidence hygiene -- fail closed on ANY missing / malformed statistic.
    n_valid = rec.get(f"{prefix}n_valid")
    n_invalid = rec.get(f"{prefix}n_invalid")
    if not _nonneg_int(n_valid) or not _nonneg_int(n_invalid):
        return CandidateStatus.INCONCLUSIVE
    if n_invalid > 0 or n_valid < min_seeds_for_accept:
        return CandidateStatus.INCONCLUSIVE
    mean = rec.get(f"{prefix}mean_improvement")
    std = rec.get(f"{prefix}std_improvement")
    baseline_mean = rec.get(f"{prefix}baseline_mean")
    if not _finite_number(mean) or not _finite_number(std) or std < 0:
        return CandidateStatus.INCONCLUSIVE
    if not _finite_number(baseline_mean) or abs(baseline_mean) <= 0.0:
        return CandidateStatus.INCONCLUSIVE

    # 2) Bidirectional Student-t significance (Holm-adjusted when in a family).
    direction = _significance_direction(rec, alpha, prefix=prefix)
    if direction is None:
        return CandidateStatus.INCONCLUSIVE

    # 3) Practical significance on the positive side.
    if direction == "positive":
        if mean < floor_rel * abs(baseline_mean):
            return CandidateStatus.INCONCLUSIVE

    # 4) Independent holdout confirmation (per-seed direction, not just mean).
    if not require_holdout_confirmation:
        if direction == "positive":
            return CandidateStatus.INCONCLUSIVE
    elif not _holdout_confirms(rec, direction, min_seeds_for_accept,
                               prefix=prefix):
        return CandidateStatus.INCONCLUSIVE

    # 5) Only statistically significant directions produce a terminal verdict.
    return (CandidateStatus.ACCEPTED if direction == "positive"
            else CandidateStatus.REJECTED)


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


def _combine_dual_layer_decisions(id_status: CandidateStatus,
                                 ood_status: CandidateStatus) -> CandidateStatus:
    """Combine the ID and OOD axis verdicts into ONE production verdict (M3.3).

    A candidate is PRODUCTION-ACCEPTED only when BOTH axes accept it -- that is
    the entire point of the OOD gate: an ID-only gain that collapses under
    distribution shift must not be shipped. Semantics:

      * ID ACCEPT + OOD ACCEPT       -> ACCEPTED  (OOD-confident production).
      * ID ACCEPT + OOD REJECTED     -> REJECTED  (ID-overfit exposed: the change
                                        helps the training regime but HARMS the
                                        deployment regime -- strictly worse than
                                        neutral, so it is rejected, not merely
                                        left undecided).
      * ID ACCEPT + OOD INCONCLUSIVE -> INCONCLUSIVE (cannot claim OOD robustness;
                                        the gate stays shut).
      * ID not ACCEPTED (REJECT / INCONCLUSIVE) -> unchanged (OOD is irrelevant).
    """
    if id_status == CandidateStatus.ACCEPTED:
        if ood_status == CandidateStatus.ACCEPTED:
            return CandidateStatus.ACCEPTED
        if ood_status == CandidateStatus.REJECTED:
            return CandidateStatus.REJECTED
        return CandidateStatus.INCONCLUSIVE
    return id_status


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
                   require_holdout_confirmation: bool = True,
                   # M3.3: an independent regime-shift (OOD) dataset. When set,
                   # every candidate is ALSO scored on this dataset with the SAME
                   # gate (namespaced "ood_"); the production verdict is the
                   # combination of the ID and OOD axes (see
                   # _combine_dual_layer_decisions). None -> behaviour unchanged.
                   ood_config_path: str | Path | None = None
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
    if domain and domain_base_config and (domain_runner is not None
                                         or ood_config_path is not None):
        # When an OOD axis is requested but no runner was supplied, build the
        # REAL engine runner with the OOD config wired in. (The M3.0 driver
        # supplies its own runner with ood_config_path; this branch only fires
        # when run_dual_layer is asked to build one.)
        if ood_config_path is not None and domain_runner is None:
            from research_runtime.cli import _build_real_domain_runner
            domain_runner = _build_real_domain_runner(
                domain, project_dir, ood_config_path=ood_config_path)
        dominant = harness_payload.get("dominant_failure", "")
        cands = generate_domain_candidates(domain=domain, dominant=dominant)
        # ---- Phase 1: evaluate EVERY candidate first (real engine runs) ----
        # M3.0: Holm correction needs the whole family's p-values, so decisions
        # can no longer be made one-by-one inside the evaluation loop. The OOD
        # axis is produced in the SAME pass (scinet_compute writes both
        # backtest_metrics.json and ood_metrics.json), so no extra engine run.
        evaluated: list[tuple[Any, dict[str, Any]]] = []
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
                holdout_seeds=domain_holdout_seeds,
                ood_config_path=ood_config_path)
            registry.transition(c.hyp_fingerprint, CandidateStatus.EVALUATED,
                               record=rec)
            evaluated.append((c, rec))
        # ---- Phase 2: family-wise correction, then per-candidate verdict ----
        holm = domain_family_significance([rec for _, rec in evaluated])
        for (c, rec), h in zip(evaluated, holm):
            rec["statistics"] = {
                "protocol": "M3.0-student-t-holm",
                "family_alpha": DEFAULT_ALPHA,
                "p_value": round(h["p_value"], 9),
                "holm_effective_alpha": h["effective_alpha"],
                "holm_significant": h["significant"],
                "holm_rank": h["rank"],
                "n_compared": h["n_compared"],
                "method": h["method"],
                "axis": "id",
            }
            decision = _decide_domain(
                rec,
                min_seeds_for_accept=min_seeds_for_accept,
                practical_improvement_floor_rel=practical_improvement_floor_rel,
                require_holdout_confirmation=require_holdout_confirmation,
                holm_alpha=h["effective_alpha"])
            rec["decision"] = decision.value
            if ood_config_path is None:
                # M3.0 (no OOD axis): the ID verdict IS the production verdict,
                # so persist it as the single terminal transition here.
                registry.transition(c.hyp_fingerprint, decision, record=rec)
            else:
                # M3.3: the production verdict is the ID+OOD COMBINATION, decided
                # in Phase 3. We must NOT terminal-transition now, because the
                # registry state machine forbids a second terminal transition
                # (e.g. ACCEPTED -> REJECTED). Phase 3 issues exactly ONE
                # terminal transition per candidate to the combined verdict.
                pass
            domain_records.append(rec)

        # ---- Phase 3 (M3.3): OOD axis -- SAME gate, namespaced "ood_" ----
        # Reuses the identical M3.0 protocol on the regime-shift axis. The final
        # production verdict combines ID + OOD via _combine_dual_layer_decisions.
        if ood_config_path is not None:
            ood_holm = domain_family_significance(
                [rec for _, rec in evaluated], prefix="ood_")
            for (c, rec), h in zip(evaluated, ood_holm):
                ood_decision = _decide_domain(
                    rec, prefix="ood_",
                    min_seeds_for_accept=min_seeds_for_accept,
                    practical_improvement_floor_rel=practical_improvement_floor_rel,
                    require_holdout_confirmation=require_holdout_confirmation,
                    holm_alpha=h["effective_alpha"])
                rec["ood_statistics"] = {
                    "protocol": "M3.0-student-t-holm",
                    "family_alpha": DEFAULT_ALPHA,
                    "p_value": round(h["p_value"], 9),
                    "holm_effective_alpha": h["effective_alpha"],
                    "holm_significant": h["significant"],
                    "holm_rank": h["rank"],
                    "n_compared": h["n_compared"],
                    "method": h["method"],
                    "axis": "ood",
                }
                rec["ood_decision"] = ood_decision.value
                id_decision = CandidateStatus(rec["decision"])
                rec["decision"] = _combine_dual_layer_decisions(
                    id_decision, ood_decision).value
                # Single terminal transition to the COMBINED verdict. In OOD mode
                # this is the ONLY terminal transition (Phase 2 withheld it), so
                # the registry state machine is respected (EVALUATED -> terminal).
                registry.transition(
                    c.hyp_fingerprint,
                    CandidateStatus(rec["decision"]), record=rec)

    ood_config_resolved = (str(ood_config_path)
                           if ood_config_path is not None else None)
    if ood_config_path is not None:
        ood_counts: dict[str, int | None] = {
            "ood_n_accepted": sum(1 for r in domain_records
                                 if r.get("ood_decision") == CandidateStatus.ACCEPTED.value),
            "ood_n_rejected": sum(1 for r in domain_records
                                 if r.get("ood_decision") == CandidateStatus.REJECTED.value),
            "ood_n_inconclusive": sum(1 for r in domain_records
                                     if r.get("ood_decision") == CandidateStatus.INCONCLUSIVE.value),
        }
    else:
        ood_counts = {
            "ood_n_accepted": None, "ood_n_rejected": None,
            "ood_n_inconclusive": None,
        }
    payload: dict[str, Any] = {
        "schema_version": "arr-1.1-dual-layer-v1",
        "elapsed_seconds": round(time.time() - started, 3),
        "harness_layer": harness_payload,
        "domain_layer": {
            "domain": domain,
            "candidate_type": CandidateType.DOMAIN_MODEL.value,
            "statistics_protocol": {
                "name": "M3.0-student-t-holm",
                "family_alpha": DEFAULT_ALPHA,
                "n_compared": len(domain_records),
                "holdout_direction_min_fraction": HOLDOUT_DIRECTION_MIN_FRACTION,
                "bidirectional": True,
            },
            "records": domain_records,
            "n_accepted": sum(1 for r in domain_records
                             if r.get("decision") == CandidateStatus.ACCEPTED.value),
            "n_rejected": sum(1 for r in domain_records
                             if r.get("decision") == CandidateStatus.REJECTED.value),
            "n_inconclusive": sum(1 for r in domain_records
                                 if r.get("decision") == CandidateStatus.INCONCLUSIVE.value),
            "ood_config_path": ood_config_resolved,
            "ood_n_accepted": ood_counts["ood_n_accepted"],
            "ood_n_rejected": ood_counts["ood_n_rejected"],
            "ood_n_inconclusive": ood_counts["ood_n_inconclusive"],
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
