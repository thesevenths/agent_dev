"""ARR 1.1 / F2 -- DOMAIN MODEL evolution layer.

Reuses the EXISTING per-domain ``ALLOWED_KEYS`` / ``VALUE_RULES`` and the
EXISTING ``research_runtime.candidate_config.CandidateConfigBuilder`` -- it does
NOT rewrite the domain config safety boundary. A domain candidate is still a
whitelisted, dotted config override that the builder verifies (forbidden roots,
frozen dataset on disk, per-seed isolation). This module only adds the
``DomainModelCandidate`` carrier, generation scoped to a domain's whitelist, and
a paired-run hook that delegates engine execution to the existing domain
campaign.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Callable

from .candidate import CandidateStatus, DomainModelCandidate, CandidateType
from adaptive_runtime.evolution_spaces import (
    space_for, DOMAIN_TO_TARGET, is_frozen,
)
from research_runtime.candidate_config import CandidateConfigBuilder

#: The canonical layer tag for Domain-Model evolution.
DOMAIN_LAYER = "domain_model"

_DOMAIN_MODULES = {
    "scinet": "research_runtime.domains.scinet",
    "embodied": "research_runtime.domains.embodied",
    "quant": "research_runtime.domains.quant",
}


def domain_whitelist(domain: str) -> tuple[tuple[str, ...], dict[str, Any]]:
    mod = importlib.import_module(_DOMAIN_MODULES[domain])
    return tuple(getattr(mod, "ALLOWED_KEYS", ())), dict(getattr(mod, "VALUE_RULES", {}))


#: Canonical fallback so the evaluation direction is always defined even if a
#: domain module has not yet exported it. The domain modules
#: (embodied/quant/scinet) are the SINGLE SOURCE OF TRUTH and expose
#: ``PRIMARY_METRIC`` / ``METRIC_DIRECTION``; this table only guards against a
#: missing export.
_METRIC_FALLBACK: dict[str, tuple[str, str]] = {
    "embodied": ("state_mse@1", "minimize"),
    "quant": ("net_return_after_cost", "maximize"),
    "scinet": ("latent_mse@1", "minimize"),
}


def domain_metric(domain: str) -> tuple[str, str]:
    """Return ``(primary_metric, direction)`` for a domain.

    ``direction`` is ``"minimize"`` or ``"maximize"``. It is read from the
    domain module's exported ``PRIMARY_METRIC`` / ``METRIC_DIRECTION`` so the
    optimization goal lives in exactly one place and is shared by the research
    task, the engine runner and the dual-layer RRSI decision.
    """
    mod = importlib.import_module(_DOMAIN_MODULES[domain])
    pm = getattr(mod, "PRIMARY_METRIC", None)
    md = getattr(mod, "METRIC_DIRECTION", None)
    if pm and md:
        return str(pm), str(md)
    return _METRIC_FALLBACK.get(domain, ("unknown", "maximize"))


def _is_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def _coerce(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return int(raw) if raw.isdigit() else float(raw)
        except ValueError:
            return raw
    return raw


def generate_domain_candidates(*, domain: str, dominant: str,
                               limit: int = 3) -> list[DomainModelCandidate]:
    """Generate DOMAIN_MODEL candidates scoped to the domain's whitelist.

    The target is chosen from ``DOMAIN_TO_TARGET`` for the dominant failure, then
    constrained to the domain's own ``ALLOWED_KEYS`` so we only ever propose
    knobs the engine actually consumes. Returns an empty list when the dominant's
    target is not whitelisted for that domain (the caller then falls back to its
    own harness loop rather than fabricating a domain change).
    """
    allowed, _ = domain_whitelist(domain)
    space = space_for(CandidateType.DOMAIN_MODEL)
    target = DOMAIN_TO_TARGET.get(dominant)
    if target is None or target not in allowed:
        target = next((k for k in space.search_space if k in allowed), None)
    if target is None or is_frozen(CandidateType.DOMAIN_MODEL, target):
        return []
    spec = space.search_space[target]
    out: list[DomainModelCandidate] = []
    for raw_new in spec.get("values", []):
        new = _coerce(raw_new)
        cand = DomainModelCandidate(
            candidate_id="", domain=domain, target=target, old=None, new=new,
            taxonomy=dominant, hypothesis=spec.get("rationale", ""))
        out.append(cand)
        if len(out) >= limit:
            break
    return out


def materialize_domain_candidate(*, domain: str, candidate: DomainModelCandidate,
                                 base_config: str | Path, project_dir: str | Path,
                                 campaign_id: str, seed: int,
                                 dataset_version: str,
                                 require_frozen_manifest: bool = True
                                 ) -> dict[str, Any]:
    """Build isolated baseline + candidate configs via the EXISTING builder.

    Returns the artifact descriptors (directories, config paths, sha256) for the
    engine to consume. The safety boundary (whitelist, forbidden roots, frozen
    dataset, per-seed isolation) is enforced by ``CandidateConfigBuilder``.
    """
    allowed, value_rules = domain_whitelist(domain)
    builder = CandidateConfigBuilder(
        project_dir, allowed_keys=allowed, value_rules=value_rules,
        require_frozen_manifest=require_frozen_manifest)
    base = builder.build_baseline(
        base_config=base_config, campaign_id=campaign_id, seed=seed,
        dataset_version=dataset_version)
    cand = builder.build(
        base_config=base_config, campaign_id=campaign_id,
        candidate_id=candidate.candidate_id,
        hypothesis_id=candidate.candidate_id, seed=seed,
        overrides={candidate.target: candidate.new},
        dataset_version=dataset_version)
    return {
        "base": {
            "directory": str(base.directory),
            "config_path": str(base.config_path),
            "baseline_config_sha256": base.baseline_config_sha256,
            "candidate_config_sha256": base.candidate_config_sha256,
        },
        "candidate": {
            "directory": str(cand.directory),
            "config_path": str(cand.config_path),
            "baseline_config_sha256": cand.baseline_config_sha256,
            "candidate_config_sha256": cand.candidate_config_sha256,
        },
    }


#: A domain engine runner takes ONE materialized artifact descriptor and returns
#: the REAL domain metric Q (latent_mse / state_mse / sharpe ...). It is injected
#: so the heavy, domain-specific engine (SciNet++ / simulator / backtest) is
#: reused from the existing campaign rather than faked here.
DomainRunner = Callable[[dict[str, Any]], float]


def evaluate_domain_candidate(*, domain: str, candidate: DomainModelCandidate,
                              base_config: str | Path, project_dir: str | Path,
                              campaign_id: str, seeds: tuple[int, ...],
                              dataset_version: str,
                              runner: DomainRunner | None = None,
                              require_frozen_manifest: bool = True,
                              holdout_seeds: tuple[int, ...] = (),
                              min_valid_seeds: int = 2
                              ) -> dict[str, Any]:
    """Paired Base/Candidate experiment for ONE domain candidate on fixed seeds.

    Engine execution is injected via ``runner`` so it reuses the existing domain
    campaign. When ``runner`` is None a clear error is raised: a fabricated
    metric is NEVER accepted as real evidence.

    The reported ``improvements`` are DIRECTION-NORMALIZED so that a strictly
    positive value always means the candidate is better than baseline, regardless
    of whether the domain metric is minimized (``state_mse@1``) or maximized
    (``net_return_after_cost``). Raw ``deltas`` (cand - base) are kept for
    traceability only; the decision is made on ``improvements``.

    Evidence hygiene (no fabricated verdicts):
      * NaN / Inf / None / missing metrics are NOT treated as a real delta. They
        increment ``n_invalid``; a candidate with any invalid run, or with fewer
        than ``min_valid_seeds`` valid paired runs, is INCONCLUSIVE -- never
        silently ACCEPTED or counted as genuine REJECTION.
      * An optional independent ``holdout_seeds`` set is evaluated; if it clearly
        contradicts the paired improvement, the decision is downgraded to
        INCONCLUSIVE (the gain is not trustworthy on unseen data).
    """
    if runner is None:
        raise NotImplementedError(
            "evaluate_domain_candidate requires an injected domain engine "
            "runner (the existing domain campaign). A fabricated metric is "
            "never accepted as real evidence.")
    primary_metric, direction = domain_metric(domain)
    # sign: +1 -> larger (cand - base) is better (maximize);
    #       -1 -> smaller (cand - base) is better (minimize).
    sign = -1.0 if direction == "minimize" else 1.0

    def _run_on(run_seeds: tuple[int, ...]) -> tuple[list[float], list[float]]:
        bQ: list[float] = []
        cQ: list[float] = []
        for seed in run_seeds:
            arts = materialize_domain_candidate(
                domain=domain, candidate=candidate, base_config=base_config,
                project_dir=project_dir, campaign_id=campaign_id, seed=seed,
                dataset_version=dataset_version,
                require_frozen_manifest=require_frozen_manifest)
            bQ.append(runner(arts["base"]))
            cQ.append(runner(arts["candidate"]))
        return bQ, cQ

    base_Q, cand_Q = _run_on(seeds)
    valid = [(b, c) for b, c in zip(base_Q, cand_Q)
             if _is_finite(b) and _is_finite(c)]
    n_invalid = len(base_Q) - len(valid)
    improvements = [sign * (c - b) for (b, c) in valid]
    if valid:
        mean_improvement = sum(improvements) / len(improvements)
        std_improvement = _std(improvements)
        # Raw (direction-native) baseline mean, kept so the decision can apply a
        # *relative* practical-improvement floor independent of the metric's scale
        # (latent_mse ~1e-2 vs net_return ~1e-1 must share one threshold).
        baseline_mean = sum(b for (b, c) in valid) / len(valid)
    else:
        mean_improvement = None
        std_improvement = None
        baseline_mean = None

    rec: dict[str, Any] = {
        "candidate_type": CandidateType.DOMAIN_MODEL.value,
        "domain": domain,
        "primary_metric": primary_metric,
        "metric_direction": direction,
        "target": candidate.target,
        "old": candidate.old,
        "new": candidate.new,
        "taxonomy": candidate.taxonomy,
        "hyp_fingerprint": candidate.hyp_fingerprint,
        "base_Q": base_Q,
        "cand_Q": cand_Q,
        # Raw (cand - base) deltas, kept for traceability only.
        "deltas": [round(c - b, 6) for b, c in zip(base_Q, cand_Q)],
        # Direction-normalized: positive == candidate is BETTER.
        "improvements": [round(x, 6) for x in improvements],
        "mean_improvement": (round(mean_improvement, 6)
                             if mean_improvement is not None else None),
        "std_improvement": (round(std_improvement, 6)
                            if std_improvement is not None else None),
        "baseline_mean": (round(baseline_mean, 6)
                          if baseline_mean is not None else None),
        "n_seeds": len(seeds),
        "n_valid": len(valid),
        "n_invalid": n_invalid,
        "min_valid_seeds": min_valid_seeds,
    }
    if holdout_seeds:
        ho_base, ho_cand = _run_on(holdout_seeds)
        ho_valid = [(b, c) for b, c in zip(ho_base, ho_cand)
                    if _is_finite(b) and _is_finite(c)]
        if ho_valid:
            ho_imp = [sign * (c - b) for (b, c) in ho_valid]
            rec["holdout_improvement"] = round(sum(ho_imp) / len(ho_imp), 6)
            rec["holdout_n_valid"] = len(ho_valid)
        else:
            rec["holdout_improvement"] = None
            rec["holdout_n_valid"] = 0
    return rec
