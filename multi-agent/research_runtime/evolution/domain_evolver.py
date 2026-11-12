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
                              require_frozen_manifest: bool = True
                              ) -> dict[str, Any]:
    """Paired Base/Candidate experiment for ONE domain candidate on fixed seeds.

    Engine execution is injected via ``runner`` so it reuses the existing domain
    campaign. When ``runner`` is None a clear error is raised: a fabricated
    metric is NEVER accepted as real evidence.
    """
    if runner is None:
        raise NotImplementedError(
            "evaluate_domain_candidate requires an injected domain engine "
            "runner (the existing domain campaign). A fabricated metric is "
            "never accepted as real evidence.")
    base_Q: list[float] = []
    cand_Q: list[float] = []
    for seed in seeds:
        arts = materialize_domain_candidate(
            domain=domain, candidate=candidate, base_config=base_config,
            project_dir=project_dir, campaign_id=campaign_id, seed=seed,
            dataset_version=dataset_version,
            require_frozen_manifest=require_frozen_manifest)
        base_Q.append(runner(arts["base"]))
        cand_Q.append(runner(arts["candidate"]))
    deltas = [round(c - b, 6) for b, c in zip(base_Q, cand_Q)]
    return {
        "candidate_type": CandidateType.DOMAIN_MODEL.value,
        "domain": domain,
        "target": candidate.target,
        "old": candidate.old,
        "new": candidate.new,
        "taxonomy": candidate.taxonomy,
        "hyp_fingerprint": candidate.hyp_fingerprint,
        "base_Q": base_Q,
        "cand_Q": cand_Q,
        "deltas": deltas,
        "mean_delta": round(sum(deltas) / max(1, len(deltas)), 6),
        "n_seeds": len(seeds),
    }
