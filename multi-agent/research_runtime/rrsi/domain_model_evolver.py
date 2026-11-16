"""M5-02 Layer 1 -- domain model evolution via the EXISTING domain evolver."""
from __future__ import annotations

from typing import Any, Callable

from ..evolution.domain_evolver import (
    evaluate_domain_candidate, generate_domain_candidates,
)
from .evolution_experiment import DOMAIN_MODEL_LAYER


def generate(*, domain: str, dominant_failure: str,
             limit: int = 3) -> list[dict[str, Any]]:
    """Propose candidates using ONLY pre-registration/diagnosis evidence.

    Held-out metrics are deliberately NOT an argument: they cannot leak into
    candidate selection.
    """
    candidates = generate_domain_candidates(
        domain=domain, dominant=dominant_failure, limit=limit)
    # DomainModelCandidate is a dataclass (to_dict), not a pydantic model.
    return [c.to_dict() for c in candidates]


def evaluate(*, domain: str, candidate: dict[str, Any],
             base_config: str, project_dir: str, campaign_id: str,
             seeds: tuple[int, ...], dataset_version: str,
             runner: Callable[[dict[str, Any]], float],
             holdout_seeds: tuple[int, ...] = (),
             ood_config_path: str | None = None) -> dict[str, Any]:
    """Evaluate with a REAL injected engine runner; no runner = no evidence."""
    from ..evolution.candidate import DomainModelCandidate
    # DomainModelCandidate is a dataclass: build it from the constructor fields
    # only (to_dict carries extra bookkeeping keys that are not ctor args).
    _fields = {"candidate_id", "domain", "target", "old", "new",
               "taxonomy", "overrides", "hypothesis", "status",
               "hyp_fingerprint"}
    kwargs = {k: v for k, v in candidate.items() if k in _fields}
    if "domain" not in kwargs:
        kwargs["domain"] = domain
    if not kwargs.get("target"):
        raise ValueError("candidate has no whitelisted target")
    model = DomainModelCandidate(**kwargs)
    result = evaluate_domain_candidate(
        domain=domain, candidate=model, base_config=base_config,
        project_dir=project_dir, campaign_id=campaign_id, seeds=seeds,
        dataset_version=dataset_version, runner=runner,
        holdout_seeds=holdout_seeds, ood_config_path=ood_config_path)
    return {"layer": DOMAIN_MODEL_LAYER, **result}


__all__ = ["generate", "evaluate", "DOMAIN_MODEL_LAYER"]
