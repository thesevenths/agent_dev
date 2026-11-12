"""Domain registry: decouple domain selection from the generic research runtime.

The generic ``ResearchService`` / ``CampaignRunner`` / ``Comparator`` never import
a specific engine, dataset or metric. A *domain* supplies BOTH a research
*policy* (``ResearchDomainPolicy``: allowed config keys, value rules, budgets,
metric protocol, approval policy) AND an *adapter* (``ModelAdapter``: how to run
the engine), plus the campaign/comparison stores and factories that are specific
to that engine's on-disk layout.

Registering a new domain (AI4S / Embodied / Quant) therefore requires NO change
to the generic orchestration, campaign or comparison layers -- exactly the R2
requirement that "a domain must be replaceable without modifying the generic
orchestration and comparison logic".

Adding a domain is a two-step, additive change:

1. implement a ``ResearchDomainPolicy`` factory and a ``ModelAdapter`` subclass;
2. ``register_domain(DomainBundle(...))``.

Nothing in ``research_service.py`` / ``campaign/`` / ``comparison/`` changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .adapters.base import ModelAdapter
from .research_policy import ResearchDomainPolicy


DatasetVerifier = Callable[[str, str], "dict[str, Any]"]


@dataclass
class DomainBundle:
    """Everything the generic runtime needs for ONE research domain.

    The generic ``ResearchService`` consumes these untouched; the domain owns the
    engine-specific pieces (adapter, stores, runner/comparator factories).
    """

    name: str
    policy_factory: Callable[[], ResearchDomainPolicy]
    adapter_factory: Callable[[], ModelAdapter]
    campaign_store: Any
    report_store: Any
    campaign_runner_factory: Callable[[], Any]
    comparator_factory: Callable[[], Any]
    dataset_verifier: DatasetVerifier
    campaign_root: str = ""
    description: str = ""

    # ARR 1.0 / D1 + D3: additive, optional. A domain may inject its OWN
    # research-quality evaluation protocol and metric interpretation. The
    # generic runtime never touches these; the unified task driver reads them
    # to score a ResearchReport with the domain's own scientific rules.
    # Defaults keep every pre-ARR DomainBundle construction valid unchanged.
    quality_evaluator: Any = None
    metric_interpreter: dict[str, Any] | None = None


class DomainRegistry:
    def __init__(self) -> None:
        self._bundles: dict[str, DomainBundle] = {}

    def register(self, bundle: DomainBundle) -> None:
        if not bundle.name:
            raise ValueError("domain bundle requires a non-empty name")
        self._bundles[bundle.name] = bundle

    def contains(self, name: str) -> bool:
        return name in self._bundles

    def get(self, name: str) -> DomainBundle:
        bundle = self._bundles.get(name)
        if bundle is None:
            raise KeyError(
                f"unknown research domain {name!r}; registered domains: "
                f"{sorted(self._bundles)}"
            )
        return bundle

    def names(self) -> list[str]:
        return sorted(self._bundles)


# Process-wide default registry. ``build_default_research_service`` and the
# registration helpers below operate on this instance.
default_registry = DomainRegistry()


def register_domain(bundle: DomainBundle) -> None:
    """Register a domain bundle in the process-wide registry."""
    default_registry.register(bundle)


def get_domain(name: str) -> DomainBundle:
    """Return a registered bundle or raise ``KeyError``."""
    return default_registry.get(name)
