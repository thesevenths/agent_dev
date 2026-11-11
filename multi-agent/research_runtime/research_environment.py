"""Real dependency assembly for the P1-06 tools (lazy, cached by tools.py).

Nothing in the generic service constructs an engine object. This module is the
composition root: it wires the P1-03 config builder, the P1-04 CampaignRunner
factory, the P1-05 comparator factory and the P1-06 trusted stores together.

ARR 0.1 / R2: the assembly is now expressed as a *domain bundle*. The single
generic ``ResearchService`` is built from whatever bundle the registry returns,
so a Quant / AI4S / Embodied domain can be selected (or registered) without any
change to ``research_service.py`` / ``campaign/`` / ``comparison/``.
"""
from __future__ import annotations

from .adapters import SciNetAdapter
from .domain_registry import (
    DomainBundle,
    default_registry,
    get_domain,
    register_domain,
)
from .settings import DATA_DIR, DEFAULT_SCINET_DIR, ensure_data_dir


def build_scinet_domain_bundle() -> DomainBundle:
    """Assemble the SciNet++ domain bundle from the real, trusted components.

    This isolates every engine-specific piece (policy factory, adapter, the two
    engine-shaped stores, the runner/comparator factories and the dataset
    verifier) behind one object. The generic runtime only ever sees the bundle.
    """
    from .campaign.campaign_store import CampaignStore
    from .campaign.scinet import (
        build_scinet_campaign_runner,
        scinet_campaign_policy,
    )
    from .candidate_config import CandidateConfigBuilder
    from .comparison.report_store import ComparisonReportStore
    from .comparison.scinet import build_scinet_comparator
    from .domains.scinet import scinet_research_policy

    ensure_data_dir()
    domain = scinet_research_policy()
    campaign_store = CampaignStore(DATA_DIR / "campaigns.sqlite")
    report_store = ComparisonReportStore(DEFAULT_SCINET_DIR / domain.campaign_root)
    config_builder = CandidateConfigBuilder(
        DEFAULT_SCINET_DIR,
        allowed_keys=domain.allowed_config_keys,
        value_rules=domain.value_rules,
        campaign_root=domain.campaign_root,
    )
    campaign_policy = scinet_campaign_policy(
        epochs=domain.epochs, timeout_per_job_s=domain.job_timeout_s,
        max_jobs=domain.max_jobs,
    )

    # ARR 0.1 / R2: the bundle's engine handle. A single local reference is
    # used both for the advertised ``adapter_factory`` field AND for the runner,
    # so the engine that actually executes experiments is the one the registry
    # advertises -- never an accidental second implementation.
    adapter_factory = SciNetAdapter

    def runner_factory():
        # The bundle's adapter must actually participate in experiment
        # execution, not merely sit on the bundle as metadata. We pass it
        # explicitly so the runner (and the ExperimentManager it builds) uses
        # the SAME adapter the registry advertises -- a swapped domain would
        # swap the engine here.
        bundle_adapter = adapter_factory()
        return build_scinet_campaign_runner(
            store=campaign_store, config_builder=config_builder,
            policy=campaign_policy, adapter=bundle_adapter,
        )

    def comparator_factory():
        return build_scinet_comparator(
            campaign_store=campaign_store, report_store=report_store,
            project_dir=DEFAULT_SCINET_DIR,
        )

    def dataset_verifier(baseline_config_path: str, dataset_version: str) -> dict:
        return config_builder.verify_dataset(baseline_config_path, dataset_version)

    return DomainBundle(
        name="scinet",
        policy_factory=scinet_research_policy,
        campaign_store=campaign_store,
        report_store=report_store,
        campaign_runner_factory=runner_factory,
        comparator_factory=comparator_factory,
        dataset_verifier=dataset_verifier,
        campaign_root=str(DEFAULT_SCINET_DIR / domain.campaign_root),
        adapter_factory=adapter_factory,
        description=(
            "SciNet++ Pendulum / robotics world-model research (P1-06 / P2-02)."
        ),
    )


_DEFAULT_REGISTERED = False


def _ensure_scinet_registered() -> None:
    """Lazily register the SciNet bundle so the default service keeps working."""
    global _DEFAULT_REGISTERED
    if not default_registry.contains("scinet"):
        register_domain(build_scinet_domain_bundle())
        _DEFAULT_REGISTERED = True


def build_default_research_service(domain_name: str = "scinet"):
    """Build the generic ``ResearchService`` for a registered domain.

    ``domain_name`` selects a ``DomainBundle`` from the process-wide registry.
    The default ``"scinet"`` reproduces the pre-ARR assembly exactly; any other
    registered domain is assembled through the identical generic path.
    """
    from .research_service import ResearchDependencies, ResearchService
    from .research_store import ResearchStores

    _ensure_scinet_registered()
    bundle = get_domain(domain_name)
    ensure_data_dir()
    stores = ResearchStores(DATA_DIR / "research.sqlite")
    return ResearchService(ResearchDependencies(
        stores=stores,
        domain=bundle.policy_factory(),
        campaign_store=bundle.campaign_store,
        report_store=bundle.report_store,
        runner_factory=bundle.campaign_runner_factory,
        comparator_factory=bundle.comparator_factory,
        dataset_verifier=bundle.dataset_verifier,
        campaign_root=bundle.campaign_root,
        adapter=bundle.adapter_factory(),
    ))
