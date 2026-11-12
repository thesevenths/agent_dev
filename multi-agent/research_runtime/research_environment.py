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


def ensure_domain_registered(domain_name: str) -> None:
    """Register a domain bundle if the process-wide registry lacks it.

    The unified ``run_research_task`` reads a domain's policy BEFORE it builds
    the ResearchService; in a fresh process that used to raise KeyError for
    SciNet because only the service builder registered it lazily. Keeping the
    registration explicit here makes every domain self-contained.
    """
    if domain_name == "scinet":
        _ensure_scinet_registered()
    elif domain_name in ("embodied", "quant"):
        if not default_registry.contains(domain_name):
            register_domain(build_local_bundle(domain_name))
    else:
        raise KeyError(f"unknown research domain {domain_name!r}")


def build_local_bundle(domain_name: str, *, data_dir: str | Path | None = None):
    """ARR 1.0 / E4: assemble a NON-SciNet bundle from the SAME generic parts.

    Both new domains reuse ``CampaignRunner``, ``CampaignComparator``,
    ``CandidateConfigBuilder``, ``ResearchStores`` and the frozen-data verifier.
    No generic module is forked.
    """
    from pathlib import Path
    if domain_name == "embodied":
        from .domains.embodied import build_embodied_domain_bundle
        return build_embodied_domain_bundle(Path(data_dir) if data_dir else None)
    if domain_name == "quant":
        from .domains.quant import build_quant_domain_bundle
        return build_quant_domain_bundle(Path(data_dir) if data_dir else None)
    raise KeyError(f"no local bundle builder for domain {domain_name!r}")


def register_all_domains(*, data_dir: str | Path | None = None,
                         include_scinet: bool = True) -> list[str]:
    """Register every available domain in the process-wide registry.

    ``include_scinet=False`` is used by CI / benchmark when no SciNet++ checkout
    is present; the two local domains still run the FULL generic pipeline.
    """
    names: list[str] = []
    if include_scinet:
        _ensure_scinet_registered()
        names.append("scinet")
    for name in ("embodied", "quant"):
        register_domain(build_local_bundle(name, data_dir=data_dir))
        names.append(name)
    return names


def build_default_research_service(domain_name: str = "scinet", *,
                                   llm_generate=None,
                                   db_path: str | Path | None = None,
                                   data_dir: str | Path | None = None,
                                   evidence_provider=None,
                                   domain_policy=None):
    """Build the generic ``ResearchService`` for a registered domain.

    ``domain_name`` selects a ``DomainBundle`` from the process-wide registry.
    The default ``"scinet"`` reproduces the pre-ARR assembly exactly; any other
    registered domain is assembled through the identical generic path.

    ARR 1.0 adds three OPTIONAL, additive injections (all default to the old
    behaviour): ``llm_generate`` for a Harness-controlled deterministic
    Planner, ``db_path`` for isolated per-run stores, ``data_dir`` for a
    per-run campaign root. ``ResearchService`` itself is NOT rewritten.
    """
    from pathlib import Path

    from .research_service import ResearchDependencies, ResearchService
    from .research_store import ResearchStores

    if domain_name == "scinet":
        _ensure_scinet_registered()
    elif not default_registry.contains(domain_name) or data_dir is not None:
        # A per-run data dir requires a FRESH bundle (campaign stores are bound
        # to their sqlite path at construction time).
        register_domain(build_local_bundle(domain_name, data_dir=data_dir))
    bundle = get_domain(domain_name)
    ensure_data_dir()
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    stores = ResearchStores(Path(db_path) if db_path is not None
                            else (root / "research.sqlite"))
    return ResearchService(ResearchDependencies(
        stores=stores,
        domain=domain_policy or bundle.policy_factory(),
        campaign_store=bundle.campaign_store,
        report_store=bundle.report_store,
        runner_factory=bundle.campaign_runner_factory,
        comparator_factory=bundle.comparator_factory,
        dataset_verifier=bundle.dataset_verifier,
        campaign_root=bundle.campaign_root,
        adapter=bundle.adapter_factory(),
        llm_generate=llm_generate,
        evidence_provider=evidence_provider,
    ))


def build_custom_research_service(*, bundle, domain_policy, stores,
                                  llm_generate=None, evidence_provider=None):
    """Assemble the SAME generic ResearchService from an explicit bundle.

    Used by Harness evolution to run a candidate in an isolated per-arm store
    while keeping the identical ResearchService code path (F3 fair comparison).
    """
    from .research_service import ResearchDependencies, ResearchService
    return ResearchService(ResearchDependencies(
        stores=stores,
        domain=domain_policy,
        campaign_store=bundle.campaign_store,
        report_store=bundle.report_store,
        runner_factory=bundle.campaign_runner_factory,
        comparator_factory=bundle.comparator_factory,
        dataset_verifier=bundle.dataset_verifier,
        campaign_root=bundle.campaign_root,
        adapter=bundle.adapter_factory(),
        llm_generate=llm_generate,
        evidence_provider=evidence_provider,
    ))
