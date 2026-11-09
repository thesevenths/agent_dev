"""Real dependency assembly for the P1-06 tools (lazy, cached by tools.py).

Nothing in the generic service constructs an engine object. This module is the
composition root: it wires the P1-03 config builder, the P1-04 CampaignRunner
factory, the P1-05 comparator factory and the P1-06 trusted stores together.
Tests inject their own ``ResearchDependencies`` and never import this module.
"""
from __future__ import annotations

from .settings import DATA_DIR, DEFAULT_SCINET_DIR, ensure_data_dir


def build_default_research_service():
    from .campaign.campaign_store import CampaignStore
    from .campaign.scinet import build_scinet_campaign_runner, scinet_campaign_policy
    from .candidate_config import CandidateConfigBuilder
    from .comparison.report_store import ComparisonReportStore
    from .comparison.scinet import build_scinet_comparator
    from .domains.scinet import scinet_research_policy
    from .research_service import ResearchDependencies, ResearchService
    from .research_store import ResearchStores

    ensure_data_dir()
    domain = scinet_research_policy()
    stores = ResearchStores(DATA_DIR / "research.sqlite")
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
        max_jobs=domain.max_jobs)

    def runner_factory():
        return build_scinet_campaign_runner(
            store=campaign_store, config_builder=config_builder,
            policy=campaign_policy)

    def comparator_factory():
        return build_scinet_comparator(
            campaign_store=campaign_store, report_store=report_store,
            project_dir=DEFAULT_SCINET_DIR)

    def dataset_verifier(baseline_config_path: str, dataset_version: str) -> dict:
        return config_builder.verify_dataset(baseline_config_path,
                                              dataset_version)

    return ResearchService(ResearchDependencies(
        stores=stores, domain=domain, campaign_store=campaign_store,
        report_store=report_store, runner_factory=runner_factory,
        comparator_factory=comparator_factory,
        dataset_verifier=dataset_verifier,
        campaign_root=str(DEFAULT_SCINET_DIR / domain.campaign_root),
    ))
