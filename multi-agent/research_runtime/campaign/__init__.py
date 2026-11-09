"""P1-04 Campaign layer: execute an approved ExperimentPlan, resumably.

    ExperimentPlan (P1-01)  ->  CampaignRunSpec + CampaignPolicy
        ->  Scheduler (seed-outer serial order)
        ->  TrialRunner (train -> validation, deterministic job ids)
        ->  CampaignStore (SQLite state + append-only events)
        ->  Manifest (facts only; no comparison, no promotion)

The runner is engine-agnostic. Domain wiring (config whitelist, value rules,
split vocabulary, resource limits) is INJECTED via CampaignPolicy; concrete
SciNet++ wiring lives in ``campaign/scinet.py`` and nowhere else.
"""
from .exceptions import (
    CampaignCancelled,
    CampaignError,
    CampaignIntegrityError,
    CampaignPreflightError,
    CampaignStoreError,
    RecoverableCampaignError,
    TrialExecutionError,
)
from .manifest import build_campaign_result, write_manifest
from .policy import CampaignPolicy
from .schemas import (
    CAMPAIGN_TERMINAL_STATUSES,
    TRIAL_TERMINAL_STATUSES,
    CampaignRecord,
    CampaignRunSpec,
    TrialRecord,
)
from .campaign_store import CampaignStore

__all__ = [
    "CampaignRecord", "CampaignRunSpec", "TrialRecord",
    "CAMPAIGN_TERMINAL_STATUSES", "TRIAL_TERMINAL_STATUSES",
    "CampaignStore", "CampaignPolicy",
    "build_campaign_result", "write_manifest",
    "CampaignError", "CampaignPreflightError", "CampaignIntegrityError",
    "CampaignStoreError", "RecoverableCampaignError",
    "TrialExecutionError", "CampaignCancelled",
]
