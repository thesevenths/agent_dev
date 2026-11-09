"""P1-04: error taxonomy for the campaign execution layer.

The distinction that matters is NOT "which exception class is prettiest" but
whether the campaign may safely continue, must pause, or must abort:

    CampaignPreflightError   nothing was executed; reject before the first job
    CampaignIntegrityError   the engine's own evidence contradicts the plan
                             (wrong split / dataset / run_id / checkpoint)
    CampaignStoreError       persistence failed -> never submit another job
    RecoverableCampaignError job may still be running / lookup undecidable
                             -> PAUSE, do not mark failed, do not resubmit
    TrialExecutionError      one trial failed; other paired trials continue
"""
from __future__ import annotations


class CampaignError(RuntimeError):
    """Base class for all campaign-layer failures."""


class CampaignPreflightError(CampaignError):
    """The plan/spec/policy is unsafe or out of budget. No job was submitted."""


class CampaignIntegrityError(CampaignError):
    """Recorded experiment evidence contradicts the approved plan."""


class CampaignStoreError(CampaignError):
    """A state/event write failed. Execution must stop before submitting work."""


class RecoverableCampaignError(CampaignError):
    """Transient: a wait timed out or a job lookup could not be decided.

    The worker may still be running. Marking the job failed (or submitting a
    second one) would either lose evidence or duplicate an experiment.
    """


class TrialExecutionError(CampaignError):
    """One arm/seed failed; its record is complete and the campaign continues."""


class CampaignCancelled(CampaignError):
    """The operator cancelled the campaign; no new jobs may be submitted."""
