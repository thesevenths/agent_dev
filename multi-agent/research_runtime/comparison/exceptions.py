"""P1-05: error taxonomy for the comparison layer.

The boundary that matters is the difference between EVIDENCE THAT IS MISSING
and EVIDENCE THAT CONTRADICTS ITSELF:

    ComparisonIntegrityError  recorded evidence contradicts the plan
                              (dataset version / split / run_id / config hash)
                              -> REFUSE TO COMPARE, never emit a verdict
    ComparisonInputError      malformed numeric input for a statistic
                              (NaN / Inf / bool / unequal lengths) -> raise
    ComparisonStoreError      immutable report persistence failed

A missing or failed trial is NEITHER of these: it is recorded as a
``MissingPair`` and caps the decision at ``inconclusive``. Never silently
drop it, and never turn it into a fabricated data point.
"""
from __future__ import annotations


class ComparisonError(RuntimeError):
    """Base class for all comparison-layer failures."""


class ComparisonIntegrityError(ComparisonError):
    """The campaign's own evidence is inconsistent; comparison is refused."""


class ComparisonInputError(ComparisonError):
    """A statistic received invalid or non-finite observations."""


class ComparisonStoreError(ComparisonError):
    """The immutable comparison report could not be written or verified."""
