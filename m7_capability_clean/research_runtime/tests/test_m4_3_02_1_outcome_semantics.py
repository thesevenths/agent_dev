"""M4.3-02.1 P0-02 -- outcome semantics, fail-closed and distinct terminal states.

ENGINE-FREE: exercises the pure classification used by the pilot runner plus
the registry vocabulary, so every director state / report outcome is covered.
"""
from __future__ import annotations

import pytest

from research_runtime.m4_3.experiment_registry import (
    DIRECTOR_STATE_OUTCOME, RUN_OUTCOMES,
)
from research_runtime.m4_3.pilot_runner import classify_director_outcome


# ------------------------------------------------------- report outcomes (COMPLETED)
@pytest.mark.parametrize(
    "report_outcome,campaign_ok,sci_ok,expected,expect_reason",
    [
        ("accepted_for_confirmation", True, True, "valid_finding", False),
        ("rejected", True, True, "negative_finding", False),
        ("rejected", False, True, "inconclusive", True),   # no real campaign
        ("rejected", True, False, "inconclusive", True),   # no valid conclusion
        ("rejected", False, False, "inconclusive", True),
        ("inconclusive", True, True, "inconclusive", False),
        ("inconclusive", False, False, "inconclusive", False),  # NEVER negative
        ("not_evaluated", True, True, "inconclusive", False),
        ("not_evaluated", False, False, "inconclusive", False),
        ("totally_unknown", True, True, "invalid", True),  # fail-closed
        ("", True, True, "invalid", True),                 # missing -> invalid
    ],
)
def test_completed_report_outcome_mapping(report_outcome, campaign_ok, sci_ok,
                                          expected, expect_reason):
    outcome, reason = classify_director_outcome(
        director_state="COMPLETED", report_outcome=report_outcome,
        campaign_succeeded=campaign_ok, scientific_outcome_valid=sci_ok)
    assert outcome == expected
    assert (bool(reason) is expect_reason) or not expect_reason
    if expect_reason:
        assert reason


def test_inconclusive_and_not_evaluated_are_never_negative_finding():
    for ro in ("inconclusive", "not_evaluated"):
        outcome, _ = classify_director_outcome(
            director_state="COMPLETED", report_outcome=ro,
            campaign_succeeded=True, scientific_outcome_valid=True)
        assert outcome != "negative_finding"
        assert outcome == "inconclusive"


def test_rejected_needs_both_real_campaign_and_valid_conclusion():
    neg, _ = classify_director_outcome(
        director_state="COMPLETED", report_outcome="rejected",
        campaign_succeeded=True, scientific_outcome_valid=True)
    assert neg == "negative_finding"
    for c, s in ((False, True), (True, False), (False, False)):
        out, reason = classify_director_outcome(
            director_state="COMPLETED", report_outcome="rejected",
            campaign_succeeded=c, scientific_outcome_valid=s)
        assert out == "inconclusive"
        assert "campaign_succeeded" in reason and "scientific_outcome_valid" in reason


# --------------------------------------------------- director terminal states
@pytest.mark.parametrize(
    "director_state,expected",
    [
        ("FAILED", "failed"),
        ("INTERRUPTED", "interrupted"),
        ("STOPPED", "stopped"),
        ("INVALID", "invalid"),
        ("REJECTED", "invalid"),
    ],
)
def test_terminal_director_states_are_distinct(director_state, expected):
    outcome, reason = classify_director_outcome(director_state=director_state)
    assert outcome == expected
    assert director_state in reason


def test_terminal_states_are_not_collapsed_into_interrupted():
    outcomes = {classify_director_outcome(director_state=s)[0]
                for s in ("FAILED", "INTERRUPTED", "STOPPED", "INVALID")}
    assert outcomes == {"failed", "interrupted", "stopped", "invalid"}


@pytest.mark.parametrize("unknown", ["", "WEIRD", "COMPLETING", "PAUSED",
                                     "something not in the state machine"])
def test_unknown_director_state_fails_closed_with_raw_value(unknown):
    outcome, reason = classify_director_outcome(director_state=unknown)
    assert outcome == "invalid"
    assert repr(unknown).strip("'\"") in reason or str(unknown) in reason


# ------------------------------------------------------------- vocabulary
def test_distinct_outcomes_are_in_the_frozen_vocabulary():
    for outcome in ("failed", "interrupted", "stopped", "invalid",
                    "valid_finding", "negative_finding", "inconclusive"):
        assert outcome in RUN_OUTCOMES
    assert set(DIRECTOR_STATE_OUTCOME.values()) <= set(RUN_OUTCOMES)
