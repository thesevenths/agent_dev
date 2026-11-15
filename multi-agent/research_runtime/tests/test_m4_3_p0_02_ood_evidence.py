"""M4.3-00 / P0-02 -- OOD candidate evidence semantics (engine-free).

Bug being fixed (M4-E2E-01.1 legacy):
``_from_ood_gaps`` treated the MERE PRESENCE of ``ood_dataset_version`` as
proof of an ID/OOD generalisation gap and emitted a "Close the gap" conclusion
with only a dataset-version ref. There was no comparable ID/OOD metric record,
no metric/split/version identity check, and no frozen-split exclusion.

Fix: a real, comparable ID/OOD metric record must be present in the Evidence
View and pass ALL FOUR identity checks (domain / dataset version / metric
identity / evaluation split, on a NON-frozen split, traced to real evidence)
before a gap claim is generated. With only a version, an EXPLORATORY candidate
is emitted and explicitly marked ``evidence_insufficient``. No historical
verdict is recomputed and no statistical/frozen-split rule is changed.
"""
from __future__ import annotations

from research_runtime.discovery_context_provider import DiscoverySnapshot
from research_runtime.objective_candidate_generator import (
    ID_OOD_METRICS_KEY, ObjectiveCandidateGenerator,
    comparable_id_ood_records, make_id_ood_metric_ref,
)
from research_runtime.objective_critic import ObjectiveCritic, _resolve_ref
from research_runtime.objective_candidate_generator import ObjectiveCandidate

DOMAIN = "scinet"
OOD_VERSION = "ood_v2"
ID_VERSION = "ds_frozen_v1"
METRIC = "mse@1"
SPLIT = "validation"
EVIDENCE_ID = "sha256:real_metrics"


def _record(**overrides) -> dict:
    rec = {
        "domain": DOMAIN,
        "id_dataset_version": ID_VERSION,
        "ood_dataset_version": OOD_VERSION,
        "id_metric": METRIC,
        "ood_metric": METRIC,
        "id_evaluation_split": SPLIT,
        "ood_evaluation_split": SPLIT,
        "id_value": 0.10,
        "ood_value": 0.31,
        "evidence_ids": [EVIDENCE_ID],
    }
    rec.update(overrides)
    return rec


def _snapshot(*, records=None, ood_version=OOD_VERSION,
              extra_reserved=None) -> DiscoverySnapshot:
    view = {
        "schema_version": "m4.0-evidence-view-v1",
        "run_id": "r_p002",
        "n_evidence": 1,
        "evidence": [{"evidence_id": EVIDENCE_ID, "verdicts": [],
                      "sources": []}],
        "provenance": {"domains": {DOMAIN: {"ood_dataset_version": ood_version}}},
        "unresolved": [],
    }
    if records is not None:
        view[ID_OOD_METRICS_KEY] = records
    if extra_reserved is not None:
        view["reserved_final_splits"] = extra_reserved
    return DiscoverySnapshot(
        run_id="r_p002",
        schema_version="m4.1-00-discovery-snapshot-v1",
        generated_at="2026-01-01T00:00:00",
        evidence_view=view,
        objectives=[],
        links={"run_stats": {}, "by_objective": {}},
        sessions=[], sessions_by_objective={},
        experience_available=False,
        failures={"session_failure_signals": [],
                  "evidence_unresolved": [], "structured_diagnoses": []},
        provenance={"read_only": True, "run_id": "r_p002"},
        unresolved_gaps=[],
    )


def _gap_candidate(ref: str) -> ObjectiveCandidate:
    return ObjectiveCandidate(
        candidate_id="cand_gap", domain=DOMAIN,
        description="investigate measured gap",
        evidence_refs=[ref], source_objective_ids=[], hypothesis_hint="h",
        proposed_intervention={"axis": "generalisation_robustness"},
        success_criteria={"criterion": "reduce gap", "measurable": True,
                          "evidence_insufficient": False},
        budget={"max_campaign_rounds": 1}, provenance={},
        generation_source="id_ood_gap")


# --------------------------------------------------- spec: comparable metrics
def test_real_comparable_metrics_allow_gap_candidate():
    """A real, fully-identity-checked ID/OOD record permits a gap claim."""
    rec = _record()
    snap = _snapshot(records=[rec])
    assert len(comparable_id_ood_records(snap, DOMAIN)) == 1

    ref = make_id_ood_metric_ref(rec)
    assert _resolve_ref(ref, snap) is True

    gen = ObjectiveCandidateGenerator(domain=DOMAIN)
    gaps = [c for c in gen.generate(snap)
            if c.generation_source == "id_ood_gap"]
    assert len(gaps) == 1, "one comparable record -> exactly one gap claim"
    assert ref in gaps[0].evidence_refs
    assert gaps[0].success_criteria.get("evidence_insufficient") is False
    passed, errors = ObjectiveCritic().critique(gaps[0], snapshot=snap)
    assert passed is True, errors


def test_only_ood_version_produces_no_gap_claim():
    """Only a dataset version -> NO 'close the gap' conclusion."""
    snap = _snapshot(records=None)     # ood_dataset_version present, no metrics
    assert comparable_id_ood_records(snap, DOMAIN) == []

    gen = ObjectiveCandidateGenerator(domain=DOMAIN)
    cands = gen.generate(snap)
    assert not [c for c in cands if c.generation_source == "id_ood_gap"], \
        "a version alone must never yield a gap claim"
    # An exploratory candidate IS allowed, and must be marked insufficient.
    exploration = [c for c in cands
                   if c.generation_source == "id_ood_exploration"]
    assert len(exploration) == 1
    assert exploration[0].success_criteria.get("evidence_insufficient") is True
    assert exploration[0].provenance.get("evidence_insufficient") is True
    assert "Investigate OOD generalization" in exploration[0].description
    # ...and it still passes the Critic (it does NOT assert a gap conclusion).
    passed, errors = ObjectiveCritic().critique(exploration[0], snapshot=snap)
    assert passed is True, errors


def test_unjustified_gap_claim_is_rejected_by_critic():
    """Critic defense-in-depth: an id_ood_gap claim with only a version ref."""
    snap = _snapshot(records=None)
    bad = _gap_candidate(f"domain:{DOMAIN}:ood_dataset_version:{OOD_VERSION}")
    passed, errors = ObjectiveCritic().critique(bad, snapshot=snap)
    assert passed is False
    assert any("comparable" in e.lower() or "gap evidence" in e.lower()
               for e in errors), errors


# --------------------------------------------------- spec: version mismatch
def test_dataset_version_mismatch_is_rejected():
    """OOD version not matching the declared domain version -> not comparable."""
    rec = _record(ood_dataset_version="ood_fabricated_v9")
    snap = _snapshot(records=[rec])
    assert comparable_id_ood_records(snap, DOMAIN) == []
    assert _resolve_ref(make_id_ood_metric_ref(rec), snap) is False
    # No gap claim; only exploration remains.
    gen = ObjectiveCandidateGenerator(domain=DOMAIN)
    assert not [c for c in gen.generate(snap)
                if c.generation_source == "id_ood_gap"]


def test_identical_id_and_ood_versions_are_not_comparable():
    rec = _record(id_dataset_version=OOD_VERSION)
    snap = _snapshot(records=[rec])
    assert comparable_id_ood_records(snap, DOMAIN) == []


# --------------------------------------------------- spec: metric mismatch
def test_metric_identity_mismatch_is_rejected():
    """ID and OOD measured on DIFFERENT metrics -> not comparable."""
    rec = _record(ood_metric="cosine@1")
    snap = _snapshot(records=[rec])
    assert comparable_id_ood_records(snap, DOMAIN) == []
    assert _resolve_ref(make_id_ood_metric_ref(rec), snap) is False
    gen = ObjectiveCandidateGenerator(domain=DOMAIN)
    assert not [c for c in gen.generate(snap)
                if c.generation_source == "id_ood_gap"]


# --------------------------------------------------- spec: split identity + frozen
def test_evaluation_split_mismatch_is_rejected():
    rec = _record(ood_evaluation_split="validation_v2")
    snap = _snapshot(records=[rec])
    assert comparable_id_ood_records(snap, DOMAIN) == []


def test_frozen_final_split_is_never_selection_evidence():
    """The reserved frozen/final split must not ground an objective gap claim."""
    rec = _record(id_evaluation_split="test", ood_evaluation_split="test")
    snap = _snapshot(records=[rec])
    assert comparable_id_ood_records(snap, DOMAIN) == []
    assert _resolve_ref(make_id_ood_metric_ref(rec), snap) is False
    # No gap claim is generated from a frozen-split record.
    gen = ObjectiveCandidateGenerator(domain=DOMAIN)
    assert not [c for c in gen.generate(snap)
                if c.generation_source == "id_ood_gap"]

    # Critic defense-in-depth: a hand-crafted gap claim citing a frozen-split
    # metric ref is rejected explicitly.
    bad = _gap_candidate(make_id_ood_metric_ref(rec))
    passed, errors = ObjectiveCritic().critique(bad, snapshot=snap)
    assert passed is False
    assert any("frozen" in e.lower() and "split" in e.lower()
               for e in errors), errors


def test_view_declared_reserved_split_is_honored():
    """A domain may declare its OWN frozen split names (generic, cross-project)."""
    rec = _record(id_evaluation_split="final_holdout",
                  ood_evaluation_split="final_holdout")
    snap = _snapshot(records=[rec], extra_reserved=["final_holdout"])
    assert comparable_id_ood_records(snap, DOMAIN) == []


# --------------------------------------------------- spec: grounding + values
def test_untraceable_evidence_is_rejected():
    rec = _record(evidence_ids=["sha256:does_not_exist"])
    snap = _snapshot(records=[rec])
    assert comparable_id_ood_records(snap, DOMAIN) == []


def test_non_finite_values_are_rejected():
    for bad_value in (None, "0.3", float("nan"), float("inf"), True):
        rec = _record(ood_value=bad_value)
        snap = _snapshot(records=[rec])
        assert comparable_id_ood_records(snap, DOMAIN) == [], bad_value


def test_domain_mismatch_is_rejected():
    rec = _record(domain="some_other_domain")
    snap = _snapshot(records=[rec])
    assert comparable_id_ood_records(snap, DOMAIN) == []
