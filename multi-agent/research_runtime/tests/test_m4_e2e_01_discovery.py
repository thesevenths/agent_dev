"""M4-E2E-01A -- Objective Discovery engine-free contract tests.

These run WITHOUT any scientific engine, LLM or model. They verify the four
spec requirements deterministically (spec M4-E2E-01A.1-A.7):

  A. At least one traceable candidate is produced from a real-shaped snapshot and
     persisted as a Proposed Objective (via the existing ObjectiveStore).
  B. The Critic rejects evidence-less / unverifiable / duplicate candidates.
  C. Same Snapshot + same config => identical candidates AND identical ranking.
  D. The Proposed Objective is persisted (idempotent) and is NOT a verdict.

No ACCEPT / REJECT / INCONCLUSIVE is produced anywhere in the pipeline.
"""
from __future__ import annotations

from pathlib import Path

from research_runtime.discovery_context_provider import DiscoverySnapshot
from research_runtime.objective_candidate_generator import (
    ObjectiveCandidate, ObjectiveCandidateGenerator,
)
from research_runtime.objective_critic import ObjectiveCritic
from research_runtime.objective_discovery import ObjectiveDiscovery
from research_runtime.objective_ranker import RANKER_VERSION, ObjectiveRanker
from research_runtime.research_store import ResearchStores

WS = Path(__file__).resolve().parents[2]      # multi-agent/


# --------------------------------------------------------------------- helpers
def _make_snapshot(run_id: str = "r1") -> DiscoverySnapshot:
    """A real-shaped snapshot with three resolvable candidate sources."""
    evidence_view = {
        "schema_version": "m4.0-evidence-view-v1",
        "run_id": run_id,
        "n_evidence": 1,
        "evidence": [
            {"evidence_id": "sha256:neg1",
             "verdicts": ["rejected"], "sources": []},
        ],
        "provenance": {
            "domains": {"scinet": {"ood_dataset_version": "ood_v2"}},
        },
        "unresolved": [],
    }
    failures = {
        "session_failure_signals": [
            {"objective_id": "obj_x", "session_id": "s_x",
             "session_index": 0, "event_type": "fail", "created_at": ""},
        ],
        "evidence_unresolved": [],
        "structured_diagnoses": [],
    }
    sessions = [{
        "objective_id": "obj_x", "session_id": "s_x",
        "session_available": True, "session_index": 0,
        "events": [], "reports": [], "experience": [],
        "experience_available": False,
    }]
    return DiscoverySnapshot(
        run_id=run_id,
        schema_version="m4.1-00-discovery-snapshot-v1",
        generated_at="2026-01-01T00:00:00",
        evidence_view=evidence_view,
        objectives=[],                 # no existing objective => no duplicate hits
        links={"run_stats": {}, "by_objective": {}},
        sessions=sessions,
        sessions_by_objective={"obj_x": sessions},
        experience_available=False,
        failures=failures,
        provenance={"read_only": True, "run_id": run_id},
        unresolved_gaps=[],
    )


def _discoverer() -> ObjectiveDiscovery:
    return ObjectiveDiscovery(
        generator=ObjectiveCandidateGenerator(domain="scinet"),
        critic=ObjectiveCritic(),
        ranker=ObjectiveRanker(),
        domain="scinet",
    )


# --------------------------------------------------------------------- A: produce + persist
def test_discovery_produces_traceable_proposed_objective(tmp_path):
    snap = _make_snapshot()
    store = ResearchStores(tmp_path / "research.sqlite")
    d = _discoverer()

    result = d.discover(snap, objective_store=store.objectives, domain="scinet")

    # At least one candidate, and it was persisted as a Proposed Objective.
    assert result.n_candidates >= 1, "must produce >=1 candidate"
    assert result.n_approved >= 1, "critic must approve >=1 candidate"
    assert result.proposed_objective_ids, "must persist >=1 Proposed Objective"

    # The persisted objective really exists and is 'proposed' (NOT a verdict).
    oid = result.proposed_objective_ids[0]
    rec = store.objectives.get(oid)
    assert rec is not None
    assert rec.status == "proposed"
    assert len(rec.content_sha256) == 64   # plain sha256 hex, no prefix
    # No verdict vocabulary leaked into the persisted objective.
    assert "ACCEPT" not in rec.status.upper()
    assert "REJECT" not in rec.status.upper()
    assert "INCONCLUSIVE" not in rec.status.upper()


def test_generated_candidates_are_not_verdicts():
    """Spec: generator must NOT emit ACCEPT/REJECT/INCONCLUSIVE."""
    gen = ObjectiveCandidateGenerator(domain="scinet")
    cands = gen.generate(_make_snapshot())
    assert cands, "generator produced no candidates"
    for c in cands:
        assert c.critic_verdict is None or "verdict" not in str(
            c.critic_verdict).upper().split()
        # The candidate carries the required provenance fields, not a verdict.
        assert c.evidence_refs
        assert c.hypothesis_hint
        assert c.proposed_intervention
        assert c.success_criteria.get("criterion")
        assert c.budget


# --------------------------------------------------------------------- B: critic rejects bad candidates
def test_critic_rejects_evidence_less_candidate():
    critic = ObjectiveCritic()
    snap = _make_snapshot()
    bad = ObjectiveCandidate(
        candidate_id="cand_test", domain="scinet",
        description="no grounding",
        evidence_refs=["nope:xyz"],        # unresolvable prefix
        source_objective_ids=[], hypothesis_hint="h",
        proposed_intervention={"axis": "x"},
        success_criteria={"criterion": "c", "measurable": True},
        budget={"max_campaign_rounds": 1}, provenance={},
        generation_source="test")
    passed, errors = critic.critique(bad, snapshot=snap)
    assert passed is False
    assert any("resolv" in e.lower() for e in errors)


def test_critic_rejects_unverifiable_candidate():
    critic = ObjectiveCritic()
    snap = _make_snapshot()
    bad = ObjectiveCandidate(
        candidate_id="cand_test2", domain="scinet",
        description="unverifiable",
        evidence_refs=["sha256:neg1"],     # resolvable
        source_objective_ids=[], hypothesis_hint="h",
        proposed_intervention={"axis": "x"},
        success_criteria={"criterion": "", "measurable": False},  # not measurable
        budget={}, provenance={}, generation_source="test")
    passed, errors = critic.critique(bad, snapshot=snap)
    assert passed is False
    assert any("measurable" in e.lower() for e in errors)


def test_critic_rejects_duplicate_of_existing_objective():
    critic = ObjectiveCritic()
    snap = _make_snapshot()
    # Inject an existing objective with the SAME description.
    snap.objectives = [{"objective_id": "obj_dup", "domain": "scinet",
                        "description": "duplicate me"}]
    dup = ObjectiveCandidate(
        candidate_id="cand_dup", domain="scinet",
        description="duplicate me",
        evidence_refs=["sha256:neg1"],
        source_objective_ids=[], hypothesis_hint="h",
        proposed_intervention={"axis": "x"},
        success_criteria={"criterion": "c", "measurable": True},
        budget={}, provenance={}, generation_source="test")
    passed, errors = critic.critique(dup, snapshot=snap)
    assert passed is False
    assert any("duplicate" in e.lower() for e in errors)


def test_critic_accepts_well_formed_candidate():
    critic = ObjectiveCritic()
    snap = _make_snapshot()
    good = ObjectiveCandidate(
        candidate_id="cand_good", domain="scinet",
        description="well formed direction",
        evidence_refs=["sha256:neg1"],
        source_objective_ids=[], hypothesis_hint="h",
        proposed_intervention={"axis": "x"},
        success_criteria={"criterion": "improve metric", "measurable": True},
        budget={"max_campaign_rounds": 1}, provenance={},
        generation_source="test")
    passed, errors = critic.critique(good, snapshot=snap)
    assert passed is True, errors


# --------------------------------------------------------------------- C: determinism
def test_discovery_is_deterministic(tmp_path):
    snap = _make_snapshot()
    store = ResearchStores(tmp_path / "research.sqlite")

    r1 = _discoverer().discover(snap, objective_store=store.objectives,
                                domain="scinet")
    # Fresh store so idempotency is not what we test here.
    store2 = ResearchStores(tmp_path / "research2.sqlite")
    r2 = _discoverer().discover(snap, objective_store=store2.objectives,
                                domain="scinet")

    # Identical candidate ids AND identical ranking order.
    ids1 = [c["candidate_id"] for c in r1.ranked]
    ids2 = [c["candidate_id"] for c in r2.ranked]
    assert ids1 == ids2, f"ranking order changed: {ids1} vs {ids2}"
    assert r1.ranker_version == r2.ranker_version == RANKER_VERSION
    # Scores must be bit-identical.
    scores1 = [(c["candidate_id"], c["rank_score"]) for c in r1.ranked]
    scores2 = [(c["candidate_id"], c["rank_score"]) for c in r2.ranked]
    assert scores1 == scores2


# --------------------------------------------------------------------- D: idempotent persistence
def test_discovery_persistence_is_idempotent(tmp_path):
    snap = _make_snapshot()
    store = ResearchStores(tmp_path / "research.sqlite")

    r1 = _discoverer().discover(snap, objective_store=store.objectives,
                                domain="scinet")
    r2 = _discoverer().discover(snap, objective_store=store.objectives,
                                domain="scinet")

    # Same objective ids produced both times.
    assert r1.proposed_objective_ids == r2.proposed_objective_ids
    # The store holds exactly one record per objective (no duplicates after
    # re-running Discovery).
    stored = store.objectives.all(domain="scinet", status="proposed")
    assert len(stored) == len(r1.proposed_objective_ids)
    assert sorted(o.objective_id for o in stored) == sorted(
        r1.proposed_objective_ids)


# --------------------------------------------------------------------- D: second-round Discovery reads Experience
def _make_experience_snapshot(run_id: str, *, with_experience: bool) -> DiscoverySnapshot:
    """A snapshot whose ONLY candidate source is recorded Experience.

    With experience -> an experience-derived candidate should appear. Without it
    -> no candidate at all. This isolates the Experience signal so the test proves
    new Experience changes Discovery output (spec M4-E2E-01D).
    """
    session = {
        "objective_id": "obj_round1", "session_id": "s_round1",
        "session_available": True, "session_index": 0,
        "events": [], "reports": [],
        "experience": (
            [{"kind": "campaign_outcome", "verdict": "inconclusive",
              "note": "intervention did not meet guardrail"}]
            if with_experience else []),
        "experience_available": with_experience,
    }
    return DiscoverySnapshot(
        run_id=run_id,
        schema_version="m4.1-00-discovery-snapshot-v1",
        generated_at="2026-01-01T00:00:00",
        evidence_view={"schema_version": "m4.0-evidence-view-v1",
                       "run_id": run_id, "n_evidence": 0, "evidence": [],
                       "provenance": {"domains": {}}, "unresolved": []},
        objectives=[],
        links={"run_stats": {}, "by_objective": {}},
        sessions=[session],
        sessions_by_objective={"obj_round1": [session]},
        experience_available=with_experience,
        failures={"session_failure_signals": [],
                  "evidence_unresolved": [], "structured_diagnoses": []},
        provenance={"read_only": True, "run_id": run_id},
        unresolved_gaps=[],
    )


def test_experience_changes_discovery_output():
    """Spec M4-E2E-01D: round-2 Experience must affect Discovery output."""
    with_exp = _make_experience_snapshot("r_exp", with_experience=True)
    without = _make_experience_snapshot("r_noexp", with_experience=False)

    d = _discoverer()
    r_exp = d.discover(with_exp, objective_store=_noop_store(),
                       domain="scinet")
    r_no = d.discover(without, objective_store=_noop_store(),
                      domain="scinet")

    # Without Experience: zero candidates. With Experience: >=1 candidate, and it
    # is explicitly experience-derived (explainable source + resolvable ref).
    assert r_no.n_candidates == 0, "no Experience must yield no candidate"
    assert r_exp.n_candidates >= 1, "recorded Experience must yield a candidate"
    sources = {c["generation_source"] for c in r_exp.ranked}
    assert "historical_experience" in sources, sources
    refs = [r for c in r_exp.ranked for r in c["evidence_refs"]]
    assert any(r.startswith("experience:") for r in refs)
    # The Experience-derived candidate is NOT treated as a success: its
    # description is cautious ("do not repeat" the failed direction).
    assert "without repeating" in r_exp.ranked[0]["description"].lower() or \
        "do not repeat" in r_exp.ranked[0]["description"].lower()


def _noop_store():
    """In-memory ObjectiveStore so Discovery persistence is exercised but isolated.

    Uses a throwaway on-disk sqlite in a temp location to avoid cross-test bleed.
    """
    import tempfile
    from pathlib import Path
    path = Path(tempfile.mkdtemp()) / "noop.sqlite"
    return ResearchStores(path).objectives

