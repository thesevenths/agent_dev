"""M4.3-02.1 P1-04 -- human blind-review closure bridge.

ENGINE-FREE. A TEST acts as two distinct human reviewers (the codebase's own
approval_cli docs explicitly permit "a test acting as the human operator").
No automated score is ever used to fill a human score. Tests prove:
  * two DISTINCT humans are required;
  * unblind is refused while any review is incomplete;
  * all review rows are preserved (append-only, no overwrite);
  * the bridge resolves objective quality ONLY after unblind;
  * the analysis reflects the resolved (non-missing) objective quality.
"""
from __future__ import annotations

import pytest

from research_runtime.m4_3 import ExperimentRegistry, FrozenProtocol
from research_runtime.m4_3.pilot_analysis import analyze_pilot, load_runs
from research_runtime.m4_3.review_bridge import (
    resolve_human_quality_overlay, review_quality_audit,
)
from research_runtime.m4_3.review_queue import (
    ReviewQueue, ReviewQueueViolation,
)
from research_runtime.research_store.db import ResearchDatabase

from research_runtime.m4_3.blind_review import RUBRIC_DIMENSIONS

RUBRIC = {dim: (4 if i % 2 == 0 else 3)
          for i, dim in enumerate(RUBRIC_DIMENSIONS)}


def _db(tmp_path):
    return ResearchDatabase(tmp_path / "r.sqlite")


def _queue(db):
    return ReviewQueue(db=db, required_human_reviews=2)


def _enqueue_three(db):
    q = _queue(db)
    ids = []
    for arm in ("A", "B", "C"):
        item = q.enqueue(experiment_id="e1",
                         objective={"objective_id": f"obj_{arm}",
                                    "description": f"objective for {arm}",
                                    "success_criteria": {}},
                         arm=arm, block_id="b1", snapshot_id="s1")
        ids.append((arm, item.review_id))
    return q, ids


# --------------------------------------------------------------- requirements
def test_two_distinct_humans_required_no_auto_fill(tmp_path):
    q, ids = _enqueue_three(_db(tmp_path))
    rid = ids[0][1]
    q.submit(review_id=rid, reviewer_id="alice", scores=RUBRIC,
             reason="solid evidence")
    # same reviewer again is refused (append-only, no overwrite)
    with pytest.raises(ReviewQueueViolation):
        q.submit(review_id=rid, reviewer_id="alice", scores=RUBRIC,
                 reason="duplicate")
    # an automated score does NOT count toward the human requirement
    q.submit(review_id=rid, reviewer_id="scorer", scores=RUBRIC,
             reason="auto", review_kind="automated")
    c = q.completeness(rid)
    assert c["complete"] is False
    assert c["n_distinct_human_reviewers"] == 1
    assert c["n_automated_scores"] == 1
    # second DISTINCT human completes it
    q.submit(review_id=rid, reviewer_id="bob", scores=RUBRIC,
             reason="independent check")
    c = q.completeness(rid)
    assert c["complete"] is True
    assert c["n_distinct_human_reviewers"] == 2
    assert c["n_human_reviews"] == 2


def test_unblind_refused_while_any_review_incomplete(tmp_path):
    q, ids = _enqueue_three(_db(tmp_path))
    # fully complete only the first
    for arm, rid in ids:
        if arm == "A":
            q.submit(review_id=rid, reviewer_id="alice", scores=RUBRIC,
                     reason="r")
            q.submit(review_id=rid, reviewer_id="bob", scores=RUBRIC,
                     reason="r")
    with pytest.raises(ReviewQueueViolation):
        q.unblind("e1")
    for arm, rid in ids:
        if arm != "A":
            q.submit(review_id=rid, reviewer_id="alice", scores=RUBRIC,
                     reason="r")
            q.submit(review_id=rid, reviewer_id="bob", scores=RUBRIC,
                     reason="r")
    out = q.unblind("e1")
    assert out["n_unblinded"] == 3


def test_bridge_refuses_before_unblind_and_resolves_after(tmp_path):
    q, ids = _enqueue_three(_db(tmp_path))
    for arm, rid in ids:
        q.submit(review_id=rid, reviewer_id="alice", scores=RUBRIC, reason="r")
        q.submit(review_id=rid, reviewer_id="bob", scores=RUBRIC, reason="r")
    # complete but NOT unblinded -> no overlay
    resolved = resolve_human_quality_overlay(queue=q, experiment_id="e1")
    assert resolved["overlay"] == {}
    assert resolved["unblinded"] is False
    q.unblind("e1")
    resolved = resolve_human_quality_overlay(queue=q, experiment_id="e1")
    assert resolved["unblinded"] is True
    # three arms -> run ids b1_A / b1_B / b1_C via sealed mapping
    assert set(resolved["overlay"]) == {"b1_A", "b1_B", "b1_C"}
    qa = resolved["overlay"]["b1_A"]
    assert qa["n_reviews"] == 2
    assert qa["reviewer_ids"] == ["alice", "bob"]
    assert qa["total"] == pytest.approx(
        sum(RUBRIC.values()) / len(RUBRIC_DIMENSIONS), abs=1e-6)
    assert qa["review_kind"] == "human_blind"


def test_audit_reports_completion_state(tmp_path):
    q, ids = _enqueue_three(_db(tmp_path))
    audit = review_quality_audit(queue=q, experiment_id="e1")
    assert audit["n_queued"] == 3 and audit["n_complete"] == 0
    assert audit["all_complete"] is False


def test_all_review_rows_preserved_no_overwrite(tmp_path):
    q, ids = _enqueue_three(_db(tmp_path))
    rid = ids[0][1]
    q.submit(review_id=rid, reviewer_id="alice", scores=RUBRIC, reason="first")
    q.submit(review_id=rid, reviewer_id="bob", scores=RUBRIC, reason="second")
    rows = q.human_reviews(rid)
    assert len(rows) == 2
    assert {r["reviewer_id"] for r in rows} == {"alice", "bob"}
    assert all(r["reason"] for r in rows)


def test_closed_reviews_make_objective_quality_non_missing(tmp_path):
    """End-to-end: a real (test-actor) human closure makes quality observed."""
    db = _db(tmp_path)
    proto = FrozenProtocol(snapshot_ids=("s1",), sample_size=0, seed=42,
                           model_version="m", prompt_version="p",
                           ranker_version="r", paired_seeds=(42, 43),
                           exclusion_rules=("none",),
                           statistical_method="paired-block",
                           stopping_rule="all blocks")
    reg = ExperimentRegistry(protocol=proto, db=db)
    q = _queue(db)
    expected_total = sum(RUBRIC.values()) / len(RUBRIC_DIMENSIONS)
    bid = "b1"
    reg.create_block(block_id=bid, snapshot_id="s1", seed=41)
    for arm in ("A", "B", "C"):
        rid = f"b1_{arm}"
        reg.register_run(block_id=bid, arm=arm)
        item = q.enqueue(experiment_id="e1",
                         objective={"objective_id": f"obj_{arm}",
                                    "description": "o", "success_criteria": {}},
                         arm=arm, block_id=bid, snapshot_id="s1")
        reg.finalize(rid, outcome="negative_finding", review_id=item.review_id,
                     campaign_id=f"c_{rid}", plan_id=f"p_{rid}",
                     approval_id=f"a_{rid}", report_id=f"r_{rid}",
                     experience_id=f"e_{rid}", gate_steps_passed=12,
                     external_approval=True, compute_units=1.0,
                     wall_time_s=10.0, paired_seeds=[42, 43],
                     gate_evidence={"gate_passed": True})
        q.submit(review_id=item.review_id, reviewer_id="alice", scores=RUBRIC,
                 reason="r")
        q.submit(review_id=item.review_id, reviewer_id="bob", scores=RUBRIC,
                 reason="r")
    q.unblind("e1")
    resolved = resolve_human_quality_overlay(queue=q, experiment_id="e1")
    assert resolved["n_runs_resolved"] == 3
    a = analyze_pilot(db=db, out_dir=tmp_path, write=False,
                      human_quality_overlay=resolved["overlay"],
                      review_closure={"status": "closed"})
    qobs = a["review_closure"]
    assert qobs["status"] == "closed"
    # objective quality now OBSERVED, not missing
    assert a["objective_quality"]["review_status"] == "complete"
    assert a["objective_quality"]["n_human_blind_reviews"] == 6
