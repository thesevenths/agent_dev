"""M4.3-02 -- real HUMAN blind-review submission queue.

ENGINE-FREE. Proves the properties the spec demands of the human review step:
opaque review ids, sealed arm mapping, >= 2 distinct reviewers, append-only
submissions, automated scores kept separate, disagreement preserved, and no
unblinding before completion.
"""
from __future__ import annotations

import pytest

from research_runtime.m4_3.blind_review import (
    FORBIDDEN_BLIND_FIELDS, RUBRIC_DIMENSIONS, RUBRIC_SCALE,
)
from research_runtime.m4_3.review_queue import (
    ReviewQueue, ReviewQueueViolation,
)


def _queue(tmp_path) -> ReviewQueue:
    return ReviewQueue(db_path=str(tmp_path / "review.sqlite"))


def _objective(oid: str = "obj_x", arm: str = "C") -> dict:
    return {"objective_id": oid,
            "description": "Reduce multi-step latent prediction error.",
            "evidence_refs": ["sha256:abc123"],
            "success_criteria": {"metric": "latent_mse@1", "direction": "minimize"},
            "proposed_intervention": {"train.lr": 0.0008},
            "generation_source": "negative_verdict_evidence",
            "rank_score": 0.91, "ranker_version": "deterministic-v1",
            "arm": arm, "block_id": "block_1"}


def _scores(value: int = 3) -> dict[str, int]:
    return {d: value for d in RUBRIC_DIMENSIONS}


def test_queued_item_is_anonymous_and_mapping_is_sealed(tmp_path):
    q = _queue(tmp_path)
    item = q.enqueue(experiment_id="exp_1", objective=_objective(),
                     arm="C", block_id="block_1", snapshot_id="snap_1")
    shown = q.show(item.review_id)
    assert set(shown) == {"review_id", "description", "evidence_refs",
                          "success_criteria", "proposed_intervention"}
    assert not (FORBIDDEN_BLIND_FIELDS & set(shown))
    assert "obj_x" not in shown["review_id"]
    with pytest.raises(ReviewQueueViolation):
        q.arm_mapping(item.review_id)      # sealed before unblinding


def test_two_independent_reviewers_are_required_and_preserved(tmp_path):
    q = _queue(tmp_path)
    item = q.enqueue(experiment_id="exp_1", objective=_objective(), arm="A",
                     block_id="block_1", snapshot_id="snap_1")
    q.submit(review_id=item.review_id, reviewer_id="alice",
             scores=_scores(4), reason="strong evidence grounding")
    assert q.completeness(item.review_id)["complete"] is False
    q.submit(review_id=item.review_id, reviewer_id="bob",
             scores=_scores(1), reason="not falsifiable enough")
    c = q.completeness(item.review_id)
    assert c["complete"] is True
    assert c["n_distinct_human_reviewers"] == 2
    # Disagreement is preserved verbatim: no averaging, no discard.
    totals = sorted(r["total"] for r in c["scores"])
    assert totals == [1.0, 4.0]
    assert c["scores"][0]["reason"] != c["scores"][1]["reason"]


def test_a_reviewer_cannot_overwrite_their_own_review(tmp_path):
    q = _queue(tmp_path)
    item = q.enqueue(experiment_id="exp_1", objective=_objective(), arm="B",
                     block_id="block_1", snapshot_id="snap_1")
    q.submit(review_id=item.review_id, reviewer_id="alice",
             scores=_scores(3), reason="ok")
    with pytest.raises(ReviewQueueViolation) as exc:
        q.submit(review_id=item.review_id, reviewer_id="alice",
                 scores=_scores(0), reason="changed my mind")
    assert "cannot be overwritten" in str(exc.value)
    assert q.completeness(item.review_id)["n_human_reviews"] == 1


def test_automated_scores_never_count_as_human_review(tmp_path):
    q = _queue(tmp_path)
    item = q.enqueue(experiment_id="exp_1", objective=_objective(), arm="C",
                     block_id="block_1", snapshot_id="snap_1")
    q.submit(review_id=item.review_id, reviewer_id="alice",
             scores=_scores(3), reason="human 1")
    q.submit(review_id=item.review_id, reviewer_id="llm_scorer",
             scores=_scores(4), reason="automated heuristic",
             review_kind="automated")
    c = q.completeness(item.review_id)
    assert c["n_human_reviews"] == 1
    assert c["n_automated_scores"] == 1
    assert c["complete"] is False           # automated never substitutes
    with pytest.raises(ReviewQueueViolation):
        q.unblind("exp_1")


def test_automated_identity_cannot_submit_a_human_review(tmp_path):
    q = _queue(tmp_path)
    item = q.enqueue(experiment_id="exp_1", objective=_objective(), arm="C",
                     block_id="block_1", snapshot_id="snap_1")
    with pytest.raises(ReviewQueueViolation):
        q.submit(review_id=item.review_id, reviewer_id="auto_scorer_1",
                 scores=_scores(3), reason="pretending to be human")


def test_scores_are_validated_against_the_frozen_rubric(tmp_path):
    q = _queue(tmp_path)
    item = q.enqueue(experiment_id="exp_1", objective=_objective(), arm="A",
                     block_id="block_1", snapshot_id="snap_1")
    with pytest.raises(ReviewQueueViolation):
        q.submit(review_id=item.review_id, reviewer_id="alice",
                 scores={d: 9 for d in RUBRIC_DIMENSIONS}, reason="out of scale")
    with pytest.raises(ReviewQueueViolation):
        q.submit(review_id=item.review_id, reviewer_id="alice",
                 scores={"novelty": 3}, reason="incomplete")
    with pytest.raises(ReviewQueueViolation):
        q.submit(review_id=item.review_id, reviewer_id="alice",
                 scores=_scores(3), reason="   ")
    assert RUBRIC_SCALE == (0, 4)


def test_unblind_releases_the_mapping_only_after_completion(tmp_path):
    q = _queue(tmp_path)
    ids = []
    for i, arm in enumerate(("A", "B", "C")):
        item = q.enqueue(experiment_id="exp_1",
                         objective=_objective(f"obj_{i}", arm), arm=arm,
                         block_id="block_1", snapshot_id="snap_1")
        ids.append(item.review_id)
    q.submit(review_id=ids[0], reviewer_id="alice", scores=_scores(3),
             reason="r1")
    q.submit(review_id=ids[0], reviewer_id="bob", scores=_scores(2),
             reason="r2")
    with pytest.raises(ReviewQueueViolation) as exc:
        q.unblind("exp_1")
    assert "still missing" in str(exc.value)
    for rid in ids[1:]:
        q.submit(review_id=rid, reviewer_id="alice", scores=_scores(3),
                 reason="ok")
        q.submit(review_id=rid, reviewer_id="bob", scores=_scores(3),
                 reason="ok")
    out = q.unblind("exp_1")
    assert out["n_unblinded"] == 3
    mapping = q.arm_mapping(ids[0])
    assert mapping["arm"] == "A" and mapping["objective_id"] == "obj_0"
    status = q.queue_status("exp_1")
    assert status["n_pending"] == 0
    assert q.list_pending("exp_1") == []
    # No new review may be added after unblinding.
    with pytest.raises(ReviewQueueViolation):
        q.submit(review_id=ids[0], reviewer_id="carol", scores=_scores(3),
                 reason="late")


def test_duplicate_queue_entry_is_refused(tmp_path):
    q = _queue(tmp_path)
    q.enqueue(experiment_id="exp_1", objective=_objective(), arm="A",
              block_id="block_1", snapshot_id="snap_1", review_id="rev_fixed")
    with pytest.raises(Exception):
        q.enqueue(experiment_id="exp_1", objective=_objective(), arm="A",
                  block_id="block_1", snapshot_id="snap_1",
                  review_id="rev_fixed")


def test_export_keeps_mapping_sealed_until_unblinded(tmp_path):
    q = _queue(tmp_path)
    item = q.enqueue(experiment_id="exp_1", objective=_objective(), arm="C",
                     block_id="block_1", snapshot_id="snap_1")
    q.submit(review_id=item.review_id, reviewer_id="alice", scores=_scores(3),
             reason="ok")
    exported = q.export("exp_1")
    assert exported["mapping"] == "SEALED"
    assert all("objective_id" not in i["blinded"] for i in exported["items"])
