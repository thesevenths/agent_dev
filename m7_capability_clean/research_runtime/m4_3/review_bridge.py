"""M4.3-02.1 P1-04 -- human blind-review CLOSURE bridge.

The ReviewQueue stores append-only human reviews with a SEALED review_id -> arm
mapping. The analysis needs an objective-quality value on each run, but a
FINALIZED run is immutable (P0-03) -- so this module never rewrites a run. It
RESOLVES the human-blind quality from the queue at analysis time and returns an
in-memory overlay ``{run_id: quality_review_dict}``.

Hard rules (P1-04):
  * at least TWO DISTINCT human reviewers must have submitted (no self-review);
  * only ``human_blind`` rows count -- automated scores are NEVER used;
  * the mapping is readable ONLY after ``ReviewQueue.unblind()`` (which itself
    refuses while any item is incomplete);
  * every submitted review row is preserved (append-only); this module only
    READS them and aggregates a mean;
  * no review is fabricated here: if a human has not submitted, the value is
    MISSING and the overlay is empty for that run.
"""
from __future__ import annotations

from typing import Any

from .blind_review import (
    REVIEW_KIND_AUTOMATED, REVIEW_KIND_HUMAN_BLIND, RUBRIC_DIMENSIONS,
)
from .review_queue import ReviewQueue, ReviewQueueViolation


def review_quality_audit(*, queue: ReviewQueue, experiment_id: str
                         ) -> dict[str, Any]:
    """Read-only audit of the human-blind review completion state."""
    status = queue.queue_status(experiment_id)
    items: list[dict[str, Any]] = []
    # queue_status returns an 'items' list with per-review detail.
    for item in status.get("items", []):
        rid = item.get("review_id")
        c = queue.completeness(rid) if rid else {}
        items.append({
            "review_id": rid,
            "n_human_reviews": c.get("n_human_reviews", 0),
            "n_distinct_human_reviewers":
                c.get("n_distinct_human_reviewers", 0),
            "n_automated_scores": c.get("n_automated_scores", 0),
            "complete": c.get("complete", False),
            "unblinded": bool(item.get("unblinded_at")),
        })
    n_complete = sum(1 for i in items if i["complete"])
    return {
        "experiment_id": experiment_id,
        "n_queued": len(items),
        "n_complete": n_complete,
        "all_complete": bool(items) and n_complete == len(items),
        "items": items,
        "required_human_reviews": queue.required_human_reviews,
    }


def resolve_human_quality_overlay(*, queue: ReviewQueue, experiment_id: str
                                  ) -> dict[str, Any]:
    """Resolve append-only human reviews into ``{run_id: quality_review}``.

    Returns an AUDIT dict, not a bare mapping, so a caller can see exactly what
    was used and why anything is missing::

        {"overlay": {run_id: {...}}, "n_runs_resolved": int,
         "skipped": [{"review_id": ..., "reason": ...}],
         "unblinded": bool, "audit": {...}}
    """
    audit = review_quality_audit(queue=queue, experiment_id=experiment_id)
    overlay: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    # actual unblind state from the queue (NOT inferred from completeness)
    actually_unblinded = bool(audit["items"]) and all(
        i["unblinded"] for i in audit["items"])
    if not audit["all_complete"]:
        return {"overlay": overlay, "n_runs_resolved": 0,
                "skipped": [{"review_id": "ALL",
                             "reason": "not all reviews complete"}],
                "unblinded": actually_unblinded, "audit": audit}
    if not actually_unblinded:
        return {"overlay": overlay, "n_runs_resolved": 0,
                "skipped": [{"review_id": "ALL",
                             "reason": "reviews complete but not yet unblinded"}],
                "unblinded": False, "audit": audit}

    for item in audit["items"]:
        rid = item["review_id"]
        human = [r for r in queue.human_reviews(rid)
                 if r.get("review_kind") == REVIEW_KIND_HUMAN_BLIND]
        reviewers = {str(r.get("reviewer_id") or "") for r in human}
        reviewers.discard("")
        if len(reviewers) < queue.required_human_reviews:
            skipped.append({"review_id": rid,
                            "reason": "fewer than required distinct humans"})
            continue
        try:
            mapping = queue.arm_mapping(rid)   # fails closed if not unblinded
        except ReviewQueueViolation as exc:
            skipped.append({"review_id": rid,
                            "reason": f"mapping sealed: {exc}"})
            continue
        run_id = str(mapping.get("run_id") or "")
        block_id = str(mapping.get("block_id") or "")
        arm = str(mapping.get("arm") or "")
        if not run_id:
            # The seal stores arm/block/objective, not always run_id; derive it
            # deterministically from the pilot's ``{block}_{arm}`` convention.
            run_id = f"{block_id}_{arm}" if block_id and arm else ""
        if not run_id:
            skipped.append({"review_id": rid,
                            "reason": "sealed mapping has no run identity"})
            continue
        totals = [float(r.get("total") or 0.0) for r in human]
        mean_total = round(sum(totals) / len(totals), 6)
        scores: dict[str, float] = {}
        for dim in RUBRIC_DIMENSIONS:
            vals = [float(r["scores"][dim]) for r in human
                    if dim in (r.get("scores") or {})]
            if vals:
                scores[dim] = round(sum(vals) / len(vals), 6)
        overlay[run_id] = {
            "total": mean_total,
            "scores": scores,
            "review_kind": REVIEW_KIND_HUMAN_BLIND,
            "reviewer_ids": sorted(reviewers),
            "n_reviews": len(human),
            "source_review_id": rid,
            "rubric_version": (human[0].get("rubric_version", "")
                               if human else ""),
            "reason": "aggregated from >=2 independent human blind reviews",
        }

    return {
        "overlay": overlay,
        "n_runs_resolved": len(overlay),
        "skipped": skipped,
        "unblinded": True,
        "audit": audit,
    }


def apply_overlay(runs: list[Any], overlay: dict[str, dict[str, Any]]
                  ) -> list[Any]:
    """Return copies of ``runs`` with human-blind quality applied in memory.

    No persisted / finalized record is mutated. ``copy`` is a shallow dataclass
    clone so the original registry objects remain untouched.
    """
    from dataclasses import replace
    out = []
    for r in runs:
        q = overlay.get(r.run_id)
        if q is not None and not getattr(r, "quality_review", None):
            out.append(replace(r, quality_review=dict(q)))
        else:
            out.append(r)
    return out


__all__ = [
    "RUBRIC_DIMENSIONS", "review_quality_audit",
    "resolve_human_quality_overlay", "apply_overlay",
]
