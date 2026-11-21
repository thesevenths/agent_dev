"""M4.3-01.1 P0-02 -- GENUINELY anonymous, blind, independent review.

Defects fixed relative to M4.3-01:
  * ``BlindedObjective`` used to carry the raw ``objective_id``. IDs like
    ``block_1_A`` leak the arm directly; reviewers were NOT actually blind.
  * there was no opaque ``review_id`` and no separately sealed mapping;
  * nothing scrubbed group labels out of descriptions / metadata;
  * nothing prevented reading the arm mapping before unblinding;
  * a ``QualityReview`` had no timestamp, and could be silently overwritten;
  * only ONE reviewer was supported, and automated scores were mixed with
    human independent review in the SAME metric.

Contract:
  * a reviewer sees ONLY ``BlindedObjective`` (opaque ``review_id``, scrubbed
    text). No objective_id, arm, rank_score, ranker_version, generation_source,
    snapshot_id or block_id.
  * ``BlindingLedger`` seals the ``review_id -> (objective_id, arm, ...)``
    mapping SEPARATELY and read-only until ``unblind()`` is explicitly called
    AFTER reviews are complete.
  * reviews are append-only: the same (review_id, reviewer_id, review_kind)
    cannot be recorded twice; the original result is never overwritten.
  * ``QualityReview`` always carries reviewer_id, rubric_version, reason and a
    timestamp.
  * ``ReviewPanel`` defaults to >= 2 independent reviewers (Pilot default).
  * ``review_kind`` distinguishes ``human_blind`` from ``automated``; only
    human_blind reviews feed ``independently_reviewed_objective_quality``.
"""
from __future__ import annotations

import datetime
import re
import secrets
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

# Fixed rubric (frozen) for the INDEPENDENT objective-quality review.
REVIEW_RUBRIC_VERSION = "m4_3_objective_quality_rubric_v1"
RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "evidence_grounding",
    "novelty",
    "falsifiability",
    "intervention_feasibility",
    "metric_measurability",
    "expected_information_gain",
)
RUBRIC_SCALE = (0, 4)

# Frozen review kinds. An automated score is NEVER an independent blind review.
REVIEW_KIND_HUMAN_BLIND = "human_blind"
REVIEW_KIND_AUTOMATED = "automated"
REVIEW_KINDS: tuple[str, ...] = (
    REVIEW_KIND_HUMAN_BLIND, REVIEW_KIND_AUTOMATED,
)

# Fields that must NEVER appear in a blinded payload.
FORBIDDEN_BLIND_FIELDS = frozenset({
    "objective_id", "arm", "group", "rank_score", "rank_breakdown",
    "ranker_version", "generation_source", "snapshot_id", "block_id",
    "review_arm", "cohort",
})

# Text patterns that directly leak grouping / Ranker signals inside a
# description or nested metadata. Fail-closed (reject) rather than silently
# rewrite: silently editing scientific text would be a hidden data change.
#
# Only STRUCTURAL leaks are intercepted (IDs that encode an arm, explicit
# "arm X" labels, Ranker field names). Bare English words such as "manual" are
# deliberately NOT blacklisted: they occur in legitimate scientific prose
# ("manual annotation") and blocking them would reject valid objectives. The
# real guarantees are (a) objective_id never enters the blinded payload,
# (b) review_id is opaque, (c) the arm mapping is sealed until unblinding.
_ARM_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bblock[_-][a-z0-9_.-]*[_-](?:a|b|c)\b",      # block_1_A
        r"\b(?:objective|obj|run|trial)[_-][a-z0-9_.-]*[_-](?:a|b|c)\b",
        r"\b(?:arm|group|cohort)\s*[:=#]?\s*[\"']?[abc][\"']?\b",
        r"\brank(?:er)?[_-]?(?:score|breakdown|version)\b",
        r"\bgeneration[_-]source\b",
        r"\bsnapshot[_-]?id\b",
        r"\bblock[_-]?id\b",
    )
)


class BlindingViolation(ValueError):
    """A leak, an early unblind, or an overwrite of a sealed review."""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def _new_review_id() -> str:
    """Opaque, non-reversible ID. Contains NO objective_id and NO arm."""
    return "rev_" + secrets.token_hex(16)


def _scrub_text(value: str) -> str:
    """Reject group/Ranker leaks anywhere in free text (fail-closed)."""
    for pattern in _ARM_LEAK_PATTERNS:
        m = pattern.search(value)
        if m:
            raise BlindingViolation(
                "blinded description/metadata leaks grouping or Ranker "
                f"signal: {m.group(0)!r}")
    return value


def is_group_leaking(text: str) -> bool:
    """True if ``text`` carries an arm / block / Ranker signal (fail-closed).

    Public so a caller can decide NOT to hand such a string to the blinding
    layer at all (e.g. an evidence reference that literally encodes the arm).
    """
    return any(p.search(str(text)) for p in _ARM_LEAK_PATTERNS)


def _scrub_obj(value: Any) -> Any:
    """Recursively validate every string (keys AND values) in a payload."""
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            _scrub_text(str(key))
            clean[str(key)] = _scrub_obj(item)
        return clean
    if isinstance(value, (list, tuple)):
        return [_scrub_obj(item) for item in value]
    return value


@dataclass(frozen=True)
class BlindedObjective:
    """What an independent reviewer is ALLOWED to see.

    There is deliberately NO ``objective_id`` and no group/Ranker signal. The
    ONLY identifier is the opaque ``review_id``.
    """
    review_id: str
    description: str
    evidence_refs: tuple[str, ...]
    success_criteria: dict[str, Any]
    proposed_intervention: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityReview:
    """Immutable, timestamped, reasoned review of ONE blinded objective."""
    review_id: str
    reviewer_id: str
    scores: dict[str, int]
    reason: str
    rubric_version: str = REVIEW_RUBRIC_VERSION
    reviewed_at: str = ""
    review_kind: str = REVIEW_KIND_HUMAN_BLIND

    @property
    def total(self) -> float:
        """Mean over the SIX fixed dimensions (0..4)."""
        if not self.scores:
            return 0.0
        return round(sum(self.scores.values()) / len(RUBRIC_DIMENSIONS), 6)

    @property
    def is_human_blind(self) -> bool:
        return self.review_kind == REVIEW_KIND_HUMAN_BLIND

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total"] = self.total
        d["is_human_blind"] = self.is_human_blind
        return d

    def identity(self) -> tuple[str, str, str]:
        """Append-only identity: no two reviews may share it."""
        return (self.review_id, self.reviewer_id, self.review_kind)


def blind(objective: dict[str, Any], *,
          review_id_factory: Callable[[], str] = _new_review_id
          ) -> BlindedObjective:
    """Produce an anonymous review payload; FAIL CLOSED on any leak.

    ``objective`` may legitimately carry ``objective_id`` etc. as INPUT (needed
    to build the sealed mapping), but NONE of the forbidden fields may survive
    into the returned payload, and no free text may contain a group label.
    """
    if not isinstance(objective, dict):
        raise BlindingViolation("objective must be a dict")
    leaked = FORBIDDEN_BLIND_FIELDS.intersection(
        k for k in objective if k != "objective_id")
    if leaked:
        raise BlindingViolation(
            "blinded objective must not expose arm/objective_id/Ranker "
            f"fields: {sorted(leaked)}")
    return BlindedObjective(
        review_id=review_id_factory(),
        description=_scrub_obj(str(objective.get("description") or "")),
        evidence_refs=tuple(
            _scrub_obj(str(r)) for r in (objective.get("evidence_refs") or [])),
        success_criteria=_scrub_obj(dict(objective.get("success_criteria") or {})),
        proposed_intervention=_scrub_obj(
            dict(objective.get("proposed_intervention") or {})),
    )


class IndependentReviewer:
    """Applies the FIXED rubric via an injected scorer, BLINDLY.

    ``scorer`` receives ONLY a ``BlindedObjective``. There is NO code path from
    a Ranker score into a review score. ``review_kind`` labels human blind
    review vs an automated score so the two can never be mixed in a metric.
    """

    def __init__(self, *, reviewer_id: str,
                 scorer: Callable[[BlindedObjective], dict[str, Any]],
                 review_kind: str = REVIEW_KIND_HUMAN_BLIND):
        if not reviewer_id or not reviewer_id.strip():
            raise ValueError("reviewer_id is required")
        if review_kind not in REVIEW_KINDS:
            raise ValueError(f"unknown review_kind {review_kind!r}")
        self.reviewer_id = reviewer_id.strip()
        self.scorer = scorer
        self.review_kind = review_kind
        self.reviews: list[QualityReview] = []

    def review(self, objective: dict[str, Any] | BlindedObjective,
               *, review_id: str | None = None) -> QualityReview:
        if isinstance(objective, BlindedObjective):
            blinded = objective
        else:
            blinded = blind(objective)
        raw = self.scorer(blinded) or {}
        scores: dict[str, int] = {}
        for dim in RUBRIC_DIMENSIONS:
            value = raw.get(dim)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"rubric dimension {dim!r} missing/non-numeric")
            if not (RUBRIC_SCALE[0] <= value <= RUBRIC_SCALE[1]):
                raise ValueError(f"rubric dimension {dim!r} out of scale: {value}")
            scores[dim] = int(value)
        reason = str(raw.get("reason") or "").strip()
        if not reason:
            raise ValueError("independent review must record a reason")
        review = QualityReview(
            review_id=review_id or blinded.review_id,
            reviewer_id=self.reviewer_id, scores=scores, reason=reason,
            rubric_version=REVIEW_RUBRIC_VERSION, reviewed_at=_now(),
            review_kind=self.review_kind)
        self.reviews.append(review)
        return review


class ReviewPanel:
    """Pilot default: at least TWO independent, blinded human reviewers."""

    def __init__(self, reviewers: list[IndependentReviewer], *,
                 min_reviewers: int = 2):
        if min_reviewers < 2:
            raise ValueError("a Pilot review panel requires >= 2 reviewers")
        self.reviewers = tuple(reviewers)
        self.min_reviewers = min_reviewers
        ids = [r.reviewer_id for r in self.reviewers]
        if len(set(ids)) != len(ids):
            raise ValueError("reviewer_id values must be distinct")

    @property
    def human_reviewers(self) -> tuple[IndependentReviewer, ...]:
        return tuple(r for r in self.reviewers
                     if r.review_kind == REVIEW_KIND_HUMAN_BLIND)

    @property
    def is_independent(self) -> bool:
        """>= min_reviewers DISTINCT human blind reviewers (no self-review)."""
        return (len(self.human_reviewers) >= self.min_reviewers
                and len({r.reviewer_id for r in self.human_reviewers})
                >= self.min_reviewers)

    def review(self, blinded: BlindedObjective) -> list[QualityReview]:
        return [r.review(blinded, review_id=blinded.review_id)
                for r in self.reviewers]


@dataclass(frozen=True)
class BlindingEntry:
    """SEALED mapping. Kept strictly separate from the blinded payload."""
    review_id: str
    objective_id: str
    arm: str
    block_id: str
    snapshot_id: str
    generation_source: str
    rank_score: Any
    sealed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BlindingLedger:
    """Append-only blinding ledger: sealed mappings + immutable reviews.

    The arm mapping is NOT readable until ``unblind()`` is explicitly invoked
    AFTER the required number of human blind reviews have been recorded.
    """

    def __init__(self, *, required_human_reviews: int = 2,
                 review_id_factory: Callable[[], str] = _new_review_id):
        if required_human_reviews < 2:
            raise ValueError("Pilot default requires >= 2 independent reviews")
        self.required_human_reviews = required_human_reviews
        self._review_id_factory = review_id_factory
        self._entries: dict[str, BlindingEntry] = {}
        self._reviews: dict[str, list[QualityReview]] = {}
        self._review_identities: set[tuple[str, str, str]] = set()
        self._unblinded = False

    # ------------------------------------------------------------- sealing
    def seal(self, objective: dict[str, Any], *, arm: str = "",
             block_id: str = "", snapshot_id: str = "") -> BlindedObjective:
        if self._unblinded:
            raise BlindingViolation(
                "ledger is unblinded; no new objective can be sealed")
        rid = self._review_id_factory()
        raw_id = str(objective.get("objective_id") or "")
        # The internal objective_id is NOT exposed to reviewers (it is stored
        # only in this sealed entry and replaced by the opaque review_id in the
        # payload). We therefore do not blacklist arbitrary internal IDs; the
        # leak test applies to the DESCRIPTION / METADATA that reviewers read.
        clean = {k: v for k, v in objective.items()
                 if k not in FORBIDDEN_BLIND_FIELDS}
        blinded = blind(
            {**clean, "description": objective.get("description") or ""},
            review_id_factory=lambda: rid)
        self._entries[rid] = BlindingEntry(
            review_id=rid, objective_id=raw_id, arm=arm, block_id=block_id,
            snapshot_id=snapshot_id,
            generation_source=str(objective.get("generation_source") or ""),
            rank_score=objective.get("rank_score"), sealed_at=_now())
        self._reviews.setdefault(rid, [])
        return blinded

    # ------------------------------------------------------------- reviews
    def record_review(self, review: QualityReview) -> None:
        """APPEND-ONLY. A completed review can never be silently overwritten."""
        if review.review_id not in self._entries:
            raise BlindingViolation("review_id is not sealed in this ledger")
        if review.identity() in self._review_identities:
            raise BlindingViolation(
                "review already recorded for "
                f"(review_id, reviewer_id, review_kind)={review.identity()}; "
                "original review result cannot be overwritten")
        self._review_identities.add(review.identity())
        self._reviews[review.review_id].append(review)

    def reviews_for(self, review_id: str) -> tuple[QualityReview, ...]:
        # Review RESULTS themselves are not secret; the ARM MAPPING is.
        if review_id not in self._entries:
            raise BlindingViolation("unknown review_id")
        return tuple(self._reviews[review_id])

    def human_blind_reviews(self, review_id: str) -> tuple[QualityReview, ...]:
        return tuple(r for r in self.reviews_for(review_id)
                     if r.is_human_blind)

    # ------------------------------------------------------------- unblinding
    @property
    def unblinded(self) -> bool:
        return self._unblinded

    def arm_mapping(self, review_id: str) -> dict[str, Any]:
        """FORBIDDEN before unblinding. Reviewers must not see the arm."""
        if not self._unblinded:
            raise BlindingViolation(
                "arm mapping is sealed until all reviews are complete; "
                "call unblind() only after the review phase has ended")
        if review_id not in self._entries:
            raise BlindingViolation("unknown review_id")
        return self._entries[review_id].to_dict()

    def unblind(self) -> None:
        """Allowed ONLY after every sealed objective has enough human reviews."""
        if self._unblinded:
            return
        incomplete = [
            rid for rid in self._entries
            if len(self.human_blind_reviews(rid)) < self.required_human_reviews]
        if incomplete:
            raise BlindingViolation(
                "cannot unblind before review completion; missing independent "
                f"human reviews for {len(incomplete)} objective(s)")
        self._unblinded = True

    def export(self) -> dict[str, Any]:
        """Full audit export; includes the mapping ONLY after unblinding."""
        return {
            "unblinded": self._unblinded,
            "required_human_reviews": self.required_human_reviews,
            "n_sealed": len(self._entries),
            "entries": ({rid: e.to_dict() for rid, e in self._entries.items()}
                        if self._unblinded else "SEALED"),
            "reviews": {rid: [r.to_dict() for r in reviews]
                        for rid, reviews in self._reviews.items()},
        }


__all__ = [
    "REVIEW_RUBRIC_VERSION", "RUBRIC_DIMENSIONS", "RUBRIC_SCALE",
    "REVIEW_KIND_HUMAN_BLIND", "REVIEW_KIND_AUTOMATED", "REVIEW_KINDS",
    "FORBIDDEN_BLIND_FIELDS", "BlindingViolation", "is_group_leaking",
    "BlindedObjective",
    "QualityReview", "blind", "IndependentReviewer", "ReviewPanel",
    "BlindingEntry", "BlindingLedger",
]
