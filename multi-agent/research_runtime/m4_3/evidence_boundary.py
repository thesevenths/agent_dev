"""M4.3-01.1 P1-02 -- protocol & execution boundary for A/B/C.

Spec P1-02, made executable (not just prose):
  * arm A reads ONLY the CURRENT evidence of the research opportunity;
  * B/C read the SAME historical Evidence / Experience / Failure;
  * all three arms share the SAME research-opportunity Snapshot;
  * CURRENT vs HISTORICAL are separated by EXPLICIT, DISJOINT ID sets;
  * NO arm may read a reserved (frozen) final split;
  * candidate budget, human selection time budget and Campaign Budget are
    FIXED per block and cannot be silently changed;
  * every block retains all three arms' COMPLETE records, including failed /
    interrupted / invalid / INCONCLUSIVE outcomes.

This module is a pure, framework-independent validator. It never reads a
database, runs an engine or touches the 12-step gate.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..schemas import RESERVED_FINAL_SPLITS

# Keys whose value denotes an evaluation split.
_SPLIT_KEYS = frozenset({
    "split", "evaluation_split", "dataset_split", "data_split",
    "final_split", "heldout_split", "eval_split",
})
# Tokens that identify the FROZEN FINAL split (the ARR reserved set plus the
# common aliases an operator might use). Never usable as selection evidence.
FROZEN_SPLIT_TOKENS: frozenset[str] = frozenset(
    {s.lower() for s in RESERVED_FINAL_SPLITS}
    | {"frozen_final", "final_test", "test_final", "frozen_test",
       "reserved_final", "final_holdout", "heldout_final"}
)
_FROZEN_SPLIT_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:" + "|".join(
        re.escape(t) for t in sorted(FROZEN_SPLIT_TOKENS)) +
    r")(?:$|[^a-z0-9])", re.IGNORECASE)

# Explicit-ID keys in an evidence/experience/failure record.
_ID_KEYS = ("evidence_id", "id", "experience_id", "ref", "session_id",
            "run_id", "record_id", "objective_id", "sha256", "source_id")


class EvidenceBoundaryViolation(ValueError):
    """A protocol boundary (A history / ID separation / frozen split) broke."""


def record_id(record: Any) -> str:
    """Return an EXPLICIT id for a record, or a stable content-hash id."""
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for key in _ID_KEYS:
            value = record.get(key)
            if value:
                return str(value)
    return "sha256:" + hashlib.sha256(
        json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _contains_frozen_split(value: Any) -> str | None:
    """Return the offending token if a payload references a frozen split."""
    if isinstance(value, str):
        token = value.strip().lower()
        if token in FROZEN_SPLIT_TOKENS:
            return token
        m = _FROZEN_SPLIT_RE.search(value)
        return m.group(0).strip() if m else None
    if isinstance(value, dict):
        for k, v in value.items():
            key = str(k).strip().lower()
            if key in _SPLIT_KEYS and isinstance(v, str) \
                    and v.strip().lower() in FROZEN_SPLIT_TOKENS:
                return v.strip().lower()
            found = _contains_frozen_split(v)
            if found:
                return found
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            found = _contains_frozen_split(item)
            if found:
                return found
    return None


@dataclass(frozen=True)
class BlockBudget:
    """FIXED per-block budgets (P1-02). No arm may exceed them."""
    candidate_budget: int
    human_time_budget_s: float
    campaign_budget: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_budget": self.candidate_budget,
                "human_time_budget_s": self.human_time_budget_s,
                "campaign_budget": dict(self.campaign_budget)}


@dataclass(frozen=True)
class EvidencePartition:
    """The EXPLICIT, disjoint ID sets for one research opportunity."""
    snapshot_id: str
    current_ids: tuple[str, ...]
    historical_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"snapshot_id": self.snapshot_id,
                "current_ids": list(self.current_ids),
                "historical_ids": list(self.historical_ids)}


def validate_evidence_partition(
    *, snapshot_id: str,
    current_evidence: Any,
    historical_experience: list[Any],
    historical_evidence: list[Any],
    historical_failure: list[Any],
    explicit_current_ids: tuple[str, ...] | None = None,
    explicit_historical_ids: tuple[str, ...] | None = None,
) -> EvidencePartition:
    """Enforce explicit-ID separation + frozen-final-split exclusion.

    Fails closed on: an ID present in BOTH current and historical sets, an
    explicit set that disagrees with the records, and ANY reference to a
    reserved/frozen final split in either current or historical evidence.
    """
    current_records = ([current_evidence] if isinstance(current_evidence, dict)
                       else list(current_evidence or []))
    history_records = (list(historical_evidence or [])
                       + list(historical_experience or [])
                       + list(historical_failure or []))

    # (P1-02) NO arm may read the frozen final split -- checked on BOTH sides.
    for label, records in (("current_evidence", current_records),
                           ("historical_context", history_records)):
        for rec in records:
            token = _contains_frozen_split(rec)
            if token:
                raise EvidenceBoundaryViolation(
                    f"{label} references the FROZEN FINAL split ({token!r}); "
                    "no arm may use the reserved final split as selection "
                    "evidence")
        token = _contains_frozen_split({"split": snapshot_id})
        if token and label == "current_evidence":
            raise EvidenceBoundaryViolation(
                f"snapshot_id references a frozen final split ({token!r})")

    derived_current = tuple(sorted({record_id(r) for r in current_records}))
    derived_history = tuple(sorted({record_id(r) for r in history_records}))

    current_ids = (tuple(explicit_current_ids)
                   if explicit_current_ids is not None else derived_current)
    historical_ids = (tuple(explicit_historical_ids)
                      if explicit_historical_ids is not None else derived_history)

    # An explicit ID set must actually match the records it claims to describe
    # (no phantom / omitted evidence).
    if explicit_current_ids is not None and \
            set(current_ids) != set(derived_current):
        raise EvidenceBoundaryViolation(
            "explicit current_evidence_ids do not match the current records")
    if explicit_historical_ids is not None and \
            set(historical_ids) != set(derived_history):
        raise EvidenceBoundaryViolation(
            "explicit historical_ids do not match the historical records")

    overlap = set(current_ids) & set(historical_ids)
    if overlap:
        raise EvidenceBoundaryViolation(
            "current and historical evidence MUST be disjoint explicit ID "
            f"sets; overlap: {sorted(overlap)}")
    return EvidencePartition(
        snapshot_id=snapshot_id,
        current_ids=tuple(sorted(current_ids)),
        historical_ids=tuple(sorted(historical_ids)))


def validate_block_budget(*, candidate_budget: int,
                          human_time_budget_s: float,
                          campaign_budget: dict[str, Any],
                          protocol_campaign_budget: dict[str, Any] | None = None,
                          ) -> BlockBudget:
    """Budgets are FIXED, positive, and must match the frozen protocol budget."""
    if candidate_budget <= 0:
        raise EvidenceBoundaryViolation("candidate_budget must be positive")
    if human_time_budget_s <= 0:
        raise EvidenceBoundaryViolation(
            "human_time_budget_s must be positive and fixed per block")
    if protocol_campaign_budget:
        if dict(campaign_budget) != dict(protocol_campaign_budget):
            raise EvidenceBoundaryViolation(
                "Campaign Budget must equal the FROZEN protocol budget; "
                "per-block budget cannot be changed")
    return BlockBudget(
        candidate_budget=int(candidate_budget),
        human_time_budget_s=float(human_time_budget_s),
        campaign_budget=dict(campaign_budget))


__all__ = [
    "FROZEN_SPLIT_TOKENS", "BlockBudget", "EvidencePartition",
    "EvidenceBoundaryViolation", "record_id", "validate_evidence_partition",
    "validate_block_budget",
]
