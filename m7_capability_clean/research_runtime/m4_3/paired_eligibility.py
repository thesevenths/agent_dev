"""M4.3-02.1 P0-01 -- ONE shared paired-block eligibility rule.

Both :mod:`pilot_analysis` and :mod:`value_report` MUST use this function so
they can never disagree on which blocks enter a paired contrast. This is the
single source of truth for P0-01.

A block is eligible for paired statistics ONLY if ALL of the following hold:
  1. the block is registered (when a block registry is supplied);
  2. all three arms A/B/C have a run record;
  3. all three runs are ``finalized``;
  4. all three outcomes are successful scientific outcomes (``GATED_OUTCOMES``)
     -- a finalized-but-failed / interrupted / invalid run is NOT a valid
     paired observation (it stays in the OVERALL statistics, never dropped);
  5. all three runs share ONE ``snapshot_id`` and it matches the registered
     block's snapshot (cross-snapshot pairing is refused);
  6. all three runs share ONE ``protocol_sha256`` (the pre-registered protocol
     version) and it matches the registered block / expected version.

Every rejected block is returned with explicit reasons so a report can list
``excluded_blocks``. Nothing is deleted or silently dropped.
"""
from __future__ import annotations

from typing import Any, Iterable

from .experiment_registry import ARMS, RunRecord
from .research_service_gate import GATED_OUTCOMES

#: Rule identifiers, exported so a report can state the exact gate contract.
ELIGIBILITY_RULES: tuple[str, ...] = (
    "all_three_arms_present", "all_three_finalized",
    "all_successful_scientific_outcomes", "one_shared_snapshot_id",
    "one_shared_protocol_version",
)


def _first_mismatch(values: Iterable[str]) -> bool:
    vals = list(values)
    return bool(vals) and any(v != vals[0] for v in vals[1:])


def select_paired_blocks(
        *, runs: list[RunRecord],
        blocks: list[dict[str, Any]] | None = None,
        expected_protocol_sha256: str = "",
) -> dict[str, Any]:
    """Partition runs into PAIR-ELIGIBLE and EXCLUDED blocks.

    Args:
        runs: every persisted run (failed / interrupted / invalid INCLUDED).
        blocks: optional registered blocks (for snapshot / protocol cross-check).
        expected_protocol_sha256: the pre-registered protocol version an
            eligible block must match, when supplied.

    Returns:
        ``{"eligible": [...], "excluded": [...], "runs_by_block": {...},
           "rules": [...]}`` where each eligible entry carries
        ``{block_id, snapshot_id, protocol_sha256, arms: {arm: RunRecord}}``
        and each excluded entry carries ``{block_id, reasons: [...]}``.
    """
    runs_by_block: dict[str, dict[str, RunRecord]] = {}
    for r in runs:
        runs_by_block.setdefault(r.block_id, {})[r.arm] = r

    block_index: dict[str, dict[str, Any]] = {}
    for b in (blocks or []):
        bid = b.get("block_id")
        if bid:
            block_index[bid] = b

    # Union of registered blocks and blocks implied by run records: a run whose
    # block is not registered is still reported (as EXCLUDED, never dropped).
    all_block_ids = sorted(set(block_index) | set(runs_by_block))

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for bid in all_block_ids:
        arms = runs_by_block.get(bid, {})
        block = block_index.get(bid)
        reasons: list[str] = []

        if block is None:
            reasons.append("block not registered")

        missing = [a for a in ARMS if a not in arms]
        if missing:
            reasons.append("arm missing: " + ",".join(missing))

        present = [arms[a] for a in ARMS if a in arms]

        if len(present) == len(ARMS):
            unfinalized = [a for a in ARMS if not arms[a].finalized]
            if unfinalized:
                reasons.append("arm not finalized: " + ",".join(unfinalized))

            bad_outcomes = [a for a in ARMS
                            if arms[a].outcome not in GATED_OUTCOMES]
            if bad_outcomes:
                reasons.append(
                    "arm outcome not a successful scientific outcome: "
                    + ",".join(f"{a}={arms[a].outcome}" for a in bad_outcomes))

            snapshots = [str(r.snapshot_id or "") for r in present]
            if _first_mismatch(snapshots):
                reasons.append("arms span different snapshot_id values")
            if block is not None:
                reg_snap = str(block.get("snapshot_id") or "")
                if reg_snap and any(s != reg_snap for s in snapshots):
                    reasons.append("arm snapshot_id differs from registered block")

            protocols = [str(r.protocol_sha256 or "") for r in present]
            if _first_mismatch(protocols):
                reasons.append("arms span different protocol versions")
            if block is not None:
                reg_proto = str(block.get("protocol_sha256") or "")
                if reg_proto and any(p != reg_proto for p in protocols):
                    reasons.append(
                        "arm protocol_sha256 differs from registered block")
            if expected_protocol_sha256:
                want = str(expected_protocol_sha256)
                if any(p != want for p in protocols):
                    reasons.append(
                        "arm protocol_sha256 differs from the pre-registered "
                        "protocol version")

        if reasons:
            excluded.append({"block_id": bid, "reasons": reasons})
        else:
            eligible.append({
                "block_id": bid,
                "snapshot_id": str(present[0].snapshot_id),
                "protocol_sha256": str(present[0].protocol_sha256),
                "arms": {a: arms[a] for a in ARMS},
            })

    return {
        "eligible": eligible,
        "excluded": excluded,
        "runs_by_block": runs_by_block,
        "rules": list(ELIGIBILITY_RULES),
    }


__all__ = ["ELIGIBILITY_RULES", "select_paired_blocks"]
