"""M4.3-01 -- Experiment registry: blocks, pairing, and COMPLETE run records.

Independent of ResearchService. It records WHAT was run, under which frozen
configuration, and keeps EVERY outcome including failed / invalid /
INCONCLUSIVE / interrupted runs (they are never dropped from the denominator).

Design rules (spec M4.3-01):
  * A/B/C use the SAME research-opportunity Snapshot (blocked pairing).
  * Snapshot / Seed hierarchy is preserved: each block has one snapshot_id and
    one seed; the three arms of that block share them.
  * All arms share budget, paired seeds, model/prompt/Ranker version and the
    SAME 12-step gate (recorded as invariants, not re-implemented here).
  * Frozen fields are hashed; a post-hoc change is detectable and rejected.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .metrics import ALL_METRICS, PRIMARY_METRICS
from .research_service_gate import GATED_OUTCOMES

ARMS: tuple[str, ...] = ("A", "B", "C")
ARM_DESCRIPTIONS: dict[str, str] = {
    "A": ("Manual: human selects an Objective from current research evidence, "
          "WITHOUT historical Experience assistance."),
    "B": ("Memory-assisted Manual: human reads the SAME historical Evidence, "
          "Experience and Failure context, then selects an Objective."),
    "C": ("Autonomous Discovery: ObjectiveCandidateGenerator + Critic + Ranker "
          "select the Objective automatically."),
}

# A run's terminal classification. Every one of these is a RECORDED outcome;
# none is silently discarded.
RUN_OUTCOMES: tuple[str, ...] = (
    "valid_finding", "negative_finding", "inconclusive", "invalid",
    "failed", "interrupted",
)


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class FrozenProtocol:
    """The COMPLETE confirmatory pre-registration (P0-04, 20 required fields).

    Field names are the SINGLE schema shared with ``M4_3_runs/protocol.yaml``
    (``confirmatory_required_fields``), enforced by a contract test. Pilot
    (M4.3-02) may leave them empty; the confirmatory experiment (M4.3-03) MUST
    set every one of them. ``budget`` is kept as a backwards-compatible alias
    for ``campaign_budget`` used by the harness' paired-seed helper.
    """
    # --- identity / design -------------------------------------------------
    confirmatory: bool = False
    snapshot_ids: tuple[str, ...] = ()
    sample_size: int = 0
    seed: int | None = None
    # --- frozen versions ---------------------------------------------------
    model_version: str = ""
    prompt_version: str = ""
    ranker_version: str = ""
    domain_policy_version: str = ""
    analysis_plan_version: str = ""
    # --- budget / pairing --------------------------------------------------
    campaign_budget: dict[str, Any] = field(default_factory=dict)
    paired_seeds: tuple[int, ...] = ()
    budget: dict[str, Any] = field(default_factory=dict)  # legacy alias
    # --- frozen measurement ------------------------------------------------
    primary_metrics: tuple[str, ...] = PRIMARY_METRICS
    metric_contract_sha256: str = ""
    exclusion_rules: tuple[str, ...] = ()
    statistical_method: str = ""
    multiple_comparison_correction: str = ""
    stopping_rule: str = ""
    smallest_effect_of_interest: float = 0.0
    alpha: float = 0.0
    target_power: float = 0.0

    def frozen_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("snapshot_ids", "primary_metrics", "exclusion_rules",
                    "paired_seeds"):
            d[key] = list(getattr(self, key))
        return d

    def freeze_sha256(self) -> str:
        return _hash(self.frozen_dict())

    def missing_confirmatory_fields(self) -> tuple[str, ...]:
        from .protocol_schema import missing_confirmatory_fields
        return missing_confirmatory_fields(self)

    def assert_confirmatory_ready(self) -> None:
        """M4.3-03 gate: ALL 20 fields must be frozen. Fail-closed."""
        from .protocol_schema import assert_confirmatory_ready
        assert_confirmatory_ready(self)


@dataclass
class Block:
    """ONE research opportunity: 3 paired arms sharing snapshot + seed."""
    block_id: str
    snapshot_id: str
    seed: int
    arms: dict[str, str] = field(default_factory=dict)  # arm -> run_id


@dataclass
class RunRecord:
    """A COMPLETE record of one arm's run. Nothing is dropped."""
    run_id: str
    block_id: str
    snapshot_id: str
    seed: int
    arm: str
    outcome: str = "interrupted"          # RUN_OUTCOMES
    objective_id: str = ""
    # ---------------------------------------------------------------- P0-03
    # All trusted IDs + frozen hashes are persisted so an exported registry is
    # self-contained and verifiable after a restart.
    plan_id: str = ""
    approval_id: str = ""
    campaign_id: str = ""
    report_id: str = ""
    experience_id: str = ""
    protocol_sha256: str = ""
    snapshot_sha256: str = ""
    finalized: bool = False
    finalized_at: str = ""
    # objective-quality evidence (P0-02 independent BLIND review)
    evidence_refs: list[str] = field(default_factory=list)
    # opaque review_id only; the raw objective_id -> arm mapping is SEALED in
    # the BlindingLedger and never stored on the reviewer-visible payload.
    review_id: str = ""
    quality_review: dict[str, Any] = field(default_factory=dict)  # PRIMARY human-blind review
    independent_reviews: list[dict[str, Any]] = field(default_factory=list)
    automated_scores: list[dict[str, Any]] = field(default_factory=list)
    # execution evidence (P0-01: DERIVED from the real read-only gate adapter,
    # never caller-supplied). ``gate_evidence`` is the full persisted evidence
    # chain that produced ``gate_steps_passed`` / ``external_approval``.
    gate_steps_passed: int = 0
    external_approval: bool = False
    gate_evidence: dict[str, Any] = field(default_factory=dict)
    compute_units: float = 0.0
    wall_time_s: float = 0.0
    paired_seeds: list[int] = field(default_factory=list)
    # audit flags
    frozen_split_violation: bool = False
    approval_bypass: bool = False
    duplicate_campaign: bool = False
    unsupported_evidence_claim: bool = False
    false_objective: bool = False
    invalid_reason: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_valid_finding(self) -> bool:
        return self.outcome == "valid_finding"

    @property
    def is_finalized(self) -> bool:
        return self.finalized

    @property
    def objective_quality(self) -> float:
        return float(self.quality_review.get("total") or 0.0)


class ExperimentRegistry:
    """Registry of blocked, paired A/B/C runs with an append-only audit trail.

    P0-03: with ``store`` (or ``db_path``/``db``) the registry persists every
    Block / Run / Review / Event into ARR's SQLite. ``update`` can NEVER
    silently overwrite a FINALIZED run, a duplicate run is refused, and a
    process restart recovers unfinished blocks via :meth:`recover`.
    """

    def __init__(self, *, protocol: FrozenProtocol, store: Any = None,
                 db_path: str | Path | None = None, db: Any = None):
        from .registry_store import RegistryStore
        self.protocol = protocol
        self.blocks: dict[str, Block] = {}
        self.runs: dict[str, RunRecord] = {}
        self._frozen_hash = protocol.freeze_sha256()
        if store is None and (db_path is not None or db is not None):
            store = RegistryStore(db=db, db_path=db_path)
        self.store = store
        self._snapshot_sha: dict[str, str] = {}

    @property
    def freeze_sha256(self) -> str:
        return self._frozen_hash

    # ------------------------------------------------------------- blocks
    def create_block(self, *, block_id: str, snapshot_id: str, seed: int,
                     snapshot: Any = None) -> Block:
        if block_id in self.blocks:
            raise ValueError(f"block {block_id!r} already exists")
        if snapshot_id not in self.protocol.snapshot_ids:
            # Pilot may run before the snapshot set is frozen; confirmatory must
            # be frozen. Enforce strictly when confirmatory.
            if self.protocol.confirmatory:
                raise ValueError(
                    f"snapshot {snapshot_id!r} is not in the frozen snapshot set")
        block = Block(block_id=block_id, snapshot_id=snapshot_id, seed=seed)
        from .registry_store import snapshot_sha256
        sha = snapshot_sha256(snapshot if snapshot is not None
                              else {"snapshot_id": snapshot_id, "seed": seed})
        self._snapshot_sha[block_id] = sha
        if self.store is not None:
            self.store.save_block(block, protocol_sha256=self._frozen_hash,
                                  snapshot_sha256=sha)
        self.blocks[block_id] = block
        return block

    # --------------------------------------------------------------- runs
    def register_run(self, *, block_id: str, arm: str, **kwargs: Any) -> RunRecord:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        block = self.blocks.get(block_id)
        if block is None:
            raise ValueError(f"unknown block {block_id!r}")
        if arm in block.arms:
            raise ValueError(f"arm {arm} already registered in block {block_id}")
        run_id = kwargs.pop("run_id", "") or f"{block_id}_{arm}"
        if run_id in self.runs:
            raise ValueError(f"duplicate run_id {run_id!r}")
        outcome = kwargs.get("outcome", "interrupted")
        if outcome not in RUN_OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        rec = RunRecord(
            run_id=run_id, block_id=block_id,
            snapshot_id=block.snapshot_id, seed=block.seed, arm=arm,
            protocol_sha256=self._frozen_hash,
            snapshot_sha256=self._snapshot_sha.get(block_id, ""), **kwargs)
        if self.store is not None:
            self.store.save_run(rec, protocol_sha256=self._frozen_hash,
                                snapshot_sha256=rec.snapshot_sha256)
        self.runs[run_id] = rec
        block.arms[arm] = run_id
        return rec

    def update(self, run_id: str, **changes: Any) -> RunRecord:
        """Amend a NON-finalized run. A finalized run is IMMUTABLE (P0-03)."""
        rec = self.runs.get(run_id)
        if rec is None:
            raise ValueError(f"unknown run {run_id!r}")
        if rec.finalized:
            from .registry_store import RegistryConflict
            raise RegistryConflict(
                f"run {run_id!r} is finalized; completed runs cannot be "
                "overwritten (append-only audit)")
        if "outcome" in changes and changes["outcome"] not in RUN_OUTCOMES:
            raise ValueError(f"unknown outcome {changes['outcome']!r}")
        for key, value in changes.items():
            if not hasattr(rec, key):
                raise ValueError(f"unknown RunRecord field {key!r}")
            setattr(rec, key, value)
        if self.store is not None:
            self.store.update_run(rec, changes=changes)
        return rec

    def finalize(self, run_id: str, **changes: Any) -> RunRecord:
        """FINALIZE exactly once (record_outcome). No double execution."""
        rec = self.runs.get(run_id)
        if rec is None:
            raise ValueError(f"unknown run {run_id!r}")
        if rec.finalized:
            from .registry_store import RegistryConflict
            raise RegistryConflict(
                f"run {run_id!r} already finalized; refusing a second outcome "
                "(no duplicate campaign / no budget re-consumption)")
        # (M4.3-02 P0 #7) ONE campaign = ONE run -- for CLAIMED SUCCESS.
        # A gated outcome (valid/negative/inconclusive) claims this campaign as
        # success evidence, so it may never be reused by another run. A real
        # failure / interruption / invalid run may still *reference* the
        # campaign for diagnostics: it claims no success, so no exclusivity is
        # asserted (that also keeps the audit trail complete).
        campaign_id = str(changes.get("campaign_id") or "")
        if campaign_id and str(changes.get("outcome", "")) in GATED_OUTCOMES:
            self.assert_campaign_unused(run_id, campaign_id)
        for key, value in changes.items():
            if not hasattr(rec, key):
                raise ValueError(f"unknown RunRecord field {key!r}")
            setattr(rec, key, value)
        rec.finalized = True
        from datetime import datetime, timezone
        rec.finalized_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        if self.store is not None:
            self.store.finalize_run(rec, changes=changes)
        return rec

    def assert_campaign_unused(self, run_id: str, campaign_id: str) -> None:
        """Refuse to bind a campaign that already belongs to another run."""
        from .registry_store import RegistryConflict
        owners = {rid for rid, r in self.runs.items()
                  if rid != run_id and getattr(r, "campaign_id", "") == campaign_id}
        if self.store is not None:
            owners |= {rid for rid in self.store.runs_with_campaign(campaign_id)
                       if rid != run_id}
        if owners:
            raise RegistryConflict(
                f"campaign {campaign_id!r} is already bound to run(s) "
                f"{sorted(owners)}; one campaign can never be reused as "
                "success evidence by another run / block / arm")

    def record_review(self, review: dict[str, Any]) -> None:
        """Persist an independent review (append-only, never overwritten)."""
        if self.store is not None:
            self.store.save_review(review)

    def append_event(self, *, scope: str, event_type: str,
                     payload: dict[str, Any]) -> None:
        """Record a REAL failure / interruption / recovery event."""
        if self.store is not None:
            self.store.append_event(scope=scope, event_type=event_type,
                                    payload=payload)

    # --------------------------------------------------- P0-03 recovery
    @classmethod
    def recover(cls, *, protocol: FrozenProtocol,
                db_path: str | Path | None = None, db: Any = None,
                store: Any = None) -> ExperimentRegistry:
        """Rebuild a registry after a process restart (incl. incomplete blocks)."""
        from .registry_store import RegistryStore
        if store is None:
            store = RegistryStore(db=db, db_path=db_path)
        reg = cls(protocol=protocol, store=store)
        for b in store.load_blocks():
            reg.blocks[b["block_id"]] = Block(
                block_id=b["block_id"], snapshot_id=b["snapshot_id"],
                seed=int(b["seed"]),
                arms={k: v for k, v in (b.get("arms") or {}).items()})
            reg._snapshot_sha[b["block_id"]] = b.get("snapshot_sha256", "")
        for d in store.load_runs():
            known = {f for f in RunRecord.__dataclass_fields__}
            rec = RunRecord(**{k: v for k, v in d.items() if k in known})
            reg.runs[rec.run_id] = rec
            if rec.block_id in reg.blocks:
                reg.blocks[rec.block_id].arms[rec.arm] = rec.run_id
        store.append_event(
            scope="registry", event_type="registry_recovered",
            payload={"n_blocks": len(reg.blocks), "n_runs": len(reg.runs),
                     "protocol_sha256": reg._frozen_hash,
                     "incomplete_blocks": reg.unfinished_blocks()})
        return reg

    def unfinished_blocks(self) -> list[str]:
        """Blocks where at least one arm is NOT yet FINALIZED (P0-03 recovery).

        ``complete_blocks()`` means "all three arms REGISTERED" (paired
        analysis). Recovery needs the stronger notion: every arm has a final,
        immutable outcome. A crash after registering all arms but before
        finalizing them MUST still be recoverable.
        """
        out = []
        for bid, block in self.blocks.items():
            arms = [self.runs.get(rid) for rid in block.arms.values()]
            if (len(arms) < len(ARMS)
                    or any(r is None or not r.finalized for r in arms)):
                out.append(bid)
        return sorted(out)

    def finalized_blocks(self) -> list[str]:
        return sorted(bid for bid in self.blocks
                      if bid not in self.unfinished_blocks())

    # ------------------------------------------------------------ export
    def export(self) -> dict[str, Any]:
        if self.store is None:
            return {"persisted": False, **self.to_dict()}
        return {"persisted": True, **self.store.export()}

    def verify_integrity(self) -> bool:
        if self.store is None:
            return False
        return self.store.verify_integrity()

    # ------------------------------------------------------------ queries
    def runs_for_block(self, block_id: str) -> dict[str, RunRecord]:
        block = self.blocks[block_id]
        return {arm: self.runs[rid] for arm, rid in block.arms.items()}

    def complete_blocks(self) -> list[str]:
        """Blocks where ALL THREE arms are recorded (for paired comparison)."""
        return sorted(bid for bid, b in self.blocks.items()
                      if all(a in b.arms for a in ARMS))

    def all_runs(self) -> list[RunRecord]:
        return list(self.runs.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.frozen_dict(),
            "freeze_sha256": self.freeze_sha256,
            "metric_names": list(ALL_METRICS),
            "blocks": {k: asdict(v) for k, v in self.blocks.items()},
            "runs": {k: v.to_dict() for k, v in self.runs.items()},
        }


__all__ = [
    "ARMS", "ARM_DESCRIPTIONS", "RUN_OUTCOMES", "FrozenProtocol", "Block",
    "RunRecord", "ExperimentRegistry",
]
