"""ARR 1.1 / G1 -- durable loop state for the autonomous research loop.

One JSON file is the SOLE source of truth for an interrupted run. Every round
appends a :class:`VersionRecord` and bumps ``current_round`` BEFORE the next
round starts, so a process that dies mid-round can be resumed from the last
COMPLETED round without re-running or silently skipping work.

Nothing here invents a result: it only records what the dual-layer controller
actually returned (accepted overrides, accepted harness targets, baseline path).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VersionRecord:
    """What round N accepted and baked into the NEXT round's baseline.

    A version is the accumulated scientific state: the domain config overrides
    and the harness patches that survived the dual-layer gate. It is the unit of
    "version inheritance" -- round N+1 literally starts from this baseline.
    """

    round_no: int
    baseline_config_path: str
    accepted_domain_overrides: dict[str, Any] = field(default_factory=dict)
    accepted_harness_targets: list[str] = field(default_factory=list)
    created_at: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "baseline_config_path": self.baseline_config_path,
            "accepted_domain_overrides": dict(self.accepted_domain_overrides),
            "accepted_harness_targets": list(self.accepted_harness_targets),
            "created_at": self.created_at,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VersionRecord":
        return cls(
            round_no=int(d.get("round_no", 0)),
            baseline_config_path=str(d.get("baseline_config_path", "")),
            accepted_domain_overrides=dict(d.get("accepted_domain_overrides") or {}),
            accepted_harness_targets=list(d.get("accepted_harness_targets") or []),
            created_at=str(d.get("created_at", "")),
            note=str(d.get("note", "")),
        )


@dataclass
class LoopState:
    """The full, resumable state of one autonomous research loop."""

    loop_id: str
    domain: str
    max_rounds: int
    current_round: int = 1
    status: str = "running"          # running | completed | failed
    baseline_config_path: str = ""
    base_harness_tag: str = ""       # opaque tag the harness layer carries round->round
    dataset_version: str = ""        # frozen dataset the loop must run against
    domain_seeds: list[int] = field(default_factory=list)
    # The round whose result has been fully ingested + checkpointed. Used purely
    # for reporting; the authoritative "next round to run" is ``current_round``.
    last_completed_round: int = 0
    accepted_domain_overrides: dict[str, Any] = field(default_factory=dict)
    version_history: list[VersionRecord] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    n_domain_accepted: int = 0
    n_harness_accepted: int = 0
    last_round_payload_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "arr-1.1-loop-v1",
            "loop_id": self.loop_id,
            "domain": self.domain,
            "max_rounds": self.max_rounds,
            "current_round": self.current_round,
            "status": self.status,
            "baseline_config_path": self.baseline_config_path,
            "base_harness_tag": self.base_harness_tag,
            "dataset_version": self.dataset_version,
            "domain_seeds": list(self.domain_seeds),
            "last_completed_round": self.last_completed_round,
            "accepted_domain_overrides": dict(self.accepted_domain_overrides),
            "version_history": [v.to_dict() for v in self.version_history],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "n_domain_accepted": self.n_domain_accepted,
            "n_harness_accepted": self.n_harness_accepted,
            "last_round_payload_sha256": self.last_round_payload_sha256,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LoopState":
        return cls(
            loop_id=str(d.get("loop_id", "")),
            domain=str(d.get("domain", "")),
            max_rounds=int(d.get("max_rounds", 1)),
            current_round=int(d.get("current_round", 1)),
            status=str(d.get("status", "running")),
            baseline_config_path=str(d.get("baseline_config_path", "")),
            base_harness_tag=str(d.get("base_harness_tag", "")),
            dataset_version=str(d.get("dataset_version", "")),
            domain_seeds=list(d.get("domain_seeds") or []),
            last_completed_round=int(d.get("last_completed_round", 0)),
            accepted_domain_overrides=dict(d.get("accepted_domain_overrides") or {}),
            version_history=[VersionRecord.from_dict(v)
                            for v in (d.get("version_history") or [])],
            started_at=str(d.get("started_at", "")),
            completed_at=str(d.get("completed_at", "")),
            n_domain_accepted=int(d.get("n_domain_accepted", 0)),
            n_harness_accepted=int(d.get("n_harness_accepted", 0)),
            last_round_payload_sha256=str(d.get("last_round_payload_sha256", "")),
        )


def save(state: LoopState, state_path: str | Path) -> None:
    """Atomically persist loop state so a crash cannot corrupt the checkpoint."""
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True,
                              ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load(state_path: str | Path) -> LoopState | None:
    path = Path(state_path)
    if not path.is_file():
        return None
    try:
        return LoopState.from_dict(
            json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        return None


def new_state(*, domain: str, max_rounds: int, baseline_config_path: str,
             base_harness_tag: str = "", loop_id: str = "",
             dataset_version: str = "", domain_seeds: list[int] | tuple[int, ...] = ()
             ) -> LoopState:
    return LoopState(
        loop_id=loop_id or ("loop_" + uuid.uuid4().hex[:12]),
        domain=domain, max_rounds=max_rounds, current_round=1,
        baseline_config_path=baseline_config_path,
        base_harness_tag=base_harness_tag,
        dataset_version=dataset_version,
        domain_seeds=list(domain_seeds),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
