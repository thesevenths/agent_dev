"""ARR 1.1 / F1 -- append-only candidate registry + audit log.

One registry tracks BOTH layers. Every register / transition is written to an
append-only JSONL so the audit trail is reconstructable and a candidate can
never silently change status. Frozen targets (the evaluation substrate) are
rejected at registration time -- candidates may only modify whitelisted model /
training / policy knobs, never the gate, the evaluator or the data split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .candidate import CandidateStatus, DomainModelCandidate

from adaptive_runtime.evolution_spaces import (
    CandidateType, is_frozen, namespaced_fingerprint,
)


@dataclass
class RegistryEntry:
    candidate_id: str
    candidate_type: str
    domain: str
    target: str
    status: CandidateStatus
    fingerprint: str
    record: dict[str, Any] = field(default_factory=dict)


class CandidateRegistry:
    """Tracks candidate lifecycle state across both RRSI layers."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.entries: dict[str, RegistryEntry] = {}
        self.log_path = Path(log_path) if log_path else None

    def register(self, candidate: DomainModelCandidate,
                 *, status: CandidateStatus | None = None) -> RegistryEntry:
        ctype = CandidateType.DOMAIN_MODEL
        if is_frozen(ctype, candidate.target):
            raise ValueError(
                f"target {candidate.target!r} is frozen; candidates must not "
                f"modify the evaluation substrate")
        st = status or candidate.status
        entry = RegistryEntry(
            candidate_id=candidate.candidate_id,
            candidate_type=ctype.value,
            domain=candidate.domain,
            target=candidate.target,
            status=st,
            fingerprint=candidate.hyp_fingerprint,
        )
        self.entries[candidate.hyp_fingerprint] = entry
        self._append({"event": "register",
                      "fingerprint": candidate.hyp_fingerprint,
                      "candidate_type": ctype.value,
                      "domain": candidate.domain,
                      "target": candidate.target,
                      "status": st.value})
        return entry

    def transition(self, fingerprint: str, to: CandidateStatus,
                   *, record: dict[str, Any] | None = None) -> RegistryEntry:
        entry = self.entries.get(fingerprint)
        if entry is None:
            raise KeyError(f"unknown candidate {fingerprint!r}")
        if not entry.status.can_transition_to(to):
            raise ValueError(
                f"illegal transition {entry.status.value} -> {to.value} for "
                f"{fingerprint!r}")
        entry.status = to
        if record is not None:
            entry.record = record
        self._append({"event": "transition", "fingerprint": fingerprint,
                      "to": to.value, "record": record})
        return entry

    def by_status(self, *statuses: CandidateStatus) -> list[RegistryEntry]:
        wanted = set(statuses)
        return [e for e in self.entries.values() if e.status in wanted]

    def _append(self, obj: dict[str, Any]) -> None:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
