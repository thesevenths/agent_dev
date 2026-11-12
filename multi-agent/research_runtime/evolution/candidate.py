"""ARR 1.1 / F1 -- the unified candidate lifecycle and the domain-model carrier.

The lifecycle is shared by BOTH layers::

    PROPOSED -> CRITIC_APPROVED -> SANDBOXED -> EVALUATED
             -> ACCEPTED | REJECTED | INCONCLUSIVE -> VERSIONED

``VERSIONED`` is only reachable from a terminal decision, so a candidate can
never jump from ``PROPOSED`` straight to a versioned state. The harness carrier
already lives in ``adaptive_runtime.harness_evolution`` and is re-exported by
``harness_evolver``; here we add the ``DomainModelCandidate`` carrier and the
shared status machine.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from adaptive_runtime.evolution_spaces import (
    CandidateType, namespaced_fingerprint,
)

#: Module-level (NOT an Enum member) ordered lifecycle, used by the state
#: machine below. Kept outside the Enum so the enum machinery never treats it as
#: a member or a reserved name.
_LIFECYCLE_ORDER = ("proposed", "critic_approved", "sandboxed", "evaluated",
                   "accepted", "rejected", "inconclusive")


class CandidateStatus(Enum):
    """Strict forward lifecycle for every candidate, regardless of layer."""

    PROPOSED = "proposed"
    CRITIC_APPROVED = "critic_approved"
    SANDBOXED = "sandboxed"
    EVALUATED = "evaluated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    VERSIONED = "versioned"

    def is_terminal(self) -> bool:
        return self in (CandidateStatus.ACCEPTED, CandidateStatus.REJECTED,
                        CandidateStatus.INCONCLUSIVE, CandidateStatus.VERSIONED)

    def can_transition_to(self, other: "CandidateStatus") -> bool:
        """A candidate must move forward; only VERSIONED may follow a terminal
        decision, and it may never be reached from a non-terminal state."""
        if self is other:
            return True
        if other is CandidateStatus.VERSIONED:
            return self.is_terminal()
        try:
            return (_LIFECYCLE_ORDER.index(other.value)
                    >= _LIFECYCLE_ORDER.index(self.value))
        except ValueError:
            return False


@dataclass
class DomainModelCandidate:
    """A candidate that improves the object of study (model / factor / policy).

    It carries a whitelisted, dot-path ``target`` in a specific domain's config
    (e.g. ``model.rollout_horizon`` for Embodied). The baseline value ``old`` is
    filled in at materialisation time from the real baseline config; ``new`` is a
    value within the domain's ``VALUE_RULES``. The fingerprint is NAME-SPACED so
    it can never collide with a harness candidate's ledger entry.
    """

    candidate_id: str
    domain: str
    target: str
    old: Any = None
    new: Any = None
    taxonomy: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""
    status: CandidateStatus = CandidateStatus.PROPOSED
    hyp_fingerprint: str = ""

    def __post_init__(self) -> None:
        self.hyp_fingerprint = namespaced_fingerprint(
            CandidateType.DOMAIN_MODEL, target=self.target, old=self.old,
            new=self.new, taxonomy=self.taxonomy)
        if not self.candidate_id:
            # The fingerprint is name-spaced and may contain '::' / '|' (unsafe
            # as a file/dir segment), so derive a safe slug for artifact ids.
            self.candidate_id = ("dom_"
                                + hashlib.sha1(self.hyp_fingerprint.encode())
                                .hexdigest()[:12])

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": CandidateType.DOMAIN_MODEL.value,
            "domain": self.domain,
            "target": self.target,
            "old": self.old,
            "new": self.new,
            "taxonomy": self.taxonomy,
            "overrides": dict(self.overrides),
            "hypothesis": self.hypothesis,
            "status": self.status.value,
            "hyp_fingerprint": self.hyp_fingerprint,
        }
