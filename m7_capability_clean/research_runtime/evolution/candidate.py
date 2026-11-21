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

#: Module-level (NOT an Enum member) ordered lifecycle, kept for reference and
#: reporting. The actual state machine is the explicit whitelist below, NOT this
#: ordering: comparison against it would wrongly allow skipping stages and
#: reversing terminal decisions.
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
        """Return True iff ``other`` is an explicitly permitted next state.

        A candidate must move through the lifecycle in order; only VERSIONED may
        follow a terminal decision, and it is a sink (no outgoing edges). This is
        enforced by the :data:`_TRANSITIONS` whitelist -- not by lifecycle index
        ordering, which would have allowed skipping stages or reversing terminal
        decisions.
        """
        if self is other:
            return True
        return other in _TRANSITIONS.get(self, set())


#: Explicit legal transitions. Declared AFTER the enum so the names resolve. Any
#: transition not listed here is FORBIDDEN, so a candidate can never skip a
#: mandatory stage (e.g. PROPOSED -> EVALUATED) nor reverse a terminal decision
#: (e.g. ACCEPTED -> REJECTED). VERSIONED is a true sink reachable exactly once,
#: only from a terminal decision.
_TRANSITIONS: dict["CandidateStatus", set["CandidateStatus"]] = {
    CandidateStatus.PROPOSED: {CandidateStatus.CRITIC_APPROVED},
    CandidateStatus.CRITIC_APPROVED: {CandidateStatus.SANDBOXED},
    CandidateStatus.SANDBOXED: {CandidateStatus.EVALUATED},
    CandidateStatus.EVALUATED: {CandidateStatus.ACCEPTED,
                               CandidateStatus.REJECTED,
                               CandidateStatus.INCONCLUSIVE},
    CandidateStatus.ACCEPTED: {CandidateStatus.VERSIONED},
    CandidateStatus.REJECTED: {CandidateStatus.VERSIONED},
    CandidateStatus.INCONCLUSIVE: {CandidateStatus.VERSIONED},
    CandidateStatus.VERSIONED: set(),
}


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
