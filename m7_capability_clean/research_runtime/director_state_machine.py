"""M4-E2E-01B -- Director state machine (pure, deterministic, auditable).

A minimal state machine that governs the Research Director's lifecycle. It is a
*validator*, not a store: it knows which transitions are legal and rejects the
rest. Persistence of each transition lives in ``research_director.py`` (an
append-only journal); this module only decides *whether* a transition is legal
and *to which* state it leads.

States (spec M4-E2E-01B)
------------------------
    IDLE -> OBJECTIVE_SELECTED -> HYPOTHESIS_READY -> PLAN_READY
         -> AWAITING_APPROVAL -> EXECUTING -> EVALUATING
         -> RECORDING_EXPERIENCE -> COMPLETED
    plus the halted / recovery states:
         REJECTED, FAILED, INTERRUPTED, RECOVERING, STOPPED

Invariants enforced here (spec M4-E2E-01B)
------------------------------------------
  * Illegal state transitions are REJECTED (raise ``IllegalTransitionError``).
  * ``COMPLETED`` and ``REJECTED`` are terminal: no further transition is legal
    (recovery cannot resurrect a completed or explicitly-rejected run).
  * Recovery only RE-ENTERS ``RECOVERING`` from an active or halted-recoverable
    state, then RESOLVES to a concrete active state -- it never grants approval,
    never re-runs a campaign, never bypasses a gate.
  * M4.3-00 / P0-04: a verified, already-finished run may take ONE dedicated
    legal recovery path ``RECOVERING --RECOVER_COMPLETE--> COMPLETED``. This is
    the ONLY way recovery reaches COMPLETED; a generic FAIL event can never
    write a terminal-success state. Eligibility (real Report + non-empty
    experience_refs + real Experience rows) is checked by the reconciliation
    layer BEFORE this event is requested.
  * The machine is side-effect free: it never touches the ResearchService, the
    stores, or the journal. That keeps it trivially testable and deterministic.
"""
from __future__ import annotations

from typing import Iterable


# --------------------------------------------------------------------------- #
# States
# --------------------------------------------------------------------------- #
S_IDLE = "IDLE"
S_OBJECTIVE_SELECTED = "OBJECTIVE_SELECTED"
S_HYPOTHESIS_READY = "HYPOTHESIS_READY"
S_PLAN_READY = "PLAN_READY"
S_AWAITING_APPROVAL = "AWAITING_APPROVAL"
S_EXECUTING = "EXECUTING"
S_EVALUATING = "EVALUATING"
S_RECORDING_EXPERIENCE = "RECORDING_EXPERIENCE"
S_COMPLETED = "COMPLETED"
S_REJECTED = "REJECTED"
S_FAILED = "FAILED"
S_INTERRUPTED = "INTERRUPTED"
S_RECOVERING = "RECOVERING"
S_STOPPED = "STOPPED"

ALL_STATES = (
    S_IDLE, S_OBJECTIVE_SELECTED, S_HYPOTHESIS_READY, S_PLAN_READY,
    S_AWAITING_APPROVAL, S_EXECUTING, S_EVALUATING, S_RECORDING_EXPERIENCE,
    S_COMPLETED, S_REJECTED, S_FAILED, S_INTERRUPTED, S_RECOVERING, S_STOPPED,
)

# Terminal: a run that reached these must not transition further.
TERMINAL_STATES = frozenset({S_COMPLETED, S_REJECTED})
# Halted but a human may choose to recover them.
HALTED_RECOVERABLE = frozenset({S_FAILED, S_INTERRUPTED, S_STOPPED})
# Active states the Director may occupy during a healthy run.
ACTIVE_STATES = frozenset({
    S_IDLE, S_OBJECTIVE_SELECTED, S_HYPOTHESIS_READY, S_PLAN_READY,
    S_AWAITING_APPROVAL, S_EXECUTING, S_EVALUATING, S_RECORDING_EXPERIENCE,
})


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
E_SELECT_OBJECTIVE = "SELECT_OBJECTIVE"
E_PROPOSE_HYPOTHESES = "PROPOSE_HYPOTHESES"
E_CREATE_PLAN = "CREATE_PLAN"
E_REQUEST_APPROVAL = "REQUEST_APPROVAL"
E_EXECUTE = "EXECUTE"
E_COMPARE = "COMPARE"
E_RECORD_EXPERIENCE = "RECORD_EXPERIENCE"
E_FINALIZE = "FINALIZE"
E_REJECT = "REJECT"
E_FAIL = "FAIL"
E_INTERRUPT = "INTERRUPT"
E_STOP = "STOP"
E_ENTER_RECOVERY = "ENTER_RECOVERY"
E_RESOLVE_RECOVERY = "RESOLVE_RECOVERY"
# M4.3-00 / P0-04: the ONE legal recovery -> COMPLETED transition. It is only
# legal from RECOVERING, so E_FAIL / generic halts can never produce COMPLETED.
E_RECOVER_COMPLETE = "RECOVER_COMPLETE"

# Generic halt events map to a single target regardless of the active source.
_HALT_TARGET = {
    E_REJECT: S_REJECTED,
    E_FAIL: S_FAILED,
    E_INTERRUPT: S_INTERRUPTED,
    E_STOP: S_STOPPED,
}


# --------------------------------------------------------------------------- #
# Forward (happy-path) transition table -- explicit and auditable
# --------------------------------------------------------------------------- #
_FORWARD = {
    (S_IDLE, E_SELECT_OBJECTIVE): S_OBJECTIVE_SELECTED,
    (S_OBJECTIVE_SELECTED, E_PROPOSE_HYPOTHESES): S_HYPOTHESIS_READY,
    (S_HYPOTHESIS_READY, E_CREATE_PLAN): S_PLAN_READY,
    (S_PLAN_READY, E_REQUEST_APPROVAL): S_AWAITING_APPROVAL,
    (S_AWAITING_APPROVAL, E_EXECUTE): S_EXECUTING,
    (S_EXECUTING, E_COMPARE): S_EVALUATING,
    (S_EVALUATING, E_RECORD_EXPERIENCE): S_RECORDING_EXPERIENCE,
    (S_RECORDING_EXPERIENCE, E_FINALIZE): S_COMPLETED,
    # Recovery entry -- from any active or halted-recoverable state.
    (S_IDLE, E_ENTER_RECOVERY): S_RECOVERING,
    (S_OBJECTIVE_SELECTED, E_ENTER_RECOVERY): S_RECOVERING,
    (S_HYPOTHESIS_READY, E_ENTER_RECOVERY): S_RECOVERING,
    (S_PLAN_READY, E_ENTER_RECOVERY): S_RECOVERING,
    (S_AWAITING_APPROVAL, E_ENTER_RECOVERY): S_RECOVERING,
    (S_EXECUTING, E_ENTER_RECOVERY): S_RECOVERING,
    (S_EVALUATING, E_ENTER_RECOVERY): S_RECOVERING,
    (S_RECORDING_EXPERIENCE, E_ENTER_RECOVERY): S_RECOVERING,
    (S_FAILED, E_ENTER_RECOVERY): S_RECOVERING,
    (S_INTERRUPTED, E_ENTER_RECOVERY): S_RECOVERING,
    (S_STOPPED, E_ENTER_RECOVERY): S_RECOVERING,
    # Recovery-to-COMPLETED: legal ONLY from RECOVERING. Reconciliation must
    # have already proven a real Report + non-empty experience_refs + real
    # Experience Store rows; the machine only validates the transition itself.
    (S_RECOVERING, E_RECOVER_COMPLETE): S_COMPLETED,
}


class IllegalTransitionError(Exception):
    """Raised when the Director attempts a transition the machine forbids."""

    def __init__(self, event: str, from_state: str, to_state: str | None = None):
        self.event = event
        self.from_state = from_state
        self.to_state = to_state
        msg = f"illegal transition: {from_state} --{event}--> {to_state or '?'}"
        super().__init__(msg)


class DirectorStateMachine:
    """Deterministic validator for Director transitions.

    It is stateless: callers own the current state and pass it in. The only
    stateful-looking method, ``resolve_recovery``, is pure -- it returns the
    resolved target without touching anything.
    """

    def __init__(self, *, transitions: dict | None = None,
                 halt_target: dict | None = None):
        self._forward = dict(_FORWARD if transitions is None else transitions)
        self._halt = dict(_HALT_TARGET if halt_target is None else halt_target)

    # -- queries ----------------------------------------------------------
    def can(self, event: str, from_state: str) -> bool:
        return self.next(event, from_state) is not None

    def is_terminal(self, state: str) -> bool:
        return state in TERMINAL_STATES

    def is_active(self, state: str) -> bool:
        return state in ACTIVE_STATES

    def is_halted_recoverable(self, state: str) -> bool:
        return state in HALTED_RECOVERABLE

    def legal_targets(self, from_state: str) -> list[str]:
        """All states reachable from ``from_state`` (for diagnostics)."""
        out: list[str] = []
        for ev in (E_SELECT_OBJECTIVE, E_PROPOSE_HYPOTHESES, E_CREATE_PLAN,
                   E_REQUEST_APPROVAL, E_EXECUTE, E_COMPARE,
                   E_RECORD_EXPERIENCE, E_FINALIZE, E_REJECT, E_FAIL,
                   E_INTERRUPT, E_STOP, E_ENTER_RECOVERY, E_RESOLVE_RECOVERY,
                   E_RECOVER_COMPLETE):
            t = self._raw_next(ev, from_state)
            if t is not None:
                out.append(t)
        return out

    # -- core -------------------------------------------------------------
    def next(self, event: str, from_state: str,
             *, recovery_target: str | None = None) -> str | None:
        """Return the target state for ``event`` from ``from_state``.

        Returns ``None`` if the transition is illegal. ``RESOLVE_RECOVERY``
        requires ``recovery_target`` (the concrete active state the recovery
        logic determined from the actual Campaign/Session status).
        """
        if from_state not in ALL_STATES:
            raise IllegalTransitionError(event, from_state,
                                         to_state="<unknown-state>")
        t = self._raw_next(event, from_state, recovery_target=recovery_target)
        return t

    def _raw_next(self, event: str, from_state: str,
                  *, recovery_target: str | None = None) -> str | None:
        # Terminal states reject everything.
        if from_state in TERMINAL_STATES and event != E_RESOLVE_RECOVERY:
            return None
        # Explicit forward / recovery-entry table first.
        if (from_state, event) in self._forward:
            return self._forward[(from_state, event)]
        # Generic halt events from any non-terminal state.
        if event in self._halt and from_state not in TERMINAL_STATES:
            return self._halt[event]
        # Resolution of a recovery: only valid while RECOVERING, and only to a
        # concrete active state (never to a terminal/halted one -- recovery does
        # not conclude the run, it only resumes it at the correct point).
        if event == E_RESOLVE_RECOVERY:
            if from_state != S_RECOVERING:
                return None
            if recovery_target in ACTIVE_STATES:
                return recovery_target
            return None
        return None

    # -- convenience guard ------------------------------------------------
    def require(self, event: str, from_state: str,
                *, recovery_target: str | None = None) -> str:
        """Like ``next`` but raise ``IllegalTransitionError`` on ``None``."""
        target = self.next(event, from_state, recovery_target=recovery_target)
        if target is None:
            raise IllegalTransitionError(event, from_state)
        return target


def describe_states() -> dict[str, list[str]]:
    """Static description for diagnostics / documentation."""
    return {
        "states": list(ALL_STATES),
        "terminal": sorted(TERMINAL_STATES),
        "halted_recoverable": sorted(HALTED_RECOVERABLE),
        "active": sorted(ACTIVE_STATES),
        "events": [
            E_SELECT_OBJECTIVE, E_PROPOSE_HYPOTHESES, E_CREATE_PLAN,
            E_REQUEST_APPROVAL, E_EXECUTE, E_COMPARE, E_RECORD_EXPERIENCE,
            E_FINALIZE, E_REJECT, E_FAIL, E_INTERRUPT, E_STOP,
            E_ENTER_RECOVERY, E_RESOLVE_RECOVERY, E_RECOVER_COMPLETE,
        ],
    }


__all__ = [
    "DirectorStateMachine", "IllegalTransitionError",
    "describe_states",
    "ALL_STATES", "TERMINAL_STATES", "HALTED_RECOVERABLE", "ACTIVE_STATES",
    "S_IDLE", "S_OBJECTIVE_SELECTED", "S_HYPOTHESIS_READY", "S_PLAN_READY",
    "S_AWAITING_APPROVAL", "S_EXECUTING", "S_EVALUATING",
    "S_RECORDING_EXPERIENCE", "S_COMPLETED", "S_REJECTED", "S_FAILED",
    "S_INTERRUPTED", "S_RECOVERING", "S_STOPPED",
    "E_SELECT_OBJECTIVE", "E_PROPOSE_HYPOTHESES", "E_CREATE_PLAN",
    "E_REQUEST_APPROVAL", "E_EXECUTE", "E_COMPARE", "E_RECORD_EXPERIENCE",
    "E_FINALIZE", "E_REJECT", "E_FAIL", "E_INTERRUPT", "E_STOP",
    "E_ENTER_RECOVERY", "E_RESOLVE_RECOVERY", "E_RECOVER_COMPLETE",
]
