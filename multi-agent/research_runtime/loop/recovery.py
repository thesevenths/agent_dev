"""ARR 1.1 / G1 -- crash recovery for the autonomous research loop.

The loop never trusts in-memory state across a process boundary: it persists a
checkpoint after EVERY completed round (see :mod:`loop_state`). On restart this
module decides whether to resume an interrupted run or start fresh.

Resumption semantics are deliberately conservative:
  * A checkpoint with ``status == "running"`` and ``current_round <= max_rounds``
    is resumed. The next round to run is exactly ``current_round`` -- the round
    that was NEVER checkpointed, so no completed work is repeated and no pending
    round is silently dropped.
  * A checkpoint that is already ``completed`` / ``failed`` is NOT re-run; the
    caller gets the finished state back and may start a new loop instead.
  * A missing or unreadable checkpoint starts a fresh loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .loop_state import LoopState, load, new_state, save


class LoopRecoveryError(Exception):
    """Raised when a checkpoint is present but cannot be safely resumed."""


def _assert_compatible(existing: LoopState, config: dict[str, Any],
                       state_path: str | Path) -> None:
    """Raise if a resume would run a DIFFERENT task on a prior run's checkpoint.

    Comparing the requested task config against what was checkpointed prevents a
    misleading "resume" that silently reuses another domain's / dataset's /
    baseline's / seed-set's state. On mismatch the old checkpoint is archived (so
    no evidence is destroyed) and a clear error is raised.
    """
    checks = [
        ("domain", existing.domain, str(config.get("domain", ""))),
        ("baseline_config_path", existing.baseline_config_path,
         str(config.get("baseline_config_path", ""))),
        ("max_rounds", existing.max_rounds, int(config.get("max_rounds", 1))),
        ("dataset_version", existing.dataset_version,
         str(config.get("dataset_version", ""))),
        ("domain_seeds", list(existing.domain_seeds),
         list(config.get("domain_seeds") or ())),
    ]
    mismatches = [f"{k}: checkpoint={a!r} != requested={b!r}"
                  for k, a, b in checks if a != b]
    if mismatches:
        _archive(state_path)
        raise LoopRecoveryError(
            "refusing to resume with mismatched configuration: "
            + "; ".join(mismatches))


def load_or_init(*, state_path: str | Path, config: dict[str, Any]
                 ) -> tuple[LoopState, bool]:
    """Return ``(state, resumed)``.

    ``resumed`` is True iff an existing running checkpoint was loaded and will be
    continued. A fresh state is persisted immediately so the very first round has
    a checkpoint on disk before any work happens.
    """
    existing = load(state_path)
    if existing is not None:
        if existing.status == "running" and existing.current_round <= existing.max_rounds:
            # Resume only if the requested task matches the checkpoint exactly.
            _assert_compatible(existing, config, state_path)
            return existing, True
        if existing.status in ("completed", "failed"):
            # Finished loop: do not re-run. Caller decides what to do next.
            return existing, False
        # Unknown / inconsistent status: treat as a fresh start but keep the old
        # file renamed so we never silently discard evidence of a prior run.
        _archive(state_path)
    state = new_state(
        domain=str(config.get("domain", "")),
        max_rounds=int(config.get("max_rounds", 1)),
        baseline_config_path=str(config.get("baseline_config_path", "")),
        base_harness_tag=str(config.get("base_harness_tag", "")),
        loop_id=str(config.get("loop_id", "")),
        dataset_version=str(config.get("dataset_version", "")),
        domain_seeds=list(config.get("domain_seeds") or ()),
    )
    save(state, state_path)
    return state, False


def _archive(state_path: str | Path) -> None:
    path = Path(state_path)
    if not path.is_file():
        return
    n = 1
    while True:
        cand = path.with_name(f"{path.stem}.archived{n}{path.suffix}")
        if not cand.exists():
            path.replace(cand)
            return
        n += 1


def mark_completed(state: LoopState, state_path: str | Path) -> LoopState:
    state.status = "completed"
    state.completed_at = __now()
    save(state, state_path)
    return state


def mark_failed(state: LoopState, state_path: str | Path) -> LoopState:
    state.status = "failed"
    state.completed_at = __now()
    save(state, state_path)
    return state


def __now() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
