"""P1-04: deterministic execution order and deterministic job identities.

Seed-OUTER ordering is deliberate:

    baseline(42), candA(42), candB(42), baseline(43), candA(43), ...

A failed baseline for seed 42 lets us skip ALL candidates of seed 42 before
spending GPU time on unpaired runs, while seed 43 continues normally.

``make_job_id`` is UUID5 over (campaign_id, trial_id, stage). It is the single
most important recovery primitive: after a process crash the campaign can
recompute the exact same id and ask the engine whether that job already exists,
instead of minting a second training run.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

# Fixed namespace: job ids are stable across processes, hosts and restarts.
JOB_NAMESPACE = uuid.UUID("a6c15c6b-95a1-4b3d-9a0d-1d5c8d2c7b01")


def make_job_id(campaign_id: str, trial_id: str, stage: str, *,
                namespace: uuid.UUID = JOB_NAMESPACE) -> str:
    """Idempotent job identity. Never use a random uuid for a retry."""
    return uuid.uuid5(namespace, f"{campaign_id}:{trial_id}:{stage}").hex


def trial_id_for(arm_id: str, seed: int) -> str:
    return f"{arm_id}_seed_{seed}"


@dataclass(frozen=True)
class ScheduleItem:
    seed: int
    arm_type: Literal["baseline", "candidate"]
    arm_id: str
    trial_id: str
    hypothesis_id: str | None = None
    candidate_id: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


def build_schedule(seeds: Sequence[int],
                   candidates: Sequence[Any]) -> list[ScheduleItem]:
    """Pure function: no clock, no I/O, no engine assumptions.

    ``candidates`` may be pydantic CandidateExperiment objects or any object
    exposing ``candidate_id``, ``hypothesis_id`` and ``config_overrides``.
    """
    out: list[ScheduleItem] = []
    for seed in seeds:
        out.append(ScheduleItem(
            seed=seed, arm_type="baseline", arm_id="baseline",
            trial_id=trial_id_for("baseline", seed),
        ))
        for candidate in candidates:
            candidate_id = str(candidate.candidate_id)
            overrides = dict(candidate.config_overrides)
            out.append(ScheduleItem(
                seed=seed, arm_type="candidate", arm_id=candidate_id,
                trial_id=trial_id_for(candidate_id, seed),
                hypothesis_id=getattr(candidate, "hypothesis_id", None),
                candidate_id=candidate_id,
                overrides=overrides,
            ))
    return out
