"""M6 -- four kinds of research Experience + retrieval ablation.

Experience is the memory of the RESEARCHER, not of the model under study. Four
kinds are tracked and each record links to its original run, task ids, evidence
and artifact SHAs:

  SUCCESS     -- what worked and under which conditions
  FAILURE     -- what failed and the traced root cause
  IMPROVEMENT -- a candidate method that was verified better
  TRANSFER    -- whether a lesson carried to a different task family/domain

The Experience Retrieval Ablation runs the SAME harness on matched tasks with
retrieval enabled vs disabled, records which experiences were injected (behaviour
change) and compares paired performance. "No experience" is never simulated as
a synthetic success -- it is a real arm of the experiment.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .identity import freeze_json, sha256_json

EXPERIENCE_KINDS = ("SUCCESS", "FAILURE", "IMPROVEMENT", "TRANSFER")


class ExperienceError(ValueError):
    """Raised for malformed or unlinked experience records."""


@dataclass
class ResearchExperience:
    experience_id: str
    kind: str
    problem: str
    claim: str
    component: str
    source_run_id: str
    source_task_ids: list[str] = field(default_factory=list)
    evidence_shas: list[str] = field(default_factory=list)
    artifact_shas: list[str] = field(default_factory=list)
    source_experience_ids: list[str] = field(default_factory=list)
    target_family: str = ""
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: time.strftime(
        "%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["experience_sha256"] = sha256_json(d)
        return d

    def validate(self) -> dict[str, Any]:
        problems = []
        if self.kind not in EXPERIENCE_KINDS:
            problems.append("invalid_kind")
        if not self.source_run_id:
            problems.append("missing_source_run")
        if not self.source_task_ids:
            problems.append("missing_source_tasks")
        if not self.evidence_shas and not self.artifact_shas:
            problems.append("missing_evidence_or_artifact_sha")
        return {"ok": not problems, "reasons": problems}


class ExperienceBank:
    """Freezes all M6 experience records to one auditable artifact."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, ResearchExperience] = {}

    def add(self, exp: ResearchExperience) -> dict[str, Any]:
        v = exp.validate()
        if not v["ok"]:
            raise ExperienceError("|".join(v["reasons"]))
        self._records[exp.experience_id] = exp
        return self._flush()

    def _flush(self) -> dict[str, Any]:
        obj = {"version": "m6_experience_v1",
               "records": [e.to_dict() for e in self._records.values()]}
        frozen = freeze_json(obj, self.path)
        return {"file_sha256": frozen["sha256"], "n_records": len(self._records)}

    def all(self) -> list[ResearchExperience]:
        return list(self._records.values())

    def retrieve(self, *, problem: str, enabled: bool = True,
                 top: int = 5, kind: str | None = None) -> list[dict[str, Any]]:
        """Retrieve relevant experiences; disabled returns NOTHING (real arm)."""
        if not enabled:
            return []
        tokens = [t for t in problem.lower().replace("_", " ").split()
                  if len(t) >= 4]
        scored: list[tuple[int, ResearchExperience]] = []
        for exp in self._records.values():
            if kind and exp.kind != kind:
                continue
            text = f"{exp.problem} {exp.claim} {exp.component}".lower()
            score = sum(1 for t in tokens if t in text)
            if score:
                scored.append((score, exp))
        scored.sort(key=lambda se: (-se[0], se[1].experience_id))
        return [e.to_dict() for _, e in scored[:top]]


def run_retrieval_ablation(*, bank: ExperienceBank, problem: str,
                           harness: Any, tasks: list[Any],
                           runner: Any, out_dir: str | Path) -> dict[str, Any]:
    """Run the SAME harness twice with retrieval on/off; compare behaviour.

    ``runner`` is the same real engine adapter used elsewhere. It is called
    with an explicit ``experience_context`` so the behavioural difference is
    observable and auditable.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    injected = bank.retrieve(problem=problem, enabled=True)
    arms: dict[str, Any] = {}
    for arm, enabled in (("with_retrieval", True), ("without_retrieval", False)):
        context = bank.retrieve(problem=problem, enabled=enabled)
        results = []
        for task in tasks:
            raw = runner({"harness": harness, "experience_context": context,
                          "arm": arm}, task)
            results.append({"task_id": getattr(task, "task_id", ""),
                            "arm": arm, "injected_experience_ids":
                            [e["experience_id"] for e in context],
                            "result": raw})
        frozen = freeze_json({"arm": arm, "n_injected": len(context),
                              "results": results}, root / f"{arm}.json")
        arms[arm] = {"n_injected": len(context),
                     "result_sha256": frozen["sha256"],
                     "results": results}
    behavior_changed = (arms["with_retrieval"]["n_injected"] > 0
                        and arms["without_retrieval"]["n_injected"] == 0
                        and any(r["injected_experience_ids"]
                                for r in arms["with_retrieval"]["results"]))
    return {"problem": problem, "arms": arms,
            "behavior_changed": behavior_changed,
            "n_experience_available": len(injected),
            "note": "performance improvement requires separate paired evaluation"}


__all__ = [
    "ExperienceError", "EXPERIENCE_KINDS", "ResearchExperience",
    "ExperienceBank", "run_retrieval_ablation",
]
