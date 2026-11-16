"""M6 -- versioned research-method benchmark with sealed splits.

This is an ENGINEERING benchmark of research tasks, not a claim of statistical
sufficiency. It implements the M6 task contract:

  * every task has a reproducible source, domain, family id, allowed tools,
    budget, seed, primary metric and grading specification;
  * dev / selection / holdout are assigned by TASK FAMILY, not by row, so a
    family seen during selection can never appear in holdout;
  * gold standards and rubrics are SEALED: they are written to a separate
    immutable artifact and are absent from the generator-visible view;
  * a contamination ledger records exactly which tasks a candidate generator
    was allowed to see, so reuse of a holdout is detectable.

The module never runs an engine and never scores a candidate. It only defines
and validates the immutable task space.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .identity import freeze_artifact, freeze_json, sha256_json, verify_frozen

# Domain -> families. Selection and holdout families are DISJOINT by design.
DOMAIN_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "scinet": {
        "all": ("control_quality", "ood_robustness", "tta_efficiency"),
        "selection": ("control_quality", "ood_robustness"),
        "holdout": ("tta_efficiency",),
    },
    "embodied": {
        "all": ("dynamics_accuracy", "safety_compliance", "transfer_robustness"),
        "selection": ("dynamics_accuracy", "safety_compliance"),
        "holdout": ("transfer_robustness",),
    },
    "quant": {
        "all": ("signal_validity", "cost_efficiency", "regime_robustness"),
        "selection": ("signal_validity", "cost_efficiency"),
        "holdout": ("regime_robustness",),
    },
}

DEFAULT_SPLIT_SIZES = {"dev": 36, "selection": 12, "holdout": 12}
PRIMARY_METRIC_BY_DOMAIN = {
    "scinet": "horizon_mse_reduction",
    "embodied": "state_mse_reduction",
    "quant": "risk_adjusted_return",
}
# Pre-registered smallest effects of interest. Fixed BEFORE any result is seen;
# a metric absent here falls back to 0.01 rather than being tuned from outcomes.
SEOI_BY_METRIC: dict[str, float] = {
    "horizon_mse_reduction": 0.01,
    "state_mse_reduction": 0.01,
    "risk_adjusted_return": 0.005,
    "objective_quality": 0.10,
    "falsifiability_rate": 0.05,
    "experiment_validity": 0.10,
    "diagnosis_accuracy": 0.10,
}
BENCHMARK_VERSION = "m6_bench_v1"


class BenchmarkError(ValueError):
    """Raised for an invalid or contaminated benchmark definition."""


@dataclass(frozen=True)
class TaskSpec:
    """One reproducible research task. No gold/rubric content lives here."""
    task_id: str
    domain: str
    family_id: str
    split: str
    source: dict[str, Any]
    allowed_tools: tuple[str, ...]
    budget: dict[str, Any]
    seed: int
    primary_metric: str
    grading_spec: dict[str, Any]
    # Sealed references are opaque ids; the answers are in the sealed artifact.
    gold_ref: str = ""
    rubric_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["allowed_tools"] = list(self.allowed_tools)
        return d

    def generation_view(self) -> dict[str, Any]:
        """The ONLY view a candidate generator may receive (dev split only).

        Drops sealed references and grading thresholds that could leak the
        evaluation answer. Failure evidence from dev runs remains allowed.
        """
        d = self.to_dict()
        d.pop("gold_ref", None)
        d.pop("rubric_ref", None)
        d["grading_spec"] = {"kind": self.grading_spec.get("kind", "deterministic")}
        return d


def _source_for(domain: str, family: str, seed: int) -> dict[str, Any]:
    """Point each task at a REAL, reproducible engine configuration."""
    if domain == "scinet":
        dataset = ("pendulum_smoke_v1" if family == "control_quality"
                   else "controlled_pendulum_pilot" if family == "ood_robustness"
                   else "pendulum_smoke_ood_v1")
        return {"engine": "scinet", "dataset": dataset,
                "config": "configs/pendulum_smoke.yaml",
                "entry": "research_api.cli", "split": "test", "seed": seed}
    if domain == "embodied":
        return {"engine": "embodied", "project": "research_runtime/domains/projects/embodied",
                "config": "config.yaml", "entry": "local_engine",
                "split": "holdout" if "robustness" in family else "validation",
                "seed": seed}
    return {"engine": "quant", "project": "research_runtime/domains/projects/quant",
            "config": "config.yaml", "entry": "local_engine",
            "split": "validation", "seed": seed,
            "data_note": "synthetic causal data only; NOT alpha evidence"}


def build_tasks(version: str = BENCHMARK_VERSION,
                split_sizes: dict[str, int] | None = None) -> list[TaskSpec]:
    """Deterministically build the full task set with family-disjoint splits."""
    sizes = dict(DEFAULT_SPLIT_SIZES if split_sizes is None else split_sizes)
    tasks: list[TaskSpec] = []
    counter = 0
    for domain, fams in DOMAIN_FAMILIES.items():
        per_domain = {s: max(1, n // 3) for s, n in sizes.items()}
        for split in ("dev", "selection", "holdout"):
            pool = (fams["all"] if split == "dev"
                    else fams["selection"] if split == "selection"
                    else fams["holdout"])
            for i in range(per_domain[split]):
                family = pool[i % len(pool)]
                counter += 1
                seed = 6000 + counter
                tid = f"{version}_{domain}_{split}_{i:03d}"
                tasks.append(TaskSpec(
                    task_id=tid, domain=domain, family_id=f"{domain}::{family}",
                    split=split, source=_source_for(domain, family, seed),
                    allowed_tools=("research_api", "local_engine", "evidence_store"),
                    budget={"max_campaign_rounds": 1, "max_calls": 4,
                            "max_wall_s": 900},
                    seed=seed,
                    primary_metric=PRIMARY_METRIC_BY_DOMAIN[domain],
                    grading_spec={"kind": "deterministic", "metric":
                                  PRIMARY_METRIC_BY_DOMAIN[domain],
                                  "direction": "maximize"},
                    gold_ref=f"gold::{tid}", rubric_ref=f"rubric::{tid}"))
    return tasks


@dataclass
class BenchmarkSpec:
    """Immutable benchmark definition plus SEALED gold artifact reference."""
    version: str
    tasks: list[TaskSpec]
    sealed: dict[str, Any] = field(default_factory=dict)
    manifest_sha: str = ""
    manifest_path: str = ""

    def by_split(self, split: str) -> list[TaskSpec]:
        return [t for t in self.tasks if t.split == split]

    def families(self, split: str) -> set[str]:
        return {t.family_id for t in self.by_split(split)}

    def generator_view(self) -> dict[str, Any]:
        """Only dev tasks, with sealed answers stripped."""
        return {"version": self.version,
                "tasks": [t.generation_view() for t in self.by_split("dev")],
                "note": "selection/holdout tasks and gold are intentionally absent"}

    def promotable_view(self) -> dict[str, Any]:
        """Task ids + families used by the strict promotion gate (no content)."""
        return {"version": self.version,
                "manifest_sha256": self.manifest_sha,
                "selection_task_ids": [t.task_id for t in self.by_split("selection")],
                "holdout_task_ids": [t.task_id for t in self.by_split("holdout")],
                "selection_families": sorted(self.families("selection")),
                "holdout_families": sorted(self.families("holdout"))}


def build_benchmark(out_dir: str | Path, *, version: str = BENCHMARK_VERSION,
                    split_sizes: dict[str, int] | None = None) -> BenchmarkSpec:
    """Build + freeze a benchmark and its sealed gold artifact to disk."""
    root = Path(out_dir)
    tasks = build_tasks(version=version, split_sizes=split_sizes)
    # Gold answers are written to a SEPARATE sealed artifact. Task specs keep
    # only opaque references, so serialising the manifest cannot leak answers.
    sealed_obj = {
        "version": version,
        "gold": {t.task_id: {"expected_direction": t.grading_spec["direction"],
                             "primary_metric": t.primary_metric,
                             "minimum_meaningful_effect": 0.01,
                             "family": t.family_id}
                 for t in tasks},
        "rubric": {t.task_id: {"dimensions": ["evidence", "falsifiability",
                                              "validity", "diagnosis"],
                               "scale": [0, 1, 2, 3]}
                   for t in tasks},
        "warning": "NEVER expose to candidate generators or evaluated harnesses",
    }
    sealed_dir = root / "sealed"
    sealed = freeze_artifact(sealed_obj, sealed_dir / "gold_rubric.json")
    manifest_obj = {
        "version": version,
        "schema": "m6_benchmark_manifest_v1",
        "split_sizes": {s: len([t for t in tasks if t.split == s])
                        for s in ("dev", "selection", "holdout")},
        "family_sets": {s: sorted({t.family_id for t in tasks if t.split == s})
                        for s in ("dev", "selection", "holdout")},
        "tasks": [t.to_dict() for t in tasks],
        "sealed_ref": {"path": sealed["path"], "sha256": sealed["sha256"]},
    }
    frozen = freeze_json(manifest_obj, root / "benchmark_manifest.json")
    # Bind each task to the manifest that defines it, for evidence SHA refs.
    tasks = [TaskSpec(**{**t.to_dict(),
                         "source": {**t.source,
                                    "manifest_sha": frozen["sha256"]}})
             for t in tasks]
    spec = BenchmarkSpec(version=version, tasks=tasks, sealed=sealed,
                         manifest_sha=frozen["sha256"],
                         manifest_path=frozen["path"])
    validate_benchmark(spec)
    return spec


def validate_benchmark(spec: BenchmarkSpec) -> dict[str, Any]:
    """Fail closed on overlap, missing fields, or tampered sealed artifact."""
    problems: list[str] = []
    ids = [t.task_id for t in spec.tasks]
    if len(ids) != len(set(ids)):
        problems.append("duplicate_task_ids")
    sel, hold = spec.families("selection"), spec.families("holdout")
    if sel & hold:
        problems.append("family_overlap_selection_holdout")
    if set(t.task_id for t in spec.by_split("selection")) & set(
            t.task_id for t in spec.by_split("holdout")):
        problems.append("task_overlap_selection_holdout")
    for t in spec.tasks:
        if not t.source.get("engine"):
            problems.append(f"missing_source:{t.task_id}")
        if not t.allowed_tools:
            problems.append(f"missing_allowed_tools:{t.task_id}")
        if t.budget.get("max_calls", 0) <= 0:
            problems.append(f"invalid_budget:{t.task_id}")
        if not t.seed or not t.primary_metric:
            problems.append(f"missing_seed_or_metric:{t.task_id}")
    if spec.sealed.get("sha256"):
        check = verify_frozen(spec.sealed["path"], spec.sealed["sha256"])
        if not check["ok"]:
            problems.extend("sealed_" + r for r in check["reasons"])
    return {"ok": not problems, "reasons": problems,
            "n_tasks": len(spec.tasks), "families_selection": sorted(sel),
            "families_holdout": sorted(hold)}


class ContaminationLedger:
    """Records which task ids were visible to which harness/candidate."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = []

    def grant(self, *, harness_sha: str, candidate_sha: str,
              visible_task_ids: list[str], phase: str) -> dict[str, Any]:
        entry = {"harness_sha": harness_sha, "candidate_sha": candidate_sha,
                 "visible_task_ids": sorted(visible_task_ids),
                 "phase": phase, "task_sha": sha256_json(sorted(visible_task_ids))}
        self._entries.append(entry)
        self._flush()
        return entry

    def assert_clean(self, *, candidate_sha: str,
                     holdout_task_ids: list[str]) -> dict[str, Any]:
        seen: set[str] = set()
        for e in self._entries:
            if e["candidate_sha"] == candidate_sha:
                seen |= set(e["visible_task_ids"])
        leaked = seen & set(holdout_task_ids)
        return {"ok": not leaked, "leaked_task_ids": sorted(leaked),
                "n_visible": len(seen)}

    def _flush(self) -> None:
        freeze_json({"entries": self._entries}, self.path)

    def all_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)


def pilot_power_analysis(paired_differences: dict[str, list[float]],
                         *, alpha: float = 0.05, target_power: float = 0.8
                         ) -> dict[str, Any]:
    """Descriptive pilot power via the project's frozen sample-size routine."""
    from ..m4_3.sample_size import ContrastSpec, estimate_sample_size
    try:
        contrasts = [ContrastSpec(name=k, seoi=SEOI_BY_METRIC.get(k, 0.01))
                     for k in paired_differences]
        est = estimate_sample_size(paired_differences=paired_differences,
                                   contrasts=contrasts, alpha=alpha,
                                   target_power=target_power)
        return {"ok": True, "estimate": est,
                "interpretation": "pilot estimate only; not a confirmatory claim"}
    except Exception as exc:  # noqa: BLE001 - report honestly
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "BenchmarkError", "TaskSpec", "BenchmarkSpec", "DOMAIN_FAMILIES",
    "BENCHMARK_VERSION", "build_tasks", "build_benchmark", "validate_benchmark",
    "ContaminationLedger", "pilot_power_analysis",
]
