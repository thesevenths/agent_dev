"""M6.1 -- Benchmark V2: three-way DISJOINT research-method benchmark.

The M6 v1 benchmark had a real P0: the dev split drew its families from
``fams["all"]`` which INCLUDED the selection and holdout families, so a family
seen during candidate generation could reappear in the holdout. V1 artifacts
are preserved for reproducibility; V2 is a NEW benchmark with:

  * dedicated, pairwise-disjoint family pools for dev / selection / holdout;
  * a per-task content SHA (``task_sha``) and per-split manifest SHAs;
  * a HOLDOUT ACCESS VERIFIER: every holdout read requires an explicit,
    preregistered purpose, is appended to a hash-chained ledger, and is
    checked before formal promotion;
  * family-level seeds inside a family (seed variation does NOT create a new
    family -- the family is the statistical unit).

No family exists only on paper: each maps to a REAL engine source (dataset /
config / allowed config override) and a distinct research objective.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .identity import freeze_artifact, freeze_json, sha256_json, verify_frozen

BENCHMARK_VERSION = "m6_bench_v2"
SPLITS = ("dev", "selection", "holdout")


class BenchmarkIsolationError(ValueError):
    """Raised on any family/task overlap or unauthorized holdout access."""


@dataclass(frozen=True)
class FamilySpec:
    """One REAL research family: distinct objective + executable source."""
    family_id: str
    domain: str
    objective: str
    source: dict[str, Any]
    primary_metric: str = ""
    task_config: dict[str, Any] = field(default_factory=dict)
    direction: str = "maximize"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scinet(ds: str, cfg: str = "configs/pendulum_smoke.yaml",
            split: str = "test") -> dict[str, Any]:
    return {"engine": "scinet", "dataset": ds, "config": cfg,
            "entry": "research_api.cli", "split": split}


def _embodied(objective: str, override: dict[str, Any],
              split: str = "validation") -> dict[str, Any]:
    return {"engine": "embodied",
            "project": "research_runtime/domains/projects/embodied",
            "config": "config.yaml", "entry": "local_engine", "split": split,
            "research_objective": objective, "task_config": dict(override)}


def _quant(objective: str, override: dict[str, Any],
           split: str = "validation") -> dict[str, Any]:
    return {"engine": "quant",
            "project": "research_runtime/domains/projects/quant",
            "config": "config.yaml", "entry": "local_engine", "split": split,
            "research_objective": objective, "task_config": dict(override),
            "data_class": "SYNTHETIC_RESEARCH"}


# Three pairwise-disjoint family pools per domain. Family identity is defined
# by a distinct research objective + a REAL executable source, never by seed.
FAMILIES: dict[str, dict[str, tuple[FamilySpec, ...]]] = {
    "scinet": {
        "dev": (
            FamilySpec("scinet::dev_pendulum_control", "scinet",
                       "short-horizon pendulum control quality",
                       _scinet("pendulum_smoke_v1"), "horizon_mse_reduction",
                       direction="minimize"),
            FamilySpec("scinet::dev_pilot_dynamics", "scinet",
                       "controlled-pilot dynamics identification",
                       _scinet("controlled_pendulum_pilot"),
                       "horizon_mse_reduction", direction="minimize"),
            FamilySpec("scinet::dev_pusher_planning", "scinet",
                       "pusher planning rollout fidelity",
                       _scinet("pusher_v1", "configs/pendulum.yaml"),
                       "horizon_mse_reduction", direction="minimize"),
        ),
        "selection": (
            FamilySpec("scinet::sel_v2_pilot", "scinet",
                       "v2 controlled-pilot generalization",
                       _scinet("controlled_pendulum_v2_pilot"),
                       "horizon_mse_reduction", direction="minimize"),
            FamilySpec("scinet::sel_ood_ident", "scinet",
                       "OOD system identification",
                       _scinet("pendulum_smoke_ood_v1",
                               "configs/pendulum_smoke_ood.yaml"),
                       "horizon_mse_reduction", direction="minimize"),
            FamilySpec("scinet::sel_lr_probe", "scinet",
                       "learning-rate sensitivity on smoke data",
                       _scinet("pendulum_smoke_v1"),
                       "horizon_mse_reduction", direction="minimize"),
        ),
        "holdout": (
            FamilySpec("scinet::ho_double_pendulum", "scinet",
                       "double-pendulum transfer (hidden family)",
                       _scinet("pendulum_smoke_v1", "configs/double_pendulum.yaml"),
                       "horizon_mse_reduction", direction="minimize"),
            FamilySpec("scinet::ho_newton", "scinet",
                       "newton dynamics transfer (hidden family)",
                       _scinet("pendulum_smoke_v1", "configs/newton.yaml"),
                       "horizon_mse_reduction", direction="minimize"),
        ),
    },
    "embodied": {
        "dev": (
            FamilySpec("embodied::dev_short_rollout", "embodied",
                       "short-horizon state prediction",
                       _embodied("short-horizon state prediction",
                                 {"model.rollout_horizon": 3}), "state_mse@1",
                       direction="minimize"),
            FamilySpec("embodied::dev_long_rollout", "embodied",
                       "long-horizon rollout stability",
                       _embodied("long-horizon rollout stability",
                                 {"model.rollout_horizon": 8}), "state_mse@1",
                       direction="minimize"),
        ),
        "selection": (
            FamilySpec("embodied::sel_dynamics", "embodied",
                       "dynamics-error accumulation",
                       _embodied("dynamics-error accumulation",
                                 {"model.drift_correction": 0.0}), "state_mse@1",
                       direction="minimize"),
            FamilySpec("embodied::sel_lr", "embodied",
                       "training-rate sensitivity",
                       _embodied("training-rate sensitivity",
                                 {"train.lr": 0.01}), "state_mse@1",
                       direction="minimize"),
        ),
        "holdout": (
            FamilySpec("embodied::ho_transfer", "embodied",
                       "cross-condition transfer (hidden family)",
                       _embodied("cross-condition transfer",
                                 {"model.rollout_horizon": 6,
                                  "model.drift_correction": 0.0},
                                 split="test"), "state_mse@1",
                       direction="minimize"),
            FamilySpec("embodied::ho_ood", "embodied",
                       "OOD dynamics prediction (hidden family)",
                       _embodied("OOD dynamics prediction",
                                 {"model.rollout_horizon": 5},
                                 split="ood"), "state_mse@1",
                       direction="minimize"),
        ),
    },
    "quant": {
        "dev": (
            FamilySpec("quant::dev_window", "quant",
                       "signal-window sensitivity (synthetic)",
                       _quant("signal-window sensitivity",
                              {"strategy.signal_window": 5}),
                       "net_return_after_cost"),
            FamilySpec("quant::dev_smoothing", "quant",
                       "signal-smoothing sensitivity (synthetic)",
                       _quant("signal-smoothing sensitivity",
                              {"strategy.signal_smoothing": 0.0}),
                       "net_return_after_cost"),
        ),
        "selection": (
            FamilySpec("quant::sel_cost", "quant",
                       "transaction-cost robustness (synthetic)",
                       _quant("transaction-cost robustness",
                              {"strategy.cost_bps": 5.0}),
                       "net_return_after_cost"),
            FamilySpec("quant::sel_holding", "quant",
                       "holding-horizon sensitivity (synthetic)",
                       _quant("holding-horizon sensitivity",
                              {"strategy.holding_horizon": 1}),
                       "net_return_after_cost"),
        ),
        "holdout": (
            FamilySpec("quant::ho_regime", "quant",
                       "regime robustness (hidden family, synthetic)",
                       _quant("regime robustness",
                              {"strategy.signal_window": 8}, split="test"),
                       "net_return_after_cost"),
            FamilySpec("quant::ho_vol", "quant",
                       "volatility-shift robustness (hidden, synthetic)",
                       _quant("volatility-shift robustness",
                              {"strategy.holding_horizon": 2}, split="ood"),
                       "net_return_after_cost"),
        ),
    },
}

DEFAULT_SPLIT_SIZES = {"dev": 36, "selection": 12, "holdout": 12}


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    family_id: str
    domain: str
    objective: str
    task_config: dict[str, Any]
    task_sha: str
    split: str
    source: dict[str, Any]
    budget: dict[str, Any]
    seed: int
    primary_metric: str
    direction: str
    gold_ref: str = ""
    rubric_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def generation_view(self) -> dict[str, Any]:
        """Dev-only, sealed refs stripped, source generalization."""
        d = self.to_dict()
        d.pop("gold_ref", None)
        d.pop("rubric_ref", None)
        # Do not expose the concrete dataset/config identity of a dev task as
        # if it were a research finding; keep only the executable handle.
        d["source"] = {"engine": self.source.get("engine", ""),
                       "split": self.source.get("split", "")}
        return d


@dataclass
class BenchmarkSpecV2:
    version: str
    tasks: list[BenchmarkTask]
    split_manifest_shas: dict[str, str] = field(default_factory=dict)
    split_manifest_paths: dict[str, str] = field(default_factory=dict)
    manifest_sha: str = ""
    manifest_path: str = ""
    sealed: dict[str, Any] = field(default_factory=dict)
    generator_version: str = "benchmark_v2_generator_v1"

    def by_split(self, split: str) -> list[BenchmarkTask]:
        return [t for t in self.tasks if t.split == split]

    def families(self, split: str) -> set[str]:
        return {t.family_id for t in self.by_split(split)}

    def generator_view(self) -> dict[str, Any]:
        return {"version": self.version,
                "tasks": [t.generation_view() for t in self.by_split("dev")],
                "note": "dev only; selection/holdout families and gold absent"}

    def promotable_view(self) -> dict[str, Any]:
        return {"version": self.version, "manifest_sha256": self.manifest_sha,
                "split_manifest_shas": dict(self.split_manifest_shas),
                "selection_task_ids": [t.task_id for t in self.by_split("selection")],
                "holdout_task_ids": [t.task_id for t in self.by_split("holdout")],
                "selection_families": sorted(self.families("selection")),
                "holdout_families": sorted(self.families("holdout"))}


def build_tasks_v2(version: str = BENCHMARK_VERSION,
                   split_sizes: dict[str, int] | None = None
                   ) -> list[BenchmarkTask]:
    """Deterministically build V2 tasks with 3-way disjoint families."""
    sizes = dict(DEFAULT_SPLIT_SIZES if split_sizes is None else split_sizes)
    tasks: list[BenchmarkTask] = []
    counter = 0
    for domain, pools in FAMILIES.items():
        for split in SPLITS:
            fams = pools[split]
            # At most 3 tasks per family; extra seeds are WITHIN a family and
            # never create a new statistical unit.
            per_domain = max(1, int(sizes[split]) // 3)
            per_family = max(1, per_domain // len(fams))
            for fam in fams:
                for j in range(per_family):
                    counter += 1
                    seed = 7000 + counter
                    tid = f"{version}_{domain}_{split}_{fam.family_id.split('::')[1]}_{j:03d}"
                    src = dict(fam.source)
                    src["seed"] = seed
                    src["family_id"] = fam.family_id
                    embedded_cfg = dict(src.get("task_config") or {})
                    task_cfg = {**embedded_cfg, **dict(fam.task_config)}
                    base = {
                        "task_id": tid, "family_id": fam.family_id,
                        "domain": domain, "objective": fam.objective,
                        "task_config": task_cfg, "split": split,
                        "source": src,
                        "budget": {"max_campaign_rounds": 1, "max_calls": 4,
                                   "max_wall_s": 900},
                        "seed": seed, "primary_metric": fam.primary_metric,
                        "direction": fam.direction,
                        "gold_ref": f"gold::{tid}",
                        "rubric_ref": f"rubric::{tid}",
                    }
                    task = BenchmarkTask(**base, task_sha="")
                    task = BenchmarkTask(**{**base,
                                            "task_sha": sha256_json(
                                                task.to_dict())})
                    tasks.append(task)
    return tasks


def validate_benchmark_v2(spec: BenchmarkSpecV2) -> dict[str, Any]:
    """Fail closed on ANY 3-way overlap, duplicate or SHA mismatch."""
    problems: list[str] = []
    dev, sel, hold = (spec.families("dev"), spec.families("selection"),
                      spec.families("holdout"))
    if dev & sel:
        problems.append("family_overlap_dev_selection")
    if dev & hold:
        problems.append("family_overlap_dev_holdout")
    if sel & hold:
        problems.append("family_overlap_selection_holdout")
    ids = [t.task_id for t in spec.tasks]
    if len(ids) != len(set(ids)):
        problems.append("duplicate_task_ids")
    fids = [t.family_id for t in spec.tasks]
    if len(set(fids)) != len(set(t.family_id for _, fams in FAMILIES.items()
                                 for pool in fams.values() for t in pool)):
        problems.append("missing_or_duplicate_family_definition")
    for t in spec.tasks:
        recomputed = sha256_json({**t.to_dict(), "task_sha": ""})
        if recomputed != t.task_sha:
            problems.append(f"task_sha_mismatch:{t.task_id}")
        if not t.source.get("engine"):
            problems.append(f"missing_source:{t.task_id}")
        if not t.task_config and t.source.get("engine") != "scinet":
            problems.append(f"missing_task_config:{t.task_id}")
        if not t.primary_metric:
            problems.append(f"missing_metric:{t.task_id}")
    for split in SPLITS:
        expected = spec.split_manifest_shas.get(split, "")
        path = spec.split_manifest_paths.get(split, "")
        if not expected or not path or not Path(path).is_file():
            problems.append(f"split_manifest_missing:{split}")
        elif verify_frozen(path, expected)["ok"] is False:
            problems.append(f"split_manifest_sha_mismatch:{split}")
    if spec.sealed.get("sha256"):
        if not verify_frozen(spec.sealed["path"], spec.sealed["sha256"])["ok"]:
            problems.append("sealed_gold_tampered")
    return {"ok": not problems, "reasons": problems,
            "n_tasks": len(spec.tasks),
            "n_families": {"dev": len(dev), "selection": len(sel),
                           "holdout": len(hold)},
            "families_dev": sorted(dev), "families_selection": sorted(sel),
            "families_holdout": sorted(hold)}


def build_benchmark_v2(out_dir: str | Path, *,
                       version: str = BENCHMARK_VERSION,
                       split_sizes: dict[str, int] | None = None,
                       mme_by_metric: dict[str, float] | None = None
                       ) -> BenchmarkSpecV2:
    """Build, freeze and validate a V2 benchmark with split manifests."""
    root = Path(out_dir)
    tasks = build_tasks_v2(version=version, split_sizes=split_sizes)
    sealed_obj = {
        "version": version,
        "warning": "NEVER expose to candidate generators or evaluated harnesses",
        "gold": {t.task_id: {"direction": t.direction,
                             "primary_metric": t.primary_metric,
                             "minimum_meaningful_effect":
                             (mme_by_metric or {}).get(t.primary_metric, 0.05),
                             "family": t.family_id}
                 for t in tasks},
        "rubric": {t.task_id: {"dimensions": ["evidence", "falsifiability",
                                              "validity", "diagnosis"],
                               "scale": [0, 1, 2, 3]} for t in tasks},
    }
    sealed = freeze_artifact(sealed_obj, root / "sealed" / "gold_rubric.json")
    split_shas: dict[str, str] = {}
    split_paths: dict[str, str] = {}
    for split in SPLITS:
        split_tasks = [t.to_dict() for t in tasks if t.split == split]
        obj = {"benchmark_version": version, "split": split,
               "families": sorted({t["family_id"] for t in split_tasks}),
               "tasks": split_tasks}
        frozen = freeze_json(obj, root / split / "manifest.json")
        split_shas[split] = frozen["sha256"]
        split_paths[split] = frozen["path"]
    manifest_obj = {
        "benchmark_version": version, "schema": "m6_benchmark_v2_manifest",
        "generator_version": "benchmark_v2_generator_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "development_families": sorted({f.family_id for d in FAMILIES
                                        for f in FAMILIES[d]["dev"]}),
        "selection_families": sorted({f.family_id for d in FAMILIES
                                      for f in FAMILIES[d]["selection"]}),
        "hidden_holdout_families": sorted({f.family_id for d in FAMILIES
                                           for f in FAMILIES[d]["holdout"]}),
        "development_manifest_sha": split_shas["dev"],
        "selection_manifest_sha": split_shas["selection"],
        "hidden_holdout_manifest_sha": split_shas["holdout"],
        "split_manifest_shas": split_shas,
        "split_manifest_paths": split_paths,
        "split_sizes": {s: len([t for t in tasks if t.split == s])
                        for s in SPLITS},
        "sealed_ref": {"path": sealed["path"], "sha256": sealed["sha256"]},
    }
    frozen = freeze_json(manifest_obj, root / "manifest.json")
    spec = BenchmarkSpecV2(
        version=version, tasks=tasks, split_manifest_shas=split_shas,
        split_manifest_paths=split_paths, manifest_sha=frozen["sha256"],
        manifest_path=frozen["path"], sealed=sealed)
    verdict = validate_benchmark_v2(spec)
    if not verdict["ok"]:
        raise BenchmarkIsolationError("|".join(verdict["reasons"]))
    freeze_json({"verdict": verdict, "manifest_sha": spec.manifest_sha},
                root / "isolation_verdict.json")
    return spec


# --------------------------------------------------------------------------- #
# Holdout access verifier: EVERY holdout read passes through here.             #
# --------------------------------------------------------------------------- #
class HoldoutAccessVerifier:
    """Append-only, hash-chained ledger of hidden-holdout access.

    The verifier is the ONLY legitimate way to obtain holdout tasks. Each
    access must present a preregistered purpose and an experiment id; the
    purpose must be in the registered allow-list (final evaluation only).
    Any attempt to read holdout outside this path is HOLDOUT_LEAKAGE.
    """

    ALLOWED_PURPOSES = frozenset({"final_holdout_evaluation",
                                  "rollback_smoke_holdout"})

    def __init__(self, spec: BenchmarkSpecV2, path: str | Path):
        self.spec = spec
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._chain = "0" * 64
        if self.path.is_file():
            obj = json.loads(self.path.read_text(encoding="utf-8"))
            self._chain = str(obj.get("chain_sha", self._chain))

    def _entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return list(json.loads(self.path.read_text(
            encoding="utf-8")).get("entries", []))

    def authorize(self, *, purpose: str, experiment_id: str,
                  requested_task_ids: list[str],
                  actor: str) -> dict[str, Any]:
        if purpose not in self.ALLOWED_PURPOSES:
            return {"ok": False, "reason": "HOLDOUT_LEAKAGE",
                    "detail": f"unauthorized purpose: {purpose}"}
        allowed = {t.task_id for t in self.spec.by_split("holdout")}
        unknown = set(requested_task_ids) - allowed
        if unknown:
            return {"ok": False, "reason": "HOLDOUT_LEAKAGE",
                    "detail": f"unknown holdout ids: {sorted(unknown)}"}
        entry = {"purpose": purpose, "experiment_id": experiment_id,
                 "requested_task_ids": sorted(requested_task_ids),
                 "actor": actor, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "prev_sha": self._chain}
        entry["entry_sha"] = sha256_json(entry)
        self._chain = entry["entry_sha"]
        obj = {"version": "m6_holdout_access_v1",
               "chain_sha": self._chain,
               "entries": self._entries() + [entry]}
        freeze_json(obj, self.path)
        return {"ok": True, "entry_sha": entry["entry_sha"],
                "chain_sha": self._chain}

    def tasks(self, *, purpose: str, experiment_id: str,
              requested_task_ids: list[str], actor: str) -> list[BenchmarkTask]:
        auth = self.authorize(purpose=purpose, experiment_id=experiment_id,
                              requested_task_ids=requested_task_ids,
                              actor=actor)
        if not auth["ok"]:
            raise BenchmarkIsolationError(json.dumps(auth, ensure_ascii=False))
        wanted = set(requested_task_ids)
        return [t for t in self.spec.by_split("holdout")
                if t.task_id in wanted]

    def verify_chain(self) -> dict[str, Any]:
        entries = self._entries()
        prev = "0" * 64
        for e in entries:
            if e.get("prev_sha") != prev:
                return {"ok": False, "reason": "holdout_ledger_broken"}
            check = dict(e)
            check.pop("entry_sha", None)
            if sha256_json(check) != e.get("entry_sha"):
                return {"ok": False, "reason": "holdout_ledger_tampered"}
            prev = e["entry_sha"]
        return {"ok": True, "n_accesses": len(entries),
                "chain_sha": prev}


def assert_no_holdout_in_generator_view(spec: BenchmarkSpecV2) -> dict[str, Any]:
    """Prove the candidate generator never sees holdout families or ids."""
    gv = json.dumps(spec.generator_view(), ensure_ascii=False)
    hold_ids = [t.task_id for t in spec.by_split("holdout")]
    hold_fams = sorted(spec.families("holdout"))
    leaked_ids = [tid for tid in hold_ids if tid in gv]
    leaked_fams = [fam for fam in hold_fams if fam in gv]
    return {"ok": not leaked_ids and not leaked_fams,
            "leaked_task_ids": leaked_ids, "leaked_families": leaked_fams}


__all__ = [
    "BENCHMARK_VERSION", "SPLITS", "FAMILIES", "BenchmarkIsolationError",
    "FamilySpec", "BenchmarkTask", "BenchmarkSpecV2", "build_tasks_v2",
    "build_benchmark_v2", "validate_benchmark_v2", "HoldoutAccessVerifier",
    "assert_no_holdout_in_generator_view",
]
