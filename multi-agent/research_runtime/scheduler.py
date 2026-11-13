"""M3.4 -- Unified Research Scheduler (Phase 2) + global state (Phase 3)
+ unified evidence index / deterministic promotion (Phase 4).

This module adds the cross-domain coordination that Phase 0 identified as
MISSING. It deliberately does NOT build a second runtime: it drives the EXISTING
dual-layer RRSI controller (``evolution_controller.run_dual_layer``) once per
domain and coordinates the results.

What it adds on top of the existing engine/statistical-gate:
  * a single top-level ``run_id`` + ``run_root`` for the whole orchestration;
  * a per-domain ISOLATED campaign namespace (unique ``campaign_id``) and a
    per-domain ``run_root`` (isolated candidate registry / evidence ledger);
  * atomic global state (``scheduler_state.json``) with IDEMPOTENT resume -- a
    crashed run re-executes only INCOMPLETE domains; a completed domain is
    skipped (its summary is reloaded, never recomputed);
  * a unified evidence index (``evidence_index.json``) consolidating config SHA,
    dataset version, baseline/candidate pairing, the ID/OOD statistical report,
    ACCEPT/REJECT/INCONCLUSIVE and the version-inheritance record;
  * a DETERMINISTIC promotion selection: when several candidates are ACCEPTED the
    scheduler records a single recommended promotion (largest absolute mean
    improvement, deterministic tie-break) and NEVER silently overwrites the
    baseline by execution order (F-1 fix at the orchestration level).

Run isolation (Phase 1): every WRITABLE artifact (per-candidate config, metrics,
checkpoints, logs) is namespaced by ``<run_id>_<domain>``; the per-domain
``candidate_registry.jsonl`` lives under ``<run_root>/<domain>/``. The frozen
datasets and baseline configs are immutable, content-addressed and SHA-verified,
shared read-only across runs BY DESIGN -- a frozen input is not a collision.
Engine-internal storage is run_id-keyed and therefore never collides; the M3.3
shared-``dual_layer``-campaign bug is fixed by the unique campaign_id.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .evolution.evolution_controller import run_dual_layer


# --------------------------------------------------------------------------- #
# Domain specification
# --------------------------------------------------------------------------- #
@dataclass
class DomainSpec:
    """One domain's slice of a unified run.

    ``project_dir`` is the canonical domain directory (datasets + baseline config
    live there, immutable). It is shared read-only across runs; writability is
    isolated via ``campaign_id``/``run_root`` (see module docstring).
    """

    domain: str
    base_config: str | Path
    dataset_version: str
    paired_seeds: tuple[int, ...]
    holdout_seeds: tuple[int, ...] = ()
    ood_config_path: str | Path | None = None
    project_dir: str | Path | None = None
    require_frozen_manifest: bool = True
    max_minutes: float = 30.0
    rounds: int = 1
    harness_evolve: Callable[..., dict[str, Any]] | None = None
    metric_direction: str = "minimize"   # "minimize" | "maximize"
    synthetic_data: bool = False         # quant MUST be flagged synthetic


# A no-op harness layer: the harness evolution is orthogonal to the domain gate
# validated here (same stance as M3.0 / M3.3). It returns a stub payload so the
# dual-layer controller records a harness layer without running 5 independent
# harness tasks (which would be expensive and is out of scope for M3.4).
def _harness_stub(**kw: Any) -> dict[str, Any]:
    return {
        "schema_version": "stub-harness-layer",
        "dominant_failure": "DECISION_UNCERTAINTY",
        "records": [],
        "accepted": [],
        "n_accepted": 0,
    }


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
class UnifiedResearchScheduler:
    def __init__(self, *, run_id: str, run_root: str | Path,
                 repo_root: str | Path, domains: Iterable[DomainSpec],
                 build_runner: Callable[..., Callable] | None = None,
                 harness_evolve: Callable[..., dict[str, Any]] | None = None):
        self.run_id = run_id
        self.run_root = Path(run_root).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.domains: list[DomainSpec] = list(domains)
        # Default runner factory: the REAL engine runner used by M3.x. Tests may
        # inject a surrogate (writes fake metrics) to run engine-free.
        self._build_runner = build_runner
        self._harness_evolve = harness_evolve or _harness_stub
        self.run_root.mkdir(parents=True, exist_ok=True)

    # -- naming -----------------------------------------------------------
    def _domain_run_root(self, domain: str) -> Path:
        return self.run_root / domain

    def _campaign_id(self, domain: str) -> str:
        return f"{self.run_id}_{domain}"

    def _domain_summary_path(self, domain: str) -> Path:
        return self._domain_run_root(domain) / "domain_summary.json"

    # -- resume support (Phase 3) -----------------------------------------
    def _domain_complete(self, domain: str) -> bool:
        p = self._domain_summary_path(domain)
        if not p.is_file():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return False
        return bool(data.get("status") == "completed")

    @staticmethod
    def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                  ensure_ascii=False, default=str),
                       encoding="utf-8")
        tmp.replace(path)

    # -- per-domain execution ---------------------------------------------
    def _run_domain(self, spec: DomainSpec) -> dict[str, Any]:
        dr_root = self._domain_run_root(spec.domain)
        dr_root.mkdir(parents=True, exist_ok=True)
        runner = None
        if self._build_runner is not None:
            runner = self._build_runner(
                spec.domain, spec.project_dir, ood_config_path=spec.ood_config_path)
        payload = run_dual_layer(
            base_harness=None, tasks=[], heldout_task=None,
            run_root=str(dr_root), repo_root=str(self.repo_root),
            rounds=spec.rounds, max_minutes=spec.max_minutes,
            domain=spec.domain, domain_base_config=str(spec.base_config),
            project_dir=str(spec.project_dir) if spec.project_dir else None,
            dataset_version=spec.dataset_version,
            domain_seeds=spec.paired_seeds,
            domain_holdout_seeds=spec.holdout_seeds,
            domain_require_frozen_manifest=spec.require_frozen_manifest,
            domain_runner=runner,
            harness_evolve=self._harness_evolve,
            ood_config_path=(str(spec.ood_config_path)
                             if spec.ood_config_path else None),
            domain_campaign_id=self._campaign_id(spec.domain),
        )
        dom = payload.get("domain_layer", {})
        # Enrich with scheduler-known identity (the domain_layer payload does not
        # otherwise surface these, but they are required for the evidence index).
        dom["campaign_id"] = self._campaign_id(spec.domain)
        dom["registry_path"] = str(dr_root / "candidate_registry.jsonl")
        for rec in dom.get("records", []):
            if rec.get("ood_dataset_version"):
                dom["ood_dataset_version"] = rec["ood_dataset_version"]
                break
        return payload

    @staticmethod
    def _norm_decision(d: Any) -> str:
        """domain_layer records carry the verdict as a lower-case string."""
        return (d or "").strip().upper()

    @staticmethod
    def _summarize_domain(spec: DomainSpec, payload: dict[str, Any]) -> dict[str, Any]:
        dom = payload.get("domain_layer", {})
        records = dom.get("records", [])
        accepted = [r for r in records
                    if UnifiedResearchScheduler._norm_decision(r.get("decision"))
                    == "ACCEPTED"]
        rejected = [r for r in records
                    if UnifiedResearchScheduler._norm_decision(r.get("decision"))
                    == "REJECTED"]
        inconclusive = [r for r in records
                        if UnifiedResearchScheduler._norm_decision(r.get("decision"))
                        == "INCONCLUSIVE"]
        promotion = UnifiedResearchScheduler._select_promotion(spec, records)
        return {
            "domain": spec.domain,
            "synthetic_data": spec.synthetic_data,
            "dataset_version": spec.dataset_version,
            "ood_dataset_version": dom.get("ood_dataset_version"),
            "paired_seeds": list(spec.paired_seeds),
            "holdout_seeds": list(spec.holdout_seeds),
            "n_candidates": len(records),
            "n_accepted": len(accepted),
            "n_rejected": len(rejected),
            "n_inconclusive": len(inconclusive),
            "campaign_id": dom.get("campaign_id"),
            "statistics_protocol": dom.get("statistics_protocol"),
            "records": records,
            "promotion_recommendation": promotion,
            "registry_path": dom.get("registry_path"),
            "status": "completed",
        }

    # -- deterministic promotion (F-1 fix) --------------------------------
    @staticmethod
    def _select_promotion(spec: DomainSpec,
                          records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Pick a SINGLE recommended promotion among ACCEPTED candidates.

        Rule (deterministic, order-independent):
          1. only candidates with combined decision ACCEPTED are eligible;
          2. prefer the largest absolute mean improvement, adjusted so that
             "improvement" is positive in the metric's direction;
          3. tie-break by target name (ascending) then by new value (ascending).
        The scheduler only RECORDS this recommendation; it never mutates the
        baseline. Baseline mutation belongs to the loop / version-inheritance
        layer, which the task book keeps separate.
        """
        eligible = [r for r in records
                    if UnifiedResearchScheduler._norm_decision(r.get("decision"))
                    == "ACCEPTED"]
        if not eligible:
            return {"chosen": None, "reason": "no ACCEPTED candidate",
                    "candidates_considered": 0}
        sign = 1.0 if spec.metric_direction == "maximize" else -1.0

        def _score(r: Mapping[str, Any]) -> float:
            mi = float(r.get("mean_improvement") or 0.0)
            return sign * mi

        ranked = sorted(
            eligible,
            key=lambda r: (-abs(_score(r)), str(r.get("target")),
                           str(r.get("new"))))
        best = ranked[0]
        return {
            "chosen": {
                "target": best.get("target"), "new": best.get("new"),
                "decision": best.get("decision"),
                "mean_improvement": best.get("mean_improvement"),
                "direction_adjusted_score": _score(best),
            },
            "reason": "largest absolute improvement among ACCEPTED, deterministic tie-break",
            "candidates_considered": len(eligible),
            "all_accepted_targets": sorted(
                {(r.get("target"), r.get("new")) for r in eligible}),
        }

    # -- top-level run with idempotent resume ------------------------------
    def run(self) -> dict[str, Any]:
        t0 = time.time()
        per_domain: dict[str, Any] = {}
        completed: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        for spec in self.domains:
            if self._domain_complete(spec.domain):
                data = json.loads(self._domain_summary_path(spec.domain)
                                  .read_text(encoding="utf-8"))
                per_domain[spec.domain] = data
                skipped.append(spec.domain)
                continue
            try:
                payload = self._run_domain(spec)
                summary = self._summarize_domain(spec, payload)
                self._atomic_write(self._domain_summary_path(spec.domain), summary)
                per_domain[spec.domain] = summary
                completed.append(spec.domain)
            except Exception as exc:  # fail-closed per domain; others survive
                failed.append(spec.domain)
                per_domain[spec.domain] = {
                    "domain": spec.domain, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        elapsed = time.time() - t0
        state = {
            "run_id": self.run_id,
            "schema_version": "m3.4-scheduler-v1",
            "run_root": str(self.run_root),
            "repo_root": str(self.repo_root),
            "n_domains": len(self.domains),
            "completed": completed,
            "skipped_resumed": skipped,
            "failed": failed,
            "domains": {d.domain: {
                "synthetic_data": d.synthetic_data,
                "dataset_version": d.dataset_version,
                "paired_seeds": list(d.paired_seeds),
                "holdout_seeds": list(d.holdout_seeds),
                "ood_config_path": (str(d.ood_config_path)
                                    if d.ood_config_path else None),
            } for d in self.domains},
            "elapsed_seconds": round(elapsed, 2),
            "status": "completed" if not failed else "partial_failure",
        }
        self._atomic_write(self.run_root / "scheduler_state.json", state)
        evidence = self._build_evidence_index(state, per_domain)
        self._atomic_write(self.run_root / "evidence_index.json", evidence)
        return {
            "run_id": self.run_id,
            "status": state["status"],
            "elapsed_seconds": state["elapsed_seconds"],
            "completed": completed, "skipped_resumed": skipped, "failed": failed,
            "per_domain": per_domain,
            "scheduler_state": state,
            "evidence_index": evidence,
        }

    # -- unified evidence index (Phase 4) ----------------------------------
    def _build_evidence_index(self, state: Mapping[str, Any],
                              per_domain: Mapping[str, Any]) -> dict[str, Any]:
        entries = {}
        for name, d in per_domain.items():
            if d.get("status") != "completed":
                entries[name] = {"status": d.get("status"),
                                 "error": d.get("error")}
                continue
            recs = d.get("records", [])
            entries[name] = {
                "domain": name,
                "synthetic_data": d.get("synthetic_data"),
                "dataset_version": d.get("dataset_version"),
                "ood_dataset_version": d.get("ood_dataset_version"),
                "campaign_id": d.get("campaign_id"),
                "registry_path": d.get("registry_path"),
                "statistics_protocol": d.get("statistics_protocol"),
                "n_accepted": d.get("n_accepted"),
                "n_rejected": d.get("n_rejected"),
                "n_inconclusive": d.get("n_inconclusive"),
                "decisions": {str(r.get("hyp_fingerprint")): {
                    "target": r.get("target"),
                    "new": r.get("new"),
                    "decision": r.get("decision"),
                    "ood_decision": r.get("ood_decision"),
                    "mean_improvement": r.get("mean_improvement"),
                    "holdout_direction_fraction": r.get("holdout_direction_fraction"),
                } for r in recs},
                "promotion_recommendation": d.get("promotion_recommendation"),
            }
        return {
            "run_id": state.get("run_id"),
            "schema_version": "m3.4-evidence-index-v1",
            "generated_by": "UnifiedResearchScheduler",
            "note": ("Frozen datasets / baseline configs are immutable, "
                     "content-addressed and SHA-verified; shared read-only "
                     "across runs by design. Writable artifacts are namespaced "
                     "by campaign_id = run_id_domain; engine-internal storage "
                     "is run_id-keyed."),
            "domains": entries,
            "promotions": {n: (d.get("promotion_recommendation")
                               if d.get("status") == "completed" else None)
                           for n, d in per_domain.items()},
        }
