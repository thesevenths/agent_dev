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

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from .evolution.evolution_controller import run_dual_layer


# --------------------------------------------------------------------------- #
# P0-2 -- READ-ONLY provenance collection from REAL engine artifacts.
#
# Nothing here invents a metric or a hash. Every value is read from an existing
# file written by the engine (``*metrics*.json``) or from the frozen dataset
# ``manifest.json``. The discovery rules are DOMAIN-AGNOSTIC on purpose: they
# follow the engine's generic artifact contract (``data.frozen_dir`` in the
# config, ``outputs/research_campaigns/<campaign_id>`` for jobs, and the
# ``*_metrics.json`` naming already used by scinet / embodied / quant), not any
# particular domain's file name or dataset name.
# --------------------------------------------------------------------------- #

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_frozen_dir(cfg_path: Path, project_dir: Path | None) -> Path | None:
    """Locate a config's frozen dataset dir using the generic ``data.frozen_dir``.

    A config may live in the domain project dir (embodied/quant) or in a
    separate engine checkout (scinet OOD), so try the plausible roots instead
    of hard-coding one project's layout.
    """
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return None
    frozen = ((cfg.get("data") or {}).get("frozen_dir") if isinstance(cfg, dict)
              else None)
    if not frozen:
        return None
    p = Path(str(frozen))
    roots = ([p] if p.is_absolute() else [])
    if not p.is_absolute():
        roots += [cfg_path.parent / p, cfg_path.parent.parent / p]
        if project_dir is not None:
            roots += [project_dir / p, project_dir.parent / p]
    for root in roots:
        if (root / "manifest.json").is_file():
            return root
    return None


def _manifest_provenance(cfg_path: Path | None,
                         project_dir: Path | None) -> dict[str, Any]:
    """SHA-pinned identity of a frozen dataset: manifest + config + splits."""
    empty = {"available": False, "reason": "no config path"}
    if cfg_path is None:
        return empty
    cfg_path = Path(cfg_path)
    if not cfg_path.is_file():
        return {"available": False, "config_path": str(cfg_path),
                "reason": "config file missing"}
    fdir = _resolve_frozen_dir(cfg_path, project_dir)
    if fdir is None:
        return {"available": False, "config_path": str(cfg_path),
                "config_sha256": _sha256_file(cfg_path),
                "reason": "frozen_dir/manifest.json not resolvable"}
    mpath = fdir / "manifest.json"
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "manifest_path": str(mpath),
                "reason": f"manifest unreadable: {exc}"}
    return {
        "available": True,
        "config_path": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "manifest_path": str(mpath),
        "manifest_sha256": _sha256_file(mpath),
        "dataset_version": manifest.get("dataset_version"),
        "declared_config_sha256": manifest.get("config_sha256"),
        "split_sha256": {name: (info or {}).get("sha256")
                         for name, info in (manifest.get("splits") or {}).items()},
        "split_count": len(manifest.get("splits") or {}),
    }


def _collect_metric_artifacts(roots: Iterable[Path]) -> list[dict[str, Any]]:
    """Read every real ``*metrics*.json`` under the given job roots."""
    found: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*metrics*.json")):
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(doc, dict):
                continue
            found.append({
                "path": str(path),
                "file_sha256": _sha256_file(path),
                "name": path.name,
                "axis": "ood" if "ood" in path.name.lower() else "id",
                "dataset_version": doc.get("dataset_version"),
                "checkpoint_sha256": doc.get("checkpoint_sha256"),
                "evaluation_run_id": doc.get("run_id"),
                "split": doc.get("split"),
                "config_path": doc.get("config_path"),
                "metrics": doc.get("metrics") if isinstance(doc.get("metrics"), dict)
                           else None,
            })
    return found


def collect_domain_provenance(*, spec: "DomainSpec", dr_root: Path,
                              campaign_id: str) -> dict[str, Any]:
    """Assemble the FULL traceability chain for one domain, from disk only.

    Covers (P0-2): OOD dataset_version, manifest SHA, checkpoint SHA,
    evaluation run ID and OOD metrics. Missing evidence is reported as an
    explicit ``unavailable`` field with a reason -- never silently filled in.
    """
    project_dir = Path(spec.project_dir) if spec.project_dir else None
    campaign_dirs: list[Path] = [dr_root]
    if project_dir is not None:
        campaign_dirs.append(
            project_dir / "outputs" / "research_campaigns" / campaign_id)
    artifacts = _collect_metric_artifacts(campaign_dirs)

    ood_cfg = Path(spec.ood_config_path) if spec.ood_config_path else None
    ood_cfg_str = str(ood_cfg).lower() if ood_cfg else None
    # Axis classification by CONTENT, not by engine file name. The ``ood``
    # infix is only a fallback: an artifact is OOD if it was evaluated against
    # the OOD config, or against a dataset version distinct from the ID one.
    for a in artifacts:
        cfg_match = bool(ood_cfg_str and a["config_path"]
                         and str(a["config_path"]).lower() == ood_cfg_str)
        ds_match = bool(a["dataset_version"] and spec.dataset_version
                        and a["dataset_version"] != spec.dataset_version)
        a["axis"] = "ood" if (cfg_match or ds_match
                              or "ood" in a["name"].lower()) else "id"
        a["axis_basis"] = ("config_path" if cfg_match else
                           "dataset_version" if ds_match else
                           "filename" if "ood" in a["name"].lower() else "id_default")
    ood_manifest = _manifest_provenance(ood_cfg, project_dir)
    id_manifest = _manifest_provenance(Path(spec.base_config), project_dir)

    ood_artifacts = [a for a in artifacts if a["axis"] == "ood"]
    id_artifacts = [a for a in artifacts if a["axis"] == "id"]

    def _uniq(values: Iterable[Any]) -> list[Any]:
        return sorted({v for v in values if v is not None}, key=str)

    ood_versions = _uniq([a["dataset_version"] for a in ood_artifacts]
                         + [ood_manifest.get("dataset_version")])
    ood_checkpoints = _uniq(a["checkpoint_sha256"] for a in ood_artifacts)
    ood_run_ids = _uniq(a["evaluation_run_id"] for a in ood_artifacts)
    ood_metrics = [a["metrics"] for a in ood_artifacts if a["metrics"]]

    required = {
        "ood_dataset_version": bool(ood_versions),
        "ood_manifest_sha256": bool(ood_manifest.get("available")),
        "ood_checkpoint_sha256": bool(ood_checkpoints),
        "ood_evaluation_run_ids": bool(ood_run_ids),
        "ood_metrics": bool(ood_metrics),
    }
    # A domain with no OOD axis is legitimately complete: there is nothing to
    # trace. Only claim completeness when an OOD config was actually requested.
    ood_requested = ood_cfg is not None
    if not ood_requested:
        status, missing = "not_applicable", []
    else:
        missing = [k for k, ok in required.items() if not ok]
        status = "complete" if not missing else "incomplete"

    return {
        "schema_version": "m3.4-provenance-v1",
        "status": status,
        "missing": missing,
        "ood_requested": ood_requested,
        "id_dataset": {
            "dataset_version": spec.dataset_version,
            "manifest": id_manifest,
        },
        "ood_dataset": {
            "dataset_versions": ood_versions,
            "dataset_version": ood_versions[0] if ood_versions else None,
            "manifest": ood_manifest,
            "config_path": str(ood_cfg) if ood_cfg else None,
        },
        "checkpoint_sha256": {
            "ood": ood_checkpoints,
            "id": _uniq(a["checkpoint_sha256"] for a in id_artifacts),
        },
        "evaluation_run_ids": {
            "ood": ood_run_ids,
            "id": _uniq(a["evaluation_run_id"] for a in id_artifacts),
        },
            "ood_metrics": {
                "n_artifacts": len(ood_artifacts),
                "n_with_metrics": len(ood_metrics),
                "artifacts": [
                    {"path": a["path"], "file_sha256": a["file_sha256"],
                     "axis": a["axis"], "axis_basis": a["axis_basis"],
                     "dataset_version": a["dataset_version"],
                     "checkpoint_sha256": a["checkpoint_sha256"],
                     "evaluation_run_id": a["evaluation_run_id"],
                     "split": a["split"], "metrics": a["metrics"]}
                    for a in ood_artifacts],
            },
            "artifact_files": [{"path": a["path"], "file_sha256": a["file_sha256"],
                                "axis": a["axis"],
                                "axis_basis": a["axis_basis"]} for a in artifacts],
    }


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
        # P0-2: collect the full OOD traceability chain from REAL engine
        # artifacts on disk (manifest SHA / checkpoint SHA / run IDs / raw OOD
        # metrics). This is READ-ONLY and never fabricates a missing value.
        prov = collect_domain_provenance(
            spec=spec, dr_root=dr_root,
            campaign_id=self._campaign_id(spec.domain))
        dom["provenance"] = prov
        # Disk evidence WINS over a version merely declared in a record. A
        # mismatch is retained explicitly (never silently reconciled), because
        # the historical arr_m34_001 index had ood_dataset_version=null while
        # the engine had in fact written 40 real OOD metric artifacts.
        declared = dom.get("ood_dataset_version")
        disk_version = prov["ood_dataset"]["dataset_version"]
        if disk_version:
            dom["ood_dataset_version"] = disk_version
        if declared and disk_version and declared != disk_version:
            prov["ood_dataset"]["declared_in_records"] = declared
            prov["ood_dataset"]["version_conflict"] = True
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
            # P0-2: full provenance chain travels into the summary and the
            # evidence index. Legacy summaries without it stay valid (None).
            "provenance": dom.get("provenance"),
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
            prov = d.get("provenance")
            decisions = {}
            for r in recs:
                # P0-2: surface the OOD axis metrics that the real statistical
                # gate already produced. Only COPY fields that exist in the raw
                # record -- never compute, infer or fabricate a metric here.
                key = str(r.get("hyp_fingerprint"))
                entry = {
                    "target": r.get("target"),
                    "new": r.get("new"),
                    "decision": r.get("decision"),
                    "ood_decision": r.get("ood_decision"),
                    "mean_improvement": r.get("mean_improvement"),
                    "holdout_direction_fraction": r.get("holdout_direction_fraction"),
                }
                for fld in ("ood_mean_improvement", "ood_baseline_mean",
                            "ood_std_improvement", "ood_n_valid", "ood_n_seeds",
                            "ood_holdout_direction_fraction", "ood_statistics",
                            "statistics", "primary_metric"):
                    if fld in r:
                        entry[fld] = r[fld]
                decisions[key] = entry
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
                "decisions": decisions,
                # P0-2 -- explicit, auditable OOD traceability chain.
                "ood_provenance": prov,
                "provenance_status": (prov.get("status") if isinstance(prov, dict)
                                      else "not_collected"),
                "provenance_missing": (prov.get("missing") if isinstance(prov, dict)
                                       else []),
                "promotion_recommendation": d.get("promotion_recommendation"),
            }
        return {
            "run_id": state.get("run_id"),
            "schema_version": "m3.4-evidence-index-v2",
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
