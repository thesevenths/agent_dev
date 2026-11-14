"""M4.0-04 -- Unified, READ-ONLY Evidence read model.

Problem (Phase 0 audit): ``ResearchService.default_evidence_provider`` reads
ONLY the append-only ``evidence.jsonl`` ledger, while M3.x writes its real
evidence to ``evidence_index.json`` (v2), ``candidate_registry.jsonl``,
``domain_summary.json``, engine ``*metrics*.json`` artifacts and (optionally)
portable ``evidence_bundle`` releases. The two halves of ARR therefore saw
DIFFERENT evidence sets for the same run.

This module is a READ MODEL, not a new evidence system:

  * it never writes, never recomputes a statistical verdict, never normalises
    a raw verdict string (``inconclusive`` stays ``inconclusive``);
  * it identifies an evidence entity by the SHA-256 of its REAL bytes, so the
    same file discovered through several carriers is ONE entity with several
    ``sources`` / ``carriers`` (duplicate recognition);
  * it keeps source paths, dataset versions, manifest SHA, checkpoint SHA and
    evaluation run IDs exactly as recorded;
  * it is deterministic (sorted, no timestamps), so ResearchService and
    UnifiedResearchScheduler return an IDENTICAL view for the same run.

``evidence_bundle.build_bundle`` decision (task book §7): audited as a MANUAL
offline publisher -- ``cli.py`` only CONSUMES its output and never calls the
builder. It is NOT auto-wired here (wiring would introduce writes and change
the historical release semantics); the tool is preserved untouched and its
published ``evidence_hashes.json`` is supported as an optional read carrier.
See M4_0_runs/evidence_bundle_decision.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

RUNTIME_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = RUNTIME_DIR.parent          # multi-agent/

CARRIERS = (
    "evidence_index", "domain_summary", "candidate_registry",
    "engine_metrics", "evidence_jsonl", "evidence_bundle",
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _rel(path: Path, workspace: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return "<external>:" + path.as_posix()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class EvidenceView:
    """Deterministic, deduplicated, read-only view of one run's evidence."""

    def __init__(self, run_id: str, workspace: Path):
        self.run_id = run_id
        self.workspace = workspace
        self._entities: dict[str, dict[str, Any]] = {}
        self.carriers: list[dict[str, Any]] = []
        self.unresolved: list[dict[str, Any]] = []
        self.provenance: dict[str, Any] = {}

    def _record_carrier(self, name: str, path: Path, n: int,
                        status: str = "read") -> None:
        self.carriers.append({
            "carrier": name,
            "path": _rel(path, self.workspace),
            "sha256": sha256_file(path),
            "n_records": n,
            "status": status,
        })

    def absorb(self, *, evidence_id: str, carrier: str, source_path: Path,
               verdict: str | None = None, **fields: Any) -> None:
        """Merge ONE real artifact into the view, deduplicated by SHA."""
        ent = self._entities.get(evidence_id)
        if ent is None:
            ent = {
                "evidence_id": evidence_id,
                "sources": [], "carriers": [],
                "verdicts": [], "dataset_versions": [],
                "manifest_sha256": [], "checkpoint_sha256": [],
                "evaluation_run_ids": [], "metrics_files": [],
            }
            self._entities[evidence_id] = ent
        sp = _rel(source_path, self.workspace)
        if sp not in ent["sources"]:
            ent["sources"].append(sp)
        if carrier not in ent["carriers"]:
            ent["carriers"].append(carrier)
        if verdict is not None and verdict not in ent["verdicts"]:
            ent["verdicts"].append(verdict)      # RAW, never recomputed
        for key in ("dataset_versions", "manifest_sha256",
                    "checkpoint_sha256", "evaluation_run_ids", "metrics_files"):
            value = fields.get(key)
            if value is None:
                continue
            values = value if isinstance(value, (list, tuple, set)) else [value]
            for v in values:
                if v is not None and v not in ent[key]:
                    ent[key].append(v)

    def as_dict(self) -> dict[str, Any]:
        entities = []
        for eid in sorted(self._entities):
            e = self._entities[eid]
            entities.append({
                "evidence_id": eid,
                "sources": sorted(e["sources"]),
                "carriers": sorted(e["carriers"]),
                "verdicts": sorted(e["verdicts"]),
                "dataset_versions": sorted(e["dataset_versions"]),
                "manifest_sha256": sorted(e["manifest_sha256"]),
                "checkpoint_sha256": sorted(e["checkpoint_sha256"]),
                "evaluation_run_ids": sorted(e["evaluation_run_ids"]),
                "metrics_files": sorted(e["metrics_files"]),
            })
        # Raw verdict census is a REPORT over stored strings, NOT a new gate.
        census: dict[str, int] = {}
        for e in entities:
            for v in e["verdicts"]:
                census[v] = census.get(v, 0) + 1
        return {
            "schema_version": "m4.0-evidence-view-v1",
            "run_id": self.run_id,
            "n_evidence": len(entities),
            "n_carriers_read": len({c["carrier"] for c in self.carriers}),
            "carriers": sorted(self.carriers,
                               key=lambda c: (c["carrier"], c["path"])),
            "evidence": entities,
            "raw_verdict_census": dict(sorted(census.items())),
            "provenance": self.provenance,
            "unresolved": sorted(self.unresolved, key=lambda x: json.dumps(x,
                                  sort_keys=True)),
            "note": ("read-only; raw verdicts preserved verbatim; no statistical "
                     "verdict recomputed; evidence identity = SHA-256 of real bytes"),
        }


class UnifiedEvidenceReader:
    """Enumerates ALL existing evidence carriers for a run, read-only."""

    def __init__(self, workspace: str | Path | None = None,
                 *, runs_root: str | Path | None = None):
        self.workspace = Path(workspace or DEFAULT_WORKSPACE).resolve()
        self.runs_root = Path(runs_root) if runs_root else \
            self.workspace / "M3_4_runs"

    def query_run(self, run_id: str) -> dict[str, Any]:
        view = EvidenceView(run_id, self.workspace)
        root = self.runs_root / run_id
        if root.is_dir():
            self._read_evidence_index(view, root / "evidence_index.json")
            self._read_domain_files(view, root)
        self._read_engine_metrics(view, run_id)
        self._read_evidence_jsonl(view, run_id)
        self._read_bundle_releases(view, run_id)
        return view.as_dict()

    # -- carrier: evidence_index.json (v1 or v2) ---------------------------
    def _read_evidence_index(self, view: EvidenceView, path: Path) -> None:
        if not path.is_file():
            return
        doc = _json(path)
        digest = sha256_file(path)
        eid = "sha256:" + digest
        n = 0
        for domain, entry in sorted((doc.get("domains") or {}).items()):
            prov = entry.get("ood_provenance") or {}
            ood = prov.get("ood_dataset") or {}
            manifest = ood.get("manifest") or {}
            view.absorb(
                evidence_id=eid, carrier="evidence_index", source_path=path,
                verdict=None,
                dataset_versions=[x for x in (entry.get("dataset_version"),
                                              entry.get("ood_dataset_version"),
                                              ood.get("dataset_version")) if x],
                manifest_sha256=[x for x in (manifest.get("manifest_sha256"),
                                             manifest.get("config_sha256")) if x],
                checkpoint_sha256=(prov.get("checkpoint_sha256") or {}).get("ood"),
                evaluation_run_ids=(prov.get("evaluation_run_ids") or {}).get("ood"),
                metrics_files=[a.get("path") for a in
                               (prov.get("ood_metrics") or {}).get("artifacts", [])
                               if a.get("path")],
            )
            # RAW verdict strings from the decision map (never recomputed).
            for fp, dec in sorted((entry.get("decisions") or {}).items()):
                n += 1
                for key in ("decision", "ood_decision"):
                    v = dec.get(key)
                    if v is not None:
                        view.absorb(evidence_id=eid, carrier="evidence_index",
                                    source_path=path, verdict=str(v))
                if dec.get("ood_dataset_version"):
                    view.absorb(evidence_id=eid, carrier="evidence_index",
                                source_path=path,
                                dataset_versions=[dec["ood_dataset_version"]])
            for a in (prov.get("ood_metrics") or {}).get("artifacts", []):
                # The SAME real artifact is also discovered by the
                # engine_metrics carrier. Use its recorded file SHA as the
                # entity id so the two carriers MERGE into one evidence entity
                # (duplicate recognition) instead of creating two.
                if not a.get("file_sha256"):
                    continue
                ap = Path(str(a.get("path"))) if a.get("path") else path
                view.absorb(
                    evidence_id="sha256:" + a["file_sha256"],
                    carrier="evidence_index", source_path=ap,
                    dataset_versions=[a.get("dataset_version")],
                    checkpoint_sha256=[a.get("checkpoint_sha256")],
                    evaluation_run_ids=[a.get("evaluation_run_id")],
                    metrics_files=[_rel(ap, self.workspace)] if a.get("path") else [],
                )
                n += 1
            view.provenance.setdefault("domains", {})[domain] = {
                "provenance_status": entry.get("provenance_status"),
                "provenance_missing": entry.get("provenance_missing"),
                "ood_dataset_version": entry.get("ood_dataset_version"),
            }
        view._record_carrier("evidence_index", path, n)

    # -- carrier: domain_summary.json + candidate_registry.jsonl -----------
    def _read_domain_files(self, view: EvidenceView, root: Path) -> None:
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            sp = d / "domain_summary.json"
            if sp.is_file():
                doc = _json(sp)
                eid = "sha256:" + sha256_file(sp)
                verdicts = [str(r.get("decision")) for r in
                            (doc.get("records") or []) if r.get("decision")]
                for v in verdicts:
                    view.absorb(evidence_id=eid, carrier="domain_summary",
                                source_path=sp, verdict=v)
                view.absorb(evidence_id=eid, carrier="domain_summary",
                            source_path=sp,
                            dataset_versions=[x for x in (
                                doc.get("dataset_version"),
                                doc.get("ood_dataset_version")) if x])
                view._record_carrier("domain_summary", sp, len(verdicts))
            rp = d / "candidate_registry.jsonl"
            if rp.is_file():
                n = 0
                for line in rp.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    n += 1
                    # Each JSONL line is its own content-addressed event; the
                    # raw status string is preserved verbatim.
                    eid = "sha256:" + _sha256_bytes(line.encode("utf-8"))
                    view.absorb(evidence_id=eid, carrier="candidate_registry",
                                source_path=rp,
                                verdict=(str(rec.get("to"))
                                         if rec.get("to") else
                                         str(rec.get("status"))
                                         if rec.get("status") else None))
                view._record_carrier("candidate_registry", rp, n)

    # -- carrier: engine *metrics*.json artifacts -------------------------
    def _read_engine_metrics(self, view: EvidenceView, run_id: str) -> None:
        base = self.workspace / "M2_7_runs"
        if not base.is_dir():
            return
        for p in sorted(base.rglob("*metrics*.json")):
            camp = p.parent.parent.name
            if not camp.startswith(run_id):     # explicit run_id scoping only
                continue
            try:
                doc = _json(p)
            except (ValueError, OSError):
                view.unresolved.append({"path": _rel(p, self.workspace),
                                        "reason": "metric artifact unreadable"})
                continue
            eid = "sha256:" + sha256_file(p)
            view.absorb(
                evidence_id=eid, carrier="engine_metrics", source_path=p,
                dataset_versions=[doc.get("dataset_version")],
                checkpoint_sha256=[doc.get("checkpoint_sha256")],
                evaluation_run_ids=[doc.get("run_id")],
                metrics_files=[_rel(p, self.workspace)],
            )
            view._record_carrier("engine_metrics", p, 1)

    # -- carrier: append-only evidence.jsonl (streamed; 38 MB) ------------
    def _read_evidence_jsonl(self, view: EvidenceView, run_id: str) -> None:
        p = self.workspace / "research_runtime" / "_data" / "evidence.jsonl"
        if not p.is_file():
            return
        n = 0
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                # Fast pre-filter, then STRUCTURAL verification: never infer
                # identity from a filename or similar text.
                if run_id not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                payload = rec.get("payload") or {}
                result = payload.get("result") or {}
                ids = {str(payload.get("run_id") or ""),
                       str(payload.get("experiment_id") or ""),
                       str(result.get("run_id") or "")}
                if run_id not in ids:
                    continue
                n += 1
                eid = "sha256:" + _sha256_bytes(line.rstrip("\n").encode("utf-8"))
                ckpt = (result.get("checkpoint") or payload.get("checkpoint") or {})
                view.absorb(
                    evidence_id=eid, carrier="evidence_jsonl", source_path=p,
                    dataset_versions=[payload.get("dataset_version"),
                                      result.get("dataset_version")],
                    checkpoint_sha256=[ckpt.get("sha256") if isinstance(ckpt, dict)
                                       else None],
                    evaluation_run_ids=[payload.get("run_id"), result.get("run_id")],
                )
        view._record_carrier("evidence_jsonl", p, n)

    # -- carrier: portable evidence_bundle releases (optional, read-only) --
    def _read_bundle_releases(self, view: EvidenceView, run_id: str) -> None:
        releases = self.workspace / "experiments" / "releases"
        if not releases.is_dir():
            return
        for hp in sorted(releases.rglob("evidence_hashes.json")):
            doc = _json(hp)
            eid = "sha256:" + sha256_file(hp)
            hashes = doc.get("files") or doc.get("hashes") or {}
            n_hashes = len(hashes) if isinstance(hashes, dict) else 0
            # The hash list is a BUNDLE INVENTORY, not a dataset manifest.
            # Record the bundle itself as one evidence entity and keep the
            # declared SHA list in the carrier record only -- never mislabel
            # those values as manifest_sha256.
            view.absorb(evidence_id=eid, carrier="evidence_bundle",
                        source_path=hp)
            view._record_carrier("evidence_bundle", hp, n_hashes)


_DEFAULT_READER: UnifiedEvidenceReader | None = None


def query_run_evidence(run_id: str, *,
                       workspace: str | Path | None = None) -> dict[str, Any]:
    """Single shared entry point used by BOTH ResearchService and Scheduler."""
    if workspace is not None:
        return UnifiedEvidenceReader(workspace).query_run(run_id)
    global _DEFAULT_READER
    if _DEFAULT_READER is None:
        _DEFAULT_READER = UnifiedEvidenceReader()
    return _DEFAULT_READER.query_run(run_id)
