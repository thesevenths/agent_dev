"""M4.0-03 -- Historical evidence backfill (READ-ONLY on historical files).

Builds entity_links from EXPLICIT identifiers only:

  * run_id            -> evidence_index.json "run_id"
  * campaign_id       -> evidence_index.json domains.<d>.campaign_id
  * candidate id      -> intersection of evidence_index decision keys
                         (hyp_fingerprint) AND candidate_registry.jsonl
                         "register" events (fingerprint)
  * evidence id       -> "sha256:<file sha256>" of the REAL file on disk
  * hypothesis / plan / session / experience -> explicit IDs already present
                         in the P1-06 research.sqlite tables

Anything that cannot be confirmed is recorded as an ``unresolved`` link with
an explicit reason. NO guessing from filenames or similar text.

The script NEVER writes to any historical file. It snapshots SHA-256 of every
input before and after and fails closed on any change. It is idempotent: all
links go through EntityLinkStore.add_link (UNIQUE constraint), so a second run
produces zero new links.

Usage:
    python M4_0_runs/backfill.py [--db PATH] [--run-root PATH]
                                 [--workspace PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO))

from research_runtime.research_store import (  # noqa: E402
    ResearchStores, database_snapshot, from_research_objective,
)
from research_runtime.research_store.entity_link_store import (  # noqa: E402
    UNRESOLVED_SENTINEL, EntityLinkStore, sha256_file,
)
from research_runtime.schemas import ResearchObjective  # noqa: E402

M34_RUN = REPO / "M3_4_runs" / "arr_m34_001"
DEFAULT_DB = REPO / "research_runtime" / "_data" / "research.sqlite"
P106_RUN_ID = "p1_06:session_34cfebd14364"


# --------------------------------------------------------------------------- #
# read-only helpers
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _registry_fingerprints(path: Path) -> set[str]:
    """Only explicit 'register' events carry a candidate identity."""
    out: set[str] = set()
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("event") == "register" and rec.get("fingerprint"):
            out.add(str(rec["fingerprint"]))
    return out


def historical_file_list(run_root: Path, workspace: Path) -> list[Path]:
    """Every historical file we READ (for the immutability SHA snapshot)."""
    files: list[Path] = []
    for name in ("evidence_index.json", "scheduler_state.json", "result.json"):
        p = run_root / name
        if p.is_file():
            files.append(p)
    for d in sorted(run_root.iterdir()):
        if not d.is_dir():
            continue
        for name in ("domain_summary.json", "candidate_registry.jsonl"):
            p = d / name
            if p.is_file():
                files.append(p)
    # Engine-written metric artifacts referenced by the OOD provenance.
    camp = workspace / "M2_7_runs" / "scinet" / "outputs" / \
        "research_campaigns" / (run_root.name + "_scinet")
    if camp.is_dir():
        files += sorted(p for p in camp.rglob("*metrics*.json") if p.is_file())
    # The frozen manifests are OUTSIDE the workspace (engine checkout); they
    # are recorded in the report but never used as a guessed entity id.
    return sorted({p.resolve() for p in files})


def sha_snapshot(files: list[Path], workspace: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in files:
        h = sha256_file(p)
        if h is None:
            continue
        try:
            key = p.relative_to(workspace).as_posix()
        except ValueError:
            key = "<external>:" + p.as_posix()
        out[key] = h
    return out


def _db_rows(db_path: Path, query: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(query)]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# backfill
# --------------------------------------------------------------------------- #
def backfill(*, run_root: Path, db_path: Path, workspace: Path,
             report_path: Path) -> dict[str, Any]:
    files = historical_file_list(run_root, workspace)
    before_sha = sha_snapshot(files, workspace)
    db_before = database_snapshot(db_path)

    stores = ResearchStores(db_path)
    links = EntityLinkStore(stores.db, workspace=workspace)
    ev_index = _load_json(run_root / "evidence_index.json")
    run_id = str(ev_index.get("run_id") or run_root.name)

    created: dict[str, int] = {}
    duplicated: dict[str, int] = {}
    unresolved: list[dict[str, Any]] = []
    reconstructed_objectives: list[str] = []

    def add(**kw) -> None:
        res = links.add_link(**kw)
        key = kw["run_id"]
        created[key] = created.get(key, 0) + (res["status"] == "created")
        duplicated[key] = duplicated.get(key, 0) + (res["status"] == "duplicate")

    def rel(p: Path) -> str | None:
        try:
            return p.relative_to(workspace).as_posix()
        except ValueError:
            return None

    def evidence_link(src_type: str, src_id: str, path: Path,
                      relation: str = "has_evidence", **kw) -> str | None:
        digest = sha256_file(path)
        rp = rel(path)
        if digest is None or rp is None:
            return None
        notes = kw.pop("notes", None) or f"real file {rp}"
        add(run_id=kw.pop("run_id", run_id),
            objective_id=kw.pop("objective_id", ""),
            src_type=src_type, src_id=src_id, dst_type="evidence",
            dst_id="sha256:" + digest, relation=relation,
            status="confirmed", basis="sha256",
            source_path=rp, source_sha256=digest,
            notes=notes, **kw)
        return "sha256:" + digest

    # ---- 1. M3.4: one RECONSTRUCTED objective per domain -----------------
    # The historical run produced NO native objective record. M4.0 creates an
    # explicitly labelled reconstruction so the graph has a root; this is
    # recorded as status='reconstructed', NEVER claimed as historical fact.
    for domain in sorted(ev_index.get("domains", {})):
        entry = ev_index["domains"][domain]
        campaign_id = entry.get("campaign_id")
        if not campaign_id:
            continue
        oid = f"obj_{run_id}_{domain}"
        summary_path = run_root / domain / "domain_summary.json"
        src_sha = sha256_file(run_root / "evidence_index.json")
        obj = ResearchObjective(
            objective_id=oid, domain=domain,
            description=(f"M4.0 RECONSTRUCTED root for historical run {run_id} "
                         f"/ campaign {campaign_id}. The M3.4 run did NOT "
                         "persist a native objective."),
            success_criteria={"statistics_protocol":
                              entry.get("statistics_protocol") or {}},
            constraints={"run_id": run_id, "campaign_id": campaign_id,
                         "dataset_version": entry.get("dataset_version")},
            created_at="2026-11-14T00:00:00",
        )
        rec = from_research_objective(
            obj, status="reconstructed",
            provenance={"basis": "m4_0_reconstruction_not_native",
                        "run_id": run_id, "campaign_id": campaign_id,
                        "evidence_index_sha256": src_sha,
                        "source_path": "M3_4_runs/arr_m34_001/evidence_index.json"})
        stores.objectives.save(rec)
        reconstructed_objectives.append(oid)

        # campaign -> evidence (real files, SHA-addressed)
        for p in (run_root / "evidence_index.json",
                  run_root / "scheduler_state.json",
                  summary_path,
                  run_root / domain / "candidate_registry.jsonl"):
            if p.is_file():
                evidence_link("campaign", campaign_id, p,
                              objective_id=oid,
                              notes=f"real file {rel(p)}")

        # OOD / ID metric artifacts produced by the real engine for THIS run.
        camp_dir = (workspace / "M2_7_runs" / domain / "outputs" /
                    "research_campaigns" / campaign_id)
        if camp_dir.is_dir():
            for p in sorted(x for x in camp_dir.rglob("*metrics*.json")
                            if x.is_file()):
                evidence_link("campaign", campaign_id, p, objective_id=oid)

        registry = _registry_fingerprints(
            run_root / domain / "candidate_registry.jsonl")
        decisions = entry.get("decisions") or {}
        for fp in sorted(decisions):
            confirmed = fp in registry
            cand_id = "candidate:" + fp
            add(run_id=run_id, objective_id=oid,
                src_type="objective", src_id=oid, dst_type="candidate",
                dst_id=cand_id, relation="drives",
                status="reconstructed" if not confirmed else "confirmed",
                basis="hyp_fingerprint",
                source_path="M3_4_runs/arr_m34_001/evidence_index.json",
                source_sha256=src_sha,
                notes=("candidate fingerprint present in BOTH evidence_index "
                       "decisions and candidate_registry register events"
                       if confirmed else
                       "fingerprint only in evidence_index; registry confirmation missing"))
            add(run_id=run_id, objective_id=oid,
                src_type="candidate", src_id=cand_id, dst_type="campaign",
                dst_id=campaign_id, relation="evaluated_in",
                status="confirmed", basis="explicit_campaign_id",
                source_path="M3_4_runs/arr_m34_001/evidence_index.json",
                source_sha256=src_sha)
            # Honest gaps: the M3.4 run never wrote a hypothesis / plan /
            # session / experience into research.sqlite. Do NOT guess them.
            for missing_type in ("hypothesis", "experiment_plan",
                                 "research_session", "experience"):
                reason = (f"no explicit {missing_type} id for run {run_id} "
                          f"candidate {fp} in evidence_index.json, "
                          "candidate_registry.jsonl, domain_summary.json or "
                          "research.sqlite; M3.4 used run_dual_layer directly")
                add(run_id=run_id, objective_id=oid,
                    src_type="candidate", src_id=cand_id,
                    dst_type=missing_type, dst_id=UNRESOLVED_SENTINEL,
                    relation=f"{missing_type}_unknown", status="unresolved",
                    basis="no_explicit_id", notes=reason)
                unresolved.append({"run_id": run_id, "domain": domain,
                                   "candidate": cand_id,
                                   "missing_entity": missing_type,
                                   "reason": reason})

    # ---- 2. P1-06 native chain (explicit IDs already in the DB) ----------
    p106 = _p106_chain(db_path, links, workspace)
    unresolved += p106["unresolved"]
    for k, v in p106["created"].items():
        created[k] = created.get(k, 0) + v
    for k, v in p106["duplicated"].items():
        duplicated[k] = duplicated.get(k, 0) + v

    # ---- 3. verify historical files unchanged ----------------------------
    after_sha = sha_snapshot(files, workspace)
    unchanged = before_sha == after_sha
    if not unchanged:
        raise RuntimeError("historical input files changed during backfill")

    db_after = database_snapshot(db_path)
    validation = links.validate_links(workspace=workspace)
    report = {
        "schema_version": "m4.0-backfill-v1",
        "run_id": run_id,
        "source": "read-only historical files + existing research.sqlite rows",
        "sources_read": [
            {"path": (p.relative_to(workspace).as_posix()
                      if _is_relative(p, workspace) else p.as_posix()),
             "sha256": sha256_file(p)} for p in files],
        "n_files_read": len(files),
        "historical_files_unchanged": unchanged,
        "objectives_reconstructed": reconstructed_objectives,
        "links_created": created,
        "links_duplicated_on_repeat": duplicated,
        "n_unresolved": len(unresolved),
        "unresolved": unresolved,
        "link_validation": validation,
        "db_before": {"statistics": db_before["statistics"],
                      "table_count": db_before["table_count"]},
        "db_after": {"statistics": db_after["statistics"],
                     "table_count": db_after["table_count"]},
        "file_sha256": after_sha,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True,
                                      ensure_ascii=False), encoding="utf-8")
    return report


def _is_relative(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def _p106_chain(db_path: Path, links: EntityLinkStore,
                workspace: Path) -> dict[str, Any]:
    """Backfill the ONE real P1-06 chain, in its OWN run namespace."""
    created: dict[str, int] = {}
    duplicated: dict[str, int] = {}
    unresolved: list[dict[str, Any]] = []
    run_id = P106_RUN_ID

    def add(**kw):
        res = links.add_link(**kw)
        created[run_id] = created.get(run_id, 0) + (res["status"] == "created")
        duplicated[run_id] = duplicated.get(run_id, 0) + (res["status"] == "duplicate")

    hypo = _db_rows(db_path, "SELECT DISTINCT hypothesis_id, objective_id, "
                             "content_sha256 FROM hypothesis_versions")
    plans = _db_rows(db_path, "SELECT plan_id, objective_id, plan_sha256, "
                              "hypothesis_ids FROM plan_versions")
    binds = _db_rows(db_path, "SELECT * FROM campaign_bindings")
    exp = _db_rows(db_path, "SELECT experience_id, session_id, objective_id "
                            "FROM research_experience")
    sess = _db_rows(db_path, "SELECT session_id, objective_id FROM research_sessions")
    reports = _db_rows(db_path, "SELECT report_id, session_id, report_sha256 "
                                "FROM research_reports")

    objective_ids = sorted({r["objective_id"] for r in hypo + plans + sess + exp
                            if r.get("objective_id")})
    if len(objective_ids) != 1:
        unresolved.append({"run_id": run_id, "missing_entity": "objective",
                           "reason": "0 or >1 distinct objective ids in DB; "
                                     "refusing to guess a single root"})
        return {"created": created, "duplicated": duplicated,
                "unresolved": unresolved}
    oid = objective_ids[0]

    # SHA fail-closed: the SAME plan_sha256 must be consistent across all
    # explicit references; a mismatch is reported, never silently reconciled.
    plan_hashes = {r["plan_sha256"] for r in plans + binds if r.get("plan_sha256")}
    if len(plan_hashes) != 1:
        unresolved.append({"run_id": run_id, "missing_entity": "experiment_plan",
                           "reason": f"inconsistent plan_sha256 {sorted(plan_hashes)}"})
        return {"created": created, "duplicated": duplicated,
                "unresolved": unresolved}
    plan_sha = next(iter(plan_hashes))

    for h in hypo:
        add(run_id=run_id, objective_id=oid,
            src_type="objective", src_id=oid, dst_type="hypothesis",
            dst_id=h["hypothesis_id"], relation="drives",
            basis="explicit_hypothesis_id",
            notes="hypothesis_versions.hypothesis_id")
    for p in plans:
        ids = json.loads(p.get("hypothesis_ids") or "[]")
        pid = p["plan_id"]
        # Preserve EVERY original entity ID verbatim: one link per explicit
        # hypothesis_id. Never synthesise a comma-joined pseudo-id.
        for hid in ids:
            add(run_id=run_id, objective_id=oid,
                src_type="hypothesis", src_id=hid,
                dst_type="experiment_plan", dst_id=pid, relation="planned_as",
                basis="plan_versions.hypothesis_ids")
    for b in binds:
        add(run_id=run_id, objective_id=oid,
            src_type="experiment_plan", src_id=b["plan_id"],
            dst_type="campaign", dst_id=b["campaign_id"],
            relation="executed_as", basis="campaign_bindings.plan_id",
            notes=f"plan_sha256={plan_sha}")
        if b.get("session_id"):
            add(run_id=run_id, objective_id=oid,
                src_type="campaign", src_id=b["campaign_id"],
                dst_type="research_session", dst_id=b["session_id"],
                relation="part_of", basis="campaign_bindings.session_id")
    for s in sess:
        add(run_id=run_id, objective_id=oid,
            src_type="objective", src_id=oid,
            dst_type="research_session", dst_id=s["session_id"],
            relation="has_session", basis="research_sessions.session_id")
    for e in exp:
        add(run_id=run_id, objective_id=oid,
            src_type="research_session", src_id=e["session_id"],
            dst_type="experience", dst_id=e["experience_id"],
            relation="accumulated", basis="research_experience.experience_id")
        add(run_id=run_id, objective_id=oid,
            src_type="objective", src_id=oid,
            dst_type="experience", dst_id=e["experience_id"],
            relation="informed_by", basis="research_experience.experience_id")
    for r in reports:
        add(run_id=run_id, objective_id=oid,
            src_type="research_session", src_id=r["session_id"],
            dst_type="evidence", dst_id="sha256:" + str(r["report_sha256"]),
            relation="published", basis="research_reports.report_sha256",
            notes="report content hash; report_id=" + str(r["report_id"]))
    return {"created": created, "duplicated": duplicated,
            "unresolved": unresolved}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--run-root", type=Path, default=M34_RUN)
    ap.add_argument("--workspace", type=Path, default=REPO)
    ap.add_argument("--out", type=Path,
                    default=REPO / "M4_0_runs" / "backfill_report.json")
    a = ap.parse_args()
    rep = backfill(run_root=a.run_root.resolve(), db_path=a.db.resolve(),
                   workspace=a.workspace.resolve(), report_path=a.out.resolve())
    print(f"run_id={rep['run_id']} files_read={rep['n_files_read']} "
          f"created={rep['links_created']} "
          f"duplicated={rep['links_duplicated_on_repeat']} "
          f"unresolved={rep['n_unresolved']} "
          f"unchanged={rep['historical_files_unchanged']} "
          f"validation_ok={rep['link_validation']['ok']}")


if __name__ == "__main__":
    main()
