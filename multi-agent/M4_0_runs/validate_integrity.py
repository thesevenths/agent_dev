"""M4.0-05 -- cross-system integrity validation for arr_m34_001.

For EACH of the 3 domains' candidates, verify the chain
Objective -> Hypothesis -> Candidate -> Campaign -> Evidence -> Experience,
plus REVERSE lookups, file-level SHA verification, unresolved honesty and
cross-run isolation. READ-ONLY on all historical files; writes only the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO))

from research_runtime.research_store import ResearchStores  # noqa: E402
from research_runtime.research_store.entity_link_store import (  # noqa: E402
    UNRESOLVED_SENTINEL, EntityLinkStore, sha256_file,
)
from research_runtime.evidence_read_model import UnifiedEvidenceReader  # noqa: E402
from M4_0_runs.backfill import historical_file_list, sha_snapshot  # noqa: E402

DEFAULT_DB = REPO / "research_runtime" / "_data" / "research.sqlite"
RUN_ID = "arr_m34_001"


def _sha_map_from_evidence_view(view: dict[str, Any]) -> set[str]:
    return {e["evidence_id"].removeprefix("sha256:") for e in view["evidence"]}


def validate(*, db_path: Path, workspace: Path, out: Path) -> dict[str, Any]:
    run_root = workspace / "M3_4_runs" / RUN_ID
    files = historical_file_list(run_root, workspace)
    before = sha_snapshot(files, workspace)

    stores = ResearchStores(db_path)
    links = EntityLinkStore(stores.db, workspace=workspace)
    view = UnifiedEvidenceReader(workspace).query_run(RUN_ID)
    ev_sha = _sha_map_from_evidence_view(view)

    ev_index = json.loads((run_root / "evidence_index.json").read_text(
        encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    reverse_ok = 0
    sha_ok = 0
    confirmed_candidates = 0

    for domain in sorted(ev_index.get("domains", {})):
        entry = ev_index["domains"][domain]
        oid = f"obj_{RUN_ID}_{domain}"
        campaign_id = entry.get("campaign_id")
        registry = run_root / domain / "candidate_registry.jsonl"
        reg_fps = {str(json.loads(l)["fingerprint"]) for l in
                   registry.read_text(encoding="utf-8").splitlines()
                   if l.strip() and json.loads(l).get("event") == "register"
                   and json.loads(l).get("fingerprint")}

        obj = stores.objectives.get(oid)
        if obj is None:
            problems.append({"domain": domain, "reason": "objective missing"})
        elif obj.status != "reconstructed":
            problems.append({"domain": domain,
                             "reason": "objective not labelled reconstructed"})

        graph = links.list_objective_graph(oid, RUN_ID)
        for fp in sorted(entry.get("decisions") or {}):
            cid = "candidate:" + fp
            fwd = links.find_by_entity("candidate", cid, run_id=RUN_ID,
                                       objective_id=oid, direction="out")
            rev = links.find_by_entity("candidate", cid, run_id=RUN_ID,
                                       objective_id=oid, direction="in")
            to_campaign = [l for l in fwd if l["dst_type"] == "campaign"]
            to_objective = [l for l in rev if l["src_type"] == "objective"]
            unresolved = [l for l in fwd if l["status"] == "unresolved"]
            evidence_links = [l for l in links.find_by_entity(
                "campaign", campaign_id, run_id=RUN_ID,
                objective_id=oid, direction="out")
                if l["dst_type"] == "evidence"]
            # Every evidence link must point at a REAL file whose SHA matches.
            verified = 0
            for l in evidence_links:
                if not l["source_sha256"] or not l["source_path"]:
                    continue
                p = workspace / l["source_path"]
                if sha256_file(p) == l["source_sha256"]:
                    verified += 1
            sha_ok += verified
            reverse_ok += (len(to_objective) > 0)
            ok = (bool(to_campaign) and bool(to_objective) and
                  len(unresolved) == 4 and bool(evidence_links) and
                  verified == len([l for l in evidence_links
                                   if l["source_sha256"]]))
            if fp in reg_fps:
                confirmed_candidates += 1
            if not ok:
                problems.append({"domain": domain, "candidate": cid,
                                 "reason": "incomplete chain",
                                 "to_campaign": len(to_campaign),
                                 "to_objective": len(to_objective),
                                 "unresolved": len(unresolved),
                                 "evidence_links": len(evidence_links),
                                 "sha_verified": verified})
            candidates.append({
                "domain": domain, "candidate": cid,
                "fingerprint_confirmed_by_registry": fp in reg_fps,
                "objective": oid, "campaign": campaign_id,
                "campaign_links": len(to_campaign),
                "objective_links": len(to_objective),
                "evidence_links": len(evidence_links),
                "evidence_sha_verified": verified,
                "unresolved_relations": [l["dst_type"] for l in unresolved],
                "reverse_query_ok": len(to_objective) > 0,
                "chain_ok": ok,
            })

    # Cross-run isolation: no M3.4 link may carry the P1-06 run id and vice versa.
    m34 = links.db.rows("SELECT * FROM entity_links WHERE run_id=?", (RUN_ID,))
    cross = [dict(r) for r in m34 if r["objective_id"] == "obj_pendulum_multistep"]
    p106 = links.db.rows(
        "SELECT * FROM entity_links WHERE run_id='p1_06:session_34cfebd14364'")
    p106_cross = [dict(r) for r in p106
                  if str(r["objective_id"]).startswith("obj_arr_m34_001")]

    validation = links.validate_links(workspace=workspace)
    after = sha_snapshot(files, workspace)
    report = {
        "schema_version": "m4.0-integrity-v1",
        "run_id": RUN_ID,
        "n_candidates": len(candidates),
        "n_candidates_confirmed_by_registry": confirmed_candidates,
        "n_reverse_queries_ok": reverse_ok,
        "n_evidence_sha_verified": sha_ok,
        "n_unresolved_relations": validation["n_unresolved"],
        "link_validation": validation,
        "cross_run": {
            "m34_links": len(m34), "p106_links": len(p106),
            "m34_using_p106_objective": len(cross),
            "p106_using_m34_objective": len(p106_cross),
            "isolated": not cross and not p106_cross,
        },
        "unified_evidence": {
            "n_entities": view["n_evidence"],
            "n_carriers": view["n_carriers_read"],
            "n_multi_carrier": sum(1 for e in view["evidence"]
                                   if len(e["carriers"]) > 1),
            "entity_ids_found_in_view": sum(
                1 for e in view["evidence"]
                if e["evidence_id"].removeprefix("sha256:") in ev_sha),
        },
        "historical_files_unchanged": before == after,
        "n_historical_files_checked": len(before),
        "problems": problems,
        "candidates": candidates,
        "all_confirmable_chains_ok": (not problems and
                                      confirmed_candidates == len(candidates)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True,
                              ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--workspace", type=Path, default=REPO)
    ap.add_argument("--out", type=Path,
                    default=REPO / "M4_0_runs" / "integrity_validation_report.json")
    a = ap.parse_args()
    r = validate(db_path=a.db.resolve(), workspace=a.workspace.resolve(),
                 out=a.out.resolve())
    print(f"candidates={r['n_candidates']} confirmed={r['n_candidates_confirmed_by_registry']} "
          f"reverse_ok={r['n_reverse_queries_ok']} sha_verified={r['n_evidence_sha_verified']} "
          f"unresolved={r['n_unresolved_relations']} "
          f"cross_run_isolated={r['cross_run']['isolated']} "
          f"files_unchanged={r['historical_files_unchanged']} "
          f"all_ok={r['all_confirmable_chains_ok']} problems={len(r['problems'])}")


if __name__ == "__main__":
    main()
