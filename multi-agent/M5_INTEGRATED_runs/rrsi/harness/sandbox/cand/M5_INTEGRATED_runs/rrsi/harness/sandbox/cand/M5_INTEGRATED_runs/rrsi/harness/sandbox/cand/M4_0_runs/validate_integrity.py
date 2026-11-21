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


def validate(*, db_path: Path, workspace: Path, out: Path,
             run_id: str = RUN_ID) -> dict[str, Any]:
    run_root = workspace / "M3_4_runs" / run_id
    files = historical_file_list(run_root, workspace)
    before = sha_snapshot(files, workspace)

    stores = ResearchStores(db_path)
    links = EntityLinkStore(stores.db, workspace=workspace)
    view = UnifiedEvidenceReader(workspace).query_run(run_id)
    ev_sha = _sha_map_from_evidence_view(view)

    ev_index = json.loads((run_root / "evidence_index.json").read_text(
        encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    reverse_ok = 0
    sha_ok = 0
    confirmed_candidates = 0
    # P1-02 -- the three relation categories are counted SEPARATELY so a
    # double-source-confirmed candidate can never be mistaken for a native
    # objective->candidate relationship.
    n_candidate_identity_confirmed = 0
    n_objective_candidate_relation_reconstructed = 0
    n_campaign_relation_confirmed = 0
    # P1-03 -- three NON-conflated SHA-verification metrics. The historical
    # single counter (n_evidence_sha_verified == sha_ok) is retained for
    # backward compatibility; its conflated semantics are documented in the
    # report. The three split fields separate:
    #   * unique artifacts  -> distinct sha256 of verified REAL files
    #   * links            -> distinct verified evidence-link ROWS
    #   * candidate-evidence associations -> per (candidate, evidence) pair
    # so a single evidence file shared by many candidates/campaigns is NEVER
    # reported as multiple different files.
    verified_artifact_shas: set[str] = set()
    link_identity_keys: set[tuple] = set()
    n_unique_artifact_sha_verified = 0
    n_link_sha_verified = 0
    n_candidate_evidence_associations_verified = 0

    for domain in sorted(ev_index.get("domains", {})):
        entry = ev_index["domains"][domain]
        oid = f"obj_{run_id}_{domain}"
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

        graph = links.list_objective_graph(oid, run_id)  # retained (audit aid)
        # P1-03 -- verify the domain's campaign->evidence links ONCE per domain.
        # Every candidate in the domain shares the SAME campaign->evidence set,
        # and the run-root-level evidence_index.json / scheduler_state.json are
        # shared across ALL domains. Computing/accumulating inside the candidate
        # loop would count the SAME physical evidence file once per candidate
        # (and 3x across domains), inflating the artifact/file count. We compute
        # the verified set once per domain and derive three NON-conflated
        # metrics (unique artifacts / links / candidate-evidence associations).
        domain_evidence_links = [l for l in links.find_by_entity(
            "campaign", campaign_id, run_id=run_id,
            objective_id=oid, direction="out")
            if l["dst_type"] == "evidence"]
        domain_verified = 0
        for l in domain_evidence_links:
            if not l["source_sha256"] or not l["source_path"]:
                continue
            p = workspace / l["source_path"]
            if sha256_file(p) == l["source_sha256"]:
                domain_verified += 1
                _sha = l["dst_id"].removeprefix("sha256:")
                if _sha not in verified_artifact_shas:
                    verified_artifact_shas.add(_sha)
                    n_unique_artifact_sha_verified += 1
                _key = (l["src_type"], l["src_id"], l["dst_type"],
                        l["dst_id"], l["source_path"])
                if _key not in link_identity_keys:
                    link_identity_keys.add(_key)
                    n_link_sha_verified += 1

        for fp in sorted(entry.get("decisions") or {}):
            cid = "candidate:" + fp
            fwd = links.find_by_entity("candidate", cid, run_id=run_id,
                                       objective_id=oid, direction="out")
            rev = links.find_by_entity("candidate", cid, run_id=run_id,
                                       objective_id=oid, direction="in")
            to_campaign = [l for l in fwd if l["dst_type"] == "campaign"]
            to_objective = [l for l in rev if l["src_type"] == "objective"]
            unresolved = [l for l in fwd if l["status"] == "unresolved"]
            # P1-02 -- the objective->candidate edge is reconstructed by
            # construction (the objective is M4.0-reconstructed). A candidate's
            # double-source identity confirmation is a SEPARATE relation.
            obj_cand_reconstructed = any(
                l["src_type"] == "objective" and l["dst_type"] == "candidate"
                and l["status"] == "reconstructed" for l in rev)
            campaign_confirmed = any(l["status"] == "confirmed"
                                    for l in to_campaign)
            identity_confirmed = fp in reg_fps
            # P1-03 -- shared per-domain verified set (computed once above);
            # the same evidence_links/verified applies to every candidate in
            # the domain because they all attach to the same campaign.
            evidence_links = domain_evidence_links
            verified = domain_verified
            # P1-03 -- n_candidate_evidence_associations_verified counts the
            # (candidate, evidence) pairs: each candidate is associated with the
            # domain's verified evidence, so it can legitimately exceed the
            # unique-artifact count when files are shared across candidates.
            # The historical n_evidence_sha_verified (== sha_ok) retains this
            # exact pre-P1-03 association semantics for backward compatibility.
            sha_ok += verified
            n_candidate_evidence_associations_verified += verified
            reverse_ok += (len(to_objective) > 0)
            ok = (bool(to_campaign) and bool(to_objective) and
                  len(unresolved) == 4 and bool(evidence_links) and
                  verified == len([l for l in evidence_links
                                   if l["source_sha256"]]))
            if identity_confirmed:
                confirmed_candidates += 1
                n_candidate_identity_confirmed += 1
            if obj_cand_reconstructed:
                n_objective_candidate_relation_reconstructed += 1
            if campaign_confirmed:
                n_campaign_relation_confirmed += 1
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
                "fingerprint_confirmed_by_registry": identity_confirmed,
                "candidate_identity_confirmed": identity_confirmed,
                "objective_candidate_relation_reconstructed": obj_cand_reconstructed,
                "campaign_relation_confirmed": campaign_confirmed,
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
    m34 = links.db.rows("SELECT * FROM entity_links WHERE run_id=?", (run_id,))
    cross = [dict(r) for r in m34 if r["objective_id"] == "obj_pendulum_multistep"]
    p106 = links.db.rows(
        "SELECT * FROM entity_links WHERE run_id='p1_06:session_34cfebd14364'")
    p106_cross = [dict(r) for r in p106
                  if str(r["objective_id"]).startswith(f"obj_{run_id}")]

    validation = links.validate_links(workspace=workspace)
    after = sha_snapshot(files, workspace)
    report = {
        "schema_version": "m4.0-integrity-v1",
        "run_id": run_id,
        "n_candidates": len(candidates),
        "n_candidates_confirmed_by_registry": confirmed_candidates,
        "n_candidate_identity_confirmed": n_candidate_identity_confirmed,
        "n_objective_candidate_relation_reconstructed":
            n_objective_candidate_relation_reconstructed,
        "n_campaign_relation_confirmed": n_campaign_relation_confirmed,
        "n_reverse_queries_ok": reverse_ok,
        "n_evidence_sha_verified": sha_ok,
        # P1-03 -- HISTORICAL field retained for backward compatibility. Its
        # value equals n_candidate_evidence_associations_verified (the
        # per-(candidate,evidence) association count). It intentionally keeps
        # its pre-P1-03 semantics and MAY over-count physically identical
        # evidence files that are referenced by multiple candidates/campaigns.
        # Prefer the three split fields below for interpretation.
        "n_evidence_sha_verified_note": (
            "HISTORICAL aggregate == n_candidate_evidence_associations_verified "
            "(per-candidate association count). May over-count shared physical "
            "evidence files. Use the three P1-03 split fields for analysis."),
        "n_unique_artifact_sha_verified": n_unique_artifact_sha_verified,
        "n_link_sha_verified": n_link_sha_verified,
        "n_candidate_evidence_associations_verified":
            n_candidate_evidence_associations_verified,
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
          f"reverse_ok={r['n_reverse_queries_ok']} "
          f"sha_unique_artifacts={r['n_unique_artifact_sha_verified']} "
          f"sha_links={r['n_link_sha_verified']} "
          f"sha_candidate_evidence_associations={r['n_candidate_evidence_associations_verified']} "
          f"sha_verified_historical={r['n_evidence_sha_verified']} "
          f"unresolved={r['n_unresolved_relations']} "
          f"cross_run_isolated={r['cross_run']['isolated']} "
          f"files_unchanged={r['historical_files_unchanged']} "
          f"all_ok={r['all_confirmable_chains_ok']} problems={len(r['problems'])}")


if __name__ == "__main__":
    main()
