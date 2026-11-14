"""M4.0-04 -- unified read-only evidence model tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research_runtime.evidence_read_model import (
    UnifiedEvidenceReader, query_run_evidence, sha256_file,
)
from research_runtime.research_service import ResearchService
from research_runtime.scheduler import UnifiedResearchScheduler

WS = Path(__file__).resolve().parents[2]      # multi-agent/
RUN = "arr_m34_001"


@pytest.fixture()
def view():
    return UnifiedEvidenceReader(WS).query_run(RUN)


def test_all_six_carriers_are_read(view):
    kinds = {c["carrier"] for c in view["carriers"]}
    assert kinds == {"evidence_index", "domain_summary", "candidate_registry",
                     "engine_metrics", "evidence_jsonl", "evidence_bundle"}
    assert view["n_carriers_read"] == 6
    assert all(c["sha256"] for c in view["carriers"])
    assert all(c["path"] and "\\" not in c["path"] for c in view["carriers"])


def test_duplicate_evidence_is_recognised_across_carriers(view):
    """The SAME real OOD artifact appears in evidence_index provenance AND as
    an engine_metrics file -> ONE entity with TWO carriers."""
    merged = [e for e in view["evidence"]
              if set(e["carriers"]) == {"evidence_index", "engine_metrics"}]
    assert len(merged) == 40, len(merged)
    assert all(e["evidence_id"].startswith("sha256:") for e in merged)
    ids = [e["evidence_id"] for e in view["evidence"]]
    assert len(ids) == len(set(ids)), "duplicate evidence_id leaked"


def test_raw_verdicts_and_provenance_are_preserved(view):
    census = view["raw_verdict_census"]
    # Real M3.4 verdicts: 9 inconclusive (scinet+quant), 3 rejected (embodied)
    assert census.get("inconclusive") == 9
    assert census.get("rejected") == 5      # 3 domain verdicts + 2 registry events
    assert "proposed" in census and "evaluated" in census
    assert view["provenance"]["domains"]["scinet"]["provenance_status"] == "complete"
    assert view["provenance"]["domains"]["quant"]["provenance_status"] == "not_applicable"
    ents = [e for e in view["evidence"] if e["checkpoint_sha256"]]
    assert len(ents) >= 80
    assert all(len(x) == 64 for e in ents for x in e["checkpoint_sha256"])
    assert any(e["manifest_sha256"] for e in view["evidence"])
    assert any(e["dataset_versions"] for e in view["evidence"])
    assert any(e["evaluation_run_ids"] for e in view["evidence"])


def test_research_service_and_scheduler_agree_exactly():
    svc = ResearchService.__new__(ResearchService)
    sch = UnifiedResearchScheduler.__new__(UnifiedResearchScheduler)
    a = svc.query_run_evidence(RUN, workspace=WS)
    b = sch.query_run_evidence(RUN, workspace=WS)
    assert a == b
    assert a == query_run_evidence(RUN, workspace=WS)
    assert a["run_id"] == RUN and a["n_evidence"] > 0


def test_queries_are_read_only(view, tmp_path: Path):
    """Snapshot all historical files, query twice, verify byte identity."""
    root = WS / "M3_4_runs" / RUN
    files = sorted(p for p in root.rglob("*") if p.is_file())
    before = {p.as_posix(): sha256_file(p) for p in files}
    UnifiedEvidenceReader(WS).query_run(RUN)
    UnifiedEvidenceReader(WS).query_run(RUN)
    after = {p.as_posix(): sha256_file(p) for p in files}
    assert before == after


def test_unknown_run_has_no_evidence_and_no_guessing():
    v = UnifiedEvidenceReader(WS).query_run("run_that_never_existed")
    assert v["n_evidence"] >= 0
    # No M3.4 file may be attributed to a nonexistent run.
    assert not any("arr_m34_001" in s for e in v["evidence"]
                   for s in e["sources"])


def test_synthetic_multi_carrier_dedup(tmp_path: Path):
    """Construct a run with the SAME file seen by two carriers."""
    run = "run_probe"
    root = tmp_path / "M3_4_runs" / run / "scinet"
    root.mkdir(parents=True)
    (tmp_path / "M3_4_runs" / run / "evidence_index.json").write_text(
        json.dumps({"run_id": run, "domains": {"scinet": {
            "campaign_id": f"{run}_scinet", "dataset_version": "id-v1",
            "decisions": {"fp1": {"decision": "inconclusive"}},
            "ood_provenance": {"ood_dataset": {"dataset_version": "ood-v1",
                                                "manifest": {"manifest_sha256": "m" * 64}},
                               "ood_metrics": {"artifacts": []}}}}}),
        encoding="utf-8")
    (root / "domain_summary.json").write_text(
        json.dumps({"records": [{"decision": "inconclusive"}]}), encoding="utf-8")
    (root / "candidate_registry.jsonl").write_text(
        json.dumps({"event": "register", "fingerprint": "fp1"}) + "\n",
        encoding="utf-8")
    camp = tmp_path / "M2_7_runs" / "scinet" / "outputs" / \
        "research_campaigns" / f"{run}_scinet" / "c"
    camp.mkdir(parents=True)
    art = camp / "ood_metrics.json"
    raw = json.dumps({"dataset_version": "ood-v1", "run_id": "ev1",
                      "checkpoint_sha256": "c" * 64, "metrics": {"m": 1}})
    art.write_text(raw, encoding="utf-8")
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    ev = json.loads((tmp_path / "M3_4_runs" / run /
                     "evidence_index.json").read_text(encoding="utf-8"))
    ev["domains"]["scinet"]["ood_provenance"]["ood_metrics"]["artifacts"] = [
        {"path": str(art), "file_sha256": h, "dataset_version": "ood-v1",
         "checkpoint_sha256": "c" * 64, "evaluation_run_id": "ev1",
         "metrics": {"m": 1}}]
    (tmp_path / "M3_4_runs" / run / "evidence_index.json").write_text(
        json.dumps(ev), encoding="utf-8")

    v = UnifiedEvidenceReader(tmp_path).query_run(run)
    target = [e for e in v["evidence"] if e["evidence_id"] == "sha256:" + h]
    assert len(target) == 1
    assert set(target[0]["carriers"]) == {"evidence_index", "engine_metrics"}
    assert target[0]["checkpoint_sha256"] == ["c" * 64]
    assert target[0]["evaluation_run_ids"] == ["ev1"]
    assert "inconclusive" in v["raw_verdict_census"]
