"""M4.3-02.1 P1-03 -- accurate group descriptions, approval boundary,
and missing-value handling for objective quality.

ENGINE-FREE: builds a recorded registry and checks the value report states.
"""
from __future__ import annotations

from research_runtime.m4_3 import ExperimentRegistry, FrozenProtocol
from research_runtime.m4_3.experiment_registry import ARM_DESCRIPTIONS
from research_runtime.m4_3.value_report import generate_value_report
from research_runtime.research_store.db import ResearchDatabase


def _protocol(**kw):
    d = dict(snapshot_ids=("s1",), sample_size=0, seed=42, model_version="m",
             prompt_version="p", ranker_version="r", paired_seeds=(42, 43),
             exclusion_rules=("none",), statistical_method="paired-block",
             stopping_rule="all blocks")
    d.update(kw)
    return FrozenProtocol(**d)


def _populate(db, *, with_quality=False, approved_by="operator",
              approval_source="human"):
    reg = ExperimentRegistry(protocol=_protocol(), db=db)
    for i in range(1, 4):
        bid = f"b{i}"
        reg.create_block(block_id=bid, snapshot_id="s1", seed=40 + i)
        for arm in ("A", "B", "C"):
            reg.register_run(block_id=bid, arm=arm)
            rid = f"{bid}_{arm}"
            qr = ({"total": 2.0, "reviewer_id": "r1",
                   "review_kind": "human_blind", "scores": {}, "reason": "x"}
                  if with_quality else {})
            reg.finalize(
                rid, outcome="negative_finding", quality_review=qr,
                campaign_id=f"c_{rid}", plan_id=f"p_{rid}",
                approval_id=f"a_{rid}", report_id=f"rep_{rid}",
                experience_id=f"e_{rid}", gate_steps_passed=12,
                external_approval=True, compute_units=1.0,
                wall_time_s=10.0, paired_seeds=[42, 43],
                gate_evidence={"gate_passed": True,
                               "approval_source": approval_source,
                               "approval_approved_by": approved_by})
    return reg


# ------------------------------------------------------- group descriptions
def test_group_descriptions_are_accurate_and_not_human():
    a, b, c = ARM_DESCRIPTIONS["A"], ARM_DESCRIPTIONS["B"], ARM_DESCRIPTIONS["C"]
    assert "delegated" in a.lower() and "current" in a.lower()
    assert "delegated" in b.lower() and "histor" in b.lower()
    assert "autonomous" in c.lower() and "discovery" in c.lower()
    # A/B must NOT be presented as real human selection.
    for text in (a, b):
        assert "NOT an independent human" in text


def test_report_design_uses_accurate_arm_descriptions(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db)
    r = generate_value_report(db=db, out_dir=tmp_path, stage="pilot",
                              write=False)
    assert r["design"]["arm_descriptions"] == dict(ARM_DESCRIPTIONS)
    md = ""
    generate_value_report(db=db, out_dir=tmp_path, stage="pilot", write=True)
    md = (tmp_path / "M4_3_PILOT_VALUE_REPORT.md").read_text(encoding="utf-8")
    assert "A manual / B memory-assisted" not in md
    assert "delegated" in md.lower()


# --------------------------------------------------------- approval boundary
def test_approval_boundary_distinguishes_cli_human_and_identity(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, approved_by="operator")
    r = generate_value_report(
        db=db, out_dir=tmp_path, stage="final", write=False,
        protocol={"protocol_sha256": "c" * 64, "approver_id": "operator"})
    ab = r["data_integrity"]["approval_boundary"]
    assert "approval RECORD" in ab["record_level_check"]
    assert ab["approval_sources_labelled"].get("human") == 9
    assert ab["approved_by_identities"] == ["operator"]
    assert ab["independent_human_approval_confirmed"] is False
    assert ab["identity_authentication_confirmed"] is False
    assert "cli_subprocess_authorisation" in ab["distinction"]
    assert "real_human_approval" in ab["distinction"]
    assert "identity_authentication" in ab["distinction"]
    assert "does NOT prove" in ab["claim"]
    # operator + configured approver -> detected as CLI subprocess grant
    assert ab["cli_subprocess_grants_detected"] == 9


def test_approval_boundary_without_config_does_not_claim_human(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, approved_by="someone")
    r = generate_value_report(db=db, out_dir=tmp_path, stage="final",
                              write=False)
    ab = r["data_integrity"]["approval_boundary"]
    assert ab["independent_human_approval_confirmed"] is False
    assert ab["identity_authentication_confirmed"] is False


def test_evidence_chain_carries_approval_identity(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, approved_by="operator")
    r = generate_value_report(db=db, out_dir=tmp_path, stage="final",
                              write=False)
    chain = r["evidence_chains"][0]
    assert chain["approval_source"] == "human"
    assert chain["approval_approved_by"] == "operator"


# ----------------------------------------------- missing objective quality
def test_missing_quality_is_none_with_zero_reviewed_not_a_zero_score(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, with_quality=False)     # NO human reviews
    r = generate_value_report(db=db, out_dir=tmp_path, stage="pilot",
                              write=False)
    q = r["primary"]["objective_quality_observation"]
    for arm in ("A", "B", "C"):
        assert q[arm]["n_reviewed"] == 0
        assert q[arm]["value"] is None
        assert q[arm]["status"] == "missing_no_human_review"
    assert q["n_reviewed_total"] == 0
    # Rendered report must show "missing", never a 0.0 score for quality.
    generate_value_report(db=db, out_dir=tmp_path, stage="pilot", write=True)
    md = (tmp_path / "M4_3_PILOT_VALUE_REPORT.md").read_text(encoding="utf-8")
    assert "missing" in md
    assert "human-blind" in md.lower()


def test_present_quality_reports_actual_value(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    _populate(db, with_quality=True)
    r = generate_value_report(db=db, out_dir=tmp_path, stage="pilot",
                              write=False)
    q = r["primary"]["objective_quality_observation"]
    for arm in ("A", "B", "C"):
        assert q[arm]["n_reviewed"] == 3
        assert q[arm]["value"] == 2.0
        assert q[arm]["status"] == "observed"
