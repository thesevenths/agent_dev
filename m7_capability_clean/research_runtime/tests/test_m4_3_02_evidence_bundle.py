"""M4.3-02 -- evidence bundle builder (ENGINE-FREE)."""
from __future__ import annotations

import json

from research_runtime.m4_3.evidence_bundle import build_evidence_bundle
from research_runtime.m4_3.experiment_controller import (
    BudgetGuard, ExperimentController,
)
from research_runtime.m4_3.experiment_registry import (
    ExperimentRegistry, FrozenProtocol,
)


def _protocol() -> FrozenProtocol:
    return FrozenProtocol(snapshot_ids=("s1",), sample_size=0, seed=42,
                          model_version="m", prompt_version="p",
                          ranker_version="r", exclusion_rules=("none",),
                          statistical_method="paired", stopping_rule="all")


def test_bundle_collects_registry_controller_and_campaign_artefacts(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    controller = ExperimentController.create(
        experiment_id="exp_x", db=db, config={}, budget=BudgetGuard())
    controller.preflight()
    reg = ExperimentRegistry(protocol=_protocol(), db=db)
    reg.create_block(block_id="b1", snapshot_id="s1", seed=42)
    for i, arm in enumerate(("A", "B", "C")):
        reg.register_run(block_id="b1", arm=arm)
        reg.finalize(f"b1_{arm}", outcome="negative_finding",
                     campaign_id=f"camp_{arm}", report_id=f"rep_{arm}",
                     experience_id=f"exp_{arm}", gate_steps_passed=12,
                     gate_evidence={"gate_passed": True})

    campaign_root = tmp_path / "campaigns"
    for arm in ("A", "B", "C"):
        trial = campaign_root / f"camp_{arm}" / "baseline_seed_42"
        trial.mkdir(parents=True)
        (trial / "metrics.json").write_text("{}", encoding="utf-8")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "job1.log").write_text("camp_A trained fine", encoding="utf-8")
    (log_dir / "job2.log").write_text("unrelated", encoding="utf-8")

    bundle = build_evidence_bundle(
        db=db, out_dir=tmp_path / "out", experiment_id="exp_x",
        campaign_root=campaign_root, job_log_dir=log_dir)
    assert bundle["n_runs"] == 3
    assert bundle["registry_integrity"] is True
    assert len(bundle["registry_export_sha256"]) == 64
    assert bundle["copied_engine_logs"] == ["job1.log"]
    assert (tmp_path / "out" / "registry_export.json").is_file()
    assert (tmp_path / "out" / "controller_export.json").is_file()
    assert (tmp_path / "out" / "campaigns.json").is_file()
    assert (tmp_path / "out" / "engine_logs" / "job1.log").is_file()
    saved = json.loads((tmp_path / "out" / "evidence_bundle.json")
                       .read_text(encoding="utf-8"))
    assert saved["n_runs"] == 3
    assert all(c["exists"] for c in saved["campaigns"])


def test_missing_campaign_dir_is_reported_not_hidden(tmp_path):
    from research_runtime.research_store.db import ResearchDatabase
    db = ResearchDatabase(tmp_path / "r.sqlite")
    reg = ExperimentRegistry(protocol=_protocol(), db=db)
    reg.create_block(block_id="b1", snapshot_id="s1", seed=42)
    reg.register_run(block_id="b1", arm="A")
    reg.finalize("b1_A", outcome="failed", campaign_id="camp_missing")
    bundle = build_evidence_bundle(
        db=db, out_dir=tmp_path / "out", experiment_id="exp_y",
        campaign_root=tmp_path / "campaigns", copy_logs=False)
    assert any("campaign dir missing" in m for m in bundle["missing_artefacts"])
