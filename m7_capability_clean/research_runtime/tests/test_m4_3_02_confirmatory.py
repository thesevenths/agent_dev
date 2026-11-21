"""M4.3-02 -- confirmatory protocol freeze + human-gated start.

ENGINE-FREE. Proves the confirmatory stage cannot be started by accident:
a complete 20-field protocol, a matching SHA, a HUMAN approval bound to that
SHA, a controller that is explicitly awaiting confirmatory approval, and a
sample size the pilot actually supports.
"""
from __future__ import annotations

import json

import pytest

from research_runtime.m4_3.confirmatory_protocol_builder import (
    APPROVAL_FILENAME, PROTOCOL_FILENAME, ProtocolBlocked, ProtocolNotApproved,
    approve_confirmatory_protocol, build_confirmatory_protocol,
    freeze_confirmatory_protocol, load_approval, load_frozen_protocol,
)
from research_runtime.m4_3.confirmatory_runner import ConfirmatoryRunner
from research_runtime.m4_3.experiment_controller import (
    BudgetGuard, ExperimentController,
)
from research_runtime.m4_3.pilot_runner import PilotConfig
from research_runtime.m4_3.protocol_schema import (
    CONFIRMATORY_REQUIRED_FIELDS, missing_confirmatory_fields,
)


def _pilot(status: str = "estimated", recommended: int | None = 4) -> dict:
    return {
        "n_blocks": 4,
        "data_integrity": {"snapshot_ids": ["snap_1", "snap_2"]},
        "blocks": [{"snapshot_id": "snap_1"}, {"snapshot_id": "snap_2"}],
        "sample_size": {"status": status, "n_blocks_recommended": recommended,
                        "contrasts_without_variance": (
                            [] if status == "estimated" else ["C-B"])},
    }


def _build(**overrides):
    kwargs = dict(
        pilot_analysis=_pilot(), snapshot_ids=["snap_1", "snap_2"],
        paired_seeds=[42, 43], sample_size=4,
        campaign_budget={"max_jobs": 8, "max_seeds": 2},
        model_version="m1", prompt_version="p1", ranker_version="r1",
        domain_policy_version="dp1", smallest_effect_of_interest=0.5)
    kwargs.update(overrides)
    return build_confirmatory_protocol(**kwargs)


def test_protocol_has_all_twenty_frozen_fields():
    protocol = _build()
    assert missing_confirmatory_fields(protocol) == ()
    assert set(CONFIRMATORY_REQUIRED_FIELDS) <= set(
        protocol.frozen_dict())
    assert len(protocol.freeze_sha256()) == 64


def test_protocol_is_refused_without_pilot_support():
    with pytest.raises(ProtocolBlocked) as exc:
        _build(pilot_analysis={})
    assert "pilot analysis" in str(exc.value)
    with pytest.raises(ProtocolBlocked) as exc:
        _build(pilot_analysis=_pilot("evidence_insufficient", None))
    assert "does not support a sample-size estimate" in str(exc.value)
    with pytest.raises(ProtocolBlocked):
        _build(sample_size=2)          # below the pilot recommendation
    with pytest.raises(ProtocolBlocked):
        _build(smallest_effect_of_interest=0.0)
    with pytest.raises(ProtocolBlocked):
        _build(snapshot_ids=[])
    with pytest.raises(ProtocolBlocked):
        _build(paired_seeds=[])


def test_freeze_is_append_only_and_idempotent(tmp_path):
    protocol = _build()
    first = freeze_confirmatory_protocol(protocol, out_dir=tmp_path)
    assert first["protocol_sha256"] == protocol.freeze_sha256()
    assert first["current_version"] == 1
    again = freeze_confirmatory_protocol(protocol, out_dir=tmp_path)
    assert again == first                                   # idempotent
    changed = _build(sample_size=6, pilot_analysis=_pilot("estimated", 4))
    second = freeze_confirmatory_protocol(changed, out_dir=tmp_path)
    assert second["current_version"] == 2
    assert second["protocol_sha256"] != first["protocol_sha256"]
    stored = json.loads((tmp_path / PROTOCOL_FILENAME).read_text(
        encoding="utf-8"))
    assert len(stored["versions"]) == 2
    assert stored["versions"][0]["protocol_sha256"] == first["protocol_sha256"]
    loaded = load_frozen_protocol(tmp_path)
    assert loaded.freeze_sha256() == second["protocol_sha256"]


def test_approval_is_bound_to_the_exact_sha(tmp_path):
    protocol = _build()
    frozen = freeze_confirmatory_protocol(protocol, out_dir=tmp_path)
    with pytest.raises(ProtocolNotApproved):
        load_approval(protocol_sha256=frozen["protocol_sha256"],
                      out_dir=tmp_path)                 # nothing approved yet
    with pytest.raises(ProtocolNotApproved):
        approve_confirmatory_protocol(protocol_sha256="b" * 64,
                                      approved_by="alice", out_dir=tmp_path)
    with pytest.raises(ProtocolNotApproved):
        approve_confirmatory_protocol(protocol_sha256=frozen["protocol_sha256"],
                                      approved_by="   ", out_dir=tmp_path)
    approval = approve_confirmatory_protocol(
        protocol_sha256=frozen["protocol_sha256"], approved_by="alice",
        out_dir=tmp_path)
    assert approval["status"] == "approved"
    assert approval["protocol_sha256"] == frozen["protocol_sha256"]
    assert load_approval(protocol_sha256=frozen["protocol_sha256"],
                         out_dir=tmp_path)["approved_by"] == "alice"
    # An approval for one SHA must never authorise another.
    with pytest.raises(ProtocolNotApproved):
        load_approval(protocol_sha256="c" * 64, out_dir=tmp_path)


def test_confirmatory_runner_refuses_an_unapproved_or_mismatched_protocol(
        tmp_path):
    from tests.rt_tests.research_fakes import Harness

    harness = Harness()
    try:
        protocol = _build()
        frozen = freeze_confirmatory_protocol(protocol, out_dir=tmp_path)
        cfg = PilotConfig(experiment_id="exp_conf", n_blocks=4,
                          paired_seeds=(42, 43), require_review_complete=True)
        controller = ExperimentController.create(
            experiment_id="exp_conf", db=harness.stores.db,
            config=cfg.to_dict(), budget=BudgetGuard())
        controller.preflight()
        controller.freeze_snapshots(["snap_1", "snap_2"])
        with pytest.raises(ProtocolNotApproved) as exc:
            ConfirmatoryRunner(
                service=harness.service, config=cfg, out_dir=tmp_path / "run",
                controller=controller, protocol=protocol,
                approval={"protocol_sha256": "d" * 64})
        assert "different protocol SHA" in str(exc.value)
        # A protocol that is not confirmatory is refused outright.
        from research_runtime.m4_3.experiment_registry import FrozenProtocol
        with pytest.raises(ProtocolBlocked):
            ConfirmatoryRunner(
                service=harness.service, config=cfg, out_dir=tmp_path / "run",
                controller=controller,
                protocol=FrozenProtocol(snapshot_ids=("snap_1",)),
                approval={"protocol_sha256": "x"})
        # A sample size that disagrees with the frozen design is refused.
        with pytest.raises(ProtocolBlocked) as exc:
            ConfirmatoryRunner(
                service=harness.service, config=PilotConfig(
                    experiment_id="exp_conf", n_blocks=2,
                    require_review_complete=True),
                out_dir=tmp_path / "run", controller=controller,
                protocol=protocol,
                approval={"protocol_sha256": frozen["protocol_sha256"]})
        assert "frozen sample size" in str(exc.value)
    finally:
        harness.close()


def test_confirmatory_run_never_starts_automatically(tmp_path):
    from tests.rt_tests.research_fakes import Harness

    harness = Harness()
    try:
        protocol = _build()
        frozen = freeze_confirmatory_protocol(protocol, out_dir=tmp_path)
        cfg = PilotConfig(experiment_id="exp_conf2", n_blocks=4,
                          paired_seeds=(42, 43), require_review_complete=True)
        controller = ExperimentController.create(
            experiment_id="exp_conf2", db=harness.stores.db,
            config=cfg.to_dict(), budget=BudgetGuard())
        controller.preflight()
        runner = ConfirmatoryRunner(
            service=harness.service, config=cfg, out_dir=tmp_path / "run",
            controller=controller, protocol=protocol,
            approval={"protocol_sha256": frozen["protocol_sha256"]})
        # The controller is NOT awaiting confirmatory approval -> refuse.
        with pytest.raises(ProtocolNotApproved) as exc:
            runner.run()
        assert "never start automatically" in str(exc.value)
    finally:
        harness.close()


def test_frozen_protocol_round_trips_through_disk(tmp_path):
    protocol = _build()
    freeze_confirmatory_protocol(protocol, out_dir=tmp_path)
    loaded = load_frozen_protocol(tmp_path)
    assert loaded.frozen_dict() == protocol.frozen_dict()
    assert loaded.freeze_sha256() == protocol.freeze_sha256()
    stored = json.loads((tmp_path / PROTOCOL_FILENAME).read_text(
        encoding="utf-8"))
    assert stored["protocol_sha256"] == protocol.freeze_sha256()
    assert stored["required_fields"] == list(CONFIRMATORY_REQUIRED_FIELDS)


def test_approval_file_is_not_reused_for_an_edited_protocol(tmp_path):
    protocol = _build()
    frozen = freeze_confirmatory_protocol(protocol, out_dir=tmp_path)
    approve_confirmatory_protocol(protocol_sha256=frozen["protocol_sha256"],
                                  approved_by="alice", out_dir=tmp_path)
    assert (tmp_path / APPROVAL_FILENAME).is_file()
    # A DIFFERENT protocol (e.g. more blocks) must not inherit the approval.
    other = _build(sample_size=6)
    freeze_confirmatory_protocol(other, out_dir=tmp_path)
    with pytest.raises(ProtocolNotApproved):
        load_approval(protocol_sha256=other.freeze_sha256(), out_dir=tmp_path)
