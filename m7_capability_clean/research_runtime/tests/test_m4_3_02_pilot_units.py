"""M4.3-02 -- pilot runner mechanics that do NOT need the engine.

The real A/B/C execution is proven by the opt-in real-engine test. Here we pin
the deterministic pieces the execution depends on: evidence partitioning,
the pre-registered selection rule, and the evidence boundary of a block.
"""
from __future__ import annotations

import pytest

from research_runtime.m4_3.evidence_boundary import validate_evidence_partition
from research_runtime.m4_3.experiment_controller import (
    BudgetGuard, ExperimentController,
)
from research_runtime.m4_3.pilot_runner import (
    PilotConfig, PilotRunner, SelectionError, delegated_select, preflight,
)
from research_runtime.objective_candidate_generator import ObjectiveCandidate


def _candidate(cid: str, refs: list[str], *, metric: str | None = "latent_mse@1",
               direction: str | None = "minimize") -> ObjectiveCandidate:
    crit = {}
    if metric:
        crit["metric"] = metric
    if direction:
        crit["direction"] = direction
    return ObjectiveCandidate(
        candidate_id=cid, domain="scinet", description=f"candidate {cid}",
        evidence_refs=refs, success_criteria=crit,
        proposed_intervention={"train.lr": 0.0008},
        source_objective_ids=[], hypothesis_hint="test hint",
        budget={"max_campaign_rounds": 1}, provenance={"source": "unit_test"})


def test_evidence_split_covers_every_id_exactly_once(tmp_path):
    runner = _runner(tmp_path)
    ids = [f"sha256:{i:04d}" for i in range(118)]
    chunks = runner._split_evidence(ids, 4)
    assert len(chunks) == 4
    flat = [i for chunk in chunks for i in chunk]
    assert flat == ids                      # order preserved, nothing dropped
    assert len(set(flat)) == len(ids)
    assert all(len(chunk) >= 1 for chunk in chunks)


def test_evidence_split_respects_an_explicit_chunk_size(tmp_path):
    cfg = PilotConfig(experiment_id="probe", n_blocks=4, evidence_per_block=10)
    runner = _runner(tmp_path, cfg=cfg)
    chunks = runner._split_evidence([f"sha256:{i}" for i in range(25)], 4)
    assert [len(c) for c in chunks][:3] == [10, 10, 5]


def test_delegated_selection_is_deterministic_and_visibility_aware():
    invisible = _candidate("c_invisible", ["sha256:other"], metric=None,
                           direction=None)
    visible = _candidate("c_visible", ["sha256:mine"])
    chosen, why = delegated_select([invisible, visible],
                                   visible_ids={"sha256:mine"})
    assert chosen.candidate_id == "c_visible"
    assert "resolvable_refs=1" in why
    # Same input -> same decision (pre-registered, not random).
    assert delegated_select([invisible, visible],
                            visible_ids={"sha256:mine"})[0].candidate_id == \
        "c_visible"
    with pytest.raises(SelectionError):
        delegated_select([], visible_ids=set())


def test_delegated_selection_falls_back_to_falsifiability_then_order():
    bare = _candidate("a_bare", ["sha256:mine"], metric=None, direction=None)
    rich = _candidate("b_rich", ["sha256:mine"])
    chosen, _why = delegated_select([bare, rich], visible_ids={"sha256:mine"})
    assert chosen.candidate_id == "b_rich"
    first = _candidate("x_first", ["sha256:mine"])
    second = _candidate("y_second", ["sha256:mine"])
    assert delegated_select([first, second],
                            visible_ids={"sha256:mine"})[0].candidate_id == \
        "x_first"


def test_block_spec_keeps_current_and_history_disjoint(tmp_path):
    runner = _runner(tmp_path)
    spec = runner._block_spec("pilot_block_1", "snap::block_1", 42,
                              ["sha256:aaa", "sha256:bbb"], 1, [])
    assert spec.current_evidence_ids and spec.historical_ids == ()
    partition = validate_evidence_partition(
        snapshot_id=spec.snapshot_id, current_evidence=spec.current_evidence,
        historical_experience=spec.historical_experience,
        historical_evidence=spec.historical_evidence,
        historical_failure=spec.historical_failure,
        explicit_current_ids=spec.current_evidence_ids)
    assert partition.current_ids == spec.current_evidence_ids
    assert spec.current_evidence["n_evidence"] == 2
    # Two different block subsets must produce different snapshot markers.
    other = runner._block_spec("pilot_block_2", "snap::block_2", 43,
                               ["sha256:ccc"], 2, [])
    assert other.current_evidence_ids != spec.current_evidence_ids


def test_preflight_refuses_without_a_real_engine(tmp_path):
    cfg = PilotConfig(experiment_id="probe", n_blocks=1)
    rep = preflight(cfg, service=None)
    assert rep.real_evidence is False
    assert rep.ready is False
    payload = rep.to_dict()
    assert payload["ready"] is False and "detail" in payload


def test_pilot_protocol_is_not_confirmatory_and_freezes_its_snapshots(tmp_path):
    runner = _runner(tmp_path, cfg=PilotConfig(experiment_id="probe",
                                              n_blocks=3))
    protocol = runner.protocol
    assert protocol.confirmatory is False
    assert protocol.snapshot_ids == ("arr_m34_001::block_1",
                                     "arr_m34_001::block_2",
                                     "arr_m34_001::block_3")
    assert protocol.paired_seeds == (42, 43)
    assert len(protocol.freeze_sha256()) == 64


# --------------------------------------------------------------- helpers
def _runner(tmp_path, cfg: PilotConfig | None = None) -> PilotRunner:
    from tests.rt_tests.research_fakes import Harness
    cfg = cfg or PilotConfig(experiment_id="probe", n_blocks=4)
    harness = Harness()
    controller = ExperimentController.create(
        experiment_id="probe", db=harness.stores.db, config=cfg.to_dict(),
        budget=BudgetGuard())
    runner = PilotRunner(service=harness.service, config=cfg,
                         out_dir=tmp_path, controller=controller)
    runner._fakes = harness          # keep the temporary directory alive
    return runner
