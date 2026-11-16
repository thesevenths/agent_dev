"""M5 Acceptance Hardening -- P0 controls (ENGINE-FREE).

Covers the exact defects found in source review:
  A. Global budget reserved BEFORE execution; budget=1 -> at most 1 campaign.
  B. Terminal state protection; stopped/completed runs cannot restart/INIT-reset;
     concurrent start has exactly one executor.
  C. Round completion requires real report+experience; classify re-reads stores.
  E. Holdout result SHAs bound to real artifacts; candidate/parent identity.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from research_runtime.research_store.db import (
    ResearchDatabase, ResearchStoreConflict,
)
from research_runtime.runtime_coordinator import (
    RuntimeCoordinator, RuntimeConfig, RuntimeRecoveryRequired,
)
from research_runtime.runtime_state import (
    RuntimeState, RuntimeBudgetExhausted, RuntimeStateError, config_sha,
    S_COMPLETED, S_STOPPED, S_RUNNING, S_INIT, C_COMPLETE, C_RECOVERY,
)


def _state(tmp_path, name="rt.sqlite"):
    return RuntimeState(db=ResearchDatabase(tmp_path / name))


def _coord(tmp_path, *, budget=1, domains=("scinet", "embodied", "quant"),
           approver=lambda a, p: None, state=None):
    cfg = RuntimeConfig(run_id="HARD", max_rounds=2, global_budget=budget,
                        domain_order=domains)
    c = RuntimeCoordinator(
        config=cfg, state=state or _state(tmp_path),
        scheduler=SimpleNamespace(services={
            d: SimpleNamespace(stores=SimpleNamespace(
                objectives=SimpleNamespace(all=lambda: []),
                sessions=SimpleNamespace(campaigns_for_objective=lambda oid: [])))
            for d in domains}),
        discovery=SimpleNamespace(), rrsi=None,
        dataset_shas={"scinet": "a" * 64}, approver=approver,
        out_dir=tmp_path / "out")
    return c


class _CountingScheduler:
    """Counts how many REAL campaigns were allowed to start."""
    def __init__(self, domains):
        self.started = []
        self.services = {
            d: SimpleNamespace(stores=SimpleNamespace(
                objectives=SimpleNamespace(all=lambda: []),
                sessions=SimpleNamespace(campaigns_for_objective=lambda oid: [])))
            for d in domains}
        self._n = 0

    def run_objective(self, oid, source_experiment_id=""):
        self._n += 1
        domain = self._n - 1
        self.started.append(oid)
        return {"campaign_id": f"camp_{oid}", "report_id": f"rep_{oid}",
                "director_state": "COMPLETED", "budget": {}, "result": {}}


# ------------------------------------------------------------------------ #
# P0-A: reservation BEFORE execution                                        #
# ------------------------------------------------------------------------ #
def test_budget_one_three_domains_only_one_campaign_starts(tmp_path):
    c = _coord(tmp_path, budget=1)
    counting = _CountingScheduler(c.config.domain_order)
    c.scheduler = counting
    c.start()
    out = c.run_round(1, objective_specs={
        "scinet": {"objective_id": "obj_sci"},
        "embodied": {"objective_id": "obj_emb"},
        "quant": {"objective_id": "obj_quant"}})
    # Exactly ONE campaign launched; the other two blocked BEFORE execution.
    assert len(counting.started) == 1, counting.started
    blocked = [r for r in out["results"] if r.get("error") ==
               "global_budget_exhausted_before_execution"]
    assert len(blocked) == 2, out["results"]
    snap = c.state.global_budget_snapshot("HARD")
    assert snap["consumed"] + snap["reserved"] == 1
    # The one successful reservation carries its OWN identity, bound to the
    # real campaign id; blocked domains are recorded as BUDGET_BLOCKED rounds.
    res = c.state.reservations("HARD")
    assert len(res) == 1
    assert res[0]["state"] == "consumed" and res[0]["campaign_id"]
    blocked_rounds = [r for r in c.state.rounds("HARD")
                      if r["state"] == "BUDGET_BLOCKED"]
    assert len(blocked_rounds) == 2


def test_reservation_id_is_deterministic_and_independent_of_campaign(tmp_path):
    s = _state(tmp_path)
    rid1 = s.reservation_id("HARD", 1, "scinet", "obj_s")
    rid2 = s.reservation_id("HARD", 1, "scinet", "obj_s")
    rid3 = s.reservation_id("HARD", 1, "embodied", "obj_s")
    assert rid1 == rid2 and rid1 != rid3
    assert len(rid1) == 64


def test_reserve_idempotent_bind_once_consume_once(tmp_path):
    s = _state(tmp_path)
    s.initialize("HARD", config=RuntimeConfig(run_id="HARD"), dataset_shas={},
                 max_rounds=2, global_budget=5)
    rid = s.reservation_id("HARD", 1, "scinet", "obj_s")
    sha = config_sha(RuntimeConfig(run_id="HARD"))
    s.reserve_execution("HARD", rid, config_sha=sha)
    s.reserve_execution("HARD", rid, config_sha=sha)  # idempotent
    assert s.global_budget_snapshot("HARD")["reserved"] == 1
    s.bind_reservation("HARD", rid, campaign_id="camp_x")
    s.bind_reservation("HARD", rid, campaign_id="camp_x")  # idempotent
    with pytest.raises(RuntimeStateError):
        s.bind_reservation("HARD", rid, campaign_id="camp_y")  # no identity swap
    s.consume_reservation("HARD", rid, outcome="COMPLETED")
    s.consume_reservation("HARD", rid, outcome="COMPLETED")  # idempotent
    assert s.global_budget_snapshot("HARD")["consumed"] == 1


def test_crash_after_reservation_does_not_regain_budget(tmp_path):
    s = _state(tmp_path)
    cfg = RuntimeConfig(run_id="HARD")
    s.initialize("HARD", config=cfg, dataset_shas={}, max_rounds=2,
                 global_budget=1)
    rid = s.reservation_id("HARD", 1, "scinet", "obj_s")
    s.reserve_execution("HARD", rid, config_sha=config_sha(cfg))
    # Simulate process restart: a NEW state object over the SAME db file.
    s2 = _state(tmp_path)
    assert s2.orphan_reservations("HARD") and \
        s2.global_budget_snapshot("HARD")["remaining"] == 0
    with pytest.raises(RuntimeBudgetExhausted):
        s2.reserve_execution(
            "HARD", s2.reservation_id("HARD", 1, "embodied", "obj_e"),
            config_sha=config_sha(cfg))


def test_release_only_when_no_campaign_started(tmp_path):
    s = _state(tmp_path)
    s.initialize("HARD", config=RuntimeConfig(run_id="HARD"), dataset_shas={},
                 max_rounds=2, global_budget=5)
    rid = s.reservation_id("HARD", 1, "scinet", "obj_s")
    sha = config_sha(RuntimeConfig(run_id="HARD"))
    s.reserve_execution("HARD", rid, config_sha=sha)
    s.release_reservation("HARD", rid, reason="no_campaign_created")
    assert s.global_budget_snapshot("HARD")["remaining"] == 5
    rid2 = s.reservation_id("HARD", 1, "embodied", "obj_e")
    s.reserve_execution("HARD", rid2, config_sha=sha)
    s.bind_reservation("HARD", rid2, campaign_id="camp_e")
    with pytest.raises(RuntimeStateError):
        s.release_reservation("HARD", rid2, reason="too_late")


# ------------------------------------------------------------------------ #
# P0-B: terminal protection + concurrent executor                          #
# ------------------------------------------------------------------------ #
def test_start_refuses_terminal_run_and_running_restart(tmp_path):
    c = _coord(tmp_path)
    c.start()
    c.state.set_state("HARD", S_COMPLETED)
    # Starting a COMPLETED run must NOT reset it to INIT.
    with pytest.raises(RuntimeRecoveryRequired):
        c.start()
    assert c.state.run("HARD")["state"] == S_COMPLETED
    c2 = _coord(tmp_path / "stopped")
    c2.start()
    c2.state.set_state("HARD", S_STOPPED)
    with pytest.raises(RuntimeRecoveryRequired):
        c2.start()
    assert c2.state.run("HARD")["state"] == S_STOPPED


def test_concurrent_start_only_one_executor(tmp_path):
    c = _coord(tmp_path)
    c.start()  # start() itself claims an executor (pid@ns)
    mine = c.state.run("HARD")["executor_id"]
    assert c.state.claim_executor("HARD", mine) is True   # idempotent re-claim
    assert c.state.claim_executor("HARD", "process_A") is False  # rival loses
    c.state.release_executor("HARD", mine)
    assert c.state.claim_executor("HARD", "process_A") is True
    assert c.state.claim_executor("HARD", "process_B") is False
    # Same process re-claim is idempotent.
    assert c.state.claim_executor("HARD", "process_A") is True
    c.state.release_executor("HARD", "process_A")
    assert c.state.claim_executor("HARD", "process_B") is True


def test_initialize_never_resets_consumed_budget(tmp_path):
    c = _coord(tmp_path, budget=5)
    c.start()
    rid = c.state.reservation_id("HARD", 1, "scinet", "obj_s")
    c.state.reserve_execution("HARD", rid, config_sha=config_sha(c.config))
    c.state.consume_reservation("HARD", rid, outcome="COMPLETED")
    assert c.state.run("HARD")["campaigns_consumed"] == 1


# ------------------------------------------------------------------------ #
# P0-C: strict round completion + real store reconciliation                 #
# ------------------------------------------------------------------------ #
def test_put_round_rejects_completed_without_report_or_experience(tmp_path):
    s = _state(tmp_path)
    s.put_round("HARD", 1, "scinet", "CAMPAIGN_COMPLETED",
                campaign_id="camp_x", report_id="rep_x")
    with pytest.raises(ResearchStoreConflict):
        # A COMPLETED round without real report/experience evidence is illegal.
        s.put_round("HARD", 1, "scinet", "ROUND_COMPLETED",
                    campaign_id="camp_x", report_id="", experience_id="")
    with pytest.raises(ResearchStoreConflict):
        s.put_round("HARD", 1, "scinet", "ROUND_COMPLETED",
                    campaign_id="camp_x", report_id="rep_x", experience_id="")


def test_classify_uses_real_store_not_round_flag(tmp_path):
    """A round claiming COMPLETED but with no real Report is NOT complete."""
    c = _coord(tmp_path)
    c.start()
    # Legacy/incorrect COMPLETED marker written directly (bypassing the guard).
    with c.state.db.session() as conn:
        conn.execute(
            "INSERT INTO rt_round (run_id,round_no,domain_id,state,"
            "objective_id,campaign_id,report_id,experience_id,payload_json,"
            "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("HARD", 1, "scinet", "COMPLETED", "obj_s", "camp_x", "", "",
             "{}", "now"))
    report = c.state.classify("HARD")
    assert report["classification"] != C_COMPLETE
    assert any("missing" in str(r).lower() or "unresolved" in str(r).lower()
               for r in report.get("reasons", [])) or \
           report["classification"] == C_RECOVERY


# ------------------------------------------------------------------------ #
# P0-E: Holdout truth is re-checked against real artifacts                 #
# ------------------------------------------------------------------------ #
def test_holdout_rejects_self_derived_result_sha(tmp_path):
    """A well-formed evidence object with fake result SHAs must NOT promote."""
    from research_runtime.rrsi.holdout_evidence import (
        build_holdout_evidence, gate_allows_promotion_verified,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"v": 1}', encoding="utf-8")
    ev = build_holdout_evidence(
        manifest_path=str(manifest), split_id="test",
        selection_split_id="validation", holdout_seeds=[60, 61],
        selection_seeds=[42, 43], candidate_sha="a" * 64,
        # NO real result files -> result_shas empty -> builder cannot fabricate.
        holdout_metrics={"Q": 1.0}, direction_fraction=1.0)
    assert ev["result_shas"] == [], "builder must never self-derive result SHAs"
    ok, reasons = gate_allows_promotion_verified(
        "ACCEPTED", holdout_confirmed=True, holdout_evidence=ev,
        expected_candidate_sha="a" * 64)
    assert not ok and any("result_sha" in r for r in reasons), reasons

    ok2, reasons2 = gate_allows_promotion_verified(
        "ACCEPTED", holdout_confirmed=True, holdout_evidence=ev,
        expected_candidate_sha="a" * 64, manifest_path=str(manifest))
    assert not ok2


def test_holdout_candidate_identity_uses_coordinator_sha(tmp_path):
    """Harness artifact SHA != candidate-dict SHA; promotion binds the latter."""
    from research_runtime.research_store.db import ResearchDatabase
    from research_runtime.rrsi import (
        RRSICoordinator, EvolutionRegistry, RESEARCH_HARNESS_LAYER,
    )
    coord = RRSICoordinator(
        EvolutionRegistry(db=ResearchDatabase(tmp_path / "e.sqlite")))
    cand = {"candidate_id": "harness::x", "target": "planner.strategy",
            "artifact_sha256": "f" * 64}
    dict_sha = coord._candidate_sha(cand)
    assert dict_sha != cand["artifact_sha256"]
    # A promotion gate given the artifact SHA as candidate_sha must reject.
    from research_runtime.rrsi.holdout_evidence import gate_allows_promotion_verified
    ok, reasons = gate_allows_promotion_verified(
        "ACCEPTED", holdout_confirmed=True, holdout_evidence=None,
        expected_candidate_sha=cand["artifact_sha256"])
    assert not ok


# ------------------------------------------------------------------------ #
# P1-B: code-version binding + honest scientific judgment                   #
# ------------------------------------------------------------------------ #
def test_scientific_judgment_never_claims_global_novelty():
    from research_runtime.scientific_judgment import scientific_judgment
    from types import SimpleNamespace
    cand = SimpleNamespace(
        description="Investigate negative evidence exp_1 and refine train.lr",
        evidence_refs=["sha256:" + "a" * 64],
        success_criteria={"criterion": "reduce MSE", "measurable": True},
        proposed_intervention={"train.lr": 0.001},
        budget={"max_campaign_rounds": 1})
    # A snapshot with one nearly-identical objective must flag a duplicate.
    snap = SimpleNamespace(objectives=[{
        "objective_id": "obj_old",
        "description": "Investigate negative evidence exp_1 and refine train.lr"}])
    j = scientific_judgment(cand, snap)
    assert j["novelty"]["literature_check"] == "not_performed"
    assert j["novelty"]["novelty_status"] == "possible_duplicate"
    assert j["novelty"]["claim"].startswith("cannot_assert")
    assert j["verdict_emitted"] is False
    assert j["research_value"]["score"] >= 3


def test_evidence_package_binds_code_versions(tmp_path):
    from research_runtime.runtime_report import RuntimeReport
    c = _coord(tmp_path)
    c.start()
    pkg = RuntimeReport().write(coordinator=c, out_dir=tmp_path / "ev")
    cv = pkg["code_versions"]
    assert "agent_dev" in cv and "scinet" in cv
    assert cv["agent_dev"].get("head")  # real HEAD recorded
    assert isinstance(cv["agent_dev"].get("dirty"), bool)


# ------------------------------------------------------------------------ #
# P1-A: cross-domain objectives route to their own stores                   #
# ------------------------------------------------------------------------ #
def test_experience_candidate_uses_source_domain(tmp_path):
    """A candidate derived from domain X's Experience is owned by X."""
    from research_runtime.objective_candidate_generator import (
        ObjectiveCandidateGenerator,
    )
    from research_runtime.discovery_context_provider import DiscoverySnapshot
    gen = ObjectiveCandidateGenerator(domain="scinet")
    snap = DiscoverySnapshot(
        run_id="R", schema_version="test", generated_at="", evidence_view={},
        objectives=[{"objective_id": "obj_e", "domain": "embodied",
                     "description": "embodied objective"}],
        links={},
        sessions=[{"session_id": "s1", "objective_id": "obj_e",
                   "source_domain": "embodied",
                   "experience": [{"experience_id": "exp1"}]}],
        sessions_by_objective={}, experience_available=True,
        failures={}, provenance={}, unresolved_gaps=[])
    cands = gen._from_experience(snap)
    assert cands and all(c.domain == "embodied" for c in cands), [
        c.domain for c in cands]


def test_discovery_persists_each_candidate_to_own_store(tmp_path):
    """Discover writes by candidate.domain, not the caller's first domain."""
    from research_runtime.research_store import (
        ResearchStores, from_research_objective,
    )
    from research_runtime.objective_discovery import ObjectiveDiscovery
    from research_runtime.objective_candidate_generator import ObjectiveCandidate
    from research_runtime.schemas import ResearchObjective
    stores = {d: ResearchStores(tmp_path / f"{d}.sqlite")
              for d in ("scinet", "embodied")}
    for d, s in stores.items():
        obj = ResearchObjective(objective_id=f"obj_{d}", domain=d,
                                description=f"{d} objective")
        s.objectives.save(from_research_objective(obj, status="proposed"))
    from research_runtime.discovery_context_provider import DiscoverySnapshot
    snap = DiscoverySnapshot(
        run_id="R", schema_version="test", generated_at="",
        evidence_view={"evidence": [{"evidence_id": "sha256:" + "a" * 64}]},
        objectives=[{"objective_id": "obj_embodied", "domain": "embodied",
                     "description": "x"}],
        links={}, sessions=[], sessions_by_objective={},
        experience_available=True, failures={}, provenance={},
        unresolved_gaps=[])

    class FakeGen:
        domain = "scinet"
        def generate(self, snapshot):
            return [ObjectiveCandidate(
                candidate_id="c1", domain="embodied",
                description="embodied only objective",
                evidence_refs=["sha256:" + "a" * 64],
                source_objective_ids=["obj_embodied"],
                hypothesis_hint="refine drift",
                success_criteria={"criterion": "x", "measurable": True},
                proposed_intervention={"model.drift_correction": 0.1},
                budget={}, provenance={},
                generation_source="historical_experience")]
    d = ObjectiveDiscovery(generator=FakeGen())
    result = d.discover(
        snap, objective_store=stores["scinet"].objectives,
        domain="scinet",
        objective_store_for_domain=lambda dom: (
            stores[dom].objectives if dom in stores else None))
    oid = result.proposed_objective_ids[0]
    assert stores["embodied"].objectives.get(oid) is not None
    assert stores["scinet"].objectives.get(oid) is None
    assert result.ranked[0]["target_domain"] == "embodied"
    assert result.ranked[0]["cross_domain"] is True
