"""M4.3-00 / P0-03 -- Campaign idempotency & crash recovery (engine-free).

Regression being fixed (M4-E2E-01.1 legacy):
``_run_campaign`` checked the per-objective round cap BEFORE looking up the
existing binding. Since ``campaigns_for_objective`` counts BINDINGS, a crashed
run leaves a binding; with ``max_campaign_rounds=1`` the RECOVERY call was then
blocked as "round_limit_reached" -- the crashed run's own round was counted as
an extra one. Also, ANY persisted CampaignRecord produced ``already_submitted``,
so an in-flight (preflighting/running/paused) execution was silently discarded
instead of resumed, and a second fresh launch could resubmit training jobs.

Fix:
  * round cap is enforced ONLY when a NEW binding is about to be created;
  * classify the REAL record state (not_started/preflighting/running/paused/
    terminal); only terminal is a genuine idempotent no-op;
  * in-flight states are RECOVERED via ``resume=True`` from persisted state,
    never freshly submitted;
  * at BOTH service and runner layers the real semantics are documented:
    at-most-once SUBMISSION per deterministic job id + resume-in-place, NOT
    strict exactly-once physical execution.

All tests here are ENGINE-FREE crash injection (FakeRunner / MockAdapter). No
real SciNet++ engine is used or claimed.
"""
from __future__ import annotations

import json

from research_runtime.campaign.schemas import CampaignRecord
from research_runtime.research_service import (
    CAMPAIGN_STATE_NOT_STARTED, CAMPAIGN_STATE_PAUSED,
    CAMPAIGN_STATE_PREFLIGHTING, CAMPAIGN_STATE_RUNNING,
    CAMPAIGN_STATE_TERMINAL, EXECUTION_SEMANTICS, classify_campaign_state,
    make_campaign_id,
)
from research_runtime.research_store.schemas import CampaignBinding

from tests.rt_tests.research_fakes import (
    Harness, FakeRunner, OBJECTIVE_ID, HYPOTHESIS_ID,
)


# --------------------------------------------------------------------- helpers
class _CrashBeforePersistRunner(FakeRunner):
    """Crash on the FIRST run() BEFORE a CampaignRecord is persisted."""

    def __init__(self, store):
        super().__init__(store)
        self.calls = 0

    def run(self, plan, run_spec):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("injected crash before persist")
        return super().run(plan, run_spec)


class _RecordingRunner(FakeRunner):
    """Healthy runner after a restart; records the ``resume`` flag it receives."""

    def __init__(self, store, *, resume_flags: list[bool]):
        super().__init__(store)
        self.resume_flags = resume_flags

    def run(self, plan, run_spec):
        self.resume_flags.append(bool(getattr(run_spec, "resume", False)))
        return super().run(plan, run_spec)


class _CrashAfterInFlightRunner(FakeRunner):
    """Crash AFTER persisting an IN-FLIGHT CampaignRecord (given status).

    Simulates: record created / configs generated / status=running but jobs not
    finished. The second call is a RECOVERY request and must resume, not relaunch.
    """

    def __init__(self, store, *, status: str, resume_flags: list[bool]):
        super().__init__(store)
        self.status = status
        self.calls = 0
        self.resume_flags = resume_flags

    def run(self, plan, run_spec):
        self.calls += 1
        self.resume_flags.append(bool(getattr(run_spec, "resume", False)))
        if self.calls == 1:
            self.store.put(CampaignRecord(
                campaign_id=run_spec.campaign_id, plan_id=plan.plan_id,
                objective_id=plan.objective_id, status=self.status,
                dataset_version=run_spec.dataset_version,
                evaluation_split=run_spec.evaluation_split,
                plan=plan.model_dump(mode="json"),
                run_spec=run_spec.model_dump(mode="json")))
            raise RuntimeError(f"injected crash at status={self.status}")
        return super().run(plan, run_spec)


def _records(h: Harness):
    return list(h.campaign_store.records.values())


def _bindings(h: Harness):
    return h.stores.sessions.campaigns_for_objective(OBJECTIVE_ID)


# --------------------------------------------------------------------- state classification (#3/#4)
def test_classify_campaign_state_five_distinct_states():
    from types import SimpleNamespace
    assert classify_campaign_state(None) == CAMPAIGN_STATE_NOT_STARTED
    for status, expected in [
        ("created", CAMPAIGN_STATE_PREFLIGHTING),
        ("preflighting", CAMPAIGN_STATE_PREFLIGHTING),
        ("running", CAMPAIGN_STATE_RUNNING),
        ("paused", CAMPAIGN_STATE_PAUSED),
        ("completed", CAMPAIGN_STATE_TERMINAL),
        ("completed_with_failures", CAMPAIGN_STATE_TERMINAL),
        ("failed", CAMPAIGN_STATE_TERMINAL),
        ("cancelled", CAMPAIGN_STATE_TERMINAL),
    ]:
        rec = CampaignRecord(
            campaign_id="c", plan_id="p", objective_id="o", status=status,
            dataset_version="d", evaluation_split="validation")
        assert classify_campaign_state(rec) == expected, status
    # Unknown status: the schema Literal cannot construct one, so use a plain
    # object to prove the fail-safe branch (never re-submit unclassified state).
    assert classify_campaign_state(SimpleNamespace(status="mystery_status")) == \
        CAMPAIGN_STATE_RUNNING


def test_execution_semantics_explicitly_not_exactly_once():
    """#7: the engine must NOT claim strict exactly-once execution."""
    assert "at-most-once" in EXECUTION_SEMANTICS
    assert "not strictly" in EXECUTION_SEMANTICS
    assert "exactly-once" in EXECUTION_SEMANTICS


# --------------------------------------------------------------------- fault 1: cap=1, crash then recover
def test_cap1_crash_then_recovery_succeeds():
    """#1/#2: a recovery call for an EXISTING binding is never a new round."""
    h = Harness()
    try:
        h.replace_domain(max_campaign_rounds=1)
        chain = h.chain(grant=True)
        pid, aid = chain["plan"]["plan_id"], chain["approval_id"]

        h.runner = _CrashBeforePersistRunner(h.campaign_store)
        r1 = h.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r1.get("ok") is False, "injected crash must surface"
        assert len(_bindings(h)) == 1, "the crashed run left ONE binding (its round)"

        # Recovery uses a healthy runner. It must NOT be blocked by the cap.
        h.runner = FakeRunner(h.campaign_store)
        r2 = h.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r2.get("ok") is True, r2
        assert r2.get("error_code") != "round_limit_reached", r2
        assert r2.get("campaign_state") == CAMPAIGN_STATE_NOT_STARTED, r2
        assert r2.get("recovered") is False  # no record -> a single first launch
        assert len(_records(h)) == 1
        assert len(_bindings(h)) == 1, "recovery must NOT consume a second round"
    finally:
        h.close()


# --------------------------------------------------------------------- fault 2: cap=1, success then duplicate
def test_cap1_success_then_duplicate_creates_no_new_campaign():
    """#2/#3: a terminal record is a genuine idempotent no-op."""
    h = Harness()
    try:
        h.replace_domain(max_campaign_rounds=1)
        chain = h.chain(grant=True)
        pid, aid = chain["plan"]["plan_id"], chain["approval_id"]

        r1 = h.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r1["ok"] and r1["status"] == "completed"
        r2 = h.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r2.get("ok") is True, r2
        assert r2.get("already_submitted") is True, r2
        assert r2.get("campaign_state") == CAMPAIGN_STATE_TERMINAL, r2
        assert len(_records(h)) == 1
        assert len(_bindings(h)) == 1
    finally:
        h.close()


# --------------------------------------------------------------------- faults 3-5: in-flight record crashes
def _assert_inflight_recovery(status: str, expected_state: str):
    """Shared driver for 'record created' / 'configs generated' / 'running'
    crash points: the second call MUST resume from persisted state."""
    h = Harness()
    try:
        h.replace_domain(max_campaign_rounds=1)
        chain = h.chain(grant=True)
        pid, aid = chain["plan"]["plan_id"], chain["approval_id"]

        resume_flags: list[bool] = []
        crash = _CrashAfterInFlightRunner(
            h.campaign_store, status=status, resume_flags=resume_flags)
        h.runner = crash
        r1 = h.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r1.get("ok") is False, "injected crash must surface"
        assert len(_records(h)) == 1, "an in-flight CampaignRecord is persisted"
        assert _records(h)[0].status == status
        assert resume_flags[-1] is False, "first launch is NOT a resume"

        # Recovery: a fresh runner object (process restart) must be told to
        # RESUME, not to launch a brand-new campaign.
        healthy = _RecordingRunner(h.campaign_store, resume_flags=resume_flags)
        h.runner = healthy
        r2 = h.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r2.get("ok") is True, r2
        assert r2.get("already_submitted") is False, \
            "an in-flight run is NOT already submitted"
        assert r2.get("recovered") is True, r2
        assert r2.get("campaign_state") == expected_state, r2
        assert resume_flags[-1] is True, \
            "recovery must set resume=True (no duplicate training jobs)"
        assert len(_records(h)) == 1, "no second CampaignRecord"
        assert len(_bindings(h)) == 1, "no extra round consumed"
    finally:
        h.close()


def test_campaign_record_created_then_crash_recovers():
    _assert_inflight_recovery("preflighting", CAMPAIGN_STATE_PREFLIGHTING)


def test_config_generated_then_crash_recovers():
    # Configs generated = still "preflighting" in the persisted record; the
    # runner's resume re-verifies config identity against persisted hashes.
    _assert_inflight_recovery("preflighting", CAMPAIGN_STATE_PREFLIGHTING)


def test_running_before_train_submit_then_crash_recovers():
    _assert_inflight_recovery("running", CAMPAIGN_STATE_RUNNING)


def test_paused_record_recovers():
    _assert_inflight_recovery("paused", CAMPAIGN_STATE_PAUSED)


# --------------------------------------------------------------------- fault 6: job id persisted before submit
def test_job_id_is_persisted_before_engine_submit():
    """#6: the deterministic job id is on disk BEFORE the engine call, so the
    'submitted but job-id event not written' window is closed by a probe.

    Uses the REAL CampaignRunner + MockAdapter (still engine-free): pause right
    after the first train submit, then assert the persisted trial carries the
    job id and the persisted-spec event precedes the submit event.
    """
    from tests.rt_tests.campaign_fakes import CampaignTestBase

    class _Probe(CampaignTestBase):
        def runTest(self):
            pass

    t = _Probe()
    t.setUp()
    try:
        adapter = t.adapter(wait_status="timeout")
        runner, adapter, store = t.runner(adapter=adapter)
        plan, spec = t.plan(), t.run_spec()
        runner.run(plan, spec)

        events = [e["event_type"] for e in store.list_events(spec.campaign_id)]
        assert "train_spec_persisted" in events, events
        assert events.index("train_spec_persisted") < events.index("train_submitted"), \
            "job id + spec must be persisted BEFORE the engine submit"
        record = store.load_campaign(spec.campaign_id)
        trial = record.trials[0]
        assert trial.train_job_id, "deterministic job id must be on the trial"
        assert trial.train_spec, "exact ExperimentSpec must be persisted"
        # Recovery may submit jobs that were NEVER submitted before (the paused
        # run stopped after the first train job), but must NEVER re-submit an
        # already-submitted deterministic job id: every entry stays unique and
        # the first job is reused via the existence probe, not submitted again.
        before = list(adapter.submits)
        adapter.opts["wait_status"] = None
        restarted, _, _ = t.runner(adapter=adapter, store=store)
        restarted.resume(spec.campaign_id)
        assert len(adapter.submits) == len(set(adapter.submits)), \
            "a deterministic job id was submitted more than once"
        assert adapter.submits[0] == before[0], \
            "recovery re-waits/re-collects the SAME first job (no re-submit)"
        assert all(j in adapter.submits for j in before)
    finally:
        t.tearDown()


# --------------------------------------------------------------------- fault 7: no double budget consumption
def test_recovery_never_double_consumes_budget():
    """#7: after a crash + recovery, the round cap must still block a genuinely
    NEW campaign -- proving the recovery did not silently free/consume budget."""
    h = Harness()
    try:
        h.replace_domain(max_campaign_rounds=1)
        chain = h.chain(grant=True)
        pid, aid = chain["plan"]["plan_id"], chain["approval_id"]

        h.runner = _CrashBeforePersistRunner(h.campaign_store)
        h.service.run_research_campaign(plan_id=pid, approval_id=aid)  # crash
        h.runner = FakeRunner(h.campaign_store)
        assert h.service.run_research_campaign(
            plan_id=pid, approval_id=aid)["ok"] is True
        assert len(_bindings(h)) == 1

        # A genuinely NEW plan+approval is a NEW binding -> cap must block.
        p2 = h.service.create_research_plan(
            objective_id=OBJECTIVE_ID,
            hypothesis_ids_json=json.dumps([HYPOTHESIS_ID]),
            plan_id="plan_second")
        assert p2["ok"]
        req2 = h.service.request_plan_approval(plan_id="plan_second")
        h.stores.approvals.grant(
            req2["approval_id"], approved_by="alice", source="human",
            current_resource_sha256=p2["plan_sha256"])
        res = h.service.run_research_campaign(
            plan_id="plan_second", approval_id=req2["approval_id"])
        assert res["ok"] is False and res["error_code"] == "round_limit_reached"
        assert len(_bindings(h)) == 1, "blocked new campaign created no binding"
    finally:
        h.close()


# --------------------------------------------------------------------- external binding w/o record still relaunches
def test_external_binding_without_record_relaunches_once():
    """Regression guard for the M4-E2E-01.1 P1 idempotency behavior."""
    h = Harness()
    try:
        h.replace_domain(max_campaign_rounds=1)
        chain = h.chain(grant=True)
        pid, aid, sha = (chain["plan"]["plan_id"], chain["approval_id"],
                         chain["plan"]["plan_sha256"])
        cid = make_campaign_id(pid, aid, sha)
        h.stores.sessions.bind_campaign(CampaignBinding(
            campaign_id=cid, plan_id=pid, approval_id=aid, plan_sha256=sha,
            objective_id=OBJECTIVE_ID, session_id=""))
        r = h.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert r["ok"] and not r.get("already_submitted"), r
        assert r["campaign_id"] == cid == _records(h)[0].campaign_id
        assert len(_bindings(h)) == 1
    finally:
        h.close()
