# M7.1 Runtime Integration Audit

Audit baseline: commit `729184c4d8a217d121f577243cc4b1f987e7922a`.

## Target chain vs actual chain

| Component | Status | Evidence at baseline |
|---|---|---|
| M5 Runtime Journal | FULL | `discovery_coordinator.py:150-154` appends each stage; `:250-252` and `:669-685` verify journal. |
| M5 Budget Reservation | PARTIAL | `discovery_coordinator.py:217-232` reserves and binds; no `consume_reservation` call. run_v4 report shows `states=['bound','bound']`. |
| M5 formal Dispatcher (`ResearchService._run_campaign`) | BYPASS | `run_m7_e2e.py:64-72` constructs `DiscoveryCoordinator` directly; `discovery_coordinator.py:106,453,534` calls domain-pack executor. No `ResearchService` import or call. |
| Campaign Lifecycle (`CampaignRunner`/`CampaignStore`/`CampaignRecord`) | BYPASS | `discovery_coordinator.py:450,532` creates campaign ID strings only. `engine_adapter.py:59-117` directly loops `LocalComputeAdapter.submit/wait/collect_result`. |
| M6 HarnessLoader | BYPASS | `run_m7_e2e.py:70` uses git HEAD as harness SHA; `discovery_coordinator.py:91,113,140,152` only labels journal events. `resolve_for_campaign` never called. |
| Formal Promotion | NOT USED | No call from discovery path; only M6 RRSI/test code. |
| Formal Rollback | NOT USED | No call from discovery path; only M6 seam/test code. |
| Evidence binding | PARTIAL | Result SHAs are re-read from disk, but they are not bound to a formal CampaignRecord or Loader-resolved Harness SHA. |

## Actual baseline call chain

```text
run_m7_e2e.main
  -> DiscoveryCoordinator.run
     -> RuntimeState.append / reserve_execution / bind_reservation
     -> domain_pack.run_experiment
        -> LocalComputeAdapter.submit/wait/collect_result
     -> local evaluator / KG / memory
  -> RuntimeState.verify_journal
```

Missing from this chain:

```text
ResearchService._authorize
ResearchService._run_campaign
CampaignRunner.run / resume
CampaignRecord.harness_sha256
HarnessLoader.resolve_for_campaign
CampaignVersionBinding
formal_promote
real_rollback
```

## Consequence

The baseline M7 run proves a local research orchestration path, not full production integration. The baseline report’s phrase “reuse HarnessLoader/formal promotion chain” is not supported by code and must be corrected.

## Required M7.1 action

- Wire a dispatcher adapter into `DiscoveryCoordinator` that invokes the formal `ResearchService` path and returns the real `campaign_id` plus Loader-resolved harness identity.
- Replace the git-HEAD harness label with Loader-returned `harness_sha256`, component SHAs and activation ID.
- Consume budget reservations with real outcomes; release only when no campaign started.
- Add an integration test asserting dispatcher invocation, pinned harness SHA and terminal reservation state.
- If full switching is not possible without rewriting M5/M6, mark runtime integration as `PARTIAL` in the final verdict; do not claim full integration.
