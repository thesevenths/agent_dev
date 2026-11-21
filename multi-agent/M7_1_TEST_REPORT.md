# M7.1 Test Report

## Commands

```bash
py -m pytest -q --basetemp=.pytest_bt_m71_fix9 \
  research_runtime/tests/test_m7_1_defect_regressions.py \
  research_runtime/tests/test_m7_discovery_coordinator.py \
  research_runtime/tests/test_m7_engine_free_and_adversarial.py

py -m pytest -q --basetemp=.pytest_bt_m71_full research_runtime/tests
```

## Results

| Suite | Result |
|---|---|
| M7.1 / Coordinator / engine-free regressions | 21 passed |
| All M7 tests | 32 passed |
| Full `research_runtime/tests` | **789 passed, 12 skipped, 37 subtests passed** |

## New fail-closed coverage

- Research Value returns `NOT_ASSESSED` without independent evidence.
- EIG requires an explicit hypothesis space, prior and observation model.
- Abstract-only no-hit Novelty returns `NOVELTY_UNCERTAIN`.
- Full-text multi-source coverage is required for `LIKELY_NOVEL`.
- Adversarial Critic records counterexamples, confounder controls and model identity.
- Zero-family family confirmation returns `INCONCLUSIVE`, never `REJECTED`.
- Missing pilot effects block scientific confirmation and force `INCONCLUSIVE`.
- Optional formal HarnessLoader binding is recorded with campaign IDs.
- Budget reservations are consumed with outcomes; they no longer remain permanently `bound`.

## Remaining runtime gap

Formal M5 `ResearchService._run_campaign` + `CampaignRunner` lifecycle remains a production integration blocker. M7.1 added an optional formal Loader seam and makes local bypass explicit; it does not falsely report full Dispatcher integration.
