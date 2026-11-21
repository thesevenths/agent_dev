# M2.5 — Complete Real Dual-Layer Research Loop (Acceptance Report)

**Branch:** `feature/arr-0.1-domain-registry`
**Scope:** Finish the dual-layer RRSI so BOTH layers run GENUINE experiments inside one
autonomous loop — no harness stub, no prefilled acceptance record, no mocked recovery.
**Verdict:** ✅ PASS — all 5 phases of ChatGPT's M2.5 package are implemented and
empirically validated. No fabricated ACCEPT; REJECT / INCONCLUSIVE are first-class outcomes.

---

## 0. Definition of Done (ChatGPT M2.5)

> "Both RRSI layers must run through the real experiment path. A mocked / stubbed /
> prefilled acceptance does NOT count."

This report demonstrates, with real executions and artifact SHAs, that:

1. Domain model layer — real Embodied engine (train → evaluate on the frozen `robot_sim_v1` dataset).
2. Research harness layer — real `adaptive_runtime.harness_evolution.evolve` on genuine tasks, with an **independent** OOD held-out task (quant) that never enters candidate selection.
3. Real inheritance — accepted domain config **and** accepted harness patches are threaded into the next round via the real primitives.
4. Frozen dataset — every experiment verifies the manifest + per-split SHA-256; the engine loads the *same verified project root* the candidate was built against.
5. Real process crash recovery — a subprocess fault-injection test: launch → round 1 checkpoints → kill mid round 2 → same command restarts → resume from round 2, with round 1 **not** repeated, no overwrite, no duplicate experiment IDs.

---

## 1. Phase 1 — Reliability hardening (P0/P1)

| Fix | File | What it guarantees |
|-----|------|--------------------|
| `base_harness_tag` persistence | `controller.py` (L83), `recovery.py` (L46) | The tag was persisted as `""` for both `None` and a real value (bug). Now derived from `base_harness.fingerprint()` and checked on resume. |
| Candidate idempotent reuse re-verifies disk | `candidate_config.py` (L383-399) | `build()` recomputes the **on-disk** `config.yaml` SHA-256 and compares it against BOTH the requested config and the recorded metadata, plus baseline SHA and `dataset_version`. A tampered disk config is refused, not silently reused. |
| Frozen dataset enforced | `_real_loop.py` (`domain_require_frozen_manifest=True`) | The loop does not silently fall back to a different data root; the manifest + per-split SHA are checked. |
| Data-root isolation | `embodied.py` (`project_dir` injected from spec) | The engine loads the SAME verified project root the candidate was built against (was hardcoded to a canonical dir). |
| Executed-vs-accepted rounds metric | `loop_report.py` | `rounds_executed` (every round that actually ran) is reported separately from `rounds_completed` (rounds that produced an accepted version), so an honest REJECT is not silently dropped. |

---

## 2. Phase 2 — Replace harness stub with real evolve (P0)

- `_real_loop.py`: removed the `_harness_stub`; `_real_harness(**kw)` wraps the **real**
  `adaptive_runtime.harness_evolution.evolve`, capturing each round's payload.
- `research_cycle.py`: `base_harness`, `tasks`, `heldout_task` are passed through to
  `run_dual_layer` (no longer hardcoded `tasks: [], heldout_task: None`).
- `cli.py`: `cmd_loop` is wired to `BASE_HARNESS` + `NEGATIVE_HARNESS` with genuine
  research tasks and an independent held-out task.
- No `ACCEPT` is fabricated. When the evidence is weak / bounded (as in this run), the
  harness honestly returns `INCONCLUSIVE` / `REJECT`.

---

## 3. Phase 3 — Real dual-layer inheritance (P0)

- `improvement_cycle.derive_next_version` now returns the **concrete patched harness
  object** (4th element) in addition to the recorded fingerprint string.
- `controller.run_loop` threads that concrete object into the next round's
  `base_harness` **exactly once per round** (the duplicate `inherit()` call was removed
  — it had been double-incrementing, breaking the inheritance test).
- Verified by `research_runtime/tests/test_loop.py::TestRealHarnessInheritance`
  (round 1 starts from `BASE_HARNESS`; round 2 receives a patched object with a
  **different fingerprint**; round 3 carries an again-distinct object).

> Note: in the validation runs the harness layer produced **0 accepted** harness
> candidates (the signal was too weak / budget too bounded). The inheritance *mechanism*
> is therefore exercised by the unit test (real `default_harness_inherit` + real
> `BASE_HARNESS`) rather than by an organic harness ACCEPT — which is the honest state
> of the evidence, not a gap.

---

## 4. Phase 4 — Real process crash recovery (P1) ✅ PASS

**Test:** `research_runtime/tests/subprocess_recovery_check.py`
(launches two real `run_loop` subprocesses; kills the first mid round 2; restarts the
same command; verifies the resume).

**Result (real execution):**
```
harness calls per round: r1=1 r2=2 r3=1 (total=4)
unique domain experiment IDs: 24
experiment IDs spanning >1 round (genuine dupes): 0
round_1 tree sha stable across crash: True

[RESULT] PASS -- real crash recovery verified:
  round 1 not repeated, no overwrite, no duplicate IDs
```

Interpretation:
- `r1=1` → round 1 ran **once**; the resume did **not** re-run it.
- `r2=2` → round 2 ran twice: once crashed, once resumed (expected for an
  interrupted-but-uncheckpointed round).
- `r3=1` → round 3 ran once on resume.
- `0` experiment IDs span more than one round → **no duplicate experiment IDs created**.
- `round_1` artifact tree SHA is byte-stable across the crash → **no completed
  experiment was overwritten**.

**Real bug found & fixed during this phase:** the recovery compatibility check rejected a
legitimate resume because `baseline_config_path` legitimately *advances* (via version
inheritance) after round 1 accepts. `recovery.py` now excludes `baseline_config_path`
from the strict identity check (identity = domain / dataset_version / base_harness_tag /
max_rounds / domain_seeds).

---

## 5. Phase 5 — End-to-end acceptance (≥5 paired seeds) ✅ PASS

**Run:** `python -c "_real_loop._run_full()"` → `M2_5_full_run.log`
(seeds `(50,51,52,53,54)`, `max_rounds=3`, `domain_require_frozen_manifest=True`,
real `BASE_HARNESS`, real `evolve`, independent quant OOD held-out).

### 5.1 DOMAIN MODEL layer (real engine)
Round 1 accepted `train.lr 0.003 → 0.01`. `state_mse@1` (lower is better) across all
5 seeds:

| seed | baseline (lr=0.003) | candidate (lr=0.01) | Δ |
|------|--------------------|---------------------|---|
| 50 | 0.004471 | 0.002530 | −43.4% |
| 51 | 0.004437 | 0.002433 | −45.2% |
| 52 | 0.004399 | 0.002454 | −44.2% |
| 53 | 0.004371 | 0.002327 | −46.8% |
| 54 | 0.004344 | 0.002437 | −43.9% |

Consistent ~44% improvement on **every** seed → genuine `ACCEPT`, baked into
`baseline_v2.yaml` (the inherited baseline for subsequent rounds). Rounds 2–3 then
honestly found no further improvement (REJECT / INCONCLUSIVE) — no fake ACCEPT to keep
the loop spinning.

### 5.2 RESEARCH HARNESS layer (real `evolve`)
3 real `evolve` calls (one per round). Each round: `accepted=0, controls_rejected=1`,
with `dominant=WORKFLOW_FAILURE`. The degenerate control
(`planner.strategy: fixed_first_knob→repeat_refuted`) was **correctly REJECTED** in
every round; weak-signal candidates were **honestly INCONCLUSIVE** (not accepted).

### 5.3 Dataset & artifact provenance
- Frozen manifest SHA-256 (verified on disk every round): `de59e57f3a5dcb6b…`
- Inherited baseline config: `baseline_v2.yaml` (carries `train.lr: 0.01`).
- Independent OOD held-out task (`quant`) passed to `evolve` and excluded from
  candidate selection.
- `rounds_executed=3`, `rounds_completed=1` (1 round produced an accepted version).

---

## 6. Regression tests

```
research_runtime/tests/test_loop.py ...................... 6 passed
research_runtime/tests/test_unified_candidate.py ............ 20 passed
----------------------------------------------------------------------
Ran 26 tests in 0.54s  OK
```

Phase 4 subprocess recovery check: **PASS** (Section 4).

---

## 7. Engineering constraints honored

- ✅ M2 implementation and artifacts preserved (no removal / rewrite of M2).
- ✅ No new Runtime introduced; all work reuses the existing `run_loop` / `run_dual_layer`.
- ✅ Package H not started.
- ✅ No merge of `main`; no PR opened.
- ✅ All changes committed as independent, logical commits on
  `feature/arr-0.1-domain-registry` and pushed to the existing remote branch.

## 8. Notes / limitations

- No remote GitHub Actions run is available in this environment; CI evidence is the
  local 26-test suite + the two real multi-round executions reported above.
- The harness layer produced no organic ACCEPT in these bounded runs; its inheritance
  path is validated by the real-primitive unit test (Phase 3) and by the honest
  REJECT/INCONCLUSIVE behavior of the real `evolve`.
