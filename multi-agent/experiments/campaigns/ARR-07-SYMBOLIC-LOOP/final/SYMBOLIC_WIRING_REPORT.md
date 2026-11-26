# ARR-07: symbolic/PySR wired into the agent loop (blocked by Julia env)

## User requirement
"接进自主循环（像 analyze 那样用标记文件触发 + LLM 读公式反思），
等我修好 Julia 后就可以继续测试" — implemented here.

## What was added
- `scinet_backend.run_symbolic()`: real SciNet++ Stage 3 (`symbolic.py`) via
  SciNetAdapter; parses `equations_<concept>.txt/.csv` (Equation column,
  complexity/loss) into `formulas`. If Julia/PySR unavailable, returns
  `ok=False, blocked=JULIA_PYSR_UNAVAILABLE` with the REAL error — NEVER
  fabricates equations.
- Autonomous loop: marker file `enable_formula_discovery` triggers symbolic
  on the baseline run inside the loop (same pattern as analyze).
  - success -> `discovered_formulas` injected into LLM's evidence (real reflection)
  - failure -> `formula_evolution` records `blocked`, LLM is TOLD formulas are
    unavailable and must not invent them; loop continues.

## VERIFIED: agent loop REALLY triggers symbolic (not a manual bypass)
ARR-07-SYMBOLIC-LOOP (both markers on):
- `formula_evolution`: [{'stage':'baseline','symbolic_ok':False,
  'blocked':'JULIA_PYSR_UNAVAILABLE', 'error':'... job_id 29cbb...,
  stage=symbolic, run_id=20261126_143426_99340b2b ...'}]
  -> the job_id/run_id were created INSIDE run_autonomous(), proving agent-triggered.
- `concept_evolution`: gamma R2=-0.106, omega R2=-0.272 (analyze, agent-triggered).
- LLM reasoning after seeing BLOCKED: quoted probe R2 ("Physics probes show
  negative R2, indicating poor concept alignment") and proposed pred_weight —
  NO invented equations. Honest degradation confirmed.

## Current blocker (user is fixing)
PySR installed but Julia project `D:\ProgramData\anaconda3\julia_env` missing.
User said manual symbolic worked before — likely a different Julia setup.
Once the user fixes the Julia env, re-run with both markers to get REAL formulas
and test whether the LLM can reflect on PySR equations (Theory Formation step).

## Honest verdict
Wiring is DONE and agent-triggered. End-to-end formula discovery is NOT verified
because of the environment. 28 unit tests pass (incl. graceful-blocked test).
