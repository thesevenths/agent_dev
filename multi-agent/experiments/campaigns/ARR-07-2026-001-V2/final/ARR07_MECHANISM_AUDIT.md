# ARR-07 mechanism-label audit — why "distribution shift never converged"

## Bottom line

The 4 "spurious convergence" verdicts involving `EXP_ACT` vs `EXP_DIST` are a
**benchmark artifact**, not proof that the LLM failed to reason from data.
On 12 of the evaluated tasks, applying the two registered experiments yields a
byte-identical effective configuration and therefore an identical MSE.
The ground-truth mechanism label is unfalsifiable from the measurements.

## The concrete distribution-shift case (v30, blind)

Baseline measured: `one_step_mse = 9.66`, `final_loss = 0.76`.

| Iter | LLM saw (measured only) | LLM reasoning (verbatim summary) | Chose | Measured MSE |
|---|---|---|---|---|
| 0 | 9.66 / 0.76 | "fits training distribution well but fails to generalize ... insufficient trajectory coverage" | EXP_COV | 8.38 (−13%) |
| 1 | 8.38 + history(COV 0.13) | "coverage failed ... model averaging over unobserved dynamics ... inadequate action conditioning" | EXP_ACT | **0.83 (−91%)** |
| 2 | 0.83 + history(ACT 0.91) | "91% reduction confirms action conditioning ... residual gap from limited capacity" | EXP_CAP | 8.12 (+spike) |
| 3 | 8.12 + history | "capacity spike suggests optimization instability, stabilise lr" | EXP_OPT | 8.13 |

Verdict was FAIL/spurious because the true label was `dynamics_distribution_shift`
and the registered fix is `EXP_DIST`. But the direct probe showed:

- `EXP_ACT` on this task → config `{action_mode: random, action_scale: 1.0}`, MSE **0.8274**
- `EXP_DIST` on this task → config `{action_mode: random, action_scale: 1.0}`, MSE **0.8274**

Identical. The LLM's inference ("action conditioning was the problem") was
reasonable from the data; the benchmark gave it two different names for the
same intervention.

## Why the LLM picked EXP_ACT and never EXP_DIST

1. `EXP_COV` failed (0.13) → coverage ruled out — this step WAS data-driven and correct.
2. The remaining high MSE + low train loss has two registered explanations
   with indistinguishable signatures; the LLM used experiment names as a tiebreak.
3. After EXP_ACT cut MSE 91%, the LLM correctly treated that as strong evidence —
   but the frozen evaluator still demanded the other label.

## What this changes about the earlier conclusions

- "Distribution shift never converged" → **misleading**. It DID converge to a
  91% MSE reduction; it was marked non-converged only because of an
  unresolvable mechanism label.
- "Spurious convergence 4/7" → **partly a benchmark artifact**. Need to re-audit
  which of the 4 are genuine mechanism errors (e.g. choosing EXP_CAP when
  capacity is not the issue) versus this label collision.
- Genuine LLM weakness remains: it never proposed a discriminating experiment
  to separate the two hypotheses, and after finding a 91% improvement it
  chased residual error into EXP_CAP/EXP_OPT instead of stopping.

## Recommended repairs (not yet executed)

1. Give `EXP_DIST` a genuinely distinct intervention (e.g. reweight training
   states toward the held-out target distribution) so it is falsifiable apart
   from `EXP_ACT`.
2. Make the evaluator accept mechanism equivalence classes when two experiments
   produce identical effective configs.
3. Re-run V1/V2 after (1)/(2) to get honest spurious-convergence numbers.
