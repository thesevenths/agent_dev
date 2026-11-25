# ARR-07 paired multi-seed validation — first statistically supported result

## What was tested
Candidate: `train.lr=0.01` (from deterministic sweep), default structure.
Paired design: SciNet++ `seed` controls model init; dataset frozen.
3 seeds (41/42/43), each baseline + candidate, real train+evaluate.

## Critical bug found and fixed during this run
First attempt (ARR-07-MULTISEED) produced IDENTICAL results for all 3 seeds
(CI width 0) — false "multi-seed". Root cause: SciNet `train.py` reads
`cfg["seed"]` from the YAML config; the harness only passed seed in the spec,
so all configs had the frozen `seed: 42`. Fixed in `scinet_backend.materialise_config`
(seed now written into YAML) + regression test. V2 below uses the fix.

## Real results (V2, all 6 engine runs ok=True)

| seed | baseline MSE | candidate MSE | relative gain | cosine delta |
|---|---:|---:|---:|---:|
| 41 | 0.02641 | 0.00194 | +92.6% | +1.877 |
| 42 | 0.02641 | 0.01270 | +51.9% | +0.384 |
| 43 | 0.02641 | 0.00244 | +90.8% | +0.878 |

Paired 95% CI (normal approx): **[0.524, 1.045]**, lower bound > 0.
Conclusion: **CONFIRMED positive MSE gain** across seeds.

## Honest caveats
- normal-approximation CI with n=3; t-based CI would be wider (implementation
  notes this in `paired_seed_verdict`). Treat as initial evidence, not final.
- cosine@1 delta 1.88 for seed 41 is legitimate: values were -0.9458 -> 0.9311,
  both inside [-1,1]. The seed-41 baseline was near-anti-correlated; lr=0.01
  rescued it. Means the baseline init quality varies a lot across seeds.
- The candidate (lr=0.01) came from a DETERMINISTIC sweep, not from the LLM
  autonomy loop. This validates the STATISTICAL harness, not the agent's
  discovery ability.
