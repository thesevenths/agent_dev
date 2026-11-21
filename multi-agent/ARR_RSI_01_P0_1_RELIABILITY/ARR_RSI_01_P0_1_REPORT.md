# ARR-RSI-01-P0.1 — reliability Report

## Verdict

**FAIL**

- n_independent_units=2 < 5
- candidate not supported against correctness/controls

## Dimension means

| Dimension | Mean |
|---|---|
| scientific_correctness | 0.8 |
| ood_performance | 0.0 |
| evidence_completeness | 1.0 |
| experimental_validity | 1.0 |
| reproducibility | 1.0 |
| budget_efficiency | 0.04291 |

## Gates

- `primary_significant`: False
- `primary_holm_p`: 0.540601293538121
- `primary_mean_difference`: 0.006780978
- `candidate_vs_placebo_delta`: -0.001702292
- `candidate_vs_shuffled_delta`: -0.019929174
- `minimum_meaningful_effect`: 0.02
- `n_independent_units`: 2
- `complete_case`: True
- `min_n_tasks`: 5

## Per-task integrity

| task | cause correct | intervention valid | applied | infra | complete-case |
|---|---|---|---|---|---|
| arr_rsi_01::pendulum_dynamics::s42 | True | True | APPLIED | 4 | True |
| arr_rsi_01::pendulum_dynamics::s43 | True | True | APPLIED | 4 | True |
| arr_rsi_01::pendulum_dynamics::s44 | True | True | APPLIED | 4 | True |
| arr_rsi_01::pendulum_dynamics::s45 | True | True | APPLIED | 4 | True |
| arr_rsi_01::newtonian_dynamics::s42 | False | True | APPLIED | 4 | True |

## Scope

Original P0 verdict remains BLOCKED and is untouched. This report covers only the P0.1 integrity repair. Engineering evidence only; NOT scientific validation. Original P0 BLOCKED verdict is preserved unchanged.

Protocol SHA: `34cdc9dae78f26787f567a82b2514370603b73168bcff4674c78f3163475d011`
Statistical SHA: `d2971c31c00a2d7d8f025c491d6698727e3349e00a96b6e9fd0a64078f448856`
