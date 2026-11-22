# ARR_EMB_RSI_02 — Final Report

Experiment: `ARR_EMB_RSI_02` · Branch: `feature/arr-emb-rsi-02-blind-validation`

## Executive verdict

Scientific claim: **NOT SUPPORTED**

| Layer | Verdict |
|---|---|
| engineering | PASS |
| benchmark_integrity | PASS |
| diagnosis | FAIL |
| intervention | PASS |
| causal_improvement | INCONCLUSIVE |

## Design

- Independent units: 10 task regimes; seeds [42, 43, 44] are repeated measurements.
- Arms: base, candidate, random, oracle; model runs: 120.
- MME (normalised): 0.05; alpha: 0.05.
- Power: n=10, dz=1.0, power=0.8031.
- Excluded regimes: {'double_pendulum::E02': 'Oracle inversion: 4 samples beat 64 samples', 'newton::E02': 'Oracle inversion: 4 samples beat 64 samples in real engine'}.

## Diagnosis benchmark

| Controller | Accuracy | 95% CI | Mean conf | ECE | Brier |
|---|---|---|---|---|---|
| agent | 0.500 | [0.332, 0.668] | 0.793 | 0.293 | 0.334 |
| rule | 0.100 | [0.035, 0.256] | 0.700 | 0.600 | 0.450 |
| random | 0.300 | [0.167, 0.479] | 0.655 | 0.355 | 0.351 |
| shuffled | 0.200 | [0.095, 0.373] | 0.617 | 0.417 | 0.338 |
| oracle | 1.000 | [0.886, 1.000] | 1.000 | 0.000 | 0.000 |

### Reliability diagram (agent)

| bin | n | mean confidence | empirical accuracy |
|---|---|---|---|
| 0 | 0 | 0.000 | 0.000 |
| 1 | 0 | 0.000 | 0.000 |
| 2 | 0 | 0.000 | 0.000 |
| 3 | 19 | 0.766 | 0.421 |
| 4 | 11 | 0.841 | 0.636 |

## Causal improvement

| Contrast | Mean normalized Δ | 95% CI | p | Holm p | Bootstrap CI |
|---|---|---|---|---|---|
| candidate_vs_base | 0.1166 | [-0.4410, 0.6743] | 0.6474 | 1 | [-0.3358, 0.5458] |
| candidate_vs_random | 0.0435 | [-1.2264, 1.3133] | 0.94 | 1 | [-0.7826, 1.2448] |
| oracle_vs_base | 0.7444 | [0.5987, 0.8900] | 1.059e-06 | 1.059e-06 | [0.6294, 0.8625] |

## Evidence audit

- Artifact integrity pass: True
- Mean citation validity: 1.000
- Mean evidence relevance: 1.000
- Sufficiency pass: True

## Engineering / failures

- Cases: 30; all arms ok: True
- Artifact integrity: True
- GT outside run root / zero leakage: True / True

## Scope statement

This campaign, even if PASS, validates only controlled state-based world-model research. It does NOT validate action-conditioned robotics, Visual JEPA, Reacher, MPC or recursive self-improvement.

## Reproducibility

```bash
python -m research_runtime.arr_emb_rsi_02.setup_preflight
python -m research_runtime.arr_emb_rsi_02.cli --stage campaign --out ARR_EMB_RSI_02_FULL
```

Canonical artifact SHA-256: `b69235e795723c808686c49f3adf7df2ff3b2684b3089f4aa9801480b6efc51f`
Preregistration SHA-256: `d83cfddaf1af55007897346892eb1fc494c00e20a2236db48aee51f172505b66`
