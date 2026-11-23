# ARR-05 Final Report — Research Capability Repair & Confirmatory Validation

## 1. Executive Summary

Overall: **AUTONOMOUS RESEARCH SYSTEM — NOT VALIDATED** · Terminal state: COMPLETED_WITH_FAILURES

| Capability | Verdict |
|---|---|
| benchmark_integrity | PASS |
| engineering | PASS |
| independent_audit | FAIL |
| m1_active_diagnosis | PARTIAL |
| m2_confirmatory_selection | PASS |
| m3_memory_transfer | INCONCLUSIVE |
| overall | AUTONOMOUS RESEARCH SYSTEM — NOT VALIDATED |
| rq_discovery | FAIL |
| rsi_recursive_methodology | INCONCLUSIVE |
| source_reproducibility | PASS |

## 2. ARR-04 Historical Baseline

- Declared source commit: fc3b49dcb7677525ac4558a0344a6487698138a9
- Actual implementation commit: eaa8ed79417550aaa3120c2f5976053ed89cf0b1
- Source recovered: True
- Raw records reconciled: True (456 records)
- ARR-04 primary result reproduced (no LLM): True, delta=0.0000

## 3. Preflight

- Benchmark identifiable: True
- Oracle sensitivity: True
- Power n26 dz0.8: 0.9749

## 4. M1 Calibration & Active Diagnosis

- Cases: 80 (calibration 40, heldout 40)
- Temperature (fit on calibration only): 0.3455
- Active heldout accuracy: 0.8000; ECE: 0.2119; Brier: 0.1342
- Active vs Passive: mean 0.1250, Holm p 0.0234
- Realized information gain: 0.4924

## 5. M2 Confirmatory Research Selection

- Independent units: 30
- arr_vs_base: mean 0.5397, CI [0.3840, 0.6954], Holm p 0.0000
- arr_vs_random: mean 0.3868, CI [0.0887, 0.6849], Holm p 0.0128
- arr_vs_rule: mean 0.5492, CI [0.1023, 0.9962], Holm p 0.0178

## 6. M3 Memory Transfer

- Units: 30; memory items: 5
- Raw vs Fresh: -0.1272
- Gated vs Fresh: 0.0271
- Negative transfer rate: 0.4000
- Applicability match: 0.8500

## 7. RQ Discovery

- n=13, ARR vs Random mean 0.1040, falsifiability 0.0000, novelty UNVERIFIED

## 8. RSI Recursive Improvement

- Methodology changes: 2
- Curve: {"evolved": [0.7, 0.9, 0.7], "frozen": [0.7, 0.6, 0.6], "memory_only": [0.6, 0.8, 0.9]}
- Evolved vs Frozen: 0.1000

## 9. Independent Audit

{
  "claims": true,
  "integrity": true,
  "reproducibility": false,
  "source": true,
  "statistics": true
}

Replay root-cause diagnostic: 3 identical-config runs spread=1.1579, deterministic=False. Root cause: PyTorch CPU multithreading + chaotic optimization (high lr / zero-action signal). Replay mismatch is an ENGINE-LEVEL non-determinism finding, not a fabricated or suppressed result.

## 10. Budget & Costs

{
  "formal": 430,
  "llm": 350,
  "max_formal": 1500,
  "max_llm": 2500,
  "max_preflight": 60,
  "max_replay": 200,
  "preflight": 12,
  "replay": 43
}

- cost_usd: null (provider pricing unavailable) — Cost Efficiency NOT certified

## 11. Negative Findings & Limitations

- Simulator-based Reacher-v5 only; no visual JEPA / real robot.
- Novelty remains UNVERIFIED (no independent literature review).
- All negative results preserved; no INCONCLUSIVE upgraded to PASS.

## 12. Evidence & Reproduction

- Evidence index: final/ARR05_EVIDENCE_INDEX.json
- Reproducibility manifest: final/ARR05_REPRODUCIBILITY_MANIFEST.json
- Canonical SHA-256: c5950913cdebff6d802da4e39f5a718ddce99ae8a7bca747dcba656ab4edcca0
