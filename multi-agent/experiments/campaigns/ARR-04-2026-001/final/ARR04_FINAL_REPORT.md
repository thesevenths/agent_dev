# ARR-04 Final Report — Autonomous Scientific Research Capability Validation

## 1. Executive Summary

```json
{
  "campaign_id": "ARR-04-2026-001",
  "campaign_status": "COMPLETED_WITH_FAILURES",
  "engineering_verdict": "PASS",
  "scientific_judgment": "NOT_VALIDATED",
  "active_diagnosis": "FAIL",
  "research_selection": "PASS",
  "knowledge_transfer": "FAIL",
  "recursive_improvement": "INCONCLUSIVE",
  "independent_audit": "PASS",
  "formal_runs": 442,
  "preflight_runs": 14,
  "llm_calls": 131,
  "invalid_runs": 0,
  "total_compute_cost": null
}
```

**Overall scientific verdict: NOT_VALIDATED**

## 2. Overall Verdict

| Layer | Verdict |
|---|---|
| engineering | PASS |
| benchmark_integrity | PASS |
| m1_active_diagnosis | FAIL |
| m2_research_selection | PASS |
| m3_knowledge_transfer | FAIL |
| recursive_improvement | INCONCLUSIVE |
| independent_audit | PASS |
| overall_scientific_verdict | NOT_VALIDATED |

## 3. Campaign Integrity

- Commit: fc3b49dcb7677525ac4558a0344a6487698138a9
- Branch: feature/arr-research-capability-04
- Protocol SHA-256: f155311b212bb5af991eab8c7a9688f7f3b43a27f9d4cfdf13f23b739d8e4a09
- Canonical SHA-256: 6c159a8bf66f9a0ac1deda307237b205219d7d2ba58c5eaccab60422a597848e
- Terminal state: COMPLETED_WITH_FAILURES

## 4. Preflight Results

- Benchmark identifiable: True
- Experiment discriminative: True
- Power n>=0.8 units: 10

## 5. Benchmark Identifiability

ARR-03's E01/E03 ambiguity (shared config) was repaired: each cause now has a distinct observable signature and one registered discriminative experiment. See `preflight/identifiability.json`.

## 6. M1 Active Diagnosis

| Strategy | Accuracy | ECE | Brier | Info gain |
|---|---|---|---|---|
| passive_arr | 0.6750 | 0.1290 | 0.2403 | - |
| active_arr | 0.7750 | 0.1520 | 0.1840 | 0.1000 |
| random_experiment | 0.7250 | 0.1250 | 0.2150 | - |
| rule_experiment | 0.5750 | 0.0250 | 0.2450 | - |
| oracle_experiment | 1.0000 | 0.4000 | 0.1600 | - |
| placebo | 0.6750 | 0.0750 | 0.2250 | - |

## 7. M2 Research Selection

| Contrast | Mean | 95% CI | Holm p | Sig |
|---|---|---|---|---|
| arr_vs_base | 0.6530 | [0.3580, 0.9480] | 0.0015 | True |
| arr_vs_random | 0.7676 | [0.2878, 1.2474] | 0.0056 | True |
| arr_vs_rule | 0.4993 | [0.1021, 0.8965] | 0.0193 | True |
| oracle_vs_base | 0.9179 | [0.8933, 0.9425] | 0.0000 | True |

## 8. M3 Knowledge Transfer

- Transfer vs Fresh: mean 0.1466, CI [-0.2147, 0.5078], sig False
- Negative transfer rate: 0.3000
- Memory items (verified only): 5

## 9. Recursive Improvement

- Capability curve: ['0.5000', '0.5000', '0.5000']
- Improvement: 0.0000, regression: False

## 10. Independent Audit

{
  "integrity_audit": true,
  "statistical_audit": true,
  "reproducibility_audit": true,
  "claim_audit": true,
  "source_audit": true
}

## 11. Limitations and Negative Findings

- Simulator-based (MuJoCo Reacher-v5); no real robot, no visual JEPA.
- Held-out/OOD ground truth inaccessible during research.
- Negative findings reported without suppression; see verdicts above.

## 12. Evidence Index

- Immutable raw records: 456
- Protocol: protocol/protocol.json; audit: independent_audit/*.json
- All artifacts under experiments/campaigns/ARR-04-2026-001/

## 13. Reproduction

```bash
cd /e/agent_dev/multi-agent
export PYTHONPATH=/e/agent_dev/multi-agent
/d/python313/python.exe -m research_runtime.arr04.cli
```
