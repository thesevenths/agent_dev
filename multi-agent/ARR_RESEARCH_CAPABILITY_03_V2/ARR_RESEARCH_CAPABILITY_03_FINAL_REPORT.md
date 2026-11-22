# ARR_RESEARCH_CAPABILITY_03 — Final Report

## Executive Verdict

**Overall Research Capability: INCONCLUSIVE**

| Layer | Verdict |
|---|---|
| benchmark_integrity | PASS |
| engineering | PASS |
| m1_diagnosis | FAIL |
| m2_causal_improvement | INCONCLUSIVE |
| m3_embodied_transfer | INCONCLUSIVE |
| overall_research_capability | INCONCLUSIVE |

## Campaign Identity

- Campaign ID: ARR_RESEARCH_CAPABILITY_03
- Branch: feature/arr-research-capability-03
- Preregistration SHA-256: 6aa6ff7c824e8a75f52d96de68f16810ed963558220469bddf1126b585bc4f28
- Canonical results SHA-256: 293b5411e2a87c22a6433a486eb700d8092cd79acbf5e56a73fdf191c6ccabda
- Ground truth seal: 862dfee5ffb122d2319136474a51049ce2e4cc025c635c2ec3c3d3be281abf0f
- Independent units: M2=10, M3=6; seeds are repeated measurements

## Budget

- M2 model runs: 120
- M3 model runs: 90
- Total: 210 (max 216)

## M1 Diagnosis

| Strategy | Accuracy | Top-2 | ECE | Brier | Abstention |
|---|---|---|---|---|---|
| arr | 0.5833 | 0.7917 | 0.2415 | 0.2882 | 0.0000 |
| oracle | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| random | 0.2708 | 0.0000 | 0.2691 | 0.2838 | 0.0000 |
| rule | 0.2500 | 0.5000 | 0.3500 | 0.3100 | 0.0000 |
| shuffled | 0.1875 | 0.0000 | 0.3363 | 0.2809 | 0.0000 |

## M2 Causal Improvement (task-level units)

| Contrast | Mean | 95% CI | Holm p | Significant |
|---|---|---|---|---|
| candidate_vs_base | 0.7587 | [0.6149, 0.9025] | 0.0000 | True |
| candidate_vs_random | 0.0480 | [-0.0324, 0.1285] | 0.2100 | False |
| oracle_vs_base | 0.7444 | [0.5987, 0.8900] | 0.0000 | True |

## M3 Embodied Transfer (Reacher-v5)

| Contrast | Mean | 95% CI | Holm p | Significant |
|---|---|---|---|---|
| arr_transfer_vs_fresh | -0.1180 | [-0.7555, 0.5195] | 0.9873 | False |
| arr_transfer_vs_random | 0.2528 | [-0.6277, 1.1333] | 0.9873 | False |
| oracle_vs_base | 0.8539 | [0.7317, 0.9761] | 0.0000 | True |

Verified knowledge items transferred: 4 (SHA-256 2e57b8fc66d5b7f2)

## Evidence Integrity

{
  "artifact_integrity_pass": true,
  "citation_validity": 0.3684027777777778,
  "evidence_relevance": 0.896875,
  "evidence_sufficiency_rate": 1.0,
  "n_cases": 48
}

## Limitations

- State-based dynamics (SciNet++) and Reacher-v5 dynamics are simulator experiments; no visual JEPA, no real robot, no action-conditioned representation learning beyond a small MLP dynamics model.
- Independent units are task regimes (10 + 6); seeds are repeated measurements.
- Overall PASS requires M1, M2 and M3 each PASS; engineering PASS cannot compensate a scientific FAIL.

## Deliverables index

- Source audit: `ARR_RESEARCH_CAPABILITY_03_SOURCE_AUDIT.md`
- Preregistration (frozen): `ARR_RESEARCH_CAPABILITY_03_V2/preregistration.json`
- Canonical results: `ARR_RESEARCH_CAPABILITY_03_V2/canonical_results.json`
- Statistical analysis: `ARR_RESEARCH_CAPABILITY_03_V2/statistical_analysis.json`
- Evidence audit: `ARR_RESEARCH_CAPABILITY_03_V2/evidence_audit.json`
- Failure analysis: `ARR_RESEARCH_CAPABILITY_03_V2/failure_analysis.json`
- Full experiment data: `ARR_RESEARCH_CAPABILITY_03_V2/m2_cases.json`, `m3_cases.json`
- Reproduction commands: see Source Audit §9

## Final scientific answer

1. Does ARR genuinely diagnose scientific failures? **PARTIAL** — accuracy
   0.583 (< 0.65 gate), over-confident (ECE 0.241); significantly above random
   but below acceptance. M1 = FAIL.
2. Does ARR select better interventions than Random Research? **NO clear
   evidence** — Candidate vs Random +0.048, CI crosses 0, Holm n.s.
3. Does ARR produce measurable causal improvements? **YES vs Base**
   (+0.759, Holm p≈1.6e-6), but not distinguishable from random research.
4. Does ARR transfer verified knowledge to a new embodied domain? **NO** —
   Transfer vs Fresh -0.118, n.s.
5. Does Research Memory improve over Fresh Research? **NO measured benefit**
   in this campaign.
6. Actual limitations: E03 evidence-ambiguity (shared config), confidence
   miscalibration, no abstention, no random-beating attribution, no transfer
   gain.

Overall: **INCONCLUSIVE**. This campaign does NOT certify ARR as an
Autonomous Scientific Research System.
