# ARR-07: analyze stage wired into loop — agent reasons about DISCOVERED CONCEPTS

## What was added
- `scinet_backend.run_analyze()`: real SciNet++ Stage 2 (`analyze.py` physics
  probe) on the training run, parses gamma/omega probe R2 + RMSE + concept names
  from real artifacts. No fabricated metrics.
- Loop: when `enable_concept_discovery` marker exists, baseline analyze runs and
  its concept metrics are injected into the LLM's evidence so it can REFLECT on
  concept discovery, not just MSE/cosine.
- Enabled by campaign marker file (generic, no hardcoded per-domain logic).

## Verified: analyze stage WORKS on the vendored smoke engine
Real run on smoke checkpoint: gamma R2=-0.119, omega R2=-0.195, RMSE 0.094/0.346,
artifacts gamma.npy/omega.npy/label_names.json/latent.npy/attribution_matrix.csv.
(Negative R2 = latent did NOT capture these physical concepts — honest result.)

## Verified: LLM actually used concept evidence (ARR-07-CONCEPT-LOOP)
Baseline concepts: gamma R2=-0.106, omega R2=-0.272.
- it0 LLM reasoning: "Baseline metrics show poor concept discovery (negative
  R2 for gamma/omega), suggesting the latent space lacks structured physical
  correlations. VICReg's covariance penalty enforces decorrelation, which may
  suppress correlated signals essential for representing coupled physical
  parameters." -> proposed reducing jepa.vicreg_cov_weight.
- it1: continued from it0 result, "if the ground-truth physics involves coupled
  parameters, this penalty actively destroys useful structure."

This is a REAL chain: observed concept-discovery failure -> formed a mechanism
hypothesis -> designed an intervention. Higher-level than MSE-only tuning.

## symbolic.py (PySR formula discovery): BLOCKED in this environment
Real probe: job failed. Root cause from engine log: PySR is installed but its
Julia project `D:\ProgramData\anaconda3\julia_env` does NOT exist. No equations
produced. NOT wired into the loop; must not be claimed as working. User said
manual use worked previously — likely a different Julia setup; needs fixing
the Julia environment first.

## Honest verdict on the 4 advanced capabilities
- Scientific Discovery: STILL NO. Concepts (gamma/omega) are DATASET-PRELABELED
  ground truth, not agent-discovered. Agent reasons about probe results.
- Theory Formation: STILL NO. symbolic/PySR blocked; no cross-experiment
  abstraction exists.
- Empirical Validation: IMPROVED. Agent now designs experiments to test a
  concept-discovery hypothesis (VICReg decorrelation), not only to lower MSE.
- Recursive Improvement: SAME benchmark only; no cross-domain evidence.

Wiring analyze closes the "agent can read concept-level evidence and reflect"
gap. It does NOT close "agent discovers new concepts/theories".
