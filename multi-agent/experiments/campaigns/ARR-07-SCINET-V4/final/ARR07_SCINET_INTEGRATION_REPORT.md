# ARR-07 SciNet++ integration — honest status

## Q1: Were earlier tests using SciNet++? NO.

ARR-03/04/05/06/07 mainline experiments used the local Reacher simulator
(`arr06.engine`: PyTorch MLP + gymnasium), NOT SciNet++. SciNet++ was used only
in a separate historical track (`arr_emb_rsi_01/02`, `campaign/scinet.py`).
The JEPA/AI4S domain-pack configs existed but were not executed in these ARR loops.

Now: `arr07/scinet_backend.py` runs REAL SciNet++ train->evaluate via
`SciNetAdapter` subprocess JSON-RPC into `D:\conda_envs\scinet_gpu\python.exe`.
Verified health_check ok; real train job succeeded (checkpoint.pt 23KB), real
evaluate returned metrics (cosine@1=0.544, latent_mse@1=0.0264). 12/12 engine
records in the first SciNet campaign were ok=True.

## Q2: New-experiment designer — principle and correctness guarantee

Principle (generic, `generic/experiment_design.py`):
1. Hold competing hypotheses with PREDICTED SIGNATURES (which metric moves up/down).
2. `design_discriminative()` searches available operators for one whose predicted
   signature DIFFERS between hypotheses (separation score).
3. `check_unfalsifiable()` flags pairs no operator can separate.

Correctness is NOT guaranteed by the module — it is a decision aid. It only
ensures the chosen intervention is, *according to the supplied signatures*,
able to discriminate. If signatures are wrong, the design is wrong. In ARR-07
it correctly flagged EXP_ACT≡EXP_DIST as unfalsifiable and proposed a
separating experiment (separation 0.67) — but that proposal was never validated
by a real run. This is an honest limitation.

## Q3: Three-step integration — done, with real negative results

1. SciNet++ adapter: DONE and operational (real engine, real metrics).
2. Designer wired into loop: DONE. Found and FIXED 4 real wiring bugs:
   - config path relative to SciNet project dir
   - candidate names all identical (LLM got no real choice)
   - LLM output with "json" prefix failed parsing
   - `ResearchBrain.plan` hardcoded Reacher menu, silently replacing valid
     SciNet choices with EXP_COV — THE root cause of repeated identical runs.
3. Stopping rule wired: DONE (GoalGate in loop, stops on goal or budget).

After all fixes, the loop DOES select diverse designed experiments (pred_weight
instead of the forced latent_dim). But: across 4 SciNet campaigns the agent has
NOT yet produced a measured improvement (best ≈ −0.003, i.e. neutral).

## Honest bottom line

The FULL CHAIN now runs on the REAL SciNet++ engine: LLM plans -> designer
proposes discriminative experiments -> real train+evaluate -> measured data ->
frozen stopping. This is genuine integration. But the agent has NOT yet shown
it can improve the SciNet++ world model — the interventions tried were neutral
or harmful. Claiming "agent + SciNet++ achieves human-level experimentation"
would be false. The infrastructure is proven; the research capability on this
engine is NOT yet demonstrated.

All failed attempts (V1..V4) preserved. 16 unit tests pass.
