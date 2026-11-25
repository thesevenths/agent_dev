# Final diagnosis: who sets SciNet++ structure/params, and which link failed

## Q1: different experiments should use different SciNet++ structures/params — who decides?

YES, each experiment uses a materialised SciNet++ config (a real YAML). The
baseline `configs/pendulum_smoke.yaml` has:
- structure: `model.latent_dim=4`, `encoder_hidden=[32,16]`, `predictor_hidden=[32,32]`
- JEPA: `ema_momentum=0.996`, `pred_weight=1.0`, `rollout_steps=[1,2,3]`
- optimisation: `train.lr=0.001`, `epochs=1`, `batch_size=16`, 64 train samples

Each agent experiment deep-merges an override into this config and submits a
FRESH train+evaluate pair (verified: distinct config hashes, 12/12 real jobs).

WHO DECIDES (honest answer): not the LLM, not SciNet++. The candidate set and
the parameter values are HARDCODED BY ME (the code author) in
`arr07/scinet_loop.py`: `HYPOTHESES` (predicted signatures) + `OPERATORS`
(parameter spaces). I initially guessed them wrong (latent_dim=64 etc.), then
corrected them from the real sweep below. The LLM only picks ONE from the
candidates I pre-defined. This is "choice within a human-defined menu", NOT
autonomous experimental design. Calling it "the agent designed the experiment"
would be false.

## Q2: why did the agent not improve the world model — which link failed?

The DECISIVE experiment was a deterministic parameter sweep (no LLM, real
engine, `ARR-07-PARAM-SWEEP`):

| override | latent_mse@1 | improvement | cosine@1 |
|---|---:|---:|---:|
| baseline (lr=0.001, dim=4) | 0.0264 | — | 0.544 |
| latent_dim=8 | 0.0427 | −61.5% | 0.123 |
| latent_dim=16 | 0.0355 | −34.3% | −0.105 |
| pred_weight=10 | 0.0264 | 0.0% | 0.544 |
| ema_momentum=0.90 | 0.0267 | −1.1% | 0.542 |
| lr=0.0001 | 0.0310 | −17.6% | 0.393 |
| **lr=0.01** | **0.0127** | **+51.9%** | **0.928** |

Conclusion: the engine, data, train/evaluate, and metrics are ALL FINE — a
simple lr change improves MSE 51.9% and cosine +0.384. So the failure is NOT
in "实验 / 看数据 / 再实验 / 达标". It is in "改方案" (planner + designer):

- I set the hypothesis `unstable_lr -> cosine flat, mse down` (wrong: lr=0.01
  improves both). After the sweep I corrected the signatures to evidence.
- Even with `lr=0.01` explicitly labelled in the candidates, the LLM chose
  `latent_dim` every time (V1..V6), reasoning "cosine low but MSE low => lack
  of representational capacity" — a text prior, ignoring the measured sweep.

Along the way I found and FIXED 6 genuine wiring bugs (config path, identical
candidate IDs, `json` prefix parse, hardcoded Reacher menu replacing valid
SciNet choices, duplicate design names, opaque candidate labels). After all
fixes the LLM still picks the non-improving knob. This is now an honest MODEL
LIMIT (strong prior overriding evidence), not a remaining plumbing bug.

## Final honest verdict

- Chain on real SciNet++: RUNS end-to-end (real train, real evaluate, real data, frozen stop).
- Agent research capability on this engine: NOT DEMONSTRATED. A known +51.9%
  improvement exists but the planner will not select it from evidence.
- Next real fix (not done): either a stronger planner, or make candidate
  proposals evidence-ranked from a pilot sweep (the generic designer already
  supports predicted signatures — feed it MEASURED ones, not guessed ones).
