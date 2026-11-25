# LeJEPA-style autonomous study on real SciNet++

## Theme (from literature, 2025-2026)

Recent JEPA work (LeJEPA/SIGReg, AMI Labs/NYU) argues that embedding
regularisation can replace hand-tuned anti-collapse tricks (stop-gradient,
teacher-student asymmetry). SciNet++ has a REAL implementation of the same
family: VICReg variance/covariance losses (`jepa.vicreg_var_weight`,
`jepa.vicreg_cov_weight`). This study asks: can the autonomous LLM agent find a
better VICReg+structure configuration on the frozen pendulum world-model task?

Reproduction SCOPE: this is a VICReg-regularised JEPA study, NOT a full
LeJEPA/SIGReg reproduction (SIGReg is not implemented in SciNet++). Claiming
otherwise would be false.

## Setup

- Engine: real SciNet++ train->evaluate via SciNetAdapter (conda scinet_gpu).
- LLM: qwen3.7-flash; objective = minimize `latent_mse@1` AND maximize `cosine@1`.
- Agent controls (no human menu): lr, pred_weight, ema_momentum, BOTH VICReg
  weights, latent_dim, predictor_hidden — free values within schema ranges.
- Baseline: latent_mse@1 = 0.02641, cosine@1 = 0.54369.

## Real results (3 iterations, all engine ok=True)

| it | LLM-proposed changes (selected) | latent_mse@1 | MSE improvement | cosine@1 | LLM finish verdict |
|---|---|---:|---:|---:|---|
| 0 | lr 0.0005, pred 5, ema 0.999, vicreg 10/30, dim 24, hidden [128,128] | 0.02226 | +15.8% | 0.339 | continue |
| 1 | vicreg 20/60, dim 32, hidden [256,256] | 0.01980 | **+25.0%** | 0.492 | continue |
| 2 | lr 0.0002, pred 8, ema 0.9995, vicreg 25/75, hidden [192,192] | 0.02329 | +11.8% | 0.270 | continue |

Run ended by budget (`budget_exhausted=True`, `finished=False`). The LLM's
"continue" verdict at every step is HONEST: although MSE improved, cosine@1
never exceeded the baseline 0.544, so the dual objective was not met.

## Verdict on the user's full-chain question

The chain `LLM designs structure/params -> SciNet tests -> LLM reviews data ->
LLM revises -> SciNet tests again -> LLM judges finish` ran end-to-end on the
real engine with ALL decisions from the LLM and all data real. Mechanically:
**YES, fully implemented and exercised**. Scientifically: on this study the
agent achieved a real +25% MSE improvement but did NOT reach the dual-objective
target; it correctly refused to claim finish. "完美达到" (perfectly achieved)
would be false — the loop is faithful, the result is partial.

## Honest limitations

- Only 3 iterations, one frozen smoke dataset (64 train samples, 1 epoch).
- Cosine@1 dropped in all three runs vs baseline; minimising MSE alone is easy
  but the joint objective is not solved.
- Not a LeJEPA/SIGReg reproduction (engine lacks SIGReg); VICReg-family study only.
