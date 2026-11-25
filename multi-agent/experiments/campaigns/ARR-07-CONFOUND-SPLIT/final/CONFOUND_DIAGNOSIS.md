# Final root-cause: "cosine below baseline" was a confound, not wrong VICReg

## Question
User asked: is the low cosine caused by LLM ability, or by the harness?
Earlier answer ("LLM's VICReg choice was wrong") was WRONG and is corrected here.

## Decisive experiments (real SciNet++, single-variable isolation)

Baseline config: latent_mse@1=0.0264, cosine@1=0.5437, VICReg var=25 / cov=1.
LLM's failing LeJEPA rounds changed 7 keys at once, incl. VICReg cov=30/60/75
(30-75x baseline). COV increases alone:

| single change | MSE | dMSE | cosine | dcosine |
|---|---:|---:|---:|---:|
| cov=30 ONLY | 0.0264 | 0 | 0.5437 | 0 |
| cov=60 ONLY | 0.0264 | 0 | 0.5437 | 0 |
| var=10 ONLY | 0.0264 | 0 | 0.5437 | 0 |
| var=20 ONLY | 0.0264 | 0 | 0.5437 | 0 |
| ema=0.999 ONLY | 0.0264 | 0 | 0.5437 | 0 |
| pred_weight=5 ONLY | 0.0264 | 0 | 0.5440 | +0.0003 |
| **latent_dim=32 ONLY** | 0.0322 | **−21.9%** | 0.0831 | **−0.461** |
| lr=0.0005 ONLY | 0.0290 | −9.8% | 0.4597 | −0.084 |
| **hidden=[256,256] ONLY** | 0.0086 | **+67.3%** | 0.9462 | **+0.403** |
| dim32 + hidden256 | 0.0121 | +54.2% | 0.7812 | +0.237 |
| LLM's full 7-key round | 0.0198 | +25.0% | 0.4921 | −0.052 |
| lr=0.01 ONLY | 0.0127 | +51.9% | 0.9281 | +0.384 |

## Verdict

1. VICReg was NOT "wrong": its knobs are INERT on this smoke engine (0 effect
   even at 75x). My previous wording was incorrect; corrected.
2. The cosine drop came from ONE key inside the 7-key bundle: `latent_dim=32`
   alone costs −0.461 cosine. The LLM's `hidden=[256,256]` was actually GOOD
   (+67% MSE, +0.40 cosine) but was masked by the bad dim in the same run.
3. Therefore the root cause is the HARNESS allowing 7 simultaneous changes
   (no attribution possible), NOT the LLM's scientific judgement being bad.
   The LLM was iteratively adjusting cov/var (30->60->75) — trying to learn —
   but could not isolate which key hurt because all moved together.
4. Fix already applied in code: `max_changes=2` per round in
   `generic/llm_autonomy.py`, plus measured-prior injection. This is exactly
   the guardrail this experiment justifies.

## Honest note
This does not prove the LLM is a strong scientist: with max_changes=2 and NO
prior evidence (ARR-07-BUDGET10) it still never found lr=0.01 in 10 rounds.
It proves the FAILURE MODE was a harness confound, not a VICReg misjudgement.
