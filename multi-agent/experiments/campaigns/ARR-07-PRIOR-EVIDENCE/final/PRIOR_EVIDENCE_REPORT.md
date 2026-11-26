# ARR-07: why cosine stayed low, and the fix that worked

## User question
"cosine 全低于 baseline 是为什么？改 harness 还是 SciNet++？预算放大到 10 轮试试？"

## Root cause (data, not speculation)
LLM's LeJEPA study changed **7 parameters at once** every round (VICReg var/cov,
ema, pred_weight, latent_dim, hidden, lr). Result: MSE improved but cosine fell
to 0.27-0.49 (baseline 0.544). In contrast, the deterministic sweep's single
parameter change `lr=0.01` gave MSE 0.0127 AND cosine 0.9281 (baseline 0.0264 /
0.5437). So: NOT a SciNet++ problem, NOT a judge problem — an agent SEARCH
STRATEGY problem (too many simultaneous changes, wrong variable).

## Experiment 1: budget 10 rounds (ARR-07-BUDGET10)
Added a harness change-budget (max 2 config keys per round) and ran 10 real
rounds. LLM tried pred_weight, VICReg, ema, latent_dim, hidden, and lr=0.0005 —
but NEVER lr=0.01. Result: cosine never exceeded baseline; all rounds
contract_met=False, LLM correctly said continue, ended at budget exhaustion.
**Lesson: more budget does not help if the agent does not know the key variable.**

## Experiment 2: inject MEASURED prior evidence (ARR-07-PRIOR-EVIDENCE)
Wrote `prior_evidence.json` into the campaign (external file, not code): the
real sweep + multi-seed finding that lr=0.01 improves both metrics (with CI).
LLM proposal reasoning quoted it verbatim:
"Verified prior evidence demonstrates that increasing train.lr to 0.01 yields
massive joint improvements ... I pair lr=0.01 with jepa.vicreg_cov_weight=1.0".

ROUND 1 RESULT (real engine, ok=True):
- proposed: {"train.lr": 0.01, "jepa.vicreg_cov_weight": 1.0}
- MSE 0.02641 -> 0.01270 (−51.9%), cosine 0.54369 -> 0.92814 (+70.7%)
- frozen contract: ALL goals met (MSE −51.9% >= 10%, cosine +70.7% >= 0)
- LLM finish verdict: "joint objective genuinely satisfied ... no further experiments"
- `finished=True`, `budget_exhausted=False` — ended by the LLM's OWN decision,
  not by budget.

## Answer to "改 harness 还是 SciNet++"
- SciNet++: NO change (pure engine, already produced the right numbers).
- Harness: YES — change budget (max 2 keys), and measured-prior injection via
  external campaign file. The harness computes the frozen contract and feeds it
  to the LLM; the LLM still decides.
- The critical ingredient was EVIDENCE, not more rounds: when the agent knows a
  measured fact (lr=0.01 works), it reaches the goal in ONE round.

## Honest scope
The successful config came from the deterministic sweep, supplied as prior
evidence — the agent did NOT discover lr=0.01 itself. What is demonstrated:
the loop can CONSUME verified evidence, reason, act, and self-terminate correctly.
Autonomous discovery of which variable matters remains unsolved.
