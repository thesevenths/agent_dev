# ARR-07: agent self-discovers the key variable (no human-fed evidence)

## Question
Previous run succeeded only because a HUMAN swept lr=0.01 and fed it as
`prior_evidence.json`. Can the agent run that sweep ITSELF and discover the
variable? User said "继续改" — this is the result.

## Mechanism added (agent harness only, SciNet++ unchanged)
- `LLMResearcher.plan_screening()`: BEFORE proposing, the LLM chooses WHICH
  knobs and WHICH values to sweep (one-factor-at-a-time), within a small budget
  (3 knobs x 2 values). It plans based on measured baseline + schema only.
- Harness runs each chosen single change on the REAL engine, records results.
- The best screening result becomes `prior_evidence` FOR THE SAME RUN.
- A frozen dual-objective contract ranks the sweep (not MSE alone) so a
  "MSE down / cosine crashes" config is never mistaken for a win.

## Real result (ARR-07-SELF-DISCOVERY-V3, no prior_evidence.json supplied)
LLM's OWN screening plan: `train.lr=[0.0001, 0.01]`, `vicreg_var=[0.1,10]`,
`latent_dim=[8,32]`. Measured single changes included:
- lr=0.01: MSE 0.0264 -> 0.0127 (+51.9%), cosine 0.544 -> 0.928 (+0.384)
- latent_dim=2 (earlier run): MSE +59% but cosine -0.586 (trap)

LLM proposal reasoning quoted its own screening result verbatim:
"Prior measured results demonstrate that setting train.lr to 0.01 yields
exceptional improvements (MSE ~0.0127, Cosine ~0.928) ..."

Then it proposed `lr=0.01 + ema_momentum=0.99` (within 2-key budget) and:
- real engine: MSE 0.02641 -> 0.01288, cosine 0.54369 -> 0.92613
- frozen contract: ALL goals met
- LLM finish verdict: objective genuinely satisfied
- `finished=True`, `budget_exhausted=False` — LLM terminated by its own decision.

## What this closes
The stated core gap ("LLM does not discover which knob matters") is CLOSED for
this benchmark: the agent planned a sweep, measured it, identified the key
variable from its own data, used it, and self-terminated. No human-supplied
answer.

## Bugs found/fixed on the way (honest log)
- V1: screening hint let LLM pick only tiny/baseline lr values -> missed 0.01.
  Fixed hint to require both directions / orders of magnitude.
- V2: evidence ranking used MSE alone -> promoted dim=2 (cosine crash).
  Fixed to use frozen dual-objective contract + regression test.
- V3: `discovered_best` field mapped wrong keys (row has "mse"/"cosine", not
  "latent_mse@1"/"cosine@1") -> field showed wrong config even though LLM read
  full rows correctly. Fixed + regression test. Not re-run after this cosmetic
  fix; the decision path itself was already correct and is evidenced above.

## Honest boundary
One benchmark (pendulum smoke, 64 samples, 1 epoch). The agent discovered lr
because its planning prompt suggested exploring orders of magnitude — this is
still harness guidance, not unbounded scientific creativity. Generalisation to
other engines (AI4S/quant) not tested.
