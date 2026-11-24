# ARR-07 Link-Level Isolation Validation

## Why this exists

ARR-07 V1/V2 showed the loop RUNS but did not prove each link works correctly. Isolation tests target the two weak links.

## V4 — Planner causality (THE finding)

- Bare LLM determinism: 100% (['EXP_COV', 'EXP_COV', 'EXP_COV', 'EXP_COV', 'EXP_COV']) — NOT random.
- Bare LLM causality: False — three distinct measurement signatures ALL produced EXP_COV. The LLM has an absolute coverage prior and ignores the data. This is the root cause of weak 'replan'.

- GATED planner causality: True (3 distinct choice sets) — with a frozen evidence gate constraining choices, the planner responds systematically to measurements.
- HONESTY NOTE: causality is achieved by the deterministic gate, NOT by LLM self-discovery. This is constrained cognition, not fully autonomous research.

## V5 — Independent evaluator audit

- Agreement: 10/10 with a SEPARATE implementation reading raw traces. Goal judgement link VERIFIED.

## Model provenance

- Campaign data produced with qwen3.6-flash (local vLLM gateway).
- On user request, .env LLM_MODEL switched to qwen3.7-flash; API key unchanged; live call verified. .env is not git-tracked; a backup .env.bak-arr07-model-switch was kept locally.
- Isolation tests V4-bare / V5 ran on qwen3.6-flash; gated causality ran on qwen3.7-flash (different model -> cross-model evidence, noted for transparency).

## Verdict

| Link | Status |
|---|---|
| replan (bare LLM) | FAIL — insensitive to evidence |
| replan (evidence-gated) | PARTIAL — causal but rule-driven |
| goal (independent audit) | PASS |
