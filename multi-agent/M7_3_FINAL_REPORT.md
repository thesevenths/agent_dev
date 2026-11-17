# M7.3 Final Report — Research Memory Benchmark

## Verdict

**INCONCLUSIVE**. The four-arm pilot ran with matched context length, but the primary decision metric had a construct-validity defect and expert review was not performed. The result cannot be used to claim that Research Memory improves research.

## What was executed

- 10 curated verified-failure memories from M7.1 audited defects.
- 10 research-method scenarios with isolated ground truth.
- 40 real LLM calls: OFF / ON / SHUFFLED / EQUAL_TOKEN.
- All raw decisions, trajectories, context SHAs and context lengths frozen under `M7_3_MEMORY/pilot_run/`.

## Key pilot signals

- ON used relevant memory IDs in 10/10 scenarios.
- SHUFFLED used relevant memory IDs in 0/10 and produced harmful memory usage in 3/10.
- Context length was matched at 900 chars for ON, SHUFFLED and EQUAL_TOKEN.
- The current `repeat_failure` metric gave ON 0.90, which manual inspection shows is partly an instrument interpretation problem rather than clean evidence of harm.

## Interpretation

Memory use and context-length effects are separable in this benchmark, which satisfies the main experimental-design requirement. However, “memory was used” does not yet mean “research quality improved”. The formal capability claim remains blocked until the instrument is repaired and expert-reviewed decisions are available.
