# M7.3 Failure Analysis

## Observed pilot result

| Arm | n | repeated_failure_rate | memory_usage_rate | harmful_memory_rate | mean context chars |
|---|---:|---:|---:|---:|---:|
| OFF | 10 | 0.30 | n/a | n/a | 0 |
| ON | 10 | 0.90 | 1.00 | 0.00 | 900 |
| SHUFFLED | 10 | 0.30 | 0.00 | 0.30 | 900 |
| EQUAL_TOKEN | 10 | 0.50 | n/a | n/a | 900 |

## Construct-validity problem

The ON arm used relevant memories in 10/10 cases and had zero harmful-memory usage, yet its `repeat_failure` flag was higher than OFF. Manual inspection of `s01_ON.json` shows the model cited the memory correctly and proposed a corrected multi-seed/held-out design while still answering `repeat_failure=False`; most later ON cases answered `True` despite similarly citing the failure. This indicates that the `repeat_failure` field is ambiguous: the model sometimes interprets it as “does the described practice repeat a known failure?” and sometimes as “will my proposed design repeat it?”.

This is a benchmark instrument defect, not evidence that memory is harmful or beneficial. The result must remain INCONCLUSIVE.

## What the pilot does show

- Memory retrieval/context use can be measured separately from context length: ON and SHUFFLED both had 900 chars.
- The model used relevant memories more often than shuffled memories.
- Harmful memory usage occurred in the SHUFFLED arm (0.30).
- The current decision-quality metric is not reliable enough for a capability claim.

## Required repair before formal M7.3

1. Replace the ambiguous `repeat_failure` boolean with two fields: `input_practice_repeats_failure` and `proposed_design_repeats_failure`.
2. Add independent expert review of proposed experiment decisions.
3. Re-run all four arms with the repaired instrument.
4. Pre-register the primary metric before inspection.
