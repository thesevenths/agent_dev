# M7.5 Integrated Benchmark Specification

## Goal

Test whether Scientific Judgment, Research Memory and Recursive Improvement work together under a fixed research budget.

## Groups

- A_BASE: base judgment instruction.
- B_MEMORY: verified-failure memory added.
- C_RSI: adversarial RSI instruction added.
- D_FULL: both memory and RSI instruction.

Groups use the same model, prompt skeleton, cases, tool permissions, temperature and max tokens. Memory context length is recorded; token cost is reported per group.

## Primary outcome

Evidence-sufficiency accuracy. Secondary: novelty and falsifiability accuracy. MME = 0.10 absolute accuracy improvement over A_BASE. Minimum families = 8.

## Pilot status

The pilot selected 7 cases, so the power requirement is not met. The verdict must be INCONCLUSIVE regardless of observed deltas.

## Anti-contamination

- Ground truth is not included in prompts.
- Case IDs are fixed before execution.
- M7.4 holdout cases were not used to tune M7.5 prompts after results were seen.
- Human approval and production adoption are not simulated.
