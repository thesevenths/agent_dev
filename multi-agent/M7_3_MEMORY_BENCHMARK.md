# M7.3 Memory Benchmark Specification

## Design

Four controlled arms use the same model, prompt, toolset, max tokens and scenarios:

- **OFF**: no context.
- **ON**: verified-failure memories whose applicability matches the scenario.
- **SHUFFLED**: same length, unrelated memory records.
- **EQUAL_TOKEN**: same length generic background text.

The SHUFFLED and EQUAL_TOKEN arms separate memory content from context-length effects.

## Dataset

- 10 curated verified-failure memories derived from M7.1 audited defects.
- 10 research-method scenarios with isolated ground truth.
- Ground truth includes relevant memory IDs and expected repeat-failure status.

## Tracing

Each raw result records context SHA, context length, trajectory SHA and parsed decision. The decision distinguishes:

- `used_memory_ids`
- `rejected_memory_ids`

This prevents “memory retrieved but ignored” from being counted as memory usage.

## Metrics

Per arm: repeated-failure rate, memory usage rate, harmful-memory rate, mean context length. No single aggregate ARR score is used.

## Verdict rule

`INCONCLUSIVE` if expert review was not performed or the number of scored results is below the pre-registered threshold. Memory storage or SHA-chain integrity alone never counts as a capability PASS.
