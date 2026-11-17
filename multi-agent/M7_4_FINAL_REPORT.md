# M7.4 Final Report — Recursive Improvement Benchmark

## Verdict

**INCONCLUSIVE**. The pilot demonstrates that ARR has a mechanism to propose single-component Candidate methods, run paired Base/Candidate experiments, evaluate an isolated holdout, and refuse adoption when the preregistered development gate is not met. It does **not** demonstrate successful recursive improvement.

## Executed

- 2 pilot rounds, each changing only `scientific_judgment_instruction`.
- 6 paired development tasks per round.
- 4 paired holdout tasks for the preregistered best Candidate.
- Real LLM trajectories, raw scores and promotion log frozen under `M7_4_RSI/`.

## Results

| Round | Candidate | Development delta | Holdout delta | Decision |
|---|---|---:|---:|---|
| 1 | evidence-gate | -0.333 | not evaluated | below MME |
| 2 | adversarial | 0.000 | +0.250 | development gate failed; no adoption |

Pre-registered MME was 0.10 and minimum families was 8. The pilot used 6 development pairs, so even a positive result would not meet the power requirement.

## Interpretation

- **Improvement mechanism exists**: yes, Candidate methods were generated, paired and evaluated.
- **Improvement succeeded**: not demonstrated. The best development Candidate failed the MME.
- **Holdout generalization**: a post-selection positive signal exists but cannot be used for adoption because it would violate the preregistered workflow.
- **Post-adoption verification**: not run.
- **Rollback**: not applicable because nothing was adopted.

A formal RSI PASS requires more rounds, at least 8 independent families per decision, expert-blind outcome scoring, and a Candidate that passes both development and untouched holdout criteria.
