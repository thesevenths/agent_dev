# M7.2 Final Report — Scientific Judgment Benchmark (Pilot)

## Verdict

**INCONCLUSIVE**. The pilot executed correctly and produced auditable raw trajectories, but sample size is 14 and independent expert blind review was not performed. These are explicit stop conditions in the benchmark specification.

## What was run

- 50-case corpus built and isolated from ground truth.
- 14 cases selected: 2 per stratum.
- Real LLM agent, fixed prompt, temperature 0, JSON output.
- Raw results and trajectories frozen under `M7_2_BENCHMARK/pilot_run/`.

## Measured pilot metrics

| Metric | Value |
|---|---|
| Scored cases | 14/14 |
| Novelty accuracy | 1.000 (only NOT_NOVEL and NOVELTY_UNCERTAIN appeared in pilot) |
| Falsifiability confusion | 9 correct FALSIFIABLE, 3 FALSIFIABLE→UNFALSIFIABLE, 2 correct UNFALSIFIABLE |
| Evidence sufficiency | 8/14 correct; 4 SUFFICIENT→INSUFFICIENT, 1 CONTRADICTORY→INSUFFICIENT |
| Abstention | TP 6, FP 4, FN 0, TN 4; precision 0.60, recall 1.00, F1 0.75 |
| Novelty CI | not decision-grade at n=14 |
| Expert blind review | NOT_PERFORMED |

## Interpretation

The pilot does not support a claim that ARR has validated Scientific Judgment. It shows:

- the benchmark runner and contamination boundary work;
- the model tends to abstain when expected, but over-abstains on cases where evidence is declared sufficient;
- falsifiability classification has errors on genuinely falsifiable cases;
- novelty accuracy is inflated by a narrow label distribution in the pilot.

## Required next step

Run the full 50 cases only after independent expert review of ground truth. If reviewers disagree, adjudicate before scoring. If more statistical power is needed, add new reviewed cases rather than repeatedly tuning on the same set.
