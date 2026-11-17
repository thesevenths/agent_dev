# M7.1 E2E Report — Clean Commit Replay

## Run identity

- Baseline commit replayed: `729184c4d8a217d121f577243cc4b1f987e7922a`
- Source snapshot: `git archive` of exact commit; no earlier dirty working tree used
- Working tree: clean by construction (`untracked_files: []`)
- Run ID: `m7_1_clean_729184c4`
- Result SHA: `c4d5105e940a15cf8c50beff79c1dcea3249bfad6d273c77b1dc240c2c20efd8`
- Manifest: `M7_1_EVIDENCE_MANIFEST.json`
- Evidence directory: `M7_1_EVIDENCE/e2e_clean`

## Result

- Engineering status: `COMPLETED`
- Scientific verdict: `INCONCLUSIVE`
- Runtime journal: verified
- Engine result SHA verification: passed
- Model: recorded in manifest from real trajectory (no API key stored)
- Environment: recorded in manifest (Python, PyYAML, pydantic, pytest, torch, transformers, CUDA availability)

## What this clean replay proves

It proves the old M7 code at the exact commit can execute the same local orchestration path from a clean source snapshot and produce the same honest class of conclusion (`INCONCLUSIVE`).

It does **not** prove full production Dispatcher / Campaign Lifecycle / HarnessLoader integration. The runtime audit found those paths bypassed in baseline code.

## What changed after this replay

M7.1 source fixes were applied after this replay. They must not be attributed to the baseline run:

- evidence-gated Research Value and explicit EIG
- abstract-grounded question evidence and Memory retrieval trace
- LLM proposal/critic/execution ledger
- evidence-limited Novelty states
- adversarial Critic checks
- family-confirmation fail-closed and pilot power gate
- budget reservation consumption
- optional formal HarnessLoader binding

A subsequent M7.1 validation run is required before claiming these fixes behave correctly in a full run; unit and engine-free regression tests provide the first verification layer.
