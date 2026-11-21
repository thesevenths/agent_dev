# M4.1-00.1 — DiscoverySnapshot Contract (stable interface)

This document is the contract that downstream M4.1 sub-steps (Candidate
Generator, Critic, Ranker) and M4.2 Director MUST depend on. The provider is
**read-only**; it never writes, never recomputes a verdict, and never fabricates
an entity.

## Entry point

```python
from research_runtime.discovery_context_provider import (
    DiscoveryContextProvider, DiscoverySnapshot,
    DISCOVERY_SNAPSHOT_SCHEMA_VERSION,
)

provider = DiscoveryContextProvider(stores=stores, workspace=workspace)
snap: DiscoverySnapshot = provider.build_snapshot(run_id, objective_ids=None)
```

* Call `build_snapshot()` exactly **once per run**. It performs a **single**
  `DiscoveryEvidenceAdapter.query_run()` read (no per-access re-reads of the
  38 MB evidence.jsonl).
* `objective_ids=None` derives the objective list from the evidence-derived
  domains: `obj_{run_id}_{domain}` for each domain.

## `DiscoverySnapshot` shape

| Field | Type | Meaning |
|-------|------|---------|
| `run_id` | str | the run being discovered |
| `schema_version` | str | `DISCOVERY_SNAPSHOT_SCHEMA_VERSION` |
| `generated_at` | str | build stamp (excluded from `data_dict()`) |
| `evidence_view` | dict | the unified Evidence View (read-only) |
| `objectives` | list[dict] | resolved objectives, each with `content_sha256` |
| `links` | dict | `run_stats` + `by_objective{oid: graph}` |
| `sessions` | list[dict] | **flat** list of ALL sessions across objectives |
| `sessions_by_objective` | dict[oid, list[dict]] | **nested** view = same data, grouped |
| `experience_available` | bool | any session has >=1 experience record |
| `failures` | dict | honest, boundary-explicit failure accounting |
| `provenance` | dict | `read_only: True`, source list, counts |
| `unresolved_gaps` | list[dict] | explicit gaps (session/experience/diagnosis/evidence) |

## Session dict (per entry in `sessions` / `sessions_by_objective[*]`)

| Field | Meaning |
|-------|---------|
| `objective_id` | owning objective |
| `session_available` | False => no session recorded (M3.4 history) |
| `session_id` / `session_status` / `session_index` | identity + position |
| `events` | session events (name + payload + created_at) |
| `reports` | research reports (pydantic dicts) |
| `experience` | experience records (verbatim; empty if none) |
| `experience_available` | `len(experience) > 0` |

## Failure accounting — three layers, never conflated (P3)

| Key | Source | Never... |
|-----|--------|----------|
| `evidence_unresolved` | unified view | guessed |
| `session_failure_signals` | session **event names** matching a failure marker | treated as a scientific conclusion |
| `structured_diagnoses` | `FailureDiagnoser.diagnose_all(result)` | inferred from log text |

* During snapshot building there is **no `ExperimentResult`**, so
  `structured_diagnoses` is always `[]` and `diagnosis_status.structured_
  diagnosis_available` is `False`.
* Structured diagnosis is produced ONLY by `provider.diagnose_result(result)`
  on a real `ExperimentResult` (forward wiring for M4.1-01/02).
* When failure signals exist but no structured result is available, an explicit
  `diagnosis_unavailable` gap is recorded. **No conclusion is guessed.**

## Traceability invariants (P4)

* Every objective carries `content_sha256`.
* Every evidence entity carries `evidence_id` = `sha256:...` (SHA-256 of real
  bytes), with `sources` / `manifest_sha256` / `metrics_files`.
* Every link carries `source_sha256` + `source_path` (or at least one).
* Every experience carries `experience_id` + `session_id` and is surfaced under
  that exact session.

## Do / Don't for downstream modules

DO:
* Read `build_snapshot()` output only.
* Treat `unresolved_gaps` as hard constraints (missing Experience => do not
  assume a conclusion).

DON'T:
* Reach into individual stores through the provider.
* Assume M3.4 history carries Experience — it does NOT; it is `unresolved`.
* Treat `session_failure_signals` as a verdict.
* Expect `structured_diagnoses` to be populated during snapshot building.

## Change policy

`DiscoverySnapshot` fields may be **added** but existing field semantics are
stable. `build_snapshot()` is the only supported constructor entry point. Any
breaking change bumps `DISCOVERY_SNAPSHOT_SCHEMA_VERSION` and is reflected here.
