# Research Autonomy Master Specification

> Scope: defines the architecture, contracts, and governance for a *research-autonomous*
> agent that can plan, execute, audit, and evolve scientific experiments across three
> domains — **AI4Science (AI4S)**, **Embodied AI / Robotics**, and **Quantitative
> Finance (Quant)** — without re-implementing the generic agent control plane.
>
> Status: P1-B1.1 (Scientific Provenance & Independent Audit Hardening) is **complete**
> and serves as the reference implementation of the cross-cutting RRSI guarantees
> (see §7). This document is the master blueprint; the executable plan is
> `research_autonomy_roadmap.yaml`.

---

## 1. Purpose & Non-Negotiable Constraints

The system must conduct real scientific work autonomously **while preserving
reproducibility and independent verifiability**. Autonomy is *bounded* by the
following standing STOP rules (inherited from P1-A / P1-B1 and applicable to all
domains unless explicitly lifted by a human approval gate):

1. **No model training without explicit approval.** JEPA / world-model / policy
   training is never triggered automatically.
2. **No full-scale data generation.** Pilots are bounded (≤ 20–50 episodes by
   default); scaling up requires an approved compute + provenance plan.
3. **No silent phase advancement.** A research line may not auto-advance to the
   next phase (e.g. P1-B1 → P1-B2) until its acceptance criteria are met *and* any
   gating flag is cleared by a human.
4. **Fail-closed on provenance.** If source identity, dependency versions, or
   dataset integrity cannot be verified, the run is rejected and any partial
   output is destroyed — never silently kept.

These rules are enforced in code (see `data/source_provenance.py`) and are
surfaced as hard gates in every domain adapter.

---

## 2. Three Research Domains

| Domain | What an "experiment" is | Domain-specific risk the autonomy layer must handle |
|--------|------------------------|----------------------------------------------------|
| **AI4S** | Simulator-based dynamics collection + representation/world-model training (e.g. controlled-pendulum → JEPA). | Numerical fidelity, determinism of ODE integration, regime preregistration, leakage across train/val/test. |
| **Embodied / Robotics** | Real or simulated robot control policies, sim-to-real transfer, manipulation/ navigation episodes. | Hardware/safety constraints, non-determinism of physics engines, actuator bounds, episode reset isolation. |
| **Quant** | Backtested strategies, signal/research notebooks, market-data pipelines. | Look-ahead bias, survivorship bias, point-in-time data correctness, regime/regime-shift detection, regulatory constraints. |

All three share the *same* control-plane lifecycle (propose → execute → audit →
decide) but differ in (a) the experiment adapter, (b) the simulator/environment
contract, and (c) the domain-specific scientific reviewer.

---

## 3. Shared Runtime

The Shared Runtime is the substrate every domain reuses. It is **not** a new
orchestrator; it is the set of primitives the adapters call.

```
┌──────────────────────────────────────────────────────────────┐
│                    Shared Runtime                              │
│                                                                │
│  • Execution sandbox      (git-worktree isolation, resource    │
│                           caps, deterministic RNG seeding)     │
│  • Compute scheduler      (budget tracking, GPU/CPU queues)   │
│  • Data store             (versioned datasets, npz + manifest) │
│  • Provenance service     (git SHA, file SHA-256, dep versions)│
│  • Audit service          (independent re-derivation, replay)  │
│  • Artifact registry      (reports, evidence JSON, models)     │
└──────────────────────────────────────────────────────────────┘
        ▲                ▲                ▲
        │ adapters       │ adapters       │ adapters
   ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
   │  AI4S   │      │ Embodied│      │  Quant  │
   └─────────┘      └─────────┘      └─────────┘
```

The Shared Runtime primitives are **domain-agnostic**: they never know what a
"pendulum" or a "portfolio" is. They only know: *run this command in an isolated
tree, record its provenance, verify the outputs, and store them immutably.*

---

## 4. Adapter Contracts

Each domain plugs in via a single typed adapter. The autonomy layer never
special-cases domain logic; it only consumes the adapter interface.

```python
class DomainAdapter(Protocol):
    domain: str                                   # "ai4s" | "embodied" | "quant"

    def build_experiment_spec(self, plan: dict) -> ExperimentSpec:
        """Turn a planner proposal into a concrete, runnable spec
        (params, seeds, bounds, preregistered hypotheses/regimes)."""

    def instantiate_environment(self, spec: ExperimentSpec) -> Environment:
        """Return an object satisfying the Environment contract below."""

    def collect(self, env: Environment, spec: ExperimentSpec) -> Dataset:
        """Drive the env, return a versioned Dataset (npz + manifest)."""

    def domain_reviewer(self) -> ScientificReviewer:
        """Return the domain-specific independent reviewer (see §6)."""

    def regime_criteria(self) -> dict:
        """Preregistered success/failure criteria, loaded from YAML
        (NEVER hardcoded — P8 lesson)."""
```

```python
class Environment(Protocol):
    def reset(self, seed: int) -> State: ...
    def step(self, action) -> tuple[State, float, bool, dict]: ...
    def action_bounds(self) -> tuple: ...          # actuator / action limits
    def deterministic(self) -> bool: ...           # must be True for replay
```

The AI4S adapter is **already implemented and proven** (SciNet++ controlled
pendulum): `data/controlled_pendulum.py`, `data/controlled_pendulum_v2.py`,
`scripts/collect_controlled_pendulum_v2.py`. Embodied and Quant adapters are
roadmap tasks (see `research_autonomy_roadmap.yaml`, P2-01 / P3-01).

---

## 5. Experiment Protocol

Every experiment, regardless of domain, follows:

1. **Propose** — Planner emits a plan with explicit, preregistered hypotheses
   (including regime/acceptance criteria loaded from YAML).
2. **Provenance gate (P6)** — Before any collection: record git HEAD, source-file
   SHA-256, dependency versions (incl. SciPy/torch/numpy), and require a clean
   tracked tree (or explicit `--allow-dirty` with provenance marked incomplete).
3. **Execute in isolation** — Run inside a git worktree so the production tree
   stays clean and the run is reproducible.
4. **Stratified design (P7)** — If the experiment produces a train/val/test split,
   allocate by strata (action profile, physical/initial-condition bins, regime)
   with deterministic largest-remainder allocation, explicit `min_per_cell`, and
   honest `BLOCKED`/`INCONCLUSIVE` status for unreachable strata. **No fabrication
   of balance.**
5. **Independent audit (P8)** — A reviewer *recomputes* every claimed quantity from
   raw observations using YAML-declared criteria and compares to the manifest.
   Hard failures on label/count/threshold tampering or config inconsistency.
6. **Decide** — Reviewer verdict + Evaluator gate decide: accept, iterate, or stop.
   Phase advancement requires the gate to clear (no auto-advance).

---

## 6. Evaluator / Scientific Reviewer

The **Scientific Reviewer is independent of the collector**: it never trusts the
manifest's self-reported numbers. It:

- reloads raw observations/conditions from the dataset,
- re-derives regime labels, aggregate counts, energy/determinism checks, and
  SHA-256 content hashes from scratch,
- compares each re-derived value to the manifest and fails closed on mismatch.

The reference implementation is
`scripts/audit_controlled_pendulum_v2.py::independent_regime_audit` plus the
`[PASS] independent_regime_audit` check (one of 10 audit checks). Every domain
adapter must supply an equivalent independent reviewer.

The **Evaluator** consumes reviewer output + a predefined acceptance rubric and
returns `ACCEPT | ITERATE | STOP`. It is the only component allowed to recommend
phase advancement, and it may only do so when the human-approval gate (§8) is
clear.

---

## 7. RRSI — Research Reproducibility & Scientific Integrity

RRSI is the cross-cutting guarantee layer. Its building blocks are **already
implemented and tested** in SciNet++ P1-B1.1:

| Guarantee | Module / Anchor | Status |
|-----------|----------------|--------|
| Source provenance (git SHA, file SHA-256, dep versions, worktree isolation, fail-closed) | `data/source_provenance.py`, `scripts/collect_controlled_pendulum_v2.py` | ✅ implemented + 14 tests |
| Stratified, honest split (no leakage, no fabrication, `BLOCKED`/`INCONCLUSIVE`) | `data/stratified_split.py`, `scripts/compute_stratified_split_v2.py` | ✅ implemented + 10 tests |
| Independent regime audit (YAML criteria, recompute-from-raw, hard gate) | `data/controlled_pendulum_v2.py::classify_regime`, `scripts/audit_controlled_pendulum_v2.py::independent_regime_audit` | ✅ implemented + 10 tests |
| Exact replay (bit-identical regeneration from recorded params) | `scripts/verify_historical_datasets.py` | ✅ verified: V1 & V2 max err = 0.0 |

RRSI is **domain-agnostic in spirit**; the AI4S adapter proves the pattern, and
the Embodied/Quant adapters must provide their own `regime_criteria` YAML and
independent reviewer while reusing the same provenance + replay machinery.

---

## 8. Safety / Approval & Compute Budgets

- **Human-approval gate**: a boolean per phase (`requires_approval`). Training,
  full-data generation, and phase advancement are blocked while `True`.
- **Prohibited actions** (system-enforced, never auto-executed):
  training runs, unbounded data generation, destructive dataset overwrites of
  historical (provenance-incomplete) datasets, and any mutation of `main`.
- **Compute budgets**: each experiment declares `episodes`, `horizon`,
  `accelerator`, and a wall-clock/budget cap. The Shared Runtime scheduler refuses
  to launch over-budget runs. Budgets are recorded in the manifest's provenance
  block.
- **Branch hygiene**: all work lands on feature branches; `main` is never
  mutated by the agent; no PR is opened without an explicit human instruction.

---

## 9. Research Memory

Research Memory persists *decisions and evidence*, not raw data. It reuses the
shared control-plane memory from `agent_dev/multi-agent` (the existing
Planner/Executor/Reviewer/Memory implementation) and adds a domain layer:

- **Decision log**: why a phase started/stopped, which hypotheses were confirmed.
- **Evidence index**: pointers to manifest + audit + replay evidence files.
- **Negative results**: failed regimes, blocked strata, rejected runs — retained
  as first-class knowledge to prevent re-attempts.

SciNet++ currently keeps this in `datasets/*/manifest.json` provenance blocks,
`*_audit_report.json`, and `historical_replay_evidence.json`. Consolidation into
the shared memory plane is roadmap task **P5-01**.

---

## 10. Diagnosis / Evolution Engine

When the Evaluator returns `ITERATE` or `STOP`, the Diagnosis Engine:

1. classifies the failure (provenance break, regime miss, non-determinism,
   budget exhaustion, scientific null result),
2. queries Research Memory for prior similar failures,
3. proposes a targeted remediation (e.g. tighten `regime_criteria`, add a stratum,
   fix an integrator), and
4. emits a *new* plan for the Planner — closing the loop without human
   intervention **except** where the remediation touches a prohibited action.

The Evolution Engine promotes *validated* remediations into durable adapter /
runtime improvements (the P6/P7/P8 hardening itself is an example of such a
promotion).

---

## 11. Dependency Graph (summary)

```
P1-A  ──► P1-B1 ──► P1-B1.1 (RRSI hardening, DONE)
                     │
                     ├──► P1-B2   [BLOCKED: requires human approval; training]
                     │
                     ├──► P2-01   Embodied adapter + provenance harness   (next auto-runnable)
                     │       └──► P2-02   Embodied independent reviewer
                     │
                     ├──► P3-01   Quant adapter + point-in-time data review
                     │
                     ├──► P4-01   Cross-domain Diagnosis/Evolution engine
                     │
                     ├──► P5-01   Research Memory consolidation
                     └──► P6-01   Safety/Approval + Compute budget enforcement
```

Full machine-readable graph with acceptance criteria, verification commands, and
evidence outputs: **`research_autonomy_roadmap.yaml`**.

---

## 12. Relationship to the Shared Control Plane

This specification deliberately **does not** re-implement Planner, Executor,
Reviewer, or Memory. Those are provided by the existing shared control plane at
`agent_dev/multi-agent` (the multi-agent orchestration framework). SciNet++ and
the other domain projects supply only:

- the **Domain Adapters** (§4),
- the **RRSI enforcement primitives** (§7),
- the **domain-specific Scientific Reviewers** (§6), and
- the **Experiment Protocol glue** (§5).

This separation keeps the autonomy logic reusable across AI4S / Embodied / Quant
while ensuring scientific integrity is a property of the runtime, not of any
single experiment.
