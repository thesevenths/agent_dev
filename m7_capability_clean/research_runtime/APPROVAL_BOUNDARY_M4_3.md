# Approval Boundary — M4.3-00 / P0-06 (statement of truth)

This document states what the ARR 1.1 approval boundary **actually** enforces
today. It is deliberately a *trust-model description*, not a security
attestation. No enterprise identity/authentication is implemented in this
stage; where the boundary is only a convention, that is called out explicitly.

## 1. Verified facts

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 1 | The Director never grants an approval for itself | **Enforced in code** | `research_director.py` exposes no grant method; `run_to_completion()` only *observes* `ApprovalStore.get()`. `test_director_has_no_self_grant_path` asserts no `approve_approval`, no `approval_granter` injection, and no `approvals.grant(` call in the module. |
| 2 | The Agent-facing tool surface exposes NO grant tool | **Enforced in code** | `research_runtime/tools.py`: `CAMPAIGN_RESEARCH_TOOLS` = 11 tools (start session → get report) and `RESEARCH_TOOLS` = 4 tools. Neither list contains `grant`/`approve`. The strongest record an Agent tool can create is `status="pending"` via `request_plan_approval`. |
| 3 | `approval_cli` is a separate, out-of-band operator entry point | **Enforced in code** | `research_runtime/research_store/approval_cli.py` runs as its own process (`python -m ... approval_cli grant ...`). It is deliberately NOT an Agent tool. The real-engine E2E and the P6 engine-free test shell out to it as a subprocess. |
| 4 | `ApprovalStore.grant()` is still a plain Python Store API | **TRUE — see §3** | `approval_store.py::grant(approval_id, *, approved_by, source, current_resource_sha256, notes)` is a public, importable method. Any Python code running in the same process/environment can call it. |

## 2. What the boundary DOES enforce

* The **12-step fail-closed gate** re-reads plan integrity, approval status,
  approval source, approver presence, scope, and the Plan/Approval SHA-256
  binding from trusted storage before any campaign can execute
  (`research_service._authorize`). A pending / revoked / SHA-mismatched
  approval cannot execute.
* `ApprovalStore.grant()` re-reads the CURRENT plan hash and refuses a grant if
  the plan changed while the request waited; a decision is immutable (a second
  grant raises `ResearchStoreConflict`).
* The ResearchDirector adds a second, independent SHA-binding check on top of
  the service gate.
* Campaign execution is idempotent per `(plan_id, approval_id, plan_sha256)`
  and recovery never re-submits an already-submitted training job (M4.3-00
  P0-03). Recovery cannot bypass an approval.

## 3. What the boundary does NOT enforce (current trust model)

> **Read this before making any security claim about the approval record.**

* **`approved_by` is an unauthenticated string.** `grant()` only checks that it
  is non-empty (`if not approved_by or not approved_by.strip()`). It is NOT a
  strong identity: no login, no signature, no certificate, no hardware token,
  no ACL lookup. `"alice"` proves nothing about who actually typed it.
* **`source="human"` is a caller-supplied label**, not a cryptographic
  provenance claim. `grant()` accepts `human` or `system_policy` from a string
  argument. The gate trusts the *stored* value, not the identity behind it.
* **`ApprovalStore.grant()` is reachable by ANY in-process Python caller.**
  The "out-of-band" property is a *deployment/tooling convention*: the Agent's
  registered tool list does not include a grant tool, and the documented human
  path is the standalone `approval_cli` process. It is NOT a sandbox or a
  process-isolation guarantee. A component with import access to the store can
  write `status="approved"` directly. The same applies to anything that can
  write to the SQLite database file.
* **No enterprise authentication is implemented in this stage** (by explicit
  scope decision). There is no per-user authorization, no audit-tamper proof,
  no separation between "operator" and "process" identities.

## 4. Accurate wording to use / avoid

| Use | Avoid |
|-----|-------|
| "The Director cannot self-grant; the Agent tool surface has no grant tool; the documented human path is the separate `approval_cli` process." | "Approvals are cryptographically authenticated." |
| "`approved_by` is an operator-supplied identity string, verified only for non-emptiness." | "`approved_by` proves the human identity." |
| "Out-of-band is a tool-surface/deployment convention, enforced against the Agent's registered tools and fail-closed gate — not a sandbox." | "Only a human can ever write an approval." |
| "The gate re-verifies plan SHA, source, scope and status from trusted storage." | "The system guarantees exactly-once, tamper-proof approval." |

## 5. Consequence for the M4.3 evaluation harness

A/B/C runs MUST all pass the same external approval step. For the E2E gates we
use the real `approval_cli` subprocess (or, in engine-free unit tests, the
exact `ApprovalStore.grant` method that console invokes) — this is a **genuine
store write**, but it is NOT an impersonation-proof human identity. The
harness records `approval_bypass_count` as a Safety metric so a run that
skips the external approval is detectable. No claim about human identity
strength is made anywhere in the value report.
