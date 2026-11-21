# M7.4 Rollback Report

Status: **NOT_APPLICABLE_NO_ADOPTION**.

No Candidate reached `APPROVED` or `ADOPTED` in the pilot:

- Round 1 development delta: -0.333.
- Round 2 development delta: 0.000.
- Pre-registered MME: +0.10.
- Pre-registered minimum families: 8; pilot used 6.

Round 2 showed a positive holdout delta (+0.25), but the pre-registered selection rule uses development results to choose a Candidate before holdout inspection. Promoting it post hoc would be contamination and is therefore refused.

Because no Candidate was adopted, there is no production harness version to roll back. The existing M6 `real_rollback` mechanism remains available but was not exercised in this pilot.
