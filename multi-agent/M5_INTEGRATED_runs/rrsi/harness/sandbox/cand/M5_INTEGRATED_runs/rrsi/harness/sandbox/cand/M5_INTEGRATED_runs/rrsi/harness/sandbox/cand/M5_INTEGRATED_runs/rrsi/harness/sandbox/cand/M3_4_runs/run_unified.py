"""M3.4 -- REAL three-domain unified orchestration (Phase 5 E2E).

Drives scinet (REAL SciNet++ engine, ID + OOD regime-shift), embodied (REAL
robot simulator) and quant (REAL synthetic-market backtest) through the SAME
UnifiedResearchScheduler used by the engine-free tests. Every domain is isolated
by a unique campaign_id = run_id_domain and a per-domain run_root; the frozen
datasets / baseline configs are shared read-only (content-addressed, SHA-verified)
by design.

This is ONE command that performs the M3.4 acceptance procedure:
  1. register the three domains,
  2. check engines / datasets / configs,
  3. run each domain's research task (baseline + candidate comparison),
  4. produce an isolated evidence package per domain,
  5. update each domain's version-inheritance record,
  6. aggregate the three domains' results,
  7. write scheduler_state.json + evidence_index.json + result.json.

No FakeRunner anywhere: the domain runner is the production
``cli._build_real_domain_runner``. Quant is flagged synthetic in the evidence
index (never presented as real-market alpha).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research_runtime.cli import _build_real_domain_runner  # noqa: E402
from research_runtime.scheduler import (  # noqa: E402
    DomainSpec, UnifiedResearchScheduler,
)
from research_runtime.domains.embodied import (  # noqa: E402
    PROJECT_DIR as EMBODIED_PROJECT_DIR, read_dataset_version as embodied_dv,
)
from research_runtime.domains.quant import (  # noqa: E402
    PROJECT_DIR as QUANT_PROJECT_DIR, read_dataset_version as quant_dv,
)
from research_runtime.settings import DEFAULT_SCINET_DIR  # noqa: E402

SCINET_PROJECT_DIR = (REPO / "M2_7_runs" / "scinet").resolve()
SCINET_BASE_CONFIG = (SCINET_PROJECT_DIR / "configs" / "pendulum_smoke.yaml").resolve()
SCINET_OOD_CONFIG = (DEFAULT_SCINET_DIR / "configs" / "pendulum_smoke_ood.yaml").resolve()
SCINET_ID_DATASET_VERSION = "pendulum-a0e03386be89"

# Formal-ACCEPT floor is 5 paired seeds (M3.0); 5 holdout seeds for independent
# confirmation. Kept at the minimum that still yields a formal verdict so the
# real SciNet++ engine run stays within a tractable wall-clock budget.
PAIRED = (50, 51, 52, 53, 54)
HOLDOUT = (60, 61, 62, 63, 64)

RUN_ID = "arr_m34_001"
RUN_ROOT = REPO / "M3_4_runs" / RUN_ID


def _check_preconditions() -> list[str]:
    problems = []
    if not SCINET_BASE_CONFIG.is_file():
        problems.append(f"scinet base config missing: {SCINET_BASE_CONFIG}")
    if not SCINET_OOD_CONFIG.is_file():
        problems.append(f"scinet OOD config missing: {SCINET_OOD_CONFIG}")
    if not (SCINET_PROJECT_DIR / "datasets" / "pendulum_smoke_v1").is_dir():
        problems.append("scinet ID frozen dataset missing")
    if not (DEFAULT_SCINET_DIR / "datasets" / "pendulum_smoke_ood_v1").is_dir():
        problems.append("scinet OOD frozen dataset missing (must live in engine repo)")
    if not (EMBODIED_PROJECT_DIR / "config.yaml").is_file():
        problems.append("embodied config missing")
    if not (QUANT_PROJECT_DIR / "config.yaml").is_file():
        problems.append("quant config missing")
    return problems


def _specs() -> list[DomainSpec]:
    return [
        DomainSpec(
            domain="scinet", base_config=str(SCINET_BASE_CONFIG),
            dataset_version=SCINET_ID_DATASET_VERSION, paired_seeds=PAIRED,
            holdout_seeds=HOLDOUT, project_dir=str(SCINET_PROJECT_DIR),
            metric_direction="minimize", ood_config_path=str(SCINET_OOD_CONFIG),
        ),
        DomainSpec(
            domain="embodied", base_config=str(EMBODIED_PROJECT_DIR / "config.yaml"),
            dataset_version=embodied_dv(), paired_seeds=PAIRED,
            holdout_seeds=HOLDOUT, project_dir=str(EMBODIED_PROJECT_DIR),
            metric_direction="minimize",
        ),
        DomainSpec(
            domain="quant", base_config=str(QUANT_PROJECT_DIR / "config.yaml"),
            dataset_version=quant_dv(), paired_seeds=PAIRED,
            holdout_seeds=HOLDOUT, project_dir=str(QUANT_PROJECT_DIR),
            metric_direction="maximize", synthetic_data=True,
        ),
    ]


def main() -> dict:
    problems = _check_preconditions()
    if problems:
        raise SystemExit("precondition failures (fail-closed):\n  - "
                         + "\n  - ".join(problems))

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"START M3.4 unified orchestration run_id={RUN_ID} "
          f"{time.strftime('%H:%M:%S')}", flush=True)
    print(f"  run_root={RUN_ROOT}", flush=True)
    print(f"  scinet project_dir={SCINET_PROJECT_DIR} (engine CWD stays "
          f"DEFAULT_SCINET_DIR)", flush=True)
    print(f"  paired_seeds={PAIRED} holdout_seeds={HOLDOUT}", flush=True)

    scheduler = UnifiedResearchScheduler(
        run_id=RUN_ID, run_root=RUN_ROOT, repo_root=REPO,
        domains=_specs(), build_runner=_build_real_domain_runner)
    out = scheduler.run()
    out["wall_clock_seconds"] = round(time.time() - t0, 2)

    result_path = RUN_ROOT / "result.json"
    result_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8")

    print(f"\nDONE run_id={RUN_ID} status={out['status']} "
          f"elapsed={out['wall_clock_seconds']:.1f}s", flush=True)
    print(f"  completed={out['completed']} "
          f"skipped_resumed={out['skipped_resumed']} failed={out['failed']}",
          flush=True)
    for name, d in out["per_domain"].items():
        if d.get("status") != "completed":
            print(f"  {name}: {d.get('status')} {d.get('error','')}", flush=True)
            continue
        promo = (d.get("promotion_recommendation") or {}).get("chosen")
        print(f"  {name}: accept={d['n_accepted']} reject={d['n_rejected']} "
              f"inconclusive={d['n_inconclusive']} "
              f"synthetic={d.get('synthetic_data')} "
              f"promotion={promo}", flush=True)
    print(f"  evidence_index -> {RUN_ROOT / 'evidence_index.json'}", flush=True)
    print(f"  scheduler_state -> {RUN_ROOT / 'scheduler_state.json'}", flush=True)
    return out


if __name__ == "__main__":
    main()
