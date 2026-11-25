# ARR-07 autonomous research loop — README

## What it is

An agent that runs the full chain **实验 → 看数据 → 改方案 → 再实验 → 达标**
on the REAL SciNet++ engine (subprocess JSON-RPC into `conda scinet_gpu`), with
the LLM holding the two decision rights previously held by harness:

1. **① LLM proposes the experiment** — model structure and hyperparameters
   (`generic/llm_autonomy.py: LLMResearcher.propose_experiment`). NOT a human menu.
2. **② Real execution** — `arr07/scinet_backend.py` materialises a fresh YAML,
   runs genuine SciNet++ `train` then `evaluate`; raw records include config
   hash, run_id, job ids, metrics, result hash.
3. **③ LLM reads measured data, decides finish / continue** — `decide_finish`
   quotes the numbers it relies on; decisions persisted verbatim as evidence.
4. Budget exhaustion recorded SEPARATELY (`budget_exhausted`); never silently
   rewritten into an LLM "finish" verdict.

## Entry point

```bash
cd /e/agent_dev/multi-agent
PYTHONPATH=/e/agent_dev/multi-agent /d/python313/python.exe -u -c "
from pathlib import Path
from research_runtime.arr07.autonomous_scinet_loop import run_autonomous
r = run_autonomous(task_id='my_study', root=Path('experiments/campaigns/MY-RUN'),
                   seed=42, max_iterations=3, target=0.10)
print(r.to_dict())
"
```

Requirements: `.env` LLM_* (loaded via `import config`); SciNet++ at
`E:\LLM\SciNet++` with `D:\conda_envs\scinet_gpu\python.exe` (override via
`SCINET_DIR` / `SCINET_PYTHON`).

## Guardrails

- Schema validation against REAL SciNet config keys + type/scale; invalid
  proposals are REJECTED and logged, never silently repaired.
- Failed engine runs produce `improvement=None`, NEVER a fake 100% gain.
- LLM sees measured numbers only; every call/metric/decision written to
  `raw_data/autonomy/` and `raw_data/scinet/` for audit.
- `budget_exhausted` is distinct from LLM's own `finish` decision.

## Evidence (real, not synthetic)

- qwen3.7-flash + real SciNet++:
  - V2 run: iter 0 degraded 25.6% -> LLM said continue; iter 1 revised params
    and improved MSE 33.2% (0.0264 -> 0.0176).
  - LeJEPA-style VICReg study: `experiments/campaigns/ARR-07-LEJEPA-AUTO`.
- 18 unit tests in `generic/tests` + `arr07/tests`.

## Honest limitations

- Schema prevents crashes, not bad science. LLM once proposed int for hidden
  layers (fixed by schema).
- A 33% MSE gain came with cosine@1 dropping 0.544 -> 0.315; LLM itself judged
  the dual objective unmet. Reported honestly.
- Small frozen smoke dataset (64 samples, 1 epoch); NOT a full V-JEPA/robot claim.

## Related modules

- `generic/identifiability.py` — detect indistinguishable experiments.
- `generic/stopping.py` — frozen GoalGate (older menu-based loop).
- `generic/experiment_design.py` — discriminative design from signatures.
- `generic/llm_autonomy.py` — the two decision rights used here.
- `arr07/scinet_loop.py` — older menu-based loop (kept for comparison).
