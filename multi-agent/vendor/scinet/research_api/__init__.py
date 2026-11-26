"""
research_api
============
Stable, engine-facing API for SciNet++ -- consumed by an external Research
Runtime (see ``multi-agent/research_runtime``).

Design contract
---------------
* The Agent Runtime never imports SciNet++ model code and never parses console
  output. It only speaks structured JSON over ``python -m research_api.cli``.
* Long-running training runs in a detached ``worker`` process that writes its
  state to a SQLite ``JobStore``. Querying a job never requires the submitting
  process to still be alive.
* Existing stage scripts (train.py / evaluate.py / analyze.py / tta.py /
  symbolic.py) are NOT modified. We read the artefacts they already produce
  (``checkpoint.pt``, ``horizon_metrics.npy``) and emit ``result.json``.

CLI examples::

    python -m research_api.cli submit --experiment-id exp1 --stage train \
        --config configs/pendulum.yaml --epochs 3
    python -m research_api.cli status --job-id <job_id>
    python -m research_api.cli wait   --job-id <job_id> --timeout 600
    python -m research_api.cli result --job-id <job_id>
"""

__version__ = "1.0.0"
