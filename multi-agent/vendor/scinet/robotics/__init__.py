"""robotics -- P2-01 Pusher-v5 environment adapter and reproducible dataset.

This package is intentionally isolated from the existing Pendulum / Double
Pendulum training pipeline (see hard constraints in the task brief). Nothing here
imports ``torch`` and nothing here touches the V-JEPA / PySR / AdaJEPA stages.
"""

from __future__ import annotations
