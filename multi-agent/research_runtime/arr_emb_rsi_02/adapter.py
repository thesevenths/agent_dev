"""ARR_EMB_RSI_02 adapter: immutable config snapshots (V1 shallow-copy bug fixed)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ..arr_emb_rsi_01.adapter import WorldModelAdapter as V1WorldModelAdapter


class WorldModelAdapter(V1WorldModelAdapter):
    """V1 adapter with deep config snapshots for main/OOD.

    V1 used a shallow dict copy; the OOD config could mutate the main config's
    nested `data` mapping. P0.2 forbids that. This subclass overrides only the
    config builder; all engine execution still uses the real SciNetAdapter.
    """

    def build_config(self, *, run, overrides: dict[str, Any],
                     config_kind: str = "main",
                     source_config: dict[str, Any] | None = None):
        cfg = copy.deepcopy(source_config if source_config is not None
                            else self._load_base())
        if config_kind == "ood" and source_config is None:
            ood = yaml.safe_load((self.project / self.ood_config).read_text(
                encoding="utf-8")) or {}
            cfg["data"] = copy.deepcopy(dict(ood.get("data") or cfg.get("data") or {}))
        for key, value in overrides.items():
            node = cfg
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):
                    raise ValueError(f"config path {key!r} is not a mapping")
            node[parts[-1]] = copy.deepcopy(value)
        path = self.config_root / (
            f"{run.failure_id}__{run.arm}__s{run.seed}__{config_kind}.yaml")
        path.write_text(yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True),
                        encoding="utf-8")
        return str(path), cfg


__all__ = ["WorldModelAdapter"]
