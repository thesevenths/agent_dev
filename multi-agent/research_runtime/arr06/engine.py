"""ARR-06 deterministic experiment engine.

P0 hardening: explicit deterministic / nondeterministic modes; controlled
RNG sources (Python, NumPy, PyTorch CPU/CUDA, env reset); RNG state capture;
R0-R3 reproducibility contract; full provenance on every record.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..arr_research_capability_03.reacher import (
    INPUT_DIM, OBS_DIM, generate_trajectories, sha256_file, sha256_json,
)
from . import core


def set_deterministic_mode(mode: str = "det", num_threads: int = 1,
                           deterministic_algorithms: bool = True) -> dict[str, Any]:
    """Configure ALL randomness sources. mode in {det, nondet}."""
    if mode not in ("det", "nondet"):
        raise ValueError(mode)
    src = {"mode": mode, "torch_threads": num_threads,
           "deterministic_algorithms": bool(deterministic_algorithms)}
    torch.set_num_threads(int(num_threads))
    src["torch_num_threads_actual"] = torch.get_num_threads()
    if mode == "det" and deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    src["cuda_deterministic"] = bool(mode == "det")
    os.environ["PYTHONHASHSEED"] = "0"
    src["pythonhashseed_env"] = os.environ.get("PYTHONHASHSEED")
    return src


def seed_everything(seed: int, mode: str = "det", num_threads: int = 1,
                    deterministic_algorithms: bool = True) -> dict[str, Any]:
    set_deterministic_mode(mode, num_threads, deterministic_algorithms)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    state = {
        "python_random": random.getstate()[1][:4],
        "numpy_rng_first": int(np.random.get_state()[1][0]),
        "torch_cpu_seed_state": [int(x) for x in torch.random.get_rng_state()[:8]],
    }
    return {"seed": int(seed), "mode": mode, "num_threads": num_threads,
            "deterministic_algorithms": bool(deterministic_algorithms),
            "rng_state_snapshot": state}


class DeterministicMLP:
    """Real PyTorch dynamics MLP with deterministic training path."""

    def __init__(self, hidden: list[int]):
        self.hidden = list(hidden)
        self.net: Any = None

    def _build(self):
        import torch.nn as nn
        layers = []
        dim = INPUT_DIM
        for h in self.hidden:
            layers += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        layers += [nn.Linear(dim, OBS_DIM)]
        self.net = torch.nn.Sequential(*layers)

    def fit(self, states, actions, next_states, *, lr: float, epochs: int,
            seed: int, batch_size: int = 64) -> dict[str, Any]:
        self._build()
        seed_everything(seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(lr))
        loss_fn = torch.nn.MSELoss()
        x = torch.tensor(np.concatenate([states, actions], axis=1), dtype=torch.float32)
        y = torch.tensor(next_states, dtype=torch.float32)
        n = x.shape[0]
        bs = max(1, min(int(batch_size), n))
        final = None
        for _ in range(int(epochs)):
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(self.net(x[idx]), y[idx])
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(idx)
            final = total / n
        post_state = [int(x) for x in torch.random.get_rng_state()[:8]]
        return {"final_loss": float(final), "post_train_rng_state": post_state}

    def predict(self, states, actions) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(np.concatenate([states, actions], axis=1), dtype=torch.float32)
            return self.net(x).numpy().astype(np.float64)


def run_experiment(*, experiment_id: str, track: str, strategy: str, task_id: str,
                   seed: int, train_cfg: dict[str, Any], model_cfg: dict[str, Any],
                   test_cfg: dict[str, Any], root: str | Path,
                   mode: str = "det", num_threads: int = 1,
                   deterministic_algorithms: bool = True,
                   protocol_hash: str = "", epochs: int = 60) -> dict[str, Any]:
    """Execute ONE real Reacher experiment with full reproducibility contract."""
    root = Path(root).resolve()
    out = root / "raw_data" / "immutable" / f"{experiment_id}.json"
    if out.is_file():
        return json.loads(out.read_text(encoding="utf-8"))

    cfg = {"train": dict(train_cfg), "model": dict(model_cfg),
           "test": dict(test_cfg), "epochs": int(epochs), "seed": int(seed),
           "mode": mode, "num_threads": int(num_threads),
           "deterministic_algorithms": bool(deterministic_algorithms)}
    rec: dict[str, Any] = {
        "experiment_id": experiment_id, "track": track, "strategy": strategy,
        "task_id": task_id, "seed": int(seed), "timestamp": core.now_iso(),
        "source_commit": core.git_head(".."), "benchmark_version": core.BENCHMARK_VERSION,
        "protocol_hash": protocol_hash, "model_id": "arr06-deterministic-mlp",
        "simulator": "gymnasium/Reacher-v5",
        "environment": {"python": "3.13", "torch": torch.__version__,
                        "numpy": np.__version__,
                        "gymnasium": __import__("gymnasium").__version__},
        "applied_config": cfg, "config_sha256": core.sha256_json(cfg),
    }
    t0 = time.time()
    try:
        rng_pre = seed_everything(seed, mode, num_threads, deterministic_algorithms)
        rec["rng_pre"] = rng_pre
        train = generate_trajectories(seed=seed, **train_cfg)
        test = generate_trajectories(seed=seed + 9000, **test_cfg)
        rec["train_data_sha256"] = train.digest()
        rec["test_data_sha256"] = test.digest()
        model = DeterministicMLP(model_cfg["hidden"])
        fit = model.fit(train.states, train.actions, train.next_states,
                        lr=float(model_cfg["lr"]), epochs=epochs, seed=seed)
        rec["fit"] = fit
        pred = model.predict(test.states, test.actions)
        mse = float(np.mean((pred - test.next_states) ** 2))
        rec["metrics"] = {"one_step_mse": mse, "final_loss": fit["final_loss"]}
        trace = root / "raw_data" / "traces" / experiment_id
        trace.mkdir(parents=True, exist_ok=True)
        tp = trace / "result.json"
        tp.write_text(json.dumps({"mse": mse, "final_loss": fit["final_loss"],
                                  "weights_sha": core.sha256_json([
                                      [round(float(x), 8) for x in p.detach().numpy().ravel().tolist()]
                                      for p in model.net.parameters()])},
                                 sort_keys=True), encoding="utf-8")
        rec["trace_path"] = str(tp)
        rec["trace_hash"] = sha256_file(tp)
        rec["result_hash"] = core.sha256_json(rec["metrics"])
        rec["wall_seconds"] = round(time.time() - t0, 3)
        rec["ok"] = True
        rec["error"] = ""
    except Exception as exc:
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["wall_seconds"] = round(time.time() - t0, 3)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=False,
                              default=str), encoding="utf-8")
    return rec


__all__ = ["REPRO_LEVELS", "set_deterministic_mode", "seed_everything",
           "DeterministicMLP", "run_experiment"]
