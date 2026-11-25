"""
symbolic.py
===========
SciNet++ (2026) -- Stage 3: AI-Feynman-style formula discovery with PySR.

Takes the frozen latent concepts saved by ``analyze.py`` and runs symbolic
regression (PySR) to search for a closed-form expression mapping the learned
latent vector z to each ground-truth physical concept. This is the modern
"AI Feynman" step: instead of hoping a human interprets the latent axes, we let
a symbolic engine propose an explicit equation.

PySR is a Julia-backed package; on first use it installs its Julia toolchain
automatically (this can take a few minutes). If PySR is unavailable the script
degrades gracefully with installation guidance instead of crashing.

Inputs (produced by analyze.py, in cfg['output_dir']):
    latent.npy, labels.npy, label_names.json

Outputs:
    equations_<concept>.txt   human-readable PySR result table
    equations_<concept>.csv   full candidate front (model.equations_)

Usage:
    python symbolic.py --config configs/pendulum.yaml
"""

from __future__ import annotations

# ----------------------------------------------------------
# JuliaCall environment (必须在导入 PySR 前设置)
# ----------------------------------------------------------
import os

os.environ.setdefault(
    "PYTHON_JULIACALL_EXE",
    r"D:\ProgramData\anaconda3\julia_env\pyjuliapkg\install\bin\julia.exe",
)
os.environ.setdefault(
    "PYTHON_JULIACALL_PROJECT",
    r"D:\ProgramData\anaconda3\julia_env",
)
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
# os.environ.setdefault("PYTHON_JULIACALL_THREADS", "12")
os.environ.setdefault(
    "PYTHON_JULIACALL_THREADS",
    "auto",
)

import argparse
import json
import logging
import sys
import time

import numpy as np

from utils.config import (
    ensure_output_dir,
    load_config,
    make_run_id,
    setup_logging,
)

try:
    from pysr import PySRRegressor

    PYSR_AVAILABLE = True
    PYSR_ERROR = None
except Exception as exc:
    PYSR_AVAILABLE = False
    PYSR_ERROR = exc


# ==========================================================
# Utilities
# ==========================================================


def parse_args():
    parser = argparse.ArgumentParser(description="SciNet++ symbolic regression")
    parser.add_argument("--config", default="configs/pendulum.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--new", action="store_true")
    return parser.parse_args()


def load_concepts(out_dir):

    latent_path = os.path.join(out_dir, "latent.npy")
    labels_path = os.path.join(out_dir, "labels.npy")

    if not (os.path.exists(latent_path) and os.path.exists(labels_path)):
        raise FileNotFoundError(
            "Missing latent.npy or labels.npy. Run analyze.py first."
        )

    Z = np.load(latent_path)
    Y = np.load(labels_path)

    names_path = os.path.join(out_dir, "label_names.json")

    if os.path.exists(names_path):
        with open(names_path, "r", encoding="utf-8") as f:
            names = json.load(f)
    else:
        names = [f"concept_{i}" for i in range(Y.shape[1])]

    return Z, Y, names


# ==========================================================
# PySR
# ==========================================================


def discover_formula(Z, y, cfg, out_dir, concept_name):

    s = cfg["symbolic"]

    equation_file = os.path.join(
        out_dir,
        f"hall_of_fame_{concept_name}.csv",
    )

    # NOTE: on Windows, parallelism="multithreading" swallows per-iteration
    # logs (terminal stuck at `[ Info: Started!`) and thread-scheduling
    # overhead actually slows down small problems.  Serial mode prints
    # `iter X/N` progress and converges faster for data of this scale.
    # early stop: config `symbolic.early_stop` > 0 时,best loss 低于该值即停止;
    # 0/缺省 -> None,即关闭(PySR 的正确参数名是 early_stop_condition)
    _esc = s.get("early_stop", 0.0)

    model = PySRRegressor(
        niterations=s.get("niterations", 40),
        binary_operators=s.get("binary_operators", ["+", "-", "*"]),
        unary_operators=s.get("unary_operators", ["exp", "cos"]),
        maxsize=s.get("max_complexity", 20),
        batch_size=s.get("batch_size", 50),

        random_state=cfg.get("seed", 42),
        parallelism=s.get("parallelism", "serial"),
        deterministic=True,

        # stop once the best loss drops below this threshold (None = disabled)
        early_stop_condition=(_esc if _esc and _esc > 0 else None),

        precision=64,

        verbosity=1,
        progress=True,

        output_directory=out_dir,
        run_id=f"symbolic_{concept_name}",

        temp_equation_file=False,
        delete_tempfiles=False,
    )

    logging.info("Start searching formula for %s", concept_name)

    t0 = time.time()

    print("Calling model.fit()...")

    model.fit(Z, y)

    elapsed = time.time() - t0

    print(f"Finished {concept_name} in {elapsed:.1f}s")

    logging.info(
        "Finished %s in %.1f seconds",
        concept_name,
        elapsed,
    )

    return model


# ==========================================================
# Main
# ==========================================================


def main():

    args = parse_args()

    cfg = load_config(args.config)

    if args.new:
        cfg["run_id"] = make_run_id()
    elif args.run_id:
        cfg["run_id"] = args.run_id

    out_dir = ensure_output_dir(cfg)

    setup_logging(out_dir, name="symbolic")

    print("=" * 70)
    print("Julia backend")
    print("=" * 70)
    print("Exe     :", os.environ["PYTHON_JULIACALL_EXE"])
    print("Project :", os.environ["PYTHON_JULIACALL_PROJECT"])
    print("Threads :", os.environ["PYTHON_JULIACALL_THREADS"])
    print("=" * 70)

    if not PYSR_AVAILABLE:

        print("=" * 70)
        print("PySR unavailable")
        print(PYSR_ERROR)
        print("=" * 70)
        sys.exit(1)

    Z, Y, names = load_concepts(out_dir)

    print(f"Loaded latents {Z.shape}")
    print(f"Loaded concepts {names}")

    for j, name in enumerate(names):

        print("\n" + "=" * 70)
        print(f"Symbolic regression: latent -> {name}")
        print("=" * 70)

        model = discover_formula(
            Z,
            Y[:, j],
            cfg,
            out_dir,
            name,
        )

        best = model.get_best()
        print("\nBest equation:")
        print(best)

        pred = model.predict(Z)
        from sklearn.metrics import r2_score
        r2 = r2_score(Y[:, j], pred)
        print(f"R2 = {r2:.6f}")

        txt_path = os.path.join(
            out_dir,
            f"equations_{name}.txt",
        )

        csv_path = os.path.join(
            out_dir,
            f"equations_{name}.csv",
        )

        with open(txt_path, "w", encoding="utf-8") as f:

            f.write(f"# Concept: {name}\n")
            f.write(f"# Latent dims: {Z.shape[1]}\n\n")
            f.write(str(model))

        try:
            model.equations_.to_csv(csv_path, index=False)
        except Exception:
            pass

        print(f"Saved {txt_path}")
        print(f"Saved {csv_path}")

    logging.info("Symbolic regression completed.")


if __name__ == "__main__":
    main()