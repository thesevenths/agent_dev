"""
utils.plotting
==============
SciNet++ (2026) -- figure generation for every stage.

All functions write PNGs into a configurable ``output_dir`` (no more hardcoded
"outputs/") and use the non-interactive Agg backend so the pipeline can run
headless on servers. Provided plots:

* ``scatter_latent``    -- one latent dimension vs a ground-truth concept.
* ``pca_plot``          -- 2D PCA of the latent space (concept geometry).
* ``prediction_plot``   -- raw-signal ground truth vs model prediction.
* ``horizon_metric_plot`` -- V-JEPA latent error / cosine vs rollout horizon.
* ``probe_parity_plot`` -- physics-probe parity (predicted vs true concept).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402


def _save(fig, output_dir: str, name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def scatter_latent(latent, target, name, output_dir="outputs"):
    """Scatter one latent dimension against a ground-truth concept."""
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(latent, target, s=8, alpha=0.6)
    plt.xlabel("Latent dimension")
    plt.ylabel(name)
    return _save(fig, output_dir, f"latent_vs_{name}")


def pca_plot(z, output_dir="outputs", name="pca"):
    """2D PCA projection of the latent space."""
    y = PCA(n_components=2).fit_transform(np.asarray(z))
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(y[:, 0], y[:, 1], s=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    return _save(fig, output_dir, name)


def prediction_plot(gt, pred, output_dir="outputs", name="prediction"):
    """Overlay a ground-truth signal and the model's prediction."""
    fig = plt.figure(figsize=(8, 4))
    plt.plot(np.asarray(gt).ravel(), label="Ground Truth")
    plt.plot(np.asarray(pred).ravel(), label="Prediction")
    plt.legend()
    return _save(fig, output_dir, name)


def horizon_metric_plot(horizons, values, ylabel, output_dir="outputs", name="horizon"):
    """Line plot of a per-horizon metric (V-JEPA multi-step evaluation)."""
    fig = plt.figure(figsize=(6, 4))
    plt.plot(list(horizons), list(values), marker="o")
    plt.xlabel("Rollout horizon (blocks)")
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.xticks(list(horizons), [str(h) for h in horizons])
    plt.grid(True, which="both", alpha=0.3)
    return _save(fig, output_dir, name)


def probe_parity_plot(true, pred, names, output_dir="outputs", name="probe_parity"):
    """Parity plots (predicted vs true) for each physical concept."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    if true.ndim == 1:
        true = true[:, None]
        pred = pred[:, None]
    n_cols = true.shape[1]

    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.2), squeeze=False)
    for j in range(n_cols):
        ax = axes[0][j]
        ax.scatter(true[:, j], pred[:, j], s=8, alpha=0.6)
        lo = float(min(true[:, j].min(), pred[:, j].min()))
        hi = float(max(true[:, j].max(), pred[:, j].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        ax.set_xlabel(f"true {names[j]}")
        ax.set_ylabel(f"predicted {names[j]}")
        ax.legend(fontsize=8)
    return _save(fig, output_dir, name)


def attribution_heatmap(
    matrix,
    label_names,
    latent_dim,
    r2_scores=None,
    pr_scores=None,
    output_dir="outputs",
    name="attribution_matrix",
):
    """Save the Latent Attribution Matrix as a CSV file and print a text table.

    Each cell [j, k] shows the effective weight from latent dim z_k to concept j,
    extracted from the trained probe's composed linear map W2 @ W1.

    Outputs:
      - ``<output_dir>/<name>.csv`` -- machine-readable CSV for downstream analysis
      - Console/log text table with R^2 and PR annotations per concept

    This replaces the previous matplotlib heatmap which suffered from a persistent
    FixedLocator bug across multiple matplotlib versions. A text table + CSV is
    more portable, diffable, and impossible to break.
    """
    import csv

    matrix = np.asarray(matrix, dtype=np.float64)
    n_concepts, n_latent = matrix.shape

    xlabels = [f"z{k}" for k in range(n_latent)]

    # ---- Save CSV --------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["concept"] + xlabels
        if r2_scores is not None:
            header.append("R2")
        if pr_scores is not None:
            header.append("PR")
        writer.writerow(header)
        for j in range(n_concepts):
            row = [label_names[j]] + [f"{matrix[j, k]:.4f}" for k in range(n_latent)]
            if r2_scores is not None:
                row.append(f"{r2_scores[j]:.4f}")
            if pr_scores is not None:
                row.append(f"{pr_scores[j]:.4f}")
            writer.writerow(row)

    # ---- Build text table for console/log --------------------------------
    col_w = 9  # width per latent column
    hdr = f"{'':>10s}" + "".join(f"{xl:>{col_w}s}" for xl in xlabels)
    lines = [hdr]
    for j in range(n_concepts):
        row_str = f"{label_names[j]:>10s}"
        row_str += "".join(f"{matrix[j, k]:>+{col_w}.3f}" for k in range(n_latent))
        if r2_scores is not None:
            row_str += f"  R2={r2_scores[j]:.3f}"
        if pr_scores is not None:
            row_str += f"  PR={pr_scores[j]:.2f}"
        lines.append(row_str)
    table_text = "\n".join(lines)
    print(table_text)

    return csv_path
