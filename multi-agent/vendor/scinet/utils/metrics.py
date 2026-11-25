"""
utils.metrics
=============
SciNet++ (2026) -- evaluation metrics shared across stages.

Includes regression metrics for the physics probe (MSE / RMSE / R^2), a cosine
similarity used to score multi-step latent prediction quality in the V-JEPA
stage, and a per-dimension latent std used as a *collapse diagnostic* (if the
std of every latent dimension goes to ~0 the encoder has collapsed).

All functions accept NumPy arrays (1D or 2D); callers convert torch tensors
with ``.detach().cpu().numpy()``.
"""

from __future__ import annotations

import numpy as np


def mse(a, b) -> float:
    """Mean squared error: average of (a-b)^2.

    Used as val_pred in training logs -- measures how close the online
    predictor's latent output is to the EMA target encoder's latent, averaged
    over all samples and all latent dimensions. Lower = better prediction.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def rmse(a, b) -> float:
    """Root mean squared error: sqrt(MSE).

    Same units as the original data, making it interpretable as an 'average
    error magnitude'. Used in the physics probe report alongside R^2.
    """
    return float(np.sqrt(mse(a, b)))


def r2(y, yhat):
    """Coefficient of determination (R^2): how much variance the model explains.

    Formula: R^2 = 1 - SS_res / SS_tot
      where SS_res = sum((y - yhat)^2)   -- residual sum of squares
            SS_tot = sum((y - y_mean)^2) -- total sum of squares

    Interpretation:
      R^2 = 1.0  perfect prediction
      R^2 = 0.0  no better than predicting the mean
      R^2 < 0    worse than predicting the mean (model is broken)

    For the physics probe: R^2=0.878 for omega means the probe can explain
    87.8% of omega's variance from the latent vector alone -- strong evidence
    that the encoder discovered omega as a concept.

    For 1D inputs returns a scalar; for 2D (multi-output) returns per-column.
    """
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    if y.ndim == 1:
        ss_res = ((y - yhat) ** 2).sum()          # unexplained variance
        ss_tot = ((y - y.mean()) ** 2).sum()       # total variance in ground truth
        return float(1.0 - ss_res / (ss_tot + 1e-12))  # eps avoids div-by-zero
    # Multi-output: compute per-column (one R^2 per physical concept)
    ss_res = ((y - yhat) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def cosine_similarity(a, b) -> float:
    """Mean row-wise cosine similarity between two sets of latent vectors.

    cos(theta) = (a . b) / (|a| * |b|), averaged over all samples in the batch.

    Why cosine instead of MSE?
      - MSE depends on the absolute scale of latents, which VICReg pushes
        toward std~1 but doesn't fix exactly. Cosine is scale-invariant.
      - Cosine measures *directional alignment*: did the predictor point in
        the right direction in latent space? This is more meaningful than
        raw distance because the V-JEPA objective cares about the structure
        of the latent dynamics, not the exact magnitude.
      - cos=1.0: perfect alignment; cos=0: orthogonal; cos=-1: opposite.

    Used in evaluate.py (per-horizon quality) and train.py (cos(h1) metric).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    # Batch: element-wise dot product / product of norms, then average
    num = (a * b).sum(axis=1)                          # dot product per sample
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12  # |a|*|b|
    return float(np.mean(num / den))                    # mean cosine over batch


def latent_std_per_dim(z) -> np.ndarray:
    """Per-dimension standard deviation of latents across the batch.

    THE collapse diagnostic: if any dim has std near 0, the encoder maps all
    inputs to the same value along that dimension -> information lost.
    VICReg's variance_loss pushes each dim's std toward 1.0; healthy training
    keeps latent_std ~ 0.8-1.5 for all dims.

    Returns shape (latent_dim,) array; callers typically take .mean() for logging.
    """
    z = np.asarray(z, dtype=np.float64)
    return z.std(axis=0)  # std of each latent dimension across all samples
