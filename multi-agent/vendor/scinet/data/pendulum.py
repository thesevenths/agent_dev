"""
data.pendulum
=============
SciNet++ (2026) -- primary 1D dataset.

Generates trajectories of a damped harmonic oscillator,

    x(t) = A * exp(-gamma * t) * cos(omega * t + phi),

which is the modern stand-in for the "motion of a (damped) pendulum" used by
the original SciNet paper (Iten et al., 2018). The two ground-truth physical
*concepts* -- the damping rate ``gamma`` and the angular frequency ``omega`` --
are what the encoder is expected to rediscover in its latent space and what the
physics probe / symbolic regression stages try to read back out.

This module is the entry point of the SciNet thread of the platform:
    SciNet (concepts) -> AI Feynman (formulas) -> V-JEPA (latent prediction)
    -> AdaJEPA (online adaptation).
"""

from __future__ import annotations

import numpy as np


class PendulumGenerator:
    """Sample damped-oscillator trajectories plus their (gamma, omega) labels."""

    #: dimensionality of the observed state at each timestep (scalar signal)
    state_dim = 1
    #: names of the ground-truth physical concepts returned as labels
    label_names = ("gamma", "omega")

    def __init__(self, length: int = 200, t_max: float = 10.0, rng=None) -> None:
        self.length = length
        self.t = np.linspace(0.0, t_max, length, dtype=np.float32)
        # 可注入的独立 RNG。默认沿用 numpy 全局 RNG，保持直接调用
        # PendulumGenerator().sample() 的既有行为；build_dataset 会显式注入
        # per-split RandomState，使同一 split 在不同进程里逐位可复现。
        self.rng = rng if rng is not None else np.random

    def sample_one(self):
        """Generate a single trajectory.

        Returns
        -------
        x : np.ndarray, shape (length, 1), float32
        gamma : float
        omega : float
        """
        amp = self.rng.uniform(0.5, 1.5)
        gamma = self.rng.uniform(0.01, 0.3)
        omega = self.rng.uniform(0.5, 2.0)
        phi = self.rng.uniform(0.0, 2.0 * np.pi)

        x = amp * np.exp(-gamma * self.t) * np.cos(omega * self.t + phi)
        return x.astype(np.float32)[:, None], float(gamma), float(omega)

    def sample(self, n: int):
        """Generate ``n`` trajectories.

        Returns
        -------
        X : np.ndarray, shape (n, length, state_dim), float32
        labels : np.ndarray, shape (n, 2), float32  -- columns (gamma, omega)
        """
        from tqdm import tqdm

        trajectories = []
        labels = []
        for _ in tqdm(range(n), desc="Generating pendulum trajectories"):
            x, gamma, omega = self.sample_one()
            trajectories.append(x)
            labels.append((gamma, omega))

        X = np.stack(trajectories).astype(np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        return X, labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gen = PendulumGenerator(length=200)
    X, labels = gen.sample(5)
    print("X shape:", X.shape, "labels shape:", labels.shape)

    plt.figure(figsize=(8, 4))
    for i in range(X.shape[0]):
        plt.plot(X[i, :, 0], label=f"g={labels[i, 0]:.2f} w={labels[i, 1]:.2f}")
    plt.xlabel("timestep")
    plt.ylabel("x(t)")
    plt.legend(fontsize=7)
    plt.title("Damped pendulum trajectories")
    plt.tight_layout()
    plt.show()
