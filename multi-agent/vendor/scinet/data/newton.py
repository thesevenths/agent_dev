"""
data.newton
===========
SciNet++ (2026) -- Newton's second law experiment.

Simulates a point mass ``m`` pushed by a constant net force ``F`` on a
frictionless track. Integrating Newton's second law,

    F = m * a   ->   a = F / m,

gives the familiar uniformly accelerated motion

    x(t) = x0 + v0 * t + 0.5 * (F / m) * t**2.

The observed signal has TWO channels:

    channel 0 : position  x(t)
    channel 1 : applied force F(t)   (constant over the window)

and the two ground-truth physical *concepts* are the force ``F`` and the mass
``m``. This is the Newton's-law analogue of the pendulum experiment: the pendulum
hides ``gamma``/``omega`` in a damped oscillation; here we hide ``F``/``m`` in a
parabolic trajectory. Because the only quantity that shapes the *curvature* of
channel 0 is ``a = F / m`` (while channel 1 exposes ``F`` directly), the latent
space must encode both ``F`` and ``a``; the physics probe then recovers
``(F, m)`` and the consistency ``F = m * a`` is exactly Newton's second law.
PySR closes the AI-Feynman loop by writing that relationship as an explicit
formula recovered purely from the trajectory.

The generator keeps the identical interface as ``data.pendulum.PendulumGenerator``
(``state_dim``, ``label_names``, ``sample_one``, ``sample``) so the encoder /
predictor / probe stack runs unchanged -- only the config and this module differ.
"""

from __future__ import annotations

import numpy as np


class NewtonGenerator:
    """Sample constant-force trajectories plus their (force, mass) labels."""

    #: dimensionality of the observed state at each timestep (position + force)
    state_dim = 2
    #: names of the ground-truth physical concepts returned as labels
    label_names = ("force", "mass")

    def __init__(
        self,
        length: int = 200,
        t_max: float = 10.0,
        force_range: tuple[float, float] = (0.5, 5.0),
        mass_range: tuple[float, float] = (0.5, 3.0),
        x0_range: tuple[float, float] = (-1.0, 1.0),
        v0_range: tuple[float, float] = (-2.0, 2.0),
        rng=None,
    ) -> None:
        self.length = length
        self.t = np.linspace(0.0, t_max, length, dtype=np.float32)
        self.force_range = force_range
        self.mass_range = mass_range
        self.x0_range = x0_range
        self.v0_range = v0_range
        # 可注入的独立 RNG（见 data.pendulum 的同名参数说明）。
        self.rng = rng if rng is not None else np.random

    def sample_one(self):
        """Generate a single trajectory.

        Returns
        -------
        sig : np.ndarray, shape (length, 2), float32
            column 0 = position x(t), column 1 = applied force F (constant)
        force : float
        mass : float
        """
        F = self.rng.uniform(*self.force_range)
        m = self.rng.uniform(*self.mass_range)
        x0 = self.rng.uniform(*self.x0_range)
        v0 = self.rng.uniform(*self.v0_range)
        t = self.t

        x = x0 + v0 * t + 0.5 * (F / m) * t ** 2          # Newton's 2nd law, integrated
        force_ch = np.full_like(t, F)                     # constant external force

        sig = np.stack([x, force_ch], axis=1)             # (length, 2)
        return sig.astype(np.float32), float(F), float(m)

    def sample(self, n: int):
        """Generate ``n`` trajectories.

        Returns
        -------
        X : np.ndarray, shape (n, length, 2), float32  -- columns (x, F)
            No normalisation is applied. The physical parameter ranges in the
            config are chosen so that position stays O(1-10) and force is O(1),
            keeping both channels on comparable scales for the encoder while
            preserving the absolute amplitude information needed to recover
            mass via m = F / a. Per-trajectory or batch-level standardisation
            would destroy this amplitude signal and make mass unidentifiable.
        labels : np.ndarray, shape (n, 2), float32       -- columns (force, mass)
        """
        from tqdm import tqdm

        trajectories = []
        labels = []
        for _ in tqdm(range(n), desc="Generating newton trajectories"):
            sig, F, m = self.sample_one()
            trajectories.append(sig)
            labels.append((F, m))

        X = np.stack(trajectories).astype(np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        return X, labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gen = NewtonGenerator(length=200)
    X, labels = gen.sample(5)
    print("X shape:", X.shape, "labels shape:", labels.shape)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for i in range(X.shape[0]):
        axs[0].plot(X[i, :, 0], label=f"F={labels[i, 0]:.2f} m={labels[i, 1]:.2f}")
        axs[1].plot(X[i, :, 1])
    axs[0].set_ylabel("position x(t)")
    axs[1].set_ylabel("force F(t)")
    axs[1].set_xlabel("timestep")
    axs[0].legend(fontsize=7)
    axs[0].set_title("Newton's 2nd law trajectories (x0 + v0 t + 0.5 (F/m) t^2)")
    plt.tight_layout()
    plt.show()
