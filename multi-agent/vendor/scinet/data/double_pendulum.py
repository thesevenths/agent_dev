"""
data.double_pendulum
====================
SciNet++ (2026) -- optional multivariate / chaotic dataset.

Integrates the standard (point-mass, massless-rod) double pendulum and exposes
it through the same interface as :class:`data.pendulum.PendulumGenerator` so the
encoder / predictor / probe stack can run unchanged on a genuinely chaotic,
4-dimensional signal. This is the "V-JEPA on a richer signal" path of the
platform.

State per timestep:
    [theta1, theta2, omega1, omega2]      -> state_dim = 4

Ground-truth concept used as the probe / symbolic-regression target:
    E = total mechanical energy (conserved along each trajectory, varies
    across trajectories) -- a textbook SciNet-style "physical concept".
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


class DoublePendulumGenerator:
    """Sample chaotic double-pendulum trajectories plus conserved-energy labels."""

    #: dimensionality of the observed state at each timestep
    state_dim = 4
    #: names of the ground-truth physical concepts returned as labels
    label_names = ("energy",)

    def __init__(
        self,
        length: int = 200,
        duration: float = 10.0,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        g: float = 9.81,
        rng=None,
    ) -> None:
        self.length = length
        self.duration = duration
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.g = g
        self.t_eval = np.linspace(0.0, duration, length)
        # 可注入的独立 RNG（见 data.pendulum 的同名参数说明）。
        self.rng = rng if rng is not None else np.random

    # ------------------------------------------------------------------ ODE --
    def dynamics(self, t, y):
        """Equations of motion for y = [theta1, theta2, omega1, omega2]."""
        theta1, theta2, omega1, omega2 = y
        delta = theta1 - theta2
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g

        c = np.cos(delta)
        s = np.sin(delta)
        denom = m1 + m2 * s ** 2

        theta1_ddot = (
            m2 * g * np.sin(theta2) * c
            - m2 * s * (l1 * omega1 ** 2 * c + l2 * omega2 ** 2)
            - (m1 + m2) * g * np.sin(theta1)
        ) / (l1 * denom)

        theta2_ddot = (
            (m1 + m2)
            * (l1 * omega1 ** 2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c)
            + m2 * l2 * omega2 ** 2 * s * c
        ) / (l2 * denom)

        return [omega1, omega2, theta1_ddot, theta2_ddot]

    # --------------------------------------------------------------- energy --
    def energy(self, states: np.ndarray) -> float:
        """Total mechanical energy of a trajectory (constant up to integrator error).

        Parameters
        ----------
        states : np.ndarray, shape (T, 4)
        """
        theta1, theta2, omega1, omega2 = states.T
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g

        kinetic = (
            0.5 * m1 * l1 ** 2 * omega1 ** 2
            + 0.5 * m2 * (
                l1 ** 2 * omega1 ** 2
                + l2 ** 2 * omega2 ** 2
                + 2.0 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2)
            )
        )
        potential = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)
        return float(np.mean(kinetic + potential))

    # -------------------------------------------------------------- sampling --
    def sample_one(self):
        """Integrate one trajectory from a random initial condition."""
        y0 = [
            self.rng.uniform(-np.pi, np.pi),
            self.rng.uniform(-np.pi, np.pi),
            self.rng.uniform(-1.0, 1.0),
            self.rng.uniform(-1.0, 1.0),
        ]
        sol = solve_ivp(
            self.dynamics,
            [0.0, self.duration],
            y0,
            t_eval=self.t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        states = sol.y.T.astype(np.float32)
        return states, self.energy(states)

    def sample(self, n: int = 1000):
        """Generate ``n`` trajectories.

        Returns
        -------
        X : np.ndarray, shape (n, length, 4), float32
        labels : np.ndarray, shape (n, 1), float32  -- column (energy,)
        """
        from tqdm import tqdm

        trajectories = []
        labels = []
        for _ in tqdm(range(n), desc="Generating double-pendulum trajectories"):
            states, energy = self.sample_one()
            trajectories.append(states)
            labels.append((energy,))

        X = np.stack(trajectories).astype(np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        return X, labels

    # ------------------------------------------------------------ utilities --
    @staticmethod
    def to_cartesian(states, l1=1.0, l2=1.0):
        """Convert angles (T, 4) to Cartesian bob positions x1, y1, x2, y2."""
        theta1 = states[:, 0]
        theta2 = states[:, 1]
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)
        return x1, y1, x2, y2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gen = DoublePendulumGenerator(length=400, duration=20.0)
    traj, energy = gen.sample_one()
    x1, y1, x2, y2 = gen.to_cartesian(traj, gen.l1, gen.l2)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(x2, y2)
    plt.scatter(x2[0], y2[0], s=40, label="start")
    plt.scatter(x2[-1], y2[-1], s=40, label="end")
    plt.axis("equal")
    plt.title(f"Double pendulum trajectory (E={energy:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()
