"""
data.controlled_pendulum
========================
Torque-controlled pendulum simulator for P2-03A (Action-conditioned JEPA) -- P1-A.

This is a *new* standalone module. It does NOT modify the existing
``data.pendulum`` (damped-oscillator) generator, the ``DynamicsBatch`` pipeline,
or any model code. It exists purely to provide a clean, deterministic,
action-conditioned ODE environment for the controlled-pendulum pilot.

Physics
-------
State ``x = [theta, omega]`` with **unwrapped** theta (never reduced mod 2*pi, so
the angle can wind past +/-pi without a spurious jump -- important for the
unwrapped-theta audit check). Equation of motion:

    theta_dot = omega
    omega_dot = -(g / L) * sin(theta) - b * omega + torque / (m * L^2)

Integrated with classical **RK4** at fixed step ``dt`` (default 0.02 s). All
arithmetic is done in **float64** for numerical fidelity; the pilot dataset on
disk is stored as float64 so the physics can be re-validated to high precision.

Action conditioning
-------------------
The single control input is the joint torque ``tau`` (U = 1). Four deterministic
action profiles are provided:

    * ``random``     -- uniform torque in [-tau_max, tau_max] (seeded)
    * ``step``       -- piecewise-constant +/- step, switching at a fraction
    * ``sinusoid``   -- A * sin(2*pi*f*t + phi)
    * ``alternating``-- square wave +tau / -tau every step

Determinism contract
--------------------
``generate_torque_sequence`` is a pure function of (profile spec, T, dt,
torque_max, seed). Given the same inputs it returns byte-identical torque arrays,
so any downstream stage can *replay* an episode exactly from the recorded
(scenario, profile, seed) without re-running the collector. This is the property
the audit script exercises.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


#: Known action-profile types.
ACTION_PROFILE_TYPES = ("random", "step", "sinusoid", "alternating")

#: Observaton dimension: [theta_unwrapped, omega].
STATE_DIM = 2
#: Action dimension: [torque].
ACTION_DIM = 1


@dataclass
class ScenarioSpec:
    """Physical + initial-condition specification of one pendulum instance."""

    name: str
    g: float = 9.81
    L: float = 1.0
    m: float = 1.0
    b: float = 0.1
    theta0: float = 2.0
    omega0: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "g": self.g,
            "L": self.L,
            "m": self.m,
            "b": self.b,
            "theta0": self.theta0,
            "omega0": self.omega0,
        }


@dataclass
class ProfileSpec:
    """Specification of one action profile (type + typed parameters)."""

    name: str
    type: str
    torque_max: float = 5.0
    # step
    fraction: float = 0.5
    magnitude: float = 0.6
    # sinusoid
    amplitude: float = 0.7
    frequency_hz: float = 0.5
    phase: float = 0.0
    # alternating
    # (reuses `magnitude`)

    def as_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "type": self.type,
            "torque_max": self.torque_max,
            "fraction": self.fraction,
            "magnitude": self.magnitude,
            "amplitude": self.amplitude,
            "frequency_hz": self.frequency_hz,
            "phase": self.phase,
        }


def episode_id_for(scenario_idx: int, profile_idx: int,
                   scenario_name: str, profile_name: str) -> str:
    """Stable, human-readable episode id, e.g. ``S0_random``."""
    return f"S{scenario_idx}_{profile_name}"


def derive_episode_seed(master_seed: int, scenario_idx: int,
                        profile_idx: int) -> int:
    """Deterministic per-episode seed (independent of generation order).

    Uses a SHA-256-based derivation so the seed is always in [0, 2**32 - 1]
    regardless of how large ``master_seed`` is (a naive ``master*1000 + ...``
    can overflow ``numpy.random.RandomState``'s 32-bit range).
    """
    payload = f"ms={int(master_seed)}|s={int(scenario_idx)}|p={int(profile_idx)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


class ControlledPendulum:
    """Torque-controlled pendulum integrated with RK4 in float64."""

    state_dim = STATE_DIM
    action_dim = ACTION_DIM

    def __init__(self, g: float = 9.81, L: float = 1.0, m: float = 1.0,
                 b: float = 0.1, dt: float = 0.02) -> None:
        self.g = float(g)
        self.L = float(L)
        self.m = float(m)
        self.b = float(b)
        self.dt = float(dt)
        # Moment of inertia of a point mass on a massless rod about the pivot.
        self.inertia = self.m * self.L * self.L

    # -- continuous dynamics ------------------------------------------------- #
    def derivatives(self, state: np.ndarray, torque: float) -> np.ndarray:
        """Return [theta_dot, omega_dot] for state ``[theta, omega]``."""
        theta = float(state[0])
        omega = float(state[1])
        dtheta = omega
        domega = (
            -(self.g / self.L) * math.sin(theta)
            - self.b * omega
            + float(torque) / self.inertia
        )
        return np.array([dtheta, domega], dtype=np.float64)

    def rk4_step(self, state: np.ndarray, torque: float) -> np.ndarray:
        """One RK4 step of length ``dt`` under constant ``torque``."""
        s = np.asarray(state, dtype=np.float64).reshape(2)
        h = self.dt
        k1 = self.derivatives(s, torque)
        k2 = self.derivatives(s + 0.5 * h * k1, torque)
        k3 = self.derivatives(s + 0.5 * h * k2, torque)
        k4 = self.derivatives(s + h * k3, torque)
        return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def simulate(self, torques: Sequence[float], theta0: float,
                 omega0: float) -> np.ndarray:
        """Roll out ``len(torques)`` transitions from ``(theta0, omega0)``.

        Returns
        -------
        observations : np.ndarray, shape (T + 1, 2), float64
            ``observations[k]`` is the state before step ``k``; ``observations[k+1]``
            is the state after applying ``torques[k]``. ``T = len(torques)``.
        """
        torques = np.asarray(torques, dtype=np.float64).reshape(-1)
        T = int(torques.shape[0])
        obs = np.empty((T + 1, 2), dtype=np.float64)
        s = np.array([float(theta0), float(omega0)], dtype=np.float64)
        obs[0] = s
        for k in range(T):
            s = self.rk4_step(s, float(torques[k]))
            obs[k + 1] = s
        return obs

    # -- independent reference integrator (for audit) ------------------------- #
    def simulate_substep(self, torques: Sequence[float], theta0: float,
                         omega0: float, substeps: int) -> np.ndarray:
        """RK4 roll-out with ``substeps`` internal steps per transition.

        Used by the audit to build a high-accuracy reference (RK4 converges as
        dt^4, so shrinking the internal step by ``substeps`` shrinks the
        per-step error by ~substeps^4). The returned array is still aligned to
        the original transition grid (one row per transition).
        """
        torques = np.asarray(torques, dtype=np.float64).reshape(-1)
        T = int(torques.shape[0])
        h = self.dt / float(substeps)
        s = np.array([float(theta0), float(omega0)], dtype=np.float64)
        out = np.empty((T + 1, 2), dtype=np.float64)
        out[0] = s
        for k in range(T):
            st = s
            for _ in range(substeps):
                st = self._rk4_step_with_h(st, float(torques[k]), h)
            s = st
            out[k + 1] = s
        return out

    def _rk4_step_with_h(self, state: np.ndarray, torque: float,
                         h: float) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64).reshape(2)
        k1 = self.derivatives(s, torque)
        k2 = self.derivatives(s + 0.5 * h * k1, torque)
        k3 = self.derivatives(s + 0.5 * h * k2, torque)
        k4 = self.derivatives(s + h * k3, torque)
        return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def generate_torque_sequence(profile: ProfileSpec, T: int, dt: float,
                             seed: int) -> np.ndarray:
    """Deterministic torque sequence of length ``T`` for one action profile.

    Pure function of ``(profile, T, dt, seed)`` -- replays byte-identically.
    """
    if profile.type not in ACTION_PROFILE_TYPES:
        raise ValueError(
            f"unknown action profile type {profile.type!r}; "
            f"expected one of {ACTION_PROFILE_TYPES}"
        )
    T = int(T)
    dt = float(dt)
    tau_max = float(profile.torque_max)

    if profile.type == "random":
        rng = np.random.RandomState(int(seed))
        return rng.uniform(-tau_max, tau_max, size=T).astype(np.float64)

    if profile.type == "step":
        mag = tau_max * float(profile.magnitude)
        frac = float(profile.fraction)
        switch = int(round(T * frac))
        seq = np.empty(T, dtype=np.float64)
        seq[:switch] = mag
        seq[switch:] = -mag
        return seq

    if profile.type == "sinusoid":
        amp = tau_max * float(profile.amplitude)
        f = float(profile.frequency_hz)
        phi = float(profile.phase)
        t = np.arange(T, dtype=np.float64) * dt
        return (amp * np.sin(2.0 * np.pi * f * t + phi)).astype(np.float64)

    # alternating
    mag = tau_max * float(profile.magnitude)
    seq = np.array(
        [mag if k % 2 == 0 else -mag for k in range(T)], dtype=np.float64
    )
    return seq


@dataclass
class Episode:
    """One collected controlled-pendulum episode."""

    episode_id: str
    scenario_idx: int
    profile_idx: int
    seed: int
    observations: np.ndarray          # (T + 1, 2) float64
    conditions: np.ndarray            # (T, 1) float64  (torque)
    scenario: ScenarioSpec
    profile: ProfileSpec

    @property
    def T(self) -> int:
        return int(self.observations.shape[0] - 1)


def generate_episode(scenario: ScenarioSpec, profile: ProfileSpec,
                    seed: int, T: int, dt: float) -> Episode:
    """Generate one full episode (observations + conditions) deterministically."""
    torques = generate_torque_sequence(profile, T, dt, seed)
    sim = ControlledPendulum(
        g=scenario.g, L=scenario.L, m=scenario.m, b=scenario.b, dt=dt
    )
    obs = sim.simulate(torques, scenario.theta0, scenario.omega0)
    cond = torques.reshape(-1, 1).astype(np.float64)
    return Episode(
        episode_id=episode_id_for(-1, -1, scenario.name, profile.name),  # placeholder
        scenario_idx=-1,
        profile_idx=-1,
        seed=int(seed),
        observations=obs,
        conditions=cond,
        scenario=scenario,
        profile=profile,
    )


def plan_pilot(scenarios: Sequence[ScenarioSpec],
               profiles: Sequence[ProfileSpec],
               master_seed: int) -> List[Dict]:
    """Enumerate the (scenario, profile, seed) plan for the pilot.

    Returns a list of dicts with keys:
        scenario_idx, profile_idx, episode_id, seed,
        scenario (ScenarioSpec), profile (ProfileSpec)
    """
    plan: List[Dict] = []
    for s_idx, sc in enumerate(scenarios):
        for p_idx, pr in enumerate(profiles):
            seed = derive_episode_seed(master_seed, s_idx, p_idx)
            epid = episode_id_for(s_idx, p_idx, sc.name, pr.name)
            plan.append({
                "scenario_idx": s_idx,
                "profile_idx": p_idx,
                "episode_id": epid,
                "seed": seed,
                "scenario": sc,
                "profile": pr,
            })
    return plan


def config_to_specs(cfg: Dict) -> tuple[List[ScenarioSpec], List[ProfileSpec], int, int, float]:
    """Extract (scenarios, profiles, T, dt, master_seed) from a config dict."""
    sim = cfg["simulator"]
    T = int(sim["horizon_steps"])
    dt = float(sim["dt"])
    master_seed = int(cfg["experiment"]["seed"])

    scenarios: List[ScenarioSpec] = []
    for sc in cfg["scenarios"]:
        scenarios.append(ScenarioSpec(
            name=str(sc["name"]),
            g=float(sc.get("g", 9.81)),
            L=float(sc.get("L", 1.0)),
            m=float(sc.get("m", 1.0)),
            b=float(sc.get("b", 0.1)),
            theta0=float(sc.get("theta0", 2.0)),
            omega0=float(sc.get("omega0", 0.0)),
        ))

    default_tau = float(sim.get("action_max_torque", 5.0))
    profiles: List[ProfileSpec] = []
    for pr in cfg["action_profiles"]:
        profiles.append(ProfileSpec(
            name=str(pr["name"]),
            type=str(pr["type"]),
            torque_max=float(pr.get("torque_max", default_tau)),
            fraction=float(pr.get("fraction", 0.5)),
            magnitude=float(pr.get("magnitude", 0.6)),
            amplitude=float(pr.get("amplitude", 0.7)),
            frequency_hz=float(pr.get("frequency_hz", 0.5)),
            phase=float(pr.get("phase", 0.0)),
        ))
    return scenarios, profiles, T, dt, master_seed


if __name__ == "__main__":
    # Quick self-check: simulate a zero-torque pendulum and print the trajectory.
    sim = ControlledPendulum(g=9.81, L=1.0, m=1.0, b=0.0, dt=0.02)
    zeros = np.zeros(50, dtype=np.float64)
    obs = sim.simulate(zeros, theta0=2.0, omega0=0.0)
    print("theta[0:5]:", np.round(obs[:5, 0], 4))
    print("omega[0:5]:", np.round(obs[:5, 1], 4))
