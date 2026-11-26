"""
data.controlled_pendulum_v2
===========================
P1-A' / P1-B1: hardened simulator + spec-compliant data generation.

Design decision (review fix P4): the ORIGINAL ``data/controlled_pendulum.py``
and its 20-episode pilot are FROZEN historical evidence and are NOT modified.
This module REUSES the v1 numerical core by inheritance (so the RK4 arithmetic
is bit-identical) and adds, purely on top:

  * strict physical-parameter validation (m>0, L>0, dt>0, g>0, b>=0, finite)
  * per-step finite-state guards (assert finite state at EVERY RK4 step)
  * torque bound validation and action-profile specification validation
  * spec-compliant sampling of initial states and physical parameters
  * configurable action_hold_steps and genuine zero-torque episodes
  * preregistered regime classification

Nothing here depends on ``robotics`` or any project-specific constant.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .controlled_pendulum import (
    ACTION_PROFILE_TYPES,
    ControlledPendulum as _V1Pendulum,
    ProfileSpec,
    ScenarioSpec,
    derive_episode_seed,
)

V2_PROFILE_TYPES = ("random", "step", "sinusoid", "alternating", "zero")


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _finite_real(x: Any, name: str) -> float:
    if isinstance(x, bool):
        raise TypeError(f"{name} must be a finite real number, got bool")
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real number: {exc}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


def validate_physical_params(g, L, m, b, dt) -> Dict[str, float]:
    """Strict, framework-independent physical parameter validation."""
    vals = {
        "g": _finite_real(g, "g"),
        "L": _finite_real(L, "L"),
        "m": _finite_real(m, "m"),
        "b": _finite_real(b, "b"),
        "dt": _finite_real(dt, "dt"),
    }
    for k, min_val, strict, in (
        ("g", 0.0, True), ("L", 0.0, True), ("m", 0.0, True),
        ("dt", 0.0, True), ("b", 0.0, False),
    ):
        v = vals[k]
        if (v <= min_val if strict else v < min_val):
            raise ValueError(
                f"{k} must be {'>' if strict else '>='} {min_val}, got {v}"
            )
    return vals


def validate_profile(profile: ProfileSpec, torque_bounds: Sequence[float]) -> None:
    """Validate an action-profile specification against the actuator bounds."""
    if profile.type not in V2_PROFILE_TYPES:
        raise ValueError(
            f"unknown action profile type {profile.type!r}; "
            f"expected one of {V2_PROFILE_TYPES}")
    lo, hi = float(torque_bounds[0]), float(torque_bounds[1])
    if not (lo < 0 < hi) or not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"torque_bounds must bracket 0 and be finite, got {torque_bounds}")
    for key in ("torque_max", "fraction", "magnitude", "amplitude",
                "frequency_hz", "phase"):
        _finite_real(getattr(profile, key), f"profile.{key}")
    if not (0.0 <= profile.fraction <= 1.0):
        raise ValueError(f"profile.fraction must be in [0,1], got {profile.fraction}")
    for key in ("magnitude", "amplitude"):
        v = float(getattr(profile, key))
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"profile.{key} must be in [0,1], got {v}")
    if profile.frequency_hz < 0:
        raise ValueError("profile.frequency_hz must be >= 0")


class ValidatedPendulum(_V1Pendulum):
    """v1 numerical core + strict validation + per-step finite guards."""

    def __init__(self, g=9.81, L=1.0, m=1.0, b=0.1, dt=0.02,
                 torque_bounds: Sequence[float] = (-1.0, 1.0)) -> None:
        v = validate_physical_params(g, L, m, b, dt)
        self.torque_bounds = (float(torque_bounds[0]), float(torque_bounds[1]))
        validate_profile(ProfileSpec(name="_bounds", type="random",
                                     torque_max=max(abs(self.torque_bounds[0]),
                                                    abs(self.torque_bounds[1]))),
                         self.torque_bounds)
        # Delegate to the v1 constructor -> identical inertia / arithmetic.
        super().__init__(g=v["g"], L=v["L"], m=v["m"], b=v["b"], dt=v["dt"])

    def _check_state(self, s: np.ndarray, where: str) -> None:
        if not np.all(np.isfinite(s)):
            raise FloatingPointError(
                f"non-finite state at {where}: {s!r}. Physical parameters or "
                f"dt are likely unstable; refusing to emit corrupt data.")

    def _check_torque(self, torque: float) -> float:
        t = _finite_real(torque, "torque")
        lo, hi = self.torque_bounds
        if not (lo - 1e-9 <= t <= hi + 1e-9):
            raise ValueError(
                f"torque {t} outside actuator bounds [{lo}, {hi}]")
        return t

    def rk4_step(self, state: np.ndarray, torque: float) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64).reshape(2)
        self._check_state(s, "rk4_step input")
        t = self._check_torque(torque)
        out = super().rk4_step(s, t)
        self._check_state(out, "rk4_step output")
        return out

    def derivatives(self, state: np.ndarray, torque: float) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64).reshape(2)
        self._check_state(s, "derivatives input")
        out = super().derivatives(s, self._check_torque(torque))
        self._check_state(out, "derivatives output")
        return out

    def simulate(self, torques: Sequence[float], theta0: float,
                 omega0: float) -> np.ndarray:
        torques = np.asarray(torques, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(torques)):
            raise ValueError("torque sequence contains NaN/Inf")
        for t in torques:
            self._check_torque(t)
        s0 = np.array([_finite_real(theta0, "theta0"),
                       _finite_real(omega0, "omega0")], dtype=np.float64)
        self._check_state(s0, "initial state")
        obs = super().simulate(torques, s0[0], s0[1])
        self._check_state(obs[-1], "final state")
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError("non-finite value appeared in rollout")
        return obs


# --------------------------------------------------------------------------- #
# Spec-compliant episode specification sampling
# --------------------------------------------------------------------------- #
@dataclass
class V2EpisodeSpec:
    episode_id: str
    seed: int
    scenario: ScenarioSpec
    profile: ProfileSpec
    hold_steps: int
    regime: Optional[str] = None
    split: Optional[str] = None


def sample_episode_specs(cfg: dict) -> List[V2EpisodeSpec]:
    """Sample 20 deterministic episode specs from the declared distributions."""
    sim = cfg["simulator"]
    T = int(sim["episode_steps"])
    dt = float(sim["dt"])
    master_seed = int(cfg["experiment"]["seed"])
    dist = cfg["distributions"]
    pr_list = cfg["action_profiles"]
    hold_options = [int(h) for h in cfg["action_hold_steps_options"]]
    n = int(cfg["pilot"]["episodes"])

    rng = np.random.RandomState(master_seed % (2 ** 31))
    specs: List[V2EpisodeSpec] = []
    for i in range(n):
        s_idx, p_idx = divmod(i, len(pr_list))
        seed = derive_episode_seed(master_seed, s_idx, p_idx)
        r = np.random.RandomState(seed)
        theta0 = float(r.uniform(*dist["theta0_range"]))
        omega0 = float(r.uniform(*dist["omega0_range"]))
        g = float(dist["gravity_g"])
        L = float(r.uniform(*dist["length_range"]))
        m = float(r.uniform(*dist["mass_range"]))
        b = float(r.uniform(*dist["damping_rate_range"]))
        sc = ScenarioSpec(name=f"sample_{i:02d}", g=g, L=L, m=m, b=b,
                          theta0=theta0, omega0=omega0)
        pr_raw = pr_list[p_idx]
        pr = ProfileSpec(
            name=str(pr_raw["name"]), type=str(pr_raw["type"]),
            torque_max=float(pr_raw.get("torque_max",
                                        max(abs(x) for x in sim["torque_bounds"]))),
            fraction=float(pr_raw.get("fraction", 0.5)),
            magnitude=float(pr_raw.get("magnitude", 0.6)),
            amplitude=float(pr_raw.get("amplitude", 0.7)),
            frequency_hz=float(pr_raw.get("frequency_hz", 0.5)),
            phase=float(pr_raw.get("phase", 0.0)),
        )
        validate_profile(pr, sim["torque_bounds"])
        hold = hold_options[i % len(hold_options)]
        specs.append(V2EpisodeSpec(
            episode_id=f"E{i:02d}_{pr.name}",
            seed=seed, scenario=sc, profile=pr, hold_steps=hold))
    return specs


def generate_torque_sequence_v2(profile: ProfileSpec, T: int, dt: float,
                                seed: int, hold_steps: int = 1,
                                torque_bounds: Sequence[float] = (-1.0, 1.0)
                                ) -> np.ndarray:
    """Deterministic torque sequence with configurable hold duration.

    ``hold_steps`` keeps each sampled torque constant for that many consecutive
    transitions, implementing the spec's ``action_hold_steps`` requirement.
    ``zero`` produces a genuine all-zero torque episode.
    """
    validate_profile(profile, torque_bounds)
    T = int(T)
    lo, hi = float(torque_bounds[0]), float(torque_bounds[1])
    span = max(abs(lo), abs(hi))
    rng = np.random.RandomState(int(seed))

    if profile.type == "zero":
        return np.zeros(T, dtype=np.float64)

    # number of independent draws
    n_holds = math.ceil(T / int(hold_steps))
    if profile.type == "random":
        base = rng.uniform(lo, hi, size=n_holds)
    elif profile.type == "step":
        mag = span * float(profile.magnitude)
        switch = int(round(n_holds * float(profile.fraction)))
        base = np.empty(n_holds, dtype=np.float64)
        base[:switch] = mag
        base[switch:] = -mag
    elif profile.type == "sinusoid":
        amp = span * float(profile.amplitude)
        t = np.arange(n_holds, dtype=np.float64) * dt * int(hold_steps)
        base = amp * np.sin(2.0 * np.pi * float(profile.frequency_hz) * t
                            + float(profile.phase))
    else:  # alternating
        mag = span * float(profile.magnitude)
        base = np.array([mag if k % 2 == 0 else -mag for k in range(n_holds)],
                        dtype=np.float64)
    seq = np.repeat(base, int(hold_steps))[:T].astype(np.float64)
    if np.any(np.abs(seq) > span + 1e-9):
        raise ValueError("generated torque exceeded actuator bounds")
    return seq


# Default regime criteria. These MUST stay byte-for-byte consistent with
# configs/dynamics/controlled_pendulum_v2_pilot.yaml :: regime_criteria. They are
# the fallback only; the audit and the P1-B2 collector MUST pass the YAML-declared
# criteria so there is a single source of truth (P1-B1.1 / P8).
DEFAULT_REGIME_CRITERIA = {
    "continuous_rotation": {"min_abs_winding": 1.0, "max_omega_sign_changes": 2},
    "bounded_oscillation": {
        "max_theta_range_rad": 2.0 * math.pi,
        "min_omega_sign_changes": 2,
    },
    "transitional": "otherwise",
}


def _validate_regime_criteria(criteria: Dict[str, Any]) -> None:
    """Raise ValueError if ``criteria`` is not a complete, well-formed spec.

    Missing keys -> 'missing criterion' failure. Non-numeric / non-finite
    thresholds -> 'invalid criterion' failure.
    """
    if not isinstance(criteria, dict):
        raise ValueError("regime_criteria must be a mapping")
    for needed in ("continuous_rotation", "bounded_oscillation", "transitional"):
        if needed not in criteria:
            raise ValueError(f"regime_criteria missing required key: {needed!r}")
    cr = criteria["continuous_rotation"]
    bo = criteria["bounded_oscillation"]
    for key in ("min_abs_winding", "max_omega_sign_changes"):
        if key not in cr:
            raise ValueError(f"continuous_rotation missing {key!r}")
        if not isinstance(cr[key], (int, float)) or not math.isfinite(cr[key]):
            raise ValueError(f"continuous_rotation.{key} must be a finite number")
    for key in ("max_theta_range_rad", "min_omega_sign_changes"):
        if key not in bo:
            raise ValueError(f"bounded_oscillation missing {key!r}")
        if not isinstance(bo[key], (int, float)) or not math.isfinite(bo[key]):
            raise ValueError(f"bounded_oscillation.{key} must be a finite number")


def classify_regime(obs: np.ndarray, p: ScenarioSpec,
                    criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Preregistered regime classification (bounded / transitional / rotation).

    Thresholds are taken from ``criteria`` (the YAML-declared
    ``regime_criteria`` block). When ``criteria`` is None the module-level
    ``DEFAULT_REGIME_CRITERIA`` is used so existing callers keep working, but the
    audit and the collector MUST pass the config so the threshold is never
    duplicated/hardcoded at the call site (P1-B1.1 / P8).
    """
    if criteria is None:
        criteria = DEFAULT_REGIME_CRITERIA
    _validate_regime_criteria(criteria)

    cr = criteria["continuous_rotation"]
    bo = criteria["bounded_oscillation"]

    I = p.m * p.L * p.L
    E_barrier = 2.0 * p.m * p.g * p.L
    th, om = obs[:, 0], obs[:, 1]
    E = 0.5 * I * om ** 2 + p.m * p.g * p.L * (1.0 - np.cos(th))
    winding = float((th[-1] - th[0]) / (2.0 * np.pi))
    sgn = np.sign(om)
    nz = sgn[sgn != 0]
    sign_changes = int(np.sum(np.diff(nz) != 0)) if nz.size > 1 else 0
    frac_above = float(np.mean(E > E_barrier))
    theta_range = float(th.max() - th.min())

    if (abs(winding) >= cr["min_abs_winding"]
            and sign_changes <= cr["max_omega_sign_changes"]):
        regime = "continuous_rotation"
    elif (theta_range <= bo["max_theta_range_rad"]
          and sign_changes >= bo["min_omega_sign_changes"]):
        regime = "bounded_oscillation"
    else:
        regime = "transitional"
    return {
        "regime": regime,
        "winding_number": winding,
        "theta_range_rad": theta_range,
        "omega_sign_changes": sign_changes,
        "max_abs_omega_rad_s": float(np.max(np.abs(om))),
        "E_barrier_J": E_barrier,
        "fraction_timesteps_above_barrier": frac_above,
        "max_energy_J": float(E.max()),
        "final_theta_rad": float(th[-1]),
    }
