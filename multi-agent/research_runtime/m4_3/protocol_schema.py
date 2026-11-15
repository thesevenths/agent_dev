"""M4.3-01.1 P0-04 -- complete, frozen Confirmatory Protocol schema.

M4.3-01's ``FrozenProtocol.assert_confirmatory_ready()`` checked only SIX
fields, and the Python dataclass and ``M4_3_runs/protocol.yaml`` had DIFFERENT
field sets. This module is the SINGLE source of truth for the 20 required
confirmatory fields, their emptiness rules, and the append-only protocol
version registry.

Rules (spec P0-04):
  * Confirmatory cannot start until ALL 20 required fields are non-empty.
  * Any change produces a NEW protocol version; an old version is immutable.
  * A started experiment keeps referencing its ORIGINAL protocol SHA.
  * primary_metrics / sample_size / exclusion_rules can NEVER be changed
    post-hoc once an experiment has been bound to that version.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .metrics import frozen_metric_contract

# The COMPLETE confirmatory pre-registration schema (20 fields). This tuple is
# mirrored verbatim in M4_3_runs/protocol.yaml::confirmatory_required_fields.
CONFIRMATORY_REQUIRED_FIELDS: tuple[str, ...] = (
    "confirmatory",
    "snapshot_ids",
    "sample_size",
    "seed",
    "model_version",
    "prompt_version",
    "ranker_version",
    "domain_policy_version",
    "campaign_budget",
    "paired_seeds",
    "primary_metrics",
    "metric_contract_sha256",
    "exclusion_rules",
    "statistical_method",
    "multiple_comparison_correction",
    "stopping_rule",
    "smallest_effect_of_interest",
    "alpha",
    "target_power",
    "analysis_plan_version",
)

# Fields that may NOT be changed after an experiment has started (spec P0-04).
POSTHOC_LOCKED_FIELDS: tuple[str, ...] = (
    "primary_metrics", "sample_size", "exclusion_rules",
)

# Canonical "empty" value for each field (used by the fail-closed tests to
# delete exactly ONE field at a time).
EMPTY_VALUES: dict[str, Any] = {
    "confirmatory": False,
    "snapshot_ids": (),
    "sample_size": 0,
    "seed": None,
    "model_version": "",
    "prompt_version": "",
    "ranker_version": "",
    "domain_policy_version": "",
    "campaign_budget": {},
    "paired_seeds": (),
    "primary_metrics": (),
    "metric_contract_sha256": "",
    "exclusion_rules": (),
    "statistical_method": "",
    "multiple_comparison_correction": "",
    "stopping_rule": "",
    "smallest_effect_of_interest": 0.0,
    "alpha": 0.0,
    "target_power": 0.0,
    "analysis_plan_version": "",
}


class ProtocolNotFrozen(ValueError):
    """A required confirmatory field is missing / empty -> fail closed."""


class ProtocolImmutable(ValueError):
    """An immutable protocol version / post-hoc-locked field was changed."""


def field_is_missing(name: str, value: Any) -> bool:
    """Return True when a required field is missing, empty or invalid."""
    if name == "confirmatory":
        return value is not True
    if value is None:
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        if not value.strip():
            return True
        if name == "metric_contract_sha256":
            return value != frozen_metric_contract()["contract_sha256"]
        return False
    if isinstance(value, (tuple, list, dict, set, frozenset)):
        return len(value) == 0
    if isinstance(value, (int, float)):
        if name == "seed":
            return False          # 0 is a valid random seed; None is missing
        if name in ("alpha", "target_power"):
            return not (0 < value <= 1)
        return value <= 0         # sample_size / smallest_effect_of_interest
    return False


def missing_confirmatory_fields(protocol: Any) -> tuple[str, ...]:
    """Return ALL required fields that are missing/empty (fail-closed)."""
    missing: list[str] = []
    for name in CONFIRMATORY_REQUIRED_FIELDS:
        if not hasattr(protocol, name):
            missing.append(name)                 # schema drift
            continue
        if field_is_missing(name, getattr(protocol, name)):
            missing.append(name)
    return tuple(missing)


def assert_confirmatory_ready(protocol: Any) -> None:
    """Refuse to start Confirmatory unless EVERY required field is frozen."""
    missing = missing_confirmatory_fields(protocol)
    if missing:
        raise ProtocolNotFrozen(
            "confirmatory experiment requires a COMPLETE frozen protocol; "
            f"missing/empty field(s): {', '.join(missing)} "
            f"({len(missing)} of {len(CONFIRMATORY_REQUIRED_FIELDS)})")


@dataclass(frozen=True)
class ProtocolVersion:
    """An IMMUTABLE, addressable protocol version."""
    version: int
    protocol_sha256: str
    frozen_dict: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version,
                "protocol_sha256": self.protocol_sha256,
                "frozen_dict": dict(self.frozen_dict)}


class ProtocolRegistry:
    """Append-only version registry. Old versions are NEVER overwritten."""

    def __init__(self):
        self._by_sha: dict[str, ProtocolVersion] = {}
        self._by_version: list[ProtocolVersion] = []
        self._experiment_sha: dict[str, str] = {}

    # ------------------------------------------------------------- freeze
    def freeze(self, protocol: Any) -> ProtocolVersion:
        """Register a COMPLETE protocol. Identical content is idempotent."""
        assert_confirmatory_ready(protocol)
        sha = protocol.freeze_sha256()
        if sha in self._by_sha:
            return self._by_sha[sha]        # idempotent, never overwritten
        version = ProtocolVersion(
            version=len(self._by_version) + 1,
            protocol_sha256=sha,
            frozen_dict=protocol.frozen_dict())
        self._by_sha[sha] = version
        self._by_version.append(version)
        return version

    def versions(self) -> tuple[ProtocolVersion, ...]:
        return tuple(self._by_version)

    def get(self, sha: str) -> ProtocolVersion:
        if sha not in self._by_sha:
            raise ProtocolImmutable(f"unknown protocol version {sha!r}")
        return self._by_sha[sha]

    # ------------------------------------------------------------- derive
    def derive(self, base: Any, *, experiment_id: str | None = None,
               **changes: Any) -> tuple[Any, ProtocolVersion]:
        """Any change produces a NEW protocol; the old one stays immutable.

        Once an experiment is bound to ``base``, changing a POSTHOC_LOCKED
        field is refused outright (not merely versioned).
        """
        base_sha = base.freeze_sha256()
        if experiment_id and self._experiment_sha.get(experiment_id) == base_sha:
            locked = POSTHOC_LOCKED_FIELDS and tuple(
                f for f in POSTHOC_LOCKED_FIELDS if f in changes)
            if locked:
                raise ProtocolImmutable(
                    "post-hoc change of primary metric / sample size / "
                    f"exclusion rules is forbidden: {', '.join(locked)}")
        new = replace(base, **changes)
        if new.freeze_sha256() == base_sha:
            raise ProtocolImmutable(
                "change produced the SAME protocol SHA; a modification must "
                "create a new protocol version")
        version = self.freeze(new)
        return new, version

    # ---------------------------------------------------- experiment binding
    def bind_experiment(self, experiment_id: str, protocol: Any) -> str:
        """A started experiment keeps referencing the ORIGINAL protocol SHA."""
        if experiment_id in self._experiment_sha:
            raise ProtocolImmutable(
                f"experiment {experiment_id!r} is already bound to a protocol")
        assert_confirmatory_ready(protocol)
        self.freeze(protocol)                 # register (idempotent)
        sha = protocol.freeze_sha256()
        self._experiment_sha[experiment_id] = sha
        return sha

    def experiment_sha(self, experiment_id: str) -> str:
        if experiment_id not in self._experiment_sha:
            raise ProtocolImmutable(
                f"experiment {experiment_id!r} has no bound protocol")
        return self._experiment_sha[experiment_id]

    def assert_experiment_unchanged(self, experiment_id: str,
                                    protocol: Any) -> None:
        """Verify a started experiment still uses its ORIGINAL protocol SHA."""
        bound = self.experiment_sha(experiment_id)
        if protocol.freeze_sha256() != bound:
            raise ProtocolImmutable(
                f"experiment {experiment_id!r} must keep its original "
                "protocol SHA; post-hoc changes are forbidden")


__all__ = [
    "CONFIRMATORY_REQUIRED_FIELDS", "POSTHOC_LOCKED_FIELDS", "EMPTY_VALUES",
    "ProtocolNotFrozen", "ProtocolImmutable", "field_is_missing",
    "missing_confirmatory_fields", "assert_confirmatory_ready",
    "ProtocolVersion", "ProtocolRegistry",
]
