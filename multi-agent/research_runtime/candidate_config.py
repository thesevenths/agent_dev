"""P1-03: the SAFE boundary between a research decision and a config file.

An Agent (or an LLM) may propose a hypothesis, but it may NOT freely rewrite a
YAML file. This builder turns an approved, whitelisted ``config_overrides`` map
into ONE isolated, hash-traceable candidate config.

Design rules
------------
* Engine-agnostic. ``allowed_keys``, ``forbidden_roots``, ``value_rules`` and
  ``seed_key`` are INJECTED POLICY. This module contains no model, metric or
  config field belonging to any particular engine -- a finance backtest, a
  robotics simulator and a world model all use this same builder.
* Safe by default. A key must be whitelisted AND already exist in the baseline
  config. Editing a path the engine never reads is not a controlled experiment.
* Data identity is verified, not asserted. ``data.*`` cannot be overridden,
  the whole data section must stay byte-identical in meaning, and the frozen
  dataset manifest's version + per-split SHA-256 digests must match on disk.
  A shared ``dataset_version`` string in JSON is NOT sufficient evidence.
* Every candidate gets its own directory and its own training seed. Existing
  directories are never overwritten.
* The safety boundary is not a scientific-correctness guarantee: the caller
  still has to know that the engine actually consumes the overridden field.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from .schemas import CONFIG_KEY_RE

# Generic research-infrastructure concepts, NOT an engine's field names:
# data source, output location, execution device, experiment identity and the
# RNG root must never be silently changed by a candidate experiment.
DEFAULT_FORBIDDEN_ROOTS = frozenset({
    "data", "dataset", "datasets", "output", "outputs", "output_dir",
    "device", "experiment", "seed", "run_id", "checkpoint",
})

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
ValueRule = Callable[[Any], "bool | str"]


class CandidateConfigError(ValueError):
    """A proposed config change violates the experiment safety boundary."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CandidateArtifact:
    """What the campaign runner (P1-04) needs to submit one candidate job."""
    directory: Path
    config_path: Path
    config_relpath: str          # relative to project_dir, for the engine CLI
    metadata_path: Path
    baseline_config_sha256: str
    candidate_config_sha256: str
    metadata: dict[str, Any]


def _check_id(value: Any, label: str) -> None:
    if (not isinstance(value, str) or not _SAFE_ID_RE.match(value)
            or ".." in value or "/" in value or "\\" in value):
        raise CandidateConfigError(
            f"{label} must be a safe single-segment identifier, got {value!r}"
        )


def _type_label(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "sequence"
    if isinstance(value, dict):
        return "mapping"
    if value is None:
        return "null"
    return type(value).__name__


def _validate_value(key: str, old: Any, new: Any) -> None:
    """A controlled intervention changes a VALUE, not the config's shape."""
    if _type_label(old) != _type_label(new):
        raise CandidateConfigError(
            f"{key}: type change {_type_label(old)} -> {_type_label(new)} "
            "is not a controlled intervention"
        )
    if isinstance(new, (int, float)) and not isinstance(new, bool):
        if not math.isfinite(float(new)):
            raise CandidateConfigError(f"{key}: value must be finite")
    if isinstance(old, (list, tuple)) and old and new:
        element_type = _type_label(old[0])
        if any(_type_label(item) != element_type for item in new):
            raise CandidateConfigError(
                f"{key}: sequence element types must match the baseline"
            )


def _get_dotted(config: Mapping[str, Any], dotted_key: str) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise CandidateConfigError(
                f"{dotted_key!r} does not exist in the baseline config"
            )
        node = node[part]
    return node


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    old = _get_dotted(config, dotted_key)   # existence check happens here
    _validate_value(dotted_key, old, value)
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = copy.deepcopy(value)


class CandidateConfigBuilder:
    def __init__(self, project_dir: str | Path, *,
                 allowed_keys: Iterable[str],
                 forbidden_roots: Iterable[str] | None = None,
                 value_rules: Mapping[str, ValueRule] | None = None,
                 seed_key: str = "seed",
                 campaign_root: str = "outputs/research_campaigns",
                 require_frozen_manifest: bool = True):
        self.project_dir = Path(project_dir).resolve()
        if not self.project_dir.is_dir():
            raise CandidateConfigError(
                f"project_dir does not exist: {self.project_dir}")
        self.allowed_keys = frozenset(allowed_keys)
        if not self.allowed_keys:
            raise CandidateConfigError(
                "allowed_keys must be injected explicitly; there is no "
                "engine-specific default whitelist in this module"
            )
        for key in self.allowed_keys:
            if not isinstance(key, str) or not CONFIG_KEY_RE.match(key):
                raise CandidateConfigError(f"invalid whitelist key: {key!r}")
        self.forbidden_roots = frozenset(
            forbidden_roots if forbidden_roots is not None
            else DEFAULT_FORBIDDEN_ROOTS
        )
        self.value_rules = dict(value_rules or {})
        self.seed_key = seed_key
        self.campaign_root = Path(campaign_root)
        self.require_frozen_manifest = require_frozen_manifest

    # ------------------------------------------------------------- checks
    def _check_override_keys(self, config: Mapping[str, Any],
                             overrides: Mapping[str, Any]) -> None:
        if not overrides:
            raise CandidateConfigError("config_overrides must not be empty")
        for key, value in overrides.items():
            if not isinstance(key, str) or not CONFIG_KEY_RE.match(key):
                raise CandidateConfigError(
                    f"invalid dotted config path: {key!r}")
            root = key.split(".")[0]
            if root in self.forbidden_roots:
                raise CandidateConfigError(
                    f"{key!r} is under a protected root ({root!r}); "
                    "candidates may not change data, outputs, device, "
                    "experiment identity or the RNG root"
                )
            if key not in self.allowed_keys:
                raise CandidateConfigError(
                    f"{key!r} is not in the injected whitelist")
            old = _get_dotted(config, key)   # rejects paths the engine never reads
            _validate_value(key, old, value)
            rule = self.value_rules.get(key)
            if rule is not None:
                verdict = rule(value)
                if verdict is False:
                    raise CandidateConfigError(
                        f"{key}={value!r} violates its injected value rule")
                if isinstance(verdict, str) and verdict:
                    raise CandidateConfigError(
                        f"{key}={value!r} violates its injected value rule: {verdict}")

    def _verify_frozen_dataset(self, config: Mapping[str, Any],
                               dataset_version: str) -> dict[str, Any]:
        """Verify data IDENTITY on disk, not just a label in a JSON file."""
        data = config.get("data")
        if not isinstance(data, dict):
            raise CandidateConfigError("baseline config has no 'data' section")
        frozen_dir = data.get("frozen_dir")
        if not frozen_dir or not isinstance(frozen_dir, str):
            if self.require_frozen_manifest:
                raise CandidateConfigError(
                    "data.frozen_dir is required: candidates must reuse the "
                    "SAME frozen dataset as the baseline")
            return {"verified": False, "reason": "no frozen_dir"}
        if Path(frozen_dir).is_absolute():
            raise CandidateConfigError(
                "data.frozen_dir must be relative to the project directory")
        root = (self.project_dir / frozen_dir).resolve()
        try:
            root.relative_to(self.project_dir)
        except ValueError as exc:
            raise CandidateConfigError(
                "data.frozen_dir escapes the project directory") from exc
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise CandidateConfigError(
                f"frozen dataset manifest not found: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise CandidateConfigError(f"unreadable frozen manifest: {exc}") from exc
        if manifest.get("dataset_version") != dataset_version:
            raise CandidateConfigError(
                f"dataset_version mismatch: manifest="
                f"{manifest.get('dataset_version')!r}, requested={dataset_version!r}"
            )
        splits = manifest.get("splits") or {}
        if not splits:
            raise CandidateConfigError("frozen manifest lists no splits")
        for name, info in splits.items():
            filename = (info or {}).get("file")
            if not filename:
                raise CandidateConfigError(f"split {name!r} has no file in manifest")
            split_path = root / filename
            if not split_path.is_file():
                raise CandidateConfigError(
                    f"frozen split file missing: {split_path}")
            if sha256_file(split_path) != (info or {}).get("sha256"):
                raise CandidateConfigError(
                    f"frozen split {name!r} failed its SHA-256 check")
        return {
            "verified": True,
            "frozen_dir": frozen_dir,
            "manifest_path": manifest_path.relative_to(self.project_dir).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "splits_verified": sorted(str(name) for name in splits),
            "dataset_seed": data.get("dataset_seed"),
        }

    # -------------------------------------------------------------- build
    def build(self, *, base_config: str | Path, campaign_id: str,
              candidate_id: str, hypothesis_id: str, seed: int,
              overrides: Mapping[str, Any],
              dataset_version: str) -> CandidateArtifact:
        _check_id(campaign_id, "campaign_id")
        _check_id(candidate_id, "candidate_id")
        _check_id(hypothesis_id, "hypothesis_id")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise CandidateConfigError(f"seed must be a non-negative int, got {seed!r}")
        if not dataset_version or not isinstance(dataset_version, str):
            raise CandidateConfigError("dataset_version is required")

        config_path = Path(base_config)
        if not config_path.is_absolute():
            config_path = self.project_dir / config_path
        config_path = config_path.resolve()
        if not config_path.is_file():
            raise CandidateConfigError(f"baseline config not found: {config_path}")
        try:
            baseline_relpath = config_path.relative_to(self.project_dir)
        except ValueError as exc:
            raise CandidateConfigError(
                "baseline config must live inside the project directory") from exc
        baseline_bytes = config_path.read_bytes()
        try:
            baseline = yaml.safe_load(baseline_bytes.decode("utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError) as exc:
            raise CandidateConfigError(f"invalid baseline YAML: {exc}") from exc
        if not isinstance(baseline, dict):
            raise CandidateConfigError("baseline config must be a YAML mapping")

        if self.seed_key not in baseline:
            raise CandidateConfigError(
                f"baseline config has no {self.seed_key!r} field; cannot set a "
                "paired training seed without changing the config contract"
            )
        self._check_override_keys(baseline, overrides)
        frozen = self._verify_frozen_dataset(baseline, dataset_version)

        candidate = copy.deepcopy(baseline)
        for key, value in overrides.items():
            _set_dotted(candidate, key, value)
        candidate[self.seed_key] = seed

        # The entire data section must be semantically identical: a paired
        # experiment changes a model/training value, never what it was fed.
        if candidate.get("data") != baseline.get("data"):
            raise CandidateConfigError(
                "the data section changed; candidate and baseline would not "
                "be trained on the same frozen dataset"
            )
        if self.seed_key == "seed" and "dataset_seed" in (baseline.get("data") or {}):
            if candidate["data"]["dataset_seed"] != baseline["data"]["dataset_seed"]:
                raise CandidateConfigError("data.dataset_seed must not change")

        # Isolated output directory; never overwrite an existing experiment.
        campaign_base = (self.project_dir / self.campaign_root).resolve()
        directory = (campaign_base / campaign_id
                     / f"{candidate_id}_seed_{seed}").resolve()
        try:
            directory.relative_to(campaign_base)
        except ValueError as exc:
            raise CandidateConfigError(
                "candidate directory escapes the campaign root") from exc
        if directory.exists():
            raise CandidateConfigError(
                f"candidate directory already exists, refusing to overwrite: "
                f"{directory}")
        directory.mkdir(parents=True, exist_ok=False)

        config_bytes = yaml.safe_dump(
            candidate, sort_keys=False, allow_unicode=True
        ).encode("utf-8")
        out_config = directory / "config.yaml"
        out_config.write_bytes(config_bytes)

        metadata = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "candidate_id": candidate_id,
            "hypothesis_id": hypothesis_id,
            "seed": seed,
            "seed_key": self.seed_key,
            "dataset_version": dataset_version,
            "config_overrides": copy.deepcopy(dict(overrides)),
            "allowed_keys": sorted(self.allowed_keys),
            "forbidden_roots": sorted(self.forbidden_roots),
            "baseline_config_path": baseline_relpath.as_posix(),
            "baseline_config_sha256": sha256_bytes(baseline_bytes),
            "candidate_config_sha256": sha256_bytes(config_bytes),
            "data_section_unchanged": True,
            "frozen_dataset": frozen,
        }
        out_metadata = directory / "metadata.json"
        out_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return CandidateArtifact(
            directory=directory,
            config_path=out_config,
            config_relpath=out_config.relative_to(self.project_dir).as_posix(),
            metadata_path=out_metadata,
            baseline_config_sha256=metadata["baseline_config_sha256"],
            candidate_config_sha256=metadata["candidate_config_sha256"],
            metadata=metadata,
        )

    # ------------------------------------------------------- build_baseline
    def build_baseline(self, *, base_config: str | Path, campaign_id: str,
                       seed: int, dataset_version: str) -> CandidateArtifact:
        """P1-04: materialise ONE isolated baseline config for ONE paired seed.

        A paired experiment needs a baseline per seed: baseline and candidate
        must share the seed, the frozen data and every config value EXCEPT the
        intervention -- but must NOT share an output directory. The only edit
        here is the root training seed; ``data.dataset_seed`` is untouched, so
        the dataset version cannot silently change between arms.

        This is additive: ``build()`` above keeps its original behaviour.
        """
        _check_id(campaign_id, "campaign_id")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise CandidateConfigError(f"seed must be a non-negative int, got {seed!r}")
        if not dataset_version or not isinstance(dataset_version, str):
            raise CandidateConfigError("dataset_version is required")

        config_path = Path(base_config)
        if not config_path.is_absolute():
            config_path = self.project_dir / config_path
        config_path = config_path.resolve()
        if not config_path.is_file():
            raise CandidateConfigError(f"baseline config not found: {config_path}")
        try:
            baseline_relpath = config_path.relative_to(self.project_dir)
        except ValueError as exc:
            raise CandidateConfigError(
                "baseline config must live inside the project directory") from exc
        baseline_bytes = config_path.read_bytes()
        try:
            baseline = yaml.safe_load(baseline_bytes.decode("utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError) as exc:
            raise CandidateConfigError(f"invalid baseline YAML: {exc}") from exc
        if not isinstance(baseline, dict):
            raise CandidateConfigError("baseline config must be a YAML mapping")
        if self.seed_key not in baseline:
            raise CandidateConfigError(
                f"baseline config has no {self.seed_key!r} field; cannot set a "
                "paired training seed without changing the config contract"
            )
        frozen = self._verify_frozen_dataset(baseline, dataset_version)

        data_before = copy.deepcopy(baseline.get("data"))
        generated = copy.deepcopy(baseline)
        generated[self.seed_key] = seed
        if generated.get("data") != data_before:
            raise CandidateConfigError(
                "the data section changed while setting the baseline seed"
            )
        data_section = generated.get("data") or {}
        if "dataset_seed" in data_section and (
            data_section["dataset_seed"] != (data_before or {}).get("dataset_seed")
        ):
            raise CandidateConfigError("data.dataset_seed must not change")

        campaign_base = (self.project_dir / self.campaign_root).resolve()
        directory = (campaign_base / campaign_id
                     / f"baseline_seed_{seed}").resolve()
        try:
            directory.relative_to(campaign_base)
        except ValueError as exc:
            raise CandidateConfigError(
                "baseline directory escapes the campaign root") from exc
        if directory.exists():
            raise CandidateConfigError(
                f"baseline directory already exists, refusing to overwrite: "
                f"{directory}")
        directory.mkdir(parents=True, exist_ok=False)

        config_bytes = yaml.safe_dump(
            generated, sort_keys=False, allow_unicode=True
        ).encode("utf-8")
        out_config = directory / "config.yaml"
        out_config.write_bytes(config_bytes)

        metadata = {
            "schema_version": 1,
            "arm_type": "baseline",
            "campaign_id": campaign_id,
            "candidate_id": None,
            "hypothesis_id": None,
            "seed": seed,
            "seed_key": self.seed_key,
            "dataset_version": dataset_version,
            "config_overrides": {},
            "allowed_keys": sorted(self.allowed_keys),
            "forbidden_roots": sorted(self.forbidden_roots),
            "baseline_config_path": baseline_relpath.as_posix(),
            "baseline_config_sha256": sha256_bytes(baseline_bytes),
            "candidate_config_sha256": sha256_bytes(config_bytes),
            "data_section_unchanged": True,
            "frozen_dataset": frozen,
        }
        out_metadata = directory / "metadata.json"
        out_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return CandidateArtifact(
            directory=directory,
            config_path=out_config,
            config_relpath=out_config.relative_to(self.project_dir).as_posix(),
            metadata_path=out_metadata,
            baseline_config_sha256=metadata["baseline_config_sha256"],
            candidate_config_sha256=metadata["candidate_config_sha256"],
            metadata=metadata,
        )
