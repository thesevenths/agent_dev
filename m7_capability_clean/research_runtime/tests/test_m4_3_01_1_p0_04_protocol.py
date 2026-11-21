"""M4.3-01.1 P0-04 -- complete frozen Confirmatory Protocol.

Spec: delete EACH of the 20 required fields in turn and confirm the
confirmatory start is refused (fail-closed). Also: unified Python/YAML schema,
new protocol version on change, old version immutable, started experiments keep
the original protocol SHA, no post-hoc primary metric / sample size /
exclusion-rule changes.
"""
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest
import yaml

from research_runtime.m4_3 import (
    CONFIRMATORY_REQUIRED_FIELDS, EMPTY_VALUES, POSTHOC_LOCKED_FIELDS,
    PRIMARY_METRICS, FrozenProtocol, ProtocolImmutable, ProtocolNotFrozen,
    ProtocolRegistry, assert_confirmatory_ready,
    missing_confirmatory_fields,
)
from research_runtime.m4_3.metrics import frozen_metric_contract
from research_runtime.m4_3.protocol_schema import field_is_missing

ROOT = Path(__file__).resolve().parents[2]


def _complete() -> FrozenProtocol:
    """A FULLY frozen confirmatory protocol (all 20 fields)."""
    return FrozenProtocol(
        confirmatory=True, snapshot_ids=("snap_1", "snap_2", "snap_3"),
        sample_size=60, seed=20261115, model_version="gpt-x-2026-11",
        prompt_version="discovery-prompt-v3", ranker_version="ranker-v1",
        domain_policy_version="scinet-policy-v2",
        analysis_plan_version="analysis-plan-v1",
        campaign_budget={"max_jobs": 30, "max_total_seconds": 3600},
        paired_seeds=(42, 43, 44), primary_metrics=PRIMARY_METRICS,
        metric_contract_sha256=frozen_metric_contract()["contract_sha256"],
        exclusion_rules=("no-frozen-split", "no-posthoc-metric-change"),
        statistical_method="blocked-paired-permutation",
        multiple_comparison_correction="holm-bonferroni",
        stopping_rule="all pre-registered blocks complete",
        smallest_effect_of_interest=0.25, alpha=0.05, target_power=0.8)


# ------------------------------------------------------- schema unification
def test_python_and_yaml_schemas_are_identical():
    cfg = yaml.safe_load(
        (ROOT / "M4_3_runs" / "protocol.yaml").read_text(encoding="utf-8"))
    assert tuple(cfg["confirmatory_required_fields"]) == \
        CONFIRMATORY_REQUIRED_FIELDS
    assert tuple(cfg["posthoc_locked_fields"]) == POSTHOC_LOCKED_FIELDS
    # YAML pre_registration has EXACTLY the same field set as the Python schema
    assert set(cfg["pre_registration"]) == set(CONFIRMATORY_REQUIRED_FIELDS)
    # Python FrozenProtocol exposes every required field
    py_fields = {f.name for f in fields(FrozenProtocol)}
    assert set(CONFIRMATORY_REQUIRED_FIELDS) <= py_fields
    assert len(CONFIRMATORY_REQUIRED_FIELDS) == 20


def test_empty_values_cover_every_required_field():
    assert set(EMPTY_VALUES) == set(CONFIRMATORY_REQUIRED_FIELDS)
    for name, value in EMPTY_VALUES.items():
        assert field_is_missing(name, value) is True, name


def test_complete_protocol_passes_confirmatory_gate():
    p = _complete()
    assert missing_confirmatory_fields(p) == ()
    p.assert_confirmatory_ready()          # no exception
    assert_confirmatory_ready(p)
    assert len(p.freeze_sha256()) == 64


# ------------------------------------------------------- per-field fail-closed
@pytest.mark.parametrize("field", CONFIRMATORY_REQUIRED_FIELDS)
def test_removing_any_single_required_field_refuses_confirmatory(field):
    """Spec: delete EACH field -> Confirmatory start is fail-closed."""
    broken = replace(_complete(), **{field: EMPTY_VALUES[field]})
    assert field in missing_confirmatory_fields(broken)
    with pytest.raises(ProtocolNotFrozen) as exc:
        broken.assert_confirmatory_ready()
    assert field in str(exc.value)
    with pytest.raises(ValueError):       # legacy ValueError contract kept
        assert_confirmatory_ready(broken)


@pytest.mark.parametrize("field", CONFIRMATORY_REQUIRED_FIELDS)
def test_protocol_registry_refuses_unfrozen_versions(field):
    reg = ProtocolRegistry()
    broken = replace(_complete(), **{field: EMPTY_VALUES[field]})
    with pytest.raises(ProtocolNotFrozen):
        reg.freeze(broken)


def test_multiple_missing_fields_are_all_reported():
    broken = replace(_complete(), alpha=0.0, target_power=0.0,
                     statistical_method="")
    missing = missing_confirmatory_fields(broken)
    assert {"alpha", "target_power", "statistical_method"} <= set(missing)
    with pytest.raises(ProtocolNotFrozen) as exc:
        assert_confirmatory_ready(broken)
    assert "3 of 20" in str(exc.value)


def test_invalid_numeric_ranges_are_fail_closed():
    for changes in ({"alpha": 1.5}, {"target_power": -0.1},
                    {"smallest_effect_of_interest": 0.0},
                    {"alpha": 0.0}, {"sample_size": -5}):
        with pytest.raises(ProtocolNotFrozen):
            replace(_complete(), **changes).assert_confirmatory_ready()


def test_metric_contract_sha_must_match_the_frozen_contract():
    wrong = replace(_complete(), metric_contract_sha256="0" * 64)
    assert "metric_contract_sha256" in missing_confirmatory_fields(wrong)
    with pytest.raises(ProtocolNotFrozen):
        wrong.assert_confirmatory_ready()


def test_seed_zero_is_valid_but_none_is_missing():
    assert replace(_complete(), seed=0).missing_confirmatory_fields() == ()
    assert "seed" in replace(_complete(), seed=None).missing_confirmatory_fields()


# ------------------------------------------------------- versioning / immutability
def test_any_change_creates_a_new_protocol_version():
    reg = ProtocolRegistry()
    v1 = reg.freeze(_complete())
    v2p, v2 = reg.derive(_complete(), model_version="gpt-x-2026-12")
    assert v1.version == 1 and v2.version == 2
    assert v1.protocol_sha256 != v2.protocol_sha256
    assert v2p.freeze_sha256() == v2.protocol_sha256
    # OLD version is untouched / immutable.
    assert reg.get(v1.protocol_sha256) == v1
    assert reg.versions() == (v1, v2)
    assert reg.freeze(_complete()) == v1   # idempotent, not overwritten


def test_no_op_change_is_refused():
    reg = ProtocolRegistry()
    reg.freeze(_complete())
    with pytest.raises(ProtocolImmutable):
        reg.derive(_complete(), model_version="gpt-x-2026-11")


def test_started_experiment_keeps_original_protocol_sha():
    reg = ProtocolRegistry()
    p = _complete()
    sha = reg.bind_experiment("exp_confirm_1", p)
    assert sha == p.freeze_sha256() == reg.experiment_sha("exp_confirm_1")
    # A new version exists, but the started experiment still references OLD.
    _, v2 = reg.derive(p, model_version="gpt-x-2027-01")
    assert reg.experiment_sha("exp_confirm_1") == sha != v2.protocol_sha256
    reg.assert_experiment_unchanged("exp_confirm_1", p)
    with pytest.raises(ProtocolImmutable):
        reg.assert_experiment_unchanged("exp_confirm_1", replace(
            p, model_version="gpt-x-2027-01"))


@pytest.mark.parametrize("locked", POSTHOC_LOCKED_FIELDS)
def test_posthoc_change_of_locked_fields_is_refused(locked):
    reg = ProtocolRegistry()
    p = _complete()
    reg.bind_experiment("exp_1", p)
    change = {"primary_metrics": ("different_metric",),
              "sample_size": 999,
              "exclusion_rules": ("changed",)}[locked]
    with pytest.raises(ProtocolImmutable) as exc:
        reg.derive(p, experiment_id="exp_1", **{locked: change})
    assert "post-hoc" in str(exc.value)


def test_non_locked_change_is_allowed_as_a_new_version_even_after_binding():
    reg = ProtocolRegistry()
    p = _complete()
    reg.bind_experiment("exp_1", p)
    new, v = reg.derive(p, experiment_id="exp_1",
                        model_version="gpt-x-2027-01")
    assert v.version == 2
    # The bound experiment still uses the ORIGINAL SHA.
    assert reg.experiment_sha("exp_1") == p.freeze_sha256()
    assert new.freeze_sha256() != p.freeze_sha256()


def test_experiment_cannot_be_rebound():
    reg = ProtocolRegistry()
    reg.bind_experiment("exp_1", _complete())
    with pytest.raises(ProtocolImmutable):
        reg.bind_experiment("exp_1", _complete())


# ------------------------------------------------------- stage gates
def test_pilot_and_confirmatory_remain_not_started():
    cfg = yaml.safe_load(
        (ROOT / "M4_3_runs" / "protocol.yaml").read_text(encoding="utf-8"))
    assert cfg["pre_registration"]["confirmatory"] is False
    assert cfg["pre_registration"]["sample_size"] == 0
    assert cfg["stage_gates"]["M4_3_02_pilot"].startswith("NOT_STARTED")
    assert cfg["stage_gates"]["M4_3_03_confirmatory"] == "NOT_STARTED"
    assert cfg["stage_gates"]["M4_3_04_value_report"] == "NOT_STARTED"
    # A fully frozen protocol is only a CAPABILITY here; no run is started.
    assert _complete().assert_confirmatory_ready() is None
