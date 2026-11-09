"""P1-03: the config-builder safety boundary.

The production module must stay engine-agnostic. All concrete config keys and
value rules below are TEST-SIDE POLICY, injected by the domain layer -- exactly
how a real campaign runner would use the builder.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_runtime.candidate_config import (
    CandidateConfigBuilder,
    CandidateConfigError,
    sha256_bytes,
)
from research_runtime.hypothesis_manager import build_experiment_plan
from research_runtime.schemas import Hypothesis, MetricPrediction

DATASET_VERSION = "frozen-test-v1"
SCINET_ROOT = Path(os.environ.get("SCINET_DIR", r"E:\LLM\SciNet++"))

# Injected domain policy (lives in the CALLER, never in the builder).
ALLOWED_KEYS = {
    "model.predictor_hidden", "model.latent_dim",
    "train.lr", "train.weight_decay",
    "jepa.pred_weight", "jepa.ema_momentum",
}
VALUE_RULES = {"jepa.ema_momentum": lambda value: 0 < value < 1}

BASE_YAML = """\
experiment: p1_smoke
seed: 42
device: cpu
output_dir: outputs
data:
  generator: pendulum
  dataset_seed: 20261109
  frozen_dir: datasets/frozen_v1
  train_samples: 64
  validation_samples: 16
  adapt_samples: 16
  test_samples: 16
model:
  latent_dim: 4
  predictor_hidden: [32, 32]
jepa:
  ema_momentum: 0.996
  pred_weight: 1.0
train:
  lr: 0.001
  weight_decay: 0.0001
"""


def make_project(root: Path) -> dict[str, Path]:
    """Build a minimal, engine-neutral project fixture in ``root``."""
    (root / "configs").mkdir(parents=True, exist_ok=True)
    base = root / "configs" / "base.yaml"
    base.write_text(BASE_YAML, encoding="utf-8")

    frozen = root / "datasets" / "frozen_v1"
    frozen.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name in ("train", "validation", "adapt", "test"):
        payload = f"frozen-split-{name}".encode()
        (frozen / f"{name}.npz").write_bytes(payload)
        splits[name] = {
            "file": f"{name}.npz",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "samples": 16,
        }
    manifest = frozen / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "splits": splits,
    }, indent=2), encoding="utf-8")
    return {"base": base, "frozen": frozen, "manifest": manifest}


class CandidateConfigTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        paths = make_project(self.root)
        self.base = paths["base"]
        self.frozen = paths["frozen"]
        self.manifest = paths["manifest"]

    def tearDown(self):
        self.tmp.cleanup()

    def builder(self, **kw) -> CandidateConfigBuilder:
        return CandidateConfigBuilder(
            self.root, allowed_keys=ALLOWED_KEYS,
            value_rules=VALUE_RULES, **kw)

    def build(self, **kw):
        defaults = dict(
            base_config="configs/base.yaml",
            campaign_id="campaign_001",
            candidate_id="candidate_A",
            hypothesis_id="H001",
            seed=42,
            overrides={"model.predictor_hidden": [64, 64]},
            dataset_version=DATASET_VERSION,
        )
        defaults.update(kw)
        return self.builder().build(**defaults)


class TestValidBuilds(CandidateConfigTestBase):
    def test_predictor_hidden_override(self):
        artifact = self.build()
        config = yaml.safe_load(artifact.config_path.read_bytes())
        self.assertEqual(config["model"]["predictor_hidden"], [64, 64])
        self.assertEqual(config["seed"], 42)
        self.assertTrue(artifact.metadata_path.is_file())
        self.assertIn("outputs/research_campaigns", artifact.config_relpath)
        self.assertEqual(artifact.metadata["config_overrides"],
                         {"model.predictor_hidden": [64, 64]})
        self.assertEqual(artifact.metadata["frozen_dataset"]["verified"], True)
        self.assertEqual(
            set(artifact.metadata["frozen_dataset"]["splits_verified"]),
            {"train", "validation", "adapt", "test"})

    def test_lr_override(self):
        artifact = self.build(overrides={"train.lr": 0.0005})
        config = yaml.safe_load(artifact.config_path.read_bytes())
        self.assertEqual(config["train"]["lr"], 0.0005)

    def test_sha256_consistency(self):
        artifact = self.build()
        self.assertEqual(
            sha256_bytes(artifact.config_path.read_bytes()),
            artifact.candidate_config_sha256)
        self.assertEqual(sha256_bytes(self.base.read_bytes()),
                         artifact.baseline_config_sha256)
        on_disk = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["candidate_config_sha256"],
                         artifact.candidate_config_sha256)

    def test_baseline_yaml_is_never_modified(self):
        before = self.base.read_bytes()
        self.build()
        self.assertEqual(self.base.read_bytes(), before)
        self.assertEqual(yaml.safe_load(before)["seed"], 42)

    def test_data_section_is_unchanged(self):
        artifact = self.build()
        baseline = yaml.safe_load(self.base.read_bytes())
        config = yaml.safe_load(artifact.config_path.read_bytes())
        self.assertEqual(config["data"], baseline["data"])

    def test_model_seed_does_not_change_dataset_seed(self):
        artifact = self.build(seed=43)
        config = yaml.safe_load(artifact.config_path.read_bytes())
        self.assertEqual(config["seed"], 43)
        self.assertEqual(config["data"]["dataset_seed"], 20261109)

    def test_each_candidate_has_an_isolated_directory(self):
        first = self.build(seed=42)
        second = self.build(seed=43)
        third = self.build(candidate_id="candidate_B", seed=42)
        paths = {first.directory, second.directory, third.directory}
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(path.is_dir() for path in paths))
        self.assertTrue(all((path / "config.yaml").is_file() for path in paths))

    def test_config_relpath_is_cli_friendly(self):
        artifact = self.build()
        self.assertFalse(Path(artifact.config_relpath).is_absolute())
        self.assertTrue((self.root / artifact.config_relpath).is_file())


class TestRejectedBuilds(CandidateConfigTestBase):
    def _assert_rejected(self, **kw):
        with self.assertRaises(CandidateConfigError):
            self.build(**kw)

    def test_unknown_field_is_rejected(self):
        self._assert_rejected(overrides={"optimizer.magic": True})

    def test_data_override_is_rejected(self):
        self._assert_rejected(overrides={"data.train_samples": 999})

    def test_path_traversal_is_rejected(self):
        self._assert_rejected(campaign_id="../../tmp")
        self._assert_rejected(candidate_id="../escape")
        self._assert_rejected(hypothesis_id="H/../../evil")

    def test_nonexistent_config_path_is_rejected(self):
        builder = self.builder()
        builder.allowed_keys = ALLOWED_KEYS | {"model.not_in_config"}
        with self.assertRaises(CandidateConfigError) as ctx:
            builder.build(
                base_config="configs/base.yaml", campaign_id="c1",
                candidate_id="A", hypothesis_id="H1", seed=42,
                overrides={"model.not_in_config": 1},
                dataset_version=DATASET_VERSION)
        self.assertIn("does not exist", str(ctx.exception))

    def test_invalid_ema_is_rejected(self):
        self._assert_rejected(overrides={"jepa.ema_momentum": 1.5})

    def test_illegal_seed_is_rejected(self):
        self._assert_rejected(seed=-1)
        self._assert_rejected(seed=True)
        self._assert_rejected(seed="42")

    def test_type_change_is_rejected(self):
        self._assert_rejected(overrides={"model.predictor_hidden": "wider"})

    def test_duplicate_directory_is_not_overwritten(self):
        first = self.build()
        before = first.config_path.read_bytes()
        self._assert_rejected()
        self.assertEqual(first.config_path.read_bytes(), before)

    def test_dataset_version_mismatch_is_rejected(self):
        self._assert_rejected(dataset_version="some-other-version")

    def test_tampered_frozen_split_is_rejected(self):
        (self.frozen / "train.npz").write_bytes(b"tampered")
        self._assert_rejected()

    def test_missing_frozen_manifest_is_rejected(self):
        self.manifest.unlink()
        self._assert_rejected()

    def test_policy_must_be_injected(self):
        with self.assertRaises(CandidateConfigError):
            CandidateConfigBuilder(self.root, allowed_keys=[])

    def test_empty_overrides_are_rejected(self):
        self._assert_rejected(overrides={})


class TestGenericity(unittest.TestCase):
    def test_production_module_contains_no_engine_specific_keys(self):
        """The whitelist is caller policy, not a hard-coded target project."""
        import research_runtime.candidate_config as module
        source = Path(module.__file__).read_text(encoding="utf-8")
        for specific in ("predictor_hidden", "latent_dim", "ema_momentum",
                         "pendulum", "SciNet", "jepa", "world_model"):
            with self.subTest(specific=specific):
                self.assertNotIn(specific, source)

    def test_plan_flows_straight_into_the_builder(self):
        """P1-01 -> P1-02 -> P1-03 data flow, no manual re-entry of values."""
        hypothesis = Hypothesis(
            hypothesis_id="H001", objective_id="obj1",
            source_experiment_id="exp_baseline_001",
            title="t", statement="s", mechanism="m",
            evidence_refs=["experiment:exp_baseline_001:metric:loss@4"],
            intervention={"model.predictor_hidden": [64, 64]},
            predicted_effects=[MetricPrediction(metric="loss@4",
                                               direction="decrease")],
            falsification_criteria=["Validation loss@4 does not improve."],
            confidence=0.5, status="approved",
        )
        plan = build_experiment_plan(
            plan_id="campaign_001", objective_id="obj1",
            baseline_experiment_id="exp_baseline_001",
            dataset_version=DATASET_VERSION, hypotheses=[hypothesis],
            primary_metric="loss@4", paired_seeds=[42, 43])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_project(root)
            builder = CandidateConfigBuilder(
                root, allowed_keys=ALLOWED_KEYS, value_rules=VALUE_RULES)
            candidate = plan.candidates[0]
            artifact = builder.build(
                base_config="configs/base.yaml",
                campaign_id=plan.plan_id,
                candidate_id=candidate.candidate_id,
                hypothesis_id=candidate.hypothesis_id,
                seed=candidate.seeds[0],
                overrides=candidate.config_overrides,
                dataset_version=plan.dataset_version,
            )
            self.assertEqual(artifact.metadata["hypothesis_id"], "H001")
            self.assertEqual(artifact.metadata["seed"], 42)
            self.assertEqual(artifact.metadata["config_overrides"],
                             {"model.predictor_hidden": [64, 64]})


@unittest.skipUnless(SCINET_ROOT.is_dir(), "SciNet++ checkout not available")
class TestRealSmokeConfigCompatibility(unittest.TestCase):
    """Real config + real frozen manifest, copied into a temp project root."""

    def test_all_whitelisted_keys_exist_and_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir(parents=True)
            shutil.copy2(SCINET_ROOT / "configs" / "pendulum_smoke.yaml",
                         root / "configs" / "base.yaml")
            shutil.copytree(SCINET_ROOT / "datasets" / "pendulum_smoke_v1",
                            root / "datasets" / "pendulum_smoke_v1")
            manifest = json.loads(
                (root / "datasets" / "pendulum_smoke_v1" / "manifest.json")
                .read_text(encoding="utf-8"))
            builder = CandidateConfigBuilder(
                root, allowed_keys=ALLOWED_KEYS, value_rules=VALUE_RULES)
            artifact = builder.build(
                base_config="configs/base.yaml",
                campaign_id="real_smoke", candidate_id="candidate_A",
                hypothesis_id="H001", seed=43,
                overrides={"model.predictor_hidden": [64, 64]},
                dataset_version=manifest["dataset_version"])
            config = yaml.safe_load(artifact.config_path.read_bytes())
            self.assertEqual(config["model"]["predictor_hidden"], [64, 64])
            self.assertEqual(config["seed"], 43)
            self.assertEqual(config["data"]["dataset_seed"],
                             manifest["data_spec"]["seed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
