"""ARR 0.1 / R2: Domain Registry wiring tests.

These prove the generic ``ResearchService`` is assembled from a selected
``DomainBundle`` and that a non-SciNet domain can be substituted WITHOUT any
change to ``research_service.py`` / ``campaign/`` / ``comparison/``.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from research_runtime.domain_registry import (
    DomainBundle,
    DomainRegistry,
    default_registry,
    get_domain,
    register_domain,
)
from research_runtime.adapters.base import ModelAdapter
from research_runtime.research_environment import (
    build_default_research_service,
    build_scinet_domain_bundle,
)
from research_runtime.research_service import ResearchService

from tests.rt_tests.research_fakes import (
    FakeCampaignStore,
    FakeComparisonReportStore,
    default_domain,
)

# ---------------------------------------------------------------------------
# CI portability (ARR 0.1 / P0 follow-up)
#
# Building the REAL SciNet++ DomainBundle requires a real engine checkout:
#   * scinet_research_policy() reads datasets/pendulum_smoke_v1/manifest.json
#   * SciNetAdapter(strict=True) checks research_api/cli.py exists
#   * CandidateConfigBuilder checks the project dir on disk
# GitHub-hosted runners have no SciNet++ checkout. Rather than fabricate a
# "fake engine" or add hardcoded --deselect entries (which hides the gap), we
# use the repo's existing RESEARCH_SKIP_E2E convention: self-skip ONLY when the
# checkout is genuinely absent. With a real SCINET_DIR (developer machine /
# the real-engine E2E runs) every one of these tests still executes for real.
_SCINET_ROOT = Path(os.getenv("SCINET_DIR", r"E:\LLM\SciNet++"))
_SCINET_CHECKOUT = (
    (_SCINET_ROOT / "research_api" / "cli.py").is_file()
    and (_SCINET_ROOT / "datasets" / "pendulum_smoke_v1" / "manifest.json").is_file()
    and (_SCINET_ROOT / "configs" / "pendulum_smoke.yaml").is_file()
)
needs_scinet_checkout = unittest.skipUnless(
    _SCINET_CHECKOUT,
    "requires a real SciNet++ checkout (SCINET_DIR); self-skips on CI runners "
    "with no engine, so this is NOT a hidden/deselected failure",
)


class TestDomainRegistry(unittest.TestCase):
    def test_register_get_contains_names(self):
        reg = DomainRegistry()
        bundle = DomainBundle(name="demo", policy_factory=lambda: default_domain(),
                              adapter_factory=lambda: None,
                              campaign_store=None, report_store=None,
                              campaign_runner_factory=lambda: None,
                              comparator_factory=lambda: None,
                              dataset_verifier=lambda a, b: {},
                              campaign_root="demo_root")
        reg.register(bundle)
        self.assertTrue(reg.contains("demo"))
        self.assertIs(reg.get("demo"), bundle)
        self.assertIn("demo", reg.names())
        with self.assertRaises(KeyError):
            reg.get("missing")

    def test_register_requires_name(self):
        reg = DomainRegistry()
        with self.assertRaises(ValueError):
            reg.register(DomainBundle(name="", policy_factory=lambda: default_domain(),
                                      adapter_factory=lambda: None,
                                      campaign_store=None, report_store=None,
                                      campaign_runner_factory=lambda: None,
                                      comparator_factory=lambda: None,
                                      dataset_verifier=lambda a, b: {}))

    @needs_scinet_checkout
    def test_default_registry_has_scinet(self):
        # Registration is separate from the factory; the default service
        # performs the lazy registration, but the bundle is the unit here.
        bundle = build_scinet_domain_bundle()
        register_domain(bundle)
        self.assertTrue(default_registry.contains("scinet"))
        self.assertIs(get_domain("scinet"), bundle)


@needs_scinet_checkout
class TestScinetBundle(unittest.TestCase):
    def test_scinet_bundle_shape(self):
        bundle = build_scinet_domain_bundle()
        self.assertEqual(bundle.name, "scinet")
        policy = bundle.policy_factory()
        self.assertEqual(policy.domain, "pendulum_world_model")
        # campaign_root must be an absolute path whose tail is the relative
        # research_campaigns dir declared by the SciNet policy.
        self.assertTrue(os.path.isabs(bundle.campaign_root))
        tail = os.path.normpath(bundle.campaign_root).split(os.sep)[-2:]
        self.assertEqual(tail, ["outputs", "research_campaigns"])

    def test_scinet_bundle_adapter_is_real_model_adapter(self):
        # R2: the adapter is not dead metadata. The bundle must advertise a real
        # engine handle that is a genuine ModelAdapter (a callable factory).
        bundle = build_scinet_domain_bundle()
        adapter = bundle.adapter_factory()
        self.assertIsInstance(adapter, ModelAdapter)
        # The runner built by the bundle's own factory must use the SAME class.
        runner = bundle.campaign_runner_factory()
        self.assertIsInstance(runner.manager.adapter, ModelAdapter)
        self.assertIs(runner.manager.adapter.__class__, adapter.__class__)


@needs_scinet_checkout
class TestServiceFromRegistry(unittest.TestCase):
    def test_default_service_uses_scinet_bundle(self):
        svc = build_default_research_service()  # no-arg -> "scinet"
        self.assertIsInstance(svc, ResearchService)
        bundle = get_domain("scinet")
        self.assertEqual(svc.deps.campaign_root, bundle.campaign_root)
        self.assertEqual(svc.deps.domain.domain, "pendulum_world_model")

    def test_default_service_exposes_real_adapter(self):
        # R2: the assembled generic service must expose a real engine adapter,
        # not None. The whole point of the bundle is that the generic runtime
        # sees the domain's adapter without naming an engine itself.
        svc = build_default_research_service()
        self.assertIsInstance(svc.adapter, ModelAdapter)
        self.assertIs(svc.adapter, svc.deps.adapter)
        # The adapter the service exposes is the bundle's advertised handle.
        bundle = get_domain("scinet")
        self.assertIs(svc.adapter.__class__, bundle.adapter_factory().__class__)

    def test_service_from_swapped_domain(self):
        # A wholly different domain (policy + stores + runner) is assembled
        # through the IDENTICAL generic path -- no generic module edited.
        fake = DomainBundle(
            name="fake_ai4s",
            policy_factory=lambda: default_domain(domain="ai4s_factor"),
            adapter_factory=lambda: None,
            campaign_store=FakeCampaignStore(),
            report_store=FakeComparisonReportStore(),
            campaign_runner_factory=lambda: None,
            comparator_factory=lambda: None,
            dataset_verifier=lambda a, b: {"verified": True},
            campaign_root="fake_root",
        )
        register_domain(fake)
        try:
            svc = build_default_research_service("fake_ai4s")
            self.assertIsInstance(svc, ResearchService)
            self.assertEqual(svc.deps.domain.domain, "ai4s_factor")
            self.assertEqual(svc.deps.campaign_root, "fake_root")
            # The generic service did NOT inherit any SciNet engine object.
            self.assertIsNone(svc.deps.comparator_factory())
        finally:
            default_registry._bundles.pop("fake_ai4s", None)

    def test_unknown_domain_raises(self):
        with self.assertRaises(KeyError):
            build_default_research_service("does_not_exist")


@needs_scinet_checkout
class TestToolsDomainSelection(unittest.TestCase):
    def test_env_override_falls_back_when_unregistered(self):
        from research_runtime import tools
        # Unknown ARR_DOMAIN must fall back to the default SciNet bundle.
        old = os.environ.get("ARR_DOMAIN")
        tools._service.cache_clear()
        os.environ["ARR_DOMAIN"] = "nope"
        try:
            svc = tools._service()
            self.assertIsInstance(svc, ResearchService)
            self.assertEqual(svc.deps.domain.domain, "pendulum_world_model")
        finally:
            tools._service.cache_clear()
            if old is None:
                os.environ.pop("ARR_DOMAIN", None)
            else:
                os.environ["ARR_DOMAIN"] = old


if __name__ == "__main__":
    unittest.main()
