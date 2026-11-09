"""P1-06 A6: ONE REAL end-to-end autonomous research loop on Pendulum.

    Objective -> real baseline evidence -> Failure Diagnosis
      -> Hypothesis -> deterministic Critic -> ExperimentPlan
      -> HUMAN approval (out-of-band console) -> CampaignRunner (20 real jobs)
      -> P1-05 Comparator -> Research Reviewer -> Experience -> ResearchReport

Everything downstream of the hypothesis is the REAL engine CLI and the REAL
P1-04/P1-05/P1-06 modules. Criterion for success is that the chain completes,
is traceable and refuses to promote -- NOT that the candidate beats baseline.

Why the hypothesis is operator-supplied: the existing rule-based proposer only
fires on genuine horizon degradation, and the frozen Pendulum smoke baseline
does NOT degrade with horizon (latent_mse@1..3 decreases). Inventing a
"diagnosis" would be worse science than an explicitly human-authored,
Critic-reviewed, evidence-referencing hypothesis. The loop being proved here is
approval -> execution -> statistical comparison -> report.

Skip with RESEARCH_SKIP_E2E=1.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from research_runtime.adapters import SciNetAdapter
from research_runtime.campaign.campaign_store import CampaignStore
from research_runtime.campaign.scinet import (
    build_scinet_campaign_runner,
    scinet_campaign_policy,
)
from research_runtime.candidate_config import CandidateConfigBuilder
from research_runtime.comparison.report_store import ComparisonReportStore
from research_runtime.comparison.scinet import build_scinet_comparator
from research_runtime.domains.scinet import (
    SCINET_INTERVENTION_POLICY,
    scinet_research_policy,
)
from research_runtime.experiment_manager import ExperimentManager
from research_runtime.orchestrator import ResearchOrchestrator
from research_runtime.research_service import (
    ResearchDependencies,
    ResearchService,
    default_evidence_provider,
)
from research_runtime.research_store import approval_cli
from research_runtime.research_store import ResearchStores
from research_runtime.schemas import ResearchObjective

SCINET_ROOT = Path(os.getenv("SCINET_DIR", r"E:\LLM\SciNet++"))
MANIFEST = SCINET_ROOT / "datasets" / "pendulum_smoke_v1" / "manifest.json"
CONFIG = "configs/pendulum_smoke.yaml"
SEEDS = (42, 43, 44, 45, 46)
VALID_DECISIONS = {"accepted_for_confirmation", "rejected", "inconclusive"}


@unittest.skipIf(os.environ.get("RESEARCH_SKIP_E2E") == "1",
                 "RESEARCH_SKIP_E2E=1")
@unittest.skipUnless(MANIFEST.is_file()
                      and (SCINET_ROOT / "research_api" / "cli.py").is_file(),
                      "SciNet++ checkout / frozen dataset unavailable")
class ResearchLoopE2ETests(unittest.TestCase):
    def test_real_objective_to_research_report(self):
        dataset_version = json.loads(
            MANIFEST.read_text(encoding="utf-8"))["dataset_version"]

        # ---- 1. REAL baseline evidence (train -> validation, frozen test
        #          is deliberately NOT touched) -----------------------------
        adapter = SciNetAdapter()
        baseline = ResearchOrchestrator(ExperimentManager(adapter)).run_baseline(
            ResearchObjective(
                objective_id="obj_p106_e2e", domain="pendulum_world_model",
                description="reduce long-horizon prediction error",
                success_criteria={"latent_mse@1": {"max": 0.05},
                                  "cosine@1": {"min": 0.40}}),
            config_path=CONFIG, epochs=1, wait_timeout=300,
            evaluation_split="validation")
        source_experiment_id = baseline["experiment_id"]
        metrics = (baseline.get("evaluate") or {}).get("metrics") or {}
        self.assertTrue(baseline["pipeline_success"], baseline)
        self.assertTrue(metrics, "real evaluation must produce structured metrics")

        # ---- 2. REAL dependencies, temporary state ----------------------
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stores = ResearchStores(root / "research.sqlite")
            campaign_store = CampaignStore(root / "campaigns.sqlite")
            domain = scinet_research_policy(
                project_dir=SCINET_ROOT, paired_seeds=SEEDS,
                report_dir=str(root / "reports"))
            self.assertEqual(domain.dataset_version, dataset_version)
            campaign_root = SCINET_ROOT / domain.campaign_root
            report_store = ComparisonReportStore(campaign_root)
            builder = CandidateConfigBuilder(
                SCINET_ROOT, allowed_keys=domain.allowed_config_keys,
                value_rules=domain.value_rules,
                campaign_root=domain.campaign_root)
            policy = scinet_campaign_policy(
                epochs=1, timeout_per_job_s=domain.job_timeout_s,
                max_jobs=domain.max_jobs)

            def runner_factory():
                return build_scinet_campaign_runner(
                    store=campaign_store, config_builder=builder, policy=policy)

            def comparator_factory():
                return build_scinet_comparator(
                    campaign_store=campaign_store, report_store=report_store,
                    project_dir=SCINET_ROOT)

            service = ResearchService(ResearchDependencies(
                stores=stores, domain=domain,
                campaign_store=campaign_store, report_store=report_store,
                runner_factory=runner_factory,
                comparator_factory=comparator_factory,
                evidence_provider=default_evidence_provider,
                dataset_verifier=lambda base, version:
                    builder.verify_dataset(base, version),
                campaign_root=str(campaign_root),
            ))

            def ok(label: str, payload: dict) -> dict:
                self.assertTrue(payload.get("ok"), f"{label}: {payload}")
                return payload

            # ---- 3. Objective -> evidence -> Critic -> Plan --------------
            session = ok("session", service.start_research_session(
                objective_id="obj_p106_e2e",
                description="reduce long-horizon prediction error"))
            sid = session["session_id"]
            hypothesis = {
                "hypothesis_id": "H_e2e_lr",
                "title": "learning-rate intervention",
                "statement": ("Reducing the optimisation step size may lower "
                              "one-step latent prediction error"),
                "mechanism": "optimisation step size",
                "evidence_refs": [
                    f"experiment:{source_experiment_id}:metric:latent_mse@1"],
                "intervention": SCINET_INTERVENTION_POLICY[
                    "dynamics_error_accumulation"],
                "predicted_effects": [{"metric": "latent_mse@1",
                                       "direction": "decrease"}],
                "falsification_criteria": [
                    "no improvement over paired seeds on validation"],
                "confidence": 0.5,
            }
            proposed = ok("propose", service.propose_research_hypotheses(
                objective_id="obj_p106_e2e",
                source_experiment_id=source_experiment_id, session_id=sid,
                hypotheses_json=json.dumps([hypothesis])))
            self.assertEqual(proposed["source_experiment_id"],
                             source_experiment_id)
            reviewed = ok("review", service.review_research_hypotheses(
                hypothesis_ids_json='["H_e2e_lr"]', session_id=sid))
            self.assertEqual(reviewed["approved_ids"], ["H_e2e_lr"])
            plan = ok("plan", service.create_research_plan(
                objective_id="obj_p106_e2e",
                hypothesis_ids_json='["H_e2e_lr"]', session_id=sid,
                plan_id="plan_p106_e2e"))
            request = ok("approval request", service.request_plan_approval(
                plan_id="plan_p106_e2e", session_id=sid))
            self.assertEqual(request["status"], "pending")

            denied = service.run_research_campaign(
                plan_id="plan_p106_e2e", approval_id=request["approval_id"])
            self.assertEqual(denied["error_code"], "approval_not_approved")

            # ---- 4. HUMAN approval through the out-of-band console -------
            with contextlib.redirect_stdout(io.StringIO()):
                rc = approval_cli.main([
                    "--db", str(root / "research.sqlite"), "grant",
                    request["approval_id"], "--approved-by", "alice", "--yes"])
            self.assertEqual(rc, 0)
            self.assertEqual(stores.approvals.get(
                request["approval_id"]).source, "human")

            # ---- 5. REAL campaign: (1 baseline + 1 candidate) x 5 seeds x 2
            run = ok("run", service.run_research_campaign(
                plan_id="plan_p106_e2e",
                approval_id=request["approval_id"], session_id=sid))
            self.assertEqual(run["status"], "completed", run)
            self.assertEqual(run["total_trials"], 10)
            self.assertEqual(run["succeeded_trials"], 10)
            self.assertEqual(run["dataset_version"], dataset_version)
            self.assertEqual(run["evaluation_split"], "validation")
            record = campaign_store.load_campaign(run["campaign_id"])
            self.assertEqual(record.completed_trials, 10)
            for trial in record.trials:
                self.assertEqual(trial.evaluation_split, "validation")
                self.assertTrue(trial.train_run_id)
                self.assertTrue(trial.evaluation_job_id)

            # Duplicate submission must NOT launch another 20 jobs.
            duplicate = ok("duplicate", service.run_research_campaign(
                plan_id="plan_p106_e2e",
                approval_id=request["approval_id"], session_id=sid))
            self.assertTrue(duplicate["already_submitted"])

            # ---- 6. REAL P1-05 comparison + Research Report -------------
            compared = ok("compare", service.compare_research_campaign(
                campaign_id=run["campaign_id"], session_id=sid))
            self.assertEqual(len(compared["comparisons"]), 1)
            verdict = compared["comparisons"][0]
            self.assertIn(verdict["decision"], VALID_DECISIONS)
            self.assertEqual(verdict["n_pairs"], 5)

            reports = ok("comparison report metadata",
                         service.get_comparison_report(
                             campaign_id=run["campaign_id"]))
            self.assertEqual(len(reports["reports"]), 1)
            full = ok("comparison report", service.get_comparison_report(
                campaign_id=run["campaign_id"],
                comparison_id=verdict["comparison_id"]))
            report = full["report"]
            self.assertEqual(report["dataset_version"], dataset_version)
            self.assertEqual(report["evaluation_split"], "validation")
            self.assertTrue(report["report_sha256"])
            self.assertEqual(report["decision"], verdict["decision"])

            final = ok("research report", service.get_research_report(
                session_id=sid))
            research = final["report"]
            self.assertEqual(research["outcome"], verdict["decision"])
            self.assertEqual(research["promotion"], "not_requested")
            self.assertFalse(research["frozen_test_used"])
            self.assertEqual(research["approvals"][0]["source"], "human")
            self.assertTrue(research["experience_refs"])
            self.assertTrue(Path(final["report_path"]).is_file())
            self.assertEqual(stores.sessions.experience(sid)[0]["kind"],
                             "research_outcome")

            # No tool / report may promote a model or use the frozen split.
            report_text = Path(final["report_path"]).read_text(encoding="utf-8")
            self.assertNotIn('"promotion": "promoted"', report_text)
            self.assertNotIn('"frozen_test_used": true', report_text)

            print(f"\n[P1-06 E2E] campaign={run['campaign_id']} "
                  f"dataset={dataset_version} "
                  f"decision={verdict['decision']} "
                  f"rule={verdict['decision_rule']} "
                  f"n_pairs={verdict['n_pairs']} "
                  f"mean_delta={verdict['mean_delta']:.6f}\n"
                  f"            research_outcome={research['outcome']} "
                  f"promotion={research['promotion']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
