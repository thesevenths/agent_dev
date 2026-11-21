"""M7 -- unified Autonomous Scientific Discovery Coordinator.

This module coordinates the scientific discovery pipeline only. It does NOT
create a second runtime, scheduler or campaign engine:

  * M5 RuntimeState owns the hash-chained journal, executor claim, checkpoints
    and budget reservations.
  * M6 Harness SHA is read from the bound protocol and written to every stage.
  * The domain pack owns real engine execution and metric evaluation.
  * The claim graph and verified memory own long-term scientific knowledge.

Every stage is idempotent and resumable from an on-disk state file. Approval,
budget and evidence gates fail closed.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from ..harness_rsi.identity import freeze_json, sha256_file, sha256_json
from ..harness_rsi.benchmark_v2 import (BenchmarkSpecV2, BenchmarkTask,
                                        HoldoutAccessVerifier)
from ..runtime_state import (RuntimeBudgetExhausted, RuntimeState,
                             RuntimeStateError, config_sha)
from .domain_packs import load_pack
from .hypotheses.engine import (IndependentCritic, ScientificHypothesis,
                                generate_competing_hypotheses, review_all)
from .knowledge import KGNode, ClaimEvidenceGraph
from .literature import (CrossrefProvider, LiteratureStore, OpenAlexProvider,
                         Paper, SemanticScholarProvider, expand_query)
from .literature.base import ResearchQuestionInput
from .memory import ResearchMemoryRecord, VerifiedResearchMemory
from .novelty.assessor import assess_novelty, extract_claims
from .questions import ResearchQuestion, assess_value, generate_questions
from .schema import install_m7_schema
from .experiment.designer import (design_experiment, freeze_preregistration,
                                  verify_preregistration)

STAGES = (
    "DISCOVERY_START", "LITERATURE_RETRIEVAL", "NOVELTY_ASSESSMENT",
    "QUESTION_GENERATION", "VALUE_ASSESSMENT", "HYPOTHESES_COMPETITION",
    "EXPERIMENT_DESIGN", "PREREGISTRATION", "HUMAN_APPROVAL",
    "EXPERIMENT_EXECUTION", "EVIDENCE_COLLECTION",
    "INDEPENDENT_VERIFICATION", "REPLICATION", "CLAIM_UPDATE",
    "EXPERIENCE_UPDATE", "DISCOVERY_COMPLETE",
)
TERMINAL = {"COMPLETED", "FAILED", "BLOCKED", "INCONCLUSIVE", "REFUTED"}
PROTOCOL_VERSION = "m7_discovery_v1"


class DiscoveryBlocked(RuntimeError):
    """A gate refused to continue; the run must end BLOCKED/INCONCLUSIVE."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _parse_json_object(content: str) -> dict[str, Any]:
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model response")
    obj = json.loads(raw[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model response JSON is not an object")
    return obj


class DiscoveryCoordinator:
    def __init__(self, *, run_id: str, db: Any, out_dir: str | Path,
                 domain_pack: str = "embodied_world_model",
                 agent: Any | None = None,
                 providers: list[Any] | None = None,
                 executor: Callable[..., dict[str, Any]] | None = None,
                 approval: Any | None = None,
                 mme: float = 0.01,
                 seeds: list[int] | None = None,
                 selection_seeds: list[int] | None = None,
                 replication_seeds: list[int] | None = None,
                 global_budget: int = 2,
                 harness_sha: str = "",
                 max_requests: int = 12):
        self.run_id = run_id
        self.db = db
        self.root = Path(out_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.pack = load_pack(domain_pack)
        self.domain = domain_pack
        self.agent = agent
        self.providers = providers if providers is not None else [
            OpenAlexProvider(cache_dir=self.root / "cache" / "openalex"),
            CrossrefProvider(cache_dir=self.root / "cache" / "crossref"),
            SemanticScholarProvider(cache_dir=self.root / "cache" / "s2")]
        self.executor = executor or self.pack.run_experiment
        self.approval = approval
        self.mme = float(mme)
        self.seeds = list(seeds or [42, 43, 44])
        self.selection_seeds = list(selection_seeds or self.seeds[:2])
        self.replication_seeds = list(replication_seeds or [45, 46])
        self.global_budget = int(global_budget)
        self.harness_sha = harness_sha
        self.max_requests = int(max_requests)
        install_m7_schema(self.db)
        self.runtime = RuntimeState(db=db)
        self.graph = ClaimEvidenceGraph(self.root / "claims")
        self.memory = VerifiedResearchMemory(
            self.root / "memory", graph=self.graph, db=db,
            evidence_resolver=self._resolve_evidence)
        self.state_path = self.root / "discovery_state.json"
        self.state: dict[str, Any] = (json.loads(self.state_path.read_text(
            encoding="utf-8")) if self.state_path.is_file() else {
                "version": PROTOCOL_VERSION, "run_id": run_id,
                "status": "RUNNING", "stage_index": 0, "stages": {},
                "artifacts": {}, "trajectories": [], "scientific_verdict": ""})
        if self.state_path.is_file():
            self.state["state_file_sha"] = sha256_file(self.state_path)
        self.dataset = self.pack.dataset_identity()
        self.dataset_shas = {
            "dataset_manifest_sha": self.dataset["dataset_manifest_sha"],
            "dataset_identity_sha": self.dataset["dataset_identity_sha"],
            "family_manifest_sha": self.dataset["family_manifest"]["family_manifest_sha"],
        }
        self.config = {
            "version": PROTOCOL_VERSION, "domain": self.domain,
            "mme": self.mme, "seeds": self.seeds,
            "selection_seeds": self.selection_seeds,
            "replication_seeds": self.replication_seeds,
            "budget": self.global_budget, "harness_sha": self.harness_sha,
            "dataset_shas": self.dataset_shas}
        self.config_sha = config_sha(self.config)

    # --------------------------------------------------------------- helpers
    def _freeze(self, name: str, obj: Any) -> dict[str, Any]:
        meta = freeze_json(obj, self.artifacts / f"{name}.json")
        self.state["artifacts"][name] = {**meta, "obj": obj}
        return self.state["artifacts"][name]

    def _stage(self, stage: str, **payload: Any) -> None:
        rec = {"stage": stage, "at": _now(),
               "harness_sha": self.harness_sha, **payload}
        self.runtime.append(self.run_id, int(self.state["stage_index"]),
                            f"m7_{stage.lower()}", rec)

    def _save(self) -> None:
        self.state["updated_at"] = _now()
        persisted = {k: v for k, v in self.state.items()
                     if k != "state_file_sha"}
        frozen = freeze_json(persisted, self.state_path)
        self.state["state_file_sha"] = frozen["sha256"]

    def _complete_stage(self, stage: str, result: dict[str, Any]) -> None:
        self.state["stages"][stage] = {"status": "DONE", "at": _now(),
                                       "result_sha": sha256_json(result)}
        self._stage(stage, result=result)
        self._save()

    def _resolve_evidence(self, digest: str) -> Path | None:
        candidates = [Path(m["path"]) for m in
                      self.state.get("artifacts", {}).values()
                      if m.get("path")]
        candidates.extend(self.root.glob("experiments/*/engine_results.json"))
        candidates.extend(self.root.rglob("result.json"))
        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            if sha256_file(path) == digest:
                return path
        return None

    def _cite_evidence(self, result: dict[str, Any]) -> str:
        if result.get("summary"):
            return str(result["summary_sha256"])
        return sha256_json(result)

    # ---------------------------------------------------------------- gates
    def _check_approval(self, protocol_sha: str) -> dict[str, Any]:
        prior = self.state.get("approval")
        if prior and prior.get("protocol_sha") == protocol_sha \
                and prior.get("decision") == "APPROVED":
            return {"ok": True, "duplicate": True}
        source = self.approval
        if callable(source):
            decision = source(protocol_sha)
        else:
            decision = source or {}
        if not isinstance(decision, dict):
            raise DiscoveryBlocked("approval gate requires structured decision")
        reasons: list[str] = []
        if str(decision.get("protocol_sha", "")) != protocol_sha:
            reasons.append("approval_not_bound_to_protocol")
        if str(decision.get("approver", "")).lower() in ("", "agent", "llm",
                                                         "research_agent"):
            reasons.append("approval_requires_human")
        if str(decision.get("decision", "")).upper() != "APPROVED":
            reasons.append("not_approved")
        if reasons:
            raise DiscoveryBlocked("approval_gate:" + "|".join(reasons))
        self.state["approval"] = {**decision, "at": _now(),
                                  "protocol_sha": protocol_sha}
        return {"ok": True, "approval_sha": sha256_json(self.state["approval"])}

    def _reserve(self, stage: str, campaign_id: str) -> str:
        rid = RuntimeState.reservation_id(
            self.run_id, int(self.state["stage_index"]), self.domain,
            f"{stage}:{campaign_id}")
        try:
            self.runtime.reserve_execution(
                self.run_id, rid, config_sha=self.config_sha,
                domain_id=self.domain,
                round_no=int(self.state["stage_index"]))
        except RuntimeBudgetExhausted as exc:
            raise DiscoveryBlocked(f"budget_exhausted:{exc}") from exc
        self.runtime.bind_reservation(self.run_id, rid,
                                      campaign_id=campaign_id)
        self.state.setdefault("reservations", {})[stage] = {
            "reservation_id": rid, "campaign_id": campaign_id}
        return rid

    # ---------------------------------------------------------------- stages
    def _start(self) -> dict[str, Any]:
        existing = self.runtime.run(self.run_id)
        if not existing:
            self.runtime.initialize(
                self.run_id, config=self.config,
                dataset_shas=self.dataset_shas, max_rounds=len(STAGES),
                global_budget=self.global_budget)
        else:
            if existing.get("config_sha") != self.config_sha:
                raise DiscoveryBlocked("runtime_config_sha_drift")
            if existing.get("dataset_shas") != json.dumps(
                    self.dataset_shas, sort_keys=True, ensure_ascii=False):
                raise DiscoveryBlocked("runtime_dataset_sha_drift")
        if not self.runtime.claim_executor(self.run_id, "m7_discovery_coordinator"):
            raise DiscoveryBlocked("executor_claim_failed_or_terminal")
        journal = self.runtime.verify_journal(self.run_id)
        if not journal["ok"]:
            raise DiscoveryBlocked("runtime_journal_tampered")
        return {"ok": True, "config_sha": self.config_sha,
                "dataset": self.dataset, "journal": journal,
                "protocol_version": PROTOCOL_VERSION}

    def _literature(self) -> dict[str, Any]:
        inp: ResearchQuestionInput = self.pack.literature_input()
        queries = expand_query(inp)[:self.max_requests]
        store = LiteratureStore(self.root / "literature")
        store.record_queries(queries)
        budget = {"max_requests": self.max_requests, "requests_used": 0,
                  "timeout_s": 20.0}
        for provider in self.providers:
            for query in queries:
                if budget["requests_used"] >= self.max_requests:
                    break
                try:
                    papers = provider.search(query, limit=5, budget=budget)
                    store.add_papers(papers)
                    store.record_run(provider=provider.name, query=query,
                                     n_returned=len(papers),
                                     events=getattr(provider, "events", []))
                except Exception as exc:  # provider failure is recorded, never hidden
                    store.record_run(provider=provider.name, query=query,
                                     n_returned=0,
                                     events=getattr(provider, "events", []),
                                     error=f"{type(exc).__name__}: {exc}")
        coverage = store.finalize()
        literature_obj = json.loads(Path(store.artifact).read_text(
            encoding="utf-8"))
        frozen = self._freeze("literature", literature_obj)
        self.state["literature_input"] = inp.to_dict()
        return {"ok": coverage["coverage"]["status"] != "INSUFFICIENT",
                "input": inp.to_dict(), "coverage": coverage,
                "literature_sha": frozen["sha256"],
                "store_path": store.artifact}

    def _novelty(self) -> dict[str, Any]:
        literature = self.state["artifacts"]["literature"]["obj"]
        papers = [Paper(**p) for p in literature["papers"]]
        inp = ResearchQuestionInput(**self.state["literature_input"])
        claims = extract_claims("pilot", inp.question, inp.target_claims)
        hints: list[str] = []
        if self.agent is not None:
            traj = self.agent.call(
                prompt=("Identify prior-art relations ONLY from these retrieved "
                        "paper titles/abstracts. Return JSON with hints. "
                        + json.dumps([{"title": p.title, "abstract": p.abstract[:500]}
                                      for p in papers[:10]])),
                system="You are a careful scientific literature reviewer.",
                toolset=["literature_store"], response_format_json=True)
            self._record_trajectory("novelty", traj)
            try:
                hints = [str(x) for x in _parse_json_object(
                    traj.content).get("hints", [])]
            except (ValueError, TypeError):
                hints = []
        result = assess_novelty(claims, papers,
                                coverage=literature["coverage"],
                                llm_hints=hints)
        frozen = self._freeze("novelty", result)
        return {"ok": True, "novelty_sha": frozen["sha256"],
                "literature_check": result["literature_check"],
                "statuses": [v["status"] for v in result["verdicts"]]}

    def _questions(self) -> dict[str, Any]:
        literature = self.state["artifacts"]["literature"]["obj"]
        novelty = self.state["artifacts"]["novelty"]["obj"]
        gaps = [f"retrieved evidence: {p['title']}" for p in
                literature["papers"][:5]] or ["insufficient literature coverage"]
        questions = generate_questions(
            domain=self.domain, novelty=novelty, literature_gaps=gaps,
            experiences=[], max_questions=4)
        if not questions:
            raise DiscoveryBlocked("no_evidence_grounded_questions")
        frozen = self._freeze("questions", [q.to_dict() for q in questions])
        self.state["questions"] = [q.to_dict() for q in questions]
        return {"ok": True, "questions_sha": frozen["sha256"],
                "n_questions": len(questions), "ids": [q.question_id
                                                      for q in questions]}

    def _value(self) -> dict[str, Any]:
        novelty = self.state["artifacts"]["novelty"]["obj"]
        questions = [ResearchQuestion(**q) for q in self.state["questions"]]
        assessments = [assess_value(q, novelty=novelty,
                                    cost_units=1.0).to_dict() for q in questions]
        ranked = sorted(zip(assessments, questions),
                        key=lambda av: (-av[0]["priority"], av[1].question_id))
        selected = ranked[0][1]
        frozen = self._freeze("value", {"assessments": assessments,
                                        "selected_question_id":
                                        selected.question_id})
        self.state["selected_question"] = selected.to_dict()
        return {"ok": True, "value_sha": frozen["sha256"],
                "selected": selected.question_id,
                "priorities": [a["priority"] for a in assessments]}

    def _hypotheses(self) -> dict[str, Any]:
        q = ResearchQuestion(**self.state["selected_question"])
        literature = self.state["artifacts"]["literature"]["obj"]
        refs = [f"doi:{p['doi']}" if p["doi"] else f"paper:{p['paper_id']}"
                for p in literature["papers"][:8]]
        base = generate_competing_hypotheses(q, literature_refs=refs)
        original_hyps = base
        llm_hyps: list[ScientificHypothesis] | None = None
        llm_used = False
        if self.agent is not None:
            traj = self.agent.call(
                prompt=("Propose competing mechanisms, observable predictions "
                        "and executable falsification conditions as JSON with "
                        "keys H1,H2,H3,H0. Question: " + q.title + "\n"
                        "Literature: " + json.dumps(refs)),
                system="You are a skeptical scientist. Every hypothesis must "
                       "be falsifiable and mechanism-distinct.",
                toolset=["literature_store", "scientific_hypothesis_engine"],
                response_format_json=True, max_tokens=1200)
            self._record_trajectory("hypotheses", traj)
            try:
                obj = _parse_json_object(traj.content)
                data = [h.to_dict() for h in base]
                for hdata in data:
                    override = obj.get(hdata["kind"])
                    if isinstance(override, dict):
                        if isinstance(override.get("mechanism"), str):
                            hdata["mechanism"] = override["mechanism"]
                        preds = (override.get("predictions")
                                 or override.get("observable_predictions")
                                 or override.get("predicted_effects"))
                        fals = (override.get("falsifying_observations")
                                or override.get("falsification_conditions")
                                or override.get("refutation_conditions"))
                        if isinstance(preds, list) and preds:
                            hdata["predictions"] = [str(x) for x in preds]
                        if isinstance(fals, list) and fals:
                            hdata["falsifying_observations"] = [str(x) for x in fals]
                candidate = [ScientificHypothesis(**hdata) for hdata in data]
                candidate_reviewed = review_all(
                    candidate, critic=IndependentCritic())
                if all(h.status == "ACCEPTED" for h in candidate_reviewed):
                    llm_hyps = candidate
                    llm_used = traj.ok
            except (ValueError, TypeError):
                llm_used = False
        base = llm_hyps or original_hyps
        reviewed = review_all(base, critic=IndependentCritic())
        accepted = [h for h in reviewed if h.status == "ACCEPTED"]
        if len(accepted) < 4:
            raise DiscoveryBlocked("hypothesis_competition_incomplete")
        frozen = self._freeze("hypotheses", {
            "llm_used": llm_used, "hypotheses": [h.to_dict() for h in reviewed]})
        self.state["hypotheses"] = [h.to_dict() for h in reviewed]
        return {"ok": True, "hypotheses_sha": frozen["sha256"],
                "llm_used": llm_used, "accepted": len(accepted),
                "kinds": sorted(h.kind for h in accepted)}

    def _experiment(self) -> dict[str, Any]:
        q = ResearchQuestion(**self.state["selected_question"])
        hyps = [ScientificHypothesis(**h) for h in self.state["hypotheses"]]
        fm = self.dataset["family_manifest"]
        protocol = design_experiment(
            question=q, hypotheses=hyps,
            family_manifest_sha=fm["family_manifest_sha"],
            split_manifest_sha=self.dataset["dataset_manifest_sha"],
            dataset_manifest_sha=self.dataset["dataset_manifest_sha"],
            primary_metric=self.config["dataset_shas"].get(
                "primary_metric", "state_mse@1"),
            mme=self.mme, seeds=self.seeds)
        frozen = self._freeze("experiment_protocol", protocol.to_dict())
        self.state["protocol"] = protocol.to_dict()
        return {"ok": True, "experiment_id": protocol.experiment_id,
                "experiment_sha": frozen["sha256"],
                "primary_metric": protocol.primary_metric,
                "mme": protocol.minimum_meaningful_effect}

    def _preregistration(self) -> dict[str, Any]:
        protocol_data = self.state["protocol"]
        protocol = type("P", (), {})()
        protocol.to_dict = lambda: protocol_data
        protocol.sha256 = sha256_json({**protocol_data,
                                       "preregistration_sha": ""})
        frozen = freeze_preregistration(protocol,
                                        self.artifacts / "preregistration.json")
        verified = verify_preregistration(
            protocol, self.artifacts / "preregistration.json")
        if not verified["ok"]:
            raise DiscoveryBlocked("preregistration_verify_failed")
        self.state["preregistration_sha"] = frozen["preregistration_sha"]
        return {"ok": True, "preregistration_sha": frozen["preregistration_sha"],
                "file_sha256": frozen["file_sha256"],
                "before_results": True}

    def _approval(self) -> dict[str, Any]:
        protocol_sha = self.state["preregistration_sha"]
        approval = self._check_approval(protocol_sha)
        return {"ok": True, "protocol_sha": protocol_sha, **approval}

    def _execute_selection(self) -> dict[str, Any]:
        protocol = self.state["protocol"]
        campaign_id = "m7sel_" + self.state["preregistration_sha"][:20]
        reservation = self._reserve("EXPERIMENT_EXECUTION", campaign_id)
        arms = ["baseline", "action_conditioned", "capacity_control"]
        result = self.executor(
            experiment_id=protocol["experiment_id"] + "_selection",
            arms=arms, seeds=self.selection_seeds, splits=["validation"],
            out_dir=str(self.root / "experiments" / "selection"))
        self.state["selection_campaign"] = campaign_id
        return {"ok": True, "campaign_id": campaign_id,
                "reservation_id": reservation,
                **{k: v for k, v in result.items() if k != "summary"}}

    def _evidence(self) -> dict[str, Any]:
        source = self.root / "experiments" / "selection" / "engine_results.json"
        if not source.is_file():
            raise DiscoveryBlocked("engine_results_missing")
        raw = json.loads(source.read_text(encoding="utf-8"))
        from .domain_packs.embodied_world_model import evaluator
        verified = evaluator.verify_result_shas({"summary": raw})
        if not verified["ok"]:
            raise DiscoveryBlocked("result_sha_verification_failed")
        evidence_sha = sha256_file(source)
        frozen = self._freeze("evidence", {
            "evidence_sha": evidence_sha, "result_sha256": evidence_sha,
            "verification": verified, "n_results": verified["n_checked"]})
        return {"ok": True, "evidence_sha": evidence_sha,
                "artifact_sha": frozen["sha256"],
                "engine_result_sha": evidence_sha,
                "n_results": verified["n_checked"]}

    def _verify(self) -> dict[str, Any]:
        source = self.root / "experiments" / "selection" / "engine_results.json"
        raw = json.loads(source.read_text(encoding="utf-8"))
        evaluation = self.pack.evaluate(
            results=raw, primary_metric="state_mse@1", mme=self.mme)
        safety = self.pack.safety_check(results=raw)
        frozen = self._freeze("evaluation", {"evaluation": evaluation,
                                             "safety": safety})
        self.state["selection_evaluation"] = evaluation
        if not evaluation["ok"] or not safety["ok"]:
            raise DiscoveryBlocked("independent_verification_failed")
        verdict = evaluation["per_arm"].get("action_conditioned", {}).get(
            "verdict", "INCONCLUSIVE")
        self.state["selection_verdict"] = verdict
        return {"ok": True, "evaluation_sha": frozen["sha256"],
                "verdict": verdict, "safety_ok": safety["ok"]}

    def _holdout_spec(self) -> BenchmarkSpecV2:
        fams = self.dataset["family_manifest"]["splits"]
        tasks: list[BenchmarkTask] = []
        for source_split, m7_split in (("test", "holdout"),
                                       ("ood", "holdout")):
            for fam in fams[source_split]["families"]:
                task_id = f"m7_{source_split}_{fam}"
                base = {"task_id": task_id, "family_id": f"m7::{fam}",
                        "domain": self.domain, "objective": "holdout replication",
                        "task_config": {"split": source_split},
                        "split": m7_split,
                        "source": {"engine": "embodied_world_model_local",
                                   "split": source_split},
                        "budget": {"max_jobs": 4}, "seed": 45,
                        "primary_metric": "state_mse@1", "direction": "minimize"}
                tasks.append(BenchmarkTask(**base, task_sha=sha256_json(base)))
        return BenchmarkSpecV2(version="m7_holdout_v1", tasks=tasks,
                               manifest_sha=self.dataset["dataset_manifest_sha"])

    def _replication(self) -> dict[str, Any]:
        protocol = self.state["protocol"]
        fams = self.dataset["family_manifest"]["splits"]
        requested = [f"m7_{s}_{f}" for s in ("test", "ood")
                     for f in fams[s]["families"]]
        verifier = HoldoutAccessVerifier(
            self._holdout_spec(), self.root / "claims" / "holdout_access.json")
        auth = verifier.authorize(
            purpose="final_holdout_evaluation",
            experiment_id=protocol["experiment_id"] + "_replication",
            requested_task_ids=requested, actor="m7_discovery_coordinator")
        if not auth["ok"]:
            raise DiscoveryBlocked("holdout_access_denied:" + auth["reason"])
        chain = verifier.verify_chain()
        if not chain["ok"]:
            raise DiscoveryBlocked("holdout_ledger_broken")
        campaign_id = "m7rep_" + self.state["preregistration_sha"][:20]
        reservation = self._reserve("REPLICATION", campaign_id)
        result = self.executor(
            experiment_id=protocol["experiment_id"] + "_replication",
            arms=["baseline", "action_conditioned"],
            seeds=self.replication_seeds, splits=["test", "ood"],
            out_dir=str(self.root / "experiments" / "replication"))
        source = self.root / "experiments" / "replication" / "engine_results.json"
        raw = json.loads(source.read_text(encoding="utf-8"))
        from .domain_packs.embodied_world_model import evaluator
        integrity = evaluator.verify_result_shas({"summary": raw})
        evaluation = self.pack.evaluate(
            results=raw, primary_metric="state_mse@1", mme=self.mme)
        safety = self.pack.safety_check(results=raw)
        evidence_sha = sha256_file(source)
        verdict = evaluation["per_arm"].get("action_conditioned", {}).get(
            "verdict", "INCONCLUSIVE")
        frozen = self._freeze("replication", {
            "replication_sha": evidence_sha,
            "holdout_authorization": auth, "holdout_chain": chain,
            "holdout_chain_sha": chain["chain_sha"], "verdict": verdict,
            "result_sha256": evidence_sha, "integrity": integrity,
            "evaluation": evaluation, "safety": safety,
            "engine_summary_sha": result.get("summary_sha256", "")})
        if not integrity["ok"] or not evaluation["ok"] or not safety["ok"]:
            raise DiscoveryBlocked("replication_evidence_invalid")
        self.state["replication_verdict"] = verdict
        return {"ok": True, "campaign_id": campaign_id,
                "reservation_id": reservation,
                "replication_sha": evidence_sha,
                "artifact_sha": frozen["sha256"],
                "holdout_chain_sha": chain["chain_sha"], "verdict": verdict,
                **{k: v for k, v in result.items() if k != "summary"}}

    def _claim(self) -> dict[str, Any]:
        q = ResearchQuestion(**self.state["selected_question"])
        selection = self.state["selection_verdict"]
        replication = self.state["replication_verdict"]
        evidence = self.state["artifacts"]["evidence"]["obj"]
        rep_sha = self.state["artifacts"]["replication"]["obj"]["replication_sha"]
        claim_id = "claim_" + hashlib.sha256(
            (q.question_id + rep_sha).encode()).hexdigest()[:16]
        source_sha = evidence["evidence_sha"]
        self.graph.add_node(KGNode(q.question_id, "ResearchQuestion",
                                   q.to_dict(), source_sha))
        for h in self.state["hypotheses"]:
            self.graph.add_node(KGNode(h["hypothesis_id"], "Hypothesis",
                                       h, source_sha))
            self.graph.add_edge(source_id=h["hypothesis_id"],
                                target_id=q.question_id,
                                relation="DERIVED_FROM",
                                evidence_sha=source_sha)
        experiment_id = self.state["protocol"]["experiment_id"]
        self.graph.add_node(KGNode(experiment_id, "Experiment",
                                   self.state["protocol"], source_sha))
        self.graph.add_edge(source_id=experiment_id, target_id=q.question_id,
                            relation="TESTED_BY", evidence_sha=source_sha)
        self.graph.add_node(KGNode(
            "ev_selection", "Evidence", {"verdict": selection}, source_sha))
        self.graph.add_node(KGNode(
            "ev_replication", "Replication", {"verdict": replication},
            rep_sha))
        self.graph.add_node(KGNode(
            claim_id, "ScientificClaim",
            {"text": q.expected_contribution, "question_id": q.question_id},
            source_sha))
        self.graph.add_edge(source_id="ev_selection", target_id=claim_id,
                            relation="SUPPORTS" if selection == "SUPPORTS"
                            else "REFUTES", evidence_sha=source_sha)
        self.graph.add_edge(source_id="ev_replication", target_id=claim_id,
                            relation="REPLICATES" if replication == "SUPPORTS"
                            else "REFUTES", evidence_sha=rep_sha)
        chain = ["LITERATURE_CHECKED", "HYPOTHESIS_REGISTERED",
                 "PREREGISTERED", "EXPERIMENTED", "EVIDENCE_REVIEWED"]
        for state in chain:
            self.graph.transition(claim_id, state, evidence_sha=source_sha)
        if selection == "SUPPORTS" and replication == "SUPPORTS":
            self.graph.transition(claim_id, "SUPPORTED",
                                  evidence_sha=source_sha)
            self.graph.transition(claim_id, "REPLICATED", evidence_sha=rep_sha)
            scientific_verdict = "REPLICATED"
        elif selection == "REFUTES" or replication == "REFUTES":
            self.graph.transition(claim_id, "REFUTED", evidence_sha=rep_sha)
            scientific_verdict = "REFUTED"
        else:
            self.graph.transition(claim_id, "INCONCLUSIVE",
                                  evidence_sha=rep_sha)
            scientific_verdict = "INCONCLUSIVE"
        integrity = self.graph.verify_integrity()
        frozen = self.graph.finalize()
        if not frozen.get("ok") or not integrity["ok"]:
            raise DiscoveryBlocked("claim_graph_integrity_failed")
        self.state["claim_id"] = claim_id
        self.state["scientific_verdict"] = scientific_verdict
        artifact = self._freeze("claim_graph", {
            "claim_id": claim_id, "scientific_verdict": scientific_verdict,
            "graph_sha": frozen["graph_sha"], "integrity": integrity})
        return {"ok": True, "claim_id": claim_id,
                "scientific_verdict": scientific_verdict,
                "graph_sha": frozen["graph_sha"],
                "artifact_sha": artifact["sha256"], "integrity": integrity}

    def _experience(self) -> dict[str, Any]:
        claim_id = self.state["claim_id"]
        protocol = self.state["protocol"]
        evidence_sha = self.state["artifacts"]["evidence"]["obj"]["evidence_sha"]
        rep_sha = self.state["artifacts"]["replication"]["obj"]["replication_sha"]
        verdict = self.state["scientific_verdict"]
        rec = ResearchMemoryRecord(
            memory_id="m7mem_" + hashlib.sha256(
                (claim_id + rep_sha).encode()).hexdigest()[:14],
            claim_id=claim_id, experiment_id=protocol["experiment_id"],
            experience_id="m7exp_" + claim_id[:12],
            evidence_ids=[evidence_sha, rep_sha],
            source_sha=rep_sha,
            verification_status=("REPLICATED" if verdict == "REPLICATED"
                                 else "UNVERIFIED"),
            domain=self.domain, applicability=["embodied_world_model",
                                               "ood_dynamics"],
            limitations=["simulated engine", "single domain pack"],
            confidence=0.85 if verdict == "REPLICATED" else 0.2,
            rationale=f"m7 discovery verdict={verdict}")
        if verdict == "REPLICATED":
            written = self.memory.add_verified(rec)
            tier = "verified"
        else:
            written = self.memory.add_candidate(rec)
            tier = "candidate"
        frozen = self.memory.finalize()
        artifact = self._freeze("memory", {
            "memory_sha": frozen["memory_sha"], "tier": tier,
            "record": rec.to_dict(), "written": written,
            "n_verified": frozen["n_verified"]})
        return {"ok": True, "tier": tier, "memory_sha": frozen["memory_sha"],
                "artifact_sha": artifact["sha256"],
                "n_verified": frozen["n_verified"], **written}

    def _complete(self) -> dict[str, Any]:
        journal = self.runtime.verify_journal(self.run_id)
        if not journal["ok"]:
            raise DiscoveryBlocked("runtime_journal_tampered_at_completion")
        self.state["status"] = "COMPLETED"
        self.state["completed_at"] = _now()
        self._stage("DISCOVERY_COMPLETE",
                    scientific_verdict=self.state["scientific_verdict"],
                    journal=journal)
        self._save()
        self.runtime.set_state(self.run_id, "COMPLETED",
                               event="m7_discovery_completed",
                               payload={"scientific_verdict":
                                        self.state["scientific_verdict"]})
        return {"ok": True, "status": "COMPLETED",
                "scientific_verdict": self.state["scientific_verdict"],
                "journal": journal}

    def _record_trajectory(self, stage: str, traj: Any) -> None:
        data = traj.to_dict() if hasattr(traj, "to_dict") else dict(traj)
        self.state["trajectories"].append({"stage": stage, **data})
        self._save()

    # ------------------------------------------------------------------ run
    def run(self) -> dict[str, Any]:
        if self.state.get("status") in TERMINAL:
            return {"ok": True, "resumed": True,
                    "status": self.state["status"],
                    "scientific_verdict": self.state.get("scientific_verdict")}
        handlers = {
            "DISCOVERY_START": self._start,
            "LITERATURE_RETRIEVAL": self._literature,
            "NOVELTY_ASSESSMENT": self._novelty,
            "QUESTION_GENERATION": self._questions,
            "VALUE_ASSESSMENT": self._value,
            "HYPOTHESES_COMPETITION": self._hypotheses,
            "EXPERIMENT_DESIGN": self._experiment,
            "PREREGISTRATION": self._preregistration,
            "HUMAN_APPROVAL": self._approval,
            "EXPERIMENT_EXECUTION": self._execute_selection,
            "EVIDENCE_COLLECTION": self._evidence,
            "INDEPENDENT_VERIFICATION": self._verify,
            "REPLICATION": self._replication,
            "CLAIM_UPDATE": self._claim,
            "EXPERIENCE_UPDATE": self._experience,
            "DISCOVERY_COMPLETE": self._complete,
        }
        for idx, stage in enumerate(STAGES):
            self.state["stage_index"] = idx
            if self.state["stages"].get(stage, {}).get("status") == "DONE":
                continue
            try:
                result = handlers[stage]()
            except DiscoveryBlocked as exc:
                self.state["status"] = "BLOCKED" if "approval" in str(exc) \
                    or "budget" in str(exc) or "holdout" in str(exc) else \
                    "INCONCLUSIVE"
                self.state["failure"] = {"stage": stage, "reason": str(exc)}
                self.state["scientific_verdict"] = self.state.get(
                    "scientific_verdict", "NOT_ASSESSED")
                self._stage(stage + "_BLOCKED", reason=str(exc))
                self._save()
                return {"ok": False, "status": self.state["status"],
                        "stage": stage, "reason": str(exc)}
            except Exception as exc:  # noqa: BLE001 - fail closed with evidence
                self.state["status"] = "FAILED"
                self.state["failure"] = {"stage": stage,
                                         "reason": f"{type(exc).__name__}: {exc}"}
                self._stage(stage + "_FAILED", reason=str(exc))
                self._save()
                return {"ok": False, "status": "FAILED", "stage": stage,
                        "reason": self.state["failure"]["reason"]}
            self._complete_stage(stage, result)
        return {"ok": True, "status": "COMPLETED",
                "scientific_verdict": self.state.get("scientific_verdict"),
                "state_sha": self.state.get("state_file_sha")}


__all__ = ["STAGES", "TERMINAL", "PROTOCOL_VERSION", "DiscoveryBlocked",
           "DiscoveryCoordinator"]
