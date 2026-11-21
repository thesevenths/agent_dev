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
from .approval_gate import verify_external_approval
from .domain_packs import load_pack
from .hypotheses.engine import (IndependentCritic, ScientificHypothesis,
                                generate_competing_hypotheses, review_all)
from .knowledge import KGNode, ClaimEvidenceGraph
from .literature import (CrossrefProvider, LiteratureStore, OpenAlexProvider,
                         Paper, SemanticScholarProvider, expand_query)
from .literature.base import ResearchQuestionInput
from .memory import ResearchMemoryRecord, VerifiedResearchMemory
from .novelty.assessor import assess_novelty, extract_claims
from .questions import (ResearchQuestion, assess_value,
                        expected_information_gain_from_model,
                        generate_questions)
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
PROTOCOL_VERSION = "m7_1_capability_validation_v1"


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
                 approval_store: Any | None = None,
                 approval_id: str = "",
                 production_mode: bool = False,
                 fixture_approval: bool = False,
                 mme: float = 0.01,
                 seeds: list[int] | None = None,
                 selection_seeds: list[int] | None = None,
                 replication_seeds: list[int] | None = None,
                 global_budget: int = 2,
                 harness_sha: str = "",
                 formal_loader: Any | None = None,
                 production_runtime: Any | None = None,
                 git_commit_sha: str = "",
                 pilot_effects: dict[str, float] | None = None,
                 evidence_map: dict[str, Any] | None = None,
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
        self.approval_store = approval_store
        self.approval_id = str(approval_id or "")
        self.production_mode = bool(production_mode)
        self.fixture_approval = bool(fixture_approval)
        # A fixture stamp is REQUIRED whenever an in-process approval source is
        # supplied outside production. Without it, auto-approval fails closed.
        self._fixture_approved = self.fixture_approval and not self.production_mode
        self._fixture_decision: dict[str, Any] | None = None
        if self._fixture_approved and callable(self.approval):
            # Evaluated lazily at the gate (needs protocol_sha); marker here only.
            self._fixture_callable = self.approval
        else:
            self._fixture_callable = None
            if self._fixture_approved and isinstance(self.approval, dict):
                self._fixture_decision = dict(self.approval)
        self.mme = float(mme)
        self.seeds = list(seeds or [42, 43, 44])
        self.selection_seeds = list(selection_seeds or self.seeds[:2])
        self.replication_seeds = list(replication_seeds or [45, 46])
        self.global_budget = int(global_budget)
        self.harness_sha = harness_sha
        self.formal_loader = formal_loader
        self.production_runtime = production_runtime
        self.git_commit_sha = str(git_commit_sha or "")  # distinct from harness SHA
        self.pilot_effects = dict(pilot_effects or {})
        self.evidence_map = dict(evidence_map or {})
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
            "dataset_shas": self.dataset_shas,
            "formal_loader_present": self.formal_loader is not None,
            "pilot_effects_present": bool(self.pilot_effects)}
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
        """M7.6 P0-A: re-read an OUT-OF-BAND approval from the trusted store.

        Production path
        ---------------
        ONLY source of truth is the M4.3 ``ApprovalStore``. The approval_id must
        reference a durable decision recorded by ``approval_cli.py`` in a
        separate process, bound to THIS preregistration SHA. In-process
        callables/dicts are auto-approvals and are rejected outright.

        Engine-free test path
        ---------------------
        A caller-supplied decision is accepted ONLY if explicitly stamped
        ``fixture=True`` AND ``trust="TEST_FIXTURE"``; it can never satisfy a
        production-mode run. Fixtures are recorded verbatim in state for audit.
        """
        prior = self.state.get("approval")
        if prior and prior.get("protocol_sha") == protocol_sha \
                and prior.get("verified_ok") is True:
            if self.production_mode and prior.get("trust") != "EXTERNAL_STORE":
                raise DiscoveryBlocked("approval_gate:cached_fixture_in_production")
            if self.production_mode:
                # Re-verify against the store on every resume: a revoked/expired
                # approval must invalidate a previously verified run.
                pass
            else:
                return {"ok": True, "duplicate": True,
                        "trust": prior.get("trust", "TEST_FIXTURE")}

        if self.production_mode:
            verdict = verify_external_approval(
                approval_store=self.approval_store, approval_id=self.approval_id,
                protocol_sha=protocol_sha)
            if not verdict.get("ok"):
                raise DiscoveryBlocked(
                    "approval_gate:" + str(verdict.get("reason", "external_verification_failed")))
            self.state["approval"] = {
                **verdict, "protocol_sha": protocol_sha, "at": _now(),
                "trust": "EXTERNAL_STORE", "verified_ok": True,
                "production_mode": True}
            return {"ok": True, "trust": "EXTERNAL_STORE",
                    "approval_sha": sha256_json(self.state["approval"]),
                    "approved_by": verdict.get("approved_by", ""),
                    "approval_id": self.approval_id}

        # ---- engine-free fixture path (never valid in production) ----
        auto = self.approval is not None or self.approval_store is not None
        if auto and not self._fixture_approved:
            raise DiscoveryBlocked(
                "approval_gate:test_fixture_requires_explicit_stamp")
        if self._fixture_callable is not None:
            produced = self._fixture_callable(protocol_sha)
            if not isinstance(produced, dict):
                raise DiscoveryBlocked("approval gate requires structured decision")
            self._fixture_decision = dict(produced)
        decision = dict(self._fixture_decision or {})
        reasons: list[str] = []
        if str(decision.get("protocol_sha", "")) != protocol_sha:
            reasons.append("approval_not_bound_to_protocol")
        if str(decision.get("approver", "")).lower() in (
                "", "agent", "llm", "research_agent", "system"):
            reasons.append("approval_requires_human_identity")
        if str(decision.get("decision", "")).upper() != "APPROVED":
            reasons.append("not_approved")
        if reasons:
            raise DiscoveryBlocked("approval_gate:" + "|".join(reasons))
        self.state["approval"] = {
            **decision, "at": _now(), "protocol_sha": protocol_sha,
            "trust": "TEST_FIXTURE", "fixture": True, "verified_ok": True,
            "production_mode": False,
            "warning": "engine-free test fixture; NOT a human approval"}
        return {"ok": True, "trust": "TEST_FIXTURE", "fixture": True,
                "approval_sha": sha256_json(self.state["approval"])}

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

    def _resolve_formal_harness(self, campaign_id: str) -> dict[str, Any]:
        """Resolve a campaign through M6 HarnessLoader when one is injected."""
        if self.formal_loader is None:
            return {"mode": "LOCAL_DIRECT_EXECUTOR", "ok": True,
                    "note": "no formal loader injected; runtime integration is "
                            "PARTIAL and must not be reported as full"}
        loaded = self.formal_loader.resolve_for_campaign(
            campaign_id=campaign_id, domain=self.domain)
        # M6 HarnessLoader.resolve_for_campaign ALREADY binds the campaign
        # (bindings.bind + adoption.bind_campaign). Re-binding here was a
        # redundant, semantically confusing duplicate; removed in M7.6 P0-B.
        binding = {"campaign_id": campaign_id,
                   "harness_sha": loaded.harness_sha,
                   "component_shas": dict(loaded.component_shas),
                   "parent_sha": loaded.parent_sha,
                   "activation_id": loaded.activation_id}
        self.state.setdefault("harness_bindings", {})[campaign_id] = binding
        return {"mode": "FORMAL_HARNESS_LOADER", "ok": True, **binding}

    def _consume_reservation(self, stage: str, outcome: str) -> None:
        rec = self.state.get("reservations", {}).get(stage)
        if not rec:
            return
        self.runtime.consume_reservation(self.run_id, rec["reservation_id"],
                                         outcome=outcome)
        rec["outcome"] = outcome

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
        if self.production_mode:
            # P0-B: production requires ALL formal contracts. No silent direct
            # executor fallback; no git HEAD standing in for a harness SHA.
            missing = []
            if self.formal_loader is None:
                missing.append("formal_harness_loader")
            if self.production_runtime is None:
                missing.append("formal_runtime_adapter")
            if not self.harness_sha:
                missing.append("harness_sha")
            if missing:
                raise DiscoveryBlocked(
                    "production_runtime_missing:" + ",".join(missing))
            # M7.6: the formal adapter MUST reuse THIS run's M5 RuntimeState
            # (same journal/budget/config), never a second runtime instance.
            self.production_runtime.runtime = self.runtime
            if hasattr(self.production_runtime, "run_id"):
                self.production_runtime.run_id = self.run_id
            self.state["runtime_integration"] = "FORMAL_PRODUCTION_RUNTIME"
            self.state["identity_separation"] = {
                "harness_sha": self.harness_sha,
                "git_commit_sha": self.git_commit_sha,
                "note": "harness identity is resolved by M6 loader, NOT git HEAD"}
        else:
            self.state["runtime_integration"] = (
                "FORMAL_HARNESS_LOADER" if self.formal_loader is not None
                else "PARTIAL_LOCAL_DIRECT_EXECUTOR")
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
        result = assess_novelty(
            claims, papers, coverage=literature["coverage"],
            llm_hints=hints, source_coverage="ABSTRACT_ONLY")
        frozen = self._freeze("novelty", result)
        return {"ok": True, "novelty_sha": frozen["sha256"],
                "literature_check": result["literature_check"],
                "statuses": [v["status"] for v in result["verdicts"]]}

    def _questions(self) -> dict[str, Any]:
        literature = self.state["artifacts"]["literature"]["obj"]
        novelty = self.state["artifacts"]["novelty"]["obj"]
        gaps: list[dict[str, Any]] = []
        for p in literature["papers"][:8]:
            abstract = str(p.get("abstract") or "").strip()
            if len(abstract) < 40:
                continue
            gaps.append({
                "text": abstract[:500],
                "paper_id": p.get("paper_id", ""),
                "doi": p.get("doi", ""),
                "title": p.get("title", ""),
                "evidence_type": "ABSTRACT_SNIPPET",
                "source": p.get("source", ""),
                "query_id": p.get("query_id", ""),
            })
        if not gaps:
            gaps = [{"text": "no usable abstract evidence was retrieved",
                     "evidence_type": "MISSING_EVIDENCE"}]

        # Memory must be actively queried and its use (or exclusion) recorded.
        retrieved = self.memory.retrieve(
            domain=self.domain,
            problem=str(self.state.get("literature_input", {}).get("question", "")),
            top=5)
        experiences = [{"memory_id": r["memory_id"], "claim": r["rationale"],
                        "confidence": r["confidence"],
                        "conditions": r["applicability"]} for r in retrieved]
        memory_trace = {
            "retrieved_ids": [r["memory_id"] for r in retrieved],
            "used_ids": [r["memory_id"] for r in experiences],
            "excluded": [],
            "reason": ("no verified memory passed the claim gate" if not retrieved
                       else "retrieved records grounded question variants"),
        }
        self._freeze("question_memory_trace", memory_trace)

        questions = generate_questions(
            domain=self.domain, novelty=novelty, literature_gaps=gaps,
            experiences=experiences, max_questions=4)
        if not questions:
            raise DiscoveryBlocked("no_evidence_grounded_questions")
        frozen = self._freeze("questions", [q.to_dict() for q in questions])
        self.state["questions"] = [q.to_dict() for q in questions]
        self.state["question_memory_trace"] = memory_trace
        return {"ok": True, "questions_sha": frozen["sha256"],
                "n_questions": len(questions),
                "source_coverage": questions[0].source_coverage,
                "memory_used": len(experiences),
                "ids": [q.question_id for q in questions]}

    def _value(self) -> dict[str, Any]:
        novelty = self.state["artifacts"]["novelty"]["obj"]
        questions = [ResearchQuestion(**q) for q in self.state["questions"]]
        eig_model = self.evidence_map.get("eig")
        # Only pass the EXPLICIT probability-model fields; extra descriptive
        # keys from an operator evidence file must not crash the gate.
        eig_keys = ("hypotheses", "priors", "likelihoods")
        eig_result = (expected_information_gain_from_model(
            **{k: eig_model[k] for k in eig_keys if k in eig_model})
            if isinstance(eig_model, dict) else None)
        evidence_map = dict(self.evidence_map)
        evidence_map.setdefault("feasibility", {})["engine_capability_check"] = [
            {"dataset_version": self.dataset["dataset_version"],
             "family_isolation": self.dataset["family_isolation"]["ok"]}]
        assessments = [assess_value(q, novelty=novelty,
                                    evidence_map=evidence_map,
                                    eig=eig_result).to_dict()
                       for q in questions]
        def sort_key(item):
            a, q = item
            priority = a.get("priority")
            return (0 if isinstance(priority, (int, float)) else 1,
                    -(priority or 0.0), q.question_id)
        ranked = sorted(zip(assessments, questions), key=sort_key)
        selected = ranked[0][1]
        selection_basis = ("EVIDENCE_BASED_PRIORITY" if
                           isinstance(ranked[0][0].get("priority"), (int, float))
                           else "NO_VALUE_EVIDENCE_FALLBACK")
        # M7.6 P1-A: production must NOT silently turn NOT_ASSESSED value into
        # a high-value decision by taking the first candidate. Fail closed.
        if self.production_mode and selection_basis == "NO_VALUE_EVIDENCE_FALLBACK":
            raise DiscoveryBlocked(
                "value_evidence_insufficient:NOT_ASSESSED_cannot_be_ranked_in_production")
        frozen = self._freeze("value", {
            "assessments": assessments,
            "selected_question_id": selected.question_id,
            "selection_basis": selection_basis,
            "eig_status": (eig_result or {}).get("status", "NOT_ASSESSED")})
        self.state["selected_question"] = selected.to_dict()
        self.state["selection_basis"] = selection_basis
        return {"ok": True, "value_sha": frozen["sha256"],
                "selected": selected.question_id,
                "selection_basis": selection_basis,
                "priorities": [a["priority"] for a in assessments]}

    def _hypotheses(self) -> dict[str, Any]:
        q = ResearchQuestion(**self.state["selected_question"])
        literature = self.state["artifacts"]["literature"]["obj"]
        refs = [f"doi:{p['doi']}" if p["doi"] else f"paper:{p['paper_id']}"
                for p in literature["papers"][:8]]
        base = generate_competing_hypotheses(q, literature_refs=refs)
        original_hyps = base
        llm_hyps: list[ScientificHypothesis] | None = None
        llm_proposals: list[dict[str, Any]] = []
        if self.agent is not None:
            traj = self.agent.call(
                prompt=("Propose competing mechanisms, observable predictions "
                        "and executable falsification conditions as JSON with "
                        "keys H1,H2,H3,H0. Research question: " + q.title
                        + "\nMotivation: " + q.motivation
                        + "\nGap: " + q.literature_gap
                        + "\nLiterature: " + json.dumps(refs)),
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
                candidate_reviewed = review_all(candidate)
                adopted = all(h.status == "ACCEPTED"
                              for h in candidate_reviewed)
                llm_proposals = [{
                    "proposal_id": f"llmprop_{traj.trajectory_id[:12]}_{h.kind}",
                    "kind": h.kind,
                    "trajectory_id": traj.trajectory_id,
                    "prompt_sha256": traj.prompt_sha256,
                    "model_id": traj.model_id,
                    "status": reviewed.status,
                    "critic_errors": reviewed.review_errors,
                    "critic_model": reviewed.critique.get("critic_model", ""),
                    "adopted": adopted,
                } for h, reviewed in zip(candidate, candidate_reviewed)]
                if adopted:
                    llm_hyps = candidate
            except (ValueError, TypeError):
                llm_proposals = [{
                    "proposal_id": "llmprop_parse_failure",
                    "kind": "*", "trajectory_id": traj.trajectory_id,
                    "prompt_sha256": traj.prompt_sha256,
                    "model_id": traj.model_id, "status": "REJECTED",
                    "critic_errors": ["llm_output_parse_failed"],
                    "adopted": False}]
        base = llm_hyps or original_hyps
        reviewed = review_all(base)
        accepted = [h for h in reviewed if h.status == "ACCEPTED"]
        competition = IndependentCritic().review_competition(reviewed)
        if len(accepted) < 4 or not competition["ok"]:
            raise DiscoveryBlocked("hypothesis_competition_incomplete")
        llm_used = any(p["adopted"] for p in llm_proposals)
        frozen = self._freeze("hypotheses", {
            "llm_proposed": bool(llm_proposals),
            "llm_used": llm_used,
            "llm_proposals": llm_proposals,
            "proposal_acceptance_rate": (
                sum(1 for p in llm_proposals if p["status"] == "ACCEPTED")
                / len(llm_proposals) if llm_proposals else None),
            "competition": competition,
            "hypotheses": [h.to_dict() for h in reviewed]})
        self.state["hypotheses"] = [h.to_dict() for h in reviewed]
        self.state["llm_decision_ledger"] = llm_proposals
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
            primary_metric="state_mse@1",
            mme=self.mme, pilot_effects=self.pilot_effects or None,
            seeds=self.seeds)
        frozen = self._freeze("experiment_protocol", protocol.to_dict())
        self.state["protocol"] = protocol.to_dict()
        power = protocol.power_analysis
        if power.get("pilot", {}).get("blocks_confirmation"):
            self.state["power_gate"] = "NOT_ASSESSED"
        return {"ok": True, "experiment_id": protocol.experiment_id,
                "experiment_sha": frozen["sha256"],
                "primary_metric": protocol.primary_metric,
                "mme": protocol.minimum_meaningful_effect,
                "power_status": power.get("pilot", {}).get("status")}

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
        # ARR v1.0 WP-01: explicit COMPOSITE binding. The trusted ApprovalStore
        # record binds the protocol SHA; a changed dataset/harness/method is
        # additionally caught by RuntimeState config_sha drift. We record a
        # composite digest so evidence consumers can verify ALL identities at
        # once and so any future schema migration has an authoritative value.
        method_sha = sha256_json({
            "mme": self.mme, "seeds": self.seeds,
            "selection_seeds": self.selection_seeds,
            "replication_seeds": self.replication_seeds,
            "domain_pack": self.domain})
        composite = sha256_json({
            "protocol_sha": protocol_sha,
            "harness_sha": self.harness_sha,
            "dataset_shas": self.dataset_shas,
            "method_sha": method_sha,
            "git_commit_sha": self.git_commit_sha})
        self.state["approval_composite_sha"] = composite
        self._stage("APPROVAL_BINDING", **{
            "protocol_sha": protocol_sha,
            "harness_sha": self.harness_sha,
            "dataset_shas": self.dataset_shas,
            "method_sha": method_sha,
            "git_commit_sha": self.git_commit_sha,
            "composite_sha": composite,
            "note": "approval_cli binds protocol_sha; composite records the "
                    "full identity set for downstream verification"})
        return {"ok": True, "protocol_sha": protocol_sha,
                "composite_binding_sha": composite, "method_sha": method_sha,
                **approval}

    def _run_experiment(self, *, experiment_id: str, stage: str,
                        arms: list[str], seeds: list[int], splits: list[str],
                        out_dir: str, round_no: int) -> dict[str, Any]:
        """Single execution funnel: formal adapter in production, engine in tests."""
        prereg = self.state["preregistration_sha"]
        if self.production_mode:
            if self.production_runtime is None:
                raise DiscoveryBlocked("production_runtime_missing")
            return self.production_runtime.dispatch(
                experiment_id=experiment_id, preregistration_sha=prereg,
                arms=arms, seeds=seeds, splits=splits, out_dir=out_dir,
                stage=stage, expected_harness_sha=self.harness_sha,
                config_sha=self.config_sha, round_no=round_no)
        # engine-free / development: explicit direct executor, recorded as such
        campaign_id = ("m7sel_" if stage == "EXPERIMENT_EXECUTION" else "m7rep_") \
            + prereg[:20]
        harness_binding = self._resolve_formal_harness(campaign_id)
        reservation = self._reserve(stage, campaign_id)
        try:
            result = self.executor(
                experiment_id=experiment_id, arms=arms, seeds=seeds,
                splits=splits, out_dir=out_dir)
        except Exception:
            # P0-B-4: settle the reservation on ANY execution failure.
            self.runtime.consume_reservation(self.run_id,
                self.state["reservations"][stage]["reservation_id"],
                outcome="failed")
            raise
        outcome = "succeeded" if result.get("ok") else "failed"
        self._consume_reservation(stage, outcome)
        return {"ok": True, "campaign_id": campaign_id,
                "reservation_id": reservation, "reservation_outcome": outcome,
                "harness_binding": harness_binding,
                "runtime_integration": self.state["runtime_integration"],
                **{k: v for k, v in result.items() if k != "summary"}}

    def _execute_selection(self) -> dict[str, Any]:
        protocol = self.state["protocol"]
        result = self._run_experiment(
            experiment_id=protocol["experiment_id"] + "_selection",
            stage="EXPERIMENT_EXECUTION",
            arms=["baseline", "action_conditioned", "capacity_control"],
            seeds=self.selection_seeds, splits=["validation"],
            out_dir=str(self.root / "experiments" / "selection"),
            round_no=int(self.state["stage_index"]))
        self.state["selection_campaign"] = result["campaign_id"]
        return {"ok": True, **result}

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
        result = self._run_experiment(
            experiment_id=protocol["experiment_id"] + "_replication",
            stage="REPLICATION",
            arms=["baseline", "action_conditioned"],
            seeds=self.replication_seeds, splits=["test", "ood"],
            out_dir=str(self.root / "experiments" / "replication"),
            round_no=int(self.state["stage_index"]))
        self.state["replication_campaign"] = result["campaign_id"]
        source = self.root / "experiments" / "replication" / "engine_results.json"
        raw = json.loads(source.read_text(encoding="utf-8"))
        from .domain_packs.embodied_world_model import evaluator
        integrity = evaluator.verify_result_shas({"summary": raw})
        evaluation = self.pack.evaluate(
            results=raw, primary_metric="state_mse@1", mme=self.mme)
        safety = self.pack.safety_check(results=raw)
        verdict = evaluation["per_arm"].get("action_conditioned", {}).get(
            "verdict", "INCONCLUSIVE")
        if self.state.get("power_gate") == "NOT_ASSESSED":
            verdict = "INCONCLUSIVE"
        evidence_sha = sha256_file(source)
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
        return {"ok": True, "replication_sha": evidence_sha,
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
        # ARR v1.0 ADR-003: EXPLICIT three-state mapping. INCONCLUSIVE must
        # NEVER be written as REFUTES -- uncertainty is not refutation, and a
        # false REFUTES edge poisons the VerifiedMemory promotion gate.
        def _selection_edge(v: str) -> str:
            return {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES"}.get(
                v, "INCONCLUSIVE_EVIDENCE")

        def _replication_edge(v: str) -> str:
            return {"SUPPORTS": "REPLICATES", "REFUTES": "REFUTES"}.get(
                v, "INCONCLUSIVE_EVIDENCE")

        self.graph.add_edge(source_id="ev_selection", target_id=claim_id,
                            relation=_selection_edge(selection),
                            evidence_sha=source_sha,
                            rationale=f"selection_verdict={selection}")
        self.graph.add_edge(source_id="ev_replication", target_id=claim_id,
                            relation=_replication_edge(replication),
                            evidence_sha=rep_sha,
                            rationale=f"replication_verdict={replication}")
        self.state["claim_edge_mapping"] = {
            "selection": {"verdict": selection, "edge": _selection_edge(selection)},
            "replication": {"verdict": replication,
                            "edge": _replication_edge(replication)}}
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
            # M7.6 P0-A/P0-B: a run that stopped at HUMAN_APPROVAL is a
            # two-phase out-of-band flow, not a failure. Phase 1 freezes the
            # preregistration and emits the approval request; a human grants it
            # out-of-band; phase 2 resumes. Only a PRODUCTION run that is still
            # blocked at the approval gate may be resumed, and only if the
            # trusted store now holds a matching approved record.
            if not (self.production_mode
                    and self.state.get("status") == "BLOCKED"
                    and (self.state.get("failure") or {}).get("stage")
                    == "HUMAN_APPROVAL"):
                return {"ok": True, "resumed": True,
                        "status": self.state["status"],
                        "scientific_verdict": self.state.get("scientific_verdict")}
            self.state["status"] = "RUNNING"
            self.state.pop("failure", None)
        # Bind the formal adapter to THIS run's M5 RuntimeState on EVERY run,
        # including resumes where DISCOVERY_START is already DONE and skipped.
        if self.production_mode and self.production_runtime is not None:
            self.production_runtime.runtime = self.runtime
            self.production_runtime.run_id = self.run_id
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
