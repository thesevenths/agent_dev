"""Build the independent M7.2 Scientific Judgment benchmark corpus.

Cases are curated from public scientific knowledge and methodology, not generated
by ARR. Ground truth is written to a SEPARATE directory and never embedded in
case prompts. Expert blind review is explicitly marked NOT_PERFORMED unless an
external review file is supplied later.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research_runtime.harness_rsi.identity import freeze_json, sha256_json

VERSION = "m7_2_scientific_judgment_v1"

SPECS = [
    # 8: already well studied
    ("KNOWN", "classical_mechanics", "Can a small-angle pendulum period depend on amplitude?", "The period is already known to be approximately amplitude-independent in the small-angle limit.", ["textbook small-angle derivation"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("KNOWN", "classical_mechanics", "Does a simple mass-spring oscillator exhibit harmonic motion?", "Linear springs produce harmonic motion under Hooke's law.", ["textbook oscillator theory"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("KNOWN", "thermodynamics", "Does ideal gas pressure increase with temperature at fixed volume?", "Ideal gas law already answers this.", ["ideal gas law"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("KNOWN", "optics", "Does light refract at a planar interface according to Snell's law?", "Snell's law is established prior art.", ["geometric optics"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("KNOWN", "circuits", "Does an RC circuit show exponential charging?", "First-order RC response is textbook knowledge.", ["RC circuit theory"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("KNOWN", "population_biology", "Does logistic growth saturate at carrying capacity?", "Logistic model is established.", ["logistic population model"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("KNOWN", "electromagnetism", "Does a resistor dissipate power proportional to current squared?", "Joule heating is established.", ["Joule's law"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("KNOWN", "statistics", "Does sample mean variance decrease with sample size under IID sampling?", "Standard error theory already covers this.", ["central limit theorem"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    # 8: genuine open problems
    ("OPEN", "world_models", "Can action-conditioned latent dynamics generalize to unseen dynamics?", "OOD generalization of learned dynamics remains incomplete.", ["World Models (Ha & Schmidhuber, 2018)", "model-based RL surveys"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", False),
    ("OPEN", "world_models", "Can latent rollout error be bounded under distribution shift?", "No general tight bound is established.", ["latent dynamics literature"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", False),
    ("OPEN", "model_based_rl", "Can model-based planning remain sample-efficient with partial observability?", "Partial observability remains difficult.", ["POMDP literature"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", False),
    ("OPEN", "causal_dynamics", "Can invariant causal dynamics be identified from limited interventions?", "Identification conditions remain context dependent.", ["causal representation learning"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", False),
    ("OPEN", "sim2real", "Can world-model calibration transfer across simulator-reality gaps?", "Transfer remains unreliable.", ["sim-to-real surveys"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", False),
    ("OPEN", "time_series", "Can multi-step forecasting remain calibrated under regime change?", "Calibration under shift is unresolved.", ["conformal prediction under shift"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", False),
    ("OPEN", "robotics", "Can data-efficient dynamics learning work with unknown actuator saturation?", "Saturation-aware learning is not generally solved.", ["robot dynamics learning"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", False),
    ("OPEN", "physics_ml", "Can neural surrogates obey conservation laws out of distribution?", "Hard-constraint generalization remains open.", ["physics-informed machine learning"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", False),
    # 8: pseudo-novel with prior art
    ("PRIOR_ART", "world_models", "Is using action as an input to a dynamics model novel?", "Action-conditioned dynamics is established prior art.", ["World Models (2018)", "action-conditioned dynamics papers"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("PRIOR_ART", "world_models", "Is training a model with multi-step prediction loss novel?", "Multi-step rollout losses are well established.", ["model-based RL literature"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("PRIOR_ART", "rl", "Is reward prediction as an auxiliary task novel?", "Auxiliary reward prediction is established.", ["UNREAL, world models"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("PRIOR_ART", "representation", "Is contrastive learning of state representations novel?", "Contrastive representation learning has extensive prior art.", ["contrastive learning surveys"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("PRIOR_ART", "control", "Is model predictive control with a learned model novel?", "Learned MPC is established.", ["model-based RL and MPC"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("PRIOR_ART", "ood", "Is evaluation on shifted dynamics novel?", "OOD evaluation is standard methodology.", ["OOD benchmark literature"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("PRIOR_ART", "planning", "Is using latent rollouts for planning novel?", "Latent rollout planning is established.", ["World Models (2018)"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    ("PRIOR_ART", "calibration", "Is uncertainty calibration for neural dynamics novel?", "Calibration literature is extensive.", ["uncertainty in deep learning"], "NOT_NOVEL", "FALSIFIABLE", "SUFFICIENT", False),
    # 6: unfalsifiable hypotheses
    ("UNFALSIFIABLE", "metaphysics", "Does the universe possess an undetectable vital force?", "No observable consequence is specified.", ["Popper falsifiability criterion"], "NOVELTY_UNCERTAIN", "UNFALSIFIABLE", "SUFFICIENT", True),
    ("UNFALSIFIABLE", "ml", "Does the model understand the world truly, regardless of any measurement?", "No measurement is specified.", ["operationalism"], "NOVELTY_UNCERTAIN", "UNFALSIFIABLE", "SUFFICIENT", True),
    ("UNFALSIFIABLE", "cognition", "Is intelligence caused by an unobservable essence that always adapts?", "The claim explains any outcome.", ["ad hoc hypotheses"], "NOVELTY_UNCERTAIN", "UNFALSIFIABLE", "SUFFICIENT", True),
    ("UNFALSIFIABLE", "physics", "Does hidden variability exist that is identical under all observations?", "No possible refutation is specified.", ["empirical testability"], "NOVELTY_UNCERTAIN", "UNFALSIFIABLE", "SUFFICIENT", True),
    ("UNFALSIFIABLE", "robotics", "Is the controller better because it is more capable, with capability defined by performance?", "Circular definition.", ["circular reasoning"], "NOVELTY_UNCERTAIN", "UNFALSIFIABLE", "SUFFICIENT", True),
    ("UNFALSIFIABLE", "science_method", "Do all future experiments support the theory because the theory is true?", "Universal confirmation claim.", ["confirmation bias"], "NOVELTY_UNCERTAIN", "UNFALSIFIABLE", "SUFFICIENT", True),
    # 8: competing explanations
    ("COMPETING", "world_models", "Improvement may come from action conditioning, extra capacity, data conditioning, or seed variance.", "Need a discriminating capacity/control ablation.", ["controlled experimentation"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "SUFFICIENT", False),
    ("COMPETING", "dynamics", "Lower error may come from correct physics or from fitting the evaluation split.", "Need frozen holdout and split audit.", ["data leakage"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "SUFFICIENT", False),
    ("COMPETING", "rl", "Higher return may come from better model or reward hacking.", "Need reward and constraint audit.", ["specification gaming"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "SUFFICIENT", False),
    ("COMPETING", "calibration", "Better calibration may reflect thinner uncertainty or genuine posterior quality.", "Need proper scoring rule and OOD test.", ["calibration literature"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "SUFFICIENT", False),
    ("COMPETING", "planning", "Better planning may come from model accuracy or longer search budget.", "Need compute-matched control.", ["planning budgets"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "SUFFICIENT", False),
    ("COMPETING", "causal", "Observed invariance may reflect causal structure or dataset selection.", "Need intervention test.", ["causal inference"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "SUFFICIENT", False),
    ("COMPETING", "ood", "OOD gain may reflect true generalization or benchmark similarity.", "Need independent OOD families.", ["distribution shift"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "SUFFICIENT", False),
    ("COMPETING", "statistics", "Significant mean effect may arise from family clustering being ignored.", "Need family-level test.", ["clustered data"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "SUFFICIENT", False),
    # 6: contradictory evidence
    ("CONTRADICTORY", "world_models", "Two studies disagree on whether latent models improve OOD control.", "Study conditions differ; evidence conflict unresolved.", ["conflicting study reports"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "CONTRADICTORY", True),
    ("CONTRADICTORY", "nutrition", "Two cohorts disagree on an observational association.", "Confounding differs.", ["observational epidemiology"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "CONTRADICTORY", True),
    ("CONTRADICTORY", "medicine", "Two trials report opposite treatment effects.", "Populations and endpoints differ.", ["clinical trial heterogeneity"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "CONTRADICTORY", True),
    ("CONTRADICTORY", "climate", "Two datasets disagree after preprocessing changes.", "Preprocessing choice is a confound.", ["measurement methodology"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "CONTRADICTORY", True),
    ("CONTRADICTORY", "rl", "Two papers disagree on sample efficiency after compute normalization.", "Compute accounting differs.", ["reproducibility in RL"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "CONTRADICTORY", True),
    ("CONTRADICTORY", "vision", "Two benchmarks disagree on robustness rankings.", "Benchmark distribution differs.", ["benchmark bias"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "CONTRADICTORY", True),
    # 6: insufficient evidence
    ("INSUFFICIENT", "world_models", "A single seed shows improvement on validation split.", "No holdout, no family-level statistics.", ["single-seed evidence"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", True),
    ("INSUFFICIENT", "medicine", "A treatment appears effective in an uncontrolled case series.", "No control group.", ["case series limitations"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", True),
    ("INSUFFICIENT", "physics", "One sensor reading supports a new force law.", "No replication or uncertainty.", ["measurement replication"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", True),
    ("INSUFFICIENT", "economics", "A correlation is presented as causal using observational data.", "No identification strategy.", ["causal identification"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", True),
    ("INSUFFICIENT", "ml", "A model is claimed superior from one benchmark without confidence interval.", "No uncertainty quantification.", ["benchmark reporting"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", True),
    ("INSUFFICIENT", "robotics", "A robot succeeds in one environment episode.", "No task family or OOD condition.", ["robot evaluation protocol"], "NOVELTY_UNCERTAIN", "FALSIFIABLE", "INSUFFICIENT", True),
]


def main() -> None:
    root = Path("M7_2_BENCHMARK")
    cases_dir = root / "cases"
    gt_dir = root / "ground_truth"
    for d in (cases_dir, gt_dir):
        d.mkdir(parents=True, exist_ok=True)
    for i, spec in enumerate(SPECS, 1):
        category, domain, question, evidence_summary, refs, novelty, fals, suff, abstain = spec
        case_id = f"m72_{i:03d}"
        case = {"case_id": case_id, "version": VERSION, "category": category,
                "domain": domain, "question": question,
                "evidence_summary": evidence_summary,
                "source_references": refs,
                "source_policy": "CURATED_PUBLIC_KNOWLEDGE_NOT_LLM_GENERATED"}
        gt = {"case_id": case_id, "novelty": novelty,
              "falsifiability": fals, "evidence_sufficiency": suff,
              "abstention_expected": abstain,
              "reviewer_annotations": [{
                  "reviewer": "CURATOR",
                  "status": "PUBLIC_KNOWLEDGE_CURATION",
                  "independent_expert_blind_review": "NOT_PERFORMED"}],
              "ground_truth_sha256": ""}
        gt["ground_truth_sha256"] = sha256_json(
            {k: v for k, v in gt.items() if k != "ground_truth_sha256"})
        freeze_json(case, cases_dir / f"{case_id}.json")
        freeze_json(gt, gt_dir / f"{case_id}.json")
    manifest = {"version": VERSION, "n_cases": len(SPECS),
                "categories": {c: sum(1 for s in SPECS if s[0] == c)
                               for c in sorted({s[0] for s in SPECS})},
                "ground_truth_isolated": True,
                "expert_blind_review": "NOT_PERFORMED"}
    freeze_json(manifest, root / "benchmark_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
