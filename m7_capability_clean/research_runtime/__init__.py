"""
research_runtime
================
Engine-agnostic research task layer for the multi-agent runtime.

Boundary (deliberate)
---------------------
* This package knows NOTHING about PyTorch, SciNet++ model internals, LangGraph
  nodes or any specific scientific framework. It only knows structured
  ``ExperimentSpec`` / ``ExperimentResult`` and an injected ``ModelAdapter``.
* SciNet++ stays an independent repository. Its ``research_api`` CLI is the
  only contact point, invoked via subprocess with its own Python interpreter.
* ``adaptive_runtime`` (Harness self-optimisation) is untouched. Improving an
  Agent's prompts and improving a World Model are two different feedback loops
  with two different scoreboards; they must not be merged.

Future domains (finance / AI4S / robotics) plug in a new Domain Adapter and a
new ModelAdapter implementation -- nothing here needs to change.

Quick start::

    from research_runtime import SciNetAdapter, ExperimentManager, ResearchOrchestrator
    adapter = SciNetAdapter()
    manager = ExperimentManager(adapter)
    orchestrator = ResearchOrchestrator(manager)
    report = orchestrator.run_baseline(objective, config_path="configs/pendulum_smoke.yaml")
"""

from .schemas import (
    CandidateExperiment,
    ComparisonReport,
    ExperimentPlan,
    ExperimentResult,
    ExperimentSpec,
    FailureReport,
    Hypothesis,
    MetricPrediction,
    ModelVersion,
    ResearchObjective,
)

__all__ = [
    # P0 execution contract
    "ResearchObjective",
    "ExperimentSpec",
    "ExperimentResult",
    "FailureReport",
    "ModelVersion",
    # P1 research decision contract
    "MetricPrediction",
    "Hypothesis",
    "CandidateExperiment",
    "ExperimentPlan",
    "ComparisonReport",
]

__version__ = "1.0.0"
