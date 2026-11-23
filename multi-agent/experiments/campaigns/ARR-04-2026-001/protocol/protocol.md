# ARR-04 Protocol (frozen)

{
  "campaign_id": "ARR-04-2026-001",
  "experiment_id": "ARR_RESEARCH_CAPABILITY_04",
  "alpha": 0.05,
  "confidence": 0.95,
  "mme_normalized": 0.05,
  "independent_units": "task regimes (seeds are repeated measurements)",
  "primary_metric": "normalized heldout one_step_mse improvement",
  "statistical_tests": [
    "paired_t",
    "permutation",
    "cluster_bootstrap",
    "holm",
    "wilson",
    "effect_size"
  ],
  "diagnosis_gates": {
    "min_accuracy": 0.65,
    "max_ece": 0.15,
    "min_information_gain": 0.05
  },
  "research_selection_gates": {
    "mme": 0.05
  },
  "transfer_gates": {
    "mme": 0.05
  },
  "multiplicity": "Holm across primary contrasts per track",
  "stopping_rules": "budget cap; preflight STOP; no post-hoc sample changes",
  "exclusion": [
    "infrastructure_failure",
    "config_mismatch",
    "artifact_hash_mismatch",
    "gt_leak",
    "protocol_violation"
  ],
  "frozen_at": "2026-11-23T09:53:38",
  "benchmark_version": "arr04-benchmark-v1",
  "benchmark_identifiable": true,
  "frozen_task_manifest": [
    "insufficient_model_capacity__train__v0",
    "insufficient_model_capacity__dev__v1",
    "insufficient_model_capacity__heldout__v2",
    "insufficient_model_capacity__ood__v3",
    "insufficient_state_action_coverage__train__v0",
    "insufficient_state_action_coverage__dev__v1",
    "insufficient_state_action_coverage__heldout__v2",
    "insufficient_state_action_coverage__ood__v3",
    "dynamics_distribution_shift__train__v0",
    "dynamics_distribution_shift__dev__v1",
    "dynamics_distribution_shift__heldout__v2",
    "dynamics_distribution_shift__ood__v3",
    "optimization_instability__train__v0",
    "optimization_instability__dev__v1",
    "optimization_instability__heldout__v2",
    "optimization_instability__ood__v3",
    "action_conditioning_failure__train__v0",
    "action_conditioning_failure__dev__v1",
    "action_conditioning_failure__heldout__v2",
    "action_conditioning_failure__ood__v3"
  ],
  "protocol_sha256": "f155311b212bb5af991eab8c7a9688f7f3b43a27f9d4cfdf13f23b739d8e4a09"
}