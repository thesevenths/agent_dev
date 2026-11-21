"""M7 Domain Pack (embodied): independent evaluation and safety gate.

The evaluator never trusts an in-memory claim: it recomputes metric deltas from
engine result artifacts and re-reads every result SHA from disk.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import yaml

from ....harness_rsi.identity import sha256_file
from . import metrics

POLICY_PATH = Path(__file__).with_name("safety_policy.yaml")


def load_safety_policy() -> dict[str, Any]:
    return yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))


def _summary(results: dict[str, Any]) -> dict[str, Any]:
    return results.get("summary", results)


def verify_result_shas(results: dict[str, Any]) -> dict[str, Any]:
    obj = _summary(results)
    problems: list[str] = []
    checked = 0
    for key, seed_result in obj["results"].items():
        entries = [("train", seed_result.get("train", {}))]
        for split, entry in (seed_result.get("evaluations") or {}).items():
            entries.append((split, entry))
        for stage, entry in entries:
            path, expected = entry.get("result_path"), entry.get("result_sha256")
            if not path or not expected or not Path(path).is_file():
                problems.append(f"missing_result:{key}:{stage}")
                continue
            if sha256_file(path) != expected:
                problems.append(f"result_sha_mismatch:{key}:{stage}")
            checked += 1
    return {"ok": not problems, "problems": problems, "n_checked": checked}


def _pair(obj: dict[str, Any], arm: str, seed: int, split: str,
          metric: str, mme: float) -> dict[str, Any] | None:
    results = obj["results"]
    base = results.get(f"baseline|s{seed}", {}).get("evaluations", {}).get(split)
    treat = results.get(f"{arm}|s{seed}", {}).get("evaluations", {}).get(split)
    if not base or not treat or base.get("status") != "succeeded" or \
            treat.get("status") != "succeeded":
        return None
    cmp = metrics.compare(base, treat, metric, mme)
    if not cmp["ok"]:
        return None
    return {**cmp, "seed": seed, "split": split, "arm": arm,
            "baseline_result_sha": base["result_sha256"],
            "treatment_result_sha": treat["result_sha256"]}


def evaluate(*, results: dict[str, Any], primary_metric: str = "state_mse@1",
             mme: float = 0.01, direction: str = "minimize") -> dict[str, Any]:
    obj = _summary(results)
    integrity = verify_result_shas(results)
    if not integrity["ok"]:
        return {"ok": False, "verdict": "INCONCLUSIVE",
                "reasons": integrity["problems"], "integrity": integrity}
    arms = [a for a in obj["arms"] if a != "baseline"]
    seeds, splits = obj["seeds"], obj["splits"]
    per_arm: dict[str, Any] = {}
    for arm in arms:
        pairs = [p for seed in seeds for split in splits
                 if (p := _pair(obj, arm, int(seed), split,
                                primary_metric, mme)) is not None]
        n_ok = len(pairs)
        expected = len(seeds) * len(splits)
        if n_ok < expected:
            per_arm[arm] = {"verdict": "INCONCLUSIVE",
                            "reason": f"only {n_ok}/{expected} valid pairs"}
            continue
        favorable_rate = sum(1 for p in pairs if p["favorable"]) / n_ok
        magnitudes = [p["magnitude"] for p in pairs]
        mean_magnitude = statistics.fmean(magnitudes)
        confound = False
        if arm == "action_conditioned" and "capacity_control" in arms:
            cap_pairs = [p for seed in seeds for split in splits
                         if (p := _pair(obj, "capacity_control", int(seed),
                                        split, primary_metric, mme))]
            if cap_pairs and statistics.fmean(
                    [p["magnitude"] for p in cap_pairs]) >= mean_magnitude:
                confound = True
        if favorable_rate >= 0.8 and mean_magnitude >= mme and not confound:
            verdict = "SUPPORTS"
        elif favorable_rate <= 0.2 and mean_magnitude >= mme:
            verdict = "REFUTES"
        else:
            verdict = "INCONCLUSIVE"
        per_arm[arm] = {
            "verdict": verdict, "n_pairs": n_ok, "favorable_rate": favorable_rate,
            "mean_magnitude": round(mean_magnitude, 6),
            "mme": mme, "direction": direction,
            "confound_not_excluded": confound,
            "pairs": pairs}
    return {"ok": True, "independent": True, "primary_metric": primary_metric,
            "integrity": integrity, "per_arm": per_arm,
            "note": "Threshold verdict only; statistical confirmation belongs "
                    "to family-level holdout/replication stages."}


def safety_check(*, results: dict[str, Any]) -> dict[str, Any]:
    obj = _summary(results)
    policy = load_safety_policy()
    violations: list[str] = []
    n = 0
    for key, seed_result in obj["results"].items():
        entries = [("train", seed_result.get("train", {}))]
        for split, entry in (seed_result.get("evaluations") or {}).items():
            entries.append((split, entry))
        for stage, entry in entries:
            n += 1
            if stage == "train":
                # Training evidence is required for provenance but evaluation
                # guardrails are defined only on rollout/evaluation stages.
                continue
            if entry.get("status") != "succeeded":
                violations.append(f"failed_job:{key}:{stage}")
                continue
            m = entry.get("metrics") or {}
            for name, rule in policy["hard_guardrails"].items():
                if name not in m:
                    violations.append(f"missing_guardrail:{key}:{stage}:{name}")
                    continue
                value = float(m[name])
                if "max" in rule and value > float(rule["max"]):
                    violations.append(f"guardrail_exceeded:{name}:{value}")
                if "min" in rule and value < float(rule["min"]):
                    violations.append(f"guardrail_below:{name}:{value}")
    return {"ok": not violations, "n_checked": n, "violations": violations,
            "policy_sha": sha256_file(POLICY_PATH)}


__all__ = ["load_safety_policy", "verify_result_shas", "evaluate",
           "safety_check"]
