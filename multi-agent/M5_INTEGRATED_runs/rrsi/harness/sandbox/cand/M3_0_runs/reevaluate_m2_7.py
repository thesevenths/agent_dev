"""M3.0 -- READ-ONLY re-evaluation of M2.7 historical verdicts.

This script NEVER writes to M2_7_runs/. It reads the persisted M2.7 candidate
registries, recomputes every domain candidate's verdict under the M3.0 frozen
statistical protocol (two-sided Student-t + Holm), and writes a NEW evidence
file under M3_0_runs/.

Two distinct questions are answered separately, because the review correctly
warned they must not be conflated:

  Q1  Is the OBSERVED metric improvement real?  -> Yes, historical on-disk
      metrics are untouched; this script does not dispute them.
  Q2  Does the candidate QUALIFY for a FORMAL statistical ACCEPT under the
      corrected protocol? -> Recomputed here. M2.7's lr=0.01 ACCEPT does NOT.

It also reports a clearly-labelled MOST-FAVOURABLE COUNTERFACTUAL (as if the
missing per-seed holdout evidence had all been same-direction). That is a
diagnostic, NOT a new verification -- even under that favourable assumption the
t+Holm qualification fails, so the conclusion does not depend on the missing
holdout fields.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research_runtime.evolution.evolution_controller import (  # noqa: E402
    DEFAULT_ALPHA, HOLDOUT_DIRECTION_MIN_FRACTION, _decide_domain,
    domain_family_significance,
)
from research_runtime.evolution.statistics import (  # noqa: E402
    paired_confidence_interval, paired_p_value, student_t_crit_two_sided,
)

HISTORY = REPO / "M2_7_runs"
OUT_DIR = REPO / "M3_0_runs"
OUT = OUT_DIR / "m2_7_reevaluation.json"
TERMINAL = {"accepted", "rejected", "inconclusive"}


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _old_decisions(events: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for e in events:
        if e.get("event") == "transition" and e.get("to") in TERMINAL:
            out[e.get("fingerprint", "")] = e["to"]
    return out


def _lookup(decisions: dict[str, str], fingerprint: str) -> str | None:
    if fingerprint in decisions:
        return decisions[fingerprint]
    return next((v for k, v in decisions.items()
                 if k.startswith(fingerprint)), None)


def _round_key(path: Path) -> str:
    return path.parent.name.replace("round_", "")


def main() -> dict:
    result: dict = {
        "protocol": "M3.0-student-t-holm",
        "protocol_details": {
            "family_alpha": DEFAULT_ALPHA,
            "t_critical_df4": student_t_crit_two_sided(0.05, 4),
            "normal_critical_used_by_M2_7": 1.96,
            "holdout_direction_min_fraction": HOLDOUT_DIRECTION_MIN_FRACTION,
            "bidirectional": True,
            "missing_statistics_fail_closed": True,
            "multiple_comparison": "holm-step-down (family = one round's candidates)",
        },
        "source": "M2_7_runs/*/round_*/candidate_registry.jsonl (READ ONLY)",
        "historical_artifacts_modified": False,
        "observed_metric_improvement_disputed": False,
        "counterfactual_note": (
            "counterfactual_most_favourable_holdout is a DIAGNOSTIC ONLY: it "
            "assumes the missing per-seed holdout results were all same-direction. "
            "It is NOT a new verification and NEVER changes a historical verdict."),
        "domains": {},
    }

    for dom in ("scinet", "embodied", "quant"):
        rounds: dict[str, list[dict]] = {}
        paths = sorted(glob.glob(str(HISTORY / dom / "round_*"
                                     / "candidate_registry.jsonl")))
        for p in paths:
            path = Path(p)
            events = _events(path)
            decisions = _old_decisions(events)
            records = [(e.get("record") or {}) for e in events
                       if e.get("event") == "transition"
                       and e.get("to") == "evaluated"]
            holm = domain_family_significance(records)
            entries: list[dict] = []
            for rec, h in zip(records, holm):
                n = rec.get("n_valid") or 0
                mean = rec.get("mean_improvement")
                std = rec.get("std_improvement")
                fp = rec.get("hyp_fingerprint", "")
                old = _lookup(decisions, fp)
                entry: dict = {
                    "target": rec.get("target"),
                    "new_value": rec.get("new"),
                    "old_decision": old,
                    "n_valid": n,
                    "mean_improvement": mean,
                    "std_improvement": std,
                    "baseline_mean": rec.get("baseline_mean"),
                    "holdout_improvement": rec.get("holdout_improvement"),
                    "holdout_n_valid": rec.get("holdout_n_valid"),
                }
                if (isinstance(mean, (int, float)) and math.isfinite(mean)
                        and isinstance(std, (int, float)) and math.isfinite(std)
                        and n >= 2):
                    lo_t, hi_t = paired_confidence_interval(mean, std, n)
                    lo_norm = mean - 1.96 * std / math.sqrt(n)
                    entry.update({
                        "M2_7_normal_ci95_lower": round(lo_norm, 8),
                        "M3_0_student_t_ci95_lower": round(lo_t, 8),
                        "M3_0_student_t_ci95_upper": round(hi_t, 8),
                        "uncorrected_two_sided_p": round(
                            paired_p_value(mean, std, n), 8),
                        "holm_rank": h["rank"],
                        "holm_effective_alpha": h["effective_alpha"],
                        "holm_significant": h["significant"],
                        "n_compared_in_family": h["n_compared"],
                        "statistical_qualification": (
                            ("significant_positive" if mean > 0
                             else "significant_negative")
                            if h["significant"] else "not_significant"),
                    })
                else:
                    entry["statistical_qualification"] = "insufficient_statistics"

                # Full M3.0 protocol on the HISTORICAL record (which lacks the
                # newly-required per-seed holdout evidence -> fail-closed).
                entry["M3_0_verdict_on_historical_evidence"] = _decide_domain(
                    rec, holm_alpha=h["effective_alpha"]).value

                # Diagnostic counterfactual: grant the most favourable holdout.
                cf = dict(rec)
                ho = cf.get("holdout_improvement")
                ho_n = cf.get("holdout_n_valid")
                if isinstance(ho, (int, float)) and math.isfinite(ho) and ho_n:
                    cf["holdout_improvements"] = [ho] * ho_n
                    cf["holdout_n_seeds"] = ho_n
                    cf["holdout_n_invalid"] = 0
                entry["counterfactual_most_favourable_holdout"] = _decide_domain(
                    cf, holm_alpha=h["effective_alpha"]).value

                # A historical terminal verdict that cannot be reproduced under
                # the corrected protocol is marked PROVISIONAL, never silently
                # re-accepted or rewritten.
                entry["provisional"] = bool(
                    old in TERMINAL
                    and entry["M3_0_verdict_on_historical_evidence"] != old
                    and entry["M3_0_verdict_on_historical_evidence"]
                    == "inconclusive")
                entries.append(entry)
            rounds[_round_key(path)] = entries
        result["domains"][dom] = rounds

    # Headline finding the review asked for.
    r1 = result["domains"].get("scinet", {}).get("1", [])
    best = next((e for e in r1 if e.get("new_value") == 0.01), None)
    result["headline"] = {
        "M2_7_scinet_round1_accepted": sum(
            1 for e in r1 if e.get("old_decision") == "accepted"),
        "M2_7_chosen_lr": 0.01,
        "chosen_lr_M2_7_normal_ci_lower": (best or {}).get(
            "M2_7_normal_ci95_lower"),
        "chosen_lr_M3_0_student_t_ci_lower": (best or {}).get(
            "M3_0_student_t_ci95_lower"),
        "chosen_lr_holm_significant": (best or {}).get("holm_significant"),
        "chosen_lr_counterfactual_verdict": (best or {}).get(
            "counterfactual_most_favourable_holdout"),
        "formal_statistical_accept_qualification": (
            "NOT_ESTABLISHED" if best and not best.get("holm_significant")
            else "RECOMPUTE_REQUIRED"),
        "observed_81_percent_metric_change": (
            "still a REAL observed metric change; NOT a statistically "
            "validated formal ACCEPT under M3.0"),
    }
    return result


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = main()
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                              sort_keys=True), encoding="utf-8")
    print(f"WROTE {OUT} (historical artifacts untouched)")
    h = report["headline"]
    print("\n=== HEADLINE: M2.7 SciNet++ round 1 under M3.0 ===")
    for k, v in h.items():
        print(f"  {k}: {v}")
    print("\n=== scinet round 1 per-candidate re-evaluation ===")
    for e in report["domains"]["scinet"]["1"]:
        print(f"  lr={e['new_value']!s:<7} old={e['old_decision']:<10} "
              f"norm_CI_lo={e.get('M2_7_normal_ci95_lower'):+.6f} "
              f"t_CI_lo={e.get('M3_0_student_t_ci95_lower'):+.6f} "
              f"p={e.get('uncorrected_two_sided_p'):.5f} "
              f"holm_sig={e.get('holm_significant')} "
              f"M3.0={e['M3_0_verdict_on_historical_evidence']:<13} "
              f"counterfactual={e['counterfactual_most_favourable_holdout']}")
    print("\n=== all domains: old -> M3.0 verdict distribution ===")
    import collections
    for dom, rounds in report["domains"].items():
        c = collections.Counter(
            (e["old_decision"], e["M3_0_verdict_on_historical_evidence"])
            for entries in rounds.values() for e in entries)
        print(f"  {dom}: {dict(c)}")
