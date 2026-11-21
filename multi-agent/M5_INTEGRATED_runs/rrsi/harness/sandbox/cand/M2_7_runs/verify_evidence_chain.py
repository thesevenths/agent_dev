"""M2.7 P1-1/P2 -- verify the REAL evidence chain of the three-domain loops.

Reads ONLY artifacts produced by the actual CLI loops (no re-computation, no
fabrication): frozen manifest SHA-256, materialised config SHA-256, checkpoint
SHA, real metrics, paired evidence, independent holdout evidence, the recorded
decision, version records and the per-round payload SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DOMAINS = {
    "scinet": ("pendulum_smoke_v1", "pendulum-a0e03386be89"),
    "embodied": ("robot_sim_v1", "robot_sim_v1-4979652dd897"),
    "quant": ("market_hist_v1", "market_hist_v1-7640575b236b"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    out: dict = {}
    for dom, (dsdir, version) in DOMAINS.items():
        proj = ROOT / dom
        entry: dict = {"dataset_version": version, "rounds": {}}
        manifest = proj / "datasets" / dsdir / "manifest.json"
        if manifest.is_file():
            m = json.loads(manifest.read_text(encoding="utf-8"))
            splits_ok = all(
                (proj / "datasets" / dsdir / str(info["file"])).is_file()
                and sha(proj / "datasets" / dsdir / str(info["file"])) == info["sha256"]
                for info in (m.get("splits") or {}).values())
            entry["dataset_manifest_sha256"] = sha(manifest)
            entry["dataset_version_on_disk"] = m.get("dataset_version")
            entry["dataset_all_split_sha_verified"] = splits_ok
            entry["dataset_version_matches"] = m.get("dataset_version") == version

        state_path = proj / "state" / "loop_state.json"
        if state_path.is_file():
            st = json.loads(state_path.read_text(encoding="utf-8"))
            entry["loop_status"] = st.get("status")
            entry["rounds_executed"] = st.get("last_completed_round")
            entry["last_round_payload_sha256"] = st.get("last_round_payload_sha256")
            entry["version_records"] = st.get("version_history", [])
            entry["accepted_domain_overrides"] = st.get("accepted_domain_overrides", {})

        for rd in sorted(proj.glob("round_*"), key=lambda p: p.name):
            rnum = rd.name.split("_")[1]
            reg = rd / "candidate_registry.jsonl"
            rinfo: dict = {}
            if reg.is_file():
                events = [json.loads(l) for l in reg.read_text(encoding="utf-8").splitlines() if l.strip()]
                decisions = Counter(e.get("to") for e in events
                                    if e.get("event") == "transition"
                                    and e.get("to") in ("accepted", "rejected", "inconclusive"))
                evals = [e.get("record") or {} for e in events
                         if e.get("event") == "transition" and e.get("to") == "evaluated"
                         and (e.get("record") or {}).get("candidate_type") == "domain_model"]
                rinfo["decisions"] = dict(decisions)
                rinfo["candidate_registry_sha256"] = sha(reg)
                rinfo["paired_evidence"] = [{
                    "target": r.get("target"),
                    "n_valid": r.get("n_valid"), "n_invalid": r.get("n_invalid"),
                    "mean_improvement": r.get("mean_improvement"),
                    "std_improvement": r.get("std_improvement"),
                    "baseline_mean": r.get("baseline_mean"),
                    "holdout_n_valid": r.get("holdout_n_valid"),
                    "holdout_improvement": r.get("holdout_improvement"),
                    "hyp_fingerprint": r.get("hyp_fingerprint"),
                } for r in evals]
            # Config + checkpoint + metric provenance from the materialised campaign.
            camp = proj / "outputs" / "research_campaigns" / f"dual_layer_r{rnum}"
            cfgs = list(camp.glob("*/config.yaml"))
            ckpts = list(camp.glob("*/checkpoint.json"))
            # Domain-agnostic: quant writes backtest_metrics.json, embodied
            # writes rollout_metrics.json. Never assume one engine's filename.
            mets = [p for p in camp.glob("*/*metrics*.json")]
            rinfo["materialised_configs"] = len(cfgs)
            rinfo["checkpoints"] = len(ckpts)
            rinfo["metric_files"] = len(mets)
            if cfgs:
                rinfo["sample_config_sha256"] = sha(cfgs[0])
            if ckpts:
                rinfo["sample_checkpoint_sha256"] = sha(ckpts[0])
            if mets:
                rinfo["sample_metric_file_sha256"] = sha(mets[0])
                try:
                    rinfo["sample_metrics"] = json.loads(mets[0].read_text(encoding="utf-8")).get("metrics")
                except (ValueError, OSError):
                    pass
            entry["rounds"][rnum] = rinfo
        out[dom] = entry

    dst = ROOT / "evidence_chain.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
