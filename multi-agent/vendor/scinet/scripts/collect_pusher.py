"""
scripts/collect_pusher.py
=========================
Collect a reproducible Pusher-v5 trajectory dataset.

    python scripts/collect_pusher.py --config configs/robotics/pusher_data.yaml
    python scripts/collect_pusher.py --config configs/robotics/pusher_data.yaml \
        --policy random --episodes 5        # quick smoke (5 random episodes)
    python scripts/collect_pusher.py --config configs/robotics/pusher_data.yaml \
        --resume                          # finish an interrupted collection

Default behaviour refuses to overwrite an existing dataset. Use --resume to
continue after an interruption; already-collected episodes are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect Pusher-v5 trajectory dataset")
    parser.add_argument("--config", required=True, type=Path, help="YAML collection config")
    parser.add_argument("--seed", type=int, default=None, help="Override dataset_seed")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override every policy's episode count (quick runs)")
    parser.add_argument("--policy", type=str, default=None,
                        help="Collect only this policy (random|guided|mixture)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output_dir")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-collected episodes instead of refusing")
    args = parser.parse_args(argv)

    # Lazy so the script still imports where gymnasium[mujoco] is absent.
    # Ensure the repo root (parent of scripts/) is importable as a package.
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import gymnasium
    import mujoco

    from robotics.data.dataset import collect_dataset, summarize
    from robotics.data.manifest import (
        build_manifest,
        verify_config_match,
        verify_manifest,
        write_manifest,
    )
    from robotics.data.policies import POLICY_VERSIONS

    config = _load_config(args.config)
    if args.seed is not None:
        config["dataset_seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.episodes is not None:
        config["collection"]["policies"] = {
            k: args.episodes for k in config["collection"]["policies"]
        }
    if args.policy is not None:
        config["collection"]["policies"] = {
            k: v for k, v in config["collection"]["policies"].items() if k == args.policy
        }

    output_dir = Path(config["output_dir"])
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists() and not args.resume:
        print(
            f"ERROR: {manifest_path} already exists. Refusing to overwrite.\n"
            f"       Use --resume to continue, or delete the dataset to start fresh.",
            file=sys.stderr,
        )
        return 2

    print(f"[collect] env={config['env']['id']} dataset_seed={config['dataset_seed']}")
    print(f"[collect] policies={config['collection']['policies']}")
    print(f"[collect] output_dir={output_dir}")

    written = collect_dataset(config, output_dir, resume=args.resume)

    # Build + verify manifest.
    manifest = build_manifest(
        config,
        output_dir,
        gym_version=gymnasium.__version__,
        mujoco_version=mujoco.__version__,
        policy_versions=dict(POLICY_VERSIONS),
    )
    # Cross-checks (must pass before we hand the dataset to any later stage).
    verify_config_match(manifest, manifest["dataset_version"])
    verify_manifest(manifest, output_dir)
    write_manifest(manifest_path, manifest)

    stats = summarize(output_dir)
    print("\n=== collection summary ===")
    print(json.dumps(stats, indent=2, default=str))
    print(f"\ndataset_version: {manifest['dataset_version']}")
    print(f"manifest: {manifest_path}")
    print(f"episodes written this run: {sum(0 if w['skipped'] else 1 for w in written)} "
          f"(skipped: {sum(1 if w['skipped'] else 0 for w in written)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
