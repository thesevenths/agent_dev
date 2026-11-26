"""
test_data.py
============
SciNet++ (2026) -- data-pipeline smoke test.

Quickly verifies, for BOTH generators (1D pendulum and chaotic double
pendulum), that:
    * the config passes ``validate_config`` (blocks vs. rollout horizons),
    * each dataset item has the expected block shape / dtype / label shape,
    * the finite-ness of the produced tensors,
    * the label ranges look sane.

Run:  python test_data.py
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from data.dataset import (
    build_dataset,
    encoder_input_dim,
    num_blocks,
    validate_config,
)
from utils.config import load_config


def check(config_path: str, n_samples: int = 8) -> None:
    """Validate the data pipeline for one config using a tiny sample count."""
    cfg = copy.deepcopy(load_config(config_path))
    cfg["data"]["train_samples"] = n_samples
    cfg["data"]["test_samples"] = n_samples
    validate_config(cfg)

    expected_blocks = (num_blocks(cfg), encoder_input_dim(cfg))
    generator = cfg["data"]["generator"]

    for split in ("train", "test"):
        dataset = build_dataset(cfg, split=split)
        assert len(dataset) == n_samples, f"{generator}/{split}: wrong length"
        assert dataset.label_names, f"{generator}: missing label_names"

        item = dataset[0]
        blocks, labels = item["blocks"], item["labels"]

        assert tuple(blocks.shape) == expected_blocks, (
            f"{generator}/{split}: blocks {tuple(blocks.shape)} != {expected_blocks}"
        )
        assert blocks.dtype == torch.float32, f"{generator}: blocks dtype"
        assert tuple(labels.shape) == (dataset.label_dim,), (
            f"{generator}/{split}: labels {tuple(labels.shape)}"
        )
        assert torch.isfinite(blocks).all(), f"{generator}/{split}: non-finite blocks"
        assert torch.isfinite(labels).all(), f"{generator}/{split}: non-finite labels"

    # report label ranges from the (small) train split
    all_labels = np.stack([dataset[i]["labels"].numpy() for i in range(len(dataset))])
    ranges = ", ".join(
        f"{name}=[{all_labels[:, j].min():.3f}, {all_labels[:, j].max():.3f}]"
        for j, name in enumerate(dataset.label_names)
    )
    print(
        f"[OK] {generator:16s} blocks={expected_blocks} "
        f"label_dim={dataset.label_dim}  {ranges}"
    )


if __name__ == "__main__":
    check("configs/pendulum.yaml")
    check("configs/double_pendulum.yaml")
    check("configs/newton.yaml")
    print("\nAll data smoke tests passed.")
