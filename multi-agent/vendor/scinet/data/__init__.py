"""
data
====
SciNet++ (2026) data package.

Provides the physical trajectory generators (1D damped pendulum and chaotic
double pendulum) and the block-sequence dataset + config-driven factory used by
every stage of the pipeline.
"""

from .pendulum import PendulumGenerator
from .double_pendulum import DoublePendulumGenerator
from .newton import NewtonGenerator
from .dataset import (
    SPLITS,
    BlockSequenceDataset,
    build_dataset,
    build_generator,
    dataset_spec,
    dataset_version,
    derive_split_seed,
    encoder_input_dim,
    load_frozen_split,
    num_blocks,
    samples_for_split,
    split_rng,
    state_dim,
    validate_config,
)

__all__ = [
    "PendulumGenerator",
    "DoublePendulumGenerator",
    "NewtonGenerator",
    "SPLITS",
    "BlockSequenceDataset",
    "build_dataset",
    "build_generator",
    "dataset_spec",
    "dataset_version",
    "derive_split_seed",
    "encoder_input_dim",
    "load_frozen_split",
    "num_blocks",
    "samples_for_split",
    "split_rng",
    "state_dim",
    "validate_config",
]
