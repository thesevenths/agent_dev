"""
models.probe
============
SciNet++ (2026) -- the "physics probe" for concept discovery.

A small MLP trained *on frozen latents* to regress the ground-truth physical
concepts (gamma, omega for the pendulum; total energy for the double pendulum).
If a low-dimensional probe can recover these quantities from the encoder's
latent space, the encoder has genuinely discovered the underlying physical
concept -- this is the SciNet "read-out" test, and it is what feeds the
AI-Feynman symbolic-regression stage.

All dimensions are configurable (no hardcoded 3 -> 2), so the same probe works
for any latent size and any number of target concepts.
"""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn

from .encoder import build_mlp


class Probe(nn.Module):
    """MLP read-out: frozen latent -> physical concept(s)."""

    def __init__(
        self,
        latent_dim: int = 3,
        hidden_dims: Iterable[int] = (32, 16),
        out_dim: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.net = build_mlp([latent_dim, *hidden_dims, out_dim])

    def forward(self, z):
        return self.net(z)

    @classmethod
    def from_config(cls, cfg: dict, latent_dim: int, out_dim: int) -> "Probe":
        """Build a Probe from a resolved config dict."""
        return cls(
            latent_dim=latent_dim,
            hidden_dims=cfg["probe"]["hidden"],
            out_dim=out_dim,
        )
