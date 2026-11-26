"""
models.predictor
================
SciNet++ (2026) -- the V-JEPA latent transition model.

A configurable transition function ``f`` that predicts the latent representation
one block into the future:  z_{b+1} = f(z_b). Multi-step prediction is obtained
by rolling the *same* ``f`` autoregressively (z_{b+k} = f^k(z_b)), which is
exactly how the model is trained (train.py) and evaluated (evaluate.py) -- the
JEPA idea of predicting latent representations instead of reconstructing raw
signal.

Two interchangeable architectures are provided via ``kind``:

* ``"mlp"``         -- a plain fully-connected MLP (default). Simple, fast, and
  well-suited because the predictor maps a low-dim latent vector to another
  low-dim latent vector with no sequential structure.
* ``"transformer"`` -- a small Transformer encoder applied to the latent vector
  treated as a single-token sequence. Note: self-attention over one token
  degenerates to an identity operation, so this is functionally similar to an
  MLP with LayerNorm. Provided as a config-switchable option for experimentation;
  no inductive-bias advantage is expected over MLP for this role.

CRITICAL INVARIANT: whichever ``kind`` is used, the predictor keeps the same
contract -- input ``(B, latent_dim)``, output ``(B, latent_dim)``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from .encoder import build_mlp


class Predictor(nn.Module):
    """Latent transition model: z_b -> z_{b+1} (MLP or Transformer)."""

    def __init__(
        self,
        latent_dim: int = 3,
        hidden_dims: Sequence[int] = (64, 64),
        kind: str = "mlp",
        tf_heads: int = 2,
        tf_layers: int = 2,
        tf_ff_mult: int = 2,
        tf_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.latent_dim = latent_dim

        if kind == "mlp":
            self.net = build_mlp([latent_dim, *hidden_dims, latent_dim])
        elif kind == "transformer":
            # Single-token transformer: attention is identity, but LayerNorm +
            # FFN provide a regularised nonlinear map. We add a learnable
            # positional embedding (length 1) for API completeness.
            d_model = latent_dim  # keep dim consistent (no projection needed)
            self.pos_embed = nn.Parameter(torch.zeros(1, 1, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=tf_heads,
                dim_feedforward=tf_ff_mult * d_model,
                dropout=tf_dropout,
                activation="relu",
                batch_first=True,
            )
            self.tf = nn.TransformerEncoder(layer, num_layers=tf_layers)
            # Final norm to match typical transformer output conventions.
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(
                f"Unknown predictor kind '{kind}'. Use 'mlp' or 'transformer'."
            )

    def forward(self, z):
        # z: (B, latent_dim)
        if self.kind == "transformer":
            tok = z.unsqueeze(1) + self.pos_embed   # (B, 1, latent_dim)
            h = self.tf(tok).squeeze(1)              # (B, latent_dim)
            return self.norm(h)
        return self.net(z)
