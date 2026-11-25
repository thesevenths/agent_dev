"""
models.encoder
==============
SciNet++ (2026) -- the concept extractor.

Maps ONE flattened block of a trajectory (``block_dim = block_size * state_dim``)
to a low-dimensional latent vector. These latent dimensions are the
*automatically discovered physical concepts* (SciNet) and the *online
representation* the V-JEPA predictor rolls forward.

Three interchangeable architectures are provided via ``kind``:

* ``"mlp"``        -- a plain fully-connected MLP over the flattened block.
* ``"conv"``       -- a temporal 1D-CNN. The flat block is reshaped to
  ``(state_dim, block_size)`` and treated as a short time series, so convolution
  filters capture local frequency content (-> omega) while the pooling + head
  capture the amplitude envelope trend (-> gamma). A strong, data-efficient
  inductive bias for sequences.
* ``"transformer"``-- a small Transformer encoder. The block becomes a sequence
  of ``block_size`` tokens (each ``state_dim``-dim) plus sinusoidal positional
  encoding; global self-attention can read the exponential decay envelope across
  the *whole* window (-> gamma) and the periodic structure (-> omega) without the
  information loss that pooling introduces. More expressive but more data-hungry.

CRITICAL INVARIANT: whichever ``kind`` is used, the encoder keeps the same
contract -- input ``(B, block_dim)`` flat tensor, output ``(B, latent_dim)``.
Because every downstream stage (predictor, physics probe, symbolic regression,
evaluate, tta) depends ONLY on this latent interface and never on the encoder
internals, swapping mlp <-> conv <-> transformer requires NO change anywhere else.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


def build_mlp(dims: Sequence[int], activation: nn.Module = nn.ReLU) -> nn.Sequential:
    """Assemble a plain MLP from a list of layer sizes ``[in, h1, ..., out]``."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # no activation after the final projection
            layers.append(activation())
    return nn.Sequential(*layers)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding of shape ``(1, max_len, d_model)``.

    Position matters here: the phase offset phi and the angular frequency omega
    of a block are only recoverable if the encoder knows the order of samples,
    which a bag-of-tokens attention would otherwise discard.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: pe[0, :, 1::2].shape[-1]])
        # Buffer (not a parameter): saved in state_dict, EMA-copied, never trained.
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: (B, L, d_model)
        return x + self.pe[:, : x.size(1), :]


class Encoder(nn.Module):
    """Block -> latent concept vector (MLP / temporal-CNN / Transformer)."""

    def __init__(
        self,
        block_size: int,
        state_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        latent_dim: int = 3,
        kind: str = "mlp",
        conv_channels: Sequence[int] = (64, 128, 128),
        pool_k: int = 4,
        tf_d_model: int = 64,
        tf_heads: int = 4,
        tf_layers: int = 2,
        tf_ff_mult: int = 2,
        tf_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.block_size = block_size
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.input_dim = block_size * state_dim
        self.pool_k = pool_k

        if kind == "mlp":
            self.net = build_mlp([self.input_dim, *hidden_dims, latent_dim])
        elif kind == "conv":
            chans = [state_dim, *conv_channels]
            conv_layers: list[nn.Module] = []
            for i in range(len(chans) - 1):
                stride = 2 if i == 0 else 1
                conv_layers += [
                    nn.Conv1d(chans[i], chans[i + 1], kernel_size=3,
                              stride=stride, padding=1),
                    nn.ReLU(),
                ]
            self.conv = nn.Sequential(*conv_layers)
            # AdaptiveAvgPool makes the encoder length-agnostic (works for any
            # block_size) while keeping pool_k coarse temporal segments so the
            # decay envelope (gamma) survives pooling.
            self.pool = nn.AdaptiveAvgPool1d(pool_k)
            self.head = build_mlp([chans[-1] * pool_k, *hidden_dims, latent_dim])
        elif kind == "transformer":
            self.in_proj = nn.Linear(state_dim, tf_d_model)
            self.pos = SinusoidalPositionalEncoding(tf_d_model, block_size)
            self.drop = nn.Dropout(tf_dropout)
            layer = nn.TransformerEncoderLayer(
                d_model=tf_d_model,
                nhead=tf_heads,
                dim_feedforward=tf_ff_mult * tf_d_model,
                dropout=tf_dropout,
                activation="relu",
                batch_first=True,
            )
            self.tf = nn.TransformerEncoder(layer, num_layers=tf_layers)
            self.head = build_mlp([tf_d_model, *hidden_dims, latent_dim])
        else:
            raise ValueError(
                f"Unknown encoder kind '{kind}'. Use 'mlp', 'conv' or 'transformer'."
            )

    def forward(self, x):
        # x: (B, block_dim) flat -- kept identical for all kinds (the contract).
        if self.kind == "conv":
            x = x.reshape(-1, self.state_dim, self.block_size)  # (B, C, L)
            h = self.pool(self.conv(x)).flatten(1)              # (B, Cn*pool_k)
            return self.head(h)
        if self.kind == "transformer":
            tok = x.reshape(-1, self.block_size, self.state_dim)  # (B, L, C)
            tok = self.drop(self.pos(self.in_proj(tok)))          # (B, L, d_model)
            h = self.tf(tok).mean(dim=1)                          # (B, d_model)
            return self.head(h)
        return self.net(x)
