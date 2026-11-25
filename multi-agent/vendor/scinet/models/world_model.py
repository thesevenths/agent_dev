"""
models.world_model
==================
SciNet++ (2026) -- the V-JEPA world model.

Assembles the three pieces of a modern Joint-Embedding Predictive Architecture:

* ``encoder``        -- online representation encoder (trained by gradient).
* ``predictor``      -- latent transition model f, rolled for multi-step prediction.
* ``target_encoder`` -- an EMA (exponential moving average) copy of the online
  encoder that produces the prediction *targets* and receives no gradient.

Prediction happens purely in latent space (no pixel/signal reconstruction), and
the EMA target encoder together with the VICReg variance/covariance penalty in
``utils.losses`` prevents the trivial collapsed solution that a shared-encoder +
plain-MSE setup (the previous version of this file) inevitably falls into.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable

import torch
import torch.nn as nn

from .encoder import Encoder
from .predictor import Predictor


class WorldModel(nn.Module):
    """V-JEPA: online encoder + latent predictor + EMA target encoder."""

    def __init__(
        self,
        block_size: int,
        state_dim: int,
        latent_dim: int = 3,
        encoder_hidden: Iterable[int] = (64, 32),
        predictor_hidden: Iterable[int] = (64, 64),
        encoder_type: str = "mlp",
        conv_channels: Iterable[int] = (64, 128, 128),
        tf_d_model: int = 64,
        tf_heads: int = 4,
        tf_layers: int = 2,
        predictor_type: str = "mlp",
        pred_tf_heads: int = 2,
        pred_tf_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            block_size, state_dim, encoder_hidden, latent_dim,
            kind=encoder_type, conv_channels=conv_channels,
            tf_d_model=tf_d_model, tf_heads=tf_heads, tf_layers=tf_layers,
        )
        self.predictor = Predictor(
            latent_dim, predictor_hidden,
            kind=predictor_type,
            tf_heads=pred_tf_heads, tf_layers=pred_tf_layers,
        )

        # EMA target encoder: a frozen copy updated by momentum, never by grad.
        # This is the key V-JEPA/BYOL trick: the target provides stable prediction
        # targets without gradient, preventing the online encoder from chasing its own tail.
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)  # exclude from optimizer (train.py filters on this)

    # ------------------------------------------------------------- encoders --
    def encode(self, x):
        """Online encoding (differentiable)."""
        return self.encoder(x)

    @torch.no_grad()
    def encode_target(self, x):
        """Target encoding via the EMA encoder (no gradient)."""
        return self.target_encoder(x)

    # ------------------------------------------------------------ prediction --
    def predict_multistep(self, z0, horizons: Iterable[int]) -> Dict[int, torch.Tensor]:
        """Roll the predictor f autoregressively: z_{b+k} = f^k(z_b).

        The SAME predictor is applied k times. This is critical: it means the
        model learns a *transition function* in latent space, not separate
        predictors per horizon. Multi-step accuracy tests whether the latent
        dynamics are truly learned (not just memorised per-horizon).
        """
        horizons = sorted(set(int(h) for h in horizons))
        preds: Dict[int, torch.Tensor] = {}
        z = z0
        for step in range(1, max(horizons) + 1):
            z = self.predictor(z)       # one step of latent dynamics
            if step in horizons:
                preds[step] = z         # save only requested horizons
        return preds

    def forward(self, blocks, horizons: Iterable[int], context_block: int = 0):
        """Multi-step latent prediction for a batch of block sequences.

        Parameters
        ----------
        blocks : Tensor, shape (batch, num_blocks, block_dim)
        horizons : iterable of int, prediction horizons in blocks (e.g. [1,2,4,8])
        context_block : int, index of the block used as the prediction context

        Returns
        -------
        preds : dict {horizon: predicted latent}        (online, differentiable)
        targets : dict {horizon: EMA target latent}      (no gradient)
        z_context : Tensor, online latent of the context block (for VICReg)
        """
        horizons = sorted(set(int(h) for h in horizons))
        # Online encoder processes the context block (differentiable path)
        z_context = self.encode(blocks[:, context_block, :])
        # Roll predictor forward from context latent to get predictions at each horizon
        preds = self.predict_multistep(z_context, horizons)

        # Target encoder provides ground-truth latents (no gradient flows here)
        with torch.no_grad():
            targets = {
                k: self.encode_target(blocks[:, context_block + k, :]) for k in horizons
            }
        return preds, targets, z_context

    # ------------------------------------------------------------------- EMA --
    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """EMA update: target <- momentum * target + (1 - momentum) * online."""
        for p_online, p_target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            p_target.data.mul_(momentum).add_((1.0 - momentum) * p_online.data)

    # ------------------------------------------------------------- factories --
    @classmethod
    def from_config(cls, cfg: dict, state_dim: int) -> "WorldModel":
        """Build a WorldModel from a resolved config dict.

        ``state_dim`` (per-timestep feature count) comes from the selected
        generator; ``block_size`` and all model shapes come from the config.
        """
        model_cfg = cfg["model"]
        return cls(
            block_size=cfg["data"]["block_size"],
            state_dim=state_dim,
            latent_dim=model_cfg["latent_dim"],
            encoder_hidden=model_cfg["encoder_hidden"],
            predictor_hidden=model_cfg["predictor_hidden"],
            encoder_type=model_cfg.get("encoder_type", "mlp"),
            conv_channels=model_cfg.get("conv_channels", [64, 128, 128]),
            tf_d_model=model_cfg.get("tf_d_model", 64),
            tf_heads=model_cfg.get("tf_heads", 4),
            tf_layers=model_cfg.get("tf_layers", 2),
            predictor_type=model_cfg.get("predictor_type", "mlp"),
            pred_tf_heads=model_cfg.get("pred_tf_heads", 2),
            pred_tf_layers=model_cfg.get("pred_tf_layers", 2),
        )
