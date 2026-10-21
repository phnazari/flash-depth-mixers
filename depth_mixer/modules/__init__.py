"""Shared building blocks. The fused norms, MLP, rotary embedding and losses are flash-linear-attention's."""

from fla.modules import (
    FusedCrossEntropyLoss,
    FusedLinearCrossEntropyLoss,
    GatedMLP,
    RMSNorm,
    RotaryEmbedding,
)

__all__ = [
    "FusedCrossEntropyLoss",
    "FusedLinearCrossEntropyLoss",
    "GatedMLP",
    "RMSNorm",
    "RotaryEmbedding",
]
