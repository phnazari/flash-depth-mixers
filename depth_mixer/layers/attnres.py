"""Attention Residuals (AttnRes): each sublayer reads a softmax mixture of earlier residual sources
instead of their sum. See `Attention Residuals <https://arxiv.org/abs/2603.15031>`_.

A source is the sum of the branch outputs of one depth block of ``block_size`` sublayers, the
embedding being the first; ``block_size=1`` (Full AttnRes) makes every branch output a source. Each
sublayer scores the RMS-normalized sources against its learned pseudo-query, mixes them with the
softmax over depth and applies its pre-norm to the mixture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from depth_mixer.modules import RMSNorm
from depth_mixer.ops.attnres import fused_attnres, naive_attnres

ATTNRES_MODES = ("fused", "naive")


class AttentionResidual(nn.Module):
    """The pseudo-query ``proj`` (zero at init, a uniform mixture) and the source norm of one sublayer."""

    def __init__(self, hidden_size: int, norm_eps: float = 1e-6, mode: str = "fused"):
        super().__init__()
        if mode not in ATTNRES_MODES:
            raise ValueError(f"unknown AttnRes mode {mode!r}; expected one of {ATTNRES_MODES}")
        self.mode = mode
        self.proj = nn.Linear(hidden_size, 1, bias=False)
        self.norm = RMSNorm(hidden_size, eps=norm_eps)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)

    def forward(
        self,
        residuals: list[torch.Tensor],
        output_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        """The mixture of ``residuals``, followed by ``output_norm`` (fused into the kernel) if given."""
        aggregate = fused_attnres if self.mode == "fused" else naive_attnres
        return aggregate(
            query=self.proj.weight,
            residuals=residuals,
            rms_weight=self.norm.weight,
            output_rms_weight=None if output_norm is None else output_norm.weight,
            key_rms_eps=self.norm.eps,
            output_rms_eps=None if output_norm is None else output_norm.eps,
        )


class AttnResState:
    """The sources of one forward pass: the completed depth blocks and the partial sum of the
    current one. :meth:`sources` lists what a sublayer mixes and closes the block at a boundary;
    :meth:`add` accumulates a branch output into the open block."""

    def __init__(self, embedding: torch.Tensor):
        self.blocks: list[torch.Tensor] = [embedding]
        self.partial: torch.Tensor | None = None

    def sources(self, boundary: bool) -> list[torch.Tensor]:
        residuals = self.blocks if self.partial is None else [*self.blocks, self.partial]
        if boundary and self.partial is not None:
            self.blocks, self.partial = residuals, None
        return residuals

    def add(self, delta: torch.Tensor) -> torch.Tensor:
        """Add ``delta`` to the open block; returns the open block's sum."""
        self.partial = delta if self.partial is None else self.partial + delta
        return self.partial
