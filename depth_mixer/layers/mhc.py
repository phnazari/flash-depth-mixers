"""mHC: manifold-constrained Hyper-Connections (DeepSeek), on the residual stream of a transformer.

The residual is widened into ``n`` streams ``[..., n, D]``. Around each branch (attention or MLP) a
connector's controller reads the RMS-normalized streams and produces three mixes: ``pre`` combines
the streams into the branch input, ``post`` spreads the branch output back onto the streams and the
doubly stochastic ``comb`` (Sinkhorn-Knopp) mixes the streams among themselves::

    x = sum_m pre_m h_m;   y = branch(norm(x));   h'_n = post_n y + sum_m comb[m, n] h_m

A learned head contracts the streams into one before the final norm. The controller and head
parameters are FP32; the streams keep the model dtype. ``fused`` runs DeepSeek's TileLang kernels
(BF16 streams, 4 streams, width divisible by 64); ``naive`` runs the PyTorch reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from depth_mixer.ops.mhc import (
    expand_to_mhc,
    fused_mhc_head,
    fused_mhc_post,
    fused_mhc_pre,
    fused_mhc_pre_inference,
    naive_mhc_head,
    naive_mhc_post,
    naive_mhc_pre,
)

MHC_MODES = ("naive", "fused")

# The mHC recipe (Appendix A of the mHC paper): pre-mix floor, Sinkhorn epsilon and post multiplier.
MHC_PRE_EPS = 1e-6
MHC_SINKHORN_EPS = 1e-6
MHC_POST_MULT = 2.0


def _init_copy_(parameter: nn.Parameter, value: torch.Tensor) -> None:
    """Copy a structured initial value, unless transformers loaded this tensor from a checkpoint
    (the flag its guarded ``nn.init`` functions check)."""
    if not getattr(parameter, "_is_hf_initialized", False):
        with torch.no_grad():
            parameter.copy_(value)


def _use_fused(mode: str, streams: torch.Tensor) -> bool:
    """The fused kernels run on CUDA with BF16 streams, 4 streams and a width divisible by 64."""
    return (
        mode == "fused" and streams.is_cuda and streams.dtype == torch.bfloat16
        and streams.shape[-2] == 4 and streams.shape[-1] % 64 == 0
    )


def _inference_splits(streams: torch.Tensor) -> int | None:
    """None (the projection split count tuned for this GPU type) when a table covers this width, else
    upstream's single split. The split count only changes the FP32 summation order."""
    from depth_mixer.ops.mhc.tile_kernels.modeling.mhc.ops.pre_big_fuse import _table, table_path

    n, d = streams.shape[-2:]
    return None if f"{n}x{d}" in _table(table_path(streams.device)) else 1


class ManifoldHyperConnection(nn.Module):
    """One connector: the controller around a branch, in DeepSeek TileKernels' parameter layout.

    ``fn`` ``[n^2 + 2n, n D]`` projects the RMS-normalized streams to the mixes (rows: pre, post,
    residual), ``norm_weight`` ``[n D]`` is that RMS norm's weight, ``b`` ``[n^2 + 2n]`` the mix biases
    and ``alpha`` ``[3]`` the gating factors of the pre, post and residual mixes. The initialization
    is Algorithm 1 of the mHC paper: zero projection, the selected stream's pre bias and the residual
    diagonal at +3, the other pre and residual biases at -3, zero post biases. ``layer_idx`` is the
    connector's index (two per block); it selects stream ``layer_idx % n``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        gating_factor_init: float = 0.01,
        norm_eps: float = 1e-6,
        sinkhorn_iters: int = 20,
        mode: str = "fused",
        layer_idx: int = 0,
    ):
        super().__init__()
        if mode not in MHC_MODES:
            raise ValueError(f"unknown mHC mode {mode!r}; expected one of {MHC_MODES}")
        n = num_streams
        self.num_streams = n
        self.hidden_size = hidden_size
        self.gating_factor_init = gating_factor_init
        self.norm_eps = norm_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.mode = mode
        self.layer_idx = layer_idx
        self.selected_stream = layer_idx % n
        self.fn = nn.Parameter(torch.zeros(n * n + 2 * n, n * hidden_size, dtype=torch.float32))
        self.norm_weight = nn.Parameter(torch.ones(n * hidden_size, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(n * n + 2 * n, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.full((3,), float(gating_factor_init), dtype=torch.float32))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        n = self.num_streams
        nn.init.zeros_(self.fn)
        nn.init.ones_(self.norm_weight)
        nn.init.constant_(self.alpha, self.gating_factor_init)
        b = torch.zeros_like(self.b)
        b[:n] = -3.0
        b[self.selected_stream] = 3.0
        residual = b[2 * n:].view(n, n)
        residual.fill_(-3.0)
        residual.diagonal().fill_(3.0)
        _init_copy_(self.b, b)

    def _controller(self) -> dict:
        # .float() is a no-op on the FP32 parameters and keeps a model cast to BF16 working.
        return dict(
            norm_weight=self.norm_weight.float(), norm_eps=self.norm_eps, mhc_mult=self.num_streams,
            post_mult_value=MHC_POST_MULT, pre_eps=MHC_PRE_EPS, sinkhorn_eps=MHC_SINKHORN_EPS,
            sinkhorn_repeat=self.sinkhorn_iters,
        )

    def _mixing(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.fn.float(), self.alpha.float(), self.b.float()

    def pre(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The branch input ``[..., D]`` and the ``post`` ``[..., n, 1]`` and ``comb`` ``[..., n, n]``
        mixes that :meth:`post` consumes."""
        if not _use_fused(self.mode, streams):
            layer_input, _, post_mix, comb = naive_mhc_pre(streams, *self._mixing(), **self._controller())
            return layer_input, post_mix, comb
        streams = streams.contiguous()
        if not torch.is_grad_enabled():
            return fused_mhc_pre_inference(
                streams, *self._mixing(), n_splits=_inference_splits(streams), **self._controller(),
            )
        layer_input, _, post_mix, comb = fused_mhc_pre(streams, *self._mixing(), **self._controller())
        return layer_input, post_mix, comb

    def post(
        self,
        branch_output: torch.Tensor,
        streams: torch.Tensor,
        post_mix: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """The streams after the branch, ``[..., n, D]``."""
        branch_output = branch_output.to(streams.dtype)
        if not _use_fused(self.mode, streams):
            return naive_mhc_post(branch_output, streams, post_mix, comb)
        return fused_mhc_post(branch_output.contiguous(), streams.contiguous(), post_mix, comb)


class MHCHead(nn.Module):
    """The learned contraction of the streams before the final norm (DeepSeek-V4 ``hc_head``):
    independent sigmoid gates per stream from the RMS-normalized streams. ``weight`` ``[n, n D]`` is
    initialized ``Normal(0.02)``, ``bias`` ``[n]`` zero, ``scale`` ``[1]`` 0.01."""

    def __init__(
        self,
        hidden_size: int,
        num_streams: int = 4,
        norm_eps: float = 1e-6,
        mode: str = "fused",
        initializer_range: float = 0.02,
    ):
        super().__init__()
        n = num_streams
        self.num_streams = n
        self.norm_eps = norm_eps
        self.mode = mode
        self.initializer_range = initializer_range
        self.weight = nn.Parameter(torch.empty(n, n * hidden_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(n, dtype=torch.float32))
        self.scale = nn.Parameter(torch.full((1,), 0.01, dtype=torch.float32))
        self.norm_weight = nn.Parameter(torch.ones(n * hidden_size, dtype=torch.float32))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self.initializer_range)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.scale, 0.01)
        nn.init.ones_(self.norm_weight)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        """``[..., n, D]`` streams to ``[..., D]``."""
        kwargs = dict(norm_weight=self.norm_weight.float(), norm_eps=self.norm_eps, pre_eps=MHC_PRE_EPS)
        weight, scale, bias = self.weight.float(), self.scale.float(), self.bias.float()
        if not _use_fused(self.mode, streams):
            return naive_mhc_head(streams, weight, scale, bias, **kwargs)
        return fused_mhc_head(streams.contiguous(), weight, scale, bias, mhc_mult=self.num_streams, **kwargs)


def expand_streams(hidden: torch.Tensor, num_streams: int) -> torch.Tensor:
    """The embedding repeated into ``num_streams`` streams, ``[..., n, D]``."""
    return expand_to_mhc(hidden, num_streams)
