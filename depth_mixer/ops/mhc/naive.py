"""Naive PyTorch mHC: DeepSeek TileKernels' references (``tile_kernels/torch/mhc.py``) composed as the
fused ops compose them.

Shapes: ``streams`` is ``[..., n, D]``; a connector's projection ``fn`` is ``[n^2 + 2n, n D]`` with rows
(pre (n), post (n), residual (n x n)), its bias ``b`` ``[n^2 + 2n]`` and its gating factors ``alpha``
``[3]`` (pre, post, residual); the head's projection is ``[n, n D]``. The controller math is FP32;
the fused kernels additionally round the projection to TF32.
"""

from __future__ import annotations

import torch


def mhc_pre_norm_fn(
    streams: torch.Tensor,
    fn: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    """The projection of the RMS-normalized flattened streams, ``[..., M]`` in FP32. The RMS weight is
    merged into the projection and the RMS is applied after the matmul, as the kernels do."""
    x = streams.flatten(-2).float()
    fn = fn.float() if norm_weight is None else fn.float() * norm_weight.float()
    mixes = x @ fn.t()
    return mixes * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)


def mhc_pre_split_mixes(
    mixes: torch.Tensor,
    alpha: torch.Tensor,
    b: torch.Tensor,
    mhc_mult: int,
    post_mult_value: float,
    pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(pre [..., n, 1], post [..., n, 1], residual logits [..., n, n])``: the gated mixes
    ``sigmoid(alpha_pre m + b) + pre_eps`` and ``post_mult sigmoid(alpha_post m + b)``."""
    n = mhc_mult
    alpha = alpha.float()
    scale = torch.cat([alpha[0].expand(n), alpha[1].expand(n), alpha[2].expand(n * n)])
    mixes = mixes * scale + b.float()
    pre = (mixes[..., :n].sigmoid() + pre_eps).unsqueeze(-1)
    post = (mixes[..., n:2 * n].sigmoid() * post_mult_value).unsqueeze(-1)
    return pre, post, mixes[..., 2 * n:].unflatten(-1, (n, n))


def sinkhorn_normalize(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Sinkhorn-Knopp on the last two axes: a softmax over rows, then alternating column and row
    normalizations, ``repeat`` column normalizations in total."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_apply_mix(streams: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """The streams mixed by ``mix [..., n, 1]`` into one ``[..., D]`` in the stream dtype."""
    return (streams.float() * mix).sum(-2).to(streams.dtype)


def naive_mhc_pre(
    streams: torch.Tensor,
    fn: torch.Tensor,
    alpha: torch.Tensor,
    b: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None,
    norm_eps: float = 1e-6,
    mhc_mult: int = 4,
    post_mult_value: float = 2.0,
    pre_eps: float = 1e-6,
    sinkhorn_eps: float = 1e-6,
    sinkhorn_repeat: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One connector's controller: ``(layer_input [..., D], pre_mix, post_mix, comb [..., n, n])``."""
    mixes = mhc_pre_norm_fn(streams, fn, norm_weight, norm_eps)
    pre_mix, post_mix, comb_logits = mhc_pre_split_mixes(mixes, alpha, b, mhc_mult, post_mult_value, pre_eps)
    comb = sinkhorn_normalize(comb_logits, repeat=sinkhorn_repeat, eps=sinkhorn_eps)
    return mhc_pre_apply_mix(streams, pre_mix), pre_mix, post_mix, comb


def naive_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """The streams after a branch: ``post_n x + sum_m comb[m, n] residual_m``, ``[..., n, D]``."""
    mixed = torch.einsum('...mn,...mc->...nc', comb.float(), residual.float())
    return (x.float().unsqueeze(-2) * post_mix.float() + mixed).to(residual.dtype)


def naive_mhc_head(
    streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None,
    norm_eps: float = 1e-6,
    pre_eps: float = 1e-6,
) -> torch.Tensor:
    """The learned final contraction (DeepSeek-V4 ``hc_head``): independent sigmoid gates per stream,
    ``[..., D]``."""
    mixes = mhc_pre_norm_fn(streams, fn, norm_weight, norm_eps)
    mix = (mixes * scale.float().reshape(()) + base.float()).sigmoid() + pre_eps
    return mhc_pre_apply_mix(streams, mix.unsqueeze(-1))


def expand_to_mhc(hidden: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    """``[..., D]`` repeated into ``mhc_mult`` streams, ``[..., n, D]``."""
    return hidden.unsqueeze(-2).expand(*hidden.shape[:-1], mhc_mult, hidden.shape[-1]).contiguous()
