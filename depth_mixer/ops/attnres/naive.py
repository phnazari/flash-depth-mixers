# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE-flash-linear-attention file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Adapted from flash-linear-attention's ``fla/ops/attnres/naive.py`` to the signature of
# ``fused_attnres`` here: separate epsilons for the source and output norms, no logit scale.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def naive_attnres(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    key_rms_eps: float = 1e-6,
    output_rms_eps: float | None = None,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""
    Apply AttnRes residual aggregation in FP32.

    AttnRes normalizes each residual source with RMSNorm, scores it against `query`, applies softmax over the
    residual-source dimension, and returns the weighted sum of residual sources.
    See `Attention Residuals <https://arxiv.org/abs/2603.15031>`_.

    Args:
        query (torch.Tensor):
            Per-layer pseudo-query with `D` elements (e.g. `[D]` or `[1, D]`), where `D` is the hidden size.
        residuals (Sequence[torch.Tensor]):
            Non-empty sequence of same-dtype, same-`D` residual sources, each of shape `[..., D]`.
        rms_weight (torch.Tensor):
            RMSNorm scale for key normalization of shape `[D]`.
        output_rms_weight (torch.Tensor, optional):
            If set, an extra RMSNorm with this weight is applied to the mixed residual before returning.
        key_rms_eps (float):
            Epsilon of the source RMSNorm. Default: `1e-6`.
        output_rms_eps (float, optional):
            Epsilon of the output RMSNorm; `key_rms_eps` when not set.
        return_weights (bool):
            Whether to return depth softmax probabilities. Default: `False`.

    Returns:
        o (torch.Tensor):
            Mixed residual of shape `[..., D]`.
        p (torch.Tensor):
            Depth softmax probabilities of shape `[L, ...]` if `return_weights=True`, otherwise not returned.
    """
    if len(residuals) == 0:
        raise ValueError("residuals must contain at least one source")

    output_shape = residuals[0].shape
    D = output_shape[-1]
    stacked = torch.stack(tuple(residual.reshape(-1, D) for residual in residuals), dim=0)

    # All math runs in FP32; the result is rounded to the residual dtype once.
    v = stacked.float()
    k = F.rms_norm(v, (D,), rms_weight.flatten().float(), key_rms_eps)
    p = torch.einsum("l...d,d->l...", k, query.flatten().float()).softmax(dim=0)
    o = torch.einsum("l...,l...d->...d", p, v).view(output_shape)

    if output_rms_weight is not None:
        eps = key_rms_eps if output_rms_eps is None else output_rms_eps
        o = F.rms_norm(o, (D,), output_rms_weight.flatten().float(), eps)

    o = o.to(stacked.dtype)

    if return_weights:
        return o, p.view(len(residuals), *output_shape[:-1])
    return o


__all__ = ["naive_attnres"]
