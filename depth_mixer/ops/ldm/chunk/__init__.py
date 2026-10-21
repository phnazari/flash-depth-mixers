"""The chunked LDM recurrence (the paper code's ``fast`` backend): CuTe DSL kernels for Hopper (SM90)
and Blackwell (SM100) that keep the memory only at the ends of windows of ``span`` edges and unwind
each window from its tapes in the backward. See ``DESIGN.md``."""

from __future__ import annotations

import torch

from .state import SegmentState


def chunk_ldm(
    keys: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    forgetting: torch.Tensor,
    queries: torch.Tensor,
    alpha: float = 1.0,
    span: int = 4,
    running_covariance_dtype: str = "bfloat16",
) -> torch.Tensor:
    """``S`` RLS edges on precomputed inputs: unit-norm FP32 ``keys`` and FP32 ``queries``
    ``[S, ..., G, K]``, BF16 ``values`` ``[S, ..., G, D]``, FP32 write ``scales`` (the square root of
    the write weight, applied to key and value) and forgetting factors broadcastable to ``[S, ..., G]``.
    Returns the BF16 reads ``[S, ..., G, D]``. Serves tests and benchmarks; the model steps an
    ``LDMState``."""
    groups, key_dim = keys.shape[-2:]
    state = SegmentState(
        total_edges=keys.shape[0], span=span, groups=groups, key_dim=key_dim, value_dim=values.shape[-1],
        covariance_scale=alpha, running_covariance_dtype=running_covariance_dtype,
    )
    token = state.new_token(values)
    reads = []
    for step in range(keys.shape[0]):
        read, token = state.step(token, keys[step], queries[step], values[step], scales[step], forgetting[step])
        reads.append(read)
    return torch.stack(reads)


__all__ = ["SegmentState", "chunk_ldm"]
