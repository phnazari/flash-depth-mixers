"""Naive PyTorch LDM: the memory M and covariance P written and read in order, in FP32.

Conventions: ``memory`` is ``[..., K, D]`` and maps an address to a value through ``M^T a``;
``covariance`` is ``[..., K, K]``; keys and queries are ``[..., K]``, values ``[..., D]`` and
forgetting factors ``[...]``. Keys and values arrive weighted by the square root of the write gate.
Every intermediate is materialized; this is the reference for the fused kernels, not a fast path.
"""

from __future__ import annotations

import torch

RECURRENCES = ("rls", "delta_rule", "outer_product")


def naive_ldm_write(
    memory: torch.Tensor,
    covariance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    forgetting: torch.Tensor,
    recurrence: str = "rls",
) -> tuple[torch.Tensor, torch.Tensor]:
    """One write. The rules differ in the gain and in whether the prediction is subtracted::

        rls            gain = Pk / (lambda + k'Pk)   error = v - M'k           P updated
        delta_rule     gain = k                      error = v - (lambda M)'k  M decayed by lambda
        outer_product  gain = k                      error = v                 M decayed by lambda

    RLS forgets through P; the other two decay M before the write and carry P unchanged.
    """
    if recurrence not in RECURRENCES:
        raise ValueError(f"unknown recurrence {recurrence!r}; expected one of {RECURRENCES}")
    if recurrence == "rls":
        covariance_key = torch.einsum("...ij,...j->...i", covariance, key)
        denominator = forgetting + (key * covariance_key).sum(-1)
        gain = covariance_key / denominator.unsqueeze(-1)
        key_covariance = torch.einsum("...i,...ij->...j", key, covariance)
        covariance = (covariance - gain.unsqueeze(-1) * key_covariance.unsqueeze(-2)) / forgetting[..., None, None]
    else:
        gain = key
        memory = memory * forgetting[..., None, None]
    if recurrence == "outer_product":
        error = value
    else:
        error = value - torch.einsum("...kd,...k->...d", memory, key)
    memory = memory + gain.unsqueeze(-1) * error.unsqueeze(-2)
    return memory, covariance


def naive_ldm_read(memory: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """The read ``M^T q``."""
    return torch.einsum("...kd,...k->...d", memory, query)


def naive_ldm_step(
    memory: torch.Tensor,
    covariance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    forgetting: torch.Tensor,
    query: torch.Tensor,
    recurrence: str = "rls",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One edge: a write followed by a read of the updated state. Returns ``(read, memory, covariance)``."""
    memory, covariance = naive_ldm_write(memory, covariance, key, value, forgetting, recurrence)
    return naive_ldm_read(memory, query), memory, covariance


def initial_state(
    shape: torch.Size | tuple[int, ...],
    key_dim: int,
    value_dim: int,
    alpha: float | torch.Tensor = 1.0,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The zero memory and the covariance ``diag(alpha)`` for a state of leading shape ``[..., G]``.
    ``alpha`` is a float or a ``[G, K]`` tensor."""
    memory = torch.zeros(*shape, key_dim, value_dim, device=device, dtype=dtype)
    if isinstance(alpha, torch.Tensor):
        diagonal = alpha.to(device=device, dtype=dtype)
    else:
        diagonal = torch.full((key_dim,), float(alpha), device=device, dtype=dtype)
    covariance = torch.diag_embed(diagonal).expand(*shape, key_dim, key_dim).clone()
    return memory, covariance


def naive_recurrent_ldm(
    keys: torch.Tensor,
    values: torch.Tensor,
    forgetting: torch.Tensor,
    queries: torch.Tensor,
    alpha: float | torch.Tensor = 1.0,
    recurrence: str = "rls",
    output_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """``S`` edges on precomputed inputs: ``keys`` and ``queries`` ``[S, ..., G, K]``, ``values``
    ``[S, ..., G, D]``, ``forgetting`` broadcastable to ``[S, ..., G]``. Returns the FP32 reads
    ``[S, ..., G, D]`` and, with ``output_final_state``, the final ``(memory, covariance)``."""
    keys, values, queries = keys.float(), values.float(), queries.float()
    forgetting = forgetting.float().expand(keys.shape[:-1])
    memory, covariance = initial_state(
        keys.shape[1:-1], keys.shape[-1], values.shape[-1], alpha, device=keys.device,
    )
    reads = []
    for step in range(keys.shape[0]):
        read, memory, covariance = naive_ldm_step(
            memory, covariance, keys[step], values[step], forgetting[step], queries[step], recurrence,
        )
        reads.append(read)
    reads = torch.stack(reads)
    if output_final_state:
        return reads, (memory, covariance)
    return reads
