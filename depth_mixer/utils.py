"""Device and backend helpers."""

from __future__ import annotations

import functools
import importlib.util

import torch


@functools.cache
def is_flash_attn_available() -> bool:
    """Whether FlashAttention 2 (``flash_attn_func`` and ``flash_attn_varlen_func``) is importable."""
    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func  # noqa: F401
    except ImportError:
        return False
    return True


def compute_capability(device: torch.device | None = None) -> tuple[int, int] | None:
    """The CUDA compute capability of ``device``, or None without CUDA."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(device)


def is_hopper_or_blackwell_datacenter(device: torch.device | None = None) -> bool:
    """SM90 (H100) or SM100 (B200): the GPUs the CuTe DSL kernels are written and tuned for."""
    return compute_capability(device) in ((9, 0), (10, 0))
