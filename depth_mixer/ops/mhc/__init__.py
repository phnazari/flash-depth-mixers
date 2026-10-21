"""mHC (manifold-constrained Hyper-Connections): naive PyTorch ops and fused TileLang ops (DeepSeek
TileKernels, vendored under ``tile_kernels``). The fused ops import TileLang on their first call;
they need a CUDA toolkit with ``nvcc``, BF16 streams, 4 streams and a stream width divisible by 64."""

from .fused import mhc_head as fused_mhc_head
from .fused import mhc_post as fused_mhc_post
from .fused import mhc_pre as fused_mhc_pre
from .fused import mhc_pre_inference as fused_mhc_pre_inference
from .naive import expand_to_mhc, naive_mhc_head, naive_mhc_post, naive_mhc_pre, sinkhorn_normalize

__all__ = [
    "expand_to_mhc",
    "fused_mhc_head",
    "fused_mhc_post",
    "fused_mhc_pre",
    "fused_mhc_pre_inference",
    "naive_mhc_head",
    "naive_mhc_post",
    "naive_mhc_pre",
    "sinkhorn_normalize",
]
