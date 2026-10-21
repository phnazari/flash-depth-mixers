from .fused_recurrent import fused_recurrent_ldm
from .naive import naive_ldm_read, naive_ldm_step, naive_ldm_write, naive_recurrent_ldm


def __getattr__(name):
    # The chunked kernels import the CuTe DSL; load them on first use.
    if name == "chunk_ldm":
        from .chunk import chunk_ldm
        return chunk_ldm
    raise AttributeError(name)


__all__ = [
    "chunk_ldm",
    "fused_recurrent_ldm",
    "naive_ldm_read",
    "naive_ldm_step",
    "naive_ldm_write",
    "naive_recurrent_ldm",
]
