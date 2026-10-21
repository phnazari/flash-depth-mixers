"""The cache is flash-linear-attention's: a depth mixer keeps no state across tokens beyond the
attention KV cache."""

from fla.models.utils import Cache, FLAGenerationMixin


class DepthMixerGenerationMixin(FLAGenerationMixin):
    """flash-linear-attention's generation mixin, without the transformers ``DynamicCache``.

    transformers 5 pre-allocates a ``DynamicCache`` whose ``len()`` is already non-zero before the
    prefill, which FLA's ``prepare_inputs_for_generation`` reads as a filled cache: the prompt would
    be cut to its last token. Without a default cache, the model creates FLA's ``Cache`` on its
    first forward and the prompt is kept whole.
    """

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False


__all__ = ["Cache", "DepthMixerGenerationMixin", "FLAGenerationMixin"]
