import warnings

from transformers.configuration_utils import PretrainedConfig


class LDMConfig(PretrainedConfig):
    """A pre-norm transformer (Llama or Qwen3 shapes) with a Linear Depth Memory.

    ``ldm_mode`` selects the recurrence implementation: ``fused_recurrent`` (one Triton step per
    write and read, BF16 memory, any recent CUDA GPU), ``chunk`` (the CuTe DSL kernels of the paper's
    main runs, H100 and B200, windows of ``ldm_span`` edges) or ``naive`` (FP32 PyTorch).
    """

    model_type = 'ldm'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        ldm_mode: str = "fused_recurrent",
        ldm_groups: int = 16,
        ldm_key_dim: int = 8,
        ldm_address_hidden_dim: int = 32,
        ldm_recurrence: str = "rls",
        ldm_alpha: float = 1.0,
        ldm_lambda_init: float = 0.99,
        ldm_learn_lambda: bool = True,
        ldm_read_gate: bool = True,
        ldm_write_gate: bool = True,
        ldm_read_gain: bool = True,
        ldm_eps: float = 1e-6,
        ldm_covariance_dtype: str = "bfloat16",
        ldm_span: int = 4,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        self.ldm_mode = ldm_mode
        self.ldm_groups = ldm_groups
        self.ldm_key_dim = ldm_key_dim
        self.ldm_address_hidden_dim = ldm_address_hidden_dim
        self.ldm_recurrence = ldm_recurrence
        self.ldm_alpha = ldm_alpha
        self.ldm_lambda_init = ldm_lambda_init
        self.ldm_learn_lambda = ldm_learn_lambda
        self.ldm_read_gate = ldm_read_gate
        self.ldm_write_gate = ldm_write_gate
        self.ldm_read_gain = ldm_read_gain
        self.ldm_eps = ldm_eps
        self.ldm_covariance_dtype = ldm_covariance_dtype
        self.ldm_span = ldm_span

        if ldm_mode not in ("naive", "fused_recurrent", "chunk"):
            raise ValueError(f"`ldm_mode` must be 'naive', 'fused_recurrent' or 'chunk', got {ldm_mode!r}")
        if ldm_mode == "chunk" and ldm_recurrence != "rls":
            raise ValueError("the chunk mode implements the RLS recurrence only")
        if ldm_recurrence not in ("rls", "delta_rule", "outer_product"):
            raise ValueError(f"`ldm_recurrence` must be 'rls', 'delta_rule' or 'outer_product', got {ldm_recurrence!r}")
        if hidden_size % ldm_groups:
            raise ValueError("`hidden_size` must be divisible by `ldm_groups`")
        if not 0 < ldm_lambda_init <= 1:
            raise ValueError("`ldm_lambda_init` must be in (0, 1]")
        if ldm_covariance_dtype not in ("bfloat16", "float32"):
            raise ValueError("`ldm_covariance_dtype` must be 'bfloat16' or 'float32'")
        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
