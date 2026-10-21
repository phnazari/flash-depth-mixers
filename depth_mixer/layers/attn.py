"""Causal softmax attention with flash-linear-attention's layer interface and cache.

flash-linear-attention's ``Attention`` requires FlashAttention 2. This layer keeps its interface,
parameter names and cache handling, and falls back to PyTorch's scaled dot-product attention when
FlashAttention 2 is not installed. It also takes an explicit ``head_dim`` (Qwen3 shapes) and the
RMS epsilon of the query and key norms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.layers.utils import pad_input, unpad_input
from fla.ops.utils.index import prepare_lens_from_mask

from depth_mixer.modules import RMSNorm, RotaryEmbedding
from depth_mixer.utils import is_flash_attn_available

if TYPE_CHECKING:
    from fla.models.utils import Cache


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    window_size: int | None,
) -> torch.Tensor:
    """Causal attention on ``[B, T, H, D]`` queries and ``[B, S, H_kv, D]`` keys and values, where the
    queries are the last ``T`` of the ``S`` positions. ``attention_mask`` is the ``[B, S']`` padding
    mask (1 for tokens), of which the last ``S`` columns apply."""
    q_len, kv_len = q.shape[1], k.shape[1]
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    if k.shape[1] != q.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    if attention_mask is None and window_size is None and q_len == kv_len:
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        rows = torch.arange(q_len, device=q.device)[:, None] + (kv_len - q_len)
        cols = torch.arange(kv_len, device=q.device)[None, :]
        mask = cols <= rows
        if window_size is not None:
            mask = mask & (cols > rows - window_size)
        mask = mask[None, None]
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, None, -kv_len:].bool()
        # A padding query with no visible key attends to every key instead of producing NaN.
        mask = mask | ~mask.any(-1, keepdim=True)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return o.transpose(1, 2)


class Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        norm_eps: float = 1e-6,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int | None = None,
        layer_idx: int | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = hidden_size // num_heads if head_dim is None else head_dim
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.q_dim, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if is_flash_attn_available():
            o = self._flash_attn(q, k, v, attention_mask, cu_seqlens, batch_size, q_len, max_seqlen)
        else:
            if cu_seqlens is not None:
                raise NotImplementedError("variable-length inputs (cu_seqlens) require FlashAttention 2")
            if attention_mask is not None and q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size:]
            o = _sdpa(q, k, v, attention_mask, self.window_size)
        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values

    def _flash_attn(self, q, k, v, attention_mask, cu_seqlens, batch_size, q_len, max_seqlen):
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        window_size = (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            if q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size:]
            q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=window_size,
            )
            return pad_input(o, indices_q, batch_size, q_len)
        if cu_seqlens is not None:
            o = flash_attn_varlen_func(
                rearrange(q, 'b l ... -> (b l) ...').contiguous(),
                rearrange(k, 'b l ... -> (b l) ...').contiguous(),
                rearrange(v, 'b l ... -> (b l) ...').contiguous(),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=window_size,
            )
            return rearrange(o, '(b l) h d -> b l h d', b=batch_size, l=q_len)
        return flash_attn_func(q, k, v, causal=True, window_size=window_size)
