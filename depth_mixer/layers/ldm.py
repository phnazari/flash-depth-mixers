"""Linear Depth Memory (LDM): a fixed-size associative memory that the residual stream writes to and
reads from across depth.

Each token carries a memory ``M`` of shape ``[G, K, D/G]`` and, for the RLS recurrence, a covariance
``P`` of shape ``[G, K, K]``. The embedding is written first. Each block then reads before attention,
writes the attention output, reads before the MLP and writes the MLP output; a final read precedes
the LM head. A read adds ``M^T q`` to the branch input only, never to the stream itself::

    a = x + read(q_attn(x));          delta = attn(norm(a));  x = x + delta;  write(k(x), delta)
    m = x + read(q_mlp(x));           delta = mlp(norm(m));   x = x + delta;  write(k(x), delta)

Addresses come from small SiLU networks on the RMS-normalized stream and are L2-normalized per
group. Gates are ``2 sigmoid(logit)``: a read gate scales the query, a write gate weights the
observation (its square root scales both key and value). A learned read gain scales each group's
query, and a learned forgetting factor ``sigmoid(logit)`` discounts each write.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_mixer.ops.ldm import fused_recurrent as _fused_recurrent
from depth_mixer.ops.ldm.naive import initial_state, naive_ldm_step

LDM_MODES = ("naive", "fused_recurrent", "chunk")


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class LDMAddressNetwork(nn.Module):
    """The address network of one site: the RMS-normalized source with a learned gain, a SiLU hidden
    layer ``w1`` and an address head ``w2`` with ``groups * key_dim`` outputs, plus an optional gate
    head with one logit per group."""

    def __init__(
        self,
        hidden_size: int,
        groups: int,
        key_dim: int,
        address_hidden_dim: int,
        gate: bool,
        gate_init: float = 1.0,
        norm_eps: float = 1e-6,
        eps: float = 1e-6,
        normalize: bool = True,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.groups = groups
        self.key_dim = key_dim
        self.eps = eps
        self.normalize = normalize
        self.gate_init = gate_init
        self.initializer_range = initializer_range
        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.w1 = nn.Linear(hidden_size, address_hidden_dim, bias=True)
        self.w2 = nn.Linear(address_hidden_dim, groups * key_dim, bias=True)
        self.gate = nn.Linear(address_hidden_dim, groups, bias=True) if gate else None

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        std = self.initializer_range
        for linear in (self.w1, self.w2):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
            nn.init.zeros_(linear.bias)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, _logit(self.gate_init / 2.0))

    def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The FP32 address ``[..., G, K]`` and the FP32 gate ``2 sigmoid(logit)`` ``[..., G]`` or None."""
        context = F.rms_norm(source.float(), self.norm.normalized_shape, self.norm.weight.float(), self.norm.eps)
        hidden = F.silu(self.w1(context.to(source.dtype)))
        address = self.w2(hidden).unflatten(-1, (self.groups, self.key_dim)).float()
        if self.normalize:
            address = address / address.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        gate = None if self.gate is None else 2.0 * torch.sigmoid(self.gate(hidden).float())
        return address, gate


class LDMRead(nn.Module):
    """One read: the query of the address network, scaled by the read gate and the learned gain."""

    def __init__(
        self,
        hidden_size: int,
        groups: int,
        key_dim: int,
        address_hidden_dim: int,
        read_gate: bool = True,
        read_gain: bool = True,
        read_gain_init: float = 1.0,
        norm_eps: float = 1e-6,
        eps: float = 1e-6,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.read_gain_init = read_gain_init
        self.proj = LDMAddressNetwork(
            hidden_size, groups, key_dim, address_hidden_dim, gate=read_gate, norm_eps=norm_eps,
            eps=eps, initializer_range=initializer_range,
        )
        self.gain = nn.Parameter(torch.full((groups,), float(read_gain_init))) if read_gain else None

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # nn.init (not Tensor.fill_), so that transformers skips tensors loaded from a checkpoint.
        if self.gain is not None:
            nn.init.constant_(self.gain, self.read_gain_init)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """The FP32 query ``[..., G, K]``."""
        query, gate = self.proj(source)
        if gate is not None:
            query = query * gate.unsqueeze(-1)
        if self.gain is not None:
            query = query * self.gain.float().unsqueeze(-1)
        return query


class LDMWrite(nn.Module):
    """One write: the key of the address network, the grouped value and the forgetting factor. The
    write gate ``w`` weights the observation: key and value are both scaled by ``sqrt(w)``."""

    def __init__(
        self,
        hidden_size: int,
        groups: int,
        key_dim: int,
        address_hidden_dim: int,
        write_gate: bool = True,
        lambda_init: float = 0.99,
        learn_lambda: bool = True,
        norm_eps: float = 1e-6,
        eps: float = 1e-6,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.groups = groups
        self.eps = eps
        self.lambda_init = lambda_init
        self.proj = LDMAddressNetwork(
            hidden_size, groups, key_dim, address_hidden_dim, gate=write_gate, norm_eps=norm_eps,
            eps=eps, initializer_range=initializer_range,
        )
        # The forgetting logit stays FP32 (see ``_keep_in_fp32_modules_strict`` of the models).
        self.forgetting = LDMForgetting(groups, lambda_init=lambda_init, learn_lambda=learn_lambda, eps=eps)

    def forward(
        self,
        source: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The unit-norm FP32 key ``[..., G, K]``, the grouped value ``[..., G, D/G]`` in its own dtype,
        the FP32 write scale ``sqrt(max(w, eps))`` ``[..., G]`` and the FP32 forgetting factors ``[G]``.
        The state applies the scale to both key and value."""
        key, gate = self.proj(source)
        value = value.unflatten(-1, (self.groups, -1))
        if gate is None:
            scale = torch.ones_like(key[..., 0])
        else:
            scale = gate.clamp_min(self.eps).sqrt()
        return key, value, scale, self.forgetting()


class LDMForgetting(nn.Module):
    """The forgetting factor ``max(sigmoid(logit), eps)`` of one write, per group; exactly one when
    ``lambda_init`` is one."""

    def __init__(self, groups: int, lambda_init: float = 0.99, learn_lambda: bool = True, eps: float = 1e-6):
        super().__init__()
        self.lambda_init = lambda_init
        self.eps = eps
        self.logit = nn.Parameter(
            torch.full((groups,), _logit(lambda_init) if lambda_init < 1.0 else 0.0, dtype=torch.float32),
            requires_grad=learn_lambda and lambda_init < 1.0,
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.constant_(self.logit, _logit(self.lambda_init) if self.lambda_init < 1.0 else 0.0)

    def forward(self) -> torch.Tensor:
        if self.lambda_init >= 1.0:
            return torch.ones_like(self.logit, dtype=torch.float32)
        return torch.sigmoid(self.logit.float()).clamp_min(self.eps)


class LDMState:
    """The depth memory of one forward pass.

    Writes and reads alternate, starting with the embedding write: :meth:`write` holds a write
    until the next :meth:`read`, which applies it and then reads the updated state (one *edge*).

    - ``naive``: FP32 tensors, differentiated by autograd.
    - ``fused_recurrent``: one Triton step per edge on a BF16 memory, unwound in place in the
      backward; any recent CUDA GPU.
    - ``chunk``: the CuTe DSL kernels (H100, B200), which commit the memory once per window of
      ``span`` edges and unwind each window from its tapes; RLS only. Their training launches need
      a tuned schedule for the GPU; launches without gradients run on any GPU the kernels compile for.
    """

    def __init__(
        self,
        hidden: torch.Tensor,
        *,
        groups: int,
        key_dim: int,
        num_writes: int,
        alpha: float = 1.0,
        mode: str = "naive",
        recurrence: str = "rls",
        covariance_dtype: torch.dtype = torch.bfloat16,
        span: int = 4,
    ):
        if mode not in LDM_MODES:
            raise ValueError(f"unknown LDM mode {mode!r}; expected one of {LDM_MODES}")
        if hidden.shape[-1] % groups:
            raise ValueError("the hidden size must be divisible by the LDM groups")
        self.mode = mode
        self.recurrence = recurrence
        self.dtype = hidden.dtype
        self.pending: tuple[torch.Tensor, ...] | None = None
        value_dim = hidden.shape[-1] // groups
        if mode == "naive":
            self.memory, self.covariance = initial_state(
                (*hidden.shape[:-1], groups), key_dim, value_dim, alpha, device=hidden.device,
            )
        elif mode == "fused_recurrent":
            rule = _fused_recurrent.RULES[recurrence]
            self.alpha = alpha
            state, self.token = _fused_recurrent.allocate(
                hidden, groups=groups, key_dim=key_dim, alpha=alpha, slots=num_writes,
                covariance_dtype=covariance_dtype, rule=rule,
            )
            _fused_recurrent.activate(state, [_fused_recurrent.slot_views(state, s) for s in range(num_writes)], rule)
        else:
            if recurrence != "rls":
                raise ValueError("the chunk mode implements the RLS recurrence only")
            from depth_mixer.ops.ldm.chunk import SegmentState

            self.segment = SegmentState(
                total_edges=num_writes, span=span, groups=groups, key_dim=key_dim, value_dim=value_dim,
                covariance_scale=alpha,
                running_covariance_dtype="float32" if covariance_dtype == torch.float32 else "bfloat16",
            )
            self.token = self.segment.new_token(hidden)

    def write(self, key: torch.Tensor, value: torch.Tensor, scale: torch.Tensor, forgetting: torch.Tensor) -> None:
        """Hold a write: the unit-norm key, the grouped value, the write scale applied to both, and the
        forgetting factors (see :meth:`LDMWrite.forward`)."""
        if self.pending is not None:
            raise RuntimeError("an LDM write must be followed by a read before the next write")
        self.pending = (key, value, scale, forgetting)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Apply the pending write and read the updated state with ``query``; returns ``[..., G, D/G]``
        (FP32 in naive mode, BF16 or the working dtype otherwise)."""
        if self.pending is None:
            raise RuntimeError("an LDM read must follow a write")
        key, value, scale, forgetting = self.pending
        self.pending = None
        if self.mode == "chunk":
            read, self.token = self.segment.step(
                self.token, key, query, value.to(torch.bfloat16), scale, forgetting,
            )
            return read
        scale = scale.unsqueeze(-1)
        key, value = key * scale, value.float() * scale
        if self.mode == "naive":
            read, self.memory, self.covariance = naive_ldm_step(
                self.memory, self.covariance, key, value, forgetting.expand(key.shape[:-1]), query, self.recurrence,
            )
            return read
        read, self.token = _fused_recurrent.step(
            key.to(self.dtype), value.to(self.dtype), query.to(self.dtype), forgetting, self.token, self.alpha,
        )
        return read


class LinearDepthMemory(nn.Module):
    """The LDM sites of one transformer block: the reads before attention and the MLP, and the writes
    of their outputs."""

    def __init__(
        self,
        hidden_size: int,
        groups: int = 16,
        key_dim: int = 8,
        address_hidden_dim: int = 32,
        read_gate: bool = True,
        write_gate: bool = True,
        read_gain: bool = True,
        read_gain_init: float = 1.0,
        lambda_init: float = 0.99,
        learn_lambda: bool = True,
        norm_eps: float = 1e-6,
        eps: float = 1e-6,
        initializer_range: float = 0.02,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        read = dict(
            hidden_size=hidden_size, groups=groups, key_dim=key_dim, address_hidden_dim=address_hidden_dim,
            read_gate=read_gate, read_gain=read_gain, read_gain_init=read_gain_init, norm_eps=norm_eps,
            eps=eps, initializer_range=initializer_range,
        )
        write = dict(
            hidden_size=hidden_size, groups=groups, key_dim=key_dim, address_hidden_dim=address_hidden_dim,
            write_gate=write_gate, lambda_init=lambda_init, learn_lambda=learn_lambda, norm_eps=norm_eps,
            eps=eps, initializer_range=initializer_range,
        )
        self.attn_read = LDMRead(**read)
        self.attn_write = LDMWrite(**write)
        self.mlp_read = LDMRead(**read)
        self.mlp_write = LDMWrite(**write)
