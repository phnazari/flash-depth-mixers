from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from depth_mixer.layers.attn import Attention
from depth_mixer.layers.ldm import LDMRead, LDMState, LDMWrite, LinearDepthMemory
from depth_mixer.models.ldm.configuration_ldm import LDMConfig
from depth_mixer.models.utils import Cache, DepthMixerGenerationMixin
from depth_mixer.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from depth_mixer.modules import GatedMLP as LDMMLP

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


def _ldm_site_kwargs(config: LDMConfig) -> dict[str, Any]:
    return dict(
        hidden_size=config.hidden_size,
        groups=config.ldm_groups,
        key_dim=config.ldm_key_dim,
        address_hidden_dim=config.ldm_address_hidden_dim,
        norm_eps=config.norm_eps,
        eps=config.ldm_eps,
        initializer_range=config.initializer_range,
    )


class LDMBlock(GradientCheckpointingLayer):

    def __init__(self, config: LDMConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            norm_eps=config.norm_eps,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = LDMMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

        self.ldm = LinearDepthMemory(
            **_ldm_site_kwargs(config),
            read_gate=config.ldm_read_gate,
            write_gate=config.ldm_write_gate,
            read_gain=config.ldm_read_gain,
            lambda_init=config.ldm_lambda_init,
            learn_lambda=config.ldm_learn_lambda,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        ldm_state: LDMState,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs: Unpack[Any],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None, Cache | None]:
        # The memory read joins the branch input only; the stream itself skips it.
        read = ldm_state.read(self.ldm.attn_read(hidden_states)).flatten(-2)
        delta, attentions, past_key_values = self.attn(
            hidden_states=self.attn_norm(hidden_states + read.to(hidden_states.dtype)),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = hidden_states + delta
        ldm_state.write(*self.ldm.attn_write(hidden_states, delta))

        read = ldm_state.read(self.ldm.mlp_read(hidden_states)).flatten(-2)
        delta = self.mlp(self.mlp_norm(hidden_states + read.to(hidden_states.dtype)), **kwargs)
        hidden_states = hidden_states + delta
        ldm_state.write(*self.ldm.mlp_write(hidden_states, delta))

        return hidden_states, attentions, past_key_values


class LDMPreTrainedModel(PreTrainedModel):

    config_class = LDMConfig
    base_model_prefix = 'model'
    # A recomputed block would repeat its writes to the depth memory.
    supports_gradient_checkpointing = False
    _no_split_modules = ['LDMBlock']
    _supports_cache_class = True
    _keep_in_fp32_modules_strict = ['forgetting']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            # Children are initialized before their parents, so an LDM address network resets its
            # linear layers (and zeroes its gate) after the generic Linear branch above.
            module.reset_parameters()

        if rescale_prenorm_residual:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class LDMModel(LDMPreTrainedModel):

    def __init__(self, config: LDMConfig) -> LDMModel:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # The embedding is the first write: it is both the key source and the value.
        self.ldm_input = LDMWrite(
            **_ldm_site_kwargs(config),
            write_gate=config.ldm_write_gate,
            lambda_init=config.ldm_lambda_init,
            learn_lambda=config.ldm_learn_lambda,
        )
        self.layers = nn.ModuleList([LDMBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        # The final read joins the stream before the last norm.
        self.ldm_output = LDMRead(
            **_ldm_site_kwargs(config),
            read_gate=config.ldm_read_gate,
            read_gain=config.ldm_read_gain,
        )
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            warnings.warn("`LDMModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        # Every token carries its own memory across depth: one write per site, 2L + 1 in total.
        ldm_state = LDMState(
            hidden_states,
            groups=self.config.ldm_groups,
            key_dim=self.config.ldm_key_dim,
            num_writes=2 * self.config.num_hidden_layers + 1,
            alpha=self.config.ldm_alpha,
            mode=self.config.ldm_mode,
            recurrence=self.config.ldm_recurrence,
            covariance_dtype=getattr(torch, self.config.ldm_covariance_dtype),
            span=self.config.ldm_span,
        )
        ldm_state.write(*self.ldm_input(hidden_states, hidden_states))

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attentions, past_key_values = layer(
                hidden_states,
                ldm_state,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            if output_attentions:
                all_attns += (attentions,)

        read = ldm_state.read(self.ldm_output(hidden_states)).flatten(-2)
        hidden_states = self.norm(hidden_states + read.to(hidden_states.dtype))

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class LDMForCausalLM(LDMPreTrainedModel, DepthMixerGenerationMixin):

    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = LDMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        loss, logits = None, None
        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
