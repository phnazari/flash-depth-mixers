"""Convert a native OLMo-core checkpoint of the Depth Memory paper into a Hugging Face depth_mixer model.

    python utils/convert_from_olmo_core.py --checkpoint checkpoints/released/400m-ldm/lr1e-3 \
        --output converted/400m-ldm

The checkpoint is rebuilt as the OLMo-core model it was trained as (``pretrain/OLMo-core`` must be
importable), so banked parameters are read through the modules that own them. Supported methods:
LDM (any LDM backend's parameter layout), mHC and AttnRes (Block and Full).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = REPO_ROOT / "pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"


def load_olmo_core(checkpoint: str | Path):
    """The OLMo-core model of ``checkpoint`` with its weights, and the checkpoint's config dict."""
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.transformer import TransformerConfig

    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    model = TransformerConfig.from_dict(config["model"]).build(init_device="cpu")
    load_model_and_optim_state(str(checkpoint / "model_and_optim"), model)
    return model.eval(), config


def _backbone_config(model, config: dict) -> dict:
    block = config["model"]["block"]
    attention = block["sequence_mixer"]
    first = model.blocks["0"]
    return dict(
        hidden_size=config["model"]["d_model"],
        num_hidden_layers=config["model"]["n_layers"],
        num_heads=attention["n_heads"],
        num_kv_heads=attention.get("n_kv_heads") or attention["n_heads"],
        head_dim=attention.get("head_dim"),
        qk_norm=getattr(first.attention, "q_norm", None) is not None,
        rope_theta=attention["rope"]["theta"],
        max_position_embeddings=config.get("train_module", {}).get("max_sequence_length", 2048),
        intermediate_size=block["feed_forward"]["hidden_size"],
        hidden_act="swish",
        norm_eps=block["layer_norm"]["eps"],
        vocab_size=config["model"]["vocab_size"],
        tie_word_embeddings=bool(config["model"].get("tie_word_embeddings", False)),
        pad_token_id=config.get("dataset", {}).get("tokenizer", {}).get("pad_token_id", 1),
        eos_token_id=config.get("dataset", {}).get("tokenizer", {}).get("eos_token_id", 50279),
        bos_token_id=config.get("dataset", {}).get("tokenizer", {}).get("bos_token_id"),
    )


def _backbone_state(model, use_head_qk_norm: bool = True) -> dict[str, torch.Tensor]:
    """The embedding, attention, MLP, norms and LM head under the HF names. A norm that sits inside a
    method's wrapper (mHC) is looked up through ``attention_norm``/``feed_forward_norm``."""
    state = {"model.embeddings.weight": model.embeddings.weight}
    for index, block in model.blocks.items():
        prefix = f"model.layers.{index}"
        attention = block.attention
        state[f"{prefix}.attn.q_proj.weight"] = attention.w_q.weight
        state[f"{prefix}.attn.k_proj.weight"] = attention.w_k.weight
        state[f"{prefix}.attn.v_proj.weight"] = attention.w_v.weight
        state[f"{prefix}.attn.o_proj.weight"] = attention.w_out.weight
        if getattr(attention, "q_norm", None) is not None:
            if not attention.use_head_qk_norm:
                raise NotImplementedError("only per-head QK norms (Qwen3) are supported")
            state[f"{prefix}.attn.q_norm.weight"] = attention.q_norm.weight
            state[f"{prefix}.attn.k_norm.weight"] = attention.k_norm.weight
        state[f"{prefix}.attn_norm.weight"] = block.attention_norm.weight
        state[f"{prefix}.mlp_norm.weight"] = block.feed_forward_norm.weight
        state[f"{prefix}.mlp.gate_proj.weight"] = block.feed_forward.w1.weight
        state[f"{prefix}.mlp.up_proj.weight"] = block.feed_forward.w3.weight
        state[f"{prefix}.mlp.down_proj.weight"] = block.feed_forward.w2.weight
    state["model.norm.weight"] = model.lm_head.norm.weight
    state["lm_head.weight"] = model.lm_head.w_out.weight
    return state


def _address_network(prefix: str, projection) -> dict[str, torch.Tensor]:
    """An LDM address network: ``w2`` in the fast layout, ``address`` in the reference layout."""
    head = projection.w2 if hasattr(projection, "w2") else projection.address
    state = {
        f"{prefix}.norm.weight": projection.context_norm.weight,
        f"{prefix}.w1.weight": projection.w1.weight,
        f"{prefix}.w1.bias": projection.w1.bias,
        f"{prefix}.w2.weight": head.weight,
        f"{prefix}.w2.bias": head.bias,
    }
    if projection.gate is not None:
        state[f"{prefix}.gate.weight"] = projection.gate.weight
        state[f"{prefix}.gate.bias"] = projection.gate.bias
    return state


def _require_zero(name: str, tensor: torch.Tensor) -> None:
    if tensor.abs().max().item() != 0:
        raise NotImplementedError(f"{name} is a learned static address, which the HF model does not support")


def convert_ldm(model, config: dict):
    from depth_mixer.models import LDMConfig, LDMForCausalLM

    ldm = config["model"]["block"]["ldm"]
    quality = ldm["quality"]
    if quality.get("replace_read") or quality.get("no_post_function_residual"):
        raise NotImplementedError("the residual-path ablations are not supported")
    if quality.get("value_source", "branch_delta") != "branch_delta":
        raise NotImplementedError("only branch-delta values are supported")
    for site in ("query", "key"):
        address = quality["addresses"][site]
        expected = "current_stream" if site == "query" else "updated_stream"
        if address["mode"] != "dynamic" or address["source"] != expected:
            raise NotImplementedError(f"only dynamic {site}s from the {expected} are supported")
    if quality["rls"].get("learn_alpha"):
        raise NotImplementedError("a learned ridge is not supported")

    hf_config = LDMConfig(
        **_backbone_config(model, config),
        ldm_groups=quality["groups"],
        ldm_key_dim=quality["key_dim"],
        ldm_address_hidden_dim=quality["addresses"]["hidden_dim"],
        ldm_recurrence=quality["recurrence"],
        ldm_alpha=quality["rls"]["alpha_init"],
        ldm_lambda_init=quality["rls"]["lambda_init"],
        ldm_learn_lambda=quality["rls"]["learn_lambda"],
        ldm_read_gate=quality["read_gate"]["enabled"],
        ldm_write_gate=quality["write_gate"]["enabled"],
        ldm_read_gain=quality["learned_read_gain"]["enabled"],
        ldm_eps=quality["eps"],
    )

    state = _backbone_state(model)
    _, block_logits = model.ldm_retention.root_logit, model.ldm_retention.block_logits
    fast_layout = ldm["execution"]["backend"]["name"] in ("fast", "fast-inference")
    for index, block in model.blocks.items():
        rule, prefix = block.ldm_rule, f"model.layers.{index}.ldm"
        for position, branch in enumerate(("attn", "mlp")):
            if fast_layout:
                read_proj, write_proj = getattr(rule, f"{branch}_q_proj"), getattr(rule, f"{branch}_k_proj")
                gain = getattr(rule, f"{branch}_read_gain")
                _require_zero(f"blocks.{index}.{branch}_query", getattr(rule, f"{branch}_query"))
                _require_zero(f"blocks.{index}.{branch}_key", getattr(rule, f"{branch}_key"))
            else:
                site = getattr(rule, branch)
                read_proj, write_proj, gain = site.read.projection, site.write.write.projection, site.read.read_gain
            state.update(_address_network(f"{prefix}.{branch}_read.proj", read_proj))
            state.update(_address_network(f"{prefix}.{branch}_write.proj", write_proj))
            if gain is not None:
                state[f"{prefix}.{branch}_read.gain"] = gain
            state[f"{prefix}.{branch}_write.forgetting.logit"] = block_logits[int(index), position]
    if fast_layout:
        input_proj, output_proj = model.ldm_boundary.input_k_proj, model.ldm_output_boundary.output_q_proj
        output_gain = model.ldm_output_boundary.output_read_gain
        _require_zero("ldm_boundary.input_key", model.ldm_boundary.input_key)
        _require_zero("ldm_output_boundary.output_query", model.ldm_output_boundary.output_query)
    else:
        input_proj = model.ldm_boundary.write.write.projection
        output_proj = model.ldm_output_boundary.output.projection
        output_gain = model.ldm_output_boundary.output.read_gain
    state.update(_address_network("model.ldm_input.proj", input_proj))
    state.update(_address_network("model.ldm_output.proj", output_proj))
    if output_gain is not None:
        state["model.ldm_output.gain"] = output_gain
    state["model.ldm_input.forgetting.logit"] = model.ldm_retention.root_logit
    return _build(LDMForCausalLM, hf_config, state)


def convert_attnres(model, config: dict):
    from depth_mixer.models import AttnResConfig, AttnResForCausalLM

    block_size = config["model"]["block"]["attnres_block_size"]
    hf_config = AttnResConfig(**_backbone_config(model, config), attnres_block_size=block_size)
    state = _backbone_state(model)
    for index, block in model.blocks.items():
        prefix = f"model.layers.{index}"
        if int(index) > 0:
            state[f"{prefix}.attn_res.proj.weight"] = block.attn_res_proj.weight
            state[f"{prefix}.attn_res.norm.weight"] = block.attn_res_norm.weight
        state[f"{prefix}.mlp_res.proj.weight"] = block.mlp_res_proj.weight
        state[f"{prefix}.mlp_res.norm.weight"] = block.mlp_res_norm.weight
    state["model.res.proj.weight"] = model.res_proj.weight
    state["model.res.norm.weight"] = model.res_norm.weight
    return _build(AttnResForCausalLM, hf_config, state)


def convert_mhc(model, config: dict):
    from depth_mixer.models.mhc import MHCConfig, MHCForCausalLM

    hyper = config["model"]["block"]["hyper_connections"]
    if hyper.get("kind", "mhc") != "mhc":
        raise NotImplementedError(f"only mHC hyper-connections are supported, got {hyper['kind']!r}")
    if not hyper.get("learned_final_reduce", True):
        raise NotImplementedError("only the learned final head is supported")

    hf_config = MHCConfig(
        **_backbone_config(model, config),
        mhc_num_streams=hyper["num_residual_streams"],
        mhc_sinkhorn_iters=hyper["sinkhorn_iters"],
        mhc_gating_factor_init=hyper["gating_factor_init"],
        mhc_norm_eps=hyper["controller_norm_eps"],
    )
    state = _backbone_state(model)
    for index, block in model.blocks.items():
        for name, connector in (("attn_hc", block.attention_hyper_connection),
                                ("mlp_hc", block.feed_forward_hyper_connection)):
            controller = connector.controller
            if controller.norm_weight_folded:
                raise ValueError("the checkpoint's model was prepared for inference (folded norm weights)")
            for parameter in ("fn", "norm_weight", "b", "alpha"):
                state[f"model.layers.{index}.{name}.{parameter}"] = getattr(controller, parameter)
    head = model.reduce_residual_streams
    for parameter in ("weight", "bias", "scale", "norm_weight"):
        state[f"model.mhc_head.{parameter}"] = getattr(head, parameter)
    return _build(MHCForCausalLM, hf_config, state)


def _build(model_class, hf_config, state: dict[str, torch.Tensor]):
    if hf_config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    with torch.device("meta"):
        hf_model = model_class(hf_config)
    expected = set(hf_model.state_dict())
    if hf_config.tie_word_embeddings:
        expected.discard("lm_head.weight")
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing {sorted(missing)[:8]}, unexpected {sorted(unexpected)[:8]}")
    state = {name: tensor.detach().clone().contiguous() for name, tensor in state.items()}
    hf_model.load_state_dict(state, strict=False, assign=True)
    if hf_config.tie_word_embeddings:
        hf_model.tie_weights()
    return hf_model


CONVERTERS = {"ldm": convert_ldm, "attnres": convert_attnres, "mhc": convert_mhc}


def method_of(config: dict) -> str:
    block = config["model"]["block"]
    return block["name"]


def convert(checkpoint: str | Path):
    model, config = load_olmo_core(checkpoint)
    method = method_of(config)
    if method not in CONVERTERS:
        raise NotImplementedError(f"no converter for block {method!r}")
    with torch.no_grad():
        return CONVERTERS[method](model, config)


def _keeps_fp32(model, name: str) -> bool:
    patterns = getattr(model, "_keep_in_fp32_modules_strict", None) or []
    return any(re.search(pattern, name) for pattern in patterns)


def save_tokenizer(output: str | Path, hf_config) -> None:
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER))
    eos = tokenizer.convert_ids_to_tokens(hf_config.eos_token_id)
    tokenizer.eos_token = eos
    if hf_config.pad_token_id is not None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(hf_config.pad_token_id)
    tokenizer.save_pretrained(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="OLMo-core checkpoint directory (with config.json)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    hf_model = convert(args.checkpoint)
    dtype = getattr(torch, args.dtype)
    for name, parameter in hf_model.named_parameters():
        # LDM's forgetting logits and mHC's controllers stay FP32.
        if not _keeps_fp32(hf_model, name):
            parameter.data = parameter.data.to(dtype)
    hf_model.config.dtype = args.dtype
    hf_model.save_pretrained(args.output)
    save_tokenizer(args.output, hf_model.config)
    print(f"saved {type(hf_model).__name__} ({sum(p.numel() for p in hf_model.parameters()) / 1e6:.1f}M parameters) to {args.output}")


if __name__ == "__main__":
    main()
