import pytest
import torch
from fla.utils import assert_close, device

from depth_mixer.models.mhc import MHCConfig

from .test_modeling_base import (
    VOCAB,
    build,
    run_test_cached_decoding,
    run_test_generation,
    run_test_left_padding,
    run_test_model_forward_backward,
    run_test_save_load,
    tiny_config,
)


@pytest.mark.parametrize('mode', ['naive', 'fused'])
@pytest.mark.parametrize('qk_norm', [False, True])
def test_forward_backward(mode, qk_norm):
    run_test_model_forward_backward(tiny_config(MHCConfig, mhc_mode=mode, qk_norm=qk_norm))


@pytest.mark.parametrize('tie', [False, True])
def test_save_load(tie):
    run_test_save_load(tiny_config(MHCConfig, mhc_mode='naive', tie_word_embeddings=tie), dtype=torch.float32)


def test_bf16_load_keeps_controllers_fp32(tmp_path):
    """A BF16 load keeps the controllers and the head FP32, as the mHC recipe trains them."""
    from transformers import AutoModelForCausalLM

    build(tiny_config(MHCConfig), torch.float32).save_pretrained(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    for name, parameter in model.named_parameters():
        expected = torch.float32 if ('_hc.' in name or 'mhc_head' in name) else torch.bfloat16
        assert parameter.dtype == expected, f"{name} is {parameter.dtype}"


@pytest.mark.parametrize('mode,dtype,ratio', [('naive', torch.float32, 1e-4), ('fused', torch.bfloat16, 0.02)])
def test_cached_decoding(mode, dtype, ratio):
    run_test_cached_decoding(tiny_config(MHCConfig, mhc_mode=mode), dtype=dtype, ratio=ratio)


def test_generation():
    run_test_generation(tiny_config(MHCConfig, mhc_mode='naive'))


def test_left_padding():
    run_test_left_padding(tiny_config(MHCConfig, mhc_mode='naive'))


@pytest.mark.skipif(device != 'cuda', reason="the fused mHC kernels need CUDA")
def test_fused_matches_naive():
    """The fused kernels reproduce the naive ops inside the model, forward and backward."""
    ids = torch.randint(2, VOCAB, (2, 64), device=device)
    results = {}
    for mode in ('naive', 'fused'):
        model = build(tiny_config(MHCConfig, mhc_mode=mode), torch.bfloat16).train()
        out = model(ids, labels=ids)
        out.loss.backward()
        results[mode] = (out.logits.float(), model.model.layers[0].attn_hc.fn.grad.float())
    assert_close('logits', results['naive'][0], results['fused'][0], 0.02)
    assert_close('dfn', results['naive'][1], results['fused'][1], 0.02)


def test_gradient_checkpointing():
    """mHC blocks are functional, so a recomputed block gives the same gradients."""
    config = tiny_config(MHCConfig, mhc_mode='naive')
    ids = torch.randint(2, VOCAB, (2, 32), device=device)
    grads = []
    for checkpointing in (False, True):
        model = build(config, torch.float32).train()
        if checkpointing:
            model.gradient_checkpointing_enable()
        model(ids, labels=ids).loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
    assert_close('grads', grads[0], grads[1], 1e-4)

