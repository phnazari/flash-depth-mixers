import pytest
import torch

from depth_mixer.models import LDMConfig

from .test_modeling_base import (
    run_test_cached_decoding,
    run_test_generation,
    run_test_left_padding,
    run_test_model_forward_backward,
    run_test_save_load,
    tiny_config,
)


@pytest.mark.parametrize('mode', ['naive', 'fused_recurrent'])
@pytest.mark.parametrize('recurrence', ['rls', 'delta_rule', 'outer_product'])
def test_forward_backward(mode, recurrence):
    run_test_model_forward_backward(tiny_config(LDMConfig, ldm_mode=mode, ldm_recurrence=recurrence, ldm_groups=4))


@pytest.mark.parametrize('qk_norm,tie', [(False, False), (True, True)])
def test_save_load(qk_norm, tie):
    run_test_save_load(tiny_config(LDMConfig, ldm_groups=4, qk_norm=qk_norm, tie_word_embeddings=tie, head_dim=128))


@pytest.mark.parametrize('mode,dtype,ratio', [('naive', torch.float32, 1e-4), ('fused_recurrent', torch.bfloat16, 0.02)])
def test_cached_decoding(mode, dtype, ratio):
    run_test_cached_decoding(tiny_config(LDMConfig, ldm_mode=mode, ldm_groups=4), dtype=dtype, ratio=ratio)


def test_generation():
    run_test_generation(tiny_config(LDMConfig, ldm_mode='naive', ldm_groups=4))


def test_left_padding():
    run_test_left_padding(tiny_config(LDMConfig, ldm_mode='naive', ldm_groups=4))


def test_fused_matches_naive():
    """The fused recurrence reproduces the naive one inside the model."""
    from fla.utils import assert_close

    from .test_modeling_base import VOCAB, build, device
    ids = torch.randint(2, VOCAB, (2, 64), device=device)
    logits = {}
    for mode in ('naive', 'fused_recurrent'):
        model = build(tiny_config(LDMConfig, ldm_mode=mode, ldm_groups=4), torch.bfloat16)
        with torch.no_grad():
            logits[mode] = model(ids).logits.float()
    assert_close('logits', logits['naive'], logits['fused_recurrent'], 0.02)
