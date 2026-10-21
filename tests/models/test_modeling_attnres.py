import pytest
import torch

from depth_mixer.models import AttnResConfig

from .test_modeling_base import (
    run_test_cached_decoding,
    run_test_generation,
    run_test_left_padding,
    run_test_model_forward_backward,
    run_test_save_load,
    tiny_config,
)


@pytest.mark.parametrize('mode', ['naive', 'fused'])
@pytest.mark.parametrize('block_size', [1, 2, 4])
def test_forward_backward(mode, block_size):
    run_test_model_forward_backward(tiny_config(AttnResConfig, attnres_mode=mode, attnres_block_size=block_size))


def test_save_load():
    run_test_save_load(tiny_config(AttnResConfig, attnres_block_size=2))


@pytest.mark.parametrize('mode,dtype,ratio', [('naive', torch.float32, 1e-4), ('fused', torch.bfloat16, 0.02)])
def test_cached_decoding(mode, dtype, ratio):
    run_test_cached_decoding(tiny_config(AttnResConfig, attnres_mode=mode, attnres_block_size=2), dtype=dtype, ratio=ratio)


def test_generation():
    run_test_generation(tiny_config(AttnResConfig, attnres_mode='naive', attnres_block_size=2))


def test_left_padding():
    run_test_left_padding(tiny_config(AttnResConfig, attnres_mode='naive', attnres_block_size=2))
