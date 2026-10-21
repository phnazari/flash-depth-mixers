"""Checks every depth-mixer model must pass: forward and backward, save/load round trip, KV-cached
decoding that reproduces the full forward, greedy generation, and left padding."""

from __future__ import annotations

import tempfile

import torch
from fla.utils import assert_close, device
from transformers import AutoModelForCausalLM

import depth_mixer  # noqa: F401  (registers the models with the Auto classes)

VOCAB = 1000


def tiny_config(config_class, **kwargs):
    defaults = dict(
        hidden_size=256, num_hidden_layers=3, num_heads=4, num_kv_heads=2, intermediate_size=512,
        vocab_size=VOCAB, max_position_embeddings=128, pad_token_id=0, eos_token_id=1, bos_token_id=None,
    )
    defaults.update(kwargs)
    return config_class(**defaults)


def build(config, dtype: torch.dtype):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).to(device=device, dtype=dtype)
    # The initialization leaves the depth mixers at the identity; perturb every parameter so that
    # the checks exercise them.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
    return model.eval()


def run_test_model_forward_backward(config, dtype: torch.dtype = torch.bfloat16):
    model = build(config, dtype).train()
    ids = torch.randint(2, VOCAB, (2, 64), device=device)
    out = model(ids, labels=ids)
    assert torch.isfinite(out.loss), out.loss
    out.loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters without a gradient: {missing[:8]}"
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients: {bad[:8]}"


def run_test_save_load(config, dtype: torch.dtype = torch.bfloat16):
    model = build(config, dtype)
    ids = torch.randint(2, VOCAB, (2, 32), device=device)
    with torch.no_grad():
        ref = model(ids).logits
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        loaded = AutoModelForCausalLM.from_pretrained(directory, dtype=dtype).to(device).eval()
    for (name, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), f"{name} changed in the save/load round trip"
    with torch.no_grad():
        assert torch.equal(loaded(ids).logits, ref)


def run_test_cached_decoding(config, dtype: torch.dtype = torch.float32, prefix: int = 20, ratio: float = 1e-3):
    """Prefill ``prefix`` tokens, then decode the rest one at a time with the KV cache; every
    position's logits must match the full forward."""
    model = build(config, dtype)
    ids = torch.randint(2, VOCAB, (2, 32), device=device)
    with torch.no_grad():
        full = model(ids).logits
        out = model(ids[:, :prefix], use_cache=True)
        logits, cache = [out.logits], out.past_key_values
        for t in range(prefix, ids.shape[1]):
            out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            logits.append(out.logits)
            cache = out.past_key_values
    assert_close('cached logits', full.float(), torch.cat(logits, 1).float(), ratio)


def run_test_generation(config, dtype: torch.dtype = torch.float32):
    model = build(config, dtype)
    ids = torch.randint(2, VOCAB, (2, 12), device=device)
    with torch.no_grad():
        cached = model.generate(ids, max_new_tokens=12, do_sample=False, use_cache=True)
        uncached = model.generate(ids, max_new_tokens=12, do_sample=False, use_cache=False)
    assert cached.shape == (2, 24)
    assert torch.equal(cached, uncached)


def run_test_left_padding(config, dtype: torch.dtype = torch.float32, ratio: float = 1e-3):
    """A left-padded sequence gives the logits of the same sequence alone."""
    model = build(config, dtype)
    short = torch.randint(2, VOCAB, (1, 10), device=device)
    long = torch.randint(2, VOCAB, (1, 16), device=device)
    batch = torch.cat([torch.cat([torch.zeros(1, 6, dtype=torch.long, device=device), short], 1), long])
    mask = torch.ones_like(batch)
    mask[0, :6] = 0
    with torch.no_grad():
        alone = model(short).logits
        padded = model(batch, attention_mask=mask).logits[:1, 6:]
    assert_close('padded logits', alone.float(), padded.float(), ratio)
