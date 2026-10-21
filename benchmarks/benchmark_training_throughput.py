"""Training throughput of the depth-mixer models: forward, backward and AdamW step on random tokens.

    python benchmarks/benchmark_training_throughput.py --model ldm --size 400m --batch-size 8 --seq-len 2048
    python benchmarks/benchmark_training_throughput.py --model ldm --size 400m --override ldm_mode=chunk

``--size`` picks the paper's backbones (400m Llama-like, 1p5b Qwen3-like) with each method's paper
settings. ``--override key=value`` changes any config field, e.g. a kernel mode. This measures the
HF models; the paper's throughput numbers come from ``paper-experiments/throughput`` on OLMo-core.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM

from depth_mixer.models import AttnResConfig, LDMConfig, MHCConfig

BACKBONES = {
    "400m": dict(hidden_size=1024, num_hidden_layers=24, num_heads=16, num_kv_heads=16, head_dim=64,
                 intermediate_size=2736, rope_theta=10000., vocab_size=50280, tie_word_embeddings=False),
    "1p5b": dict(hidden_size=2048, num_hidden_layers=28, num_heads=16, num_kv_heads=8, head_dim=128,
                 qk_norm=True, intermediate_size=6144, rope_theta=1e6, vocab_size=50280, tie_word_embeddings=True),
}
METHODS = {
    "ldm": (LDMConfig, {"400m": dict(ldm_address_hidden_dim=32), "1p5b": dict(ldm_address_hidden_dim=64)}),
    "mhc": (MHCConfig, {"400m": {}, "1p5b": {}}),
    "attnres": (AttnResConfig, {"400m": dict(attnres_block_size=6), "1p5b": dict(attnres_block_size=8)}),
}


def parse_override(text: str):
    key, value = text.split("=", 1)
    for cast in (int, float):
        try:
            return key, cast(value)
        except ValueError:
            pass
    return key, {"true": True, "false": False}.get(value.lower(), value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(METHODS), required=True)
    parser.add_argument("--size", choices=sorted(BACKBONES), default="400m")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--override", action="append", default=[], type=parse_override)
    args = parser.parse_args()

    config_class, settings = METHODS[args.model]
    config = config_class(**BACKBONES[args.size], **settings[args.size], **dict(args.override))
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).cuda().to(torch.bfloat16).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device="cuda")

    def step():
        loss = model(ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / args.steps
    parameters = sum(p.numel() for p in model.parameters())
    overrides = " ".join(f"{k}={v}" for k, v in args.override)
    print(f"{args.model} {args.size} ({parameters / 1e6:.0f}M) {overrides}  batch {args.batch_size}x{args.seq_len}: "
          f"{elapsed * 1e3:.0f} ms/step, {args.batch_size * args.seq_len / elapsed:,.0f} tokens/s, "
          f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
