"""The LDM recurrence of one model forward (2L + 1 edges) in each mode, forward and forward+backward.

    python benchmarks/ops/benchmark_ldm.py --groups 16 --key-dim 8 --value-dim 64 --edges 49

``chunk`` training launches need a tuned schedule for the GPU (``depth_mixer/ops/ldm/chunk/tuned``);
a shape without one is reported as such.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from depth_mixer.ops.ldm import fused_recurrent_ldm, naive_recurrent_ldm


def inputs(edges, tokens, groups, key_dim, value_dim, requires_grad):
    k = F.normalize(torch.randn(edges, tokens, groups, key_dim, device="cuda"), dim=-1)
    q = F.normalize(torch.randn(edges, tokens, groups, key_dim, device="cuda"), dim=-1)
    v = torch.randn(edges, tokens, groups, value_dim, device="cuda", dtype=torch.bfloat16)
    scale = (2 * torch.sigmoid(torch.randn(edges, tokens, groups, device="cuda"))).sqrt()
    f = torch.sigmoid(torch.randn(edges, 1, groups, device="cuda") + 4).expand(edges, tokens, groups).contiguous()
    tensors = [k, q, v, scale, f]
    if requires_grad:
        tensors = [t.requires_grad_() for t in tensors]
    return tensors


def run(mode, k, q, v, scale, f):
    if mode == "chunk":
        from depth_mixer.ops.ldm import chunk_ldm
        return chunk_ldm(k, v, scale, f, q)
    weighted_k, weighted_v = k * scale[..., None], v.float() * scale[..., None]
    if mode == "naive":
        return naive_recurrent_ldm(weighted_k, weighted_v, f, q)
    return fused_recurrent_ldm(weighted_k.bfloat16(), weighted_v.bfloat16(), f, q.bfloat16())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--edges", type=int, default=49)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096, 16384, 32768])
    parser.add_argument("--modes", nargs="+", default=["naive", "fused_recurrent", "chunk"])
    args = parser.parse_args()

    print(f"{torch.cuda.get_device_name()}  G={args.groups} K={args.key_dim} D={args.value_dim} edges={args.edges}")
    print(f"{'tokens':>8} {'mode':>16} {'fwd ms':>10} {'fwd+bwd ms':>12}")
    for tokens in args.tokens:
        for mode in args.modes:
            shape = (args.edges, tokens, args.groups, args.key_dim, args.value_dim)
            with torch.no_grad():
                tensors = inputs(*shape, requires_grad=False)
                forward = do_bench(lambda: run(mode, *tensors))
            try:
                tensors = inputs(*shape, requires_grad=True)
                grad = torch.randn(args.edges, tokens, args.groups, args.value_dim, device="cuda")
                backward = f"{do_bench(lambda: run(mode, *tensors).float().backward(grad)):12.3f}"
            except RuntimeError as error:
                backward = "untuned" if "no tuned schedule" in str(error) else "error"
            print(f"{tokens:>8} {mode:>16} {forward:10.3f} {backward:>12}")


if __name__ == "__main__":
    main()
