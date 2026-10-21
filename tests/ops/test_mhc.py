import pytest
import torch
from fla.utils import assert_close, device

from depth_mixer.ops.mhc import (
    fused_mhc_head,
    fused_mhc_post,
    fused_mhc_pre,
    fused_mhc_pre_inference,
    naive_mhc_head,
    naive_mhc_post,
    naive_mhc_pre,
)

requires_cuda = pytest.mark.skipif(device != 'cuda', reason="the fused mHC kernels need CUDA")

N = 4
CONTROLLER = dict(norm_eps=1e-6, mhc_mult=N, post_mult_value=2.0, pre_eps=1e-6, sinkhorn_eps=1e-6, sinkhorn_repeat=20)


def _controller(D):
    fn = 0.02 * torch.randn(N * N + 2 * N, N * D, device=device)
    norm_weight = 1 + 0.1 * torch.randn(N * D, device=device)
    alpha = torch.tensor([0.5, 0.4, 0.8], device=device)
    b = torch.randn(N * N + 2 * N, device=device)
    return fn, norm_weight, alpha, b


@requires_cuda
@pytest.mark.parametrize(
    ('B', 'T', 'D'),
    [
        pytest.param(*test, id="B{}-T{}-D{}".format(*test))
        for test in [
            (2, 256, 1024),   # 400M
            (1, 1000, 2048),  # 1.5B, a token count that is not a multiple of the kernels' blocks
            (3, 37, 256),
        ]
    ],
)
def test_mhc_pre_post(B, T, D):
    torch.manual_seed(42)
    streams = torch.randn(B, T, N, D, device=device, dtype=torch.bfloat16)
    branch = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    params = _controller(D)
    d_input, d_out = torch.randn(B, T, D, device=device), torch.randn(B, T, N, D, device=device)

    outputs = {}
    for name, pre, post in (('ref', naive_mhc_pre, naive_mhc_post), ('tri', fused_mhc_pre, fused_mhc_post)):
        s, y, fn, nw, alpha, b = (t.clone().requires_grad_(True) for t in (streams, branch, *params))
        layer_input, _, post_mix, comb = pre(s, fn, alpha, b, norm_weight=nw, **CONTROLLER)
        out = post(y, s, post_mix, comb)
        loss = (layer_input.float() * d_input).sum() + (out.float() * d_out).sum()
        grads = torch.autograd.grad(loss, (s, y, fn, nw, alpha, b))
        outputs[name] = (layer_input, post_mix, comb, out, grads)

    (ref_x, ref_post, ref_comb, ref_out, ref_grads), (tri_x, tri_post, tri_comb, tri_out, tri_grads) = outputs.values()
    assert_close('layer_input', ref_x, tri_x, 0.005)
    assert_close('post_mix', ref_post, tri_post, 0.005)
    assert_close('comb', ref_comb, tri_comb, 0.005)
    assert_close('out', ref_out, tri_out, 0.005)
    for name, ref_grad, tri_grad in zip(('dstreams', 'dbranch', 'dfn', 'dnorm_weight', 'dalpha', 'db'), ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 0.01)


@requires_cuda
@pytest.mark.parametrize(('B', 'T', 'D'), [(2, 256, 1024), (1, 1000, 2048)])
def test_mhc_head(B, T, D):
    torch.manual_seed(42)
    streams = torch.randn(B, T, N, D, device=device, dtype=torch.bfloat16)
    weight = 0.02 * torch.randn(N, N * D, device=device)
    scale, base = torch.tensor([0.3], device=device), torch.randn(N, device=device)
    norm_weight = 1 + 0.1 * torch.randn(N * D, device=device)
    do = torch.randn(B, T, D, device=device)

    outputs = {}
    for name, head in (('ref', naive_mhc_head), ('tri', fused_mhc_head)):
        inputs = tuple(t.clone().requires_grad_(True) for t in (streams, weight, scale, base, norm_weight))
        s, w, sc, bs, nw = inputs
        o = head(s, w, sc, bs, norm_weight=nw, norm_eps=1e-6, pre_eps=1e-6)
        outputs[name] = (o, torch.autograd.grad((o.float() * do).sum(), inputs))

    (ref, ref_grads), (tri, tri_grads) = outputs.values()
    assert_close('o', ref, tri, 0.005)
    for name, ref_grad, tri_grad in zip(('dstreams', 'dweight', 'dscale', 'dbase', 'dnorm_weight'), ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 0.01)


@requires_cuda
@pytest.mark.parametrize(('T', 'D', 'n_splits'), [(256, 1024, 1), (8, 1024, None), (4096, 2048, None)])
def test_mhc_pre_inference(T, D, n_splits):
    """The fused no-grad path; ``n_splits=None`` reads the per-GPU projection split table."""
    torch.manual_seed(42)
    streams = torch.randn(1, T, N, D, device=device, dtype=torch.bfloat16)
    fn, norm_weight, alpha, b = _controller(D)
    with torch.no_grad():
        ref_x, _, ref_post, ref_comb = naive_mhc_pre(streams, fn, alpha, b, norm_weight=norm_weight, **CONTROLLER)
        try:
            tri_x, tri_post, tri_comb = fused_mhc_pre_inference(
                streams, fn, alpha, b, norm_weight=norm_weight, n_splits=n_splits, **CONTROLLER,
            )
        except RuntimeError as error:
            if "no tuned projection split" in str(error):
                pytest.skip(str(error).splitlines()[0])
            raise
    assert_close('layer_input', ref_x, tri_x, 0.005)
    assert_close('post_mix', ref_post, tri_post, 0.005)
    assert_close('comb', ref_comb, tri_comb, 0.005)


@requires_cuda
def test_sinkhorn_backward_token_block():
    """The Sinkhorn backward's token block only partitions tokens: halving it (as SM120 requires)
    changes the gradients by FP32 rounding only."""
    from depth_mixer.ops.mhc.tile_kernels.mhc.sinkhorn_kernel import _mhc_sinkhorn_bwd, sinkhorn_bwd_token_block

    torch.manual_seed(42)
    logits = torch.randn(1000, N, N, device=device)
    grad = torch.randn_like(logits)
    largest = sinkhorn_bwd_token_block(N, 20)
    results = []
    for block in sorted({largest, max(1, largest // 2), max(1, largest // 4)}):
        out = torch.empty_like(logits)
        _mhc_sinkhorn_bwd(N, block, 20, 1e-6)(grad, logits, out)
        results.append(out)
    for out in results[1:]:
        assert ((out - results[0]).norm() / results[0].norm()).item() < 1e-6


def test_naive_ops_on_cpu():
    """The naive ops run anywhere; a connector configured as fused falls back to them on CPU."""
    from depth_mixer.layers.mhc import ManifoldHyperConnection

    torch.manual_seed(42)
    connector = ManifoldHyperConnection(64, mode='fused', layer_idx=1)
    streams = torch.randn(2, 8, N, 64)
    branch_input, post_mix, comb = connector.pre(streams)
    out = connector.post(torch.randn(2, 8, 64), streams, post_mix, comb)
    assert out.shape == streams.shape and torch.isfinite(out).all()
    # At initialization the projection is zero: the pre mix is sigmoid(+3) on stream layer_idx % n and
    # sigmoid(-3) on the others, and the residual mix is doubly stochastic.
    weights = torch.full((N,), torch.sigmoid(torch.tensor(-3.0)).item()) + 1e-6
    weights[1] = torch.sigmoid(torch.tensor(3.0)).item() + 1e-6
    assert_close('branch input', (streams * weights[:, None]).sum(-2), branch_input, 1e-5)
    assert torch.allclose(comb.sum(-2), torch.ones_like(comb.sum(-2)), atol=1e-4)
