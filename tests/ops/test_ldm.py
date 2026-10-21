import pytest
import torch
import torch.nn.functional as F
from fla.utils import assert_close, device

from depth_mixer.ops.ldm import fused_recurrent_ldm, naive_recurrent_ldm


@pytest.mark.parametrize(
    ('S', 'B', 'T', 'G', 'K', 'D', 'recurrence', 'covariance_dtype'),
    [
        pytest.param(*test, id="S{}-B{}-T{}-G{}-K{}-D{}-{}-{}".format(*test))
        for test in [
            (9, 2, 64, 16, 8, 64, 'rls', torch.float32),
            (9, 2, 64, 16, 8, 64, 'rls', torch.bfloat16),
            (49, 1, 128, 16, 8, 64, 'rls', torch.float32),   # 400M: 2 * 24 + 1 writes
            (13, 2, 32, 16, 4, 128, 'rls', torch.float32),   # K=4, 1.5B group width
            (9, 2, 64, 1, 8, 1024, 'rls', torch.float32),    # one group
            (9, 2, 64, 16, 8, 64, 'delta_rule', torch.float32),
            (9, 2, 64, 16, 8, 64, 'outer_product', torch.float32),
        ]
    ],
)
def test_fused_recurrent_ldm(S, B, T, G, K, D, recurrence, covariance_dtype):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    k = F.normalize(torch.randn(S, B, T, G, K, device=device), dim=-1).to(dtype).requires_grad_()
    v = torch.randn(S, B, T, G, D, device=device).to(dtype).requires_grad_()
    q = F.normalize(torch.randn(S, B, T, G, K, device=device), dim=-1).to(dtype).requires_grad_()
    f = torch.sigmoid(torch.randn(S, 1, 1, G, device=device) + 4).requires_grad_()
    do = torch.randn(S, B, T, G, D, device=device)

    ref = naive_recurrent_ldm(k, v, f, q, recurrence=recurrence)
    ref_dk, ref_dv, ref_dq, ref_df = torch.autograd.grad((ref * do).sum(), (k, v, q, f))
    tri = fused_recurrent_ldm(k, v, f, q, recurrence=recurrence, covariance_dtype=covariance_dtype)
    tri_dk, tri_dv, tri_dq, tri_df = torch.autograd.grad((tri.float() * do).sum(), (k, v, q, f))

    # The fused step keeps M (and optionally P) in BF16, so it is held to BF16 tolerances.
    ratio = 0.01 if covariance_dtype == torch.float32 else 0.02
    assert_close('o', ref, tri.float(), ratio)
    assert_close('dk', ref_dk.float(), tri_dk.float(), ratio)
    assert_close('dv', ref_dv.float(), tri_dv.float(), ratio)
    assert_close('dq', ref_dq.float(), tri_dq.float(), ratio)
    assert_close('df', ref_df.float(), tri_df.float(), ratio)


def test_naive_ldm_rls_matches_least_squares():
    """Without forgetting, the RLS memory is the ridge regression of the values on the keys."""
    torch.manual_seed(0)
    S, K, D, alpha = 20, 8, 16, 1.0
    keys = torch.randn(S, 1, K, dtype=torch.float64)
    values = torch.randn(S, 1, D, dtype=torch.float64)
    queries = torch.randn(S, 1, K, dtype=torch.float64)
    forgetting = torch.ones(S, 1, dtype=torch.float64)
    reads, (memory, _) = naive_recurrent_ldm(
        keys.float(), values.float(), forgetting.float(), queries.float(), alpha=alpha, output_final_state=True,
    )
    X, Y = keys[:, 0], values[:, 0]
    ridge = torch.linalg.solve(X.T @ X + torch.eye(K, dtype=torch.float64) / alpha, X.T @ Y)
    torch.testing.assert_close(memory[0].double(), ridge, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ('S', 'N', 'G', 'K', 'D'),
    [
        pytest.param(*test, id="S{}-N{}-G{}-K{}-D{}".format(*test))
        for test in [
            (7, 2048, 4, 8, 64),
            (49, 16384, 16, 8, 64),    # 400M, micro-batch 8 x 2048
        ]
    ],
)
def test_chunk_ldm(S, N, G, K, D):
    """The CuTe kernels against the naive recurrence. Training launches need a tuned schedule for
    the GPU and shape; an untuned shape is skipped."""
    pytest.importorskip("cutlass")
    from depth_mixer.ops.ldm import chunk_ldm

    torch.manual_seed(42)
    k = F.normalize(torch.randn(S, N, G, K, device=device), dim=-1).requires_grad_()
    q = F.normalize(torch.randn(S, N, G, K, device=device), dim=-1).requires_grad_()
    v = torch.randn(S, N, G, D, device=device).to(torch.bfloat16).requires_grad_()
    scale = (2 * torch.sigmoid(torch.randn(S, N, G, device=device))).sqrt().requires_grad_()
    f = torch.sigmoid(torch.randn(S, N, G, device=device) + 4).requires_grad_()
    do = torch.randn(S, N, G, D, device=device)

    ref = naive_recurrent_ldm(k * scale[..., None], v.float() * scale[..., None], f, q)
    ref_grads = torch.autograd.grad((ref * do).sum(), (k, q, v, scale, f))
    try:
        tri = chunk_ldm(k, v, scale, f, q)
    except RuntimeError as error:
        if "no tuned schedule" in str(error):
            pytest.skip(str(error).splitlines()[0])
        raise
    tri_grads = torch.autograd.grad((tri.float() * do).sum(), (k, q, v, scale, f))
    assert_close('o', ref, tri.float(), 0.01)
    for name, ref_grad, tri_grad in zip(('dk', 'dq', 'dv', 'dscale', 'df'), ref_grads, tri_grads):
        assert_close(name, ref_grad.float(), tri_grad.float(), 0.01)
