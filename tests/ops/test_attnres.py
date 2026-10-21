import pytest
import torch
from fla.utils import assert_close, device

from depth_mixer.ops.attnres import fused_attnres, naive_attnres


@pytest.mark.parametrize(
    ('L', 'B', 'T', 'D', 'fuse_output_norm', 'dtype'),
    [
        pytest.param(*test, id="L{}-B{}-T{}-D{}-onorm{}-{}".format(*test))
        for test in [
            (2, 2, 256, 1024, True, torch.bfloat16),
            (9, 2, 256, 1024, True, torch.bfloat16),     # 400M Block AttnRes, final mixture
            (49, 1, 128, 1024, True, torch.bfloat16),    # 400M Full AttnRes, final mixture
            (9, 2, 256, 2048, False, torch.bfloat16),
            (5, 1, 1000, 1000, True, torch.float32),
        ]
    ],
)
def test_attnres(L, B, T, D, fuse_output_norm, dtype):
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    residuals = [torch.randn(B, T, D, dtype=dtype, device=device).requires_grad_(True) for _ in range(L)]
    query = (0.1 * torch.randn(D, dtype=dtype, device=device)).requires_grad_(True)
    rms_weight = (1 + 0.1 * torch.randn(D, dtype=dtype, device=device)).requires_grad_(True)
    output_rms_weight = (
        (1 + 0.1 * torch.randn(D, dtype=dtype, device=device)).requires_grad_(True) if fuse_output_norm else None
    )
    inputs = (query, rms_weight, *residuals) + ((output_rms_weight,) if fuse_output_norm else ())
    do = torch.randn(B, T, D, dtype=dtype, device=device)

    outputs = {}
    for name, fn in (('ref', naive_attnres), ('tri', fused_attnres)):
        o, p = fn(query=query, residuals=residuals, rms_weight=rms_weight, output_rms_weight=output_rms_weight,
                  key_rms_eps=1e-6, return_weights=True)
        grads = torch.autograd.grad((o * do).sum(), inputs)
        outputs[name] = (o, p, grads)

    (ref, ref_p, ref_grads), (tri, tri_p, tri_grads) = outputs['ref'], outputs['tri']
    assert_close('o', ref, tri, 0.005)
    assert_close('p', ref_p, tri_p, 0.005)
    names = ['dq', 'dw', *[f'dres{i}' for i in range(L)]] + (['dow'] if fuse_output_norm else [])
    for name, ref_grad, tri_grad in zip(names, ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 0.01)
