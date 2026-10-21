"""The fused Metal acoustic attention matches dense FP64 attention to BF16 rounding."""
import pytest
import torch

from yue2 import mps_attention, nar

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")


def dense(q, k, v):
    groups = q.shape[1] // k.shape[1]
    query = q.double().transpose(0, 1)
    key = k.double().repeat_interleave(groups, 1).transpose(0, 1)
    value = v.double().repeat_interleave(groups, 1).transpose(0, 1)
    return ((query @ key.transpose(-1, -2)) * q.shape[-1] ** -0.5).softmax(-1).matmul(value).transpose(0, 1)


def inputs(rows, keys, seed):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(rows, 16, 128, generator=generator).bfloat16()
    k = torch.randn(keys, 8, 128, generator=generator).bfloat16()
    v = torch.randn(keys, 8, 128, generator=generator).bfloat16()
    return q, k, v


@pytest.mark.parametrize("rows,keys", [(1, 1), (31, 33), (37, 45), (100, 1000), (257, 2049)])
def test_kernel_matches_dense_attention_including_ragged_tiles(rows, keys):
    q, k, v = inputs(rows, keys, rows * 7919 + keys)
    expected = dense(q, k, v)
    result = mps_attention.attention(q.to("mps"), k.to("mps"), v.to("mps")).cpu().double()
    assert result.shape == expected.shape
    # One FP32 result rounded to BF16: error stays within a BF16 step of the output scale.
    assert (result - expected).abs().max() <= 2 ** -7


def test_acoustic_attention_uses_the_kernel_and_causal_calls_do_not():
    q, k, v = (t.to("mps") for t in inputs(37, 45, 3))
    torch.testing.assert_close(nar.attention(q, k, v), mps_attention.attention(q, k, v), atol=0, rtol=0)
    qc, kc, vc = (t.to("mps") for t in inputs(9, 9, 4))
    expected = dense(qc.cpu(), kc.cpu(), vc.cpu())
    causal = nar.attention(qc, kc, vc, causal=True).cpu().double()
    assert not torch.allclose(causal, expected, atol=1e-2)  # causal masking changed the result


def test_unsupported_shapes_fall_back_to_sdpa():
    q, k, v = (t.to("mps") for t in inputs(9, 9, 5))
    q4, k4 = q[:, :4, :64].contiguous(), k[:, :2, :64].contiguous()
    assert not mps_attention.supported(q4, k4, k4)
    expected = dense(q4.cpu(), k4.cpu(), k4.cpu())
    torch.testing.assert_close(nar.attention(q4, k4, k4).cpu().double(), expected, atol=2e-2, rtol=2e-2)
