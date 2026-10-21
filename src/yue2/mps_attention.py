"""Fused non-causal grouped-query attention kernel for Apple GPUs.

On MPS, PyTorch runs long-query SDPA as an MPSGraph that writes the full FP32
score matrix to memory several times. This Metal kernel streams 32-key blocks
through threadgroup memory with an online softmax (the FlashAttention scheme)
and writes only the output. Scores, softmax and accumulation are FP32; inputs
and output are BF16, as with SDPA. Results match SDPA to rounding, not bit for
bit, because the softmax is accumulated block by block.
"""
import torch

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;
constant constexpr uint D = 128, BK = 32, NSG = 8, KVH = 8, H = 16, SD = D + 8;
#define UNROLL _Pragma("clang loop unroll(full)")
typedef simdgroup_matrix<float, 8, 8> frag;

kernel void flash_gqa(device const bfloat* Q [[buffer(0)]], device const ushort* K [[buffer(1)]],
                      device const ushort* V [[buffer(2)]], device bfloat* O [[buffer(3)]],
                      constant uint& n [[buffer(4)]], constant uint& L [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup ushort KS[BK * SD], VS[BK * SD];
  const uint g = tg.y, h = 2 * g + (sg & 1), row0 = tg.x * (NSG / 2) * 8 + (sg >> 1) * 8;
  const float scale = 0.08838834764831845f;
  const ushort qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
  const uint p = row0 + fm;

  frag qf[D / 8], of[D / 8], sf[BK / 8], bf;
  UNROLL for (uint k = 0; k < D / 8; k++) {
    thread auto& e = qf[k].thread_elements();
    e[0] = p < n ? float(Q[(p * H + h) * D + k * 8 + fn]) : 0.0f;
    e[1] = p < n ? float(Q[(p * H + h) * D + k * 8 + fn + 1]) : 0.0f;
    of[k] = frag(0);
  }
  float m = -INFINITY, l = 0.0f;
  for (uint k0 = 0; k0 < L; k0 += BK) {
    UNROLL for (uint it = 0; it < BK * D / (NSG * 32 * 4); it++) {
      uint e = (lid + it * NSG * 32) * 4, i = e / D, d = e % D, key = k0 + i;
      ushort4 kv = 0, vv = 0;
      if (key < L) {
        kv = *(device const ushort4*)(K + (key * KVH + g) * D + d);
        vv = *(device const ushort4*)(V + (key * KVH + g) * D + d);
      }
      *(threadgroup ushort4*)(KS + i * SD + d) = kv;
      *(threadgroup ushort4*)(VS + i * SD + d) = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    UNROLL for (uint j = 0; j < BK / 8; j++) sf[j] = frag(0);
    UNROLL for (uint k = 0; k < D / 8; k++)
      UNROLL for (uint j = 0; j < BK / 8; j++) {
        thread auto& e = bf.thread_elements();   // B = K^T tile: B[r][c] = K[key c][dim r]
        e[0] = float(as_type<bfloat>(KS[(j * 8 + fn) * SD + k * 8 + fm]));
        e[1] = float(as_type<bfloat>(KS[(j * 8 + fn + 1) * SD + k * 8 + fm]));
        simdgroup_multiply_accumulate(sf[j], qf[k], bf, sf[j]);
      }
    float rmax = -INFINITY;
    UNROLL for (uint j = 0; j < BK / 8; j++) {
      thread auto& e = sf[j].thread_elements();
      UNROLL for (uint c = 0; c < 2; c++) {
        float s = k0 + j * 8 + fn + c < L ? e[c] * scale : -INFINITY;
        e[c] = s; rmax = max(rmax, s);
      }
    }
    rmax = max(rmax, simd_shuffle_xor(rmax, 1)); rmax = max(rmax, simd_shuffle_xor(rmax, 8));
    float mn = max(m, rmax), factor = precise::exp(m - mn), rsum = 0.0f;
    UNROLL for (uint j = 0; j < BK / 8; j++) {
      thread auto& e = sf[j].thread_elements();
      UNROLL for (uint c = 0; c < 2; c++) { float pr = precise::exp(e[c] - mn); e[c] = pr; rsum += pr; }
    }
    rsum += simd_shuffle_xor(rsum, 1); rsum += simd_shuffle_xor(rsum, 8);
    l = l * factor + rsum; m = mn;
    UNROLL for (uint k = 0; k < D / 8; k++) { thread auto& e = of[k].thread_elements(); e[0] *= factor; e[1] *= factor; }
    UNROLL for (uint j = 0; j < BK / 8; j++)
      UNROLL for (uint k = 0; k < D / 8; k++) {
        thread auto& e = bf.thread_elements();   // B = V tile: B[r][c] = V[key r][dim c]
        e[0] = float(as_type<bfloat>(VS[(j * 8 + fm) * SD + k * 8 + fn]));
        e[1] = float(as_type<bfloat>(VS[(j * 8 + fm) * SD + k * 8 + fn + 1]));
        simdgroup_multiply_accumulate(of[k], sf[j], bf, of[k]);
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (p < n) {
    float inv = 1.0f / l;
    UNROLL for (uint k = 0; k < D / 8; k++) {
      thread auto& e = of[k].thread_elements();
      O[(p * H + h) * D + k * 8 + fn] = bfloat(e[0] * inv);
      O[(p * H + h) * D + k * 8 + fn + 1] = bfloat(e[1] * inv);
    }
  }
}
"""
_lib = None


def supported(q, k, v):
    """The kernel is specialised for YuE2's acoustic attention shapes."""
    return (q.device.type == "mps" and q.dtype == k.dtype == v.dtype == torch.bfloat16 and
            q.ndim == 3 and q.shape[1:] == (16, 128) and k.shape[1:] == (8, 128) and v.shape == k.shape)


def _library():
    global _lib
    if _lib is None:
        library = torch.mps.compile_shader(SOURCE)
        _self_test(library)
        _lib = library
    return _lib


def _launch(library, q, k, v):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    library.flash_gqa(q, k.view(torch.int16), v.view(torch.int16), out, q.shape[0], k.shape[0],
                      threads=[(q.shape[0] + 31) // 32 * 256, 8, 1], group_size=[256, 1, 1])
    return out


def _self_test(library):
    """Reject GPUs whose simdgroup fragment layout differs from the one the kernel assumes."""
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(45, 16, 128, generator=generator).bfloat16()
    k = torch.randn(77, 8, 128, generator=generator).bfloat16()
    v = torch.randn(77, 8, 128, generator=generator).bfloat16()
    expected = _reference(q, k, v)
    result = _launch(library, q.to("mps"), k.to("mps"), v.to("mps")).cpu().float()
    error = (result - expected).abs().max().item()
    if not error < 2e-2:
        raise RuntimeError(f"Metal attention self-test failed on this GPU (max error {error:.3g}); "
                           "use attention='sdpa'")


def _reference(q, k, v):
    groups = q.shape[1] // k.shape[1]
    query, key = q.float().transpose(0, 1), k.float().repeat_interleave(groups, 1).transpose(0, 1)
    value = v.float().repeat_interleave(groups, 1).transpose(0, 1)
    weights = (query @ key.transpose(-1, -2) * q.shape[-1] ** -0.5).softmax(-1)
    return (weights @ value).transpose(0, 1)


def attention(q, k, v):
    """Attend q [n,16,128] to k/v [L,8,128] without a mask; returns [n,16,128]."""
    return _launch(_library(), q, k, v)
