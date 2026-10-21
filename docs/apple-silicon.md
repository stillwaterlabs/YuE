# YuE2 on Apple Silicon

This branch runs YuE2 on Apple Silicon Macs about twice as fast as upstream `main`, and makes MPS generation reproducible. It is a small set of changes on top of upstream `main` @ 1647252; the model, sampling and outputs format are unchanged.

## Install

macOS 14 or later, Python 3.12, an Apple Silicon Mac with enough unified memory for the 3B model (tested on 64 GB).

```bash
git clone -b apple-silicon https://github.com/zequnyu/YuE.git
cd YuE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
yue2 generate --device mps --request examples/song.json --output outputs/first-song
```

`examples/generate.py` selects CUDA, so use `yue2 generate --device mps` on a Mac. This branch pins torch 2.14.1 for every platform; use upstream `main` on Linux and Windows.

## What changes

1. **torch 2.14.1 instead of 2.10.0.** torch 2.10 MPS SDPA writes past its scratch buffers in BF16 decode once the KV cache holds 1,024 keys, so the same request and seed give different songs and attention can be wrong (#268). torch 2.13+ also fixes the causal-mask bug in #176, implements grouped-query attention in its MPS kernels, and no longer grows the MPS allocator cache without bound during decode.
2. **Native grouped-query attention on MPS.** `sdpa()` and `nar.attention()` pass `enable_gqa` instead of copying K/V with `repeat_interleave`.
3. **Decode head folding.** For single-token decode, the query heads that share a KV head are folded into the query axis (`[B, 16, 1, D]` → `[B, 8, 2, D]`), which keeps PyTorch on its vector kernel. Same keys, same weights; results match to BF16 rounding.
4. **Fused Metal attention for the acoustic stage.** `src/yue2/mps_attention.py` is a FlashAttention-style kernel compiled with `torch.mps.compile_shader`: 32-key blocks stream through threadgroup memory with an online softmax and FP32 accumulation, and only the output is written. It covers YuE2's 16/8-head, 128-dim BF16 acoustic attention, checks itself against an FP32 reference on first use, and other shapes fall back to SDPA.

## Benchmark

*Amazing Grace* (public domain, six stanzas; request in #265), seed 831001, 3 min 28 s of audio, 1,527-token prefix, 5,190 semantic tokens. MacBook Pro (Mac15,8), Apple M3 Max (40-core GPU), 64 GB, macOS 26.7.

| Stage | Upstream `main`, torch 2.10.0 | This branch, torch 2.14.1 |
|---|---:|---:|
| Score planning | 32 s (1,266 tokens) | 30 s (1,194 tokens)⁵ |
| Semantic decode | does not finish on 64 GB¹; 231 s with the cache released² | 146 s³ |
| Acoustic synthesis | 598 s | ≈225 s⁴ |
| Audio decoding | 9 s | 5 s |
| **Total** | **≈870 s** | **≈405 s** |

1. Real decode with sampling: the MPS driver allocation stays near 8 GB for 3,000 tokens, then grows by about 10 GB every 256 tokens and reaches 48.9 GB at 4,096 tokens, where the run was stopped. With torch 2.14.1 it stays at 7.9 GB for all 5,190 tokens.
2. Teacher-forced replay of the same tokens with `torch.mps.empty_cache()` every 32 steps (#265).
3. Teacher-forced replay, mean of three runs (142.8, 150.1, 144.1 s). Native `enable_gqa` without folding: 146.4, 156.1, 157.2 s.
4. Two measured neighbours of this branch's configuration, each one full synthesis: torch 2.14.1 with one native SDPA call per layer, 224.8 s; torch 2.10.0 with the Metal kernel, 225.1 s. This branch keeps the kernel because it was faster than the native call on longer songs in our own use.
5. Both about 39–40 tokens/s. The seeded score differs between torch versions, so its length differs too.

Totals are sums of the stage measurements, each taken after a cool-down; a back-to-back run on a warm laptop is slower.

## Validation

Against upstream `main` on torch 2.10.0, on the same machine:

- The upstream CI checks pass on torch 2.14.1: `yue2 --help`, `generate --help`, `batch --help`, `doctor`, the test suite, the instrumental tool test and the skill bundle. This branch adds MPS tests for decode folding and the Metal kernel (206 passed, 11 skipped).
- Precision against CPU FP32 is unchanged: over 400 decode steps the mean KL divergence of the codec distribution is 0.0006 on both versions and top-1 agreement is 97.3% (2.10.0) vs 97.5% (2.14.1); one acoustic velocity evaluation has 1.60% vs 1.62% relative error.
- The VAE decodes the same latent to audio that differs by at most 5.7 × 10⁻⁶ between versions.
- `examples/song.json` with planning, generated twice, now gives identical semantic tokens and latents (1,759 tokens). On upstream `main` the two runs diverge at token 286.

Results are rounding-level different from upstream, so a given seed produces a different song than it does on `main`. Upstream already notes that runtime versions can change a seeded generation.
