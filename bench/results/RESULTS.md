# Cross-engine benchmark: millrace vs MLX vs Ollama (Qwen2.5, Apple M4)

Measured with `bench/bench.py` (two-point method: prefill = T(1), decode tok/s =
(tok@N − tok@1)/(T(N) − T(1)); per-request nonce defeats prompt caches; median of
5, 1 warmup, temp 0, max_tokens 128).

**Quantization (not identical — read accordingly):**
- millrace: 0.5B **bf16**, 3B **group-128 int4** (this repo's quantizer)
- MLX: `mlx-community/*-4bit`; Ollama: `qwen2.5:*` (Q4_K_M). Both ~4-bit.

The **3B row is the fair fight** (all ~4-bit) and was run **isolated** (one engine
resident at a time — no shared unified-memory pressure). The 0.5B row had all
three loaded at once (slightly soft absolutes) and pits our bf16 against their
4-bit, so it flatters them on decode; treat it as directional.

## Qwen2.5-3B  (all ~4-bit, isolated — the apples-to-apples comparison)

| metric                     | millrace int4 | MLX 4-bit | Ollama 4-bit |
|----------------------------|--------------:|----------:|-------------:|
| decode tok/s               |      **10.5** |      51.8 |         46.5 |
| prefill, 71-tok prompt     |     552 ms    |    220 ms |       165 ms |
| prefill, 104-tok prompt    |     830 ms    |    276 ms |       221 ms |
| prefill, 1570-tok prompt   |  **22 237 ms**|   2773 ms |      2896 ms |
| cold→warm prefix reuse     |        7.6×   |     15.1× |        22.9× |

## Qwen2.5-0.5B  (ours bf16 vs their 4-bit; non-isolated — directional)

| metric                     | millrace bf16 | MLX 4-bit | Ollama 4-bit |
|----------------------------|--------------:|----------:|-------------:|
| decode tok/s               |       22.4    |     222.0 |        164.6 |
| prefill, 71-tok prompt     |       59 ms   |     94 ms |        63 ms |
| prefill, 1570-tok prompt   |     2103 ms   |    474 ms |       486 ms |
| cold→warm prefix reuse     |      10.8×    |      4.9× |         7.5× |

## What it says about millrace

- **Decode is ~5× slower than MLX/Ollama even at matched 4-bit (3B: 10.5 vs ~50
  tok/s).** Our validated int4 GEMV is ~2× the bf16 kernel, but that win is hidden
  at the server level: decode is **per-token kernel-launch-overhead bound** — ~36
  layers × many tiny Metal dispatches/token + non-incremental generation. MLX and
  llama.cpp fuse/batch dispatches; that's the gap, not the matmul.
- **Long-prompt prefill is the worst weakness (1570 tok: 22 s vs ~2.8 s, ~8×).**
  The simd-matrix GEMM is fast, but attention over a long context (O(T²), 36
  layers) is not — it dominates and doesn't scale like the mature kernels.
- **Short-prompt prefill is within ~2.5–3×** (552 vs ~200 ms) — closer than decode.
- **Prefix caching works well** (3B 7.6×, 0.5B 10.8×) and is a genuine strength for
  multi-turn agent use, though MLX/Ollama also cache and reach higher ratios.

**Takeaways for where to invest:** (1) kernel fusion / fewer per-token dispatches
is the single biggest decode lever (would close most of the 5× before any further
quant); (2) a scalable long-context attention kernel for prefill; (3) incremental
streaming so first-token latency reflects prefill, not full generation. int4
landed the memory + matmul wins; the remaining gap is dispatch overhead, not
arithmetic.

Raw JSON: `qwen3b_{millrace,mlx,ollama}.json`, `qwen05b.json`.
