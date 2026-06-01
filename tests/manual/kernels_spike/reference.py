"""From-scratch NumPy references for the Qwen2 building-block kernels.

These define "correct" for the Phase-2 GPU kernels (ARCHITECTURE.md §3, §6):
RMSNorm, dense matmul + bias (projections / LM head), and the SwiGLU MLP. All
float32 to match a CPU/f32 run and the Mojo GPU/f32 kernels.
"""

import numpy as np

DTYPE = np.float32


def rmsnorm(x, w, eps):
    """x: [T, H], w: [H]  ->  [T, H].  Qwen2 RMSNorm (variance in f32)."""
    x = x.astype(DTYPE)
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    xn = x / np.sqrt(var + np.float32(eps))
    return (w.astype(DTYPE) * xn).astype(DTYPE)


def matmul_bias(x, W, b=None):
    """x: [T, K], W: [N, K] (torch layout), b: [N] or None  ->  [T, N] = x @ W.T + b."""
    y = x.astype(DTYPE) @ W.astype(DTYPE).T
    if b is not None:
        y = y + b.astype(DTYPE)
    return y.astype(DTYPE)


def silu(x):
    x = x.astype(DTYPE)
    return (x / (1.0 + np.exp(-x))).astype(DTYPE)


def swiglu_mlp(x, w_gate, w_up, w_down):
    """x: [T, H]; w_gate/w_up: [I, H]; w_down: [H, I]  ->  [T, H]."""
    g = silu(matmul_bias(x, w_gate))
    u = matmul_bias(x, w_up)
    return matmul_bias((g * u).astype(DTYPE), w_down)
