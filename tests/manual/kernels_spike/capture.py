"""Generate kernel-spike fixtures + validate the NumPy references vs HF.

Writes tests/fixtures/kernels/<name>/ : the named input .bin files (raw
little-endian float32, C-contiguous), expected.bin (NumPy reference), and
meta.txt (space-separated dims the Mojo harness reads).

  syn_rmsnorm / syn_matmul / syn_swiglu   small synthetic dims, COMMITTED so the
                                          gate runs from a clean checkout.
  real_rmsnorm / real_matmul / real_swiglu  full Qwen2 dims with real layer-0
                                          weights + activations, GITIGNORED
                                          (weights are large). Each is also
                                          cross-checked against HF's own output.

Run via `pixi run kernels-capture`.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as ref

FIX_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "kernels")
)
F32 = np.float32


def to_bf16(a):
    """Round f32 -> bf16 -> f32 (round-to-nearest-even). The engine keeps weights
    as bf16 on device and widens per element in the matmul (ARCHITECTURE §11 #12),
    so synthetic matmul/SwiGLU references must use bf16 weights to match the
    kernel. (Real fixtures use the model's already-bf16 weights, so this is a
    no-op for them.)"""
    u = np.ascontiguousarray(a, dtype=F32).view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


def save(name, arrays, meta):
    d = os.path.join(FIX_ROOT, name)
    os.makedirs(d, exist_ok=True)
    for k, a in arrays.items():
        np.ascontiguousarray(a, dtype=F32).tofile(os.path.join(d, k + ".bin"))
    with open(os.path.join(d, "meta.txt"), "w") as f:
        f.write(" ".join(str(x) for x in meta))


def synthetic(eps):
    rng = np.random.RandomState(0)

    T, H = 8, 128
    x = rng.randn(T, H).astype(F32)
    w = (rng.randn(H) * 0.1 + 1.0).astype(F32)
    save("syn_rmsnorm", {"x": x, "w": w, "expected": ref.rmsnorm(x, w, eps)}, [T, H])

    M, K, N = 8, 128, 256
    x = rng.randn(M, K).astype(F32)
    W = to_bf16(rng.randn(N, K) * 0.05)   # weights live as bf16 on device
    b = (rng.randn(N) * 0.05).astype(F32)
    save("syn_matmul", {"x": x, "W": W, "b": b, "expected": ref.matmul_bias(x, W, b)}, [M, K, N, 1])

    T, H, I = 8, 128, 512
    x = rng.randn(T, H).astype(F32)
    wg = to_bf16(rng.randn(I, H) * 0.05)
    wu = to_bf16(rng.randn(I, H) * 0.05)
    wd = to_bf16(rng.randn(H, I) * 0.05)
    save(
        "syn_swiglu",
        {"x": x, "w_gate": wg, "w_up": wu, "w_down": wd, "expected": ref.swiglu_mlp(x, wg, wu, wd)},
        [T, H, I],
    )
    print(f"synthetic: rmsnorm[{8}x{128}] matmul[{8}x{128}->{256}] swiglu[{8}x{128}x{512}]")


def real(eps):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").float().eval()

    L = model.model.layers[0]
    cap = {}
    hooks = [
        L.register_forward_pre_hook(lambda m, a: cap.__setitem__("h_in", a[0].detach().float()[0].numpy())),
        L.input_layernorm.register_forward_hook(lambda m, i, o: cap.__setitem__("ln1", o.detach().float()[0].numpy())),
        L.self_attn.q_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("q", o.detach().float()[0].numpy())),
        L.post_attention_layernorm.register_forward_hook(lambda m, i, o: cap.__setitem__("ln2", o.detach().float()[0].numpy())),
        L.mlp.register_forward_hook(lambda m, i, o: cap.__setitem__("mlp", o.detach().float()[0].numpy())),
    ]
    enc = tok.apply_chat_template(
        [{"role": "user", "content": "In one sentence, what is the capital of France?"}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    with torch.no_grad():
        model(enc["input_ids"])
    for h in hooks:
        h.remove()

    def npw(mod):
        return mod.weight.detach().float().numpy()

    T, Hd = cap["h_in"].shape
    ln1_w = npw(L.input_layernorm)
    qw = npw(L.self_attn.q_proj)
    qb = L.self_attn.q_proj.bias.detach().float().numpy()
    wg, wu, wd = npw(L.mlp.gate_proj), npw(L.mlp.up_proj), npw(L.mlp.down_proj)
    Inter = wg.shape[0]
    N = qw.shape[0]

    # references
    r_ln1 = ref.rmsnorm(cap["h_in"], ln1_w, eps)
    r_q = ref.matmul_bias(cap["ln1"], qw, qb)
    r_mlp = ref.swiglu_mlp(cap["ln2"], wg, wu, wd)

    # cross-check vs HF
    checks = {
        "rmsnorm": float(np.abs(r_ln1 - cap["ln1"]).max()),
        "matmul": float(np.abs(r_q - cap["q"]).max()),
        "swiglu": float(np.abs(r_mlp - cap["mlp"]).max()),
    }
    ok = True
    for k, v in checks.items():
        st = "OK" if v < 1e-3 else "FAIL"
        print(f"real {k}: NumPy-ref vs HF max_abs={v:.3e} [{st}]")
        ok = ok and v < 1e-3
    if not ok:
        raise SystemExit("a NumPy reference does NOT match HF — fix it")

    save("real_rmsnorm", {"x": cap["h_in"], "w": ln1_w, "expected": r_ln1}, [T, Hd])
    save("real_matmul", {"x": cap["ln1"], "W": qw, "b": qb, "expected": r_q}, [T, Hd, N, 1])
    save("real_swiglu", {"x": cap["ln2"], "w_gate": wg, "w_up": wu, "w_down": wd, "expected": r_mlp}, [T, Hd, Inter])
    print(f"real: saved layer-0 fixtures T={T} H={Hd} I={Inter}")


def main():
    eps = 1e-6
    os.makedirs(FIX_ROOT, exist_ok=True)
    synthetic(eps)
    try:
        from transformers import AutoModelForCausalLM  # noqa: F401
    except Exception:
        print("transformers unavailable — skipping real fixtures")
        return
    eps = 1e-6
    real(eps)
    print("OK: NumPy references match HF on real activations")


if __name__ == "__main__":
    main()
