"""Capture a per-layer reference for Gemma 4 12B: run transformers' single
Gemma4TextDecoderLayer (the gold reference — pins down the proportional/partial
RoPE) on a fixed random input, for one sliding layer (0) and one full-attention
layer (5). Dumps input hidden, output hidden, and meta so the Mojo gemma_layer
can be validated against it. Single-layer → fits in 24 GB. Run in the oracle env."""
import os, json, glob, struct
import numpy as np, torch
from safetensors import safe_open
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

D = os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-bf16/snapshots/afb7b215e9fe3b3eaef462b27d5c9d9b1ba0565b")
OUT = "tests/fixtures/gemma"
torch.manual_seed(0)

raw = json.load(open(os.path.join(D, 'config.json')))
tc = Gemma4TextConfig(**raw['text_config'])
print("layers", tc.num_hidden_layers, "hidden", tc.hidden_size, "head_dim", tc.head_dim,
      "layer_types[0,5]", tc.layer_types[0], tc.layer_types[5])

# map tensor name -> shard file
wmap = json.load(open(os.path.join(D, "model.safetensors.index.json")))["weight_map"]
def load(name):
    f = os.path.join(D, wmap[name])
    with safe_open(f, framework="pt") as h:
        return h.get_tensor(name).to(torch.float32)

rotary = Gemma4TextRotaryEmbedding(tc).to(torch.float32)
S = 12
hidden = torch.randn(1, S, tc.hidden_size, dtype=torch.float32) * 0.1
pos = torch.arange(S).unsqueeze(0)

def causal_mask(window=0):
    m = torch.full((S, S), float("-inf"))
    for i in range(S):
        for j in range(S):
            ok = j <= i and (window == 0 or i - j < window)
            if ok: m[i, j] = 0.0
    return m.view(1, 1, S, S)

for L in (0, 5):
    lt = tc.layer_types[L]
    layer = Gemma4TextDecoderLayer(tc, layer_idx=L).to(torch.float32).eval()
    pfx = f"language_model.model.layers.{L}."
    sd = {}
    for k in layer.state_dict().keys():
        full = pfx + k
        if full in wmap:
            sd[k] = load(full)
    missing = [k for k in layer.state_dict() if k not in sd]
    layer.load_state_dict(sd, strict=False)
    print(f"layer {L} ({lt}) loaded; missing(not-in-ckpt)={missing}")
    cos, sin = rotary(hidden, pos, layer_type=lt)
    window = tc.sliding_window if lt == "sliding_attention" else 0
    mask = causal_mask(window)
    with torch.no_grad():
        out = layer(hidden, position_embeddings=(cos, sin), attention_mask=mask, position_ids=pos)
        out = out[0] if isinstance(out, tuple) else out
    def dump(name, t):
        a = t.detach().numpy().astype(np.float32).ravel()
        with open(os.path.join(OUT, name), "wb") as f: f.write(a.tobytes())
    dump(f"layer{L}_in.bin", hidden)
    dump(f"layer{L}_out.bin", out)
    print(f"  layer {L} out: shape={tuple(out.shape)} mean={out.mean().item():.5f} std={out.std().item():.5f}")

json.dump({"S": S, "hidden": tc.hidden_size, "layers_captured": [0, 5],
           "layer_types": [tc.layer_types[0], tc.layer_types[5]]},
          open(os.path.join(OUT, "meta.json"), "w"))
print("wrote fixtures to", OUT)
