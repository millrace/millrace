"""Capture the embedding oracle for Qwen3-Embedding-0.6B.

The from-scratch Mojo embedder must reproduce, token-for-token / float-for-float,
what this reference produces. This script pins down BOTH:

  1. the exact architecture config (dims, head_dim, eps, rope_theta, qk-norm
     presence, tie_word_embeddings) — read empirically, not from memory; and
  2. reference embeddings for a handful of strings — last-token pooled +
     L2-normalized, the official Qwen3-Embedding recipe.

Writes tests/fixtures/embed/ (GITIGNORED — the checkpoint is ~1.2 GB):

  config.txt    key=value lines (hidden, layers, n_heads, n_kv, head_dim,
                intermediate, vocab, rms_eps, rope_theta, tie_word_embeddings,
                eos_id, appends_eos)
  ids.txt       one line per sample: the exact token ids fed to the model
  expected.bin  for each sample: the embedding dim D as float32, then D floats
                (the L2-normalized last-token embedding) — the ground truth.

Run via `pixi run -e oracle embed-capture`.
"""

import os
import struct

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "Qwen/Qwen3-Embedding-0.6B"
FIX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "embed"))

# A few documents + one instructed query, mirroring real vault usage.
SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Insurance policy renewal date: 2026-09-15.",
    "Instruct: Given a question, retrieve passages that answer it.\nQuery: when does my insurance renew?",
]


def last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # Official Qwen3-Embedding pooling: the last non-pad token's hidden state.
    left_padded = attn_mask[:, -1].sum() == attn_mask.shape[0]
    if left_padded:
        return last_hidden[:, -1]
    lengths = attn_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0]), lengths]


def main() -> None:
    os.makedirs(FIX, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    cfg = model.config

    def rope_theta(c) -> float:
        # transformers moved rope config around across versions.
        if getattr(c, "rope_theta", None) is not None:
            return c.rope_theta
        rp = getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None)
        if isinstance(rp, dict) and rp.get("rope_theta") is not None:
            return rp["rope_theta"]
        return 1000000.0

    # Empirically read the architecture — do NOT trust remembered constants.
    config = {
        "hidden": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "n_heads": cfg.num_attention_heads,
        "n_kv": cfg.num_key_value_heads,
        "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
        "intermediate": cfg.intermediate_size,
        "vocab": cfg.vocab_size,
        "rms_eps": cfg.rms_norm_eps,
        "rope_theta": rope_theta(cfg),
        "tie_word_embeddings": int(bool(getattr(cfg, "tie_word_embeddings", False))),
        "eos_id": tok.eos_token_id,
    }

    # Does the tokenizer append EOS? Check on a trivial input.
    probe = tok("hi", return_tensors="pt")["input_ids"][0].tolist()
    config["appends_eos"] = int(len(probe) > 0 and probe[-1] == tok.eos_token_id)

    ids_lines = []
    vectors = []
    for text in SAMPLES:
        batch = tok(text, return_tensors="pt")
        ids_lines.append(" ".join(str(i) for i in batch["input_ids"][0].tolist()))
        with torch.no_grad():
            out = model(**batch)
        emb = last_token_pool(out.last_hidden_state, batch["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)[0]
        vectors.append(emb.float().numpy())

    with open(os.path.join(FIX, "config.txt"), "w") as f:
        for k, v in config.items():
            f.write(f"{k}={v}\n")
    with open(os.path.join(FIX, "ids.txt"), "w") as f:
        f.write("\n".join(ids_lines) + "\n")
    with open(os.path.join(FIX, "expected.bin"), "wb") as f:
        for v in vectors:
            f.write(struct.pack("<i", v.shape[0]))
            f.write(v.astype("<f4").tobytes())

    print("== Qwen3-Embedding-0.6B config ==")
    for k, v in config.items():
        print(f"  {k} = {v}")
    print(f"\nembedding dim: {vectors[0].shape[0]}")
    print(f"sample 0 first 8: {np.round(vectors[0][:8], 5).tolist()}")
    print(f"wrote fixtures -> {FIX}")


if __name__ == "__main__":
    main()
