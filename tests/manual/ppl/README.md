# Perplexity harnesses

Manual harnesses built while measuring model quality (perplexity) and chasing the
**Gemma-4-12B int4 PPL bug**. The user-facing PPL tool is the sibling
[`millfolio/assay`](../../../../assay) repo (pure-Mojo: download corpus → for each
served model launch server, score PPL via `/v1/completions`, stop). These are the
lower-level, model-internal probes that `assay` can't do (e2b isn't served; HF
oracle; per-layer/per-position dumps).

## The numbers (Time Machine corpus, 24 windows, ~11k scored tokens)

| model | corpus PPL | status |
|---|---|---|
| Qwen2.5-3B bf16 | 9.3 | ✓ |
| Qwen2.5-3B int4 | 13.6 | ✓ |
| Gemma e2b bf16 | 231.9 | ✓ works |
| Gemma e2b int4 | 220.5 | ✓ works |
| Gemma-4-12B-QAT int4 | 1666 | ✗ broken |
| Gemma-4-12B int4 (RTN) | 8593 | ✗ broken |

## The open bug (Gemma-4-12B, `gemma.mojo` dense path)

The 12B produces **under-confident** logits: correct argmax (greedy chat works) but
a too-flat softmax (PPL broken). Ruled out with evidence — tokenization (exact HF
match), interior-position forward (self-consistent), embed/LM-head scaling, norms
(actual-scale, no `(1+w)`), all constants, GQA mapping (Qwen-validated), and the
**shared kernels**: millfolio e2b matches HF gemma4-e2b per-position to **<1%**. int4
is a red herring (graceful for e2b; finer group-32 made the 12B *worse*). The tiny
~2B e2b (PPL 221) even **beats** the 12B (1666) — a 2B can't legitimately outscore a
12B, so it's a forward bug, not quantization. Per-layer hidden magnitudes are healthy
(no blow-up), so it's a representation-**direction** degradation. Pinning the exact op
needs a **bf16 12B HF oracle**, which doesn't fit 24 GB RAM (needs ~48 GB). See the
`assay-ppl-harness` memory note for the full trail.

## Harnesses

Mojo harnesses build with `pixi run -- mojo build <file> -I src -I ../jinja2.mojo/src -I ../flare -o build/<name>` then run the binary. They hardcode local checkpoint snapshot paths.

| file | what it does |
|---|---|
| `e2b_ppl.mojo` | millfolio e2b per-position logprobs for a fixed 27-token sentence (bf16; flip `load_e2b_weights(...,True)` for int4). Matches `e2b_ppl_ref.py` to <1%. |
| `e2b_corpus_ppl.mojo` | millfolio e2b **corpus** PPL over the windows in `.scratch/e2b_corpus_ids.txt`. Arg `int4` selects int4 (default bf16). |
| `g12_layerdump.mojo` | per-layer hidden-state magnitude trace through the 12B int4 forward + corpus-sentence PPL — shows magnitudes are healthy (no blow-up at full-attention layers). |
| `e2b_ppl_ref.py` | **HF** gemma4-e2b per-position logprobs reference (the oracle). Run in an env with `transformers` ≥5.9 that has the `gemma4` model (the max-backend pixi env): `python e2b_ppl_ref.py`. |
| `ppl_selfconsistency.py` | hits a running server's `/v1/completions`: proves `full[k] == prefix[k]` (the multi-position forward is self-consistent — interior positions aren't the bug). |
| `tokenize_corpus.py` | regenerates `.scratch/e2b_corpus_ids.txt` (assay windowing + Gemma tokenizer + `<bos>`). Run in the max-backend env. |

### Typical flow

```sh
# 1. (re)generate tokenized windows  — max-backend transformers env
python tests/manual/ppl/tokenize_corpus.py

# 2. millfolio e2b corpus PPL          — inference-server (Metal GPU)
pixi run -- mojo build tests/manual/ppl/e2b_corpus_ppl.mojo \
    -I src -I ../jinja2.mojo/src -I ../flare -o build/e2b_corpus_ppl
./build/e2b_corpus_ppl        # bf16
./build/e2b_corpus_ppl int4   # int4

# 3. HF oracle for the same sentence  — max-backend transformers env
python tests/manual/ppl/e2b_ppl_ref.py
```

## Next step to fix the 12B

Get the bf16 12B onto a ≥48 GB machine (CPU is fine; the oracle is portable), extend
`e2b_ppl_ref.py` to the 12B (`Gemma4ForCausalLM`, text-only), dump per-layer hidden
states + per-position logprobs, and diff against `g12_layerdump.mojo` — the first
layer whose direction diverges is the bug.
