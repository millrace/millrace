"""Tokenize the assay corpus into BOS-prepended windows for e2b_corpus_ppl.mojo.

Replicates assay's windowing (group lines up to ~1800 chars, max 24 windows),
tokenizes each window with the Gemma tokenizer, prepends <bos> (id 2), and writes
one space-separated token-id line per window to .scratch/e2b_corpus_ids.txt.

Run in an env that has `transformers` with the Gemma tokenizer (e.g. the
max-backend pixi env):  python tokenize_corpus.py
"""
import sys
from transformers import AutoTokenizer

SNAP = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/22a2753af6114b0c364f09921771b458e40b9e09"
CORPUS = "/Users/mseritan/dev/millrace/assay/corpus.txt"
OUT = "/Users/mseritan/dev/millrace/inference-server/.scratch/e2b_corpus_ids.txt"
WINDOW_CHARS, MAX_WINDOWS = 1800, 24


def main():
    tok = AutoTokenizer.from_pretrained(SNAP)
    body = open(CORPUS, encoding="utf-8").read()
    lines = body.split("\n")
    windows, cur = [], ""
    for ln in lines:
        if len(cur) + len(ln) > WINDOW_CHARS and len(cur) > 40:
            windows.append(cur); cur = ""
        cur += ln + "\n"
    if len(cur.strip()) > 40:
        windows.append(cur)
    windows = windows[:MAX_WINDOWS]
    out, total = [], 0
    for w in windows:
        ids = [2] + tok(w, add_special_tokens=False)["input_ids"]  # <bos> + tokens
        out.append(" ".join(str(i) for i in ids)); total += len(ids) - 1
    open(OUT, "w").write("\n".join(out))
    print(f"wrote {len(windows)} windows, {total} scored tokens -> {OUT}")


if __name__ == "__main__":
    main()
