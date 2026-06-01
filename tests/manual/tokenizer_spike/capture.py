"""Dump Qwen2 tokenizer tables (resolved to ids) + expected encodings.

Key trick (keeps the Mojo side free of unicode): the GPT-2 byte<->unicode map is
a bijection and every BPE symbol — including all 256 single bytes — is itself a
vocab token. So we resolve everything to integer ids in Python and Mojo runs BPE
purely on id lists.

Writes tests/fixtures/tokenizer/ (GITIGNORED — large, derived from HF files):
  vocab.tsv    "<id>\\t<hexbytes>"  per vocab entry (hex of the token's bytes)
  merges.tsv   "<leftid> <rightid> <mergedid>"  per merge, in rank order
  specials.tsv "<id>\\t<text>"  per added special token (matched verbatim)
  prompts/p<i>.txt  raw prompt bytes
  expected.tsv "<i>\\t<id> <id> ..."  transformers' token ids for prompt i

Run via `pixi run tok-capture`.
"""

import glob
import os

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FIX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "tokenizer"))


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def main():
    os.makedirs(FIX, exist_ok=True)
    os.makedirs(os.path.join(FIX, "prompts"), exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    def tok_bytes(s):
        return bytes(u2b[ch] for ch in s)

    vocab = tok.get_vocab()  # unicode token -> id
    id_to_token = {i: t for t, i in vocab.items()}

    # vocab.tsv
    with open(os.path.join(FIX, "vocab.tsv"), "w") as f:
        for i in range(len(id_to_token)):
            if i not in id_to_token:
                continue
            t = id_to_token[i]
            # special/added tokens are real strings, not byte-mapped; skip here
            # (handled via specials.tsv). Detect: chars not in u2b.
            if any(ch not in u2b for ch in t):
                f.write(f"{i}\t\n")  # placeholder; not a byte-BPE token
                continue
            f.write(f"{i}\t{tok_bytes(t).hex()}\n")

    # merges.tsv (read raw merges.txt from the snapshot)
    snap = glob.glob(
        os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}/snapshots/*")
    )[0]
    merges_path = os.path.join(snap, "merges.txt")
    n_merges = 0
    with open(merges_path, encoding="utf-8") as mf, open(os.path.join(FIX, "merges.tsv"), "w") as out:
        for line in mf:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            left, right = line.split(" ")
            merged = left + right
            if left not in vocab or right not in vocab or merged not in vocab:
                continue
            out.write(f"{vocab[left]} {vocab[right]} {vocab[merged]}\n")
            n_merges += 1

    # specials.tsv
    specials = tok.added_tokens_decoder  # id -> AddedToken
    with open(os.path.join(FIX, "specials.tsv"), "w") as f:
        for i, atok in specials.items():
            f.write(f"{i}\t{str(atok.content)}\n")

    # prompts + expected ids
    chat = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True, tokenize=False,
    )
    prompts = [
        chat,
        "Hello world",
        "hello   world",     # multiple spaces
        "The year 2024 was great!!!",   # digits + punctuation run
        "  leading and trailing   ",
        "Mixing CASE and punctuation: a,b.c?d!",
        "Line one\nLine two\n\nLine four",
        "<|im_start|>user\nHi there<|im_end|>\n",
    ]
    with open(os.path.join(FIX, "expected.tsv"), "w") as ef:
        ef.write(f"{len(prompts)}\n")
        for i, p in enumerate(prompts):
            with open(os.path.join(FIX, "prompts", f"p{i}.txt"), "w") as pf:
                pf.write(p)
            ids = tok.encode(p, add_special_tokens=False)
            ef.write(f"{i}\t{' '.join(str(x) for x in ids)}\n")

    print(f"vocab={len(id_to_token)} merges={n_merges} specials={len(specials)} prompts={len(prompts)}")
    # show special-token recognition behavior on a raw string
    sample = tok.encode("<|im_start|>user", add_special_tokens=False)
    print(f"encode('<|im_start|>user') = {sample}  (expect 151644 then BPE of 'user')")
    print(f"OK: wrote tokenizer fixtures to {FIX}")


if __name__ == "__main__":
    main()
