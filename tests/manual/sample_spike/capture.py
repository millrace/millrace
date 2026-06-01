"""Dump HF's logits-processor output as the Phase-5 sampling oracle.

Sampling can't be checked token-for-token (RNG differs), so we verify the
*distribution* instead: apply Qwen's generation_config processors
(repetition_penalty 1.1 → temperature 0.7 → top_k 20 → top_p 0.8 → softmax) to a
real last-position logits vector and dump the resulting (token id, prob) pairs.
The Mojo sampler must reproduce this distribution; the draw is then its own RNG.

Writes tests/fixtures/sample/ (GITIGNORED). Run via `pixi run sample-capture`.
"""

import os

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
    TopKLogitsWarper, TopPLogitsWarper,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FIX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "sample"))
TEMP, TOPK, TOPP, REP = 0.7, 20, 0.8, 1.1


def main():
    os.makedirs(FIX, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager").float().eval()

    enc = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    ids = enc["input_ids"]
    with torch.no_grad():
        logits = model(ids).logits[0, -1, :].float()  # [V]

    logits.numpy().astype(np.float32).tofile(os.path.join(FIX, "logits.bin"))
    ids.numpy().astype(np.int32).reshape(-1).tofile(os.path.join(FIX, "context_ids.bin"))

    # case 0 = the real generation_config; case 1 = high-entropy to exercise
    # multi-token top-k/top-p ordering + renormalization on the same logits.
    cases = [(TEMP, TOPK, TOPP, REP), (1.0, 20, 0.95, 1.1)]
    with open(os.path.join(FIX, "cases.txt"), "w") as cf:
        cf.write(str(len(cases)) + "\n")
        for ci, (t, k, p, r) in enumerate(cases):
            procs = LogitsProcessorList([
                RepetitionPenaltyLogitsProcessor(r),
                TemperatureLogitsWarper(t),
                TopKLogitsWarper(k),
                TopPLogitsWarper(p),
            ])
            scores = procs(ids, logits.clone().unsqueeze(0))[0]
            probs = torch.softmax(scores, dim=-1)
            keep = torch.nonzero(probs > 0).reshape(-1).tolist()
            keep.sort(key=lambda i: -probs[i].item())
            np.array(keep, dtype=np.int32).tofile(os.path.join(FIX, f"ids{ci}.bin"))
            np.array([probs[i].item() for i in keep], dtype=np.float32).tofile(os.path.join(FIX, f"probs{ci}.bin"))
            cf.write(f"{t} {k} {p} {r}\n")
            print(f"case {ci} ({t},{k},{p},{r}): kept {len(keep)} tokens; top: "
                  + ", ".join(f"{i}:{probs[i].item():.3f}" for i in keep[:5]))
    print(f"OK: wrote sample fixtures to {FIX}")


if __name__ == "__main__":
    main()
