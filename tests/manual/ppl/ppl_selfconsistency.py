"""Self-consistency probe for the Gemma teacher-forced logprobs.

For a fixed token sequence, logP(token_k | t0..t_{k-1}) can be computed two ways
via /v1/completions:

  full[k]   = token_logprobs[k] from ONE forward of the whole sequence
              (the interior-position path token_logprobs uses — suspect).
  prefix[k] = the LAST token_logprob from a forward over just t0..t_k
              (the last position — the path generation/lm_logits uses — trusted).

If full[k] != prefix[k] for interior k, the multi-position forward is wrong and
this says exactly where (which positions, how badly).
"""
import json, sys, urllib.request

BASE = "http://127.0.0.1:8000"
SENT = "The Time Traveller (for so it will be convenient to speak of him) was expounding a recondite matter to us."


def post(prompt):
    body = json.dumps({"prompt": prompt, "echo": True, "logprobs": True, "max_tokens": 0}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main():
    # 1) full pass: get token ids (incl. server-prepended BOS) + per-position logprobs
    resp = post(SENT)
    toks = resp["choices"][0]["logprobs"]["tokens"]
    full = resp["choices"][0]["logprobs"]["token_logprobs"]
    n = len(toks)
    print(f"n={n} tokens (incl BOS); first few ids: {toks[:6]}")

    # 2) prefix passes: for each k>=2, score t0..t_k (VLIST) and take the LAST logprob
    print(f"{'k':>3} {'tok':>7} {'full[k]':>10} {'prefix[k]':>10} {'Δ':>9}")
    worst = (0.0, -1)
    for k in range(2, n):
        pr = post(toks[: k + 1])           # token-id array → no extra BOS added
        prefix_k = pr["choices"][0]["logprobs"]["token_logprobs"][-1]
        fk = full[k]
        d = (fk - prefix_k) if (fk is not None and prefix_k is not None) else float("nan")
        flag = "  <<<" if abs(d) > 0.5 else ""
        print(f"{k:>3} {toks[k]:>7} {fk:>10.4f} {prefix_k:>10.4f} {d:>9.4f}{flag}")
        if abs(d) > abs(worst[0]):
            worst = (d, k)
    print(f"\nworst divergence: Δ={worst[0]:.4f} at k={worst[1]}")


if __name__ == "__main__":
    main()
