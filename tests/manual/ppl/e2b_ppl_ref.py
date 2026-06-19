"""HF gemma4-e2b per-position logprobs reference, for a fixed token sequence.

Loads the e2b checkpoint into Gemma4ForCausalLM (same path as e2b_hf_ref.py),
teacher-forces a token sequence, and prints logP(token_k | t0..t_{k-1}) for each
position. Compare against millrace's e2b token_logprobs to validate the shared
forward kernels are *calibration*-correct (not just argmax-correct)."""
import os, glob, json, math
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-bf16/snapshots/*"))[0]
cfg_json = json.load(open(os.path.join(SNAP, "config.json")))["text_config"]
cfg = Gemma4TextConfig(**cfg_json)

sd = {}
for f in glob.glob(os.path.join(SNAP, "*.safetensors")):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.startswith("language_model.model."):
                sd["model." + k[len("language_model.model."):]] = h.get_tensor(k)
sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
model = Gemma4ForCausalLM(cfg).to(torch.bfloat16)
model.load_state_dict(sd, strict=False)
model.eval()

tok = AutoTokenizer.from_pretrained(SNAP)
SENT = "The Time Traveller (for so it will be convenient to speak of him) was expounding a recondite matter to us."
ids = tok(SENT, add_special_tokens=False)["input_ids"]
ids = [2] + ids   # prepend BOS, matching millrace's Gemma raw-text path
print("IDS=" + json.dumps(ids))

with torch.no_grad():
    logits = model(torch.tensor([ids])).logits[0].float()   # [T, vocab]
logp = torch.log_softmax(logits, dim=-1)
out = [None]
nll = 0.0
for k in range(1, len(ids)):
    lp = float(logp[k - 1, ids[k]])   # logits at pos k-1 predict token k
    out.append(round(lp, 4))
    nll += -lp
print("HF_E2B_LP=" + json.dumps(out))
print(f"HF_E2B mean_nll={nll/(len(ids)-1):.4f}  PPL={math.exp(nll/(len(ids)-1)):.3f}")
