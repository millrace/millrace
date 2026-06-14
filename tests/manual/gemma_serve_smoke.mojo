"""End-to-end smoke test of the Gemma serving path, mirroring server.gen_full:
render the chat template -> Gemma tokenizer -> persistent KV session sized like
the server (GEMMA_MAX_SEQ) -> prefill the suffix -> greedy decode until a Gemma
EOS (<eos>=1 or <turn|>=106) -> detokenize. Validates the whole stack the HTTP
server drives, minus flare. Loads the ~7 GB int4 model, so it's a manual test.

  pixi run mojo run -I src -I ../jinja2.mojo/src -I ../flare tests/manual/gemma_serve_smoke.mojo
"""

from std.gpu.host import DeviceContext
from gemma import load_gemma_weights, GemmaWeights, G_NLAYERS
from engine import new_session, sess_prefill_suffix, sess_step, argmax_f, Session
from chat import load_chat_template, render_request
from tokenizer import load_gemma_tokenizer_json, Tokenizer
from tensor_ops import probe_simd_gemm
from template import Template

comptime GEMMA_MAX_SEQ = 4096   # mirror server.GEMMA_MAX_SEQ
comptime TMPL = "assets/gemma4-chat-template.jinja"
comptime SNAP = "/Users/mseritan/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-bf16/snapshots/afb7b215e9fe3b3eaef462b27d5c9d9b1ba0565b"


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8]()
    for byte in s.as_bytes():
        b.append(byte)
    return b^


def run_prompt(ctx: DeviceContext, mut gw: GemmaWeights, mut sess: Session,
               tok: Tokenizer, tmpl: Template, eos1: Int, eos2: Int,
               idx: Int, body: String) raises:
    sess.pos = 0
    var rendered = render_request(tmpl, body)
    var ids = tok.encode(_bytes(rendered))
    var logits = sess_prefill_suffix(ctx, gw, sess, ids, 0, True)
    var gen = List[Int]()
    var nxt = argmax_f(logits)
    var steps = 0
    while steps < 40 and nxt != eos1 and nxt != eos2:
        gen.append(nxt)
        logits = sess_step(ctx, gw, sess, nxt)
        nxt = argmax_f(logits)
        steps += 1
    var text = String(StringSlice(unsafe_from_utf8=Span(tok.decode(gen))))
    print("Q", idx, " prompt_toks=", len(ids), " gen_toks=", len(gen), sep="")
    print("  -> ", text, sep="")


def main() raises:
    var ctx = DeviceContext()
    print("loading gemma tokenizer + template…")
    var tok = load_gemma_tokenizer_json(SNAP + "/tokenizer.json")
    var tmpl = load_chat_template(TMPL)

    print("loading gemma weights (int4, all 48 layers)…")
    var alllayers = List[Int]()
    for i in range(G_NLAYERS):
        alllayers.append(i)
    var gw = load_gemma_weights(ctx, SNAP, alllayers, True)
    gw.simd_ok = probe_simd_gemm(ctx)
    var cfg = gw.config()
    print("  simd_ok=", gw.simd_ok, " nlayers=", cfg.nlayers, " nkv=", cfg.nkv,
          " eos=", cfg.eos1, "/", cfg.eos2, sep="")

    print("allocating persistent session (max_seq=", GEMMA_MAX_SEQ, ")…", sep="")
    var sess = new_session(ctx, GEMMA_MAX_SEQ, cfg.nlayers, cfg.nkv)

    run_prompt(ctx, gw, sess, tok, tmpl, cfg.eos1, cfg.eos2, 0,
        '{"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}]}')
    run_prompt(ctx, gw, sess, tok, tmpl, cfg.eos1, cfg.eos2, 1,
        '{"messages":[{"role":"user","content":"Name the first four planets from the Sun."}]}')
