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
from chat import load_chat_template, render_value
from tokenizer import load_gemma_tokenizer_json, Tokenizer
from tensor_ops import probe_simd_gemm
from template import Template
from model_iface import FAMILY_GEMMA
from toolcall import parse_gemma_tool_calls
from json import parse_json

comptime GEMMA_MAX_SEQ = 4096   # mirror server.GEMMA_MAX_SEQ
comptime TMPL = "assets/qwen2.5-chat-template.jinja"   # placeholder; render_value(FAMILY_GEMMA) renders in Mojo
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
    var rendered = render_value(tmpl, parse_json(body), FAMILY_GEMMA)
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

    # Tool-call round trip: render a tool definition, generate, parse the call.
    sess.pos = 0
    var tbody = String('{"messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}}}]}')
    var trender = render_value(tmpl, parse_json(tbody), FAMILY_GEMMA)
    var tids = tok.encode(_bytes(trender))
    var tlogits = sess_prefill_suffix(ctx, gw, sess, tids, 0, True)
    var tgen = List[Int]()
    var tnxt = argmax_f(tlogits)
    var tsteps = 0
    while tsteps < 60 and tnxt != cfg.eos1 and tnxt != cfg.eos2:
        tgen.append(tnxt)
        tlogits = sess_step(ctx, gw, sess, tnxt)
        tnxt = argmax_f(tlogits)
        tsteps += 1
    var ttext = String(StringSlice(unsafe_from_utf8=Span(tok.decode(tgen))))
    print("TOOL prompt_toks=", len(tids), " raw=", repr(ttext), sep="")
    var parsed = parse_gemma_tool_calls(ttext)
    print("  parsed calls=", len(parsed.calls), " content=", repr(parsed.content), sep="")
    for ci in range(len(parsed.calls)):
        print("  call ", ci, ": ", parsed.calls[ci].name, " args=", parsed.calls[ci].arguments, sep="")

    # Multi-turn: feed the tool result back; the model should answer in NL.
    sess.pos = 0
    var mbody = String('{"messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\\"city\\": \\"Paris\\"}"}}]},{"role":"tool","tool_call_id":"c1","content":"22C and sunny"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}}}]}')
    var mrender = render_value(tmpl, parse_json(mbody), FAMILY_GEMMA)
    var mids = tok.encode(_bytes(mrender))
    var mlogits = sess_prefill_suffix(ctx, gw, sess, mids, 0, True)
    var mgen = List[Int]()
    var mnxt = argmax_f(mlogits)
    var msteps = 0
    while msteps < 40 and mnxt != cfg.eos1 and mnxt != cfg.eos2:
        mgen.append(mnxt)
        mlogits = sess_step(ctx, gw, sess, mnxt)
        mnxt = argmax_f(mlogits)
        msteps += 1
    var mtext = String(StringSlice(unsafe_from_utf8=Span(tok.decode(mgen))))
    var mclean = parse_gemma_tool_calls(mtext)
    print("MULTITURN prompt_toks=", len(mids), " raw=", repr(mtext), sep="")
    print("  served content=", repr(mclean.content), sep="")
