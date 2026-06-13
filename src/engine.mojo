"""Model-agnostic decode runtime: KV-cache `Session`, the per-token forward
(`sess_step`)/prefill (`sess_prefill`, `sess_prefill_suffix`), the tied LM head
(`logits_last`/`argmax_last`), embedding pooling (`sess_embed`), and the greedy /
sampled `generate` loops. Nothing here is Qwen-specific: the per-layer forward is
dispatched on `w.cfg.family` (`_forward_layer`), so a new model family is served
by adding its `*_layer` and one dispatch arm — the session/generate machinery is
reused unchanged."""

from std.math import sqrt
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from tensor_ops import DevBuf, embed_tokens, last_row, rmsnorm, mm_norm
from sampling import process_logits, sample, argmax_f
from qwen import Weights, qwen_layer, FAMILY_QWEN


def _forward_layer(ctx: DeviceContext, mut w: Weights, l: Int, mut h: DevBuf,
                   mut kc: DevBuf, mut vc: DevBuf, Tq: Int, q_offset: Int,
                   cache_len: Int, mut dummy: DevBuf) raises -> DevBuf:
    """Dispatch one decoder layer to its model family's forward. Qwen today; a new
    family adds `elif w.cfg.family == FAMILY_GEMMA: return gemma_layer(...)`."""
    if w.cfg.family == FAMILY_QWEN:
        return qwen_layer(ctx, w, l, h, kc, vc, Tq, q_offset, cache_len, dummy)
    return qwen_layer(ctx, w, l, h, kc, vc, Tq, q_offset, cache_len, dummy)


def argmax_last(ctx: DeviceContext, mut w: Weights, mut h: DevBuf, T: Int, mut dummy: DevBuf) raises -> Int:
    """Final RMSNorm + tied LM head; argmax over the last position's logits.

    Only row T-1 feeds the LM head: the VOCAB-wide (151936) head is the largest
    matmul in the net, so at prefill running it on all T rows and keeping one was
    the dominant cost (§11 #12). Slice the last hidden row first → one GEMV."""
    var hl = last_row(ctx, h, T, w.hidden)
    var logits = mm_norm(ctx, hl, w.final_norm, w.embed, dummy, 1, w.hidden, w.vocab, 0)
    ctx.synchronize()
    var best = -1
    var best_v = Float32(-1.0e30)
    with logits.map_to_host() as m:
        var mt = TileTensor(m, row_major(w.vocab))
        for i in range(w.vocab):
            var v = rebind[Scalar[DType.float32]](mt[i])
            if v > best_v:
                best_v = v
                best = i
    return best

def logits_last(ctx: DeviceContext, mut w: Weights, mut h: DevBuf, T: Int, mut dummy: DevBuf) raises -> List[Float32]:
    """Final RMSNorm + tied LM head; returns the last position's logits on host.
    Slices row T-1 before the head so prefill runs it once, not T times (§11 #12)."""
    var hl = last_row(ctx, h, T, w.hidden)
    var logits = mm_norm(ctx, hl, w.final_norm, w.embed, dummy, 1, w.hidden, w.vocab, 0)
    ctx.synchronize()
    var out = List[Float32]()
    with logits.map_to_host() as m:
        var mt = TileTensor(m, row_major(w.vocab))
        for i in range(w.vocab):
            out.append(rebind[Scalar[DType.float32]](mt[i]))
    return out^


def upload_ids(ctx: DeviceContext, vals: List[Int]) raises -> DeviceBuffer[DType.int32]:
    var n = len(vals)
    var d = ctx.enqueue_create_buffer[DType.int32](n)
    with d.map_to_host() as m:
        var mt = TileTensor(m, row_major(n))
        for i in range(n):
            mt[i] = rebind[mt.ElementType](Int32(vals[i]))
    return d^


# ── decode session: KV caches + position, the per-step primitive ──────────────

@fieldwise_init
struct Session(Movable):
    """Holds the per-layer KV caches and the current position. `prefill` runs the
    prompt and returns the last-position logits; `step` advances one token. Shared
    by greedy/sampled generate and the server's streaming loop."""
    var kcs: List[DevBuf]
    var vcs: List[DevBuf]
    var dummy: DevBuf
    var cache_len: Int
    var pos: Int


def new_session(ctx: DeviceContext, max_seq: Int, nlayers: Int, nkv: Int) raises -> Session:
    var cache_len = max_seq * nkv
    var kcs = List[DevBuf]()
    var vcs = List[DevBuf]()
    for _ in range(nlayers):
        kcs.append(ctx.enqueue_create_buffer[DType.float32](cache_len))
        vcs.append(ctx.enqueue_create_buffer[DType.float32](cache_len))
    return Session(kcs^, vcs^, ctx.enqueue_create_buffer[DType.float32](1), cache_len, 0)


def sess_prefill(ctx: DeviceContext, mut w: Weights, mut s: Session, prompt: List[Int]) raises -> List[Float32]:
    var P = len(prompt)
    var ids_dev = upload_ids(ctx, prompt)
    var h = embed_tokens(ctx, ids_dev, w.embed, P, w.hidden, w.vocab)
    for l in range(w.nlayers):
        h = _forward_layer(ctx, w, l, h, s.kcs[l], s.vcs[l], P, 0, s.cache_len, s.dummy)
    s.pos = P
    return logits_last(ctx, w, h, P, s.dummy)


def sess_embed(ctx: DeviceContext, mut w: Weights, prompt: List[Int]) raises -> List[Float32]:
    """Qwen3-Embedding sentence vector for `prompt`: run the full decoder, take the
    LAST token's hidden state (the official Qwen3-Embedding last-token pooling — the
    input ids already carry the appended EOS, so no padding/length handling is
    needed here), apply the final RMSNorm, then L2-normalize.

    `AutoModel.last_hidden_state` is post-`model.norm`, so the reference vector this
    must match is final_norm(h[T-1]); we normalize it to unit length (HF recipe:
    `F.normalize(pool, p=2, dim=1)`). Runs its own one-shot Session (no KV reuse).

    Returns the D-element unit vector on the host (D = w.hidden = 1024)."""
    var P = len(prompt)
    var s = new_session(ctx, P + 2, w.nlayers, w.nkv)
    var ids_dev = upload_ids(ctx, prompt)
    var h = embed_tokens(ctx, ids_dev, w.embed, P, w.hidden, w.vocab)
    for l in range(w.nlayers):
        h = _forward_layer(ctx, w, l, h, s.kcs[l], s.vcs[l], P, 0, s.cache_len, s.dummy)
    # Last-token pool → final RMSNorm (so we norm only one row, not all P).
    var hl = last_row(ctx, h, P, w.hidden)
    var hn = rmsnorm(ctx, hl, w.final_norm, 1, w.hidden)
    ctx.synchronize()
    var out = List[Float32]()
    var ss = Float32(0.0)
    with hn.map_to_host() as m:
        var mt = TileTensor(m, row_major(w.hidden))
        for i in range(w.hidden):
            var v = rebind[Scalar[DType.float32]](mt[i])
            out.append(v)
            ss += v * v
    var inv = Float32(1.0) / sqrt(ss)
    for i in range(len(out)):
        out[i] = out[i] * inv
    return out^


# Below this suffix length, prefill is sub-second and frequent (the prefix-cache
# common case), so progress reporting — and its per-layer synchronize — is skipped
# entirely, leaving that hot path byte-for-byte as before.
comptime PROGRESS_MIN_TOK = 2048
comptime PROGRESS_EVERY_NS = 5_000_000_000   # ~5 s between progress lines


def _ktok(n: Int) -> String:
    """Compact token count: 9562 -> '9.5k', 800 -> '800'."""
    if n < 1000:
        return String(n)
    return String(n // 1000) + "." + String((n % 1000) // 100) + "k"

def _dur(secs: Float64) -> String:
    var s = Int(secs + 0.5)
    if s < 90:
        return String(s) + "s"
    return String(s // 60) + "m" + String(s % 60) + "s"


def sess_prefill_suffix(ctx: DeviceContext, mut w: Weights, mut s: Session,
                        suffix: List[Int], offset: Int, progress: Bool = False) raises -> List[Float32]:
    """Prefill `suffix` tokens at cache position `offset`, reusing the K/V already
    stored in rows [0, offset). Returns the last-row logits. This is the engine
    behind the server's cross-request prefix cache; `sess_prefill` is just the
    offset==0 / whole-prompt special case. RoPE positions come from `offset`, so
    the rotated K and the attention mask stay correct for the reused prefix.

    With `progress` (and a large enough suffix), prints a throttled stdout line
    with percent done + ETA. Layers are uniform cost, so elapsed/layers-done
    extrapolates accurately. It synchronizes per layer to time real GPU progress;
    that adds no throughput cost (layers are a sequential dependency chain anyway)
    but is gated off below PROGRESS_MIN_TOK so the frequent tiny prefills are
    untouched. Granularity is per-layer — a kernel already running can't be
    interrupted — so very long contexts tick per layer rather than exactly 5 s."""
    var Q = len(suffix)
    var ids_dev = upload_ids(ctx, suffix)
    var h = embed_tokens(ctx, ids_dev, w.embed, Q, w.hidden, w.vocab)
    var report = progress and Q >= PROGRESS_MIN_TOK
    var t0 = perf_counter_ns()
    var last = t0
    for l in range(w.nlayers):
        h = _forward_layer(ctx, w, l, h, s.kcs[l], s.vcs[l], Q, offset, s.cache_len, s.dummy)
        if report:
            ctx.synchronize()
            var now = perf_counter_ns()
            if Float64(now - last) >= Float64(PROGRESS_EVERY_NS):
                var done = l + 1
                var elapsed = Float64(now - t0) / 1.0e9
                var eta = elapsed * Float64(w.nlayers - done) / Float64(done)
                print("  prefill ", _ktok(Q), "tok: ", (done * 100) // w.nlayers,
                      "% (layer ", done, "/", w.nlayers, "), ~", _dur(eta), " left", sep="")
                last = now
    s.pos = offset + Q
    return logits_last(ctx, w, h, Q, s.dummy)


def sess_step(ctx: DeviceContext, mut w: Weights, mut s: Session, token: Int) raises -> List[Float32]:
    var one = upload_ids(ctx, [token])
    var h = embed_tokens(ctx, one, w.embed, 1, w.hidden, w.vocab)
    for l in range(w.nlayers):
        h = _forward_layer(ctx, w, l, h, s.kcs[l], s.vcs[l], 1, s.pos, s.cache_len, s.dummy)
    s.pos += 1
    return logits_last(ctx, w, h, 1, s.dummy)


def generate(ctx: DeviceContext, mut w: Weights, prompt: List[Int], max_new: Int) raises -> List[Int]:
    """Greedy decode: prefill the prompt then emit tokens until EOS or max_new."""
    var s = new_session(ctx, len(prompt) + max_new + 2, w.nlayers, w.nkv)
    var nxt = argmax_f(sess_prefill(ctx, w, s, prompt))
    var gen = List[Int]()
    gen.append(nxt)
    while len(gen) < max_new and nxt != w.cfg.eos1 and nxt != w.cfg.eos2:
        nxt = argmax_f(sess_step(ctx, w, s, nxt))
        gen.append(nxt)
    return gen^


def generate_sample(ctx: DeviceContext, mut w: Weights, prompt: List[Int], max_new: Int,
                    temp: Float32, top_k: Int, top_p: Float32, rep_pen: Float32,
                    seed: UInt64) raises -> List[Int]:
    """Greedy-structure decode but draw each token from the processed distribution."""
    var s = new_session(ctx, len(prompt) + max_new + 2, w.nlayers, w.nkv)
    var rng = seed if seed != 0 else UInt64(0x9E3779B97F4A7C15)
    var context = prompt.copy()
    var nxt = sample(process_logits(sess_prefill(ctx, w, s, prompt), context, temp, top_k, top_p, rep_pen), rng)
    var gen = List[Int]()
    gen.append(nxt)
    context.append(nxt)
    while len(gen) < max_new and nxt != w.cfg.eos1 and nxt != w.cfg.eos2:
        var dist = process_logits(sess_step(ctx, w, s, nxt), context, temp, top_k, top_p, rep_pen)
        nxt = sample(dist, rng)
        context.append(nxt)
        gen.append(nxt)
    return gen^
