"""Byte-level BPE tokenizer for Qwen2 (ARCHITECTURE.md §5.2), pure Mojo.

Encode/decode in integer token-id space (the byte↔unicode map is a bijection and
every BPE symbol is a vocab token, so no unicode is needed in Mojo). Special-token
splitting + ASCII-correct Qwen pretokenization + rank-greedy BPE. Loads the tables
`tok-capture` dumps (vocab/merges resolved to ids). Verified by test_tokenizer.mojo.

Scope: ASCII-correct; non-ASCII \\p{L}/\\p{N} pretokenization is deferred.

Two loaders build the same `Tokenizer`:
  - `load_tokenizer(dir)`         — tok-capture's resolved `.tsv` dumps (dev/tests).
  - `load_tokenizer_json(path)`   — HuggingFace `tokenizer.json` directly (what the
                                    native downloader fetches), so a freshly
                                    downloaded checkpoint serves with no tok-capture.
"""

from json import parse_json

comptime MUL = 200000          # pair key = left*MUL + right (ids < 151936)
comptime APOS = 39
comptime SPACE = 32


def is_letter(b: Int) -> Bool:
    return (b >= 65 and b <= 90) or (b >= 97 and b <= 122)

def is_digit(b: Int) -> Bool:
    return b >= 48 and b <= 57

def is_space(b: Int) -> Bool:
    return b == 32 or b == 9 or b == 10 or b == 13 or b == 11 or b == 12

def is_nl(b: Int) -> Bool:
    return b == 10 or b == 13

def lower(b: Int) -> Int:
    return b + 32 if (b >= 65 and b <= 90) else b


@fieldwise_init
struct SpMatch(Copyable, Movable):
    var id: Int
    var length: Int


@fieldwise_init
struct Tokenizer(Movable):
    var byte_id: List[Int]                 # [256] id of each single-byte token
    var merge_rank: Dict[Int, Int]         # pairkey -> rank
    var merge_id: Dict[Int, Int]           # pairkey -> merged id
    var id_to_bytes: Dict[Int, List[UInt8]]
    var sp_text: List[List[UInt8]]         # special token texts
    var sp_id: List[Int]                   # parallel ids

    def bpe(self, mut ids: List[Int]) raises:
        while len(ids) >= 2:
            var best_rank = 1 << 60
            var best_i = -1
            for i in range(len(ids) - 1):
                var key = ids[i] * MUL + ids[i + 1]
                if key in self.merge_rank:
                    var r = self.merge_rank[key]
                    if r < best_rank:
                        best_rank = r
                        best_i = i
            if best_i < 0:
                break
            var key = ids[best_i] * MUL + ids[best_i + 1]
            ids[best_i] = self.merge_id[key]
            _ = ids.pop(best_i + 1)

    def next_chunk(self, buf: List[UInt8], pos: Int, end: Int) -> Int:
        var c = Int(buf[pos])

        # rule 1: contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == APOS:
            var sufs = [String("s"), String("t"), String("re"), String("ve"),
                        String("m"), String("ll"), String("d")]
            for s in sufs:
                var sb = s.as_bytes()
                if pos + 1 + len(sb) <= end:
                    var ok = True
                    for j in range(len(sb)):
                        if lower(Int(buf[pos + 1 + j])) != Int(sb[j]):
                            ok = False
                            break
                    if ok:
                        return pos + 1 + len(sb)

        # rule 2: [^\r\n\p{L}\p{N}]? \p{L}+
        if is_letter(c):
            var q = pos
            while q < end and is_letter(Int(buf[q])):
                q += 1
            return q
        elif (not is_digit(c)) and (not is_nl(c)):
            if pos + 1 < end and is_letter(Int(buf[pos + 1])):
                var q = pos + 1
                while q < end and is_letter(Int(buf[q])):
                    q += 1
                return q

        # rule 3: \p{N} (single digit)
        if is_digit(c):
            return pos + 1

        # rule 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        var p = pos
        if c == SPACE:
            p = pos + 1
        if p < end:
            var d = Int(buf[p])
            if (not is_space(d)) and (not is_letter(d)) and (not is_digit(d)):
                var q = p
                while q < end:
                    var e = Int(buf[q])
                    if is_space(e) or is_letter(e) or is_digit(e):
                        break
                    q += 1
                while q < end and is_nl(Int(buf[q])):
                    q += 1
                return q

        # rule 5: \s*[\r\n]
        if is_space(c):
            var w = pos
            while w < end and is_space(Int(buf[w])):
                w += 1
            var lastnl = -1
            for i in range(pos, w):
                if is_nl(Int(buf[i])):
                    lastnl = i
            if lastnl >= 0:
                return lastnl + 1
            # rule 6/7: \s+(?!\S) then \s+
            if w == end:
                return w
            if w - 1 > pos:
                return w - 1
            return w

        return pos + 1  # non-ASCII / unhandled byte

    def encode_normal(self, buf: List[UInt8], start: Int, stop: Int, mut out: List[Int]) raises:
        var p = start
        while p < stop:
            var e = self.next_chunk(buf, p, stop)
            if e <= p:
                e = p + 1
            var ids = List[Int]()
            for i in range(p, e):
                ids.append(self.byte_id[Int(buf[i])])
            self.bpe(ids)
            for x in ids:
                out.append(x)
            p = e

    def match_special(self, buf: List[UInt8], pos: Int) raises -> SpMatch:
        var best_id = -1
        var best_len = 0
        for s in range(len(self.sp_text)):
            ref t = self.sp_text[s]
            var n = len(t)
            if n > best_len and pos + n <= len(buf):
                var ok = True
                for j in range(n):
                    if buf[pos + j] != t[j]:
                        ok = False
                        break
                if ok:
                    best_id = self.sp_id[s]
                    best_len = n
        return SpMatch(best_id, best_len)

    def encode(self, buf: List[UInt8]) raises -> List[Int]:
        var out = List[Int]()
        var n = len(buf)
        var pos = 0
        var seg = 0
        while pos < n:
            var m = self.match_special(buf, pos)
            if m.id >= 0:
                self.encode_normal(buf, seg, pos, out)
                out.append(m.id)
                pos += m.length
                seg = pos
            else:
                pos += 1
        self.encode_normal(buf, seg, n, out)
        return out^

    def decode(self, ids: List[Int]) raises -> List[UInt8]:
        var out = List[UInt8]()
        for k in range(len(ids)):
            var id = ids[k]
            if id in self.id_to_bytes:
                for b in self.id_to_bytes[id]:
                    out.append(b)
            else:
                for s in range(len(self.sp_id)):
                    if self.sp_id[s] == id:
                        for b in self.sp_text[s]:
                            out.append(b)
                        break
        return out^


def hex_val(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    return c - 97 + 10  # a-f (Python .hex() is lowercase)

def hex_to_bytes(s: String) -> List[UInt8]:
    var sb = s.as_bytes()
    var out = List[UInt8]()
    var i = 0
    while i + 2 <= len(sb):
        out.append(UInt8((hex_val(Int(sb[i])) << 4) | hex_val(Int(sb[i + 1]))))
        i += 2
    return out^

def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()

def _byte_decoder() -> Dict[Int, Int]:
    """GPT-2 byte-level decoder: token-string codepoint -> raw byte (inverse of
    `bytes_to_unicode`). Printable bytes map to themselves; the rest to 256+n in
    order. HF `tokenizer.json` vocab keys are encoded with this bijection."""
    var dec = Dict[Int, Int]()
    var inbs = List[Bool]()
    for _ in range(256):
        inbs.append(False)
    for v in range(33, 127):        # '!'..'~'
        inbs[v] = True; dec[v] = v
    for v in range(161, 173):       # '¡'..'¬'
        inbs[v] = True; dec[v] = v
    for v in range(174, 256):       # '®'..'ÿ'
        inbs[v] = True; dec[v] = v
    var n = 0
    for b in range(256):
        if not inbs[b]:
            dec[256 + n] = b
            n += 1
    return dec^


def _decode_token(s: String, dec: Dict[Int, Int]) raises -> List[UInt8]:
    """Decode one byte-level token string back to its raw bytes."""
    var out = List[UInt8]()
    for cp in s.codepoints():
        var v = Int(cp)
        if v in dec:
            out.append(UInt8(dec[v]))
    return out^


def load_tokenizer_json(path: String) raises -> Tokenizer:
    """Build a `Tokenizer` straight from a HuggingFace `tokenizer.json` (byte-level
    BPE, `model.type == "BPE"`): decode each vocab token to raw bytes, resolve each
    `"A B"` merge to an (id,id)->merged-id rule, and take `added_tokens` as the
    special-token table. Produces the same tables as `load_tokenizer`."""
    var root = parse_json(_read(path))
    var model = root.map_get("model").value()
    var vocab = model.map_get("vocab").value()
    var merges = model.map_get("merges").value()
    var added = root.map_get("added_tokens").value()
    var dec = _byte_decoder()

    var byte_id = List[Int]()
    for _ in range(256):
        byte_id.append(-1)
    var id_to_bytes = Dict[Int, List[UInt8]]()
    var vocab_map = Dict[String, Int]()       # token string -> id (merge resolution)

    ref vkeys = vocab.c[].keys
    ref vvals = vocab.c[].vals
    for k in range(len(vkeys)):
        var tok = vkeys[k]
        var id = vvals[k].i
        vocab_map[tok] = id
        var raw = _decode_token(tok, dec)
        if len(raw) == 1:
            byte_id[Int(raw[0])] = id
        id_to_bytes[id] = raw^

    var merge_rank = Dict[Int, Int]()
    var merge_id = Dict[Int, Int]()
    ref mvals = merges.c[].vals
    for r in range(len(mvals)):
        # "A B": neither side contains a literal space (byte 0x20 encodes as 'Ġ'),
        # so split on the single separator; merged token = A concatenated with B.
        var parts = String(mvals[r].s).split(" ")
        if len(parts) != 2:
            continue
        var a = String(parts[0])
        var b = String(parts[1])
        var merged = a + b
        if a not in vocab_map or b not in vocab_map or merged not in vocab_map:
            continue
        var key = vocab_map[a] * MUL + vocab_map[b]
        merge_rank[key] = r
        merge_id[key] = vocab_map[merged]

    var sp_text = List[List[UInt8]]()
    var sp_id = List[Int]()
    ref avals = added.c[].vals
    for a in range(len(avals)):
        ref obj = avals[a]
        sp_id.append(obj.map_get("id").value().i)
        var tb = List[UInt8]()
        var cb = String(obj.map_get("content").value().s).as_bytes()
        for i in range(len(cb)):
            tb.append(cb[i])
        sp_text.append(tb^)

    return Tokenizer(byte_id^, merge_rank^, merge_id^, id_to_bytes^, sp_text^, sp_id^)


def load_tokenizer(dir: String) raises -> Tokenizer:
    var byte_id = List[Int]()
    for _ in range(256):
        byte_id.append(-1)
    var id_to_bytes = Dict[Int, List[UInt8]]()

    for line in _read(dir + "vocab.tsv").split("\n"):
        var ls = String(line)
        if ls.byte_length() == 0:
            continue
        var parts = ls.split("\t")
        if len(parts) < 2:
            continue
        var id = Int(atol(String(parts[0]).strip()))
        var hx = String(parts[1]).strip()
        if hx.byte_length() == 0:
            continue
        var bytes = hex_to_bytes(String(hx))
        if len(bytes) == 1:
            byte_id[Int(bytes[0])] = id
        id_to_bytes[id] = bytes^

    var merge_rank = Dict[Int, Int]()
    var merge_id = Dict[Int, Int]()
    var rank = 0
    for line in _read(dir + "merges.tsv").split("\n"):
        var ls = String(line)
        if ls.byte_length() == 0:
            continue
        var parts = ls.split(" ")
        if len(parts) < 3:
            continue
        var key = Int(atol(String(parts[0]).strip())) * MUL + Int(atol(String(parts[1]).strip()))
        merge_rank[key] = rank
        merge_id[key] = Int(atol(String(parts[2]).strip()))
        rank += 1

    var sp_text = List[List[UInt8]]()
    var sp_id = List[Int]()
    for line in _read(dir + "specials.tsv").split("\n"):
        var ls = String(line)
        if ls.byte_length() == 0:
            continue
        var parts = ls.split("\t")
        if len(parts) < 2:
            continue
        sp_id.append(Int(atol(String(parts[0]).strip())))
        var tb = List[UInt8]()
        var txt = String(parts[1]).as_bytes()
        for i in range(len(txt)):
            tb.append(txt[i])
        sp_text.append(tb^)

    return Tokenizer(byte_id^, merge_rank^, merge_id^, id_to_bytes^, sp_text^, sp_id^)
