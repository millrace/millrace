# mojo-backend

A native-Mojo, GPU-only re-implementation of Qwen2.5-0.5B-Instruct inference on
Apple Silicon (Metal). Pure Mojo forward pass, served over an OpenAI-compatible
HTTP API. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## Prerequisites

- Apple Silicon Mac (Metal GPU).
- [pixi](https://pixi.sh) — the environment is pinned in `pixi.toml` (nightly
  Mojo + a separate `oracle` env with torch/transformers for fixture capture).

## Start the server

The server loads the tokenizer tables and the checkpoint path from captured
fixtures, so generate those once (these run in the `oracle` env and download the
HF model on first use):

```sh
pixi run -e oracle tok-capture       # tokenizer vocab/merges -> tests/fixtures/tokenizer/
pixi run -e oracle forward-capture   # checkpoint path        -> tests/fixtures/forward/meta.txt
```

Then launch the server:

```sh
pixi run serve
```

It compiles `src/server.mojo`, builds the native TLS helper (`libflare_tls.so`),
loads the weights onto the GPU, and listens on **http://127.0.0.1:8000**:

```
millrace serving on http://127.0.0.1:8000  (flare)
  GET  /v1/models
  POST /v1/chat/completions  (stream + non-stream)
  POST /v1/responses         (stream + non-stream)
```

Smoke-test it from another terminal:

```sh
curl -s localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"In one sentence, what is the capital of France?"}]}'
```

## Connect OpenCode

`assets/opencode.json` declares a `millrace` provider (via
`@ai-sdk/openai-compatible`, pointed at `http://127.0.0.1:8000/v1`) so the local
model appears in OpenCode's picker. With the server running in another terminal:

```sh
pixi run opencode                       # interactive
pixi run opencode -- run "your prompt"   # one-shot
```

The task sets `OPENCODE_CONFIG` to `assets/opencode.json` and the
`OPENAI_BASE_URL`/`OPENAI_API_KEY` env vars for you; the active model is
`millrace/qwen2.5-0.5b-instruct`.

To point an existing OpenCode install at the server instead of using the task,
copy the `provider` block from `assets/opencode.json` into your
`~/.config/opencode/opencode.json` and select the `millrace` model.
