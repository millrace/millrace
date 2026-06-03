"""Generate an OpenCode config from the running millrace server's /v1/models.

OpenCode's `@ai-sdk/openai-compatible` provider can't auto-discover a local model
(its bundled models.dev catalog won't list a custom id), so it only surfaces
models declared in its config. This queries the running server's `/v1/models`,
builds a `millrace` provider whose `models` map is exactly what the server
reports (default = the first / only served model), writes it to a temp file, and
prints the path — which the `opencode` pixi task feeds to `OPENCODE_CONFIG`.

So `pixi run serve -- <model>` decides what's served, and `pixi run opencode`
picks it up automatically. Mirrors ../max-backend/millrace_opencode.py but
stdlib-only (urllib), so it runs in the default env. Exits non-zero (so the task
stops) if the server isn't reachable.

Usage: python opencode_config.py [base-url]   # default http://127.0.0.1:8000/v1
"""

import json
import os
import sys
import tempfile
import urllib.request


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1").rstrip("/")
    try:
        with urllib.request.urlopen(base + "/models", timeout=3) as r:
            data = json.load(r)
    except Exception as e:
        sys.exit(f"millrace not reachable at {base}/models ({e}).\n"
                 f"  start it first:  pixi run serve            (0.5B)\n"
                 f"             or:  pixi run serve -- <hf-id>  (e.g. Qwen/Qwen2.5-3B-Instruct)")

    ids = [m["id"] for m in data.get("data", []) if m.get("id")]
    if not ids:
        sys.exit(f"no models reported at {base}/models")

    # opencode keys models by id; `name` is the picker label (drop the org prefix).
    models = {mid: {"name": mid.rsplit("/", 1)[-1]} for mid in ids}
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": "millrace/" + ids[0],
        "provider": {
            "millrace": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "millrace (local)",
                "options": {
                    "baseURL": base,
                    "apiKey": os.environ.get("OPENAI_API_KEY", "millrace"),
                },
                "models": models,
            }
        },
    }
    path = os.path.join(tempfile.gettempdir(), "millrace-opencode.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(path)


if __name__ == "__main__":
    main()
