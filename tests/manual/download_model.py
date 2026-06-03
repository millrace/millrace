"""Download a Hugging Face model's weights into the local HF cache and print the
snapshot directory — the path the engine's auto-detecting loader wants.

Only the safetensors shards + the small metadata/index json are fetched (not the
PyTorch .bin duplicates), so the printed directory holds either a single
`*.safetensors` (e.g. Qwen2.5-0.5B) or the shards plus
`model.safetensors.index.json` (e.g. Qwen2.5-3B) — both of which
`model.load_weights` detects automatically. Re-uses the cache if already present.

Usage (runs in the oracle env, which has huggingface_hub):
    pixi run -e oracle download-model -- Qwen/Qwen2.5-3B-Instruct

Drop the printed path into tests/fixtures/forward/meta.txt line 2, then
`pixi run serve` (it logs the detected arch).
"""

import sys

from huggingface_hub import snapshot_download


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1].startswith("-"):
        sys.exit("usage: pixi run -e oracle download-model -- <hf-model-id>\n"
                 "   e.g. Qwen/Qwen2.5-3B-Instruct")
    model = sys.argv[1]
    print(f"downloading {model} (safetensors + metadata) …", file=sys.stderr)
    path = snapshot_download(model, allow_patterns=["*.safetensors", "*.json"])
    # stderr gets the human note; stdout is just the path so it's easy to capture.
    print(path)


if __name__ == "__main__":
    main()
