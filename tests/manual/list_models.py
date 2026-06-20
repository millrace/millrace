"""List downloaded models and flag which millfolio can serve.

Scans the HuggingFace cache for models that actually have safetensors weights on
disk (skipping metadata-only entries and .bin-only repos the engine can't load),
reads each one's hidden_size from config.json, and resolves it to what `serve`
accepts: 896 -> Qwen2.5-0.5B, 2048 -> Qwen2.5-3B; anything else is shown as
unsupported. Sibling to download_model.py; runs in the oracle env.

Usage:
    pixi run -e oracle list-models
"""

import json
import sys

from huggingface_hub import scan_cache_dir

# hidden_size -> human label of the arch millfolio's loader auto-detects.
SERVABLE = {896: "Qwen2.5-0.5B", 2048: "Qwen2.5-3B"}


def _probe(repo) -> tuple[int | None, bool]:
    """(hidden_size, has_weights) from any revision of `repo` that has
    safetensors weights on disk. has_weights is False for metadata-only repos."""
    for rev in repo.revisions:
        files = {f.file_name: f for f in rev.files}
        if not any(n.endswith(".safetensors") for n in files):
            continue
        hidden = None
        cfg = files.get("config.json")
        if cfg is not None:
            try:
                with open(cfg.file_path) as fh:
                    hidden = json.load(fh).get("hidden_size")
            except (OSError, ValueError):
                pass
        return hidden, True
    return None, False


def main() -> None:
    info = scan_cache_dir()
    rows = []  # (servable, repo_id, size_str, verdict)
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        hidden, has_weights = _probe(repo)
        if not has_weights:
            continue  # weights not actually downloaded — not serveable
        if hidden in SERVABLE:
            rows.append((True, repo.repo_id, repo.size_on_disk_str, SERVABLE[hidden]))
        else:
            label = "unsupported" + (f" (hidden={hidden})" if hidden else "")
            rows.append((False, repo.repo_id, repo.size_on_disk_str, label))

    if not rows:
        print("no models with downloaded weights in the HF cache.")
        print("  download one:  pixi run -e oracle download-model -- Qwen/Qwen2.5-3B-Instruct")
        return

    rows.sort(key=lambda r: (not r[0], r[1]))  # servable first, then by id
    print("downloaded models  (✓ = `pixi run serve -- <id>` can serve it):")
    for servable, repo_id, size, verdict in rows:
        mark = "✓" if servable else " "
        print(f"  {mark} {size:>9}  {repo_id:<42} {verdict}")


if __name__ == "__main__":
    main()
