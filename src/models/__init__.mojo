"""The `models` package: one module per supported model family.

Each family module (`qwen`, `gemma`, `gemma_e2b`) exposes a `Weights` struct that
conforms to the `ModelWeights` trait plus a `load_weights` loader and the
per-layer decoder forward, so the model-agnostic `runtime.engine` can drive any
of them. Add a sibling module here to support a new family. The stable public
import surface is re-exported from the top-level `model` facade."""
