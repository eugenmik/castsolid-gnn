"""Loading the published weight files.

- GNN: one safetensors file with tensors `model.<state_dict key>` and
  `norm.<key>`; the architecture config (`cfg`) and the container type of
  each normalization entry are JSON strings in the safetensors metadata.
- Gradient-boosted heads: skops files. Before loading, every type a file
  names is checked against an allowlist (scikit-learn and NumPy only).

Neither format runs code when loaded.
"""
from __future__ import annotations

import json
import os

GNN_FORMAT = "foundry-gnn-release/1"

_SKOPS_ALLOWED_PREFIXES = ("sklearn.", "numpy.")


def load_gnn_state(path: str, device="cpu") -> dict:
    """The `{model, norm, cfg}` dict stored in a GNN safetensors file."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if meta.get("format") != GNN_FORMAT:
            raise ValueError(f"{path}: not a {GNN_FORMAT} file "
                             f"(format={meta.get('format')!r})")
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    norm_kinds = json.loads(meta["norm_kinds"])
    model = {k[len("model."):]: v.to(device) for k, v in tensors.items()
             if k.startswith("model.")}
    norm = dict(json.loads(meta["norm_scalars"]))
    for k, kind in norm_kinds.items():
        t = tensors[f"norm.{k}"]
        norm[k] = t.numpy() if kind == "numpy" else t
    return {"model": model, "norm": norm, "cfg": json.loads(meta["cfg"])}


def load_gbm(path):
    """A gradient-boosted head from a skops file, after the type check."""
    import skops.io as sio
    untrusted = sio.get_untrusted_types(file=os.fspath(path))
    bad = [t for t in untrusted if not t.startswith(_SKOPS_ALLOWED_PREFIXES)]
    if bad:
        raise ValueError(f"{path}: refusing to load unexpected types {bad}")
    return sio.load(os.fspath(path), trusted=untrusted)
