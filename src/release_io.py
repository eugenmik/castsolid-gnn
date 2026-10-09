"""Public-release artifact formats (Hugging Face model repo).

The internal artifacts are pickles: the GNN checkpoint is a `torch.save`d
`{model, norm, cfg}` dict (loaded with `weights_only=False`) and the GBM heads
are `joblib` dumps. The public release ships them in non-executable formats:

- GNN  -> one `.safetensors` file: tensors `model.<state_dict key>` +
  `norm.<key>`, and a JSON `cfg` plus the norm entries' original container
  type in the safetensors metadata header. `load_gnn_state` rebuilds the
  exact `{model, norm, cfg}` dict `torch.load` would have returned, so every
  downstream loader (`src.evaluate_ft2b.load_ft2b_gnn`, `load_hybrid`) takes
  it unchanged.
- GBM  -> `.skops` (sklearn's safe serialization). Loading checks the file's
  untrusted types against a fixed sklearn/numpy allowlist instead of
  trusting whatever the file names.

`resolve_artifact` maps the internal default paths (`runs/...`) onto the
release layout (`weights/...`) when only the latter exists, so the public
snapshot runs `python -m src.predict_casting part.step` with zero flags
while the internal checkout keeps using its original files.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

GNN_FORMAT = "foundry-gnn-release/1"

# Internal default path -> public release path (repo-root relative).
RELEASE_PATHS = {
    "runs/ft3_stage1/ckpt_ep021.pt": "weights/gnn_v5.1.safetensors",
    "runs/ft3_gbm/hgb_v3.joblib": "weights/gbm_total_v3.skops",
    "runs/ft3_gbm/hgb_std_v2.joblib": "weights/gbm_std_v2.skops",
    "data/features_v2/ood_ranges.json": "weights/ood_ranges.json",
}

# Types a release `.skops` GBM file may name. Anything else fails the load.
_SKOPS_ALLOWED_PREFIXES = ("sklearn.", "numpy.")


def resolve_artifact(path):
    """`path` unchanged unless it is a known internal default that is absent
    on disk while its release counterpart exists."""
    if isinstance(path, (str, os.PathLike)):
        p = os.fspath(path)
        alt = RELEASE_PATHS.get(p)
        if alt is not None and not os.path.exists(p) and os.path.exists(alt):
            return alt
    return path


def _norm_to_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous(), "torch"
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value)), "numpy"
    return None, None


def save_gnn_release(state: dict, out_path: str, extra_metadata: dict | None = None) -> None:
    """Write a `{model, norm, cfg}` checkpoint dict as one safetensors file."""
    from safetensors.torch import save_file
    tensors, norm_kinds, norm_scalars = {}, {}, {}
    for k, v in state["model"].items():
        tensors[f"model.{k}"] = v.detach().cpu().contiguous()
    for k, v in state["norm"].items():
        t, kind = _norm_to_tensor(v)
        if t is None:
            norm_scalars[k] = v
        else:
            tensors[f"norm.{k}"] = t
            norm_kinds[k] = kind
    meta = {"format": GNN_FORMAT,
            "cfg": json.dumps(state["cfg"], sort_keys=True),
            "norm_kinds": json.dumps(norm_kinds, sort_keys=True),
            "norm_scalars": json.dumps(norm_scalars, sort_keys=True)}
    for k, v in (extra_metadata or {}).items():
        meta[k] = v if isinstance(v, str) else json.dumps(v)
    save_file(tensors, out_path, metadata=meta)


def load_gnn_state(path: str, device="cpu") -> dict:
    """Inverse of `save_gnn_release`: the `{model, norm, cfg}` dict."""
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


def save_gbm_release(model, out_path: str) -> None:
    import skops.io as sio
    sio.dump(model, out_path)


def load_gbm(path):
    """Load a GBM head: `.skops` through the type allowlist, else joblib."""
    if path is None or not isinstance(path, (str, os.PathLike)):
        return path
    if os.fspath(path).endswith(".skops"):
        import skops.io as sio
        untrusted = sio.get_untrusted_types(file=path)
        bad = [t for t in untrusted if not t.startswith(_SKOPS_ALLOWED_PREFIXES)]
        if bad:
            raise ValueError(f"{path}: refusing to load unexpected types {bad}")
        return sio.load(path, trusted=untrusted)
    import joblib
    return joblib.load(path)


def load_gnn_any(path, device):
    """A GNN checkpoint path (`.safetensors` release or internal `.pt`) as
    the `{model, norm, cfg}` dict; an already-loaded dict passes through."""
    if not isinstance(path, (str, os.PathLike)):
        return path
    if os.fspath(path).endswith(".safetensors"):
        return load_gnn_state(path, device)
    return torch.load(path, map_location=device, weights_only=False)
