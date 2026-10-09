"""Convert the internal v5.1 artifacts into the public release layout.

    python -m tools.export_release --out weights

Writes `gnn_v5.1.safetensors`, `gbm_total_v3.skops`, `gbm_std_v2.skops`,
`ood_ranges.json` and `SHA256SUMS` into `--out`, then reloads every file
through `src.release_io` and checks it against the original:
GNN tensors / norm / cfg bit-identical, GBM predictions identical on
randomized feature rows spanning each feature's training split range.
"""
import argparse
import hashlib
import json
import os
import shutil

import joblib
import numpy as np
import torch

from src import release_io as rio

SOURCES = {
    "gnn": "runs/ft3_stage1/ckpt_ep021.pt",
    "gbm_total": "runs/ft3_gbm/hgb_v3.joblib",
    "gbm_std": "runs/ft3_gbm/hgb_std_v2.joblib",
    "ood": "data/features_v2/ood_ranges.json",
}


def _check_gnn(orig: dict, back: dict) -> None:
    assert back["cfg"] == orig["cfg"], "cfg mismatch"
    assert set(back["model"]) == set(orig["model"]), "state_dict keys differ"
    for k, v in orig["model"].items():
        assert torch.equal(back["model"][k].cpu(), v.cpu()), f"tensor {k} differs"
    assert set(back["norm"]) == set(orig["norm"]), "norm keys differ"
    for k, v in orig["norm"].items():
        b = back["norm"][k]
        if isinstance(v, torch.Tensor):
            assert torch.equal(b, v.cpu()), f"norm {k} differs"
        elif isinstance(v, np.ndarray):
            assert b.dtype == v.dtype and np.array_equal(b, v), f"norm {k} differs"
        else:
            assert b == v, f"norm {k} differs"


def _gbm_probe_rows(model, n=10_000, seed=0) -> np.ndarray:
    """Random rows inside each feature's histogram bin range (the binner's
    thresholds are exactly the training split points), plus the edges."""
    rng = np.random.default_rng(seed)
    thresholds = model._bin_mapper.bin_thresholds_
    cols = []
    for t in thresholds:
        lo, hi = (float(t[0]), float(t[-1])) if len(t) else (0.0, 1.0)
        pad = 0.1 * (hi - lo + 1e-9)
        cols.append(rng.uniform(lo - pad, hi + pad, size=n))
    X = np.stack(cols, axis=1)
    edges = np.array([[float(t[0]) if len(t) else 0.0 for t in thresholds],
                      [float(t[-1]) if len(t) else 1.0 for t in thresholds]])
    return np.vstack([X, edges])


def _check_gbm(orig, back) -> int:
    X = _gbm_probe_rows(orig)
    assert np.array_equal(orig.predict(X), back.predict(X)), "GBM predictions differ"
    return len(X)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(out_dir: str = "weights") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "gnn": os.path.join(out_dir, "gnn_v5.1.safetensors"),
        "gbm_total": os.path.join(out_dir, "gbm_total_v3.skops"),
        "gbm_std": os.path.join(out_dir, "gbm_std_v2.skops"),
        "ood": os.path.join(out_dir, "ood_ranges.json"),
    }
    state = torch.load(SOURCES["gnn"], map_location="cpu", weights_only=False)
    rio.save_gnn_release(state, out["gnn"], extra_metadata={
        "release": "v5.1", "source_checkpoint": "ft3_stage1/ckpt_ep021",
        "license": "apache-2.0"})
    _check_gnn(state, rio.load_gnn_state(out["gnn"]))
    report = {"gnn_params": int(sum(v.numel() for v in state["model"].values()))}

    for key in ("gbm_total", "gbm_std"):
        model = joblib.load(SOURCES[key])
        rio.save_gbm_release(model, out[key])
        report[f"{key}_rows_checked"] = _check_gbm(model, rio.load_gbm(out[key]))
        report[f"{key}_n_features"] = int(model.n_features_in_)

    with open(SOURCES["ood"]) as f:
        json.load(f)
    shutil.copyfile(SOURCES["ood"], out["ood"])

    with open(os.path.join(out_dir, "SHA256SUMS"), "w") as f:
        for path in out.values():
            f.write(f"{_sha256(path)}  {os.path.basename(path)}\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights")
    print(json.dumps(export(ap.parse_args().out), indent=2))
