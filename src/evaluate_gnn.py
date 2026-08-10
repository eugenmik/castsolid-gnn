"""Evaluate a trained GNN on cache stems: GNN-only and (optionally) hybrid,
optional rotation TTA, per-split comparison table. No augmentation leakage:
TTA rotations are seeded and averaged in seconds-space."""
import argparse
import json
import os.path as osp

import numpy as np
import torch

from src.graph_build import (knn_edge_index, edge_attr, random_rotation_matrix, rotate,
                             union_edge_index, union_edge_attr)
from src.model_gnn import SolidGNN
from src.dataload import load_norm, standardize
from src.splits import load_splits
from src.baseline import predict_seconds
from src.hybrid import hybrid_field
from src import metrics

# The HGB baseline was trained on the 9 v2 features = first 9 columns of any
# newer feature version (ordering contract, SP1 spec §3).
N_BASELINE_FEATURES = 9
SPLIT_NAMES = ("interpolation", "family", "fully_cooled")


def predict_seconds_gnn(pred_std_log, norm) -> np.ndarray:
    pred_log = pred_std_log * norm["y_log_std"] + norm["y_log_mean"]
    return np.clip(np.expm1(pred_log), 0.0, None)


def load_gnn(ckpt, device):
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg, norm = state["cfg"], state["norm"]
    model = SolidGNN(node_in=cfg.get("node_in", 9), edge_in=cfg.get("edge_in", 4),
                     hidden=cfg["hidden"], n_blocks=cfg["n_blocks"],
                     two_head=cfg.get("two_head", False)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg, norm


def _build_edges(pos, cfg, mesh_edges=None):
    if cfg.get("graph", "knn") == "union":
        if mesh_edges is None:
            raise ValueError("checkpoint expects a union graph but mesh_edges is missing")
        ei, is_mesh = union_edge_index(pos, mesh_edges, k=cfg["k"])
        return ei, union_edge_attr(pos, ei, is_mesh)
    ei = knn_edge_index(pos, k=cfg["k"])
    return ei, edge_attr(pos, ei)


def predict_graph_seconds(model, cfg, norm, x, pos, mesh_edges=None, device="cpu", tta=0):
    """Mean GNN prediction (seconds) over identity + `tta` seeded random rotations."""
    outs = []
    rots = [None] + [random_rotation_matrix(seed=s) for s in range(tta)]
    zeros = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    for R in rots:
        p = pos if R is None else rotate(pos, R)
        ei, ea = _build_edges(p, cfg, mesh_edges)
        with torch.no_grad():
            yn, _ = model(x.to(device), ei.to(device), ea.to(device), zeros)
        outs.append(predict_seconds_gnn(yn.cpu().numpy(), norm))
    return np.mean(outs, axis=0)


def _metric_dict(y, p, xyz, gid) -> dict:
    return {
        "mae": metrics.mae(y, p),
        "rmse": metrics.rmse(y, p),
        "median_rel_error": metrics.median_rel_error(y, p),
        "pred_std_ratio": metrics.pred_std_ratio(y, p, gid),
        "hotspot_accuracy": metrics.hotspot_accuracy(y, p, xyz, gid),
    }


def evaluate(ckpt, features_dir, test_stems, device=None, tta=0,
             hgb_path=None, baseline_features_dir="data/features_v2") -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, norm = load_gnn(ckpt, device)
    fmean = torch.tensor(norm["feat_mean"], dtype=torch.float32)
    fstd = torch.tensor(norm["feat_std"], dtype=torch.float32)
    hgb = hgb_norm = None
    if hgb_path is not None:
        import joblib
        hgb = joblib.load(hgb_path)
        hgb_norm = load_norm(baseline_features_dir)

    ys, preds, hyb, xyz, gid = [], [], [], [], []
    for i, stem in enumerate(test_stems):
        d = np.load(osp.join(features_dir, "graphs", f"{stem}.npz"))
        X, y, pos_np = d["X"], d["y"], d["pos"]
        x = (torch.tensor(X, dtype=torch.float32) - fmean) / fstd
        pos = torch.tensor(pos_np, dtype=torch.float32)
        me = (torch.tensor(d["mesh_edges"], dtype=torch.long)
              if "mesh_edges" in d.files else None)
        p = predict_graph_seconds(model, cfg, norm, x, pos, me, device=device, tta=tta)
        preds.append(p); ys.append(y); xyz.append(pos_np)
        gid.append(np.full(y.shape[0], i, dtype=np.int64))
        if hgb is not None:
            Xb = standardize(X[:, :N_BASELINE_FEATURES], hgb_norm)
            hyb.append(hybrid_field(p, predict_seconds(hgb, Xb, hgb_norm)))

    y = np.concatenate(ys); p = np.concatenate(preds)
    xyz = np.concatenate(xyz); gid = np.concatenate(gid)
    out = _metric_dict(y, p, xyz, gid)
    if hgb is not None:
        out["hybrid"] = _metric_dict(y, np.concatenate(hyb), xyz, gid)
    return out


def format_table(results) -> str:
    lines = [f"{'split':<15}{'model':<8}{'MAE':>8}{'RMSE':>8}{'hot-spot':>10}{'std-ratio':>11}"]
    for split, res in results.items():
        rows = [("gnn", res)] + ([("hybrid", res["hybrid"])] if "hybrid" in res else [])
        for label, m in rows:
            lines.append(f"{split:<15}{label:<8}{m['mae']:>8.2f}{m['rmse']:>8.2f}"
                         f"{m['hotspot_accuracy']:>10.3f}{m['pred_std_ratio']:>11.3f}")
    return "\n".join(lines)


def evaluate_all_splits(ckpt, features_dir, splits_path, hgb_path=None, tta=0,
                        device=None, out_json=None,
                        baseline_features_dir="data/features_v2") -> dict:
    splits = load_splits(splits_path)
    results = {}
    for name in SPLIT_NAMES:
        results[name] = evaluate(ckpt, features_dir, splits[name]["test"], device=device,
                                 tta=tta, hgb_path=hgb_path,
                                 baseline_features_dir=baseline_features_dir)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
    print(format_table(results))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--features-dir", default="data/features_v2")
    ap.add_argument("--splits", default=None, help="default: <features-dir>/splits.json")
    ap.add_argument("--hgb", default=None)
    ap.add_argument("--baseline-features-dir", default="data/features_v2")
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    evaluate_all_splits(a.ckpt, a.features_dir,
                        a.splits or osp.join(a.features_dir, "splits.json"),
                        hgb_path=a.hgb, tta=a.tta, out_json=a.out,
                        baseline_features_dir=a.baseline_features_dir)
