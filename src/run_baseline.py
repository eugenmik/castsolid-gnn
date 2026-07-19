"""End-to-end Phase-0 baseline: build splits, train HGB (+ ablation) and MLP,
evaluate across splits, write a Markdown report and per-prefix MAE plot."""
import os
import os.path as osp

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataload import load_matrix, load_norm, standardize
from src.baseline import train_hgb, train_mlp, predict_seconds
from src.splits import build_splits, save_splits, load_splits
from src import metrics


def _xyz_mm(features_dir, data_root, stem, n):
    """Per-node positions in mm. Prefer the feature cache (v2 stores `pos` in mm);
    fall back to the raw .pt (metres) when the cache lacks positions."""
    gpath = osp.join(features_dir, "graphs", f"{stem}.npz")
    d = np.load(gpath)
    if "pos" in d.files:
        return d["pos"][:n]                       # already mm
    raw = torch.load(osp.join(data_root, "pt", f"{stem}.pt"), weights_only=False)
    return raw["pos"].numpy()[:n] * 1000.0


def evaluate_model(model, features_dir, test_stems, norm, data_root="data"):
    X, y, gid = load_matrix(features_dir, test_stems)
    # Both model types are trained on standardized features (see run()), so
    # standardize uniformly here too.
    Xs = standardize(X, norm)
    pred = predict_seconds(model, Xs, norm)
    xyz = np.concatenate([_xyz_mm(features_dir, data_root, s, (gid == i).sum())
                          for i, s in enumerate(test_stems)])
    return {
        "mae": metrics.mae(y, pred),
        "rmse": metrics.rmse(y, pred),
        "median_rel_error": metrics.median_rel_error(y, pred),
        "pred_std_ratio": metrics.pred_std_ratio(y, pred, gid),
        "hotspot_accuracy": metrics.hotspot_accuracy(y, pred, xyz, gid),
    }


def per_prefix_mae(model, features_dir, test_stems, norm):
    by = {}
    for s in test_stems:
        X, y, _ = load_matrix(features_dir, [s])
        pred = predict_seconds(model, standardize(X, norm), norm)
        by.setdefault(s.split("_")[0], []).append(metrics.mae(y, pred))
    return {k: float(np.mean(v)) for k, v in sorted(by.items())}


def _balanced(train_stems, meta_lookup):
    """Config C: downsample so each n_adiab regime has ~equal stem count."""
    from collections import defaultdict
    import random
    groups = defaultdict(list)
    for s in train_stems:
        groups[meta_lookup[s]].append(s)
    if not groups:
        return train_stems
    k = min(len(v) for v in groups.values())
    rng = random.Random(0)
    out = []
    for v in groups.values():
        out.extend(rng.sample(v, k))
    return out


def run(features_dir="data/features", splits_path=None, out_dir="runs/baseline",
        ablation=True, max_nodes_per_graph=2000, data_root="data"):
    os.makedirs(out_dir, exist_ok=True)
    splits_path = splits_path or osp.join(features_dir, "splits.json")
    meta = np.load(osp.join(features_dir, "meta.npz"), allow_pickle=True)
    if not osp.exists(splits_path):
        save_splits(build_splits(meta), splits_path)
    sp = load_splits(splits_path)
    norm = load_norm(features_dir)
    meta_nad = {str(s): int(n) for s, n in zip(meta["stem"], meta["n_adiab"])}
    fully = {str(s) for s, f in zip(meta["stem"], meta["is_fully_cooled"]) if f}

    configs = {"A_all": sp["interpolation"]["train"]}
    if ablation:
        configs["B_cooled_only"] = [s for s in sp["interpolation"]["train"] if s in fully]
        configs["C_balanced"] = _balanced(sp["interpolation"]["train"], meta_nad)

    eval_sets = {"interpolation": sp["interpolation"]["test"],
                 "family": sp["family"]["test"],
                 "fully_cooled": sp["fully_cooled"]["test"]}

    lines = ["# Phase 0 Baseline Report\n"]
    results = {}
    for cfg, train_stems in configs.items():
        if not train_stems:
            print(f"  skipping config {cfg}: no training stems")
            continue
        Xtr, ytr, _ = load_matrix(features_dir, train_stems,
                                  max_nodes_per_graph=max_nodes_per_graph)
        model = train_hgb(standardize(Xtr, norm), np.log1p(ytr))
        lines.append(f"\n## HGB — config {cfg}  (train graphs: {len(train_stems)})\n")
        lines.append("| split | MAE(s) | RMSE(s) | med.rel | pred/GT std | hotspot |")
        lines.append("|---|---|---|---|---|---|")
        for name, stems in eval_sets.items():
            if not stems:
                continue
            r = evaluate_model(model, features_dir, stems, norm, data_root)
            results[(cfg, name)] = r
            lines.append(f"| {name} | {r['mae']:.2f} | {r['rmse']:.2f} | "
                         f"{r['median_rel_error']:.3f} | {r['pred_std_ratio']:.3f} | "
                         f"{r['hotspot_accuracy']:.3f} |")
        if cfg == "A_all":
            import joblib
            joblib.dump(model, osp.join(out_dir, "hgb_A.joblib"))
            # per-prefix plot on interpolation test
            pmae = per_prefix_mae(model, features_dir, eval_sets["interpolation"], norm)
            plt.figure(figsize=(10, 4))
            plt.bar(list(pmae.keys()), list(pmae.values()))
            plt.ylabel("MAE (s)"); plt.xticks(rotation=90); plt.tight_layout()
            plt.savefig(osp.join(out_dir, "per_prefix_mae.png"), dpi=120)
            plt.close()

    # MLP on config A
    if not configs["A_all"]:
        print(f"  skipping MLP training: config A_all has no training stems")
    else:
        Xtr, ytr, _ = load_matrix(features_dir, configs["A_all"],
                                  max_nodes_per_graph=max_nodes_per_graph)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlp = train_mlp(standardize(Xtr, norm), np.log1p(ytr), device=device)
        lines.append("\n## MLP — config A_all\n")
        lines.append("| split | MAE(s) | RMSE(s) | med.rel | pred/GT std | hotspot |")
        lines.append("|---|---|---|---|---|---|")
        for name, stems in eval_sets.items():
            if not stems:
                continue
            r = evaluate_model(mlp, features_dir, stems, norm, data_root)
            lines.append(f"| {name} | {r['mae']:.2f} | {r['rmse']:.2f} | "
                         f"{r['median_rel_error']:.3f} | {r['pred_std_ratio']:.3f} | "
                         f"{r['hotspot_accuracy']:.3f} |")

    with open(osp.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report → {osp.join(out_dir, 'report.md')}")
    return results


if __name__ == "__main__":
    run()
