"""Load cached per-node features for a set of stems into flat matrices."""
import os.path as osp
import numpy as np


def load_norm(features_dir: str) -> dict:
    d = np.load(osp.join(features_dir, "norm_stats.npz"))
    return {"feat_mean": d["feat_mean"], "feat_std": d["feat_std"],
            "y_log_mean": float(d["y_log_mean"]), "y_log_std": float(d["y_log_std"])}


def load_matrix(features_dir: str, stems, max_nodes_per_graph=None, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys, gids = [], [], []
    for gi, stem in enumerate(stems):
        d = np.load(osp.join(features_dir, "graphs", f"{stem}.npz"))
        X, y = d["X"], d["y"]
        if max_nodes_per_graph is not None and X.shape[0] > max_nodes_per_graph:
            idx = rng.choice(X.shape[0], max_nodes_per_graph, replace=False)
            X, y = X[idx], y[idx]
        Xs.append(X)
        ys.append(y)
        gids.append(np.full(X.shape[0], gi, dtype=np.int64))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gids))


def standardize(X, norm) -> np.ndarray:
    return (X - norm["feat_mean"]) / norm["feat_std"]
