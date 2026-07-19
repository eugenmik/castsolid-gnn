"""Evaluation metrics for the solidification-time field (all in seconds)."""
import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def median_rel_error(y_true, y_pred, eps: float = 1.0) -> float:
    return float(np.median(np.abs(y_true - y_pred) / (y_true + eps)))


def _per_graph(graph_id):
    order = np.argsort(graph_id, kind="stable")
    gid = graph_id[order]
    bounds = np.flatnonzero(np.diff(gid)) + 1
    return order, np.split(order, bounds)


def pred_std_ratio(y_true, y_pred, graph_id) -> float:
    _, groups = _per_graph(np.asarray(graph_id))
    ratios = []
    for idx in groups:
        st = np.std(y_true[idx])
        if st > 1e-6:
            ratios.append(np.std(y_pred[idx]) / st)
    return float(np.mean(ratios)) if ratios else 0.0


def hotspot_accuracy(y_true, y_pred, xyz, graph_id,
                     top_frac: float = 0.10, tol_mm: float = 5.0) -> float:
    _, groups = _per_graph(np.asarray(graph_id))
    accs = []
    for idx in groups:
        n = len(idx)
        k = max(1, int(round(n * top_frac)))
        true_top = idx[np.argsort(y_true[idx])[-k:]]
        pred_top = idx[np.argsort(y_pred[idx])[-k:]]
        true_xyz = xyz[true_top]
        hits = 0
        for p in pred_top:
            d = np.linalg.norm(true_xyz - xyz[p], axis=1)
            if d.min() <= tol_mm:
                hits += 1
        accs.append(hits / k)
    return float(np.mean(accs)) if accs else 0.0
