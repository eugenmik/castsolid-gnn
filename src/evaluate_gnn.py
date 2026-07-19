"""Evaluate a trained GNN on cache stems, reusing Phase 0 metrics. No augmentation."""
import numpy as np
import torch

from src.gnn_dataset import SolidGraphDataset
from src.model_gnn import SolidGNN
from src import metrics


def predict_seconds_gnn(pred_std_log, norm) -> np.ndarray:
    pred_log = pred_std_log * norm["y_log_std"] + norm["y_log_mean"]
    return np.clip(np.expm1(pred_log), 0.0, None)


def evaluate(ckpt, features_dir, test_stems, device=None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg, norm = state["cfg"], state["norm"]
    model = SolidGNN(hidden=cfg["hidden"], n_blocks=cfg["n_blocks"]).to(device)
    model.load_state_dict(state["model"]); model.eval()
    ds = SolidGraphDataset(features_dir, test_stems, k=cfg["k"], augment=False, norm=norm)

    ys, preds, xyz, gid = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            d = ds[i].to(device)
            yn, _ = model(d.x, d.edge_index, d.edge_attr,
                          torch.zeros(d.x.shape[0], dtype=torch.long, device=device))
            preds.append(predict_seconds_gnn(yn.cpu().numpy(), norm))
            # ground truth in seconds: invert the standardized-log y stored on d
            y_log = d.y.cpu().numpy() * norm["y_log_std"] + norm["y_log_mean"]
            ys.append(np.expm1(y_log))
            xyz.append(d.pos.cpu().numpy())
            gid.append(np.full(d.x.shape[0], i, dtype=np.int64))
    y = np.concatenate(ys); p = np.concatenate(preds)
    xyz = np.concatenate(xyz); gid = np.concatenate(gid)
    return {
        "mae": metrics.mae(y, p),
        "rmse": metrics.rmse(y, p),
        "median_rel_error": metrics.median_rel_error(y, p),
        "pred_std_ratio": metrics.pred_std_ratio(y, p, gid),
        "hotspot_accuracy": metrics.hotspot_accuracy(y, p, xyz, gid),
    }
