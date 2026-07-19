import numpy as np
import torch
from pathlib import Path
from src.run_baseline import evaluate_model
from src.baseline import train_hgb
from src.dataload import load_matrix, load_norm


def _cache_with_pt(root: Path):
    g = root / "features" / "graphs"; g.mkdir(parents=True)
    (root / "pt").mkdir(parents=True)
    rng = np.random.default_rng(0)
    stems = []
    for name in ["A", "B"]:
        N = 50
        X = rng.normal(size=(N, 15)).astype(np.float32)
        y = np.expm1(0.7 * X[:, 0] + 2.0).astype(np.float32)   # signal in feat 0
        np.savez(g / f"{name}.npz", X=X, y=y)
        torch.save({"pos": torch.tensor(rng.normal(size=(N, 3)) / 1000.0)},
                   root / "pt" / f"{name}.pt")   # metres
        stems.append(name)
    np.savez(root / "features" / "norm_stats.npz",
             feat_mean=np.zeros(15), feat_std=np.ones(15),
             y_log_mean=np.float64(0.0), y_log_std=np.float64(1.0))
    return stems


def test_evaluate_model_reports_metrics(tmp_path):
    stems = _cache_with_pt(tmp_path)
    fdir = str(tmp_path / "features")
    norm = load_norm(fdir)
    X, y, _ = load_matrix(fdir, stems)
    model = train_hgb(X, np.log1p(y), max_iter=50)
    res = evaluate_model(model, fdir, stems, norm, data_root=str(tmp_path))
    assert set(res) == {"mae", "rmse", "median_rel_error",
                        "pred_std_ratio", "hotspot_accuracy"}
    assert res["pred_std_ratio"] > 0.5     # learned real variance, not collapsed
