import numpy as np
from src.train_gnn import train
from src.evaluate_gnn import evaluate


def _cache(tmp_path, n=6):
    g = tmp_path / "graphs"; g.mkdir(parents=True)
    rng = np.random.default_rng(0); stems = []
    for i in range(n):
        N = 30
        X = rng.normal(size=(N, 9)).astype(np.float32)
        pos = rng.normal(size=(N, 3)).astype(np.float32) * 15
        y = np.abs(0.6 * X[:, 0] + 1).astype(np.float32) * 5
        np.savez(g / f"G{i}.npz", X=X, y=y, pos=pos); stems.append(f"G{i}")
    np.savez(tmp_path / "norm_stats.npz", feat_mean=np.zeros(9), feat_std=np.ones(9),
             y_log_mean=np.float64(0.0), y_log_std=np.float64(1.0))
    return stems


def test_evaluate_returns_metrics(tmp_path):
    stems = _cache(tmp_path, 6)
    res = train(str(tmp_path), stems[:4], stems[4:], epochs=3, hidden=16, n_blocks=2,
                batch_size=2, k=6, device="cpu", out_dir=str(tmp_path / "run"))
    m = evaluate(res["ckpt"], str(tmp_path), stems[4:], device="cpu")
    assert set(m) == {"mae", "rmse", "median_rel_error", "pred_std_ratio", "hotspot_accuracy"}
    assert np.isfinite(m["mae"])
