import numpy as np
from src.train_gnn import train


def _cache(tmp_path, n_graphs=6):
    g = tmp_path / "graphs"; g.mkdir(parents=True)
    rng = np.random.default_rng(0); stems = []
    for i in range(n_graphs):
        N = 30
        X = rng.normal(size=(N, 9)).astype(np.float32)
        pos = rng.normal(size=(N, 3)).astype(np.float32) * 15
        y = np.abs(0.5 * X[:, 0] + 1).astype(np.float32) * 5      # signal in feat 0
        np.savez(g / f"G{i}.npz", X=X, y=y, pos=pos); stems.append(f"G{i}")
    np.savez(tmp_path / "norm_stats.npz", feat_mean=np.zeros(9), feat_std=np.ones(9),
             y_log_mean=np.float64(0.0), y_log_std=np.float64(1.0))
    return stems


def test_train_smoke_cpu(tmp_path):
    stems = _cache(tmp_path, 6)
    res = train(str(tmp_path), stems[:4], stems[4:], epochs=3, hidden=16, n_blocks=2,
                batch_size=2, k=6, device="cpu", out_dir=str(tmp_path / "run"), augment=True)
    assert res["epochs_run"] >= 1
    assert np.isfinite(res["best_val"])
    assert (tmp_path / "run" / "best_gnn.pt").exists()
