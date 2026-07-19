import numpy as np
from pathlib import Path
from src.dataload import load_matrix, standardize, load_norm


def _cache(tmp_path):
    g = tmp_path / "graphs"; g.mkdir(parents=True)
    np.savez(g / "A.npz", X=np.ones((10, 15), np.float32), y=np.full(10, 5.0, np.float32))
    np.savez(g / "B.npz", X=np.full((4, 15), 2.0, np.float32), y=np.full(4, 9.0, np.float32))
    np.savez(tmp_path / "norm_stats.npz",
             feat_mean=np.zeros(15), feat_std=np.ones(15),
             y_log_mean=np.float64(0.0), y_log_std=np.float64(1.0))
    return tmp_path


def test_load_matrix_and_ids(tmp_path):
    root = _cache(tmp_path)
    X, y, gid = load_matrix(str(root), ["A", "B"])
    assert X.shape == (14, 15)
    assert y.shape == (14,)
    assert set(gid.tolist()) == {0, 1}
    assert (gid == 0).sum() == 10 and (gid == 1).sum() == 4


def test_subsample_per_graph(tmp_path):
    root = _cache(tmp_path)
    X, y, gid = load_matrix(str(root), ["A", "B"], max_nodes_per_graph=3, seed=0)
    assert (gid == 0).sum() == 3      # A capped at 3
    assert (gid == 1).sum() == 3      # B has 4 -> capped at 3


def test_standardize(tmp_path):
    root = _cache(tmp_path)
    norm = load_norm(str(root))
    Xs = standardize(np.full((2, 15), 3.0), norm)
    np.testing.assert_allclose(Xs, 3.0)   # mean 0 std 1 -> unchanged
