import numpy as np
from src.metrics import (mae, rmse, median_rel_error, pred_std_ratio,
                         hotspot_accuracy)


def test_basic_errors():
    yt = np.array([0.0, 10.0, 20.0])
    yp = np.array([1.0, 10.0, 18.0])
    np.testing.assert_allclose(mae(yt, yp), (1 + 0 + 2) / 3)
    np.testing.assert_allclose(rmse(yt, yp), np.sqrt((1 + 0 + 4) / 3))


def test_pred_std_ratio_collapse():
    gid = np.array([0, 0, 0, 0])
    yt = np.array([0.0, 10.0, 20.0, 30.0])
    yp_flat = np.full(4, 15.0)          # total collapse
    assert pred_std_ratio(yt, yp_flat, gid) == 0.0
    assert abs(pred_std_ratio(yt, yt, gid) - 1.0) < 1e-9


def test_hotspot_accuracy_perfect():
    gid = np.zeros(10, int)
    yt = np.arange(10.0)                 # node 9 is the hot spot (top 10%)
    yp = np.arange(10.0)
    xyz = np.zeros((10, 3))
    assert hotspot_accuracy(yt, yp, xyz, gid, top_frac=0.1, tol_mm=5.0) == 1.0


def test_hotspot_accuracy_wrong_far():
    gid = np.zeros(10, int)
    yt = np.arange(10.0)                 # true hot spot = node 9
    yp = 9.0 - np.arange(10.0)           # predicted hot spot = node 0
    xyz = np.zeros((10, 3)); xyz[0] = [100, 0, 0]   # node0 far from node9
    assert hotspot_accuracy(yt, yp, xyz, gid, top_frac=0.1, tol_mm=5.0) == 0.0
