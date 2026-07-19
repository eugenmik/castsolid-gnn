import numpy as np
from src.targets import solidification_time, global_time, T_SOLIDUS


def test_linear_crossing():
    # one node cools 1200 -> 1100 between t=0 and t=10; crosses 1120 at t=8
    times = np.array([0.0, 10.0])
    temps = np.array([[1200.0], [1100.0]])
    y = solidification_time(temps, times)
    np.testing.assert_allclose(y, [8.0])


def test_never_crosses_uses_last_time():
    times = np.array([0.0, 5.0])
    temps = np.array([[1300.0], [1200.0]])  # stays above solidus
    y = solidification_time(temps, times)
    np.testing.assert_allclose(y, [5.0])


def test_multi_node_and_global():
    times = np.array([0.0, 10.0])
    # node A crosses at 8s, node B already below at t=0
    temps = np.array([[1200.0, 1000.0], [1100.0, 900.0]])
    y = solidification_time(temps, times)
    np.testing.assert_allclose(y, [8.0, 0.0])
    assert global_time(y) == 8.0


def test_solidus_constant():
    assert T_SOLIDUS == 1120.0
