import numpy as np
import pytest

from src import gbm_heads
from src.features import FEATURE_NAMES

PROPS = {"k": 32.9, "rho": 6800.0, "cp": 515.0, "L_kJ_kg": 220.0,
         "TL": 1160.0, "TS": 1120.0, "Tpour": 1260.0, "mold_effusivity": 1018.0}


def _X(n=10, vol=8000.0, area=2400.0, n_adiab=0):
    X = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    X[:, FEATURE_NAMES.index("volume_mm3")] = vol
    X[:, FEATURE_NAMES.index("surface_area_mm2")] = area
    X[:6, FEATURE_NAMES.index("is_cooling")] = 1
    X[6:6 + n_adiab, FEATURE_NAMES.index("is_adiabatic")] = 1
    return X


def test_total_row_layout():
    row = gbm_heads.total_row(PROPS, _X())
    assert row.shape == (gbm_heads.N_TOTAL_FEATURES,)
    assert row[:7].tolist() == [32.9, 6800.0, 515.0, 220.0, 1160.0, 1120.0, 1260.0]
    assert row[7] == pytest.approx(8000.0 / 2400.0)
    assert row[8] == 1018.0
    assert row[9:].tolist() == [1, 0, 0, 0, 0, 0, 0]


def test_spread_row_appends_adiabatic_fraction():
    X = _X(n_adiab=2)
    row = gbm_heads.spread_row(gbm_heads.total_row(PROPS, X), X)
    assert row.shape == (gbm_heads.N_SPREAD_FEATURES,)
    assert row[-1] == pytest.approx(2 / 8)


class _Stub:
    def __init__(self, value, n):
        self.value, self.n_features_in_ = value, n

    def predict(self, x):
        return np.full(len(x), self.value)


def test_predictions_are_expm1_of_log_output():
    row = np.zeros(16)
    assert gbm_heads.predict_total_seconds(_Stub(np.log1p(500.0), 16), row) == pytest.approx(500.0)
    assert gbm_heads.predict_spread_seconds(_Stub(-5.0, 17), np.zeros(17)) == 0.0
    with pytest.raises(ValueError):
        gbm_heads.predict_total_seconds(_Stub(-1.0, 16), row)


def test_check_head_rejects_wrong_width():
    gbm_heads.check_head(_Stub(0.0, 16), 16, "total-time")
    with pytest.raises(ValueError):
        gbm_heads.check_head(_Stub(0.0, 9), 16, "total-time")
