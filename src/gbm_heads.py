"""The two gradient-boosted heads: total solidification time and field spread.

Both are scikit-learn HistGradientBoostingRegressor models that see one row
per part and predict log1p of their target in seconds.

Row layout (16 columns for the total head, 17 for the spread head):
  0-6   alloy: k, rho, cp, L_kJ_kg, TL, TS, Tpour
  7     part volume / surface area (mm)
  8     mold effusivity
  9-15  boundary-condition descriptors from training. A casting that cools
        into the sand mold over its whole surface is the regime encoded as
        [1, 0, 0, 0, 0, 0, 0]; this release always uses it.
  16    (spread head only) fraction of surface nodes tagged adiabatic
"""
import numpy as np

from src.features import FEATURE_NAMES

_IS_COOLING_IDX = FEATURE_NAMES.index("is_cooling")
_IS_ADIABATIC_IDX = FEATURE_NAMES.index("is_adiabatic")
_VOL_IDX = FEATURE_NAMES.index("volume_mm3")
_AREA_IDX = FEATURE_NAMES.index("surface_area_mm2")

_SAND_MOLD_BOUNDARY = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
N_TOTAL_FEATURES = 16
N_SPREAD_FEATURES = 17


def total_row(props: dict, X: np.ndarray) -> np.ndarray:
    """The 16-column row for the total-time head."""
    v_over_a = float(X[0, _VOL_IDX]) / float(X[0, _AREA_IDX])
    return np.array([
        props["k"], props["rho"], props["cp"], props["L_kJ_kg"],
        props["TL"], props["TS"], props["Tpour"],
        float(v_over_a), props["mold_effusivity"],
        *_SAND_MOLD_BOUNDARY,
    ], dtype=np.float64)


def adiabatic_node_fraction(X: np.ndarray) -> float:
    """Adiabatic surface nodes / (adiabatic + cooling surface nodes)."""
    n_adiabatic = float(np.sum(X[:, _IS_ADIABATIC_IDX]))
    n_cooling = float(np.sum(X[:, _IS_COOLING_IDX]))
    return n_adiabatic / max(n_adiabatic + n_cooling, 1.0)


def spread_row(row_total: np.ndarray, X: np.ndarray) -> np.ndarray:
    """The 17-column row for the spread head."""
    return np.concatenate([row_total, [adiabatic_node_fraction(X)]])


def predict_total_seconds(model, row: np.ndarray) -> float:
    total = float(np.expm1(float(model.predict(row.reshape(1, -1))[0])))
    if total <= 0:
        raise ValueError(f"total-time head returned a non-positive total ({total!r})")
    return total


def predict_spread_seconds(model, row: np.ndarray) -> float:
    x = np.asarray(row, dtype=np.float64).reshape(1, -1)
    log_pred = float(model.predict(x)[0])
    return max(0.0, float(np.expm1(log_pred)))


def check_head(model, n_features: int, name: str) -> None:
    n = getattr(model, "n_features_in_", None)
    if n is not None and int(n) != n_features:
        raise ValueError(f"{name} head expects {n} input columns, this code builds "
                         f"{n_features}")
