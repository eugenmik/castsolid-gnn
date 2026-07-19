import numpy as np
from src.baseline import train_hgb, train_mlp, predict_seconds


def _norm():
    return {"feat_mean": np.zeros(3), "feat_std": np.ones(3),
            "y_log_mean": 0.0, "y_log_std": 1.0}


def test_hgb_learns_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = np.expm1(0.8 * X[:, 0] + 2.0)              # seconds; monotone in feature 0
    ylog = np.log1p(y)
    model = train_hgb(X, ylog, max_iter=50)
    pred = predict_seconds(model, X, _norm())
    # correlation with truth should be strong
    assert np.corrcoef(pred, y)[0, 1] > 0.9


def test_mlp_runs_and_predicts_positive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    ylog = (0.5 * X[:, 0] + 1.0).astype(np.float32)
    model = train_mlp(X, ylog, epochs=5)
    pred = predict_seconds(model, X, _norm())
    assert pred.shape == (200,)
    assert (pred >= 0).all()
