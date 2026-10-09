"""Public-release artifact formats round-trip exactly (src/release_io.py)."""
import os

import numpy as np
import pytest
import torch

from src.model_gnn import SolidGNN
from src import release_io as rio


def _ckpt():
    torch.manual_seed(0)
    cfg = {"hidden": 16, "n_blocks": 2, "node_in": 15, "edge_in": 5, "two_head": True,
           "film": True, "g_dim": 5, "graph": "union", "k": 16, "feature_version": 4}
    model = SolidGNN(node_in=15, edge_in=5, hidden=16, n_blocks=2, two_head=True,
                     film=True, g_dim=5)
    norm = {"feat_mean": np.arange(15, dtype=np.float32),
            "feat_std": np.ones(15, dtype=np.float64),
            "g_mean": torch.zeros(5), "y_log_mean": 3.5}
    return {"model": model.state_dict(), "norm": norm, "cfg": cfg}


def test_gnn_safetensors_roundtrip(tmp_path):
    pytest.importorskip("safetensors")
    state = _ckpt()
    out = str(tmp_path / "gnn.safetensors")
    rio.save_gnn_release(state, out)
    back = rio.load_gnn_state(out)
    assert back["cfg"] == state["cfg"]
    assert set(back["model"]) == set(state["model"])
    for k, v in state["model"].items():
        assert torch.equal(back["model"][k], v)
    n = back["norm"]
    assert isinstance(n["feat_mean"], np.ndarray) and n["feat_mean"].dtype == np.float32
    assert n["feat_std"].dtype == np.float64
    assert isinstance(n["g_mean"], torch.Tensor)
    assert n["y_log_mean"] == 3.5
    np.testing.assert_array_equal(n["feat_mean"], state["norm"]["feat_mean"])
    # load_gnn_any dispatches on the extension
    assert rio.load_gnn_any(out, "cpu")["cfg"] == state["cfg"]


def test_gnn_rejects_foreign_safetensors(tmp_path):
    st = pytest.importorskip("safetensors.torch")
    out = str(tmp_path / "other.safetensors")
    st.save_file({"w": torch.zeros(2)}, out)
    with pytest.raises(ValueError, match="not a"):
        rio.load_gnn_state(out)


def test_gbm_skops_roundtrip(tmp_path):
    pytest.importorskip("skops")
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 4)), rng.normal(size=200)
    m = HistGradientBoostingRegressor(max_iter=20).fit(X, y)
    out = str(tmp_path / "gbm.skops")
    rio.save_gbm_release(m, out)
    back = rio.load_gbm(out)
    np.testing.assert_array_equal(back.predict(X), m.predict(X))


def test_load_gbm_passthrough():
    assert rio.load_gbm(None) is None
    obj = object()
    assert rio.load_gbm(obj) is obj


def test_resolve_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    internal = "runs/ft3_gbm/hgb_v3.joblib"
    release = rio.RELEASE_PATHS[internal]
    assert rio.resolve_artifact(internal) == internal  # neither exists
    os.makedirs(os.path.dirname(release))
    open(release, "w").close()
    assert rio.resolve_artifact(internal) == release   # only release exists
    os.makedirs(os.path.dirname(internal))
    open(internal, "w").close()
    assert rio.resolve_artifact(internal) == internal  # internal wins
    assert rio.resolve_artifact("some/other.joblib") == "some/other.joblib"
    assert rio.resolve_artifact(None) is None
