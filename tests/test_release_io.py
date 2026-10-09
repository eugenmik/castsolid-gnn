import collections
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from src import release_io as rio


def _write_gnn(path, fmt=rio.GNN_FORMAT):
    tensors = {"model.w": torch.arange(6, dtype=torch.float32).reshape(2, 3),
               "norm.feat_mean": torch.from_numpy(np.arange(3, dtype=np.float32))}
    meta = {"format": fmt, "cfg": json.dumps({"hidden": 16, "k": 4}),
            "norm_kinds": json.dumps({"feat_mean": "numpy"}),
            "norm_scalars": json.dumps({"y_log_mean": 3.5})}
    save_file(tensors, str(path), metadata=meta)


def test_load_gnn_state(tmp_path):
    p = tmp_path / "gnn.safetensors"
    _write_gnn(p)
    state = rio.load_gnn_state(str(p))
    assert state["cfg"] == {"hidden": 16, "k": 4}
    assert torch.equal(state["model"]["w"], torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert isinstance(state["norm"]["feat_mean"], np.ndarray)
    assert state["norm"]["feat_mean"].dtype == np.float32
    assert state["norm"]["y_log_mean"] == 3.5


def test_load_gnn_state_rejects_other_files(tmp_path):
    p = tmp_path / "other.safetensors"
    _write_gnn(p, fmt="something-else")
    with pytest.raises(ValueError, match="not a"):
        rio.load_gnn_state(str(p))


def test_load_gbm_roundtrip(tmp_path):
    skops = pytest.importorskip("skops.io")
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 4)), rng.normal(size=200)
    m = HistGradientBoostingRegressor(max_iter=20).fit(X, y)
    p = tmp_path / "gbm.skops"
    skops.dump(m, str(p))
    np.testing.assert_array_equal(rio.load_gbm(p).predict(X), m.predict(X))


def test_load_gbm_refuses_foreign_types(tmp_path):
    skops = pytest.importorskip("skops.io")
    p = tmp_path / "bad.skops"
    skops.dump({"c": collections.Counter(a=1)}, str(p))
    with pytest.raises(ValueError, match="unexpected types"):
        rio.load_gbm(p)
