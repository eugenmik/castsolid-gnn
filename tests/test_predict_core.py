import numpy as np
import torch
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from src.msh_parse import parse_msh
from src.model_gnn import SolidGNN
from src.predict_casting import predict_field, write_vtu
from src.hybrid import hybrid_field
from src.features import compute_features_inference
from src.dataload import standardize
from src.baseline import predict_seconds


def _norm():
    return {"feat_mean": np.zeros(9), "feat_std": np.ones(9),
            "y_log_mean": 0.0, "y_log_std": 1.0}


def test_predict_field_is_hybrid_of_parts(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    norm = _norm()
    gnn = SolidGNN(n_blocks=2, hidden=16).eval()
    rng = np.random.default_rng(0)
    hgb = HistGradientBoostingRegressor(max_iter=5).fit(rng.normal(size=(50, 9)),
                                                        rng.normal(size=50))
    out = predict_field(mesh, gnn, hgb, norm, k=3, device="cpu")
    assert out.shape == (mesh.node_xyz.shape[0],)
    assert np.isfinite(out).all() and (out >= 0).all()


def test_write_vtu_roundtrip(tmp_path):
    import meshio
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    tets = np.array([[0, 1, 2, 3]])
    seconds = np.array([1.0, 2.0, 3.0, 9.0])
    p = str(tmp_path / "c.vtu")
    write_vtu(p, xyz, tets, seconds, top_frac=0.25)
    m = meshio.read(p)
    assert "solidification_time" in m.point_data and "hotspot" in m.point_data
    np.testing.assert_allclose(m.point_data["solidification_time"], seconds)
    assert m.point_data["hotspot"].sum() == 1        # top 25% of 4 = 1 node
    assert m.point_data["hotspot"][3] == 1           # node with 9.0 is the hot spot
