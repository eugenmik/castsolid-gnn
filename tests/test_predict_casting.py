import os.path as osp
import numpy as np
import torch
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from src.model_gnn import SolidGNN
from src.predict_casting import predict_casting


def _tiny_artifacts(tmp_path):
    # tiny GNN checkpoint
    gnn = SolidGNN(n_blocks=2, hidden=16)
    norm = {"feat_mean": np.zeros(9), "feat_std": np.ones(9),
            "y_log_mean": 0.0, "y_log_std": 1.0}
    ck = str(tmp_path / "gnn.pt")
    torch.save({"model": gnn.state_dict(), "norm": norm,
                "cfg": {"hidden": 16, "n_blocks": 2, "k": 6}}, ck)
    # tiny HGB
    rng = np.random.default_rng(0)
    hgb = HistGradientBoostingRegressor(max_iter=5).fit(rng.normal(size=(80, 9)),
                                                        np.abs(rng.normal(size=80)))
    hp = str(tmp_path / "hgb.joblib"); joblib.dump(hgb, hp)
    return ck, hp


def _box_step(path):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("b"); gmsh.model.occ.addBox(0, 0, 0, 20, 20, 20)
        gmsh.model.occ.synchronize(); gmsh.write(path)
    finally:
        gmsh.finalize()


def test_predict_casting_end_to_end(tmp_path):
    step = str(tmp_path / "part.step"); _box_step(step)
    ck, hp = _tiny_artifacts(tmp_path)
    out = str(tmp_path / "out")
    res = predict_casting(step, out_dir=out, elem_size=5.0, gnn_ckpt=ck, hgb_path=hp,
                          ood_json="/nonexistent.json", device="cpu")
    assert osp.exists(res["vtu"]) and osp.exists(res["summary"])
    assert res["total_time_s"] > 0 and res["n_nodes"] > 0
    import meshio
    m = meshio.read(res["vtu"])
    assert "solidification_time" in m.point_data and "hotspot" in m.point_data
