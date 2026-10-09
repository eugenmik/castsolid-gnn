"""Shared pytest fixtures: a minimal gmsh-2.2 mesh for unit tests.

The mesh is a unit-ish tetrahedron-based shape with 5 nodes:
  - one internal-ish apex, 4 base nodes forming two tets
  - two boundary triangles tagged 'Cooling', one tagged 'Adiabatic'
Coordinates are in mm.
"""
import textwrap
import pytest

# node ids 1..5 ; coords in mm
MINI_MSH = textwrap.dedent("""\
    $MeshFormat
    2.2 0 8
    $EndMeshFormat
    $PhysicalNames
    3
    2 1 "Adiabatic"
    2 2 "Cooling"
    3 3 "Volume"
    $EndPhysicalNames
    $Nodes
    5
    1 0 0 0
    2 10 0 0
    3 0 10 0
    4 0 0 10
    5 3 3 3
    $EndNodes
    $Elements
    4
    1 2 2 2 1 1 2 3
    2 2 2 2 1 1 2 4
    3 2 2 1 1 2 3 4
    4 4 2 3 1 1 2 3 4
    $EndElements
    """)
# Element lines: "id type ntags phys geom n1 n2 n3[ n4]"
#  el1: triangle (type 2), phys 2 = Cooling, nodes 1,2,3
#  el2: triangle, phys 2 = Cooling, nodes 1,2,4
#  el3: triangle, phys 1 = Adiabatic, nodes 2,3,4
#  el4: tetrahedron (type 4), phys 3 = Volume, nodes 1,2,3,4


@pytest.fixture
def mini_msh_file(tmp_path):
    p = tmp_path / "MINI_00.msh"
    p.write_text(MINI_MSH)
    return str(p)


@pytest.fixture
def tiny_weights_dir(tmp_path):
    """A folder in the published layout (config.json + safetensors GNN + two
    skops heads + OOD ranges) holding a tiny randomly initialised model."""
    import json
    import numpy as np
    import torch
    import skops.io as sio
    from safetensors.torch import save_file
    from sklearn.ensemble import HistGradientBoostingRegressor
    from src.model_gnn import SolidGNN
    from src.release_io import GNN_FORMAT

    d = tmp_path / "weights"
    d.mkdir()
    torch.manual_seed(0)
    cfg = {"node_in": 15, "edge_in": 5, "hidden": 16, "n_blocks": 2, "g_dim": 5, "k": 4}
    tensors = {f"model.{k}": v.contiguous()
               for k, v in SolidGNN(15, 5, 16, 2).state_dict().items()}
    tensors["norm.feat_mean"] = torch.zeros(15)
    tensors["norm.feat_std"] = torch.ones(15)
    save_file(tensors, str(d / "gnn.safetensors"), metadata={
        "format": GNN_FORMAT, "cfg": json.dumps(cfg),
        "norm_kinds": json.dumps({"feat_mean": "numpy", "feat_std": "numpy"}),
        "norm_scalars": json.dumps({"y_log_mean": 3.0, "y_log_std": 0.5})})
    rng = np.random.default_rng(0)
    for name, n in (("total.skops", 16), ("spread.skops", 17)):
        X = rng.normal(size=(100, n))
        m = HistGradientBoostingRegressor(max_iter=10).fit(X, 4.0 + 0.1 * X[:, 0])
        sio.dump(m, str(d / name))
    (d / "ood.json").write_text(json.dumps(
        {"volume_mm3_max": 1e9, "bbox_diag_max": 1e4, "geodesic_d_cool_max": 1e3}))
    (d / "config.json").write_text(json.dumps({
        "version": "test",
        "files": {"gnn": "gnn.safetensors", "total_head": "total.skops",
                  "spread_head": "spread.skops", "ood_ranges": "ood.json"},
        "recomposition": {"quantile": 0.995, "w_total": 1.0, "w_std": 4.0}}))
    return str(d)
