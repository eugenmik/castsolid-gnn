import numpy as np

from src.msh_parse import parse_msh
from src.features import (_node_features, compute_features_inference,
                          FEATURE_NAMES, FEATURE_NAMES_V3, LOCAL_MODULUS_RADII)


def test_v3_names_extend_v2():
    assert FEATURE_NAMES_V3[:9] == FEATURE_NAMES and len(FEATURE_NAMES_V3) == 13
    assert LOCAL_MODULUS_RADII == (5.0, 15.0, 40.0)


def test_v3_first_columns_match_v2(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    X2, pos2, e2 = _node_features(mesh, version=2)
    X3, pos3, e3 = _node_features(mesh, version=3)
    assert X2.shape[1] == 9 and X3.shape[1] == 13
    assert np.allclose(X3[:, :9], X2) and np.array_equal(e2, e3)


def test_geodesic_d_adiab_values(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    X3, _, _ = _node_features(mesh, version=3)
    col = FEATURE_NAMES_V3.index("geodesic_d_adiab")
    # 0-based nodes 1,2,3 lie on the Adiabatic triangle -> 0; node 0 reaches one
    # of them along a 10 mm tet edge
    assert np.allclose(X3[[1, 2, 3], col], 0.0)
    assert np.isclose(X3[0, col], 10.0)


def test_inference_dict_has_mesh_edges(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    out = compute_features_inference(mesh, version=3)
    assert out["X"].shape[1] == 13
    me = out["mesh_edges"]
    assert me.ndim == 2 and me.shape[1] == 2 and me.dtype == np.int64
