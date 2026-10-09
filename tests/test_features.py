import numpy as np
import pytest

from src.features import FEATURE_NAMES, alloy_g_vector, node_features
from src.msh_parse import parse_msh

MOLD_EFF = 1018.0


def test_shapes_and_order(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    f = node_features(mesh, MOLD_EFF)
    assert f["X"].shape == (5, len(FEATURE_NAMES)) == (5, 15)
    assert f["X"].dtype == np.float32
    assert f["pos"].shape == (5, 3)
    assert f["mesh_edges"].shape == (6, 2)          # one tet -> 6 edges


def test_flags_and_boundary_channels(mini_msh_file):
    X = node_features(parse_msh(mini_msh_file), MOLD_EFF)["X"]
    col = {n: X[:, i] for i, n in enumerate(FEATURE_NAMES)}
    # nodes 0..3 lie on the tagged surface, node 4 (index 4) is not in any tet
    assert col["is_cooling"][:4].tolist() == [1, 1, 1, 1]
    assert col["is_adiabatic"].sum() == 0           # adiabatic nodes are also cooling here
    assert col["is_interior"][4] == 1
    np.testing.assert_allclose(col["boundary_extraction"][:4], MOLD_EFF)
    assert col["boundary_extraction"][4] == 0
    assert np.all(col["boundary_heat_input"] == 0)


def test_global_columns_are_constant(mini_msh_file):
    X = node_features(parse_msh(mini_msh_file), MOLD_EFF)["X"]
    for name in ("volume_mm3", "surface_area_mm2", "cooling_area_frac"):
        c = X[:, FEATURE_NAMES.index(name)]
        assert np.all(c == c[0])
    assert X[0, FEATURE_NAMES.index("volume_mm3")] == pytest.approx(1000.0 / 6.0)


def test_translation_invariance(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    a = node_features(mesh, MOLD_EFF)["X"]
    mesh.node_xyz = mesh.node_xyz + np.array([100.0, -50.0, 7.0])
    b = node_features(mesh, MOLD_EFF)["X"]
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)


def test_g_vector():
    props = {"k": 32.9, "rho": 6800.0, "cp": 515.0, "L_kJ_kg": 220.0,
             "TL": 1160.0, "TS": 1120.0, "Tpour": 1260.0, "mold_effusivity": 1250.0}
    g = alloy_g_vector(props)
    assert g.shape == (5,)
    assert g[0] == pytest.approx(515.0 * 140.0 / 220000.0)        # Stefan number
    assert g[1] == pytest.approx(0.04) and g[3] == pytest.approx(0.1)
    assert g[2] == pytest.approx(32.9 / (6800.0 * 515.0) / 1e-5)
    assert g[4] == pytest.approx(1.0)
