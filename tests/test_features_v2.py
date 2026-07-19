import numpy as np
from src.msh_parse import parse_msh
from src.features import compute_features, FEATURE_NAMES


def test_feature_names_v2():
    assert FEATURE_NAMES == [
        "geodesic_d_cool", "d_surf", "d_adiab",
        "is_cooling", "is_adiabatic", "is_interior",
        "volume_mm3", "surface_area_mm2", "cooling_area_frac",
    ]


def test_compute_features_v2_shapes_and_pos(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    N = mesh.node_xyz.shape[0]
    times = np.array([0.0, 10.0]); temps = np.full((2, N), 1200.0); temps[1] = 1100.0
    out = compute_features(mesh, temps, times)
    assert out["X"].shape == (N, 9)
    assert out["X"].dtype == np.float32
    assert out["pos"].shape == (N, 3)
    np.testing.assert_allclose(out["pos"], mesh.node_xyz, rtol=0, atol=0)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    X = out["X"]
    # one-hot boundary flags
    flags = X[:, [idx["is_cooling"], idx["is_adiabatic"], idx["is_interior"]]]
    np.testing.assert_array_equal(flags.sum(1), np.ones(N))
    # geodesic_d_cool: cooling nodes are 0, finite everywhere, >= euclidean would be >=0
    assert np.isfinite(X[:, idx["geodesic_d_cool"]]).all()
    assert (X[:, idx["geodesic_d_cool"]] >= 0).all()


def test_geodesic_ge_euclidean_on_mini(mini_msh_file):
    # geodesic distance-to-cooling must be >= straight-line distance-to-cooling
    from src.geometry import nearest_distance, boundary_node_mask
    mesh = parse_msh(mini_msh_file)
    N = mesh.node_xyz.shape[0]
    times = np.array([0.0, 10.0]); temps = np.full((2, N), 1200.0); temps[1] = 1100.0
    out = compute_features(mesh, temps, times)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    cool = boundary_node_mask(N, mesh.tris.get("Cooling", np.empty((0, 3), int)))
    euclid = nearest_distance(mesh.node_xyz, mesh.node_xyz[cool])
    assert (out["X"][:, idx["geodesic_d_cool"]] >= euclid - 1e-4).all()
