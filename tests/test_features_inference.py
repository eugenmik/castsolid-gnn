import numpy as np
from src.msh_parse import parse_msh
from src.features import compute_features, compute_features_inference, FEATURE_NAMES


def test_inference_matches_training_features(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    N = mesh.node_xyz.shape[0]
    times = np.array([0.0, 10.0]); temps = np.full((2, N), 1200.0); temps[1] = 1100.0
    full = compute_features(mesh, temps, times)
    inf = compute_features_inference(mesh)
    assert inf["X"].shape == (N, len(FEATURE_NAMES))
    assert "y" not in inf
    np.testing.assert_array_equal(inf["X"], full["X"])       # identical feature columns
    np.testing.assert_array_equal(inf["pos"], full["pos"])
    assert np.isfinite(inf["X"]).all()
