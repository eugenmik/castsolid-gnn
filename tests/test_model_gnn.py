import numpy as np
import torch

from src.model_gnn import (SolidGNN, knn_edge_index, load_gnn, predict_graph_seconds,
                           union_edge_attr, union_edge_index)

CFG = {"node_in": 15, "edge_in": 5, "hidden": 16, "n_blocks": 2, "g_dim": 5, "k": 4}


def _norm():
    return {"feat_mean": np.zeros(15, np.float32), "feat_std": np.ones(15, np.float32),
            "y_log_mean": 3.0, "y_log_std": 0.5}


def _state(seed=0):
    torch.manual_seed(seed)
    model = SolidGNN(node_in=15, edge_in=5, hidden=16, n_blocks=2)
    return {"model": model.state_dict(), "norm": _norm(), "cfg": CFG}


def test_knn_edges_are_symmetric():
    pos = torch.rand(30, 3)
    ei = knn_edge_index(pos, k=4)
    pairs = set(map(tuple, ei.t().tolist()))
    assert all((b, a) in pairs for a, b in pairs)


def test_union_graph_contains_mesh_edges_and_flags_them():
    pos = torch.rand(20, 3)
    mesh_edges = torch.tensor([[0, 19], [3, 7]])
    ei, is_mesh = union_edge_index(pos, mesh_edges, k=3)
    pairs = {tuple(p): f for p, f in zip(ei.t().tolist(), is_mesh.tolist())}
    for a, b in [(0, 19), (19, 0), (3, 7), (7, 3)]:
        assert pairs[(a, b)] == 1.0
    assert len(pairs) == ei.shape[1]                 # no duplicates
    ea = union_edge_attr(pos, ei, is_mesh)
    assert ea.shape == (ei.shape[1], 5)
    src, dst = ei
    np.testing.assert_allclose(ea[:, 3], (pos[dst] - pos[src]).norm(dim=1), rtol=1e-6)


def test_film_conditioning_changes_output():
    model, cfg, norm = load_gnn(_state(), "cpu")
    with torch.no_grad():
        for film in model.films:                     # make FiLM non-trivial
            film.net[-1].weight.normal_(0, 0.5)
    rng = np.random.default_rng(0)
    X, pos = rng.normal(size=(25, 15)), rng.uniform(0, 50, size=(25, 3))
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    a = predict_graph_seconds(model, cfg, norm, X, pos, edges, np.zeros(5))
    b = predict_graph_seconds(model, cfg, norm, X, pos, edges, np.ones(5))
    assert a.shape == (25,) and np.all(a >= 0)
    assert not np.allclose(a, b)


def test_prediction_ignores_node_order():
    model, cfg, norm = load_gnn(_state(1), "cpu")
    rng = np.random.default_rng(1)
    X, pos = rng.normal(size=(25, 15)), rng.uniform(0, 50, size=(25, 3))
    edges = np.array([[0, 1], [1, 2], [5, 9]])
    g = rng.normal(size=5)
    a = predict_graph_seconds(model, cfg, norm, X, pos, edges, g)
    perm = rng.permutation(25)
    inv = np.argsort(perm)
    b = predict_graph_seconds(model, cfg, norm, X[perm], pos[perm], inv[edges], g)
    np.testing.assert_allclose(a[perm], b, rtol=1e-4)
