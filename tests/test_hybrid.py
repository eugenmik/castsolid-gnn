import numpy as np
from src.hybrid import hybrid_field


def test_hybrid_takes_shape_from_gnn_level_from_baseline():
    # GNN shape [-1,0,+1] around mean 5; baseline mean 20 (shape irrelevant)
    gnn = np.array([4.0, 5.0, 6.0])          # deviations -1,0,+1
    base = np.array([18.0, 22.0, 20.0])      # mean 20
    out = hybrid_field(gnn, base)
    # deviations preserved from GNN, level set to baseline mean (20)
    np.testing.assert_allclose(out, [19.0, 20.0, 21.0])


def test_hybrid_clips_negative():
    gnn = np.array([0.0, 10.0])              # mean 5, devs -5,+5
    base = np.array([1.0, 1.0])              # mean 1 -> level 1: -4, +6 -> clip -4 to 0
    out = hybrid_field(gnn, base)
    np.testing.assert_allclose(out, [0.0, 6.0])


def test_hybrid_preserves_within_graph_ranking():
    # hot-spot ranking (argsort) must come from the GNN shape, unchanged by re-leveling
    gnn = np.array([3.0, 1.0, 9.0, 5.0])
    base = np.array([100.0, 100.0, 100.0, 100.0])
    out = hybrid_field(gnn, base)
    assert list(np.argsort(out)) == list(np.argsort(gnn))
