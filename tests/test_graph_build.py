import torch
from src.graph_build import knn_edge_index, edge_attr, random_rotation_matrix, rotate


def test_knn_symmetric_no_selfloops():
    pos = torch.tensor([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    ei = knn_edge_index(pos, k=1)
    assert ei.shape[0] == 2
    assert (ei[0] != ei[1]).all()                       # no self-loops
    pairs = {(int(a), int(b)) for a, b in ei.t()}
    assert all((b, a) in pairs for a, b in pairs)        # symmetric


def test_edge_attr_mm():
    pos = torch.tensor([[0., 0, 0], [3, 4, 0]])
    ei = torch.tensor([[0], [1]])
    ea = edge_attr(pos, ei)
    torch.testing.assert_close(ea[0], torch.tensor([3., 4., 0., 5.]))  # dst-src + norm


def test_rotation_is_orthonormal_and_preserves_distances():
    R = random_rotation_matrix(seed=0)
    torch.testing.assert_close(R @ R.t(), torch.eye(3), atol=1e-5, rtol=1e-5)
    pos = torch.randn(10, 3)
    rp = rotate(pos, R)
    d0 = torch.cdist(pos, pos); d1 = torch.cdist(rp, rp)
    torch.testing.assert_close(d0, d1, atol=1e-4, rtol=1e-4)  # rotation preserves distances


def test_union_contains_mesh_edges_flagged():
    from src.graph_build import union_edge_index, union_edge_attr
    pos = torch.tensor([[0., 0, 0], [1, 0, 0], [2, 0, 0], [10, 10, 10]])
    me = torch.tensor([[0, 3]])                 # pair kNN(k=1) would never pick
    ei, ism = union_edge_index(pos, me, k=1)
    pairs = {(int(a), int(b)): float(f) for (a, b), f in zip(ei.t(), ism)}
    assert pairs[(0, 3)] == 1.0 and pairs[(3, 0)] == 1.0       # both directions, flagged
    assert all(f == 0.0 for p, f in pairs.items() if p not in {(0, 3), (3, 0)})
    key = ei[0] * pos.shape[0] + ei[1]
    assert len(torch.unique(key)) == ei.shape[1]               # no duplicates
    assert all((b, a) in pairs for a, b in pairs)              # symmetric
    ea = union_edge_attr(pos, ei, ism)
    assert ea.shape == (ei.shape[1], 5)
    torch.testing.assert_close(ea[:, 4], ism)


def test_union_dedups_overlap():
    from src.graph_build import union_edge_index
    pos = torch.tensor([[0., 0, 0], [1, 0, 0]])
    me = torch.tensor([[0, 1]])                 # same pair kNN also finds
    ei, ism = union_edge_index(pos, me, k=1)
    assert ei.shape[1] == 2                     # (0,1) and (1,0), once each
    assert (ism == 1.0).all()                   # mesh flag wins on overlap
