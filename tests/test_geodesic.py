import numpy as np
from src.geometry import edges_from_tets, geodesic_distance


def test_edges_from_tets_unique_undirected():
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    e = edges_from_tets(tets)
    # each tet has 6 edges; shared edges {1,2},{1,3},{2,3} dedup -> 9 unique
    assert e.shape == (9, 2)
    assert (e[:, 0] < e[:, 1]).all()          # sorted per row
    assert len({tuple(r) for r in e}) == 9     # unique


def test_geodesic_path_line():
    # chain 0-1-2-3 with unit edges; source={0}
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    edge_len = np.array([1.0, 1.0, 1.0])
    d = geodesic_distance(4, edges, edge_len, np.array([0]))
    np.testing.assert_allclose(d, [0, 1, 2, 3])


def test_geodesic_multisource_and_weights():
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    edge_len = np.array([2.0, 5.0, 2.0])
    d = geodesic_distance(4, edges, edge_len, np.array([0, 3]))
    # node1: min(2 from 0, 5+2 from 3)=2 ; node2: min(2+5,2 from 3)=2
    np.testing.assert_allclose(d, [0, 2, 2, 0])


def test_geodesic_empty_sources_is_inf():
    edges = np.array([[0, 1]])
    d = geodesic_distance(2, edges, np.array([1.0]), np.array([], dtype=int))
    assert np.isinf(d).all()
