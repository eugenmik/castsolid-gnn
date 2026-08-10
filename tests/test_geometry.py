import numpy as np
import pytest
from unittest.mock import patch
from src.geometry import (triangle_areas, total_area, mesh_volume,
                          nearest_distance, boundary_node_mask,
                          geodesic_diameter_two_sweep)


def test_triangle_area_right_triangle():
    xyz = np.array([[0, 0, 0], [4, 0, 0], [0, 3, 0]], float)
    tri = np.array([[0, 1, 2]])
    np.testing.assert_allclose(triangle_areas(xyz, tri), [6.0])  # 0.5*4*3
    np.testing.assert_allclose(total_area(xyz, tri), 6.0)


def test_tet_volume_unit():
    # tetrahedron with legs 3,3,3 along axes from origin -> vol = 3^3/6 = 4.5
    xyz = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]], float)
    tets = np.array([[0, 1, 2, 3]])
    np.testing.assert_allclose(mesh_volume(xyz, tets), 4.5)


def test_nearest_distance():
    pts = np.array([[0, 0, 0], [10, 0, 0]], float)
    tgt = np.array([[0, 0, 0], [1, 0, 0]], float)
    np.testing.assert_allclose(nearest_distance(pts, tgt), [0.0, 9.0])


def test_nearest_distance_empty_targets():
    pts = np.array([[0, 0, 0]], float)
    tgt = np.empty((0, 3), float)
    d = nearest_distance(pts, tgt)
    assert np.isinf(d).all()


def test_boundary_node_mask():
    tri = np.array([[0, 1, 2], [1, 2, 3]])
    m = boundary_node_mask(5, tri)
    np.testing.assert_array_equal(m, [True, True, True, True, False])


# --------------------------------------------------------------------- #
# geodesic_diameter_two_sweep (bc-reach re-review N1): direct unit tests.
# The bc-reach features-v5 tests only ever use this helper as an
# expected-value oracle on real/synthetic meshes -- a "d2 = d1" single-
# sweep mutant (skip the second Dijkstra entirely, just report sweep-1's
# own max) passes all of them, because none of those fixtures put the
# fixed `start=0` at a graph MIDPOINT where a single sweep provably can't
# see the true diameter. These tests target that property directly.
# --------------------------------------------------------------------- #
def _mid_start_path_graph(edge_len_mm=10.0):
    """A 5-node path graph -- the chain order (by position) is
    node1-node2-node0-node3-node4 -- where node INDEX 0 (this function's
    default `start`) sits at the graph's MIDDLE, not at either end. True
    diameter (node1 <-> node4, both ends) = 4 hops = 4*edge_len_mm. A
    single Dijkstra sweep from node0 only reaches 2 hops in either
    direction (both ends are tied at that distance) -- it structurally
    cannot see the two ends' mutual distance -- so a "d2 = d1" mutant
    (reporting sweep-1's own max instead of re-sweeping from sweep-1's
    farthest node) returns 2*edge_len_mm here: HALF the true diameter."""
    edges = np.array([[1, 2], [2, 0], [0, 3], [3, 4]], dtype=np.int64)
    edge_len = np.full(4, edge_len_mm, dtype=float)
    return 5, edges, edge_len


def test_geodesic_diameter_two_sweep_mid_start_finds_true_diameter():
    n_nodes, edges, edge_len = _mid_start_path_graph(edge_len_mm=10.0)
    d_geo = geodesic_diameter_two_sweep(n_nodes, edges, edge_len)  # default start=0
    # 4 hops * 10mm = 40.0 -- a single-sweep-from-the-middle mutant would
    # instead report 20.0 (2 hops, the farthest EITHER end appears from
    # the midpoint), so this assertion fails under that mutant.
    assert d_geo == pytest.approx(40.0)


def test_geodesic_diameter_two_sweep_edgeless_returns_zero():
    edges = np.empty((0, 2), dtype=np.int64)
    edge_len = np.empty(0, dtype=float)
    assert geodesic_diameter_two_sweep(5, edges, edge_len) == 0.0


def test_geodesic_diameter_two_sweep_zero_nodes_returns_zero():
    edges = np.empty((0, 2), dtype=np.int64)
    edge_len = np.empty(0, dtype=float)
    assert geodesic_diameter_two_sweep(0, edges, edge_len) == 0.0


def test_geodesic_diameter_two_sweep_no_finite_distances_returns_zero():
    # The documented "no finite distances" guard is unreachable through any
    # REAL Dijkstra call (a node's distance to itself is always 0, so
    # sweep-1 always has at least one finite entry) -- exercised directly
    # here by forcing the underlying primitive to report all-inf, to prove
    # the guard degrades to 0.0 instead of crashing (e.g. np.argmax over an
    # all-False finite mask) if that invariant is ever violated.
    n_nodes, edges, edge_len = _mid_start_path_graph()
    with patch("src.geometry.geodesic_distance",
              return_value=np.full(n_nodes, np.inf)):
        assert geodesic_diameter_two_sweep(n_nodes, edges, edge_len) == 0.0


def test_geodesic_diameter_two_sweep_deterministic():
    n_nodes, edges, edge_len = _mid_start_path_graph()
    d_a = geodesic_diameter_two_sweep(n_nodes, edges, edge_len, start=0)
    d_b = geodesic_diameter_two_sweep(n_nodes, edges, edge_len, start=0)
    assert d_a == d_b   # bit-equal: deterministic, fixed start, no RNG
