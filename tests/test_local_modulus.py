import numpy as np
from scipy.spatial import Delaunay

from src.geometry import (local_modulus, tet_volumes, edges_from_tets,
                          mesh_volume, total_area)


def _cube(a=10.0):
    pts = np.array([[x, y, z] for x in (0, a) for y in (0, a) for z in (0, a)], float)
    dt = Delaunay(pts)
    tets = dt.simplices.astype(np.int64)
    cool = dt.convex_hull.astype(np.int64)          # all 12 boundary triangles
    edges = edges_from_tets(tets)
    elen = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
    return pts, tets, cool, edges, elen


def _box_grid(n=4, a=10.0):
    """A structured grid of n^3 points spanning a cube of side `a`, Delaunay
    tetrahedralized, with the convex-hull triangles as the cooling surface.
    Denser than `_cube` (n=2) -> stands in for a training-density mesh."""
    lin = np.linspace(0.0, a, n)
    pts = np.array([[x, y, z] for x in lin for y in lin for z in lin], float)
    dt = Delaunay(pts)
    tets = dt.simplices.astype(np.int64)
    cool = dt.convex_hull.astype(np.int64)
    edges = edges_from_tets(tets)
    elen = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
    return pts, tets, cool, edges, elen


def test_tet_volumes_sum_to_mesh_volume():
    xyz, tets, _, _, _ = _cube()
    assert np.isclose(tet_volumes(xyz, tets).sum(), mesh_volume(xyz, tets))
    assert np.isclose(mesh_volume(xyz, tets), 1000.0)


def test_large_radius_equals_global_v_over_a():
    xyz, tets, cool, edges, elen = _cube()
    m = local_modulus(xyz, tets, cool, edges, elen, radii=[1e6])
    expected = mesh_volume(xyz, tets) / total_area(xyz, cool)   # 1000/600
    assert m.shape == (8, 1) and np.allclose(m[:, 0], expected, rtol=1e-6)


def test_no_cooling_gives_bbox_sentinel():
    xyz, tets, _, edges, elen = _cube()
    m = local_modulus(xyz, tets, np.empty((0, 3), np.int64), edges, elen, radii=[1e6])
    assert np.allclose(m[:, 0], np.linalg.norm(xyz.max(0) - xyz.min(0)))


def test_tiny_radius_gives_sentinel():
    # ball = the node itself: no complete tet, no complete cooling tri -> sentinel
    xyz, tets, cool, edges, elen = _cube()
    m = local_modulus(xyz, tets, cool, edges, elen, radii=[0.1])
    assert np.allclose(m[:, 0], np.linalg.norm(xyz.max(0) - xyz.min(0)))


def test_multiple_radii_columns_ordered():
    xyz, tets, cool, edges, elen = _cube()
    m = local_modulus(xyz, tets, cool, edges, elen, radii=[0.1, 1e6])
    assert m.shape == (8, 2)


