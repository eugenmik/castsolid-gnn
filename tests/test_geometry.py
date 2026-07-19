import numpy as np
from src.geometry import (triangle_areas, total_area, mesh_volume,
                          nearest_distance, boundary_node_mask)


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
