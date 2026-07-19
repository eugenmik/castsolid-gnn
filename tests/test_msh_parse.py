import numpy as np
from src.msh_parse import parse_msh


def test_parse_mini(mini_msh_file):
    mesh = parse_msh(mini_msh_file)

    # 5 nodes, in id order, in mm
    assert mesh.node_xyz.shape == (5, 3)
    np.testing.assert_allclose(mesh.node_xyz[0], [0, 0, 0])
    np.testing.assert_allclose(mesh.node_xyz[4], [3, 3, 3])

    # physical groups present, 0-based indices
    assert set(mesh.tris.keys()) == {"Cooling", "Adiabatic"}
    assert mesh.tris["Cooling"].shape == (2, 3)
    assert mesh.tris["Adiabatic"].shape == (1, 3)
    # cooling triangle 1 = nodes 1,2,3 (ids) -> indices 0,1,2
    np.testing.assert_array_equal(sorted(mesh.tris["Cooling"][0]), [0, 1, 2])

    # one tetrahedron, 0-based
    assert mesh.tets.shape == (1, 4)
    np.testing.assert_array_equal(sorted(mesh.tets[0]), [0, 1, 2, 3])
