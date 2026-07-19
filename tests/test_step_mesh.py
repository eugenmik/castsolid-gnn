import numpy as np
import pytest
from src.step_mesh import auto_elem_size, mesh_step
from src.msh_parse import parse_msh
from src.geometry import boundary_node_mask


def test_auto_elem_size():
    assert auto_elem_size(600.0) == pytest.approx(10.0)


def _make_box_step(path, dx=10.0, dy=20.0, dz=30.0):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, dx, dy, dz)
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def test_mesh_step_box(tmp_path):
    step = str(tmp_path / "box.step")
    _make_box_step(step, 10, 20, 30)
    out = str(tmp_path / "box.msh")
    info = mesh_step(step, out, elem_size=5.0)   # coarse -> fast

    assert info["n_nodes"] > 0 and info["n_tets"] > 0
    np.testing.assert_allclose(info["bbox_mm"], (10, 20, 30), atol=1e-3)

    mesh = parse_msh(out)
    assert mesh.node_xyz.shape[0] == info["n_nodes"]
    assert "Cooling" in mesh.tris and mesh.tris["Cooling"].shape[0] > 0
    # box has no interior nodes at coarse size: every node is on the boundary
    N = mesh.node_xyz.shape[0]
    cool = boundary_node_mask(N, mesh.tris["Cooling"])
    assert cool.sum() >= 8   # at least the corners are cooling-tagged
