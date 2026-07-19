"""Shared pytest fixtures: a minimal gmsh-2.2 mesh for unit tests.

The mesh is a unit-ish tetrahedron-based shape with 5 nodes:
  - one internal-ish apex, 4 base nodes forming two tets
  - two boundary triangles tagged 'Cooling', one tagged 'Adiabatic'
Coordinates are in mm.
"""
import textwrap
import pytest

# node ids 1..5 ; coords in mm
MINI_MSH = textwrap.dedent("""\
    $MeshFormat
    2.2 0 8
    $EndMeshFormat
    $PhysicalNames
    3
    2 1 "Adiabatic"
    2 2 "Cooling"
    3 3 "Volume"
    $EndPhysicalNames
    $Nodes
    5
    1 0 0 0
    2 10 0 0
    3 0 10 0
    4 0 0 10
    5 3 3 3
    $EndNodes
    $Elements
    4
    1 2 2 2 1 1 2 3
    2 2 2 2 1 1 2 4
    3 2 2 1 1 2 3 4
    4 4 2 3 1 1 2 3 4
    $EndElements
    """)
# Element lines: "id type ntags phys geom n1 n2 n3[ n4]"
#  el1: triangle (type 2), phys 2 = Cooling, nodes 1,2,3
#  el2: triangle, phys 2 = Cooling, nodes 1,2,4
#  el3: triangle, phys 1 = Adiabatic, nodes 2,3,4
#  el4: tetrahedron (type 4), phys 3 = Volume, nodes 1,2,3,4


@pytest.fixture
def mini_msh_file(tmp_path):
    p = tmp_path / "MINI_00.msh"
    p.write_text(MINI_MSH)
    return str(p)
