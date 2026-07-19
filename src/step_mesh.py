"""STEP -> tetrahedral gmsh 2.2 mesh with all external surface tagged 'Cooling'.

Uses the gmsh Python API (bundled OpenCASCADE) to import a STEP solid, tag the
volume and its full boundary, mesh uniformly, and export in the gmsh 2.2 format
consumed by src/msh_parse.py. All lengths in mm."""
import numpy as np
import gmsh


def auto_elem_size(bbox_diag: float) -> float:
    """Target element size (mm): bbox diagonal / 60, the validated Phase 2A
    density chosen to give training-comparable node density on real castings."""
    return bbox_diag / 60.0


def mesh_step(step_path: str, out_msh: str, elem_size: float | None = None) -> dict:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("casting")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()

        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        bbox = (xmax - xmin, ymax - ymin, zmax - zmin)
        diag = float(np.linalg.norm(bbox))
        size = float(elem_size) if elem_size is not None else auto_elem_size(diag)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)

        vols = [t for (d, t) in gmsh.model.getEntities(3)]
        surfs = [t for (d, t) in gmsh.model.getEntities(2)]
        pv = gmsh.model.addPhysicalGroup(3, vols); gmsh.model.setPhysicalName(3, pv, "Volume")
        pc = gmsh.model.addPhysicalGroup(2, surfs); gmsh.model.setPhysicalName(2, pc, "Cooling")

        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(out_msh)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        tet_tags = gmsh.model.mesh.getElementsByType(4)[0]
        return {"n_nodes": len(node_tags), "n_tets": len(tet_tags),
                "bbox_mm": bbox, "elem_size": size}
    finally:
        gmsh.finalize()
