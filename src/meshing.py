"""STEP to tetrahedral mesh with Gmsh (OpenCASCADE kernel).

`heal_and_mesh` is the default path: import the STEP file, keep the largest
solid, tag every surface "Cooling" and the volume "Volume", and search the
element size until the mesh lands near `target_nodes` (the density the model
was trained on). If the unhealed geometry cannot be meshed, a second pass heals
it and retries. `mesh_step` meshes at one fixed element size instead.

Both write gmsh 2.2 ASCII meshes, the format `src.msh_parse` reads. All
lengths are in millimetres.
"""
from __future__ import annotations

import numpy as np

DEFAULT_TARGET_NODES = 8500
# Part of the mesh-cache key (src/mesh_cache.py): bump it whenever a change
# here would produce a different mesh for the same STEP file.
MESH_PIPELINE_VERSION = 3

# Element-size seed: sqrt(K_SURF * surface_area / target_nodes), calibrated so
# the first guess usually lands within the density search's tolerance.
K_SURF = 2.5

# Physical-group tags are pinned (not auto-assigned) so that repeated meshing
# attempts in one Gmsh session keep the "Volume"/"Cooling" names bound.
_VOLUME_PHYS_TAG = 1
_COOLING_PHYS_TAG = 2


def _reduce_to_largest_solid():
    """Keep only the largest solid of the imported OCC model.

    Returns (n_solids_in, kept_tag, kept_volume_mm3)."""
    import gmsh

    vols = [t for (d, t) in gmsh.model.occ.getEntities(3)]
    n_in = len(vols)
    if n_in == 0:
        raise ValueError("no 3D solids found in STEP import")
    masses = [(t, gmsh.model.occ.getMass(3, t)) for t in vols]
    kept_tag, kept_vol = max(masses, key=lambda tv: tv[1])
    if n_in > 1:
        drop = [(3, t) for (t, _v) in masses if t != kept_tag]
        gmsh.model.occ.remove(drop, recursive=True)
        gmsh.model.occ.synchronize()
    return n_in, kept_tag, float(kept_vol)


def _import_and_reduce(step_path: str) -> tuple[int, float, str | None]:
    """Import `step_path` into the current Gmsh session and keep its largest
    solid. Returns (n_solids_in, kept_volume_mm3, error_or_None)."""
    import gmsh

    try:
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
    except Exception as e:
        return 0, 0.0, f"import failed: {e}"
    try:
        n_solids, _kept_tag, volume_mm3 = _reduce_to_largest_solid()
    except Exception as e:
        return 0, 0.0, f"largest-solid reduction failed: {e}"
    return n_solids, volume_mm3, None


def _elem_size(bbox_diag_mm: float) -> float:
    """Fallback element-size seed: bounding-box diagonal / 28."""
    return bbox_diag_mm / 28.0


def _elem_size_from_area(area_mm2: float, target_nodes: int) -> float:
    """Element-size seed from the total surface area of the model."""
    return float(np.sqrt(K_SURF * area_mm2 / target_nodes))


def _total_surface_area_mm2() -> float:
    """Total area of all surface entities of the current OCC model."""
    import gmsh

    surfs = [t for (d, t) in gmsh.model.occ.getEntities(2)]
    return float(sum(gmsh.model.occ.getMass(2, t) for t in surfs))


def _first_guess_elem_size(bbox_diag_mm: float, target_nodes: int) -> float:
    """Surface-area seed, or the bounding-box seed if the area query fails."""
    try:
        area_mm2 = _total_surface_area_mm2()
    except Exception as e:
        print(f"[meshing] WARNING: OCC surface-area query failed ({e}); "
              f"falling back to bbox_diag/28 seed", flush=True)
        return _elem_size(bbox_diag_mm)
    if area_mm2 <= 0:
        print(f"[meshing] WARNING: OCC surface-area query returned "
              f"{area_mm2!r} (<=0); falling back to bbox_diag/28 seed", flush=True)
        return _elem_size(bbox_diag_mm)
    return _elem_size_from_area(area_mm2, target_nodes)


def _search_elem_size(build_fn, out_msh: str, target_nodes: int,
                      max_iter: int = 3) -> tuple[int, float]:
    """Refine the element size assuming the node count scales as h^-3."""
    n_nodes, elem_size = build_fn(out_msh, None)
    for _ in range(max_iter - 1):
        if n_nodes <= 0:
            break
        elem_size = elem_size * (n_nodes / target_nodes) ** (1.0 / 3.0)
        n_nodes, elem_size = build_fn(out_msh, elem_size)
    return n_nodes, elem_size


def _search_target_nodes(build_fn, out_msh: str, target_nodes: int,
                         tol: float = 0.35, max_iter: int = 3) -> tuple[int, float, int]:
    """Density search towards `target_nodes`.

    `build_fn(out_msh, elem_size) -> (n_nodes, elem_size_used)`, with
    `elem_size=None` meaning "use the seed" and `n_nodes == 0` meaning the
    mesh failed outright. Stops after the first attempt if it is already
    within `tol` of the target (or failed); otherwise refines for up to
    `max_iter` attempts. Returns (n_nodes, elem_size_used, attempts)."""
    calls = {"n": 0}

    def counted_build_fn(msh_path, elem_size):
        calls["n"] += 1
        return build_fn(msh_path, elem_size)

    n0, es0 = counted_build_fn(out_msh, None)
    if n0 <= 0 or abs(n0 - target_nodes) <= tol * target_nodes:
        return n0, es0, calls["n"]
    n, es = _search_elem_size(counted_build_fn, out_msh, target_nodes, max_iter=max_iter)
    return n, es, calls["n"]


def _tag_and_mesh(size: float, out_msh: str) -> tuple[bool, int, str]:
    """Tag the current model ("Volume" + every surface as "Cooling") and mesh
    it at element size `size`. Returns (ok, n_nodes, note).

    The tag numbers are pinned: Gmsh keeps a removed group's *name* bound to
    its old tag, so auto-assigned tags on a second attempt in the same
    session would leave the written mesh with unnamed groups."""
    import gmsh

    gmsh.model.mesh.clear()
    gmsh.model.removePhysicalGroups()
    vols = [t for (d, t) in gmsh.model.getEntities(3)]
    surfs = [t for (d, t) in gmsh.model.getEntities(2)]
    if not vols:
        return False, 0, "no volume entities to mesh"
    pv = gmsh.model.addPhysicalGroup(3, vols, tag=_VOLUME_PHYS_TAG)
    gmsh.model.setPhysicalName(3, pv, "Volume")
    pc = gmsh.model.addPhysicalGroup(2, surfs, tag=_COOLING_PHYS_TAG)
    gmsh.model.setPhysicalName(2, pc, "Cooling")
    # The mesh reader matches elements by group name, so an unnamed group
    # would silently look like a mesh without boundary. Fail loudly instead.
    if (gmsh.model.getPhysicalName(3, pv) != "Volume"
            or gmsh.model.getPhysicalName(2, pc) != "Cooling"):
        return False, 0, ("physical-group tagging did not register "
                          "(Volume/Cooling name binding missing)")
    gmsh.option.setNumber("Mesh.MeshSizeMin", size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    try:
        gmsh.model.mesh.generate(3)
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        try:
            tet_tags = gmsh.model.mesh.getElementsByType(4)[0]
        except Exception:
            tet_tags = []
        if len(node_tags) > 0 and len(tet_tags) > 0:
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(out_msh)
            return True, len(node_tags), ""
        return False, 0, f"empty mesh at size={size:.2f}mm"
    except Exception as e:
        return False, 0, f"mesh generate failed at size={size:.2f}mm: {e}"


def _apply_heal() -> tuple[bool, str]:
    """Run OCC shape healing. `healShapes()` can finish without raising and
    still leave no solid, so a heal only counts if a volume survives.
    Returns (effective, note)."""
    import gmsh

    try:
        gmsh.model.occ.healShapes()
        gmsh.model.occ.synchronize()
    except Exception as e:
        return False, f"healShapes raised: {e}"
    n_vols = len(gmsh.model.getEntities(3))
    if n_vols == 0:
        return False, ("healShapes ran but yielded 0 volume entities (solid could not be "
                       "reconstructed -- likely a self-intersecting B-rep)")
    return True, ""


def _apply_defeature() -> tuple[bool, str]:
    """Merge literally duplicate OCC entities (`removeAllDuplicates`), a
    cheap clean-up for CAD exports. Returns (ok, note); not ok if the call
    raises or the solid is lost."""
    import gmsh

    n_vols_before = len(gmsh.model.getEntities(3))
    try:
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
    except Exception as e:
        return False, f"defeature (removeAllDuplicates) raised: {e}"
    n_vols_after = len(gmsh.model.getEntities(3))
    if n_vols_before > 0 and n_vols_after == 0:
        return False, "defeature (removeAllDuplicates) yielded 0 volume entities (solid lost)"
    return True, ""


def heal_and_mesh(step_path: str, out_msh: str, target_nodes: int = DEFAULT_TARGET_NODES,
                  defeature: bool = False) -> dict:
    """Mesh the largest solid of `step_path` near `target_nodes` nodes.

    Pass 1 meshes the geometry as imported (healing can destroy a solid
    that would have meshed fine, so it is not applied by default). Only if
    pass 1 fails outright does pass 2, in a fresh Gmsh session, heal the
    geometry and retry the same density search. `defeature=True` runs
    `_apply_defeature` before meshing in both passes.

    Returns {n_solids, volume_mm3, bbox_diag_mm, n_nodes, ok, healed, notes,
    attempts}; `notes` collects human-readable diagnostics.
    """
    import gmsh

    notes: list[str] = []

    # ---- Pass 1: unhealed ----
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part")
        n_solids, volume_mm3, err = _import_and_reduce(step_path)
        if err is not None:
            notes.append(err)
            return dict(n_solids=0, volume_mm3=0.0, bbox_diag_mm=0.0, n_nodes=0,
                        ok=False, healed=False, notes="; ".join(notes), attempts=0)
        if n_solids > 1:
            notes.append(f"multi-body: kept largest of {n_solids} solids (vol={volume_mm3:.1f} mm3)")

        if defeature:
            defeature_ok, defeature_note = _apply_defeature()
            if defeature_note:
                notes.append(defeature_note)
            if not defeature_ok:
                return dict(n_solids=n_solids, volume_mm3=volume_mm3, bbox_diag_mm=0.0,
                            n_nodes=0, ok=False, healed=False, notes="; ".join(notes),
                            attempts=0)

        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        bbox_diag_mm = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
        base_size = _first_guess_elem_size(bbox_diag_mm, target_nodes)

        fail_notes: list[str] = []

        def build_fn(msh_path, elem_size):
            size = elem_size if elem_size is not None else base_size
            ok_, n_nodes_, note_ = _tag_and_mesh(size, msh_path)
            if not ok_:
                fail_notes.append(note_)
            return (n_nodes_ if ok_ else 0), size

        n_nodes, _used_size, attempts = _search_target_nodes(build_fn, out_msh, target_nodes)
        if n_nodes > 0:
            return dict(n_solids=n_solids, volume_mm3=volume_mm3, bbox_diag_mm=bbox_diag_mm,
                        n_nodes=n_nodes, ok=True, healed=False, notes="; ".join(notes),
                        attempts=attempts)
        notes.append(f"unhealed mesh attempt failed: {fail_notes[-1] if fail_notes else 'unknown'}")
    finally:
        gmsh.finalize()

    # ---- Pass 2: heal + density-searched retry (fresh session) ----
    healed = False
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part_healed")
        n_solids2, volume_mm32, err2 = _import_and_reduce(step_path)
        if err2 is not None:
            notes.append(f"pass-2 {err2}")
            return dict(n_solids=n_solids, volume_mm3=volume_mm3, bbox_diag_mm=bbox_diag_mm,
                        n_nodes=0, ok=False, healed=False, notes="; ".join(notes),
                        attempts=attempts)

        if defeature:
            defeature_ok2, defeature_note2 = _apply_defeature()
            if defeature_note2:
                notes.append(defeature_note2)
            if not defeature_ok2:
                return dict(n_solids=n_solids, volume_mm3=volume_mm3, bbox_diag_mm=bbox_diag_mm,
                            n_nodes=0, ok=False, healed=False, notes="; ".join(notes),
                            attempts=attempts)

        heal_ok, heal_note = _apply_heal()
        if heal_note:
            notes.append(heal_note)
        if not heal_ok:
            notes.append("meshing failed after heal attempt")
            return dict(n_solids=n_solids, volume_mm3=volume_mm3, bbox_diag_mm=bbox_diag_mm,
                        n_nodes=0, ok=False, healed=False, notes="; ".join(notes),
                        attempts=attempts)
        healed = True

        fail_notes2: list[str] = []

        def build_fn2(msh_path, elem_size):
            size = elem_size if elem_size is not None else base_size
            ok_, n_nodes_, note_ = _tag_and_mesh(size, msh_path)
            if not ok_:
                fail_notes2.append(note_)
            return (n_nodes_ if ok_ else 0), size

        n_nodes2, _used_size2, attempts2 = _search_target_nodes(build_fn2, out_msh, target_nodes)
        if n_nodes2 > 0:
            notes.append("meshed after heal")
            return dict(n_solids=n_solids, volume_mm3=volume_mm3, bbox_diag_mm=bbox_diag_mm,
                        n_nodes=n_nodes2, ok=True, healed=True, notes="; ".join(notes),
                        attempts=attempts + attempts2)
        notes.append(f"post-heal mesh attempt failed: {fail_notes2[-1] if fail_notes2 else 'unknown'}")
        notes.append("meshing failed after heal attempt")
        return dict(n_solids=n_solids, volume_mm3=volume_mm3, bbox_diag_mm=bbox_diag_mm,
                    n_nodes=0, ok=False, healed=healed, notes="; ".join(notes),
                    attempts=attempts + attempts2)
    finally:
        gmsh.finalize()


def auto_elem_size(bbox_diag: float) -> float:
    """Default fixed element size for `mesh_step`: bounding-box diagonal / 60."""
    return bbox_diag / 60.0


def mesh_step(step_path: str, out_msh: str, elem_size: float | None = None) -> dict:
    """Mesh the whole STEP model at one fixed element size (no healing, no
    density search). Every surface is tagged "Cooling"."""
    import gmsh

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
