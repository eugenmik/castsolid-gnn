"""Dataset-D part preparation and density-targeted STEP meshing.

Two failure modes, both owner-decided (DECISIONS-2026-07-22 §11):
  - Multi-body (Koleso_otl 4 solids, Track 3, TP 2): these are castings
    accidentally split into multiple solids by the source CAD. Fix: keep
    ONLY the largest solid by volume (`largest_solid`).
  - Mesh-fail "Invalid boundary mesh (overlapping facets)" (20508M, SSG0750,
    TP, DO678078): attempt an OCC heal (`gmsh.model.occ.healShapes()`)
    before meshing.
DO678078 is additionally OOD (bbox diag 2663.5mm > envelope 2177.7mm,
`data/features_v2/ood_ranges.json`). In practice its heal attempt does NOT
recover a meshable solid (same genuine self-intersecting-B-rep class as
20508M/SSG0750/TP, confirmed 2026-07-23) -- it is reported FAILED for
meshing, same as the other 3 mesh-fail parts. It still gets a cleaned
(largest-solid) `.step` export and is tagged OOD in the report (useful if a
future CAD repair fixes the mesh, and as a record that this geometry sits
outside the training envelope), but it is NOT currently a usable training
geometry -- the "healed + KEPT (OOD)" framing in the original task brief
was aspirational and did not hold up against the real STEP file. 5D49.78.01
is not broken, just slow/heavy -- it gets a longer per-part time-box (see
`SLOW_PART_TIMEOUTS_S` below) rather than any geometry fix.

Mesh density: FT1 production density (DECISIONS-2026-07-22 §9, ~7.5-10k
nodes, target_nodes=8500 default). A single fixed `bbox_diag/28` element
size (the original approach) misses this badly across parts of very
different shapes -- observed 1855..18343 nodes in a real batch run at that
fixed size. Instead, `_search_target_nodes` reuses the validated iterative
density search from `tools.doe.mesh_study.search_elem_size` (same
node_count ~ elem_size^-3 rescaling, imported not reimplemented), seeded
with `bbox_diag/28` as the first guess, refining up to 2 more times to
converge near `target_nodes`; a node-count early exit skips the refinement
entirely when the first guess already lands within 35% of the target (the
common case). If the (density-searched) unhealed attempt still fails
outright (not a density problem -- a topology error), heal + one more
density-searched attempt is tried before giving up.

Every part is time-boxed via a `--single`-mode SUBPROCESS (not an in-process
signal alarm: `gmsh.model.mesh.generate(3)` is a blocking C call that does
not yield to Python signal handlers, so `signal.alarm` cannot reliably
interrupt a hung mesh) -- `nice -n 10 <python> -u -m tools.doe.prep_parts
--single ...` under `subprocess.run(..., timeout=...)`. Default time-box is
300s; 5D49.78.01 (known slow/heavy) gets a longer one.

CLI:
    python -m tools.doe.prep_parts --parts-dir pc_parts \\
        --out-dir doe_out/parts_prepped
writes, per part, a cleaned single-solid `<stem>_cleaned.step` (best-effort:
largest-solid + healed, written regardless of whether meshing itself
succeeds) and, when meshing succeeds, `<stem>.msh` (gmsh 2.2, all-surface
"Cooling" + "Volume"); plus a summary `prep_report.md`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tools.doe.mesh_study import search_elem_size

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TARGET_NODES = 8500

# Bump on ANY change that affects mesh output (heal/mesh logic, gmsh options,
# density search, etc.) -- the mesh cache (src/mesh_cache.py) keys on this.
# `fast` (HXT/multithreading/curvature-off, see `_apply_fast_options`) is
# OFF by default and thus does not change today's output, but it DOES
# participate in mesh output whenever a caller passes fast=True -- if
# `fast`'s DEFAULT is ever flipped to True, this version must bump (a
# cache keyed only on this version, not on the `fast` argument itself,
# would otherwise silently reuse a pre-flip mesh for a post-flip call).
# v1 -> v2 (Task 4, 2026-07-29): surface-area density-search seed
# (`_first_guess_elem_size`/`_elem_size_from_area`/`K_SURF`) replaced the
# bbox_diag/28 first-attempt guess UNCONDITIONALLY (default-on, not
# flag-gated like `fast`) -- this changes the first elem-size tried (and
# thus can change the converged mesh) for every part, so a cache entry
# written under v1 must not be served after this change.
# `defeature` (Task 5, 2026-07-29): a best-effort geometry-simplification
# pass (`_apply_defeature` -- see its docstring for the full virtual-
# topology investigation), gated OFF by default and flag-gated like `fast`
# -- does NOT bump this version, for the same reason `fast` doesn't: it
# only participates in mesh output when a caller explicitly passes
# `defeature=True`. `src/mesh_cache.py::get_or_mesh`'s auto-trigger policy
# decides `defeature` deterministically from the input alone (file size,
# then a same-input retry-on-failure), so a given STEP file always
# resolves to the same `defeature` choice across runs -- the cache key
# (which does not include `defeature` itself) stays consistent per input.
# v2 -> v3 (Part B Task 4, 2026-07-29): v3 = multi-body bodies path (Part B)
# -- `heal_and_mesh`'s `keep_solid_id` (user-designated casting solid,
# replacing the largest-solid default) and `src/mesh_cache.py::get_or_mesh`'s
# `bodies_key`/`post_mesh` (tagged-mesh caching) both participate in mesh
# output whenever `predict_casting(bodies=...)` is used -- bump so a
# pre-Part-B cache entry (meshed without any of this) is never served to a
# bodies-aware call.
MESH_PIPELINE_VERSION = 3
DEFAULT_TIMEOUT_S = 300.0
# 5D49.78.01 was killed at 180s in the inventory scan's coarser (diag/25)
# attempt; at FT1's finer diag/28 density it needs more room, not a geometry
# fix -- give it a longer per-part time-box instead of the 300s default.
SLOW_PART_TIMEOUTS_S = {"5D49.78.01": 1800.0}

DEFAULT_OOD_RANGES_PATH = "data/features_v2/ood_ranges.json"

# Fixed physical-group tags for the volume/cooling groups (see `_tag_and_mesh`
# docstring for why these must be pinned rather than auto-assigned).
_VOLUME_PHYS_TAG = 1
_COOLING_PHYS_TAG = 2


# --------------------------------------------------------------------------
# Largest-solid extraction
# --------------------------------------------------------------------------
def _reduce_to_largest_solid():
    """Assumes an OCC model is already imported + synchronized in the
    current gmsh session. If >1 solid (dim=3 OCC entity) exists, keeps only
    the max-volume one (removes the rest, recursively, from the OCC CAD
    representation) and re-synchronizes. If exactly 1 (or 0) solids exist,
    it's a no-op. Returns (n_solids_in, kept_tag, kept_volume_mm3)."""
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


def largest_solid(step_path: str, out_step: str) -> tuple[int, float]:
    """Import `step_path` (STEP); if it contains >1 solid, keep only the
    max-volume one; export a single-solid STEP to `out_step` either way
    (a 1-solid input is copied through via a plain gmsh re-export).

    Returns (n_solids_in, kept_volume_mm3)."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("largest_solid")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        n_in, _kept_tag, kept_vol = _reduce_to_largest_solid()
        gmsh.write(out_step)
        return n_in, kept_vol
    finally:
        gmsh.finalize()


# --------------------------------------------------------------------------
# Heal + mesh
# --------------------------------------------------------------------------
def _elem_size(bbox_diag_mm: float) -> float:
    """Starting guess for the density search: bbox diagonal / 28 (roughly
    the FT1 ~7.5-10k node density on a "typical" casting, DECISIONS-2026-07-22
    §9) -- `_search_target_nodes` then refines from here towards the actual
    `target_nodes`, since this fixed heuristic alone was found to miss badly
    across parts of very different shapes (see module docstring). Used as
    the FALLBACK first guess (see `_first_guess_elem_size`) when the OCC
    surface-area query is unavailable."""
    return bbox_diag_mm / 28.0


# Calibrated 2026-07-29 (mesh-accel Task 4) against BUKSA.STEP + H17.STEP
# (real dataset-D parts, target_nodes=8500) + 2 synthetic box fixtures
# (30x20x10mm and 40x40x40mm, target_nodes=500) over K_SURF in
# {1.0, 1.5, 2.0, 2.5, 3.0} -- see
# The calibration table is retained in the private experiment record.
# K_SURF=2.5 was the outright winner: 4/4 fixtures land within
# +-35% of target_nodes on the FIRST attempt (K=2.0 and K=3.0 each get 3/4;
# K<=1.5 get <=1/4).
K_SURF = 2.5


def _elem_size_from_area(area_mm2: float, target_nodes: int) -> float:
    """Starting guess for the density search, seeded from the model's total
    OCC surface area (mm^2) rather than the fixed bbox_diag/28 heuristic:
    `sqrt(K_SURF * area_mm2 / target_nodes)`. Surface area tracks the
    number of surface elements (and, via the tet-to-surface-element ratio
    for a roughly-uniform mesh, the total node count) far more directly
    than a bounding-box diagonal, which is blind to shape (a thin plate and
    a compact block with the same diagonal need very different element
    sizes for the same target_nodes). `K_SURF` is calibrated in this task
    (see module-level comment above `K_SURF`)."""
    return float(np.sqrt(K_SURF * area_mm2 / target_nodes))


def _total_surface_area_mm2() -> float:
    """Assumes an already-imported + synchronized OCC model in the current
    gmsh session. Returns the total area (mm^2) of all dim=2 (surface)
    entities via `gmsh.model.occ.getMass(2, tag)` -- confirmed to work
    directly on an imported STEP model (no re-synchronize needed beyond the
    caller's own `occ.synchronize()`). Raises whatever gmsh raises on
    failure; callers must catch (see `_first_guess_elem_size`)."""
    import gmsh

    surfs = [t for (d, t) in gmsh.model.occ.getEntities(2)]
    return float(sum(gmsh.model.occ.getMass(2, t) for t in surfs))


def _first_guess_elem_size(bbox_diag_mm: float, target_nodes: int) -> float:
    """First-attempt element-size guess for the density search: prefers the
    OCC-surface-area-seeded guess (`_elem_size_from_area`, using the
    CURRENTLY-imported/synchronized gmsh model), since the fixed
    bbox_diag/28 heuristic alone was found to miss badly on non-typical
    (thin-walled, elongated, etc.) shapes (see module docstring). Falls
    back to `_elem_size(bbox_diag_mm)` -- loudly logged -- if the area query
    raises or returns a non-positive area."""
    try:
        area_mm2 = _total_surface_area_mm2()
    except Exception as e:
        print(f"[prep_parts] WARNING: OCC surface-area query failed ({e}); "
              f"falling back to bbox_diag/28 seed", flush=True)
        return _elem_size(bbox_diag_mm)
    if area_mm2 <= 0:
        print(f"[prep_parts] WARNING: OCC surface-area query returned "
              f"{area_mm2!r} (<=0); falling back to bbox_diag/28 seed", flush=True)
        return _elem_size(bbox_diag_mm)
    return _elem_size_from_area(area_mm2, target_nodes)


def _search_target_nodes(build_fn, out_msh: str, target_nodes: int,
                          tol: float = 0.35, max_iter: int = 3) -> tuple[int, float, int]:
    """Node-count-targeted density search: a thin wrapper around the
    validated `tools.doe.mesh_study.search_elem_size` iterative refinement
    (same node_count ~ elem_size^-3 rescaling law, imported and reused
    as-is, not reimplemented) that adds a node-count early exit.

    `build_fn(out_msh, elem_size) -> (n_nodes, elem_size_used)` -- same
    contract as `search_elem_size` expects (`elem_size=None` means "pick a
    default"); `n_nodes` is 0 on outright mesh failure (not a density
    problem).

    If the FIRST attempt (the surface-area-seeded guess, see
    `_first_guess_elem_size`) is already within `tol` (relative) of
    `target_nodes`, or fails outright, return immediately -- no point in
    further (wasted) refinement iterations in either case. Otherwise,
    delegate to `search_elem_size` for up to `max_iter` total attempts
    (this redoes the first attempt once more as its own internal first
    step -- one redundant mesh generation in this less-common branch,
    accepted for simplicity over threading extra state through the reused
    function).

    Returns `(n_nodes, elem_size_used, attempts)` -- `attempts` is the
    total number of `build_fn` invocations across the whole search (1 when
    the first guess already lands in-band; useful both for this module's
    own tests and for a future bench harness)."""
    calls = {"n": 0}

    def counted_build_fn(msh_path, elem_size):
        calls["n"] += 1
        return build_fn(msh_path, elem_size)

    n0, es0 = counted_build_fn(out_msh, None)
    if n0 <= 0 or abs(n0 - target_nodes) <= tol * target_nodes:
        return n0, es0, calls["n"]
    n, es = search_elem_size(counted_build_fn, out_msh, target_nodes, max_iter=max_iter)
    return n, es, calls["n"]


def _tag_and_mesh(size: float, out_msh: str) -> tuple[bool, int, str]:
    """Assumes an already-imported (and possibly reduced/healed) OCC model
    is synchronized in the current gmsh session. (Re)tags all volume/surface
    entities "Volume"/"Cooling" (fresh physical groups -- entity tags can
    change after healing) and attempts `gmsh.model.mesh.generate(3)` at
    element size `size`. Returns (ok, n_nodes, note).

    Pins the physical-group tag numbers to `_VOLUME_PHYS_TAG`/
    `_COOLING_PHYS_TAG` instead of letting gmsh auto-assign them. Root cause
    (found 2026-07-28 debugging H17.STEP producing an untagged mesh): this
    function can run MULTIPLE times within one gmsh session -- the
    density-search retries in `_search_target_nodes`/`search_elem_size` call
    it again with a different `size` when the first attempt misses the node
    target. `removePhysicalGroups()` frees the *tag number* for reuse but
    does NOT clear the *name registration* for that string -- so on the 2nd+
    call in a session, `gmsh.model.addPhysicalGroup(dim, ents)` auto-assigns
    a NEW tag, but `setPhysicalName(dim, new_tag, "Volume"/"Cooling")`
    silently no-ops (gmsh will not bind a name string that's already
    registered to a different, even now-removed, tag). The result: the
    LAST-written `.msh` has its actual `$Elements` referencing an unnamed
    physical group, while `$PhysicalNames` still lists the stale first-round
    (tag -> name) binding -- `parse_msh` matches elements by name and finds
    none, i.e. `tris == {}`. Confirmed with a minimal gmsh repro (no meshing
    involved) that reusing the same tag number every round keeps the name
    bound correctly. BUKSA never hit this because its first density guess
    already converged (one call, no retries); H17's did not (multiple
    calls), which is why the two parts diverged.
    """
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
    # Fail loudly rather than silently writing an untagged mesh: this is a
    # defense-in-depth check, not a substitute for the tag-pinning above --
    # it catches ANY other way the name binding could fail to register,
    # since a downstream consumer (`parse_msh`) trusts these two names
    # unconditionally and would otherwise silently see empty groups.
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
    """Assumes a fresh gmsh session with an imported + largest-solid-reduced
    OCC model synchronized. Runs `gmsh.model.occ.healShapes()` and checks
    whether it actually yielded >=1 volume entity -- a real bug found during
    development (see module docstring / task report) is that `healShapes()`
    can run WITHOUT raising and still silently reduce the model to 0 volumes
    (all surfaces, no solid), which must be treated as an ineffective heal,
    not a successful one. Returns (effective, note); `note` is "" when
    effective."""
    import gmsh

    try:
        gmsh.model.occ.healShapes()
        gmsh.model.occ.synchronize()
    except Exception as e:
        return False, f"healShapes raised: {e}"
    n_vols = len(gmsh.model.getEntities(3))
    if n_vols == 0:
        return False, ("healShapes ran but yielded 0 volume entities (solid could not be "
                        "reconstructed -- likely a genuine self-intersecting B-rep, not a "
                        "density artifact)")
    return True, ""


def _apply_defeature() -> tuple[bool, str]:
    """Assumes a fresh gmsh session with an imported (and possibly largest-
    solid-reduced) OCC model synchronized. Best-effort geometry-
    simplification pass ahead of meshing, gated behind `defeature=True`
    (see `heal_and_mesh`). Investigated for this task (mesh-accel Task 5,
    2026-07-29; full probe log in
    the meshing-acceleration experiment record)
    against real dataset-D parts (4474_443.STEP: 3009 faces / 1 solid /
    diag 1283mm; TP.STEP: 981 faces after largest-solid reduction):

      - `gmsh.model.occ.removeAllDuplicates()` -- APPLIED. Fast (<2s even
        on the 3009-face part) and, in every part probed, never reduced
        the volume count to 0. Its actual effect is narrow (it only merges
        LITERALLY duplicate/coincident OCC entities -- e.g. a CAD-export
        artifact) -- it was a no-op on face count on every real part
        probed here (no exact duplicates present), but it is gmsh's own
        recommended call whenever "the OCC model may contain redundant
        geometry" and costs nothing to run unconditionally.
      - `gmsh.model.occ.healShapes(tolerance=<elem-size-scale>, ...)` as a
        "merge faces smaller than the mesh scale" lever -- TRIED AND
        REJECTED. Even the library's DEFAULT tolerance (1e-8, i.e. no
        intentional defeaturing at all) silently reduced 4474_443.STEP
        (a clean, single-solid part) to 0 volume entities after a 59s
        call; scaling the tolerance up to merge small faces (tried 5mm on
        TP.STEP, whose bbox-diag/28 scale is ~30mm) instead raised
        "Could not fix wire in surface 54" outright. This is the exact
        class of silent-destruction failure `_apply_heal` already guards
        against for the pass-2 heal -- deliberately enlarging the
        tolerance to defeature makes it MORE likely to trigger, not less.
        Not a safe unconditional lever; not applied.
      - `gmsh.option.setNumber("Geometry.Tolerance", ...)` ahead of
        `removeAllDuplicates()` -- TRIED AND REJECTED. Swept up to 10mm
        (already far above a typical small-part mesh element size) on
        4474_443.STEP with zero change in face count -- in this gmsh
        version this option does not make `removeAllDuplicates` merge
        near-but-not-exactly-coincident faces.
      - `gmsh.model.mesh.setCompound(dim, tags)` -- NOT APPLIED. Per the
        gmsh API this is a manual meshing CONSTRAINT (mesh a caller-
        supplied group of face tags as one reparametrized patch), not an
        automatic small-face detector. Using it for defeaturing would
        need a from-scratch face-area + adjacency analysis (per-face area
        via `occ.getMass(2, tag)`, threshold against the target element
        scale, then group adjacent small faces via shared-edge topology
        queries) -- real engineering beyond a best-effort pass; flagged as
        a follow-up, not implemented.
      - `gmsh.model.mesh.classifySurfaces(...)` -- NOT APPLICABLE. Per its
        own docs it operates on an ALREADY-DISCRETIZED surface mesh and
        (via `exportDiscrete=True`, the only way to get usable output)
        REPLACES the OCC CAD entities with discrete ones -- this is gmsh's
        STL/reverse-engineering workflow, not a pre-mesh OCC B-rep
        operation, and would break `_tag_and_mesh`'s OCC-entity-based
        physical-group tagging downstream. Confirmed inapplicable from the
        API contract alone (angle-based reclassification assumes a
        discrete/triangulated model); not probed further, per the task
        brief's own suspicion.

    Net effect: the applied pass is `removeAllDuplicates()` only, with a
    defensive "did we just lose the solid" check (never observed to
    trigger during probing, but cheap insurance -- mirrors `_apply_heal`'s
    `(ok, note)` return shape and MUST be branched on by the caller the
    same way: a `removeAllDuplicates()` call that raises, or that zeroes
    out the volume count, leaves the gmsh session with 0 solids -- letting
    the caller fall through to `gmsh.model.getBoundingBox(-1, -1)` on an
    empty model raises an uncaught `Exception: Empty bounding box` instead
    of a clean `ok=False` (found in review, 2026-07-29; see task-5-report.md
    "Review fix"). Returns `(ok, note)`; `note` is "" when `ok` (dedup ran
    and the solid survived), and a human-readable diagnostic when not."""
    import gmsh

    n_vols_before = len(gmsh.model.getEntities(3))
    try:
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
    except Exception as e:
        return False, f"defeature (removeAllDuplicates) raised: {e}"
    n_vols_after = len(gmsh.model.getEntities(3))
    if n_vols_before > 0 and n_vols_after == 0:
        return False, ("defeature (removeAllDuplicates) yielded 0 volume entities "
                        "(solid lost) -- not observed during development, but "
                        "reported honestly rather than silently proceeding")
    return True, ""


def _reduce_to_solid_id(keep_solid_id: int) -> tuple[int, float]:
    """Assumes an OCC model is already imported + synchronized in the current
    gmsh session (mirrors `_reduce_to_largest_solid`, but keeps a CALLER-
    designated OCC tag instead of the max-volume one -- used by Part B's
    multi-body pipeline, where the user has already assigned a specific
    solid the "casting" role and it need not be the largest body in the
    assembly). Deletes every other dim=3 entity (recursively, from the OCC
    CAD representation) and re-synchronizes; a no-op if `keep_solid_id` is
    already the only solid. Raises `ValueError` if `keep_solid_id` is not
    among the imported model's 3-D entities -- callers (here,
    `_import_and_reduce`) turn that into a loud error string, never a
    silent fallback to some other solid. Returns (n_solids_in, kept_volume_mm3)."""
    import gmsh

    vols = [t for (d, t) in gmsh.model.occ.getEntities(3)]
    n_in = len(vols)
    if n_in == 0:
        raise ValueError("no 3D solids found in STEP import")
    if keep_solid_id not in vols:
        raise ValueError(
            f"keep_solid_id={keep_solid_id} not found among 3D solids {sorted(vols)}")
    kept_vol = float(gmsh.model.occ.getMass(3, keep_solid_id))
    if n_in > 1:
        drop = [(3, t) for t in vols if t != keep_solid_id]
        gmsh.model.occ.remove(drop, recursive=True)
        gmsh.model.occ.synchronize()
    return n_in, kept_vol


def _import_and_reduce(step_path: str, keep_solid_id: int | None = None) -> tuple[int, float, str | None]:
    """Assumes a fresh gmsh session (already initialized). Imports
    `step_path` and reduces multi-body parts to a single solid: the largest
    by volume (`keep_solid_id=None`, existing/default behavior, byte-
    identical) or the caller-designated OCC tag (`keep_solid_id=<int>` --
    Part B's multi-body pipeline, where the "casting" solid is whichever one
    the user assigned that role to, not necessarily the largest). A missing
    `keep_solid_id` is reported the same way as any other reduction failure
    -- a loud error string, not an exception -- so `heal_and_mesh` turns it
    into `ok=False` with the reason in `notes`, never a silent fallback.
    Returns (n_solids_in, kept_volume_mm3, error_or_None)."""
    import gmsh

    try:
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
    except Exception as e:
        return 0, 0.0, f"import failed: {e}"

    if keep_solid_id is None:
        try:
            n_solids, _kept_tag, volume_mm3 = _reduce_to_largest_solid()
        except Exception as e:
            return 0, 0.0, f"largest-solid reduction failed: {e}"
        return n_solids, volume_mm3, None

    try:
        n_solids, volume_mm3 = _reduce_to_solid_id(keep_solid_id)
    except Exception as e:
        return 0, 0.0, f"keep_solid_id reduction failed: {e}"
    return n_solids, volume_mm3, None


def _apply_fast_options() -> None:
    """Assumes an already-initialized gmsh session. Sets the flag-gated
    "fast" meshing option set (mesh-accel Task 3, default OFF -- see
    `heal_and_mesh`'s `fast` kwarg): `Mesh.Algorithm3D=10` (HXT, gmsh's
    multithreaded Delaunay-refinement tet algorithm, vs. the single-
    threaded classical Delaunay default), `General.NumThreads` capped at
    min(8, os.cpu_count()) (diminishing returns/oversubscription risk past
    8 on this box -- see the CPU-oversubscription rule), and
    `Mesh.MeshSizeFromCurvature=0` (curvature-adaptive refinement is
    unneeded for this project's uniform analytic-density workflow and only
    costs solve time). Does NOT touch `Mesh.MeshSizeMin`/`Mesh.MeshSizeMax`
    -- those are already set per-call by `_tag_and_mesh`, right before
    `gmsh.model.mesh.generate(3)`."""
    import gmsh

    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("General.NumThreads", min(8, os.cpu_count() or 1))
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


def heal_and_mesh(step_path: str, out_msh: str, target_nodes: int = DEFAULT_TARGET_NODES,
                   out_step: str | None = None, fast: bool = False,
                   defeature: bool = False, keep_solid_id: int | None = None) -> dict:
    """Import `step_path`, reduce multi-body parts to their largest solid,
    tag all surfaces "Cooling" / the volume "Volume", and mesh at the FT1
    density via `_search_target_nodes` (node-count-targeted, seeded from
    `_first_guess_elem_size`'s OCC-surface-area guess, falling back to
    `_elem_size`'s bbox-diag/28 guess if the area query is unavailable --
    see module docstring). Writes gmsh 2.2 `out_msh` on success.

    Two passes, each a FRESH gmsh session (an OCC heal is not reliably
    undoable within one session):
      1. Unhealed, density-searched towards `target_nodes`. This is tried
         FIRST and is the only pass run for parts that already mesh fine --
         a real finding during development was that
         `gmsh.model.occ.healShapes()` can silently reduce a perfectly
         meshable solid to 0 volumes (all surfaces, no solid) even though it
         raises no exception, so it is NOT applied unconditionally to every
         part (that would regress the 9 already-clean parts). `out_step`, if
         given, is exported here -- the single-solid (but unhealed)
         geometry -- so a usable cleaned STEP exists regardless of what
         pass 2 does.
      2. Only if pass 1's mesh attempt fails outright (a topology error, not
         a density problem -- `_search_target_nodes` already handles density):
         heal (`_apply_heal`, default tolerance) and, only if the heal
         actually yielded a volume, retry the SAME density search. If
         healing runs but yields 0 volume entities (observed on
         20508M/SSG0750/TP/DO678078 in this repo's pc_parts -- a genuine
         self-intersecting B-rep, not a mesh-density artifact: confirmed by
         testing heal tolerances from 1e-6 to 0.7, none recover a solid, and
         tolerance=1.0 raises "Could not fix wire in surface ..." -- gmsh's
         healer simply cannot repair it), that is reported honestly as a
         failed heal, not silently treated as success -- meshing is not
         even attempted in that case.

    Returns a dict: {n_solids, volume_mm3, bbox_diag_mm, n_nodes, ok,
    healed, notes, attempts}. `ok` is True iff meshing produced a non-empty
    tetrahedral mesh (in either pass). `healed` is True iff pass 2 ran AND
    its heal yielded at least one volume entity (i.e. the heal was
    actually effective, not just exception-free). `notes` is a single
    human-readable string (diagnostics joined by "; ") for the prep report.
    `attempts` is the total number of density-search `build_fn` calls
    across whichever pass(es) actually ran a search (1 = the first,
    surface-area-seeded guess already landed within tolerance; 0 if no
    density search was ever attempted, e.g. import failed).

    `fast`, if True, applies `_apply_fast_options()` (HXT + multithreading
    + curvature-refinement off) in BOTH passes' gmsh sessions, right after
    each `gmsh.initialize()`. Default False -> zero behavior change (no
    option calls at all, gmsh's own defaults apply, current single-
    threaded classical-Delaunay behavior is unchanged) -- gated OFF until a
    later benchmark task flips the default (see `MESH_PIPELINE_VERSION`).

    `defeature`, if True, applies `_apply_defeature()` (best-effort
    geometry simplification -- see its docstring for what was investigated
    and what actually ships) in BOTH passes, right after that pass's
    import/largest-solid-reduction step. Default False -> zero behavior
    change. This is opt-in and does not bump `MESH_PIPELINE_VERSION` (see
    the comment above it); `src/mesh_cache.py::get_or_mesh` decides when to
    pass `defeature=True` via its own size/failure auto-trigger policy, not
    this function. The Volume/Cooling pinned-tag + loud-guard machinery in
    `_tag_and_mesh` applies completely unchanged after defeaturing -- it is
    the safety net that a defeatured model still gets tagged correctly (or
    fails loudly if it doesn't), not something `defeature` needs to
    duplicate.

    `keep_solid_id`, if given (Part B multi-body pipeline: the user has
    assigned solid roles and the "casting" is whichever solid got that
    role, not necessarily the largest), is threaded into `_import_and_reduce`
    on BOTH passes in place of the largest-solid default -- see its
    docstring. Default `None` -> existing largest-solid behavior, byte-
    identical to before this parameter existed. An invalid/missing tag comes
    back as an `_import_and_reduce` error string, which this function
    already turns into a loud `ok=False` (same code path as an import
    failure) -- no new branch needed.
    """
    import gmsh

    notes: list[str] = []

    # ---- Pass 1: unhealed ----
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        if fast:
            _apply_fast_options()
        gmsh.model.add("part")
        n_solids, volume_mm3, err = _import_and_reduce(step_path, keep_solid_id=keep_solid_id)
        if err is not None:
            notes.append(err)
            return dict(n_solids=0, volume_mm3=0.0, bbox_diag_mm=0.0, n_nodes=0,
                        ok=False, healed=False, notes="; ".join(notes), attempts=0)
        if n_solids > 1:
            if keep_solid_id is None:
                notes.append(f"multi-body: kept largest of {n_solids} solids (vol={volume_mm3:.1f} mm3)")
            else:
                notes.append(f"multi-body: kept solid_id={keep_solid_id} of {n_solids} solids "
                              f"(vol={volume_mm3:.1f} mm3)")

        if defeature:
            defeature_ok, defeature_note = _apply_defeature()
            if defeature_note:
                notes.append(defeature_note)
            if not defeature_ok:
                return dict(n_solids=n_solids, volume_mm3=volume_mm3, bbox_diag_mm=0.0,
                            n_nodes=0, ok=False, healed=False, notes="; ".join(notes),
                            attempts=0)

        if out_step is not None:
            try:
                gmsh.write(out_step)
            except Exception as e:
                notes.append(f"cleaned-STEP export failed: {e}")

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
        if fast:
            _apply_fast_options()
        gmsh.model.add("part_healed")
        n_solids2, volume_mm32, err2 = _import_and_reduce(step_path, keep_solid_id=keep_solid_id)
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


# --------------------------------------------------------------------------
# OOD check
# --------------------------------------------------------------------------
def load_ood_ranges(path: str = DEFAULT_OOD_RANGES_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def is_ood(volume_mm3: float, bbox_diag_mm: float, ranges: dict, tol: float = 1.05) -> bool:
    """True if either the kept volume or the bbox diagonal exceeds the
    training envelope (`data/features_v2/ood_ranges.json`) by more than
    `tol` (default 1.05 = 5% margin) -- same convention as `src.ood.ood_flags`,
    which applies the identical 5% tolerance before flagging, so this
    parts-prep OOD check and the real-casting-inference OOD check agree."""
    vol_max = ranges.get("volume_mm3_max", float("inf"))
    diag_max = ranges.get("bbox_diag_max", float("inf"))
    return bool(volume_mm3 > vol_max * tol or bbox_diag_mm > diag_max * tol)


# --------------------------------------------------------------------------
# Per-part subprocess isolation (real 300s-class time-box)
# --------------------------------------------------------------------------
def _timeout_for(stem: str, default_timeout_s: float) -> float:
    return SLOW_PART_TIMEOUTS_S.get(stem, default_timeout_s)


def _run_one_bounded(step_path: str, out_msh: str, out_step: str, target_nodes: int,
                      timeout_s: float) -> dict:
    """Run `heal_and_mesh` for ONE part in a fresh subprocess (`nice -n 10`),
    bounded by `timeout_s`. A hang or crash in this one part never blocks
    the rest of the batch -- it's reported as a failed/timed-out row."""
    cmd = [
        "nice", "-n", "10", sys.executable, "-u", "-m", "tools.doe.prep_parts",
        "--single", step_path, "--out-msh", out_msh, "--out-step", out_step,
        "--target-nodes", str(target_nodes),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout_s, cwd=str(REPO_ROOT))
    except subprocess.TimeoutExpired:
        return dict(n_solids=0, volume_mm3=0.0, bbox_diag_mm=0.0, n_nodes=0,
                    ok=False, healed=False, notes=f"TIMEOUT after {timeout_s:.0f}s")

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        return dict(n_solids=0, volume_mm3=0.0, bbox_diag_mm=0.0, n_nodes=0,
                    ok=False, healed=False,
                    notes=f"worker exit {proc.returncode}: {' | '.join(tail)}")

    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return dict(n_solids=0, volume_mm3=0.0, bbox_diag_mm=0.0, n_nodes=0,
                    ok=False, healed=False, notes="worker produced no output")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return dict(n_solids=0, volume_mm3=0.0, bbox_diag_mm=0.0, n_nodes=0,
                    ok=False, healed=False, notes=f"worker output not JSON: {lines[-1][:200]}")


# --------------------------------------------------------------------------
# Batch CLI
# --------------------------------------------------------------------------
_STEP_SUFFIXES = {".step", ".stp"}


def run_all(parts_dir: str, out_dir: str, target_nodes: int = DEFAULT_TARGET_NODES,
            timeout_s: float = DEFAULT_TIMEOUT_S,
            ood_ranges_path: str = DEFAULT_OOD_RANGES_PATH) -> list[dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ranges = load_ood_ranges(ood_ranges_path) if Path(ood_ranges_path).exists() else {}
    if not ranges:
        print(f"[prep_parts] WARNING: OOD ranges not found at {ood_ranges_path}; "
              f"OOD column will read 'unknown' for every part", flush=True)

    step_files = sorted(p for p in Path(parts_dir).iterdir()
                         if p.suffix.lower() in _STEP_SUFFIXES)

    rows: list[dict] = []
    for step_path in step_files:
        stem = step_path.stem
        out_msh = out_path / f"{stem}.msh"
        out_step = out_path / f"{stem}_cleaned.step"
        part_timeout = _timeout_for(stem, timeout_s)

        print(f"[prep_parts] {step_path.name} (timeout={part_timeout:.0f}s) ...", flush=True)
        t0 = time.time()
        try:
            result = _run_one_bounded(str(step_path), str(out_msh), str(out_step),
                                       target_nodes, part_timeout)
        except Exception as e:
            result = dict(n_solids=0, volume_mm3=0.0, bbox_diag_mm=0.0, n_nodes=0,
                          ok=False, healed=False, notes=f"controller error: {e}")
        dt = time.time() - t0

        if ranges:
            ood = is_ood(result["volume_mm3"], result["bbox_diag_mm"], ranges)
            ood_str = "yes" if ood else "no"
        else:
            ood = False
            ood_str = "unknown"

        status = "OK" if result["ok"] else "FAILED"
        if result["ok"] and ood:
            status = "OK (OOD)"

        row = dict(file=step_path.name, n_solids_in=result["n_solids"],
                    kept_vol_mm3=result["volume_mm3"], diag_mm=result["bbox_diag_mm"],
                    healed=result["healed"], mesh_nodes=result["n_nodes"],
                    ood=ood_str, status=status, notes=result.get("notes", ""),
                    seconds=dt)
        rows.append(row)
        print(f"  -> {status}, nodes={result['n_nodes']}, {dt:.1f}s -- {result.get('notes', '')}",
              flush=True)

    write_report(rows, str(out_path / "prep_report.md"))
    return rows


def write_report(rows: list[dict], out_path: str) -> None:
    n_ok = sum(1 for r in rows if r["status"].startswith("OK"))
    lines = [
        "# Dataset-D Parts Prep Report\n\n",
        f"Processed {len(rows)} parts -- {n_ok} usable (mesh OK), {len(rows) - n_ok} failed.\n\n",
        "| file | n_solids_in | kept_vol_mm3 | diag_mm | healed? | mesh_nodes | OOD? | status | notes |\n",
        "|---|---|---|---|---|---|---|---|---|\n",
    ]
    for r in rows:
        lines.append(
            f"| {r['file']} | {r['n_solids_in']} | {r['kept_vol_mm3']:.1f} | "
            f"{r['diag_mm']:.1f} | {r['healed']} | {r['mesh_nodes']} | {r['ood']} | "
            f"{r['status']} | {r['notes']} |\n"
        )
    Path(out_path).write_text("".join(lines))


# --------------------------------------------------------------------------
# argparse entry point
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--parts-dir", default="pc_parts")
    ap.add_argument("--out-dir", default="doe_out/parts_prepped")
    ap.add_argument("--target-nodes", type=int, default=DEFAULT_TARGET_NODES)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--ood-ranges", default=DEFAULT_OOD_RANGES_PATH)
    # Hidden single-part worker mode, used internally via subprocess for a
    # real per-part time-box (see `_run_one_bounded`). Not part of the
    # documented CLI surface.
    ap.add_argument("--single", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-msh", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-step", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.single:
        result = heal_and_mesh(args.single, args.out_msh, target_nodes=args.target_nodes,
                                out_step=args.out_step)
        print(json.dumps(result))
        return 0

    run_all(args.parts_dir, args.out_dir, target_nodes=args.target_nodes,
            timeout_s=args.timeout, ood_ranges_path=args.ood_ranges)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
