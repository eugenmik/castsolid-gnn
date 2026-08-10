"""Reprocess raw .pt + paired .msh into cached per-node feature files.

Writes one .npz per casting under <data_root>/features/graphs/ plus a global
meta.npz and norm_stats.npz. Parallelised across CPU cores.
"""
import os
# CPU-hygiene: cap every BLAS backend to 1 thread PER WORKER PROCESS, set
# before numpy (or anything that transitively imports it) is first imported
# in this process. This must run before `import numpy as np` below to take
# effect for numpy's own BLAS init; ProcessPoolExecutor workers either
# re-import this module from scratch ("spawn", re-running this) or fork from
# a parent that already applied it ("fork", inherited via copy-on-write) --
# mirrors tools/doe/doe.py's identical top-of-file pattern.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import itertools
import json
import os.path as osp
import re
import shutil
import sys
from glob import glob

import meshio
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))
from src.msh_parse import parse_msh
from src.features import (compute_features, FEATURE_NAMES, FEATURE_NAMES_V3,
                          FEATURE_NAMES_V4, FEATURE_NAMES_V5, G_VECTOR_NAMES,
                          _node_features, alloy_g_vector, chvorinov_t_ref)
from src.geometry import mesh_volume, total_area

_FACE_RE = re.compile(r'_[+-][XYZ]')
_MSH_SETS = ("msh_none", "msh_half", "msh_all")

# Canonical raw temperature backup (dolfinx-DOF-order snapshots_nodal), on the
# large Media drive. Used when data/pt/*.pt has been deleted to free space.
# The .pt temps are byte-identical to these npz (verified). Override if moved.
RAW_NPZ_DIRS = ("data/raw",)


def find_msh(stem: str, data_root: str = "data"):
    for s in _MSH_SETS:
        hits = glob(osp.join(data_root, s, "*", f"{stem}.msh"))
        if hits:
            return hits[0]
    return None


def _load_raw_temps(stem: str, data_root: str = "data"):
    """Return (temps_dof (T,N) float32, times (T,) float32) or None.

    Prefers data/pt/<stem>.pt; falls back to the canonical raw npz backup so the
    pipeline still rebuilds after data/pt is deleted."""
    ptp = osp.join(data_root, "pt", f"{stem}.pt")
    if osp.exists(ptp):
        raw = torch.load(ptp, weights_only=False)
        return raw["temps"].numpy().astype(np.float32), raw["times"].numpy().astype(np.float32)
    for d in RAW_NPZ_DIRS:
        hits = glob(osp.join(d, "*", f"{stem}.npz"))
        if hits:
            z = np.load(hits[0])
            return z["snapshots_nodal"].astype(np.float32), z["times"].astype(np.float32)
    return None


def process_one(stem: str, data_root: str = "data", feature_version: int = 2):
    mpath = find_msh(stem, data_root)
    if mpath is None:
        print(f"  WARN missing msh for {stem}")
        return None
    src = _load_raw_temps(stem, data_root)
    if src is None:
        print(f"  WARN no raw temps for {stem} (.pt or npz backup)")
        return None
    temps_dof, times = src
    mesh = parse_msh(mpath)
    n_pt = temps_dof.shape[1]
    if mesh.node_xyz.shape[0] != n_pt:
        print(f"  WARN node count mismatch {stem}: msh={mesh.node_xyz.shape[0]} raw={n_pt}")
        return None

    # Apply the node-ordering recovery permutation: the raw temps are in dolfinx
    # DOF order, which is scrambled relative to the gmsh node order used by pos/
    # mesh labels. perm g maps DOF->gmsh node: temps_gmsh[:, g] = temps_dof.
    # Files without a validated perm are excluded (their per-node target is
    # unrecoverable). See docs spec §1.5.
    perm_path = osp.join(data_root, "features", "perms", f"{stem}.npy")
    if not osp.exists(perm_path):
        print(f"  WARN no recovery perm for {stem} (excluded)")
        return None
    g = np.load(perm_path)
    if g.shape[0] != n_pt:
        print(f"  WARN perm/temps size mismatch {stem}: perm={g.shape[0]} temps_N={n_pt}")
        return None
    temps = np.empty_like(temps_dof)
    temps[:, g] = temps_dof                 # reorder into gmsh node order

    feat = compute_features(mesh, temps, times, version=feature_version)

    n_adiab = len(_FACE_RE.findall(stem))
    prefix = stem.split("_")[0]
    return {
        "stem": stem,
        "X": feat["X"],
        "y": feat["y"],
        "pos": feat["pos"],
        "mesh_edges": feat["mesh_edges"],
        "y_global": feat["y_global"],
        "prefix": prefix,
        "n_adiab": n_adiab,
        "is_fully_cooled": n_adiab == 0,
        "n_nodes": n_pt,
    }


import argparse
from concurrent.futures import ProcessPoolExecutor


def _init_worker():
    """ProcessPoolExecutor initializer: lower scheduling priority + force
    every BLAS backend down to 1 thread at RUNTIME via threadpoolctl. The
    module-top env vars already cover a fresh/"spawn" worker's own numpy
    import; this is the belt-and-braces fallback for a "fork" worker that
    inherited an ALREADY-initialized BLAS library from the parent (in which
    case env vars set post-fork would be too late) -- mirrors
    tools/doe/doe.py's `_init_worker`."""
    try:
        os.nice(10)
    except Exception:
        pass
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass


def write_record(rec, out_dir):
    gdir = osp.join(out_dir, "graphs")
    os.makedirs(gdir, exist_ok=True)
    np.savez(osp.join(gdir, f"{rec['stem']}.npz"), X=rec["X"], y=rec["y"], pos=rec["pos"], mesh_edges=rec["mesh_edges"])


def _all_stems(data_root):
    """Enumerate casting stems. Prefer data/pt; after it is deleted, fall back to
    the recovery perms (the exact recoverable set), then the raw npz backup."""
    stems = [osp.splitext(osp.basename(p))[0]
             for p in glob(osp.join(data_root, "pt", "*.pt"))]
    if not stems:
        stems = [osp.splitext(osp.basename(p))[0]
                 for p in glob(osp.join(data_root, "features", "perms", "*.npy"))]
    if not stems:
        for d in RAW_NPZ_DIRS:
            stems = [osp.splitext(osp.basename(p))[0]
                     for p in glob(osp.join(d, "*", "*.npz"))]
            if stems:
                break
    return sorted(stems)


# ============================================================================
# feature_version == 4: join raw FEniCSx solves (`data/raw_v4/<run_id>.npz`,
# {t,gradT,dTdt} already in gmsh node order -- see tools/doe/adapters.py's
# map_to_gmsh) with the DOE manifest (per-run spec: material, mold/chill/
# sleeve BC, hold_out). One graph per manifest row; `run_id` is the cache key
# ("stem").
#
# IMPORTANT interface note: `src.features.compute_features(mesh, temps, times,
# version=4, spec=...)` (Task 3) still expects raw (T,N) temperature snapshots
# + times, and internally derives y via `solidification_time` -- that is the
# OLD raw-pipeline contract (v2/v3). The FT2 raw solves are NOT snapshot
# histories: `fenicsx_solver.py` already outputs the per-node crossing time
# `t` directly (see its module docstring). So this v4 path does NOT call
# `compute_features`; it composes the same lower-level building blocks
# (`_node_features`, `alloy_g_vector`, `chvorinov_t_ref`) directly, using the
# solver-provided `t` array as `y` with no re-derivation.
# ============================================================================
def _read_manifest_rows(manifest_path: str) -> list:
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def _archive_mesh_work_dir(data_root: str) -> str:
    d = osp.join(data_root, "mesh_work_v4")
    os.makedirs(d, exist_ok=True)
    return d


def _find_msh_v4(row: dict, data_root: str = "data"):
    """Resolve a manifest row's geometry to a .msh path.
    - "path": geometry_id verbatim (a full .msh path).
    - "primitive"/"scaled": reuse the existing 42-primitive lookup
      (`find_msh`) by geometry_id.
    - "archive": geometry_id is a STEP file path; mesh it via
      `src.step_mesh.mesh_step` (same tool `tools.doe.adapters.mesh()` uses
      for dataset D) into `<data_root>/mesh_work_v4/<run_id>.msh`, cached
      across calls (the pre-pass Chvorinov-C fit and the main pass would
      otherwise mesh the same archive part twice). Returns None (with a
      printed WARN) if the STEP import/meshing itself fails -- a bad archive
      part must WARN+skip its row, not crash the batch.
    `scale_factor` (if != 1.0) is applied to the resolved mesh's node
    coordinates by the caller, not here.

    ft2a feature-BC fix: a non-empty `mesh_variant` column (written by
    `tools/doe/augment_manifest_bc.py` for chill/sleeve rows) is an explicit
    face-tagged msh_half path that OVERRIDES every other resolution branch
    below -- mirrors `tools.doe.adapters.mesh()`'s identical check, which the
    SOLVER side already uses. msh_none (what `find_msh` would otherwise
    return) has no "Adiabatic" (BC-patch) facet tag, so without this override
    the chill/sleeve/exo BC-dependent feature channels (is_adiabatic,
    boundary_extraction, exothermic_input) would be computed as if the row
    were plain sand, even though the row's SOLVER target used the tagged
    msh_half mesh -- silently dropping the BC input signal from X. msh_none
    and its msh_half variant share the same node order/coordinates (verified
    -- only the facet tagging differs), so this override changes only the
    tag-derived feature channels, not the geometric features or the
    per-node alignment with the (msh_half-based) solver targets.

    Task 3 (ft3-assembly-finetune): dataset=="E" rows (assembly/integration)
    have mesh_variant as a literal direct path. For consistency with FT2a's
    batch-resilience pattern, missing mesh_variant is WARN+skip (row-level
    failure, not batch-killing), same as non-E rows."""
    mesh_variant = (row.get("mesh_variant") or "").strip()
    if mesh_variant:
        if not osp.exists(mesh_variant):
            print(f"  WARN mesh_variant {mesh_variant!r} (run_id="
                  f"{row.get('run_id')!r}) does not exist -- stale path?")
            return None
        return mesh_variant
    source = row.get("geometry_source") or "primitive"
    gid = row["geometry_id"]
    if source == "path":
        return gid if osp.exists(gid) else None
    if source == "archive":
        out_msh = osp.join(_archive_mesh_work_dir(data_root),
                          f"{row.get('run_id', 'archive')}.msh")
        if osp.exists(out_msh):
            return out_msh
        try:
            from src.step_mesh import mesh_step
            mesh_step(gid, out_msh)
        except Exception as e:  # noqa
            print(f"  WARN mesh_step failed for run_id={row.get('run_id')} "
                  f"(STEP={gid!r}): {e}")
            return None
        return out_msh if osp.exists(out_msh) else None
    return find_msh(gid, data_root)


def _resolve_material_props_v4(material_id: str, manifest_dir: str) -> dict:
    """material_id -> {k, rho, cp, L_kJ_kg, TL, TS, Tpour}, mirroring
    `tools.doe.adapters._resolve_material`'s 3-step fallback: the
    owner-verified CSV, then `materials_synthetic.csv` next to the manifest
    (SYN_* ids), then the 12 named anchor grades."""
    from tools.doe.load_properties import load_materials
    for m in load_materials():
        if m.material_id == material_id:
            return dict(k=m.k, rho=m.rho, cp=m.cp, L_kJ_kg=m.L,
                       TL=m.TL, TS=m.TS, Tpour=m.Tpour)

    synth_csv = osp.join(manifest_dir, "materials_synthetic.csv")
    if osp.exists(synth_csv):
        with open(synth_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("material_id") == material_id:
                    return dict(
                        k=float(row["k_W_mK"]), rho=float(row["rho_kg_m3"]),
                        cp=float(row["cp_J_kgK"]), L_kJ_kg=float(row["L_kJ_kg"]),
                        TL=float(row["TL_C"]), TS=float(row["TS_C"]),
                        Tpour=float(row["Tpour_C"]))

    from tools.doe.doe import _anchor_materials
    for m in _anchor_materials():
        if m.material_id == material_id:
            return dict(k=m.k, rho=m.rho, cp=m.cp, L_kJ_kg=m.L,
                       TL=m.TL, TS=m.TS, Tpour=m.Tpour)

    raise KeyError(
        f"material_id {material_id!r} not resolvable via load_properties, "
        f"{synth_csv}, or tools.doe.doe._anchor_materials()")


def _resolve_chill_effusivity_v4(row: dict) -> float:
    """0.0 unless bc_config=='chills', in which case
    `tools.doe.doe.get_chill_effusivities()[chill_material]` -- raises
    ValueError (never a silent 0.0) if chill_material is empty/unknown."""
    if (row.get("bc_config") or "none") != "chills":
        return 0.0
    from tools.doe.doe import get_chill_effusivities
    chill_material = (row.get("chill_material") or "").strip()
    effs = get_chill_effusivities()
    if not chill_material or chill_material not in effs:
        raise ValueError(
            f"run_id={row.get('run_id')!r}: bc_config=='chills' but "
            f"chill_material={chill_material!r} not in get_chill_effusivities() "
            f"(known: {sorted(effs)}) -- refusing to silently use 0.0")
    return float(effs[chill_material])


def _resolve_exo_q0_v4(row: dict,
                       exo_calibration_csv: str = "data/properties/exo_calibration.csv",
                       tol: float = 0.05) -> float:
    """0.0 unless bc_config=='exo_sleeve', in which case the nearest-MEF
    `q0_W_m2` row of `exo_calibration_csv` (the sweep output of
    `tools/doe/calibrate_exo.py`), resolved via the shared
    `tools.doe.load_properties.resolve_exo_q0_by_mef` lookup (also used by
    `src.assembly.validate_bodies` for Part B) -- raises ValueError (never a
    silent 0.0) if the calibration table is missing or has no row within
    `tol` of the row's sleeve_MEF, mirroring
    `tools.doe.adapters._resolve_exo_params`. The ValueError message is
    re-wrapped here with this row's run_id for context."""
    if (row.get("bc_config") or "none") != "exo_sleeve":
        return 0.0
    from tools.doe.load_properties import resolve_exo_q0_by_mef
    mef = float(row.get("sleeve_MEF") or 0.0)
    try:
        return resolve_exo_q0_by_mef(mef, exo_calibration_csv, tol)
    except ValueError as e:
        raise ValueError(f"run_id={row.get('run_id')!r}: {e}") from e


# --------------------------------------------------------------------- #
# I1 (ft2a-final-review): archive/DF feature-mesh coordinate correspondence.
#
# For archive rows the FEATURE builder (`_find_msh_v4`, above) re-meshes the
# row's STEP file via `mesh_step` INDEPENDENTLY of the SOLVER side, which
# also calls `mesh_step` (via `tools.doe.adapters.mesh()`) but into a
# DIFFERENT work directory (`<doe_dir>/mesh_work/` vs this module's
# `<data_root>/mesh_work_v4/`) -- two separate meshing invocations of the
# same STEP file. Until now only the resulting node COUNT was checked to
# match the raw npz's array length (see the `t.shape[0] != n_pt` check in
# `_process_row_stage1`); node ORDER/coordinates were never verified. The
# solver's `t`/`gradT`/`dTdt`/`t_liquidus` raw_v4 arrays are in the node
# order of THAT solve-time mesh (`fenicsx_solver._dof_to_gmsh_perm`
# coordinate-matches its dolfinx DOFs onto it, at ~1e-9 m tolerance); if a
# re-mesh of the identical STEP ever produced the same node count in a
# DIFFERENT order (a gmsh non-determinism this repo has not observed but
# cannot rule out -- different gmsh version, different meshing-algorithm
# code path, etc.), targets and features would silently misalign for that
# row with no other symptom. This mirrors the strict bijection
# `tools.doe.adapters.map_to_gmsh` already enforces on the solver side, on
# the feature side.
# --------------------------------------------------------------------- #
_ARCHIVE_COORD_TOL_MM = 1e-6


def _archive_reference_mesh_path(row: dict, manifest_dir: str) -> str:
    """The ORIGINAL solve-time .msh for an archive/DF row, as cached by
    `tools.doe.adapters.mesh()` under `<doe_dir>/mesh_work/<run_id>[_scaled]
    .msh` (`adapters._work_dir()`'s naming convention). `manifest_dir`
    (where manifest.csv lives) IS `<doe_dir>` in the standard layout (both
    default to "doe_out"), so no separate config knob is needed."""
    run_id = row.get("run_id", "<unknown>")
    scale = float(row.get("scale_factor") or 1.0)
    suffix = "_scaled" if scale != 1.0 else ""
    return osp.join(manifest_dir, "mesh_work", f"{run_id}{suffix}.msh")


def _assert_archive_mesh_coord_correspondence(mesh, row: dict, manifest_dir: str) -> None:
    """No-op unless `row["geometry_source"] == "archive"` (msh_none/msh_half
    primitive rows already share node order with the raw npz by construction
    -- verified -- so this cost is deliberately not paid there).

    For archive rows, raises a loud `ValueError` naming the run_id --
    NEVER silently proceeds -- when:
      - the reference solve-time mesh can't be found (an archive row with a
        raw_v4 solve implies `tools.doe.adapters.mesh()` ran and cached one;
        its absence means this check literally cannot be performed, which is
        treated as a failure, not a silent pass);
      - node counts differ; or
      - node coordinates fail to correspond AT THE SAME INDEX, within
        `_ARCHIVE_COORD_TOL_MM`.
    The last check is deliberately NOT a plain nearest-neighbour SET
    membership test (which a pure reordering of the identical point set
    would pass): it uses a KDTree to find, for every feature-mesh node, its
    nearest reference-mesh node (mirroring `fenicsx_solver.
    _dof_to_gmsh_perm`'s own coordinate-matching approach) but then requires
    BOTH that the match be within tolerance AND that the matched index equal
    the feature node's own index (identity correspondence) -- since the two
    meshes are expected to already share node order (same deterministic
    `mesh_step` call on the same STEP), a same-count/same-position-set but
    DIFFERENT-order mesh (exactly the silent-misalignment risk I1 flags)
    must fail this check, not pass it."""
    if (row.get("geometry_source") or "primitive") != "archive":
        return
    run_id = row.get("run_id", "<unknown>")
    ref_path = _archive_reference_mesh_path(row, manifest_dir)
    if not osp.exists(ref_path):
        raise ValueError(
            f"run_id={run_id!r}: archive-row coordinate-correspondence check "
            f"cannot find the original solve-time mesh at {ref_path!r} (the "
            f"one tools.doe.adapters.mesh() caches for the FEniCSx solve) -- "
            "refusing to trust the feature builder's independently-remeshed "
            "STEP without verifying it against it")

    ref_pts = meshio.read(ref_path).points[:, :3].astype(np.float64)
    feat_pts = np.asarray(mesh.node_xyz, dtype=np.float64)[:, :3]
    if ref_pts.shape[0] != feat_pts.shape[0]:
        raise ValueError(
            f"run_id={run_id!r}: archive feature-mesh node count "
            f"({feat_pts.shape[0]}) != solve-time mesh node count "
            f"({ref_pts.shape[0]}) at {ref_path!r} -- a gmsh remesh of the "
            "same STEP produced a DIFFERENT mesh; targets<->features cannot "
            "be trusted to align for this row")

    dist, idx = cKDTree(ref_pts).query(feat_pts)
    max_dist = float(dist.max())
    if max_dist > _ARCHIVE_COORD_TOL_MM:
        raise ValueError(
            f"run_id={run_id!r}: archive feature-mesh node coordinates do "
            f"NOT correspond to the solve-time mesh at {ref_path!r} within "
            f"{_ARCHIVE_COORD_TOL_MM} mm (max nearest-neighbour distance = "
            f"{max_dist:.6g} mm) -- a gmsh remesh of the same STEP produced "
            "a mesh whose node positions don't match; targets<->features "
            "would silently misalign for this row")
    if not np.array_equal(idx, np.arange(feat_pts.shape[0])):
        n_reordered = int((idx != np.arange(feat_pts.shape[0])).sum())
        raise ValueError(
            f"run_id={run_id!r}: archive feature-mesh node ORDER does not "
            f"match the solve-time mesh at {ref_path!r} -- {n_reordered} "
            "node(s) matched a DIFFERENT index by nearest-neighbour "
            "coordinate (same node positions, different order) -- a gmsh "
            "remesh of the same STEP produced the same node count/positions "
            "but a DIFFERENT order; targets<->features would silently "
            "misalign for this row")


def _load_mesh_for_row(row: dict, data_root: str):
    """Resolve + parse a manifest row's mesh, applying scale_factor (if any)
    to node coordinates. Returns None (with a printed warning) if the .msh
    can't be found."""
    mpath = _find_msh_v4(row, data_root)
    if mpath is None:
        print(f"  WARN missing msh for run_id={row.get('run_id')} "
              f"(geometry_id={row.get('geometry_id')!r})")
        return None
    mesh = parse_msh(mpath)
    scale = float(row.get("scale_factor") or 1.0)
    if scale != 1.0:
        mesh.node_xyz = mesh.node_xyz * scale
    return mesh


def _fit_chvorinov_c_per_family(rows: list, raw_dir: str, data_root: str) -> dict:
    """Per alloy_family, a fixed-exponent (Chvorinov exponent==2) Chvorinov
    constant C fit from the generated (t_total, V/A) pairs: log(C) =
    mean(log(t_total) - 2*log(M)) -- the geometric mean of the per-row ratio
    t_total/M^2. This is the "underlying fit" `gate_b.fit_chvorinov` performs
    a free two-parameter (slope+intercept) regression of, specialized to a
    FIXED slope of 2 (the assumed t=C*M^2 model): more robust than the free
    regression when a family has few/degenerate (M, t) points (e.g. a single
    geometry solved once -- `gate_b.fit_chvorinov` requires >=3 non-degenerate
    points and would raise), which is expected for early/small manifests."""
    # M-4 (ft2a-branch-review): dataset=="S" (sanity cross-check pairs, always
    # a duplicate of an existing A/B/C row) rows are EXCLUDED here -- they
    # must not double-weight their base row's (M, t_total) point in the
    # per-family geometric-mean fit, mirroring `_build_splits_v4`'s existing
    # S exclusion from train/test.
    fam_points = {}
    for row in rows:
        if row.get("dataset") == "S":
            continue
        run_id = row["run_id"]
        mesh = _load_mesh_for_row(row, data_root)
        if mesh is None:
            continue
        raw_path = osp.join(raw_dir, f"{run_id}.npz")
        if not osp.exists(raw_path):
            continue
        with np.load(raw_path) as z:
            if "t" not in z.files:
                continue
            t_total = float(np.max(z["t"]))
        cool_tri = mesh.tris.get("Cooling", np.empty((0, 3), np.int64))
        adiab_tri = mesh.tris.get("Adiabatic", np.empty((0, 3), np.int64))
        area = total_area(mesh.node_xyz, cool_tri) + total_area(mesh.node_xyz, adiab_tri)
        vol = mesh_volume(mesh.node_xyz, mesh.tets)
        if area <= 0 or vol <= 0 or t_total <= 0:
            continue
        M = vol / area
        fam = row.get("alloy_family") or "unknown"
        fam_points.setdefault(fam, []).append((M, t_total))

    return _fit_c_from_points(fam_points)


def _fit_c_from_points(fam_points: dict) -> dict:
    """Shared core of the per-family Chvorinov-C geometric-mean fit: given
    {alloy_family: [(M, t_total), ...]} (M = volume/area), returns
    {alloy_family: C}. Factored out so `_fit_chvorinov_c_per_family`
    (mesh-parsing, standalone/tested directly) and `build_v4`'s parallel
    stage-1 pipeline (which already has volume/area/t_total from stage 1 and
    must NOT re-parse the mesh to get them -- see M-3) can never silently
    diverge in the actual fit math."""
    C_by_family = {}
    for fam, pts in fam_points.items():
        M = np.array([p[0] for p in pts], dtype=np.float64)
        T = np.array([p[1] for p in pts], dtype=np.float64)
        C_by_family[fam] = float(np.exp(np.mean(np.log(T) - 2.0 * np.log(M))))
    return C_by_family


# ----------------------------------------------------------------------- #
# M-3 (ft2a-branch-review): build_v4 used to parse each row's .msh TWICE --
# once inside `_fit_chvorinov_c_per_family`'s pre-pass (for volume/area, to
# compute M=V/A) and again inside `process_one_v4`'s main pass (for X/pos/
# mesh_edges) -- and did both passes SERIALLY, row by row, which is the slow
# path for the ~4725-row FT2 manifest. Fixed by splitting the per-row work
# into two stages:
#   stage 1 (`_process_row_stage1`, embarrassingly parallel, one mesh parse
#     per row): resolve+parse the mesh ONCE, load the raw solve, resolve the
#     BC/material spec, and compute X/pos/mesh_edges/volume/area/t_total.
#     None of this needs the cross-row Chvorinov C fit -- `_node_features`
#     never reads `spec["C_material"]` (only `chvorinov_t_ref` does, in stage
#     2), so X can be computed here without a fitted C yet.
#   (serial aggregate, cheap, no mesh reparse): fit C_by_family straight from
#     stage 1's (volume, area, t_total) triples via `_fit_c_from_points` --
#     the same fit math `_fit_chvorinov_c_per_family` uses, so the two paths
#     can never silently diverge.
#   stage 2 (`_finalize_row_stage2`, pure arithmetic, no mesh/IO): plugs the
#     fitted C into t_ref/y_nondim/g and assembles the final record.
# `build_v4` calls the two stages directly (stage 1 parallelised across the
# whole manifest, once) instead of going through `process_one_v4`, which
# remains as a back-compat single-row wrapper composing both stages in one
# call for anyone calling it directly.
# ----------------------------------------------------------------------- #
def _process_row_stage1(row: dict, manifest_dir: str, raw_dir: str, data_root: str,
                        exo_calibration_csv: str = "data/properties/exo_calibration.csv",
                        feature_version: int = 4):
    """Parallel-safe, picklable: parses the mesh exactly once. Returns a dict
    of picklable (numpy/scalar) results for stage 2, or None (+ printed WARN)
    if this row must be skipped -- every skip condition here is byte-for-byte
    the same check `process_one_v4` used to make (missing mesh/raw/t, node-
    count mismatch, unknown material_id, unresolvable chill/exo BC props).

    `feature_version` (Task 2, bc-reach): threaded straight into
    `_node_features` -- 4 (default, v4's 15 cols) or 5 (v4's 15 cols +
    Task 1's 5 bc-reach cols, `build_v5`'s 20-wide X). Nothing else in this
    function depends on the feature version; the BC-spec resolution above
    is shared verbatim by both."""
    run_id = row["run_id"]
    mesh = _load_mesh_for_row(row, data_root)
    if mesh is None:
        return None

    raw_path = osp.join(raw_dir, f"{run_id}.npz")
    if not osp.exists(raw_path):
        print(f"  WARN no raw_v4 npz for run_id={run_id}: {raw_path}")
        return None
    with np.load(raw_path) as z:
        if "t" not in z.files:
            print(f"  WARN raw_v4 npz for {run_id} has no 't' field (keys={z.files})")
            return None
        t = z["t"].astype(np.float32)
        gradT = z["gradT"].astype(np.float32) if "gradT" in z.files else np.zeros_like(t)
        dTdt = z["dTdt"].astype(np.float32) if "dTdt" in z.files else np.zeros_like(t)
        # t_liquidus (FT2a: liquidus-crossing time, enables t - t_liquidus =
        # local solidification time without re-simulating) is OPTIONAL --
        # older raw_v4 npz files predate this field. Tolerate its absence
        # (or a malformed length) by simply not carrying it forward; never
        # crash and never fabricate a value (unlike gradT/dTdt, a missing
        # t_liquidus has no safe zero-fallback -- see _finalize_row_stage2).
        t_liquidus = None
        if "t_liquidus" in z.files:
            _tl = z["t_liquidus"].astype(np.float32)
            if _tl.shape[0] == t.shape[0]:
                t_liquidus = _tl
            else:
                print(f"  WARN t_liquidus length mismatch run_id={run_id}: "
                      f"t={t.shape[0]} t_liquidus={_tl.shape[0]} -- skipping "
                      "t_liquidus/local_solidification_time for this row")

    n_pt = mesh.node_xyz.shape[0]
    if t.shape[0] != n_pt:
        print(f"  WARN node count mismatch run_id={run_id}: msh={n_pt} raw={t.shape[0]}")
        return None

    # I1 (ft2a-final-review): archive/DF rows only -- see the function's
    # docstring above. Deliberately NOT wrapped in try/except: a mismatch (or
    # a missing reference mesh) here is a hard, batch-aborting failure, not a
    # row-level WARN+skip like every other check in this function -- it
    # signals a potential systemic feature<->target misalignment, not a bad
    # individual row.
    _assert_archive_mesh_coord_correspondence(mesh, row, manifest_dir)

    try:
        spec = _resolve_material_props_v4(row["material_id"], manifest_dir)
    except KeyError as e:
        print(f"  WARN {e}")
        return None

    bc_config = row.get("bc_config") or "none"
    spec["mold_effusivity"] = float(row["mold_effusivity"])
    spec["bc_config"] = bc_config
    spec["sleeve_MEF"] = float(row.get("sleeve_MEF") or 0.0)
    # Row-level BC-property resolution failures (unknown chill_material, an
    # exo_sleeve with no calibrated MEF / no exo_calibration.csv yet) must
    # WARN+skip this one row, exactly like every other row-level failure
    # above (missing mesh/raw/t, node-count mismatch, unknown material_id) --
    # NOT abort the whole build_v4 batch. _resolve_chill_effusivity_v4 /
    # _resolve_exo_q0_v4 raise ValueError by design (never a silent 0.0); this
    # is where that "loud" behavior is turned into a "skip THIS row" outcome
    # instead of letting it propagate out of build_v4 mid-manifest.
    try:
        spec["chill_effusivity"] = _resolve_chill_effusivity_v4(row)
        spec["exo_q0"] = _resolve_exo_q0_v4(row, exo_calibration_csv)
    except ValueError as e:
        print(f"  WARN {e}")
        return None

    # `spec` here deliberately has NO "C_material" key yet -- `_node_features`
    # (version=4/5) never reads it (only mold_effusivity/bc_config/
    # chill_effusivity/sleeve_MEF/exo_q0), so X/pos/mesh_edges are fully
    # computable before the cross-row Chvorinov fit exists.
    X, pos, mesh_edges = _node_features(mesh, version=feature_version, spec=spec)

    volume = float(X[0, FEATURE_NAMES.index("volume_mm3")])
    area = float(X[0, FEATURE_NAMES.index("surface_area_mm2")])
    fam = row.get("alloy_family") or "unknown"

    return {
        "run_id": run_id, "dataset": row.get("dataset", ""),
        "alloy_family": fam, "bc_config": bc_config,
        "hold_out": row.get("hold_out", ""),
        "X": X, "pos": pos, "mesh_edges": mesh_edges,
        "spec": spec, "volume": volume, "area": area,
        "t": t, "gradT": gradT, "dTdt": dTdt, "t_liquidus": t_liquidus,
        "t_total": float(np.max(t)), "n_nodes": n_pt,
    }


def _finalize_row_stage2(s1: dict, C_by_family: dict):
    """Pure arithmetic + dict assembly, no mesh/IO: plugs the fitted
    per-family Chvorinov C into stage 1's output. Returns None (+ printed
    WARN) if no C was fitted for this row's family (e.g. every row in that
    family failed stage 1)."""
    fam = s1["alloy_family"]
    run_id = s1["run_id"]
    if fam not in C_by_family:
        print(f"  WARN no Chvorinov C fitted for alloy_family={fam!r} "
              f"(run_id={run_id}) -- skipped")
        return None

    spec = dict(s1["spec"])
    spec["C_material"] = C_by_family[fam]
    t = s1["t"]
    t_ref = chvorinov_t_ref(s1["volume"], s1["area"], spec["C_material"])
    g = alloy_g_vector(spec)
    y_nondim = (t.astype(np.float64) / max(t_ref, 1e-9)).astype(np.float32)

    rec = {
        "stem": run_id, "run_id": run_id, "dataset": s1["dataset"],
        "alloy_family": fam, "bc_config": s1["bc_config"],
        "hold_out": s1["hold_out"],
        "X": s1["X"], "pos": s1["pos"], "mesh_edges": s1["mesh_edges"],
        "g": g, "y": t, "y_nondim": y_nondim, "t_ref": t_ref,
        "gradT": s1["gradT"], "dTdt": s1["dTdt"],
        "y_global": float(np.max(t)), "n_nodes": s1["n_nodes"],
    }
    t_liquidus = s1["t_liquidus"]
    if t_liquidus is not None:
        # local solidification time = t_solidus - t_liquidus, clipped at 0
        # (the two independent threshold-crossing interpolations can, in
        # principle, disagree by a tiny numerical amount right at a node
        # that crosses both thresholds within the same accepted step).
        rec["t_liquidus"] = t_liquidus
        rec["local_solidification_time"] = np.clip(t - t_liquidus, 0.0, None).astype(np.float32)
    return rec


def process_one_v4(row: dict, manifest_dir: str, raw_dir: str, data_root: str,
                   C_by_family: dict,
                   exo_calibration_csv: str = "data/properties/exo_calibration.csv"):
    """Back-compat single-row entrypoint (still parses the mesh exactly
    once). `build_v4` does NOT call this -- it calls stage 1 across the
    whole manifest first (parallelised) and stage 2 after the C fit, so a
    row's mesh is parsed once per BUILD, not once per stage."""
    s1 = _process_row_stage1(row, manifest_dir, raw_dir, data_root, exo_calibration_csv)
    if s1 is None:
        return None
    return _finalize_row_stage2(s1, C_by_family)


def write_record_v4(rec, out_dir):
    gdir = osp.join(out_dir, "graphs")
    os.makedirs(gdir, exist_ok=True)
    extra = {}
    if "t_liquidus" in rec:
        extra["t_liquidus"] = rec["t_liquidus"]
        extra["local_solidification_time"] = rec["local_solidification_time"]
    np.savez(osp.join(gdir, f"{rec['stem']}.npz"),
             X=rec["X"], pos=rec["pos"], mesh_edges=rec["mesh_edges"],
             g=rec["g"], y=rec["y"], y_raw=rec["y"], y_nondim=rec["y_nondim"],
             t_ref=np.float64(rec["t_ref"]),
             gradT=rec["gradT"], dTdt=rec["dTdt"], **extra)


def _build_splits_v4(meta: dict) -> dict:
    """{"interp": {"train","test"}, "extrap": {"train","test"}} from the
    manifest's hold_out column. dataset=="S" (sanity cross-check pairs) rows
    are EXCLUDED from every train (and test) list here, regardless of their
    hold_out value -- they are not part of the GNN train/val protocol."""
    stems = list(meta["stem"]); dataset = list(meta["dataset"]); hold_out = list(meta["hold_out"])
    def _stems_where(pred):
        return sorted(s for s, d, h in zip(stems, dataset, hold_out)
                     if d != "S" and pred(h))
    train = _stems_where(lambda h: h == "train")
    val_interp = _stems_where(lambda h: h == "val_interp")
    val_extrap = _stems_where(lambda h: h == "val_extrap")
    return {
        "interp": {"train": train, "test": val_interp},
        "extrap": {"train": train, "test": val_extrap},
    }


def _init_norm_accum(feature_version: int = 4, n_feat: int = None):
    """Fresh accumulator state for the norm-stats running sums, shared by
    `build_v4`/`build_v5`'s Phase 2 fold and `recompute_norm_stats_v4` (so
    these can never silently diverge in the accumulation math).

    `n_feat`, if given, overrides the width implied by `feature_version`
    (used by `recompute_norm_stats_v4` to size the accumulator from an
    on-disk graph's ACTUAL X width -- 15 for a v4 cache, 20 for a v5 one --
    instead of assuming a version from the caller)."""
    if n_feat is None:
        n_feat = len(FEATURE_NAMES_V5) if feature_version >= 5 else len(FEATURE_NAMES_V4)
    n_g = len(G_VECTOR_NAMES)
    return {
        "fsum": np.zeros(n_feat), "fsq": np.zeros(n_feat), "fcnt": 0,
        "gsum": np.zeros(n_g), "gsq": np.zeros(n_g), "gcnt": 0,
        "ylsum": 0.0, "ylsq": 0.0, "ycnt": 0,
    }


def _accumulate_norm_stats(acc: dict, X: np.ndarray, g: np.ndarray, y: np.ndarray) -> None:
    """Fold one row's (X, g, y) into `acc` in place."""
    X = X.astype(np.float64)
    acc["fsum"] += X.sum(0); acc["fsq"] += (X ** 2).sum(0); acc["fcnt"] += X.shape[0]
    g = g.astype(np.float64)
    acc["gsum"] += g; acc["gsq"] += g ** 2; acc["gcnt"] += 1
    yl = np.log1p(y.astype(np.float64))
    acc["ylsum"] += yl.sum(); acc["ylsq"] += (yl ** 2).sum(); acc["ycnt"] += yl.size


def _finalize_norm_stats(acc: dict) -> dict:
    """Turn the running sums into the final {feat,g,y_log}_{mean,std} dict --
    exactly the arithmetic `build_v4` used to do inline."""
    feat_mean = acc["fsum"] / max(acc["fcnt"], 1)
    feat_std = np.sqrt(np.clip(acc["fsq"] / max(acc["fcnt"], 1) - feat_mean ** 2, 1e-12, None))
    g_mean = acc["gsum"] / max(acc["gcnt"], 1)
    g_std = np.sqrt(np.clip(acc["gsq"] / max(acc["gcnt"], 1) - g_mean ** 2, 1e-12, None))
    y_log_mean = acc["ylsum"] / max(acc["ycnt"], 1)
    y_log_std = float(np.sqrt(max(acc["ylsq"] / max(acc["ycnt"], 1) - y_log_mean ** 2, 1e-12)))
    return {
        "feat_mean": feat_mean, "feat_std": feat_std,
        "g_mean": g_mean, "g_std": g_std,
        "y_log_mean": np.float64(y_log_mean), "y_log_std": np.float64(y_log_std),
    }


def recompute_norm_stats_v4(features_dir: str) -> dict:
    """Recompute `norm_stats.npz` from EVERY graph `.npz` currently on disk
    under `<features_dir>/graphs/`, using exactly the same accumulation
    conventions as `build_v4`'s Phase 2 (`_accumulate_norm_stats` /
    `_finalize_norm_stats` are the SAME helpers build_v4 folds each row
    through -- this cannot silently diverge from a full build's math).

    dataset=="S" rows are excluded, mirroring build_v4's own exclusion --
    looked up via `<features_dir>/meta.npz`'s stem/dataset columns (still
    the full, untouched manifest-wide meta written by the last full build;
    a selective `only_stems` rebuild deliberately never rewrites it).

    Overwrites `<features_dir>/norm_stats.npz` in place and returns the new
    stats dict. Intended for use AFTER a selective (`only_stems`) rebuild,
    to fold the rebuilt rows' (possibly changed) features back into the
    global normalization without re-running the whole manifest.

    Version-agnostic (v4 15-wide or v5 20-wide cache): the accumulator's
    feature width is read off the FIRST graph `.npz` actually found on disk
    (`X.shape[1]`), not assumed from a hardcoded FEATURE_NAMES_V4 -- so this
    one function serves both a v4 and a v5 `features_dir` without a
    parameter, and can never disagree with what's actually cached there."""
    meta_path = osp.join(features_dir, "meta.npz")
    dataset_by_stem = {}
    if osp.exists(meta_path):
        with np.load(meta_path, allow_pickle=True) as mz:
            for stem, ds in zip(mz["stem"], mz["dataset"]):
                dataset_by_stem[str(stem)] = str(ds)

    gpaths = sorted(glob(osp.join(features_dir, "graphs", "*.npz")))
    n_feat = None
    if gpaths:
        with np.load(gpaths[0]) as z0:
            n_feat = int(z0["X"].shape[1])
    acc = _init_norm_accum(n_feat=n_feat)
    for gp in gpaths:
        stem = osp.splitext(osp.basename(gp))[0]
        if dataset_by_stem.get(stem) == "S":
            continue
        with np.load(gp) as z:
            _accumulate_norm_stats(acc, z["X"], z["g"], z["y"])

    stats = _finalize_norm_stats(acc)
    np.savez(osp.join(features_dir, "norm_stats.npz"), **stats)
    return stats


def _recover_c_by_family_from_cache(rows: list, out_dir: str) -> dict:
    """For a SELECTIVE (`only_stems`) rebuild: recover each needed
    alloy_family's Chvorinov C by inverting `t_ref = C * (V/A)**2` from the
    row's OWN currently-cached graph `.npz` (read BEFORE this rebuild
    overwrites it). Volume/area are geometry-only and unchanged by Task 1's
    exothermic_input re-encoding, so this exactly reproduces the ORIGINAL
    full-manifest fit's C for that family -- to floating-point precision,
    not a re-fit from the (possibly much smaller/skewed) `only_stems`
    subset, which would silently shift t_ref/y_nondim for every row in that
    family (see the Task 2 brief's Chvorinov-C correctness point).

    One point per family suffices (algebraic inversion of a constant, not a
    regression); if more than one row of a family is present among `rows`,
    their recovered C values are averaged purely as a sanity cross-check
    (they should already agree to float precision)."""
    v_idx = FEATURE_NAMES.index("volume_mm3")
    a_idx = FEATURE_NAMES.index("surface_area_mm2")
    fam_c_vals = {}
    for row in rows:
        run_id = row["run_id"]
        fam = row.get("alloy_family") or "unknown"
        gpath = osp.join(out_dir, "graphs", f"{run_id}.npz")
        if not osp.exists(gpath):
            continue
        with np.load(gpath) as z:
            if "t_ref" not in z.files or "X" not in z.files:
                continue
            X = z["X"]
            t_ref_old = float(z["t_ref"])
            V = float(X[0, v_idx]); A = float(X[0, a_idx])
        if V <= 0 or A <= 0 or t_ref_old <= 0:
            continue
        M = V / A
        fam_c_vals.setdefault(fam, []).append(t_ref_old / (M ** 2))
    return {fam: float(np.mean(vals)) for fam, vals in fam_c_vals.items()}


def _run_stage1_v4(rows: list, manifest_dir: str, raw_dir: str, data_root: str,
                   exo_calibration_csv: str, n_workers, feature_version: int = 4) -> list:
    """Stage 1 across the whole manifest. Returns a list of `dict | None`,
    ONE PER ROW, in the SAME ORDER as `rows` (required for the serial
    aggregation phase below to be byte-identical to the pre-parallel serial
    build, which folded rows into norm_stats/meta in manifest order).
    `n_workers in (None, 1)` runs serially in-process (no pool overhead,
    easiest to debug); otherwise a bounded `ProcessPoolExecutor` -- BLAS
    capped (module-top env vars + `_init_worker`'s threadpoolctl fallback)
    and niced, mirroring tools/doe/doe.py's execute() pool. `ex.map` (not
    submit+as_completed) is used specifically because it yields results in
    submission order regardless of completion order.

    `feature_version` (Task 2, bc-reach): forwarded to every
    `_process_row_stage1` call (serial or pooled) -- `build_v4` passes 4
    (the default), `build_v5` passes 5."""
    if n_workers is None or n_workers <= 1:
        return [_process_row_stage1(r, manifest_dir, raw_dir, data_root,
                                    exo_calibration_csv, feature_version)
                for r in rows]
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        return list(ex.map(
            _process_row_stage1, rows,
            itertools.repeat(manifest_dir), itertools.repeat(raw_dir),
            itertools.repeat(data_root), itertools.repeat(exo_calibration_csv),
            itertools.repeat(feature_version)))


def _build_v4_family(manifest_path: str, feature_version: int, raw_dir: str,
                     data_root: str, out_dir: str, limit: int,
                     exo_calibration_csv: str, n_workers, only_stems,
                     write_splits) -> None:
    """Shared implementation behind `build_v4` (feature_version=4, 15-wide
    X) and `build_v5` (feature_version=5, 20-wide X, Task 2/bc-reach).
    Everything here is feature-version-agnostic: X's width is entirely
    determined by `_process_row_stage1`/`_node_features(version=
    feature_version)` in stage 1 -- the per-family Chvorinov-C fit (or, for
    a selective `only_stems` rebuild, its cache-recovery + same-family
    fallback), stage-2 finalize + write, and norm-stats/meta accumulation
    never read a hardcoded column count.

    `write_splits(meta, out_dir)` is the ONE place the two callers diverge:
    called only for a FULL (non-selective) build, after meta.npz/
    norm_stats.npz are already written, it decides how splits.json is
    produced -- `build_v4` fits it from this build's own meta
    (`_build_splits_v4`); `build_v5` instead copies an existing v4 cache's
    splits.json VERBATIM (same manifest/rows, so the split membership is
    identical by construction, never independently re-derived and risking
    divergence). A selective (`only_stems`) rebuild never calls
    `write_splits` at all -- splits.json is left untouched, same as
    meta.npz/norm_stats.npz.

    `only_stems` (optional set/iterable of `run_id`): a SELECTIVE rebuild --
    process only the manifest rows whose `run_id` is in `only_stems`, write
    (only) their graph `.npz`, and skip `meta.npz`/`norm_stats.npz`/
    `splits.json` entirely (those describe the WHOLE manifest and a partial
    rebuild must never truncate/corrupt them). The per-family Chvorinov C
    each selected row needs is NOT re-fit from the (possibly tiny/skewed)
    subset -- it is recovered from the already-cached graph .npz on disk
    (`_recover_c_by_family_from_cache`), so t_ref/y_nondim for unaffected
    rows never drifts. Use `recompute_norm_stats_v4` afterwards to fold the
    rebuilt rows back into `norm_stats.npz`."""
    rows = _read_manifest_rows(manifest_path)
    if limit:
        rows = rows[:limit]
    rows_full = rows
    if only_stems is not None:
        only_stems = set(only_stems)
        rows = [r for r in rows if r["run_id"] in only_stems]
    manifest_dir = osp.dirname(osp.abspath(manifest_path))
    os.makedirs(out_dir, exist_ok=True)

    # Phase 1 (parallel, embarrassingly-parallel per row -- M-3): resolve +
    # parse each row's mesh EXACTLY ONCE (previously parsed twice: once in
    # `_fit_chvorinov_c_per_family`'s pre-pass, once in `process_one_v4`'s
    # main pass), load its raw solve, resolve its BC/material spec, and
    # compute X/pos/mesh_edges/volume/area/t_total -- everything that does
    # NOT depend on the cross-row Chvorinov C fit.
    stage1 = _run_stage1_v4(rows, manifest_dir, raw_dir, data_root,
                            exo_calibration_csv, n_workers,
                            feature_version=feature_version)

    # Phase 1.5 (serial aggregate, cheap -- no mesh reparse, just a fold over
    # already-computed scalars): fit the per-family Chvorinov C, EXCLUDING
    # dataset=="S" rows (M-4, mirrored from `_fit_chvorinov_c_per_family`)
    # from the fit, using the exact same fit math (`_fit_c_from_points`) so
    # this can never silently diverge from the standalone/tested function.
    # Selective (only_stems) rebuild: NEVER re-fit C from this row subset --
    # recover the ORIGINAL full-manifest C per family from the already-cached
    # graph .npz on disk instead (see build_v4's docstring + the Task 2
    # brief's Chvorinov-C correctness point).
    if only_stems is not None:
        C_by_family = _recover_c_by_family_from_cache(rows, out_dir)
        # NEW stems (no graph .npz of their own cached yet) contribute no
        # point to the lookup above, so a family made up ENTIRELY of new
        # stems comes back with no C at all -- fall back to recovering that
        # family's C from any OTHER (already-cached, from a prior full or
        # selective build) row of the same family across the FULL manifest,
        # excluding dataset=="S" rows (mirrors the fit's own exclusion, see
        # `_fit_chvorinov_c_per_family`). This is still the same algebraic
        # inversion of a per-family constant, just sourced from a wider row
        # set -- never a re-fit from the (possibly tiny/skewed) only_stems
        # subset.
        needed = {r.get("alloy_family") or "unknown" for r in rows}
        missing = needed - set(C_by_family)
        if missing:
            fallback = [r for r in rows_full
                        if (r.get("alloy_family") or "unknown") in missing
                        and r.get("dataset") != "S"]
            C_by_family.update(_recover_c_by_family_from_cache(fallback, out_dir))
        # If a needed family STILL has no C after the full-manifest fallback,
        # every one of its rows would otherwise be silently skipped by
        # `_finalize_row_stage2`'s warn-and-skip -- a selective rebuild must
        # fail loudly instead of quietly writing zero graphs.
        still_missing = needed - set(C_by_family)
        if still_missing:
            raise RuntimeError(
                f"build_v{feature_version}(only_stems=...): no cached graph "
                f"found anywhere in {out_dir!r} to recover Chvorinov C for "
                f"alloy_family(s) {sorted(still_missing)!r} -- run a full "
                f"(non-selective) build_v{feature_version} first so at least "
                f"one graph of each family is cached.")
    else:
        fam_points = {}
        for s1 in stage1:
            if s1 is None or s1["dataset"] == "S":
                continue
            fam_points.setdefault(s1["alloy_family"], []).append(
                (s1["volume"] / s1["area"], s1["t_total"]))
        C_by_family = _fit_c_from_points(fam_points)

    # Phase 2 (serial -- pure arithmetic + file writes, no mesh parsing or
    # other heavy work, so parallelising it would only add IPC/pickling
    # overhead): finalize each row now that C_by_family is known, write its
    # graph .npz, and fold it into the running norm-stats/meta accumulators
    # in the SAME `rows` order the old serial build used -- byte-identical
    # accumulation order (floating-point sums are order-sensitive in
    # principle, though not in practice at this row count/precision).
    names = FEATURE_NAMES_V5 if feature_version >= 5 else FEATURE_NAMES_V4
    acc = _init_norm_accum(feature_version=feature_version)
    meta = {k: [] for k in ("stem", "run_id", "dataset", "alloy_family",
                            "bc_config", "hold_out", "y_global", "n_nodes")}
    n_written = 0

    for s1 in stage1:
        if s1 is None:
            continue
        rec = _finalize_row_stage2(s1, C_by_family)
        if rec is None:
            continue
        write_record_v4(rec, out_dir)
        n_written += 1
        if only_stems is not None:
            # Selective rebuild: meta/norm_stats/splits describe the WHOLE
            # manifest and are deliberately left untouched (see
            # `_build_v4_family`'s docstring) -- use `recompute_norm_stats_v4`
            # afterwards instead.
            continue
        # M-4 (ft2a-branch-review): dataset=="S" rows still get a cached
        # graph .npz (and a meta.npz row, so tooling that looks them up by
        # run_id keeps working) but must NOT fold into norm_stats -- they
        # duplicate an existing A/B/C geometry+material and would otherwise
        # mildly double-weight that point in feature/g/y normalization,
        # mirroring their existing exclusion from the Chvorinov-C fit above
        # and from train/test splits (`_build_splits_v4`).
        if rec["dataset"] != "S":
            _accumulate_norm_stats(acc, rec["X"], rec["g"], rec["y"])
        for k in meta:
            meta[k].append(rec[k])

    if only_stems is not None:
        print(f"Rebuilt {n_written} v{feature_version} graphs (of {len(rows)} "
              f"selected stems) -> {out_dir}/graphs (meta.npz/norm_stats.npz/"
              "splits.json untouched -- run recompute_norm_stats_v4 separately)")
        return

    stats = _finalize_norm_stats(acc)

    np.savez(osp.join(out_dir, "meta.npz"),
             stem=np.array(meta["stem"]), run_id=np.array(meta["run_id"]),
             dataset=np.array(meta["dataset"]),
             alloy_family=np.array(meta["alloy_family"]),
             bc_config=np.array(meta["bc_config"]),
             hold_out=np.array(meta["hold_out"]),
             y_global=np.array(meta["y_global"], dtype=np.float32),
             n_nodes=np.array(meta["n_nodes"]),
             feature_names=np.array(names))
    np.savez(osp.join(out_dir, "norm_stats.npz"), **stats)

    write_splits(meta, out_dir)

    print(f"Built {len(meta['stem'])} v{feature_version} graphs "
          f"(of {len(rows)} manifest rows) -> {out_dir}")


def build_v4(manifest_path: str, raw_dir: str = "data/raw_v4", data_root: str = "data",
            out_dir: str = "data/features_v4", limit: int = None,
            exo_calibration_csv: str = "data/properties/exo_calibration.csv",
            n_workers=1, only_stems=None):
    """v4 (15-wide X) feature-cache build. See `_build_v4_family`'s
    docstring for the full contract (stage 1/2 split, only_stems selective
    rebuild + Chvorinov-C cache-recovery). splits.json is fit from this
    build's own meta via `_build_splits_v4` -- the v4-specific case of
    `_build_v4_family`'s `write_splits` hook."""
    def _write_splits_v4(meta: dict, out_dir: str) -> None:
        splits = _build_splits_v4(meta)
        with open(osp.join(out_dir, "splits.json"), "w") as f:
            json.dump(splits, f, indent=2)

    return _build_v4_family(manifest_path, 4, raw_dir, data_root, out_dir,
                            limit, exo_calibration_csv, n_workers, only_stems,
                            _write_splits_v4)


def build_v5(manifest_path: str, raw_dir: str = "data/raw_v4", data_root: str = "data",
            out_dir: str = "data/features_v5", limit: int = None,
            exo_calibration_csv: str = "data/properties/exo_calibration.csv",
            n_workers=1, only_stems=None,
            v4_splits_path: str = "data/features_v4/splits.json"):
    """Task 2 (bc-reach): the 20-channel feature-cache build -- `build_v4`
    parameterized to feature_version=5 via the shared `_build_v4_family`
    implementation (see its docstring for the full stage 1/2 + only_stems
    contract, unchanged here). X is 20-wide: v4's 15 columns UNCHANGED
    (byte-identical to a v4 build of the same row -- Task 1's
    `compute_bc_reach_channels`/`FEATURE_NAMES_V5` only APPEND 5 new
    columns) + the 5 bc-reach columns from Task 1.

    Two deliberate differences from `build_v4`:
      - norm_stats.npz is FRESH over the full 20-wide X -- NOT v4's stats
        reused or zero-padded. `_init_norm_accum(feature_version=5)` sizes
        the accumulator to `len(FEATURE_NAMES_V5) == 20` and every row's
        real 20-column X is folded into it (`_accumulate_norm_stats`,
        unchanged -- it never assumes a column count).
      - splits.json is NOT computed from this build's own meta -- v5 uses
        the IDENTICAL manifest/rows/hold_out protocol as the v4 cache it's
        built alongside, so `splits.json` is copied VERBATIM from
        `v4_splits_path` (default `data/features_v4/splits.json`, the
        production v4 cache) instead of re-deriving it a second time --
        guaranteeing byte-for-byte the same split membership (incl. e_val)
        with zero risk of the two derivations silently diverging. Raises
        `FileNotFoundError` if `v4_splits_path` doesn't exist -- a v5 build
        requires the v4 cache to already exist; this never silently skips
        or fabricates a split. Selective (`only_stems`) rebuilds never call
        this at all (same as `build_v4`), so `v4_splits_path` need not even
        be reachable for those."""
    def _copy_splits_from_v4(meta: dict, out_dir: str) -> None:
        if not osp.exists(v4_splits_path):
            raise FileNotFoundError(
                f"build_v5: v4_splits_path {v4_splits_path!r} does not exist "
                "-- build_v5 copies splits.json VERBATIM from an existing v4 "
                "cache (same manifest/rows) instead of re-deriving it; run "
                "build_v4 on this manifest first, or pass the correct "
                "v4_splits_path")
        shutil.copyfile(v4_splits_path, osp.join(out_dir, "splits.json"))

    return _build_v4_family(manifest_path, 5, raw_dir, data_root, out_dir,
                            limit, exo_calibration_csv, n_workers, only_stems,
                            _copy_splits_from_v4)


def build(data_root="data", out_dir="data/features", n_workers=None, limit=None,
         feature_version=2, manifest_path=None, raw_dir="data/raw_v4",
         exo_calibration_csv="data/properties/exo_calibration.csv"):
    if feature_version >= 4:
        if manifest_path is None:
            raise ValueError("build(feature_version=4) requires manifest_path")
        return build_v4(manifest_path=manifest_path, raw_dir=raw_dir,
                        data_root=data_root, out_dir=out_dir, limit=limit,
                        exo_calibration_csv=exo_calibration_csv,
                        n_workers=(n_workers if n_workers is not None else 1))
    stems = _all_stems(data_root)
    if limit:
        stems = stems[:limit]
    os.makedirs(out_dir, exist_ok=True)

    # Determine feature names based on version
    names = FEATURE_NAMES_V3 if feature_version >= 3 else FEATURE_NAMES

    # accumulators for global normalization (Welford-free: sum, sumsq, count)
    fsum = np.zeros(len(names)); fsq = np.zeros(len(names)); fcnt = 0
    ylsum = ylsq = ycnt = 0.0
    meta = {k: [] for k in ("stem", "prefix", "n_adiab", "is_fully_cooled",
                            "y_global", "n_nodes")}

    def _accumulate(rec):
        nonlocal fsum, fsq, fcnt, ylsum, ylsq, ycnt
        write_record(rec, out_dir)
        X = rec["X"].astype(np.float64)
        fsum += X.sum(0); fsq += (X ** 2).sum(0); fcnt += X.shape[0]
        yl = np.log1p(rec["y"].astype(np.float64))
        ylsum += yl.sum(); ylsq += (yl ** 2).sum(); ycnt += yl.size
        for k in meta:
            meta[k].append(rec[k])

    worker = process_one
    if n_workers == 1:
        for i, s in enumerate(stems):
            rec = worker(s, data_root, feature_version)
            if rec is not None:
                _accumulate(rec)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(stems)}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(worker, s, data_root, feature_version): s for s in stems}
            done = 0
            for fut in futs:
                rec = fut.result()
                done += 1
                if rec is not None:
                    _accumulate(rec)
                if done % 500 == 0:
                    print(f"  {done}/{len(stems)}")

    feat_mean = fsum / max(fcnt, 1)
    feat_std = np.sqrt(np.clip(fsq / max(fcnt, 1) - feat_mean ** 2, 1e-12, None))
    y_log_mean = ylsum / max(ycnt, 1)
    y_log_std = float(np.sqrt(max(ylsq / max(ycnt, 1) - y_log_mean ** 2, 1e-12)))

    np.savez(osp.join(out_dir, "meta.npz"),
             stem=np.array(meta["stem"]),
             prefix=np.array(meta["prefix"]),
             n_adiab=np.array(meta["n_adiab"]),
             is_fully_cooled=np.array(meta["is_fully_cooled"]),
             y_global=np.array(meta["y_global"], dtype=np.float32),
             n_nodes=np.array(meta["n_nodes"]),
             feature_names=np.array(names))
    np.savez(osp.join(out_dir, "norm_stats.npz"),
             feat_mean=feat_mean, feat_std=feat_std,
             y_log_mean=np.float64(y_log_mean), y_log_std=np.float64(y_log_std))
    print(f"Built {len(meta['stem'])} graphs → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="data/features")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--feature-version", type=int, default=2)
    ap.add_argument("--manifest", default=None,
                   help="DOE manifest.csv (required for --feature-version 4)")
    ap.add_argument("--raw-dir", default="data/raw_v4",
                   help="raw FEniCSx solve npz dir ({run_id}.npz -> t/gradT/dTdt), v4 only")
    a = ap.parse_args()
    build(a.data_root, a.out_dir, a.workers, a.limit, a.feature_version,
         manifest_path=a.manifest, raw_dir=a.raw_dir)
