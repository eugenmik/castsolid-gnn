"""Assemble per-node physics features (v2) + target for one casting.

v2 vs v1: distance-to-cooling is GEODESIC (through the material); the degenerate
engineered ratios (d_cool_sq, d_cool_over_dsurf) are dropped; node positions are
returned for downstream graph construction. All features mm, invariant, from mesh
+ facet labels only.
"""
import logging

import numpy as np

from src.geometry import (triangle_areas, total_area, mesh_volume,
                          nearest_distance, boundary_node_mask,
                          edges_from_tets, geodesic_distance, local_modulus,
                          geodesic_diameter_two_sweep)
from src.targets import solidification_time, global_time

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "geodesic_d_cool", "d_surf", "d_adiab",
    "is_cooling", "is_adiabatic", "is_interior",
    "volume_mm3", "surface_area_mm2", "cooling_area_frac",
]

FEATURE_NAMES_V3 = FEATURE_NAMES + [
    "geodesic_d_adiab",
    "local_modulus_r5", "local_modulus_r15", "local_modulus_r40",
]
LOCAL_MODULUS_RADII = (5.0, 15.0, 40.0)

# v4: v3's 13 columns UNCHANGED (X[:, :9] == v2, X[:, :13] == v3 -- the HGB
# baseline + backward-compat contract), with two continuous BC channels
# APPENDED at the end (replacing is_cooling/is_adiabatic/is_interior as MODEL
# input signal, though those binary tags stay in the first 9 columns as-is).
FEATURE_NAMES_V4 = FEATURE_NAMES_V3 + ["boundary_extraction", "exothermic_input"]

# v5: v4's 15 columns UNCHANGED, with five multi-scale part-normalized BC
# influence channels APPENDED -- exo_mid/exo_long extend the v4
# exothermic_input ("exo_short") reach to 0.15*D_geo/0.40*D_geo, and
# chill_short/chill_mid/chill_long give chills the same three-scale
# treatment (chills previously only had the surface-local
# boundary_extraction). See BC_REACH_MID_FRAC/BC_REACH_LONG_FRAC above.
FEATURE_NAMES_V5 = FEATURE_NAMES_V4 + [
    "exo_mid", "exo_long", "chill_short", "chill_mid", "chill_long",
]

# Geodesic-diffusion length scale (mm) for the exothermic_input channel: the
# sleeve's calibrated heat flux q0 decays as exp(-d_geo/lambda) away from the
# adiab_only (BC-patch) source nodes, instead of being painted only on those
# 1-2% of nodes as a sparse indicator (see docs/superpowers exo-fix design).
# This is also reused, unchanged, as the "short" reach scale for the v5
# chill_short channel below (bc-reach spec: same short-range mesh-scale
# decay length, not part-size-normalized).
EXO_DIFFUSION_LAMBDA_MM = 15.0

# v5 "mid"/"long" BC-influence reach fractions -- PART-NORMALIZED (of
# D_geo = the part's GEODESIC DIAMETER: a two-sweep-Dijkstra estimate over
# the mesh-edge graph -- `geodesic_diameter_two_sweep` -- patch-INDEPENDENT
# and through-metal), unlike the fixed EXO_DIFFUSION_LAMBDA_MM above.
# CORRECTION (2026-08-02, Task-1 review round 1, F1/F3): D_geo was
# originally (wrongly) `max(geodesic_d_cool)` -- that is max WALL DEPTH
# (~35-94mm on real parts, and BC-PATCH-DEPENDENT since it depends on which
# facets are tagged Cooling), not part size; it made L_long 6-17x too
# short (shorter even than the v4 fixed 15mm scale on some rows), inverting
# the intended short<mid<long ladder. The geodesic diameter is a single
# per-part constant, identical across every BC row of the same geometry.
# Motivation (bc-reach investigation): the fixed 15mm short scale decays to
# ~0 by ~140mm, but a real exo sleeve's influence on a 600mm-class casting
# extends past 400mm -- normalizing by the part's own geodesic diameter
# lets the encoding transfer across part sizes instead of a single global
# mm constant under/over-reaching.
BC_REACH_MID_FRAC = 0.15
BC_REACH_LONG_FRAC = 0.40

# Global dimensionless alloy+mould vector (broadcast to every node by the
# cache builder / FT2b; computed here RAW -- not standardized).
G_VECTOR_NAMES = ["Ste", "dTf_norm", "alpha_ratio", "superheat_norm", "mold_eff_norm"]
T_REF = 1000.0          # deg C, normalizes temperature-difference channels
ALPHA_REF = 1e-5        # m^2/s, ballpark alloy thermal diffusivity
MOLD_EFF_REF = 1250.0   # W*s^0.5/(m^2*K), green-sand mold effusivity baseline


def alloy_g_vector(props: dict) -> np.ndarray:
    """Global dimensionless alloy+mould vector `g` (raw, not standardized).

    `props` keys: k, rho, cp, L_kJ_kg, TL, TS, Tpour, mold_effusivity."""
    k = float(props["k"]); rho = float(props["rho"]); cp = float(props["cp"])
    L = float(props["L_kJ_kg"]); TL = float(props["TL"]); TS = float(props["TS"])
    Tpour = float(props["Tpour"]); b_mold = float(props["mold_effusivity"])

    Ste = cp * (Tpour - TS) / (L * 1000.0)
    dTf_norm = (TL - TS) / T_REF
    alpha_ratio = (k / (rho * cp)) / ALPHA_REF
    superheat_norm = (Tpour - TL) / T_REF
    mold_eff_norm = b_mold / MOLD_EFF_REF
    return np.array([Ste, dTf_norm, alpha_ratio, superheat_norm, mold_eff_norm],
                     dtype=np.float64)


def chvorinov_t_ref(volume_mm3, area_mm2, C_material) -> float:
    """Chvorinov total-solidification-time estimate: C * (V/A)^2."""
    return float(C_material) * (float(volume_mm3) / float(area_mm2)) ** 2


def geodesic_diffuse(source_values: np.ndarray, length_mm: float, N: int,
                     edges: np.ndarray, edge_len: np.ndarray) -> np.ndarray:
    """Multi-source geodesic exponential-decay diffusion, factored out of
    the original inline `exothermic_input` construction.

    For every node: the max over distinct nonzero source magnitudes of
    (magnitude * exp(-geodesic distance to the nearest node carrying that
    magnitude / length_mm)). Distinct magnitudes are grouped and Dijkstra'd
    independently, then combined by elementwise max -- for the single
    uniform-magnitude source group this codebase actually uses (every BC
    source node carries the same scalar peak), this reduces EXACTLY to
    `peak * exp(-d_geo/length_mm)` via one ordinary multi-source Dijkstra
    call, byte-identical to the original `exothermic_input` formula --
    PROVIDED that peak is non-negative. The `out = np.maximum(out, ...)`
    combine is an implicit >= 0 clip: a genuinely negative source magnitude
    would get clamped toward 0 instead of decaying to a negative value (a
    node's own zero-init `out` beats any negative contribution). No caller
    in this codebase produces a negative peak today (`exo_q0/1e6` and
    `(chill_eff-mold_eff)/mold_eff` are both >= 0 in every resolved spec),
    so this is currently unreachable, not a live bug -- documented so a
    future negative-magnitude source doesn't rediscover it the hard way.

    Returns all-zeros WITHOUT invoking the Dijkstra solver when there are
    no nonzero sources or `length_mm <= 0` (cheap all-sand / degenerate
    path -- `np.exp(-inf) == 0.0` would otherwise handle disconnected mesh
    components fine, but a zero/negative length is not physically
    meaningful and must not divide-by-zero).
    """
    source_values = np.asarray(source_values, dtype=np.float64)
    src_idx = np.flatnonzero(source_values != 0.0)
    if src_idx.size == 0 or length_mm <= 0:
        return np.zeros(N, dtype=np.float32)
    out = np.zeros(N, dtype=np.float64)
    for val in np.unique(source_values[src_idx]):
        grp = src_idx[source_values[src_idx] == val]
        d = geodesic_distance(N, edges, edge_len, grp)
        # np.exp(-inf) == 0.0 handles disconnected components
        out = np.maximum(out, val * np.exp(-d / length_mm))
    return out.astype(np.float32)


def compute_bc_reach_channels(N: int, edges: np.ndarray, edge_len: np.ndarray,
                              adiab_only: np.ndarray, bc_config: str,
                              mold_eff: float, chill_eff: float,
                              exo_q0: float) -> np.ndarray:
    """The five v5 multi-scale part-normalized BC-influence channels:
    `[exo_mid, exo_long, chill_short, chill_mid, chill_long]`, shape (N, 5)
    float32.

    `D_geo` (the part-normalizing length) is the part's GEODESIC DIAMETER --
    `geodesic_diameter_two_sweep(N, edges, edge_len, start=0)`, a two-sweep
    Dijkstra estimate over the SAME mesh-edge graph the other geodesic
    features use. This is patch-INDEPENDENT (unlike `geodesic_d_cool`,
    whose max is wall DEPTH, not part size -- see the CORRECTION note above
    `BC_REACH_MID_FRAC`) and identical across every BC row of the same
    geometry. exo_mid/exo_long reuse the exo_sleeve's `adiab_only` source
    patch and peak (`exo_q0/1e6`, matching `exothermic_input`'s "exo_short"
    scale) at `BC_REACH_MID_FRAC`/`BC_REACH_LONG_FRAC` * D_geo.
    chill_short/chill_mid/chill_long give the chills bc_config the same
    three-scale treatment, sourced at the chill contact nodes (`adiab_only`)
    with peak magnitude `(chill_eff - mold_eff) / mold_eff` -- chill_short
    reuses the fixed `EXO_DIFFUSION_LAMBDA_MM` short scale, chill_mid/
    chill_long use the same part-normalized fractions as the exo channels.

    D_geo (and therefore the two extra Dijkstra sweeps) is computed ONLY
    when `bc_config` is `"exo_sleeve"` or `"chills"` -- an all-sand
    (`"none"`) or `"insulating_sleeve"` part returns all-zeros without any
    solver call, same contract as `exothermic_input`/`geodesic_diffuse`. A
    degenerate/disconnected mesh (D_geo <= 0) logs a warning and leaves the
    mid/long channels at zero (short-scale channels are unaffected, since
    they don't depend on D_geo).
    """
    out = np.zeros((N, 5), dtype=np.float32)
    if bc_config not in ("exo_sleeve", "chills"):
        return out

    d_geo = geodesic_diameter_two_sweep(N, edges, edge_len, start=0)
    if d_geo <= 0:
        logger.warning(
            "compute_bc_reach_channels: D_geo (geodesic diameter) <= 0 for "
            "bc_config=%r -- degenerate/disconnected mesh, zeroing the "
            "part-normalized mid/long BC-reach channels", bc_config)
    lambda_mid = BC_REACH_MID_FRAC * d_geo
    lambda_long = BC_REACH_LONG_FRAC * d_geo

    if bc_config == "exo_sleeve":
        src = np.zeros(N, dtype=np.float64)
        src[adiab_only] = exo_q0 / 1e6
        out[:, 0] = geodesic_diffuse(src, lambda_mid, N, edges, edge_len)
        out[:, 1] = geodesic_diffuse(src, lambda_long, N, edges, edge_len)

    if bc_config == "chills":
        chill_val = (chill_eff - mold_eff) / mold_eff if mold_eff != 0 else 0.0
        src = np.zeros(N, dtype=np.float64)
        src[adiab_only] = chill_val
        out[:, 2] = geodesic_diffuse(src, EXO_DIFFUSION_LAMBDA_MM, N, edges, edge_len)
        out[:, 3] = geodesic_diffuse(src, lambda_mid, N, edges, edge_len)
        out[:, 4] = geodesic_diffuse(src, lambda_long, N, edges, edge_len)

    return out


def _node_features(mesh, version=2, spec=None):
    """Compute the (N,9/13/15/20) feature matrix (v2/v3/v4/v5) + pos (mm) +
    mesh edges for a mesh. No target."""
    xyz = mesh.node_xyz
    N = xyz.shape[0]
    cool_tri = mesh.tris.get("Cooling", np.empty((0, 3), np.int64))
    adiab_tri = mesh.tris.get("Adiabatic", np.empty((0, 3), np.int64))

    cool_mask = boundary_node_mask(N, cool_tri)
    adiab_mask = boundary_node_mask(N, adiab_tri)
    adiab_only = adiab_mask & ~cool_mask
    surf_mask = cool_mask | adiab_mask
    interior_mask = ~surf_mask

    bbox = xyz.max(0) - xyz.min(0)
    bbox_diag = float(np.linalg.norm(bbox))

    edges = edges_from_tets(mesh.tets)
    edge_len = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1) if edges.shape[0] else np.empty(0)
    src = np.flatnonzero(cool_mask)
    g_cool = geodesic_distance(N, edges, edge_len, src)
    eucl_cool = nearest_distance(xyz, xyz[cool_mask])
    eucl_cool = np.where(np.isinf(eucl_cool), bbox_diag, eucl_cool)
    g_cool = np.where(np.isinf(g_cool), eucl_cool, g_cool)

    d_surf = nearest_distance(xyz, xyz[surf_mask])
    d_adiab = nearest_distance(xyz, xyz[adiab_mask])
    d_surf = np.where(np.isinf(d_surf), bbox_diag, d_surf)
    d_adiab = np.where(np.isinf(d_adiab), bbox_diag, d_adiab)

    area_cool = total_area(xyz, cool_tri)
    area_total = area_cool + total_area(xyz, adiab_tri)
    cool_frac = area_cool / area_total if area_total > 0 else 0.0
    volume = mesh_volume(xyz, mesh.tets)

    cols = {
        "geodesic_d_cool": g_cool, "d_surf": d_surf, "d_adiab": d_adiab,
        "is_cooling": cool_mask.astype(np.float32),
        "is_adiabatic": adiab_only.astype(np.float32),
        "is_interior": interior_mask.astype(np.float32),
        "volume_mm3": np.full(N, volume),
        "surface_area_mm2": np.full(N, area_total),
        "cooling_area_frac": np.full(N, cool_frac),
    }
    if version >= 3:
        g_adiab = geodesic_distance(N, edges, edge_len, np.flatnonzero(adiab_mask))
        g_adiab = np.where(np.isinf(g_adiab), d_adiab, g_adiab)
        lm = local_modulus(xyz, mesh.tets, cool_tri, edges, edge_len, LOCAL_MODULUS_RADII)
        cols.update({
            "geodesic_d_adiab": g_adiab,
            "local_modulus_r5": lm[:, 0],
            "local_modulus_r15": lm[:, 1],
            "local_modulus_r40": lm[:, 2],
        })
    if version >= 4:
        if spec is None:
            raise ValueError("_node_features(version=4) requires a `spec` dict "
                              "(BC info: mold_effusivity, bc_config, "
                              "chill_effusivity, sleeve_MEF, exo_q0)")
        mold_eff = float(spec["mold_effusivity"])
        bc_config = spec["bc_config"]
        chill_eff = float(spec.get("chill_effusivity", 0.0))
        sleeve_mef = float(spec.get("sleeve_MEF", 0.0))
        exo_q0 = float(spec.get("exo_q0", 0.0))

        # Mirrors tools/doe/fenicsx_solver.py's tag convention: tag "Cooling"
        # is always the ordinary sand-mold Robin term; tag "Adiabatic" is the
        # BC-patch footprint, reinterpreted per bc_config. Nodes shared
        # between the two triangle groups follow the same cooling-takes-
        # priority convention as `is_adiabatic` (adiab_only).
        boundary_extraction = np.zeros(N, dtype=np.float32)
        boundary_extraction[cool_mask] = mold_eff
        if bc_config == "chills":
            adiab_val = chill_eff
        elif bc_config in ("insulating_sleeve", "exo_sleeve"):
            adiab_val = mold_eff / max(sleeve_mef ** 2, 1.0)
        else:  # "none" -- BC patch treated as ordinary mold (matches solver)
            adiab_val = mold_eff
        boundary_extraction[adiab_only] = adiab_val

        exothermic_input = np.zeros(N, dtype=np.float32)
        if bc_config == "exo_sleeve":
            exo_src_vals = np.zeros(N, dtype=np.float64)
            exo_src_vals[adiab_only] = exo_q0 / 1e6
            exothermic_input = geodesic_diffuse(exo_src_vals, EXO_DIFFUSION_LAMBDA_MM,
                                                N, edges, edge_len)

        cols.update({
            "boundary_extraction": boundary_extraction,
            "exothermic_input": exothermic_input,
        })

        if version >= 5:
            bc_reach = compute_bc_reach_channels(
                N, edges, edge_len, adiab_only, bc_config,
                mold_eff, chill_eff, exo_q0)
            cols.update({
                "exo_mid": bc_reach[:, 0], "exo_long": bc_reach[:, 1],
                "chill_short": bc_reach[:, 2], "chill_mid": bc_reach[:, 3],
                "chill_long": bc_reach[:, 4],
            })

    names = (FEATURE_NAMES_V5 if version >= 5 else
             FEATURE_NAMES_V4 if version >= 4 else
             FEATURE_NAMES_V3 if version >= 3 else FEATURE_NAMES)
    X = np.stack([cols[n] for n in names], axis=1).astype(np.float32)
    return X, xyz.astype(np.float32), edges.astype(np.int64)


def compute_features_inference(mesh, version=2, spec=None) -> dict:
    X, pos, edges = _node_features(mesh, version, spec=spec)
    return {"X": X, "pos": pos, "mesh_edges": edges}


def compute_features(mesh, temps, times, version=2, spec=None) -> dict:
    X, pos, edges = _node_features(mesh, version, spec=spec)
    y = solidification_time(temps, times)
    out = {"X": X, "y": y.astype(np.float32), "y_global": global_time(y),
           "pos": pos, "mesh_edges": edges}
    if version >= 4:
        g = alloy_g_vector(spec)
        volume = float(X[0, FEATURE_NAMES.index("volume_mm3")])
        area = float(X[0, FEATURE_NAMES.index("surface_area_mm2")])
        t_ref = chvorinov_t_ref(volume, area, spec["C_material"])
        y_nondim = y.astype(np.float64) / max(t_ref, 1e-9)
        out.update({"g": g, "t_ref": t_ref, "y_nondim": y_nondim.astype(np.float32)})
    return out
