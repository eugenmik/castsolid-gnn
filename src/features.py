"""Assemble per-node physics features (v2) + target for one casting.

v2 vs v1: distance-to-cooling is GEODESIC (through the material); the degenerate
engineered ratios (d_cool_sq, d_cool_over_dsurf) are dropped; node positions are
returned for downstream graph construction. All features mm, invariant, from mesh
+ facet labels only.
"""
import numpy as np

from src.geometry import (triangle_areas, total_area, mesh_volume,
                          nearest_distance, boundary_node_mask,
                          edges_from_tets, geodesic_distance)
from src.targets import solidification_time, global_time

FEATURE_NAMES = [
    "geodesic_d_cool", "d_surf", "d_adiab",
    "is_cooling", "is_adiabatic", "is_interior",
    "volume_mm3", "surface_area_mm2", "cooling_area_frac",
]


def _node_features(mesh):
    """Compute the (N,9) v2 feature matrix + pos (mm) for a mesh. No target."""
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
    X = np.stack([cols[n] for n in FEATURE_NAMES], axis=1).astype(np.float32)
    return X, xyz.astype(np.float32)


def compute_features_inference(mesh) -> dict:
    X, pos = _node_features(mesh)
    return {"X": X, "pos": pos}


def compute_features(mesh, temps, times) -> dict:
    X, pos = _node_features(mesh)
    y = solidification_time(temps, times)
    return {"X": X, "y": y.astype(np.float32), "y_global": global_time(y), "pos": pos}
