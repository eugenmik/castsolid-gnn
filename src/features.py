"""Node features and the alloy/mold conditioning vector.

Every feature comes from the mesh and its surface tags ("Cooling", optionally
"Adiabatic"), in millimetres. Nothing depends on node numbering, file names or
absolute position.
"""
import numpy as np

from src.geometry import (total_area, mesh_volume, nearest_distance, boundary_node_mask,
                          edges_from_tets, geodesic_distance, local_modulus)

FEATURE_NAMES = [
    "geodesic_d_cool", "d_surf", "d_adiab",
    "is_cooling", "is_adiabatic", "is_interior",
    "volume_mm3", "surface_area_mm2", "cooling_area_frac",
    "geodesic_d_adiab",
    "local_modulus_r5", "local_modulus_r15", "local_modulus_r40",
    "boundary_extraction", "boundary_heat_input",
]
LOCAL_MODULUS_RADII = (5.0, 15.0, 40.0)

# Alloy/mold conditioning vector (raw, not standardized).
G_VECTOR_NAMES = ["Ste", "dTf_norm", "alpha_ratio", "superheat_norm", "mold_eff_norm"]
T_REF = 1000.0          # deg C, normalizes temperature differences
ALPHA_REF = 1e-5        # m^2/s, typical alloy thermal diffusivity
MOLD_EFF_REF = 1250.0   # W*s^0.5/(m^2*K), reference mold effusivity


def alloy_g_vector(props: dict) -> np.ndarray:
    """Five dimensionless numbers describing alloy and mold.

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


def node_features(mesh, mold_effusivity: float) -> dict:
    """The (N, 15) float32 feature matrix `X`, node positions `pos` (mm) and
    the unique tetrahedral mesh edges `mesh_edges` for one casting."""
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

    g_adiab = geodesic_distance(N, edges, edge_len, np.flatnonzero(adiab_mask))
    g_adiab = np.where(np.isinf(g_adiab), d_adiab, g_adiab)
    lm = local_modulus(xyz, mesh.tets, cool_tri, edges, edge_len, LOCAL_MODULUS_RADII)

    # Boundary channels. Surface nodes carry the mold effusivity (how fast the
    # mold draws heat); interior nodes carry 0. The heat-input channel is for
    # heat sources on the boundary, and a plain sand-mold surface has none.
    boundary_extraction = np.zeros(N, dtype=np.float32)
    boundary_extraction[cool_mask] = mold_effusivity
    boundary_extraction[adiab_only] = mold_effusivity
    boundary_heat_input = np.zeros(N, dtype=np.float32)

    cols = {
        "geodesic_d_cool": g_cool, "d_surf": d_surf, "d_adiab": d_adiab,
        "is_cooling": cool_mask.astype(np.float32),
        "is_adiabatic": adiab_only.astype(np.float32),
        "is_interior": interior_mask.astype(np.float32),
        "volume_mm3": np.full(N, volume),
        "surface_area_mm2": np.full(N, area_total),
        "cooling_area_frac": np.full(N, cool_frac),
        "geodesic_d_adiab": g_adiab,
        "local_modulus_r5": lm[:, 0],
        "local_modulus_r15": lm[:, 1],
        "local_modulus_r40": lm[:, 2],
        "boundary_extraction": boundary_extraction,
        "boundary_heat_input": boundary_heat_input,
    }
    X = np.stack([cols[n] for n in FEATURE_NAMES], axis=1).astype(np.float32)
    return {"X": X, "pos": xyz.astype(np.float32), "mesh_edges": edges.astype(np.int64)}
