import numpy as np
import pytest
from scipy.spatial import Delaunay

from src.msh_parse import parse_msh, Mesh
from src.geometry import edges_from_tets
from src.features import (
    _node_features, compute_features,
    FEATURE_NAMES, FEATURE_NAMES_V3, FEATURE_NAMES_V4,
    G_VECTOR_NAMES, T_REF, ALPHA_REF, MOLD_EFF_REF,
    EXO_DIFFUSION_LAMBDA_MM,
    alloy_g_vector, chvorinov_t_ref,
)

MOLD_EFFUSIVITY = 1250.0
CHILL_EFFUSIVITY = 11000.0
SLEEVE_MEF = 2.0
EXO_Q0 = 5000.0


def _base_spec(bc_config="none", exo_q0=0.0):
    return {
        "mold_effusivity": MOLD_EFFUSIVITY,
        "bc_config": bc_config,
        "chill_effusivity": CHILL_EFFUSIVITY,
        "sleeve_MEF": SLEEVE_MEF,
        "exo_q0": exo_q0,
        # alloy props (for g / t_ref -- only needed by compute_features tests)
        "k": 35.0, "rho": 7100.0, "cp": 500.0, "L_kJ_kg": 220.0,
        "TL": 1180.0, "TS": 1140.0, "Tpour": 1380.0,
        "C_material": 6.5e5,
    }


def _cube_with_bc_patch(n=5, a=20.0):
    """A Delaunay-tetrahedralized cube where the z=0 face is tagged
    'Adiabatic' (the BC-patch footprint for chills/sleeves) and every other
    hull face is tagged 'Cooling'. Interior points of the z=0 face (not on
    a cube edge) are touched ONLY by Adiabatic triangles -- a clean
    adiab_only region, unlike the tiny conftest mini_msh where every
    Adiabatic node is also a Cooling node."""
    lin = np.linspace(0.0, a, n)
    pts = np.array([[x, y, z] for x in lin for y in lin for z in lin], float)
    dt = Delaunay(pts)
    tets = dt.simplices.astype(np.int64)
    hull = dt.convex_hull.astype(np.int64)
    is_bottom = np.isclose(pts[hull, 2], 0.0).all(axis=1)
    adiab_tri = hull[is_bottom]
    cool_tri = hull[~is_bottom]
    mesh = Mesh(node_xyz=pts, tris={"Cooling": cool_tri, "Adiabatic": adiab_tri}, tets=tets)
    interior_bottom = (np.isclose(pts[:, 2], 0.0)
                       & (pts[:, 0] > 1e-9) & (pts[:, 0] < a - 1e-9)
                       & (pts[:, 1] > 1e-9) & (pts[:, 1] < a - 1e-9))
    on_cool = ~np.isclose(pts[:, 2], 0.0) | ~interior_bottom
    return mesh, interior_bottom


# --------------------------------------------------------------------- #
# Names / shape contract
# --------------------------------------------------------------------- #
def test_v4_names_extend_v3():
    assert FEATURE_NAMES_V4[:9] == FEATURE_NAMES
    assert FEATURE_NAMES_V4[:13] == FEATURE_NAMES_V3
    assert len(FEATURE_NAMES_V4) == 15
    assert FEATURE_NAMES_V4[13:] == ["boundary_extraction", "exothermic_input"]


def test_g_vector_names_and_constants():
    assert G_VECTOR_NAMES == ["Ste", "dTf_norm", "alpha_ratio", "superheat_norm", "mold_eff_norm"]
    assert T_REF == 1000.0
    assert ALPHA_REF == 1e-5
    assert MOLD_EFF_REF == 1250.0


# --------------------------------------------------------------------- #
# version==4 requires spec
# --------------------------------------------------------------------- #
def test_version4_without_spec_raises(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    with pytest.raises(ValueError):
        _node_features(mesh, version=4)


def test_compute_features_version4_without_spec_raises(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    N = mesh.node_xyz.shape[0]
    times = np.array([0.0, 10.0]); temps = np.full((2, N), 1200.0); temps[1] = 1100.0
    with pytest.raises(ValueError):
        compute_features(mesh, temps, times, version=4)


# --------------------------------------------------------------------- #
# Backward-compat: all-sand (bc_config="none", exo_q0=0)
# --------------------------------------------------------------------- #
def test_v4_backward_compat_first13_columns_match_v3(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    spec = _base_spec(bc_config="none", exo_q0=0.0)
    X4, pos4, e4 = _node_features(mesh, version=4, spec=spec)
    X3, pos3, e3 = _node_features(mesh, version=3)
    assert X4.shape[1] == 15
    assert np.array_equal(X4[:, :13], X3)
    assert np.array_equal(pos4, pos3) and np.array_equal(e4, e3)


def test_v4_backward_compat_boundary_extraction_and_exo(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    spec = _base_spec(bc_config="none", exo_q0=0.0)
    X4, _, _ = _node_features(mesh, version=4, spec=spec)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES_V4)}
    is_interior = X4[:, idx["is_interior"]].astype(bool)
    extraction = X4[:, idx["boundary_extraction"]]
    exo = X4[:, idx["exothermic_input"]]
    # all boundary nodes get the baseline mold effusivity, interior nodes 0
    assert np.allclose(extraction[~is_interior], MOLD_EFFUSIVITY)
    assert np.allclose(extraction[is_interior], 0.0)
    assert np.allclose(exo, 0.0)


def test_v4_backward_compat_on_cube_with_distinct_adiab_patch():
    # Same all-sand check, but on a mesh with a genuinely distinct Adiabatic
    # footprint (not overlapping Cooling), to make sure "none" really treats
    # the BC patch as ordinary mold (matches fenicsx_solver's bc_config="none"
    # branch: patch shares the SAME h_mold term).
    mesh, interior_bottom = _cube_with_bc_patch()
    spec = _base_spec(bc_config="none", exo_q0=0.0)
    X4, _, _ = _node_features(mesh, version=4, spec=spec)
    X3, _, _ = _node_features(mesh, version=3)
    assert np.array_equal(X4[:, :13], X3)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES_V4)}
    extraction = X4[:, idx["boundary_extraction"]]
    is_interior = X4[:, idx["is_interior"]].astype(bool)
    assert np.allclose(extraction[interior_bottom], MOLD_EFFUSIVITY)
    assert np.allclose(extraction[is_interior], 0.0)
    assert np.allclose(X4[:, idx["exothermic_input"]], 0.0)


# --------------------------------------------------------------------- #
# BC channel mapping: chills / insulating_sleeve / exo_sleeve
# --------------------------------------------------------------------- #
def test_v4_chills_mapping():
    mesh, interior_bottom = _cube_with_bc_patch()
    spec = _base_spec(bc_config="chills", exo_q0=0.0)
    X4, _, _ = _node_features(mesh, version=4, spec=spec)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES_V4)}
    extraction = X4[:, idx["boundary_extraction"]]
    is_interior = X4[:, idx["is_interior"]].astype(bool)
    assert np.allclose(extraction[interior_bottom], CHILL_EFFUSIVITY)
    assert np.allclose(extraction[is_interior], 0.0)
    assert np.allclose(X4[:, idx["exothermic_input"]], 0.0)
    # cooling-only nodes still get the mold baseline
    cool_only = ~is_interior & ~interior_bottom
    assert np.allclose(extraction[cool_only], MOLD_EFFUSIVITY)


def test_v4_insulating_sleeve_mapping():
    mesh, interior_bottom = _cube_with_bc_patch()
    spec = _base_spec(bc_config="insulating_sleeve", exo_q0=0.0)
    X4, _, _ = _node_features(mesh, version=4, spec=spec)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES_V4)}
    extraction = X4[:, idx["boundary_extraction"]]
    expected = MOLD_EFFUSIVITY / max(SLEEVE_MEF ** 2, 1.0)
    assert np.allclose(extraction[interior_bottom], expected)
    assert np.allclose(X4[:, idx["exothermic_input"]], 0.0)


def test_v4_exo_sleeve_mapping():
    # NEW contract (geodesic-diffused field, not a sparse indicator): peak
    # exo_q0/1e6 AT the patch nodes, decaying smoothly with geodesic distance
    # -- see test_exo_channel_is_diffused_field/test_exo_channel_zero_for_non_exo
    # below for the full shape/decay assertions. This test still checks that
    # boundary_extraction (unrelated column) keeps its sleeve-scaled mapping
    # and that the peak value at the patch is exactly exo_q0/1e6.
    mesh, interior_bottom = _cube_with_bc_patch()
    spec = _base_spec(bc_config="exo_sleeve", exo_q0=EXO_Q0)
    X4, _, _ = _node_features(mesh, version=4, spec=spec)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES_V4)}
    extraction = X4[:, idx["boundary_extraction"]]
    exo = X4[:, idx["exothermic_input"]]
    expected = MOLD_EFFUSIVITY / max(SLEEVE_MEF ** 2, 1.0)
    assert np.allclose(extraction[interior_bottom], expected)
    assert np.isclose(exo[interior_bottom].max(), EXO_Q0 / 1e6, rtol=1e-5)
    # diffused, not sparse: some nodes OUTSIDE the patch are nonzero too.
    outside_patch = ~interior_bottom
    assert (exo[outside_patch] > 0).any()


def test_exo_channel_is_diffused_field():
    mesh, interior_bottom = _cube_with_bc_patch()
    X, _, _ = _node_features(mesh, version=4, spec=_base_spec("exo_sleeve", exo_q0=EXO_Q0))
    ei = X[:, FEATURE_NAMES_V4.index("exothermic_input")]
    peak = EXO_Q0 / 1e6
    assert np.isclose(ei[interior_bottom].max(), peak, rtol=1e-5)   # peak at patch
    far = np.isclose(mesh.node_xyz[:, 2], 20.0)                     # opposite face
    assert (ei[far] < peak * np.exp(-20.0 / EXO_DIFFUSION_LAMBDA_MM) * 1.5).all()  # decayed far away
    assert (ei[far] > 0).any()                                      # but still nonzero (diffused)
    mid = np.isclose(mesh.node_xyz[:, 2], 10.0)
    assert ei[mid].max() < ei[interior_bottom].max()                # monotone trend
    assert ei[mid].max() > ei[far].max()


def test_exo_channel_zero_for_non_exo():
    mesh, _ = _cube_with_bc_patch()
    for bc in ("none", "chills", "insulating_sleeve"):
        X, _, _ = _node_features(mesh, version=4, spec=_base_spec(bc))
        assert (X[:, FEATURE_NAMES_V4.index("exothermic_input")] == 0).all()


def test_v4_sleeve_mef_floor_at_one():
    # max(MEF**2, 1) guards MEF < 1 from amplifying extraction above baseline
    mesh, interior_bottom = _cube_with_bc_patch()
    spec = _base_spec(bc_config="insulating_sleeve", exo_q0=0.0)
    spec["sleeve_MEF"] = 0.5
    X4, _, _ = _node_features(mesh, version=4, spec=spec)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES_V4)}
    extraction = X4[:, idx["boundary_extraction"]]
    assert np.allclose(extraction[interior_bottom], MOLD_EFFUSIVITY)  # divided by 1, not 0.25


# --------------------------------------------------------------------- #
# alloy_g_vector
# --------------------------------------------------------------------- #
def test_alloy_g_vector_hand_computed():
    props = {
        "k": 35.0, "rho": 7100.0, "cp": 500.0, "L_kJ_kg": 220.0,
        "TL": 1180.0, "TS": 1140.0, "Tpour": 1380.0,
        "mold_effusivity": MOLD_EFFUSIVITY,
    }
    g = alloy_g_vector(props)
    assert g.shape == (5,) and g.dtype == np.float64
    expected_Ste = 500.0 * (1380.0 - 1140.0) / (220.0 * 1000.0)
    expected_dTf_norm = (1180.0 - 1140.0) / T_REF
    expected_alpha_ratio = (35.0 / (7100.0 * 500.0)) / ALPHA_REF
    expected_superheat_norm = (1380.0 - 1180.0) / T_REF
    expected_mold_eff_norm = MOLD_EFFUSIVITY / MOLD_EFF_REF
    np.testing.assert_allclose(
        g, [expected_Ste, expected_dTf_norm, expected_alpha_ratio,
            expected_superheat_norm, expected_mold_eff_norm])


# --------------------------------------------------------------------- #
# chvorinov_t_ref
# --------------------------------------------------------------------- #
def test_chvorinov_t_ref_monotonic_in_modulus_squared():
    C = 6.5e5
    area = 100.0
    t_small = chvorinov_t_ref(volume_mm3=500.0, area_mm2=area, C_material=C)
    t_large = chvorinov_t_ref(volume_mm3=2000.0, area_mm2=area, C_material=C)
    assert t_large > t_small
    # exact formula
    assert np.isclose(t_small, C * (500.0 / area) ** 2)
    assert np.isclose(t_large, C * (2000.0 / area) ** 2)


# --------------------------------------------------------------------- #
# compute_features(version=4): g / t_ref / y_nondim
# --------------------------------------------------------------------- #
def test_compute_features_v4_adds_g_tref_ynondim(mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    N = mesh.node_xyz.shape[0]
    times = np.array([0.0, 10.0]); temps = np.full((2, N), 1200.0); temps[1] = 1100.0
    spec = _base_spec(bc_config="none", exo_q0=0.0)
    out = compute_features(mesh, temps, times, version=4, spec=spec)
    assert out["X"].shape == (N, 15)
    assert "y" in out and "g" in out and "t_ref" in out and "y_nondim" in out
    assert out["g"].shape == (5,) and out["g"].dtype == np.float64
    expected_g = alloy_g_vector(spec)
    np.testing.assert_allclose(out["g"], expected_g)

    volume = out["X"][0, FEATURE_NAMES.index("volume_mm3")]
    area = out["X"][0, FEATURE_NAMES.index("surface_area_mm2")]
    expected_t_ref = chvorinov_t_ref(volume, area, spec["C_material"])
    assert np.isclose(out["t_ref"], expected_t_ref)

    expected_y_nondim = out["y"] / max(expected_t_ref, 1e-9)
    np.testing.assert_allclose(out["y_nondim"], expected_y_nondim, rtol=1e-5)
