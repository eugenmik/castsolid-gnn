"""Task 3: film-aware v4 inference path in predict_field/predict_casting.

Real tiny in-process FiLM SolidGNN + a stub v2-total GBM (mirrors
tests/test_evaluate_ft2b.py / tests/test_select_ckpt_ft2b.py's fixture
style) exercise the new v4 (film) branch of `predict_field`, while a
plain SP1-shaped (non-film) call proves the OLD code path is untouched.

Task 8: prep-aligned inference meshing -- `predict_casting`'s meshing
stage routes through `tools.doe.prep_parts.heal_and_mesh` (largest-solid
+ OCC heal + density search, same pipeline as the training/prep dataset)
by default (`mesh_mode="prep"`), with `mesh_mode="raw"` (or an explicit
`elem_size`) as an escape hatch to the old `src.step_mesh.mesh_step`
path. The routing tests below monkeypatch both meshing functions as
recorders (plus `load_hybrid`/`predict_field`) -- no real gmsh/model
checkpoints are touched.

Part B Task 4: `predict_casting(bodies=...)` wiring -- `validate_bodies` ->
mesh via the cache with `keep_solid_id`/`bodies_key` -> `tag_contact_patches`
(cached as a `post_mesh` hook) -> the film spec/gbm bc_spec carry the
resolved `BcPlan`. All of it is exercised with monkeypatched recorders
(`validate_bodies`/`tag_contact_patches`/`get_or_mesh`/`predict_field`) --
no real gmsh/model checkpoints are touched here either.
"""
import json
import logging
import os.path as osp
import shutil

import numpy as np
import pytest
import torch
from scipy.spatial import Delaunay
from sklearn.ensemble import HistGradientBoostingRegressor

import src.predict_casting as pc_mod
from src.features import FEATURE_NAMES_V5
from src.inference_props import resolve_props
from src.model_gnn import SolidGNN
from src.msh_parse import Mesh, parse_msh
from src.predict_casting import predict_field

N_COLS_V4 = 15


def _all_cooling_cube(n=4, a=20.0):
    """A Delaunay-tetrahedralized cube with every hull face tagged
    'Cooling' (no Adiabatic patch) -- mirrors test_features_v4.py's
    `_cube_with_bc_patch` mesh-construction style."""
    lin = np.linspace(0.0, a, n)
    pts = np.array([[x, y, z] for x in lin for y in lin for z in lin], float)
    dt = Delaunay(pts)
    tets = dt.simplices.astype(np.int64)
    hull = dt.convex_hull.astype(np.int64)
    return Mesh(node_xyz=pts, tris={"Cooling": hull, "Adiabatic": np.empty((0, 3), np.int64)},
                tets=tets)


def _make_film_gnn(seed=0):
    """A real tiny film SolidGNN with a NON-zero-init FiLM (the last FiLM
    layer is zero-init by construction -- see model_gnn.FiLM -- so g would
    be a no-op unless we perturb it here, exactly as the brief requires)."""
    torch.manual_seed(seed)
    model = SolidGNN(node_in=N_COLS_V4, edge_in=5, hidden=16, n_blocks=2,
                     two_head=True, film=True, g_dim=5)
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for film in model.films:
            w = rng.normal(scale=2.0, size=tuple(film.net[-1].weight.shape)).astype(np.float32)
            b = rng.normal(scale=2.0, size=tuple(film.net[-1].bias.shape)).astype(np.float32)
            film.net[-1].weight.copy_(torch.tensor(w))
            film.net[-1].bias.copy_(torch.tensor(b))
    model.eval()
    return model


def _norm15():
    # y_log_mean=3.0 (not 0.0): an untrained model's raw per-node output has
    # only a tiny spread (~1e-2, this architecture's "dev" term); centering
    # log1p(seconds) near 0 would expm1-clip that whole spread to exactly
    # 0 for every node (seen in practice), masking any FiLM/g effect. A
    # moderate y_log_mean keeps the (tiny but real) per-node spread inside
    # expm1's non-degenerate region so alloy-driven differences survive.
    return {"feat_mean": np.zeros(N_COLS_V4, dtype=np.float32),
            "feat_std": np.ones(N_COLS_V4, dtype=np.float32),
            "g_mean": np.zeros(5), "g_std": np.ones(5),
            "y_log_mean": np.float64(3.0), "y_log_std": np.float64(1.0)}


def _film_cfg():
    return {"hidden": 16, "n_blocks": 2, "k": 4, "graph": "union",
            "feature_version": 4, "film": True, "g_dim": 5, "two_head": True,
            "node_in": N_COLS_V4, "edge_in": 5}


class _StubHGB:
    """A v2-total GBM stand-in: `.predict` ignores its input and always
    returns the SAME log1p(total_seconds) row -- enough to pin the
    max-anchor recomposition exactly, without a real gbm_ft2b fit."""

    def __init__(self, n_features_in_, total_seconds=50.0):
        self.n_features_in_ = n_features_in_
        self._log_pred = float(np.log1p(total_seconds))

    def predict(self, X):
        return np.full(X.shape[0], self._log_pred)


# --------------------------------------------------------------------------- #
# film (v4) path
# --------------------------------------------------------------------------- #
def test_film_path_max_anchors_to_stub_gbm_total():
    mesh = _all_cooling_cube()
    gnn = _make_film_gnn(seed=0)
    hgb = _StubHGB(n_features_in_=16, total_seconds=50.0)
    props = resolve_props()

    out, feat = predict_field(mesh, gnn, hgb, _norm15(), cfg=_film_cfg(),
                              device="cpu", props=props)

    assert out.shape == (mesh.node_xyz.shape[0],)
    assert np.isfinite(out).all() and (out >= 0).all()
    # Task 7: hybrid_field_max_anchor's default anchor is a high quantile
    # (q=0.995), not the exact max, so `out.max()` need only be AT LEAST the
    # stub total (the anchor node itself, and any node above it, gets
    # shifted to >= total) -- it is no longer pinned to exactly 50.0. The
    # excess is bounded by the top-quantile spread of the field, which is
    # small here.
    assert out.max() >= 50.0 - 1e-6
    assert out.max() <= 50.0 + 1.0
    assert feat["X"].shape == (mesh.node_xyz.shape[0], N_COLS_V4)


def test_film_path_changing_alloy_changes_field():
    mesh = _all_cooling_cube()
    gnn = _make_film_gnn(seed=0)
    hgb = _StubHGB(n_features_in_=16, total_seconds=50.0)

    props_a = resolve_props(alloy="EN-GJL-250")
    props_b = resolve_props(alloy="EN-GJS-500-7")
    out_a, _ = predict_field(mesh, gnn, hgb, _norm15(), cfg=_film_cfg(),
                             device="cpu", props=props_a)
    out_b, _ = predict_field(mesh, gnn, hgb, _norm15(), cfg=_film_cfg(),
                             device="cpu", props=props_b)

    # Both anchor to the SAME stub total (50.0) -- only the field SHAPE
    # (the FiLM-conditioned deviations from that quantile anchor) can
    # differ. Per Task 7, the default quantile anchor means `max() >= total`
    # rather than `== total` exactly (see the assertion above this fixture).
    assert out_a.max() >= 50.0 - 1e-6
    assert out_b.max() >= 50.0 - 1e-6
    assert not np.allclose(out_a, out_b)


def test_film_path_with_bc_plan_uses_plan_feature_spec_and_gbm_bc_spec(monkeypatch):
    """Part B Task 4: `predict_field(bc_plan=...)` feeds the film branch
    `plan.feature_spec(...)` (the chill/sleeve spec) instead of the
    all-sand literal, and `plan.gbm_bc_spec()` into `gbm_v2_total_seconds`.
    Spies on the module-level `compute_features_inference` / `gbm_v2_total_seconds`
    (both real, just observed) to capture exactly what spec/bc_spec were
    built, without needing a real multi-body mesh."""
    from src.assembly import BcPlan

    mesh = _all_cooling_cube()
    gnn = _make_film_gnn(seed=0)
    hgb = _StubHGB(n_features_in_=16, total_seconds=50.0)
    props = resolve_props()

    plan = BcPlan(bc_config="chills", casting_solid_id=1, role_solid_ids=[2],
                  n_chills=2, chill_effusivity=9876.5)

    captured_spec = {}
    real_compute = pc_mod.compute_features_inference

    def spy_compute(mesh_, version=2, spec=None):
        captured_spec["spec"] = spec
        return real_compute(mesh_, version=version, spec=spec)

    captured_bc_spec = {}
    real_gbm = pc_mod.gbm_v2_total_seconds

    def spy_gbm(hgb_model, props_, v_over_a, bc_spec=None):
        captured_bc_spec["bc_spec"] = bc_spec
        return real_gbm(hgb_model, props_, v_over_a, bc_spec=bc_spec)

    monkeypatch.setattr(pc_mod, "compute_features_inference", spy_compute)
    monkeypatch.setattr(pc_mod, "gbm_v2_total_seconds", spy_gbm)

    predict_field(mesh, gnn, hgb, _norm15(), cfg=_film_cfg(), device="cpu",
                 props=props, bc_plan=plan)

    assert captured_spec["spec"]["bc_config"] == "chills"
    assert captured_spec["spec"]["chill_effusivity"] == pytest.approx(9876.5)
    assert captured_spec["spec"]["mold_effusivity"] == pytest.approx(props["mold_effusivity"])
    assert captured_bc_spec["bc_spec"]["bc_config"] == "chills"
    assert captured_bc_spec["bc_spec"]["n_chills"] == 2


def test_film_path_without_bc_plan_spec_stays_all_sand_literal(monkeypatch):
    """`bc_plan=None` (the default) -- byte-identical to the pre-Part-B
    behavior: the all-sand literal spec, and `bc_spec=None` into the GBM
    call (its own documented default, "plain sand mold")."""
    mesh = _all_cooling_cube()
    gnn = _make_film_gnn(seed=0)
    hgb = _StubHGB(n_features_in_=16, total_seconds=50.0)
    props = resolve_props()

    captured_spec = {}
    real_compute = pc_mod.compute_features_inference

    def spy_compute(mesh_, version=2, spec=None):
        captured_spec["spec"] = spec
        return real_compute(mesh_, version=version, spec=spec)

    captured_bc_spec = {}
    real_gbm = pc_mod.gbm_v2_total_seconds

    def spy_gbm(hgb_model, props_, v_over_a, bc_spec=None):
        captured_bc_spec["bc_spec"] = bc_spec
        return real_gbm(hgb_model, props_, v_over_a, bc_spec=bc_spec)

    monkeypatch.setattr(pc_mod, "compute_features_inference", spy_compute)
    monkeypatch.setattr(pc_mod, "gbm_v2_total_seconds", spy_gbm)

    predict_field(mesh, gnn, hgb, _norm15(), cfg=_film_cfg(), device="cpu", props=props)

    assert captured_spec["spec"] == {"mold_effusivity": props["mold_effusivity"],
                                     "bc_config": "none", "chill_effusivity": 0.0,
                                     "sleeve_MEF": 0.0, "exo_q0": 0.0}
    assert captured_bc_spec["bc_spec"] is None


def test_film_ckpt_with_wrong_dim_hgb_raises():
    mesh = _all_cooling_cube()
    gnn = _make_film_gnn(seed=0)
    hgb = _StubHGB(n_features_in_=9, total_seconds=50.0)  # SP1-shaped baseline, wrong for v4
    props = resolve_props()

    with pytest.raises(ValueError):
        predict_field(mesh, gnn, hgb, _norm15(), cfg=_film_cfg(), device="cpu", props=props)


def test_film_cfg_without_props_raises():
    mesh = _all_cooling_cube()
    gnn = _make_film_gnn(seed=0)
    hgb = _StubHGB(n_features_in_=16, total_seconds=50.0)

    with pytest.raises(ValueError):
        predict_field(mesh, gnn, hgb, _norm15(), cfg=_film_cfg(), device="cpu", props=None)


# --------------------------------------------------------------------------- #
# Task 4 (bc-reach): cfg-driven v5 (20-column) feature building at inference.
# Mirrors the v4 fixtures above (_make_film_gnn/_norm15/_film_cfg) at
# node_in=20/feature_version=5 -- `predict_field`'s film branch must read
# `cfg.get("feature_version", 4)` (the SAME pattern the non-film branch
# already uses, predict_casting.py:354) instead of hardcoding `version=4`,
# so a v5 (bc-reach) checkpoint's cfg drives `compute_features_inference`
# into the 20-column builder. `cfg.feature_version` absent or ==4 must stay
# byte-identical to every test above (unmodified).
# --------------------------------------------------------------------------- #
N_COLS_V5 = 20


def _cube_with_bc_patch(n=5, a=20.0):
    """Same fixture as tests/test_features_v5.py: a Delaunay cube with the
    z=0 face tagged 'Adiabatic' (BC patch) and every other hull face
    'Cooling' -- needed (unlike `_all_cooling_cube`) so a bc_plan's
    chill/exo source patch has somewhere to sit."""
    lin = np.linspace(0.0, a, n)
    pts = np.array([[x, y, z] for x in lin for y in lin for z in lin], float)
    dt = Delaunay(pts)
    tets = dt.simplices.astype(np.int64)
    hull = dt.convex_hull.astype(np.int64)
    is_bottom = np.isclose(pts[hull, 2], 0.0).all(axis=1)
    adiab_tri = hull[is_bottom]
    cool_tri = hull[~is_bottom]
    return Mesh(node_xyz=pts, tris={"Cooling": cool_tri, "Adiabatic": adiab_tri}, tets=tets)


def _make_film_gnn_v5(seed=0):
    """`_make_film_gnn` at node_in=N_COLS_V5 (20) -- otherwise identical
    (non-zero-init FiLM last layer, so `g` is not a no-op)."""
    torch.manual_seed(seed)
    model = SolidGNN(node_in=N_COLS_V5, edge_in=5, hidden=16, n_blocks=2,
                     two_head=True, film=True, g_dim=5)
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for film in model.films:
            w = rng.normal(scale=2.0, size=tuple(film.net[-1].weight.shape)).astype(np.float32)
            b = rng.normal(scale=2.0, size=tuple(film.net[-1].bias.shape)).astype(np.float32)
            film.net[-1].weight.copy_(torch.tensor(w))
            film.net[-1].bias.copy_(torch.tensor(b))
    model.eval()
    return model


def _norm20():
    return {"feat_mean": np.zeros(N_COLS_V5, dtype=np.float32),
            "feat_std": np.ones(N_COLS_V5, dtype=np.float32),
            "g_mean": np.zeros(5), "g_std": np.ones(5),
            "y_log_mean": np.float64(3.0), "y_log_std": np.float64(1.0)}


def _film_cfg_v5():
    return {"hidden": 16, "n_blocks": 2, "k": 4, "graph": "union",
            "feature_version": 5, "film": True, "g_dim": 5, "two_head": True,
            "node_in": N_COLS_V5, "edge_in": 5}


def test_v4_cfg_still_builds_15_columns_unchanged():
    """(a) The existing v4 cfg (feature_version=4, node_in=15) must still
    produce exactly the 15-column X every test above already asserts --
    the binding "cfg.feature_version==4 (or absent) -> byte-identical
    current behavior" contract, re-stated as its own direct test rather
    than relying only on the untouched tests above."""
    mesh = _all_cooling_cube()
    gnn = _make_film_gnn(seed=0)
    hgb = _StubHGB(n_features_in_=16, total_seconds=50.0)
    props = resolve_props()

    out, feat = predict_field(mesh, gnn, hgb, _norm15(), cfg=_film_cfg(),
                              device="cpu", props=props)
    assert feat["X"].shape == (mesh.node_xyz.shape[0], N_COLS_V4)


def test_film_cfg_missing_feature_version_key_defaults_to_15_columns():
    """(a) `cfg.get("feature_version", 4)` -- a film cfg with NO
    "feature_version" key at all (not just one explicitly set to 4) must
    still default to the v4 (15-column) builder."""
    mesh = _all_cooling_cube()
    gnn = _make_film_gnn(seed=0)
    hgb = _StubHGB(n_features_in_=16, total_seconds=50.0)
    props = resolve_props()
    cfg = {k: v for k, v in _film_cfg().items() if k != "feature_version"}

    out, feat = predict_field(mesh, gnn, hgb, _norm15(), cfg=cfg,
                              device="cpu", props=props)
    assert feat["X"].shape == (mesh.node_xyz.shape[0], N_COLS_V4)


def test_v5_cfg_film_path_builds_20_column_x_all_sand_last_five_zero():
    """(b) A v5-cfg stub ckpt (cfg.feature_version=5, node_in=20) drives
    `compute_features_inference` into the 20-column builder; a bare
    all-sand mesh (no bc_plan, no Adiabatic patch) leaves the 5 new
    BC-reach channels (exo_mid/exo_long/chill_short/chill_mid/chill_long)
    exactly zero -- matching src.features's own all-sand contract."""
    mesh = _all_cooling_cube()
    gnn = _make_film_gnn_v5(seed=0)
    hgb = _StubHGB(n_features_in_=16, total_seconds=50.0)
    props = resolve_props()

    out, feat = predict_field(mesh, gnn, hgb, _norm20(), cfg=_film_cfg_v5(),
                              device="cpu", props=props)
    X = feat["X"]
    assert X.shape == (mesh.node_xyz.shape[0], N_COLS_V5)
    assert out.shape == (mesh.node_xyz.shape[0],)
    assert np.isfinite(out).all()
    assert (X[:, 15:] == 0.0).all()


def test_v5_cfg_bodies_exo_patch_populates_exo_channels_not_chill(monkeypatch):
    """(c) A bodies/BcPlan exo_sleeve patch resolved EXACTLY the way the
    training builder does (`bc_plan.feature_spec(...)`, same spec keys)
    must populate exo_mid/exo_long nonzero while leaving every chill
    channel exactly zero -- the SAME spec resolution `compute_features_
    inference`'s v4 branch already uses, now flowing into
    `compute_bc_reach_channels` via the version=5 dispatch."""
    from src.assembly import BcPlan

    mesh = _cube_with_bc_patch()
    gnn = _make_film_gnn_v5(seed=0)
    hgb = _StubHGB(n_features_in_=16, total_seconds=50.0)
    props = resolve_props()
    plan = BcPlan(bc_config="exo_sleeve", casting_solid_id=1, role_solid_ids=[2],
                  sleeve_MEF=2.0, exo_q0=5000.0)

    out, feat = predict_field(mesh, gnn, hgb, _norm20(), cfg=_film_cfg_v5(),
                              device="cpu", props=props, bc_plan=plan)
    X = feat["X"]
    idx = {n: i for i, n in enumerate(FEATURE_NAMES_V5)}
    assert X[:, idx["exo_mid"]].max() > 0.0
    assert X[:, idx["exo_long"]].max() > 0.0
    assert (X[:, idx["chill_short"]] == 0.0).all()
    assert (X[:, idx["chill_mid"]] == 0.0).all()
    assert (X[:, idx["chill_long"]] == 0.0).all()


def test_v5_cfg_bodies_chill_patch_populates_chill_channels_not_exo(monkeypatch):
    """(c), vice versa: a chills BcPlan populates chill_short/mid/long
    nonzero while exo_mid/exo_long stay exactly zero."""
    from src.assembly import BcPlan

    mesh = _cube_with_bc_patch()
    gnn = _make_film_gnn_v5(seed=0)
    hgb = _StubHGB(n_features_in_=16, total_seconds=50.0)
    props = resolve_props()
    plan = BcPlan(bc_config="chills", casting_solid_id=1, role_solid_ids=[2],
                  n_chills=2, chill_effusivity=9876.5)

    out, feat = predict_field(mesh, gnn, hgb, _norm20(), cfg=_film_cfg_v5(),
                              device="cpu", props=props, bc_plan=plan)
    X = feat["X"]
    idx = {n: i for i, n in enumerate(FEATURE_NAMES_V5)}
    assert X[:, idx["chill_short"]].max() > 0.0
    assert X[:, idx["chill_mid"]].max() > 0.0
    assert X[:, idx["chill_long"]].max() > 0.0
    assert (X[:, idx["exo_mid"]] == 0.0).all()
    assert (X[:, idx["exo_long"]] == 0.0).all()


# --------------------------------------------------------------------------- #
# SP1 (non-film) path: byte-identical regression
# --------------------------------------------------------------------------- #
def test_sp1_path_regression_still_routes_through_hybrid_field(monkeypatch, mini_msh_file):
    mesh = parse_msh(mini_msh_file)
    gnn = SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2)
    norm = {"feat_mean": np.zeros(9), "feat_std": np.ones(9),
            "y_log_mean": 0.0, "y_log_std": 1.0}
    rng = np.random.default_rng(0)
    hgb = HistGradientBoostingRegressor(max_iter=5).fit(rng.normal(size=(50, 9)),
                                                        rng.normal(size=50))

    sentinel = np.array([111.0])
    called = {}

    def _fake_hybrid_field(g, b):
        called["yes"] = True
        return sentinel

    monkeypatch.setattr(pc_mod, "hybrid_field", _fake_hybrid_field)

    cfg = {"k": 3, "graph": "knn", "feature_version": 2}  # no "film" key -> non-film path
    out, _ = predict_field(mesh, gnn, hgb, norm, cfg=cfg, device="cpu")

    assert called.get("yes") is True
    assert np.array_equal(out, sentinel)


# --------------------------------------------------------------------------- #
# load_hybrid: film-ckpt detection
# --------------------------------------------------------------------------- #
def test_load_hybrid_detects_film_ckpt_and_skips_baseline_norm(tmp_path):
    gnn = _make_film_gnn(seed=0)
    ck_path = tmp_path / "film.pt"
    torch.save({"model": gnn.state_dict(), "norm": _norm15(), "cfg": _film_cfg()}, ck_path)

    loaded_gnn, hgb, norm, cfg, baseline_norm, hgb_std = pc_mod.load_hybrid(
        str(ck_path), hgb_path=None, device="cpu")

    assert cfg.get("film") is True
    assert baseline_norm is None
    assert hgb is None
    assert hgb_std is None
    assert norm["y_log_mean"] == pytest.approx(3.0)


def test_load_hybrid_non_film_ckpt_unchanged(tmp_path):
    gnn = SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2)
    ck_path = tmp_path / "sp1.pt"
    norm = {"feat_mean": np.zeros(9), "feat_std": np.ones(9),
            "y_log_mean": 0.0, "y_log_std": 1.0}
    torch.save({"model": gnn.state_dict(), "norm": norm,
                "cfg": {"hidden": 16, "n_blocks": 2, "k": 6}}, ck_path)

    loaded_gnn, hgb, loaded_norm, cfg, baseline_norm, hgb_std = pc_mod.load_hybrid(
        str(ck_path), hgb_path=None, device="cpu")

    assert not cfg.get("film", False)
    assert baseline_norm is None  # hgb_path is None here too, so unchanged either way
    assert hgb_std is None


def test_sp1_path_with_wrong_dim_hgb_raises(mini_msh_file):
    """Vice-versa guard: a v4 (16-feature) GBM fed to the SP1/non-film path
    must also fail loudly, not silently misbehave or crash inside sklearn."""
    mesh = parse_msh(mini_msh_file)
    gnn = SolidGNN(node_in=9, edge_in=4, hidden=16, n_blocks=2)
    norm = {"feat_mean": np.zeros(9), "feat_std": np.ones(9),
            "y_log_mean": 0.0, "y_log_std": 1.0}
    hgb = _StubHGB(n_features_in_=16, total_seconds=50.0)

    cfg = {"k": 3, "graph": "knn", "feature_version": 2}
    with pytest.raises(ValueError):
        predict_field(mesh, gnn, hgb, norm, cfg=cfg, device="cpu")


# --------------------------------------------------------------------------- #
# Task 8: mesh_mode routing (prep vs. raw) -- recorder-only, no real gmsh
# --------------------------------------------------------------------------- #
def _stub_prediction_stage(monkeypatch, n_nodes):
    """Bypass the model-loading + field-prediction stages entirely (no real
    checkpoints touched) so these tests exercise ONLY predict_casting's
    meshing-stage routing."""
    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {}, None, None))
    monkeypatch.setattr(pc_mod, "predict_field",
                        lambda mesh, *a, **kw: (np.arange(n_nodes, dtype=np.float64) + 1.0,
                                                {"X": np.zeros((n_nodes, 9))}))


def _fake_get_or_mesh(calls, mini_msh_file, ok=True, cache="miss"):
    def _fn(step_path, out_msh, target_nodes=None, cache_dir=None,
            keep_solid_id=None, bodies_key="none", post_mesh=None):
        calls["get_or_mesh"] = dict(step_path=step_path, out_msh=out_msh,
                                    target_nodes=target_nodes, cache_dir=cache_dir,
                                    keep_solid_id=keep_solid_id, bodies_key=bodies_key,
                                    post_mesh=post_mesh)
        shutil.copy(mini_msh_file, out_msh)
        result = dict(n_solids=1, volume_mm3=1.0, bbox_diag_mm=17.3, n_nodes=5,
                      ok=ok, healed=False, notes="", cache=cache)
        # Part B Task 4: exercise the tagging-before-store contract exactly
        # as the real get_or_mesh does -- call the post_mesh hook (if any)
        # on this "miss" and merge its return into the meta.
        if post_mesh is not None:
            result.update(post_mesh(out_msh))
        return result
    return _fn


def _fake_tag_contact_patches(calls, per_body_patch_area_mm2=None, n_adiab_tris=7,
                              nodes_by_solid=None):
    per_body_patch_area_mm2 = per_body_patch_area_mm2 or {2: 42.0}

    def _fn(msh_path, step_path, role_solid_ids, tol_mm=None):
        calls.setdefault("tag_contact_patches", []).append(
            dict(msh_path=msh_path, step_path=step_path,
                 role_solid_ids=list(role_solid_ids), tol_mm=tol_mm))
        result = {"per_body_patch_area_mm2": dict(per_body_patch_area_mm2),
                  "n_adiab_tris": n_adiab_tris}
        if nodes_by_solid is not None:
            result["nodes_by_solid"] = {k: list(v) for k, v in nodes_by_solid.items()}
        return result
    return _fn


def _fake_mesh_step(calls, mini_msh_file):
    def _fn(step_path, out_msh, elem_size=None):
        calls["mesh_step"] = dict(step_path=step_path, out_msh=out_msh, elem_size=elem_size)
        shutil.copy(mini_msh_file, out_msh)
        return dict(n_nodes=5, n_tets=1, bbox_mm=(10.0, 10.0, 10.0), elem_size=elem_size or 1.0)
    return _fn


def test_predict_casting_default_routes_through_get_or_mesh(
        monkeypatch, mini_msh_file, tmp_path):
    calls = {}
    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    _stub_prediction_stage(monkeypatch, n_nodes=5)

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 ood_json="does-not-exist.json")

    assert "get_or_mesh" in calls
    assert "mesh_step" not in calls
    assert calls["get_or_mesh"]["target_nodes"] == pc_mod.DEFAULT_TARGET_NODES
    assert out["mesh_mode"] == "prep"
    assert out["n_nodes"] == 5
    assert "cache" in out
    assert out["cache"] == "miss"


def test_mesh_mode_raw_routes_through_mesh_step(monkeypatch, mini_msh_file, tmp_path):
    calls = {}
    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    _stub_prediction_stage(monkeypatch, n_nodes=5)

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 mesh_mode="raw", ood_json="does-not-exist.json")

    assert "mesh_step" in calls
    assert "get_or_mesh" not in calls
    assert out["mesh_mode"] == "raw"


def test_explicit_elem_size_implies_raw_even_with_default_mesh_mode(
        monkeypatch, mini_msh_file, tmp_path):
    calls = {}
    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    _stub_prediction_stage(monkeypatch, n_nodes=5)

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 elem_size=5.0, ood_json="does-not-exist.json")

    assert "mesh_step" in calls
    assert "get_or_mesh" not in calls
    assert calls["mesh_step"]["elem_size"] == 5.0
    assert out["mesh_mode"] == "raw"


def test_prep_meshing_failure_raises_loudly(monkeypatch, mini_msh_file, tmp_path):
    calls = {}
    monkeypatch.setattr(pc_mod, "get_or_mesh",
                        _fake_get_or_mesh(calls, mini_msh_file, ok=False))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    _stub_prediction_stage(monkeypatch, n_nodes=5)

    with pytest.raises(RuntimeError):
        pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                               ood_json="does-not-exist.json")


def test_predict_casting_includes_cache_hit_in_result(
        monkeypatch, mini_msh_file, tmp_path):
    """Verify that a cache hit result dict includes the 'cache' field."""
    calls = {}
    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file, cache="hit"))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    _stub_prediction_stage(monkeypatch, n_nodes=5)

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 ood_json="does-not-exist.json")

    assert "cache" in out
    assert out["cache"] == "hit"


# --------------------------------------------------------------------------- #
# Part B Task 4: predict_casting(bodies=...) wiring -- monkeypatched
# recorders for validate_bodies/tag_contact_patches/get_or_mesh/predict_field,
# no real gmsh/model checkpoints touched.
# --------------------------------------------------------------------------- #
def _fake_validate_bodies(calls, plan):
    def _fn(bodies, **kw):
        calls["validate_bodies_arg"] = bodies
        return plan
    return _fn


def test_bodies_given_wires_keep_solid_id_bodies_key_and_tagging(
        monkeypatch, mini_msh_file, tmp_path):
    from src.assembly import BcPlan

    calls = {}
    plan = BcPlan(bc_config="chills", casting_solid_id=1, role_solid_ids=[2],
                  n_chills=1, chill_effusivity=12345.0)

    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    monkeypatch.setattr(pc_mod, "tag_contact_patches", _fake_tag_contact_patches(calls))

    captured = {}

    def fake_predict_field(mesh, gnn_model, hgb_model, norm, cfg=None, device=None, tta=0,
                           baseline_norm=None, props=None, bc_plan=None, hgb_std=None,
                           lambda_shrink=1.0, w_total=1.0, w_std=1.0):
        captured["bc_plan"] = bc_plan
        return np.arange(5, dtype=np.float64) + 1.0, {"X": np.zeros((5, 9))}

    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {"film": True}, None, None))
    monkeypatch.setattr(pc_mod, "predict_field", fake_predict_field)

    bodies = [{"solid_id": 1, "role": "casting"},
             {"solid_id": 2, "role": "chill", "params": {"material": "cast_iron"}}]

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 ood_json="does-not-exist.json", bodies=bodies)

    assert calls["validate_bodies_arg"] == bodies
    assert calls["get_or_mesh"]["keep_solid_id"] == 1
    assert calls["get_or_mesh"]["bodies_key"] != "none"
    assert "tag_contact_patches" in calls
    assert calls["tag_contact_patches"][0]["role_solid_ids"] == [2]
    assert captured["bc_plan"] is plan
    assert out["bc_config"] == "chills"
    assert out["per_body_patch_area_mm2"] == {2: 42.0}


# --------------------------------------------------------------------------- #
# Shrinkage-screening spec ADDENDUM 2 (2026-08-01, owner-approved SCOPED
# SHIP): Gate C passed on all-sand/sleeve-only inputs but FAILED on the
# chill-bearing slice (rho 0.313) -- a chill-bearing BC plan
# (`bc_config == "chills"`) must mark "niyama_confidence": "low" + a summary
# caveat; sleeve-only/bare inputs stay "full". Follows the same
# validate_bodies/BcPlan monkeypatch pattern as the Part B tests above.
# --------------------------------------------------------------------------- #
def test_bodies_chill_role_marks_niyama_confidence_low_with_summary_caveat(
        monkeypatch, mini_msh_file, tmp_path):
    from src.assembly import BcPlan

    calls = {}
    plan = BcPlan(bc_config="chills", casting_solid_id=1, role_solid_ids=[2],
                  n_chills=1, chill_effusivity=12345.0)

    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    monkeypatch.setattr(pc_mod, "tag_contact_patches", _fake_tag_contact_patches(calls))
    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {"film": True}, None, None))
    monkeypatch.setattr(pc_mod, "predict_field",
                        lambda mesh, *a, **kw: (np.arange(5, dtype=np.float64) + 1.0,
                                                {"X": np.zeros((5, 9))}))

    bodies = [{"solid_id": 1, "role": "casting"},
             {"solid_id": 2, "role": "chill", "params": {"material": "cast_iron"}}]

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 ood_json="does-not-exist.json", bodies=bodies)

    assert out["shrinkage"] is not None
    assert out["shrinkage"]["niyama_confidence"] == "low"
    with open(out["summary"]) as f:
        summary_text = f.read()
    assert "niyama confidence: LOW" in summary_text
    assert "rho 0.313" in summary_text


def test_bodies_sleeve_only_role_keeps_niyama_confidence_full_no_caveat(
        monkeypatch, mini_msh_file, tmp_path):
    """A sleeve-only BC plan (bc_config="insulating_sleeve", no chill role
    at all) is UNAFFECTED by the chills-only Gate C gap -- stays "full",
    no caveat line."""
    from src.assembly import BcPlan

    calls = {}
    plan = BcPlan(bc_config="insulating_sleeve", casting_solid_id=1, role_solid_ids=[2])

    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    monkeypatch.setattr(pc_mod, "tag_contact_patches", _fake_tag_contact_patches(calls))
    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {"film": True}, None, None))
    monkeypatch.setattr(pc_mod, "predict_field",
                        lambda mesh, *a, **kw: (np.arange(5, dtype=np.float64) + 1.0,
                                                {"X": np.zeros((5, 9))}))

    bodies = [{"solid_id": 1, "role": "casting"},
             {"solid_id": 2, "role": "insulating_sleeve"}]

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 ood_json="does-not-exist.json", bodies=bodies)

    assert out["shrinkage"] is not None
    assert out["shrinkage"]["niyama_confidence"] == "full"
    with open(out["summary"]) as f:
        summary_text = f.read()
    assert "niyama confidence: LOW" not in summary_text
    assert "rho 0.313" not in summary_text


def test_bodies_chill_role_learned_niyama_source_uses_neutral_caveat(
        monkeypatch, mini_msh_file, tmp_path):
    """Learned-heads Task 3 review round 1, F3: when `predict_field`'s
    `feat` carries learned `gradT`/`dTdt` (so `niyama_source` is "learned",
    not "proxy"), the chill-bearing low-confidence caveat text must be the
    NEUTRAL "learned-head Niyama pending slice validation" -- NOT the
    proxy-specific "rho 0.313" text, a claim about the PROXY identity's own
    measured Gate C shortfall that has nothing to do with a learned
    checkpoint's heads. The marking itself ("low") is UNCHANGED -- still
    keyed only on `plan.bc_config == "chills"`, per the brief ("stays until
    the promote step")."""
    from src.assembly import BcPlan

    calls = {}
    plan = BcPlan(bc_config="chills", casting_solid_id=1, role_solid_ids=[2],
                  n_chills=1, chill_effusivity=12345.0)

    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    monkeypatch.setattr(pc_mod, "tag_contact_patches", _fake_tag_contact_patches(calls))
    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {"film": True}, None, None))

    def fake_predict_field(mesh, *a, **kw):
        n = 5
        seconds = np.arange(n, dtype=np.float64) + 1.0
        feat = {"X": np.zeros((n, 9)), "gradT": np.full(n, 500.0), "dTdt": np.full(n, 2.0)}
        return seconds, feat

    monkeypatch.setattr(pc_mod, "predict_field", fake_predict_field)

    bodies = [{"solid_id": 1, "role": "casting"},
             {"solid_id": 2, "role": "chill", "params": {"material": "cast_iron"}}]

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 ood_json="does-not-exist.json", bodies=bodies)

    assert out["shrinkage"] is not None
    assert out["shrinkage"]["niyama_confidence"] == "low"
    assert out["shrinkage"]["niyama_source"] == "learned"
    with open(out["summary"]) as f:
        summary_text = f.read()
    assert "niyama confidence: LOW" in summary_text
    assert "learned-head Niyama pending slice validation" in summary_text
    assert "rho 0.313" not in summary_text


# --------------------------------------------------------------------------- #
# Part-B roles task (2026-08-03): role-aware feed-node selection, replacing
# review finding I1's coarse `plan.bc_config == "chills"` special case (which
# forced feed_nodes empty for the WHOLE input, rather than per body role).
# `tag_contact_patches` relabels EVERY role body's contact patch "Adiabatic"
# -- chills included -- so on a chill-bearing input every "Adiabatic"
# triangle is a chill footprint, not a riser/sleeve; on a sleeve-bearing
# input it genuinely is a feed patch. `mini_msh_file` already carries one
# real "Adiabatic" tri (0-indexed nodes 1,2,3 -- file node ids 2,3,4), so
# both branches are exercised against the SAME underlying mesh, with the
# fake `tag_contact_patches` now also returning `nodes_by_solid={2: [1, 2,
# 3]}` (what the real tagging pass would have found for that one role body)
# and each `BcPlan` carrying the matching `role_by_solid` -- only the ROLE
# differs between the two tests, not merely `bc_config`.
# --------------------------------------------------------------------------- #
def _spy_screen_casting(monkeypatch, captured):
    real_screen_casting = pc_mod.screen_casting

    def _fn(*args, **kwargs):
        captured["feed_nodes"] = kwargs.get("feed_nodes")
        return real_screen_casting(*args, **kwargs)

    monkeypatch.setattr(pc_mod, "screen_casting", _fn)


def test_bodies_chill_role_forces_feed_nodes_empty(monkeypatch, mini_msh_file, tmp_path):
    from src.assembly import BcPlan

    calls = {}
    plan = BcPlan(bc_config="chills", casting_solid_id=1, role_solid_ids=[2],
                  n_chills=1, chill_effusivity=12345.0, role_by_solid={2: "chill"})

    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    monkeypatch.setattr(pc_mod, "tag_contact_patches",
                        _fake_tag_contact_patches(calls, nodes_by_solid={2: [1, 2, 3]}))
    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {"film": True}, None, None))
    monkeypatch.setattr(pc_mod, "predict_field",
                        lambda mesh, *a, **kw: (np.arange(5, dtype=np.float64) + 1.0,
                                                {"X": np.zeros((5, 9))}))
    captured = {}
    _spy_screen_casting(monkeypatch, captured)

    bodies = [{"solid_id": 1, "role": "casting"},
             {"solid_id": 2, "role": "chill", "params": {"material": "cast_iron"}}]

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 ood_json="does-not-exist.json", bodies=bodies)

    # the chill body's own patch DID find nodes (1,2,3) -- they are simply
    # never fed nodes, proving this is a real per-role exclusion and not an
    # accident of an empty/missing nodes_by_solid.
    assert captured["feed_nodes"].size == 0
    # with feed_nodes forced empty, is_fed is always False -- EVERY hot
    # region hot_regions itself finds must survive into macro_shrinkage
    # (none silently dropped as "fed" by a chill contact patch).
    from src.shrinkage import hot_regions
    mesh = parse_msh(osp.join(str(tmp_path), f"{out['name']}.msh"))
    seconds = np.arange(mesh.node_xyz.shape[0], dtype=np.float64) + 1.0
    all_hot_regions = hot_regions(mesh.node_xyz, mesh.tets, seconds)
    assert out["shrinkage"]["n_macro_regions"] == len(all_hot_regions)


def test_bodies_sleeve_role_keeps_adiabatic_feed_nodes(monkeypatch, mini_msh_file, tmp_path):
    from src.assembly import BcPlan

    calls = {}
    plan = BcPlan(bc_config="insulating_sleeve", casting_solid_id=1, role_solid_ids=[2],
                  role_by_solid={2: "insulating_sleeve"})

    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    monkeypatch.setattr(pc_mod, "tag_contact_patches",
                        _fake_tag_contact_patches(calls, nodes_by_solid={2: [1, 2, 3]}))
    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {"film": True}, None, None))
    monkeypatch.setattr(pc_mod, "predict_field",
                        lambda mesh, *a, **kw: (np.arange(5, dtype=np.float64) + 1.0,
                                                {"X": np.zeros((5, 9))}))
    captured = {}
    _spy_screen_casting(monkeypatch, captured)

    bodies = [{"solid_id": 1, "role": "casting"},
             {"solid_id": 2, "role": "insulating_sleeve"}]

    pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                          ood_json="does-not-exist.json", bodies=bodies)

    # sleeve assembly -> feed == that patch's nodes EXACTLY (spec §3).
    assert captured["feed_nodes"].tolist() == [1, 2, 3]


def test_bodies_sleeve_role_feed_nodes_byte_identical_to_pre_change_adiabatic_union(
        monkeypatch, mini_msh_file, tmp_path):
    """Screening byte-identity (spec §3): on a legal (single-role, non-mixed)
    sleeve input, the NEW per-role feed_nodes selection must produce EXACTLY
    the same result as the pre-change code (`_shrinkage_surface_and_feed_
    nodes`'s plain "Adiabatic"-tag node union, which is what `predict_
    casting` used unconditionally for any non-chills bc_config before this
    task)."""
    from src.assembly import BcPlan
    from src.msh_parse import parse_msh as _parse_msh

    calls = {}
    plan = BcPlan(bc_config="exo_sleeve", casting_solid_id=1, role_solid_ids=[2],
                  role_by_solid={2: "exo_sleeve"})

    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    monkeypatch.setattr(pc_mod, "tag_contact_patches",
                        _fake_tag_contact_patches(calls, nodes_by_solid={2: [1, 2, 3]}))
    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {"film": True}, None, None))
    monkeypatch.setattr(pc_mod, "predict_field",
                        lambda mesh, *a, **kw: (np.arange(5, dtype=np.float64) + 1.0,
                                                {"X": np.zeros((5, 9))}))
    captured = {}
    _spy_screen_casting(monkeypatch, captured)

    bodies = [{"solid_id": 1, "role": "casting"},
             {"solid_id": 2, "role": "exo_sleeve"}]

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 ood_json="does-not-exist.json", bodies=bodies)

    old_mesh = _parse_msh(osp.join(str(tmp_path), f"{out['name']}.msh"))
    _old_surface, old_feed_nodes = pc_mod._shrinkage_surface_and_feed_nodes(old_mesh)

    assert np.array_equal(captured["feed_nodes"], old_feed_nodes)


def test_bodies_legacy_cache_hit_sleeve_falls_back_to_adiabatic_union_with_warning(
        monkeypatch, mini_msh_file, tmp_path, caplog):
    """Review round 1, H1: a mesh-cache HIT served from a side-car written
    BEFORE the Part-B roles task never had `nodes_by_solid` at all --
    `runs/mesh_cache/` holds exactly such legacy entries today (`per_body_
    patch_area_mm2` present, `nodes_by_solid` absent), incl. both gold
    parts. On a sleeve plan, feed_nodes must fall back to the PRE-CHANGE
    rule (the plain "Adiabatic"-tag node union) rather than silently going
    empty, and a loud warning naming the legacy situation must fire."""
    from src.assembly import BcPlan

    calls = {}
    plan = BcPlan(bc_config="insulating_sleeve", casting_solid_id=1, role_solid_ids=[2],
                  role_by_solid={2: "insulating_sleeve"})

    monkeypatch.setattr(pc_mod, "get_or_mesh",
                        _fake_get_or_mesh_hit_with_str_keys(calls, mini_msh_file, {"2": 42.0}))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    monkeypatch.setattr(pc_mod, "tag_contact_patches", _fake_tag_contact_patches(calls))
    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {"film": True}, None, None))
    monkeypatch.setattr(pc_mod, "predict_field",
                        lambda mesh, *a, **kw: (np.arange(5, dtype=np.float64) + 1.0,
                                                {"X": np.zeros((5, 9))}))
    captured = {}
    _spy_screen_casting(monkeypatch, captured)

    bodies = [{"solid_id": 1, "role": "casting"},
             {"solid_id": 2, "role": "insulating_sleeve"}]

    with caplog.at_level(logging.WARNING):
        out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                     ood_json="does-not-exist.json", bodies=bodies)

    mesh = parse_msh(osp.join(str(tmp_path), f"{out['name']}.msh"))
    _surface, expected_adiab_union = pc_mod._shrinkage_surface_and_feed_nodes(mesh)
    assert expected_adiab_union.size > 0   # sanity: mini_msh_file's real Adiabatic tri
    assert np.array_equal(captured["feed_nodes"], expected_adiab_union)
    assert any("legacy" in rec.message.lower() for rec in caplog.records)


def test_bodies_legacy_cache_hit_chills_falls_back_to_empty_with_warning(
        monkeypatch, mini_msh_file, tmp_path, caplog):
    """Same legacy-cache-HIT scenario as above, chills plan: falls back to
    empty feed_nodes (the pre-change chills rule), with the same loud
    warning -- and the warning fires regardless of `bc_config`."""
    from src.assembly import BcPlan

    calls = {}
    plan = BcPlan(bc_config="chills", casting_solid_id=1, role_solid_ids=[2],
                  n_chills=1, chill_effusivity=12345.0, role_by_solid={2: "chill"})

    monkeypatch.setattr(pc_mod, "get_or_mesh",
                        _fake_get_or_mesh_hit_with_str_keys(calls, mini_msh_file, {"2": 42.0}))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    monkeypatch.setattr(pc_mod, "tag_contact_patches", _fake_tag_contact_patches(calls))
    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {"film": True}, None, None))
    monkeypatch.setattr(pc_mod, "predict_field",
                        lambda mesh, *a, **kw: (np.arange(5, dtype=np.float64) + 1.0,
                                                {"X": np.zeros((5, 9))}))
    captured = {}
    _spy_screen_casting(monkeypatch, captured)

    bodies = [{"solid_id": 1, "role": "casting"},
             {"solid_id": 2, "role": "chill", "params": {"material": "cast_iron"}}]

    with caplog.at_level(logging.WARNING):
        pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                               ood_json="does-not-exist.json", bodies=bodies)

    assert captured["feed_nodes"].size == 0
    assert any("legacy" in rec.message.lower() for rec in caplog.records)


def test_nodes_by_role_from_tagging_incomplete_role_by_solid_warns_and_drops(caplog):
    """Review round 1, M1: a solid_id present in `nodes_by_solid` but absent
    from `role_by_solid` must be dropped LOUDLY (not silently) -- defense in
    depth for any caller other than `_role_aware_feed_nodes` (which, in the
    normal `predict_casting` path, never lets an incomplete mapping reach
    this function at all -- see its own docstring)."""
    with caplog.at_level(logging.WARNING):
        grouped = pc_mod._nodes_by_role_from_tagging(
            {2: [1, 2, 3], 5: [7, 8]}, {2: "insulating_sleeve"})

    assert grouped == {"insulating_sleeve": [1, 2, 3]}
    assert any("solid_id=5" in rec.message for rec in caplog.records)


def test_feed_nodes_from_role_patches_mixed_mapping_bypassing_validate_bodies():
    """A directly-constructed mixed role -> nodes mapping (spec §3) --
    `validate_bodies` itself forbids combining chills with a sleeve, but
    `_feed_nodes_from_role_patches` takes no `BcPlan`/`bc_config` at all, so
    it must still exclude the chill nodes and keep only the sleeve nodes
    even when both roles co-occur in the same mapping."""
    nodes_by_role = {"chill": [10, 11, 12], "insulating_sleeve": [11, 20],
                     "exo_sleeve": [30]}

    feed = pc_mod._feed_nodes_from_role_patches(nodes_by_role)

    assert feed.tolist() == [11, 20, 30]
    assert 10 not in feed.tolist() and 12 not in feed.tolist()


def test_nodes_by_role_from_tagging_bare_mesh_empty_mapping_is_empty():
    """Bare mesh (spec §3): no role bodies at all -> an empty `nodes_by_solid`
    (or `role_by_solid`) grouping produces an empty per-role mapping, and
    `_feed_nodes_from_role_patches` on that empty mapping stays empty --
    matching `predict_casting`'s `if plan is not None and plan.role_solid_
    ids:` guard, which never even enters this code path for a bare/bodies=
    None input."""
    assert pc_mod._nodes_by_role_from_tagging({}, {}) == {}
    assert pc_mod._feed_nodes_from_role_patches({}).size == 0


def test_nodes_by_role_from_tagging_normalizes_string_solid_id_keys():
    """A mesh-cache HIT round-trips `nodes_by_solid` through JSON, turning
    int solid_id keys into str -- `_nodes_by_role_from_tagging` must still
    match them against `role_by_solid`'s int keys."""
    grouped = pc_mod._nodes_by_role_from_tagging(
        {"2": [1, 2, 3]}, {2: "insulating_sleeve"})
    assert grouped == {"insulating_sleeve": [1, 2, 3]}


def _fake_get_or_mesh_hit_with_str_keys(calls, mini_msh_file, per_body_patch_area_mm2):
    """Simulates a real cache HIT: `per_body_patch_area_mm2` comes back with
    STRING solid_id keys (the JSON-round-trip artifact -- see
    tests/test_mesh_cache.py), and `post_mesh` is never called (a HIT never
    re-tags)."""
    def _fn(step_path, out_msh, target_nodes=None, cache_dir=None,
            keep_solid_id=None, bodies_key="none", post_mesh=None):
        calls["get_or_mesh"] = dict(step_path=step_path, out_msh=out_msh,
                                    target_nodes=target_nodes, cache_dir=cache_dir,
                                    keep_solid_id=keep_solid_id, bodies_key=bodies_key,
                                    post_mesh=post_mesh)
        shutil.copy(mini_msh_file, out_msh)
        return dict(n_solids=1, volume_mm3=1.0, bbox_diag_mm=17.3, n_nodes=5,
                    ok=True, healed=False, notes="", cache="hit",
                    per_body_patch_area_mm2=dict(per_body_patch_area_mm2), n_adiab_tris=3)
    return _fn


def test_bodies_cache_hit_normalizes_string_keys_to_int(
        monkeypatch, mini_msh_file, tmp_path):
    """Review fix (LOW): a cache HIT's `per_body_patch_area_mm2` comes back
    with STRING dict keys (json.dump has no int-key mode) -- predict_casting
    must normalize these back to int in its result dict, and must NOT call
    tag_contact_patches again on a HIT."""
    from src.assembly import BcPlan

    calls = {}
    plan = BcPlan(bc_config="chills", casting_solid_id=1, role_solid_ids=[2],
                  n_chills=1, chill_effusivity=12345.0)

    monkeypatch.setattr(pc_mod, "get_or_mesh",
                        _fake_get_or_mesh_hit_with_str_keys(calls, mini_msh_file, {"2": 99.5}))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    monkeypatch.setattr(pc_mod, "tag_contact_patches", _fake_tag_contact_patches(calls))
    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {"film": True}, None, None))
    monkeypatch.setattr(pc_mod, "predict_field",
                        lambda mesh, *a, **kw: (np.arange(5, dtype=np.float64) + 1.0,
                                                {"X": np.zeros((5, 9))}))

    bodies = [{"solid_id": 1, "role": "casting"},
             {"solid_id": 2, "role": "chill", "params": {"material": "cast_iron"}}]

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 ood_json="does-not-exist.json", bodies=bodies)

    assert "tag_contact_patches" not in calls          # HIT never re-tags
    assert out["cache"] == "hit"
    assert out["per_body_patch_area_mm2"] == {2: 99.5}
    assert all(isinstance(k, int) for k in out["per_body_patch_area_mm2"])


def test_bodies_none_never_calls_tag_contact_patches_and_default_bc_config(
        monkeypatch, mini_msh_file, tmp_path):
    calls = {}
    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "tag_contact_patches", _fake_tag_contact_patches(calls))
    _stub_prediction_stage(monkeypatch, n_nodes=5)

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 ood_json="does-not-exist.json")

    assert "tag_contact_patches" not in calls
    assert calls["get_or_mesh"]["keep_solid_id"] is None
    assert calls["get_or_mesh"]["bodies_key"] == "none"
    assert out["bc_config"] == "none"
    assert out["per_body_patch_area_mm2"] == {}


def test_bodies_as_json_path_loads(monkeypatch, mini_msh_file, tmp_path):
    from src.assembly import BcPlan

    calls = {}
    plan = BcPlan(bc_config="none", casting_solid_id=1, role_solid_ids=[])

    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    # bodies requires a film-capable checkpoint (see the guard test below) --
    # this test is only about JSON-path loading, so stub a film-capable cfg.
    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {"film": True}, None, None))
    monkeypatch.setattr(pc_mod, "predict_field",
                        lambda mesh, *a, **kw: (np.arange(5, dtype=np.float64) + 1.0,
                                                {"X": np.zeros((5, 9))}))

    bodies = [{"solid_id": 1, "role": "casting"}]
    bodies_path = tmp_path / "bodies.json"
    bodies_path.write_text(json.dumps(bodies))

    out = pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                                 ood_json="does-not-exist.json", bodies=str(bodies_path))

    assert calls["validate_bodies_arg"] == bodies
    assert calls["get_or_mesh"]["keep_solid_id"] == 1


def test_bodies_with_raw_mesh_mode_raises(monkeypatch, mini_msh_file, tmp_path):
    """`bodies` requires the prep/get_or_mesh pipeline (role-based casting
    solid selection + tagging) -- combining it with mesh_mode='raw' (or an
    explicit elem_size, which forces raw) must fail loudly rather than
    silently ignore the role assignment."""
    from src.assembly import BcPlan

    calls = {}
    plan = BcPlan(bc_config="none", casting_solid_id=1, role_solid_ids=[])
    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    _stub_prediction_stage(monkeypatch, n_nodes=5)

    bodies = [{"solid_id": 1, "role": "casting"}]
    with pytest.raises(ValueError):
        pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                               ood_json="does-not-exist.json", bodies=bodies,
                               mesh_mode="raw")


def test_bodies_with_non_film_checkpoint_raises_and_never_tags(
        monkeypatch, mini_msh_file, tmp_path):
    """Review fix (MEDIUM): `bodies` with chill/sleeve roles + a NON-film
    checkpoint (e.g. SP1 via --gnn) must fail loudly, not silently proceed.
    A resolved BcPlan means tag_contact_patches would relabel the contact
    patch "Adiabatic" -- for a chill that's backwards physics (a chill
    COOLS, Adiabatic INSULATES) -- and SP1 has no BC channel to honor the
    role assignment at all. The check runs BEFORE meshing (load_hybrid is
    called first), so tag_contact_patches must never even be invoked."""
    from src.assembly import BcPlan

    calls = {}
    plan = BcPlan(bc_config="chills", casting_solid_id=1, role_solid_ids=[2],
                  n_chills=1, chill_effusivity=12345.0)

    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "validate_bodies", _fake_validate_bodies(calls, plan))
    monkeypatch.setattr(pc_mod, "tag_contact_patches", _fake_tag_contact_patches(calls))
    # A non-film (SP1-shaped) cfg -- no "film" key at all, matching a real
    # SP1 checkpoint's cfg.
    monkeypatch.setattr(pc_mod, "load_hybrid",
                        lambda *a, **kw: (None, None, {}, {"hidden": 16}, None, None))

    bodies = [{"solid_id": 1, "role": "casting"},
             {"solid_id": 2, "role": "chill", "params": {"material": "cast_iron"}}]

    with pytest.raises(ValueError, match="film-conditioned"):
        pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                               ood_json="does-not-exist.json", bodies=bodies)

    assert "tag_contact_patches" not in calls
    assert "get_or_mesh" not in calls


def test_bodies_cache_key_order_invariant_and_role_sensitive():
    from src.assembly import BcPlan

    plan = BcPlan(bc_config="chills", casting_solid_id=1, role_solid_ids=[2, 3],
                  n_chills=2, chill_effusivity=12345.0)
    bodies = [{"solid_id": 1, "role": "casting"},
             {"solid_id": 2, "role": "chill"},
             {"solid_id": 3, "role": "chill"}]
    bodies_reordered = [bodies[2], bodies[0], bodies[1]]

    k1 = pc_mod._bodies_cache_key(bodies, plan)
    k2 = pc_mod._bodies_cache_key(bodies_reordered, plan)
    assert k1 == k2

    plan2 = BcPlan(bc_config="chills", casting_solid_id=1, role_solid_ids=[3],
                   n_chills=1, chill_effusivity=12345.0)
    bodies_role_changed = [{"solid_id": 1, "role": "casting"},
                          {"solid_id": 2, "role": "ignore"},
                          {"solid_id": 3, "role": "chill"}]
    k3 = pc_mod._bodies_cache_key(bodies_role_changed, plan2)
    assert k3 != k1


# --------------------------------------------------------------------------
# target_nodes pass-through (app Phase A, 2026-08-01)
#
# `get_or_mesh`/`heal_and_mesh` have honoured `target_nodes` all along and it is
# part of the mesh-cache key, but `predict_casting` pinned it to the default, so
# no caller could choose a mesh density without dropping to the legacy `raw`
# path via `elem_size` -- which in turn is refused when `bodies` is present.
# Density and per-body roles were therefore mutually exclusive from outside.
# --------------------------------------------------------------------------


def test_target_nodes_is_forwarded_to_get_or_mesh(monkeypatch, mini_msh_file, tmp_path):
    calls = {}
    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh(calls, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step(calls, mini_msh_file))
    _stub_prediction_stage(monkeypatch, n_nodes=5)

    pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                           ood_json="does-not-exist.json", target_nodes=9600)

    assert calls["get_or_mesh"]["target_nodes"] == 9600


def test_target_nodes_with_an_explicit_elem_size_raises(monkeypatch, mini_msh_file,
                                                       tmp_path):
    """Density search and a pinned element size are mutually exclusive requests.

    `elem_size` forces the `raw` path, which never calls `get_or_mesh`, so a
    silently ignored `target_nodes` would mean the caller asked for a mesh
    density and got a different one with no indication. Same reasoning as the
    existing `bodies`-with-`raw` guard, and the same loud failure.
    """
    monkeypatch.setattr(pc_mod, "get_or_mesh", _fake_get_or_mesh({}, mini_msh_file))
    monkeypatch.setattr(pc_mod, "mesh_step", _fake_mesh_step({}, mini_msh_file))
    _stub_prediction_stage(monkeypatch, n_nodes=5)

    with pytest.raises(ValueError) as excinfo:
        pc_mod.predict_casting("fake_part.step", out_dir=str(tmp_path),
                               ood_json="does-not-exist.json",
                               target_nodes=9600, elem_size=4.0)

    message = str(excinfo.value)
    assert "target_nodes" in message and "elem_size" in message
