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

    props_a = resolve_props(alloy="EN-GJS-400-15")
    props_b = resolve_props(alloy="G20Mn5")
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




# --------------------------------------------------------------------------
# target_nodes pass-through (app Phase A, 2026-08-01)
#
# `get_or_mesh`/`heal_and_mesh` have honoured `target_nodes` all along and it is
# part of the mesh-cache key, but `predict_casting` pinned it to the default, so
# no caller could choose a mesh density without dropping to the legacy `raw`
# path via `elem_size`.
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
    density and got a different one with no indication, so it fails loudly.
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
