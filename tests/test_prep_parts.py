from pathlib import Path

import numpy as np
import pytest

import tools.doe.prep_parts as pp
from tools.doe.prep_parts import (
    largest_solid, is_ood, heal_and_mesh, _elem_size_from_area, _elem_size,
    _first_guess_elem_size, K_SURF, _import_and_reduce,
)
from src.msh_parse import parse_msh


def _make_one_box_step(path, dx=30.0, dy=20.0, dz=10.0):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, dx, dy, dz)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def _make_two_box_step(path, big=40.0, small=10.0):
    """A synthetic 2-solid STEP: one `big`^3 box and one `small`^3 box,
    positioned so they don't touch (kept as two separate OCC solids)."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("two_box")
        gmsh.model.occ.addBox(0, 0, 0, big, big, big)
        gmsh.model.occ.addBox(big + 100, 0, 0, small, small, small)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def test_largest_solid_keeps_max_volume_box(tmp_path):
    step = tmp_path / "two_box.step"
    _make_two_box_step(step, big=40.0, small=10.0)

    out_step = tmp_path / "kept.step"
    n_in, kept_vol = largest_solid(str(step), str(out_step))

    assert n_in == 2
    assert kept_vol == pytest.approx(40.0 ** 3, rel=1e-3)

    # Re-import the exported STEP and confirm exactly 1 solid remains.
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("check")
        gmsh.model.occ.importShapes(str(out_step))
        gmsh.model.occ.synchronize()
        vols = gmsh.model.occ.getEntities(3)
        assert len(vols) == 1
        vol = gmsh.model.occ.getMass(3, vols[0][1])
        assert vol == pytest.approx(40.0 ** 3, rel=1e-3)
    finally:
        gmsh.finalize()


def test_largest_solid_single_solid_passthrough(tmp_path):
    step = tmp_path / "one_box.step"
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("one_box")
        gmsh.model.occ.addBox(0, 0, 0, 20, 20, 20)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step))
    finally:
        gmsh.finalize()

    out_step = tmp_path / "kept.step"
    n_in, kept_vol = largest_solid(str(step), str(out_step))
    assert n_in == 1
    assert kept_vol == pytest.approx(20.0 ** 3, rel=1e-3)


def test_is_ood_flags_large_diag():
    ranges = {"volume_mm3_max": 1.0e12, "bbox_diag_max": 2177.7}
    assert is_ood(volume_mm3=1.0, bbox_diag_mm=2663.5, ranges=ranges) is True
    assert is_ood(volume_mm3=1.0, bbox_diag_mm=100.0, ranges=ranges) is False


def test_is_ood_flags_large_volume():
    ranges = {"volume_mm3_max": 390663008.0, "bbox_diag_max": 2177.7}
    assert is_ood(volume_mm3=5.0e9, bbox_diag_mm=100.0, ranges=ranges) is True
    assert is_ood(volume_mm3=1.0e6, bbox_diag_mm=100.0, ranges=ranges) is False


def test_heal_and_mesh_synthetic_box(tmp_path):
    """Small synthetic box -- fast, not a real dataset-D part -- exercises
    the heal+mesh+tag pipeline end to end."""
    step = tmp_path / "box.step"
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 30, 20, 10)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step))
    finally:
        gmsh.finalize()

    out_msh = tmp_path / "box.msh"
    out_step = tmp_path / "box_cleaned.step"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=500, out_step=str(out_step))

    assert result["ok"] is True
    assert result["n_solids"] == 1
    assert result["n_nodes"] > 0
    assert result["volume_mm3"] == pytest.approx(30 * 20 * 10, rel=1e-2)
    assert np.isfinite(result["bbox_diag_mm"]) and result["bbox_diag_mm"] > 0
    assert out_msh.exists()
    assert out_step.exists()


# --------------------------------------------------------------------------
# Surface-area density-search seed (mesh-accel Task 4): the first-attempt
# elem size is now seeded from the OCC surface area of the imported model
# (`_elem_size_from_area`, via `_first_guess_elem_size`) instead of a fixed
# bbox_diag/28 guess, since the latter misses badly on non-typical shapes
# (see module docstring). K_SURF calibrated against BUKSA/H17 + 2 synthetic
# boxes (see task-4-report.md).
# --------------------------------------------------------------------------
def test_elem_size_from_area_matches_formula():
    area_mm2, target_nodes = 2200.0, 500
    expected = np.sqrt(K_SURF * area_mm2 / target_nodes)
    assert _elem_size_from_area(area_mm2, target_nodes) == pytest.approx(expected)


def test_heal_and_mesh_area_seed_converges_on_first_attempt(tmp_path):
    """The calibrated K_SURF must make the 30x20x10mm box @ target_nodes=500
    converge within the +-35% band on the FIRST density-search attempt (no
    refinement rounds needed) -- this is the whole point of seeding from
    the OCC surface area instead of bbox_diag/28, which missed this same
    fixture badly (see the multi-round test below, forced via a bad
    monkeypatched K_SURF)."""
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30, 20, 10)

    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=500)

    assert result["ok"] is True
    assert result["attempts"] == 1
    assert 500 * 0.65 <= result["n_nodes"] <= 500 * 1.35


def test_first_guess_elem_size_falls_back_when_area_query_raises(monkeypatch):
    """If the OCC surface-area query raises, `_first_guess_elem_size` must
    fall back to the bbox_diag/28 heuristic rather than propagating the
    exception."""
    def _boom():
        raise RuntimeError("simulated gmsh area-query failure")

    monkeypatch.setattr(pp, "_total_surface_area_mm2", _boom)

    bbox_diag_mm = 37.4
    assert _first_guess_elem_size(bbox_diag_mm, 500) == pytest.approx(_elem_size(bbox_diag_mm))


def test_first_guess_elem_size_falls_back_when_area_nonpositive(monkeypatch):
    """If the OCC surface-area query returns <= 0 (no exception, just a
    degenerate/empty answer), `_first_guess_elem_size` must fall back to
    the bbox_diag/28 heuristic rather than dividing by (or sqrt-ing) a
    non-positive area."""
    monkeypatch.setattr(pp, "_total_surface_area_mm2", lambda: 0.0)

    bbox_diag_mm = 37.4
    assert _first_guess_elem_size(bbox_diag_mm, 500) == pytest.approx(_elem_size(bbox_diag_mm))


# --------------------------------------------------------------------------
# Heal-only-as-fallback regression tests (guards the bug fixed in 5220dbc:
# `healShapes()` can run without raising and still silently destroy the
# solid -- monkeypatched, deterministic, no real-part meshing).
# --------------------------------------------------------------------------
def test_heal_and_mesh_clean_part_never_calls_heal(tmp_path, monkeypatch):
    """A part that meshes fine on pass 1 must never invoke `_apply_heal` --
    this is the exact regression found in commit 5220dbc, where
    `healShapes()` was called unconditionally and silently reduced a
    perfectly meshable solid to 0 volume entities."""
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30, 20, 10)

    def _boom():
        raise AssertionError("_apply_heal must not be called when pass 1 succeeds")

    monkeypatch.setattr(pp, "_apply_heal", _boom)

    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=500)
    assert result["ok"] is True
    assert result["healed"] is False


def test_heal_and_mesh_ineffective_heal_reported_as_failed(tmp_path, monkeypatch):
    """If pass 1's mesh attempt fails and the pass-2 heal runs but does NOT
    yield a volume (the historical bug: `healShapes()` silently zeroing out
    the solid), `heal_and_mesh` must (a) actually attempt the heal, and
    (b) report ok=False / healed=False -- never silently accept a heal
    that produced no usable geometry."""
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30, 20, 10)

    monkeypatch.setattr(pp, "_tag_and_mesh",
                         lambda size, out_msh: (False, 0, "simulated mesh failure"))

    heal_calls = []

    def _fake_heal():
        heal_calls.append(1)
        return False, "simulated heal ineffective (0 volumes)"

    monkeypatch.setattr(pp, "_apply_heal", _fake_heal)

    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=500)

    assert len(heal_calls) == 1        # (a) heal WAS attempted after pass-1 failure
    assert result["ok"] is False       # (b) not silently accepted
    assert result["healed"] is False
    assert "simulated heal ineffective" in result["notes"]
    assert not out_msh.exists()


def test_heal_and_mesh_effective_heal_then_succeeds(tmp_path, monkeypatch):
    """If pass 1 fails but the pass-2 heal IS effective, pass 2 should
    retry meshing and succeed (ok=True, healed=True)."""
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30, 20, 10)

    calls = {"n": 0}

    def _fake_tag_and_mesh(size, out_msh):
        calls["n"] += 1
        if calls["n"] == 1:
            return False, 0, "simulated pass-1 failure"
        Path(out_msh).write_text("fake mesh")
        return True, 777, ""

    monkeypatch.setattr(pp, "_tag_and_mesh", _fake_tag_and_mesh)
    monkeypatch.setattr(pp, "_apply_heal", lambda: (True, ""))

    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=777)

    assert calls["n"] >= 2
    assert result["ok"] is True
    assert result["healed"] is True
    assert result["n_nodes"] == 777


# --------------------------------------------------------------------------
# Untagged-mesh regression (H17.STEP bug, found 2026-07-28): `_tag_and_mesh`
# can run more than once within a single gmsh session (density-search
# retries) -- `removePhysicalGroups()` frees the *tag number* but not the
# *name registration*, so on the 2nd+ call an auto-assigned new tag number
# silently fails to bind "Volume"/"Cooling" again. The fix pins explicit
# tag numbers (`_VOLUME_PHYS_TAG`/`_COOLING_PHYS_TAG`) across every call in
# a session, plus a loud-fail guard in case the binding fails for any other
# reason.
# --------------------------------------------------------------------------
def test_heal_and_mesh_multi_round_density_search_keeps_tags(tmp_path, monkeypatch):
    """A part whose first-attempt guess misses `target_nodes` by more than
    the 35% early-exit tolerance forces `_search_target_nodes` into
    `search_elem_size`'s multi-round refinement -- i.e. `_tag_and_mesh` runs
    MORE THAN ONCE in the same gmsh session (verified during triage:
    without the tag-pinning fix, this reliably ends with an untagged mesh
    -- `parse_msh(...).tris == {}` -- because only the FIRST round's
    physical names actually bind). This test's purpose is that TAG
    behavior, not seed quality -- since the calibrated K_SURF now makes
    this exact 30x20x10 @ target_nodes=500 fixture converge in ONE attempt
    (see `test_heal_and_mesh_area_seed_converges_on_first_attempt` above),
    K_SURF is monkeypatched to a deliberately bad value (50.0, which misses
    this fixture by >90% -- see task-4-report.md) purely to force the
    multi-round path back open. Regardless of how many rounds ran, the
    final written mesh must have a nonempty "Cooling" surface group and a
    nonempty tet volume."""
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30, 20, 10)

    monkeypatch.setattr(pp, "K_SURF", 50.0)

    calls = []
    orig_tag_and_mesh = pp._tag_and_mesh

    def _counting(size, out_msh):
        calls.append(size)
        return orig_tag_and_mesh(size, out_msh)

    monkeypatch.setattr(pp, "_tag_and_mesh", _counting)

    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=500)

    assert len(calls) >= 2, "test setup must force a multi-round density search"
    assert result["ok"] is True

    mesh = parse_msh(str(out_msh))
    assert "Cooling" in mesh.tris
    assert mesh.tris["Cooling"].shape[0] > 0
    assert mesh.tets.shape[0] > 0


# --------------------------------------------------------------------------
# Fast meshing options (mesh-accel Task 3): HXT (Mesh.Algorithm3D=10) +
# multithreading + curvature-refinement off, flag-gated via `fast=` and
# OFF by default -- these tests must pass with the default (fast=False)
# behavior completely unchanged.
# --------------------------------------------------------------------------
def test_heal_and_mesh_fast_true_synthetic_box(tmp_path):
    """fast=True must still produce a usable mesh: ok, a nonempty "Cooling"
    surface group, and node count within the same +-35% density-search
    band used elsewhere in this file for target_nodes=500 on this box."""
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30, 20, 10)

    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=500, fast=True)

    assert result["ok"] is True
    assert 500 * 0.65 <= result["n_nodes"] <= 500 * 1.35

    mesh = parse_msh(str(out_msh))
    assert "Cooling" in mesh.tris
    assert mesh.tris["Cooling"].shape[0] > 0
    assert mesh.tets.shape[0] > 0


def test_apply_fast_options_sets_hxt_threads_and_curvature_off():
    """Direct test of `_apply_fast_options()` inside an initialized gmsh
    session: HXT tet algorithm, >=1 threads, curvature-based refinement
    off."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        pp._apply_fast_options()
        assert gmsh.option.getNumber("Mesh.Algorithm3D") == 10
        assert gmsh.option.getNumber("General.NumThreads") >= 1
        assert gmsh.option.getNumber("Mesh.MeshSizeFromCurvature") == 0
    finally:
        gmsh.finalize()


def test_fast_options_not_applied_in_plain_session():
    """A plain fresh gmsh session (nothing fast-related called) must show
    gmsh's own, un-touched default for Mesh.Algorithm3D -- never HXT (10).
    Read BEFORE any fast call so this session can't have been polluted by
    another test in the same process."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        algo = gmsh.option.getNumber("Mesh.Algorithm3D")
        assert algo != 10
        assert algo == 1  # gmsh's documented default (classical Delaunay)
    finally:
        gmsh.finalize()


def test_heal_and_mesh_fast_default_never_calls_apply_fast_options(tmp_path, monkeypatch):
    """`fast` defaults to False -- `heal_and_mesh` must not invoke
    `_apply_fast_options` at all in that case (zero behavior change from
    before this flag existed)."""
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30, 20, 10)

    def _boom():
        raise AssertionError("_apply_fast_options must not be called when fast=False (default)")

    monkeypatch.setattr(pp, "_apply_fast_options", _boom)

    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=500)
    assert result["ok"] is True


# --------------------------------------------------------------------------
# Defeaturing pass (mesh-accel Task 5): best-effort `removeAllDuplicates()`
# simplification, flag-gated via `defeature=` and OFF by default -- see
# `_apply_defeature`'s docstring / task-5-report.md for the fuller virtual-
# topology investigation (why nothing beyond `removeAllDuplicates` shipped).
# --------------------------------------------------------------------------
def test_heal_and_mesh_defeature_true_synthetic_box(tmp_path):
    """defeature=True on a clean box must still succeed with a nonempty
    "Cooling" surface group and tet volume -- defeaturing a clean part must
    not break the Volume/Cooling tag guarantee."""
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30, 20, 10)

    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=500, defeature=True)

    assert result["ok"] is True
    assert result["n_nodes"] > 0

    mesh = parse_msh(str(out_msh))
    assert "Cooling" in mesh.tris
    assert mesh.tris["Cooling"].shape[0] > 0
    assert mesh.tets.shape[0] > 0


def test_heal_and_mesh_defeature_true_calls_apply_defeature(tmp_path, monkeypatch):
    """defeature=True must actually invoke `_apply_defeature` (in pass 1)."""
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30, 20, 10)

    calls = []
    orig = pp._apply_defeature

    def _wrap():
        calls.append(1)
        return orig()

    monkeypatch.setattr(pp, "_apply_defeature", _wrap)

    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=500, defeature=True)

    assert len(calls) >= 1
    assert result["ok"] is True


def test_heal_and_mesh_destructive_defeature_returns_ok_false_not_a_crash(tmp_path, monkeypatch):
    """Regression for the review finding (2026-07-29): if `_apply_defeature`
    reports the solid was lost (ok=False), `heal_and_mesh` must branch on
    that and return a clean `ok=False` dict with the solid-loss note --
    NOT fall through to `gmsh.model.getBoundingBox(-1, -1)` on an empty
    model, which raises an uncaught `Exception: Empty bounding box`."""
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30, 20, 10)

    monkeypatch.setattr(pp, "_apply_defeature",
                         lambda: (False, "defeature (removeAllDuplicates) yielded 0 volume entities"))

    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=500, defeature=True)

    assert result["ok"] is False
    assert "0 volume entities" in result["notes"]
    assert not out_msh.exists()


def test_heal_and_mesh_defeature_default_never_calls_apply_defeature(tmp_path, monkeypatch):
    """`defeature` defaults to False -- `heal_and_mesh` must not invoke
    `_apply_defeature` at all in that case (zero behavior change from
    before this flag existed)."""
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30, 20, 10)

    def _boom():
        raise AssertionError("_apply_defeature must not be called when defeature=False (default)")

    monkeypatch.setattr(pp, "_apply_defeature", _boom)

    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=500)
    assert result["ok"] is True


def test_apply_defeature_removes_duplicates_without_losing_solid():
    """Direct test of `_apply_defeature()` inside an initialized gmsh
    session: runs cleanly on a plain box and does not destroy the solid."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 30, 20, 10)
        gmsh.model.occ.synchronize()
        ok, note = pp._apply_defeature()
        assert ok is True
        assert note == ""
        assert len(gmsh.model.getEntities(3)) == 1
    finally:
        gmsh.finalize()


# --------------------------------------------------------------------------
# `keep_solid_id` (Part B Task 3): `_import_and_reduce`/`heal_and_mesh` can
# keep a CALLER-designated OCC solid instead of the largest-volume one --
# needed when the user has assigned solid roles and the "casting" is not
# necessarily the biggest body in the assembly (e.g. a small casting next to
# a bulky chill). `keep_solid_id=None` (default) must remain byte-identical
# to the pre-existing largest-solid behavior.
# --------------------------------------------------------------------------
def test_import_and_reduce_keep_solid_id_keeps_designated_solid(tmp_path):
    step = tmp_path / "two_box.step"
    _make_two_box_step(step, big=40.0, small=10.0)

    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part")
        # tag 2 is the SMALL box (see _make_two_box_step: big added first).
        n_solids, kept_vol, err = _import_and_reduce(str(step), keep_solid_id=2)
        assert err is None
        assert n_solids == 2
        assert kept_vol == pytest.approx(10.0 ** 3, rel=1e-3)
        vols = gmsh.model.occ.getEntities(3)
        assert len(vols) == 1
        assert vols[0][1] == 2
    finally:
        gmsh.finalize()


def test_import_and_reduce_keep_solid_id_none_keeps_largest_unchanged(tmp_path):
    """`keep_solid_id=None` (default) must reproduce the pre-existing
    largest-solid behavior exactly."""
    step = tmp_path / "two_box.step"
    _make_two_box_step(step, big=40.0, small=10.0)

    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part")
        n_solids, kept_vol, err = _import_and_reduce(str(step))
        assert err is None
        assert n_solids == 2
        assert kept_vol == pytest.approx(40.0 ** 3, rel=1e-3)
    finally:
        gmsh.finalize()


def test_import_and_reduce_keep_solid_id_missing_tag_returns_error_string(tmp_path):
    step = tmp_path / "two_box.step"
    _make_two_box_step(step, big=40.0, small=10.0)

    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("part")
        n_solids, kept_vol, err = _import_and_reduce(str(step), keep_solid_id=999)
        assert err is not None
        assert "999" in err
        assert n_solids == 0
        assert kept_vol == 0.0
    finally:
        gmsh.finalize()


def test_heal_and_mesh_keep_solid_id_meshes_designated_small_solid(tmp_path):
    """`heal_and_mesh(..., keep_solid_id=<small solid's tag>)` must mesh the
    SMALL solid (volume/node-count consistent with its own size), while the
    largest-solid default on the SAME input is untouched (still meshes the
    big one) -- regression coverage for both branches on one fixture."""
    step = tmp_path / "two_box.step"
    _make_two_box_step(step, big=40.0, small=10.0)

    out_msh_small = tmp_path / "small.msh"
    result_small = heal_and_mesh(str(step), str(out_msh_small), target_nodes=300,
                                  keep_solid_id=2)
    assert result_small["ok"] is True
    assert result_small["volume_mm3"] == pytest.approx(10.0 ** 3, rel=1e-2)
    assert "solid_id=2" in result_small["notes"]

    out_msh_default = tmp_path / "default.msh"
    result_default = heal_and_mesh(str(step), str(out_msh_default), target_nodes=300)
    assert result_default["ok"] is True
    assert result_default["volume_mm3"] == pytest.approx(40.0 ** 3, rel=1e-2)
    assert "kept largest" in result_default["notes"]


def test_heal_and_mesh_keep_solid_id_missing_tag_ok_false(tmp_path):
    step = tmp_path / "two_box.step"
    _make_two_box_step(step, big=40.0, small=10.0)

    out_msh = tmp_path / "missing.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=300, keep_solid_id=999)
    assert result["ok"] is False
    assert "999" in result["notes"]
    assert not out_msh.exists()


def test_tag_and_mesh_fails_loudly_when_name_binding_missing(tmp_path, monkeypatch):
    """Direct regression test for the loud-fail guard itself: if gmsh ever
    fails to bind the "Volume"/"Cooling" physical-group names (the exact
    silent-failure mode that produced H17's untagged mesh), `_tag_and_mesh`
    must return ok=False and must NOT write `out_msh` -- never silently
    hand back an untagged mesh."""
    import gmsh

    step = tmp_path / "box.step"
    _make_one_box_step(step, 30, 20, 10)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.importShapes(str(step))
        gmsh.model.occ.synchronize()

        monkeypatch.setattr(gmsh.model, "getPhysicalName", lambda dim, tag: "")

        out_msh = tmp_path / "box.msh"
        ok, n_nodes, note = pp._tag_and_mesh(2.0, str(out_msh))

        assert ok is False
        assert n_nodes == 0
        assert "name binding missing" in note
        assert not out_msh.exists()
    finally:
        gmsh.finalize()
