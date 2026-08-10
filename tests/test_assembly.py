"""Tests for src/assembly.py -- Part B Task 1: assembly inspection
(`list_solids`) + the `compose_assembly` test/fixture helper used by later
Part B fixtures (proximity classification, BC-patch tagging, etc.), and
Part B Task 2: `validate_bodies -> BcPlan` (no gmsh needed for Task 2)."""
from __future__ import annotations

import numpy as np
import pytest

from src.assembly import (BcPlan, list_solids, compose_assembly, main, validate_bodies,
                           tag_contact_patches, _triangle_areas, _rewrite_msh_adiabatic)
from src.msh_parse import parse_msh
from tests.test_prep_parts import _make_one_box_step
from tools.doe.load_properties import resolve_exo_q0_by_mef
from tools.doe.prep_parts import heal_and_mesh


def test_compose_and_list_two_solid_assembly(tmp_path):
    """Base box (30x20x10, the "casting") + a smaller box (10x10x6, a
    "chill") attached at the +z face. `compose_assembly` places the chill
    sunk `overlap_mm` into the base (default 2.0mm) -- NO boolean op, so the
    chill's own reported volume is its full analytic 10*10*6, unaffected by
    the intentional overlap with the base solid. `list_solids` must report
    both solids, sorted by volume descending (casting first)."""
    base_step = tmp_path / "base.step"
    _make_one_box_step(base_step, 30.0, 20.0, 10.0)

    out_step = tmp_path / "assembly.step"
    attachments = [
        {"shape": "box", "size": (10.0, 10.0, 6.0), "at_face": "+z"},
    ]
    n = compose_assembly(str(base_step), attachments, str(out_step))
    assert n == 2

    rows = list_solids(str(out_step))
    assert len(rows) == 2
    assert rows[0]["volume_mm3"] == pytest.approx(30.0 * 20.0 * 10.0, rel=0.05)
    assert rows[1]["volume_mm3"] == pytest.approx(10.0 * 10.0 * 6.0, rel=0.05)
    # sorted desc by volume
    assert rows[0]["volume_mm3"] >= rows[1]["volume_mm3"]
    for row in rows:
        assert set(row.keys()) == {"solid_id", "name", "volume_mm3", "bbox", "com"}
        assert len(row["bbox"]) == 6
        assert len(row["com"]) == 3


def test_list_solids_single_solid(tmp_path):
    """A plain single-solid STEP (reusing the existing box fixture from
    tests/test_prep_parts.py) must yield exactly 1 row."""
    step = tmp_path / "one_box.step"
    _make_one_box_step(step, 20.0, 20.0, 20.0)

    rows = list_solids(str(step))
    assert len(rows) == 1
    assert rows[0]["volume_mm3"] == pytest.approx(20.0 ** 3, rel=0.05)


def test_compose_assembly_cylinder_attachment(tmp_path):
    """Cylinder-shaped attachment (radius, height) at a different face,
    exercised end to end -- also confirms non-"+z" faces work."""
    base_step = tmp_path / "base.step"
    _make_one_box_step(base_step, 30.0, 20.0, 10.0)

    out_step = tmp_path / "assembly.step"
    attachments = [
        {"shape": "cylinder", "size": (3.0, 15.0), "at_face": "+x", "overlap_mm": 1.5},
    ]
    n = compose_assembly(str(base_step), attachments, str(out_step))
    assert n == 2

    rows = list_solids(str(out_step))
    assert len(rows) == 2
    import math
    cyl_vol = math.pi * 3.0 ** 2 * 15.0
    volumes = sorted(row["volume_mm3"] for row in rows)
    assert volumes[0] == pytest.approx(cyl_vol, rel=0.05)
    assert volumes[1] == pytest.approx(30.0 * 20.0 * 10.0, rel=0.05)


def test_main_prints_solid_ids(tmp_path, capsys):
    """`python -m src.assembly part.step` (exercised here via a direct
    `main()` call rather than a real subprocess, to keep the test fast)
    prints a table that includes both solids' ids."""
    base_step = tmp_path / "base.step"
    _make_one_box_step(base_step, 30.0, 20.0, 10.0)
    out_step = tmp_path / "assembly.step"
    compose_assembly(str(base_step),
                      [{"shape": "box", "size": (10.0, 10.0, 6.0), "at_face": "+z"}],
                      str(out_step))

    rows = list_solids(str(out_step))
    rc = main([str(out_step)])
    captured = capsys.readouterr()

    assert rc == 0
    for row in rows:
        assert str(row["solid_id"]) in captured.out


# ---------------------------------------------------------------------- #
# Part B Task 2 -- validate_bodies -> BcPlan (no gmsh; pure dict/CSV logic).
# All tests use the real, committed registries (data/properties/chills.csv,
# tools/sleeves/sleeve_registry.csv, data/properties/exo_calibration.csv)
# since those ARE the production registries and their values are pinned
# (see module docstrings): cast_iron effusivity 11968, sleeve "SD_4_7K"
# MEF 1.547 (2026-08-09 sleeve-mef-correction: catalog_klx/catalog_geom =
# 1.16/0.75, was the synthetic-spread value 1.49 before the fix -- see
# tools/sleeves/gen_sleeves.py's module docstring).
# ---------------------------------------------------------------------- #
def test_validate_bodies_casting_plus_two_chills():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "chill"},
        {"solid_id": 3, "role": "chill"},
    ]
    plan = validate_bodies(bodies)
    assert isinstance(plan, BcPlan)
    assert plan.bc_config == "chills"
    assert plan.casting_solid_id == 1
    assert sorted(plan.role_solid_ids) == [2, 3]
    assert plan.n_chills == 2
    assert plan.chill_effusivity == pytest.approx(11968.0)
    assert plan.sleeve_MEF == 0.0
    assert plan.exo_q0 == 0.0


def test_validate_bodies_chills_explicit_material_matches_default():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "chill", "params": {"material": "cast_iron"}},
    ]
    plan = validate_bodies(bodies)
    assert plan.bc_config == "chills"
    assert plan.n_chills == 1
    assert plan.chill_effusivity == pytest.approx(11968.0)


def test_validate_bodies_chills_differing_materials_raises():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "chill", "params": {"material": "cast_iron"}},
        {"solid_id": 3, "role": "chill", "params": {"material": "graphite"}},
    ]
    with pytest.raises(ValueError, match="all chills must share one material"):
        validate_bodies(bodies)


def test_validate_bodies_unknown_chill_material_raises_listing_options():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "chill", "params": {"material": "unobtainium"}},
    ]
    with pytest.raises(ValueError) as exc_info:
        validate_bodies(bodies)
    msg = str(exc_info.value)
    assert "unobtainium" in msg
    assert "cast_iron" in msg  # a known material is listed


def test_validate_bodies_exo_sleeve_by_registry_name():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "exo_sleeve", "params": {"sleeve": "SD_4_7K"}},
    ]
    plan = validate_bodies(bodies)
    assert plan.bc_config == "exo_sleeve"
    assert plan.role_solid_ids == [2]
    assert plan.sleeve_MEF == pytest.approx(1.547)
    assert plan.exo_q0 > 0.0
    assert plan.exo_q0 == pytest.approx(resolve_exo_q0_by_mef(1.547))


def test_validate_bodies_exo_sleeve_manual_mef_override():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "exo_sleeve", "params": {"MEF": 1.49}},
    ]
    plan = validate_bodies(bodies)
    assert plan.bc_config == "exo_sleeve"
    assert plan.sleeve_MEF == pytest.approx(1.49)
    assert plan.exo_q0 == pytest.approx(resolve_exo_q0_by_mef(1.49))


def test_validate_bodies_insulating_sleeve_no_exo_q0():
    # FOSECO_V90L is registry type=insulating -- matches role insulating_sleeve.
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "insulating_sleeve", "params": {"sleeve": "FOSECO_V90L"}},
    ]
    plan = validate_bodies(bodies)
    assert plan.bc_config == "insulating_sleeve"
    assert plan.sleeve_MEF == pytest.approx(1.15)
    assert plan.exo_q0 == 0.0


def test_validate_bodies_sleeve_both_name_and_mef_raises():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "exo_sleeve",
         "params": {"sleeve": "SD_4_7K", "MEF": 1.49}},
    ]
    with pytest.raises(ValueError, match="not both"):
        validate_bodies(bodies)


def test_validate_bodies_sleeve_neither_name_nor_mef_raises():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "exo_sleeve", "params": {}},
    ]
    with pytest.raises(ValueError, match="'sleeve'.*'MEF'|'MEF'.*'sleeve'"):
        validate_bodies(bodies)


def test_validate_bodies_exo_role_with_insulating_typed_sleeve_raises():
    # FOSECO_V88E is registry type=insulating, MEF=1.45 -- collides with an
    # exo_calibration.csv row at MEF=1.45, so without the type cross-check
    # this would silently resolve a confident nonzero exo_q0 for a
    # non-exothermic product.
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "exo_sleeve", "params": {"sleeve": "FOSECO_V88E"}},
    ]
    with pytest.raises(ValueError, match="FOSECO_V88E.*type=insulating"):
        validate_bodies(bodies)


def test_validate_bodies_insulating_role_with_exo_typed_sleeve_raises():
    # SD_4_7K is registry type=exo -- must not be accepted under
    # insulating_sleeve.
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "insulating_sleeve", "params": {"sleeve": "SD_4_7K"}},
    ]
    with pytest.raises(ValueError, match="SD_4_7K.*type=exo"):
        validate_bodies(bodies)


def test_validate_bodies_sleeve_type_matches_role_ok():
    # exo role + exo-typed sleeve, and insulating role + insulating-typed
    # sleeve, both resolve cleanly (matched pairs).
    exo_bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "exo_sleeve", "params": {"sleeve": "SD_4_7K"}},
    ]
    plan = validate_bodies(exo_bodies)
    assert plan.bc_config == "exo_sleeve"
    assert plan.sleeve_MEF == pytest.approx(1.547)

    insulating_bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "insulating_sleeve", "params": {"sleeve": "FOSECO_V88E"}},
    ]
    plan2 = validate_bodies(insulating_bodies)
    assert plan2.bc_config == "insulating_sleeve"
    assert plan2.sleeve_MEF == pytest.approx(1.45)


def test_validate_bodies_manual_mef_bypasses_type_check_either_role():
    # Manual MEF override has no registry row to check against -- expert
    # mode, allowed under either sleeve role.
    exo_bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "exo_sleeve", "params": {"MEF": 1.45}},
    ]
    plan = validate_bodies(exo_bodies)
    assert plan.bc_config == "exo_sleeve"
    assert plan.sleeve_MEF == pytest.approx(1.45)

    insulating_bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "insulating_sleeve", "params": {"MEF": 1.49}},
    ]
    plan2 = validate_bodies(insulating_bodies)
    assert plan2.bc_config == "insulating_sleeve"
    assert plan2.sleeve_MEF == pytest.approx(1.49)
    assert plan2.exo_q0 == 0.0


def test_validate_bodies_unknown_sleeve_name_raises_listing_options():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "exo_sleeve", "params": {"sleeve": "NOT_A_SLEEVE"}},
    ]
    with pytest.raises(ValueError) as exc_info:
        validate_bodies(bodies)
    msg = str(exc_info.value)
    assert "NOT_A_SLEEVE" in msg
    assert "SD_4_7K" in msg


def test_validate_bodies_casting_only_bc_none():
    bodies = [{"solid_id": 1, "role": "casting"}]
    plan = validate_bodies(bodies)
    assert plan.bc_config == "none"
    assert plan.casting_solid_id == 1
    assert plan.role_solid_ids == []
    assert plan.n_chills == 0
    assert plan.chill_effusivity == 0.0
    assert plan.sleeve_MEF == 0.0
    assert plan.exo_q0 == 0.0


def test_validate_bodies_ignore_bodies_dropped():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "ignore"},
        {"solid_id": 3, "role": "ignore", "params": {"whatever": 1}},
    ]
    plan = validate_bodies(bodies)
    assert plan.bc_config == "none"
    assert plan.casting_solid_id == 1


def test_validate_bodies_zero_castings_raises():
    bodies = [{"solid_id": 2, "role": "chill"}]
    with pytest.raises(ValueError, match="exactly one"):
        validate_bodies(bodies)


def test_validate_bodies_two_castings_raises():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "casting"},
    ]
    with pytest.raises(ValueError, match="exactly one"):
        validate_bodies(bodies)


def test_validate_bodies_chills_plus_sleeve_mixed_raises():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "chill"},
        {"solid_id": 3, "role": "exo_sleeve", "params": {"sleeve": "SD_4_7K"}},
    ]
    with pytest.raises(ValueError, match="chills and a sleeve cannot be combined"):
        validate_bodies(bodies)


def test_validate_bodies_mixed_sleeve_roles_raises():
    # SP4 ADDENDUM (2026-08-08): N sleeve bodies are accepted now, but only
    # when they share one role -- an exo_sleeve + an insulating_sleeve still
    # raises, because the model has a single sleeve BC channel and cannot
    # represent both kinds of sleeve heat exchange at once.
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "exo_sleeve", "params": {"sleeve": "SD_4_7K"}},
        {"solid_id": 3, "role": "insulating_sleeve", "params": {"sleeve": "FOSECO_V90L"}},
    ]
    with pytest.raises(ValueError, match="one role"):
        validate_bodies(bodies)


def test_validate_bodies_two_same_role_sleeves_resolve():
    # SP4 ADDENDUM (2026-08-08): two sleeve bodies of the SAME role and the
    # SAME registry-resolved MEF must now be accepted (this used to raise
    # "at most one sleeve body is supported"), resolving to that single MEF
    # exactly and covering both solid_ids in role_solid_ids.
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "exo_sleeve", "params": {"sleeve": "SD_4_7K"}},
        {"solid_id": 3, "role": "exo_sleeve", "params": {"sleeve": "SD_4_7K"}},
    ]
    plan = validate_bodies(bodies)
    assert plan.bc_config == "exo_sleeve"
    assert sorted(plan.role_solid_ids) == [2, 3]
    assert plan.sleeve_MEF == pytest.approx(1.547)
    assert plan.mef_weighted is False
    assert plan.mef_spread == pytest.approx(0.0)


def test_validate_bodies_unknown_role_raises():
    bodies = [
        {"solid_id": 1, "role": "casting"},
        {"solid_id": 2, "role": "typo_role"},
    ]
    with pytest.raises(ValueError, match="unknown role"):
        validate_bodies(bodies)


def test_bcplan_feature_spec_keys_and_values():
    plan = BcPlan(bc_config="chills", casting_solid_id=1, role_solid_ids=[2],
                  n_chills=1, chill_effusivity=11968.0)
    spec = plan.feature_spec(mold_effusivity=1250.0)
    assert spec == {
        "mold_effusivity": 1250.0,
        "bc_config": "chills",
        "chill_effusivity": 11968.0,
        "sleeve_MEF": 0.0,
        "exo_q0": 0.0,
    }


def test_bcplan_gbm_bc_spec_keys_and_exo_q0_mw_conversion():
    plan = BcPlan(bc_config="exo_sleeve", casting_solid_id=1, role_solid_ids=[2],
                  sleeve_MEF=1.49, exo_q0=960000.0)
    spec = plan.gbm_bc_spec()
    assert spec == {
        "bc_config": "exo_sleeve",
        "n_chills": 0,
        "sleeve_MEF": 1.49,
        "exo_q0_MW": pytest.approx(0.96),
    }


# ---------------------------------------------------------------------- #
# Part B Task 3 -- tag_contact_patches: mesh the user-designated casting
# solid (via prep_parts' new `keep_solid_id`), classify its "Cooling"
# triangles by proximity to role bodies, and rewrite the mesh's Adiabatic
# facet group. Real gmsh throughout -- these build a fresh composer fixture
# each time (see `src.assembly.compose_assembly`'s module docstring for why
# a few mm of intentional overlap is the fixture convention, not a bug).
# ---------------------------------------------------------------------- #
def _elem_est_mm(mesh) -> float:
    """Same `bbox_diag / n_nodes**(1/3)` local element-size estimate
    `tag_contact_patches` uses for its default `tol_mm` -- used here only to
    size test assertion margins, not to drive the function itself."""
    bbox_diag = float(np.linalg.norm(mesh.node_xyz.max(axis=0) - mesh.node_xyz.min(axis=0)))
    return bbox_diag / (mesh.node_xyz.shape[0] ** (1.0 / 3.0))


def _build_chill_assembly(tmp_path, base_size=(30.0, 20.0, 10.0), chill_size=(10.0, 10.0, 6.0),
                           overlap_mm=2.0, target_nodes=500):
    """Base box (the "casting") + a smaller box (the "chill") attached at
    the "+z" face, exactly the `compose_assembly` fixture pattern used
    throughout this file. Meshes ONLY the casting (`keep_solid_id`, Task 3's
    other new capability -- the casting need not be the largest solid in
    general, though it is here) and returns
    `(casting_msh_path, assembly_step_path, casting_id, chill_id,
    heal_and_mesh_result)`."""
    base_step = tmp_path / "base.step"
    _make_one_box_step(base_step, *base_size)

    assembly_step = tmp_path / "assembly.step"
    attachments = [{"shape": "box", "size": chill_size, "at_face": "+z", "overlap_mm": overlap_mm}]
    compose_assembly(str(base_step), attachments, str(assembly_step))

    rows = list_solids(str(assembly_step))
    casting_id = max(rows, key=lambda r: r["volume_mm3"])["solid_id"]
    chill_id = min(rows, key=lambda r: r["volume_mm3"])["solid_id"]

    out_msh = tmp_path / "casting.msh"
    result = heal_and_mesh(str(assembly_step), str(out_msh), target_nodes=target_nodes,
                            keep_solid_id=casting_id)
    assert result["ok"] is True
    return str(out_msh), str(assembly_step), casting_id, chill_id, result


def test_tag_contact_patches_exact_contact_matches_chill_footprint(tmp_path):
    """Overlap 2mm box chill at "+z" center (compose_assembly's own default
    overlap) -- the Adiabatic patch must land on the chill's footprint: every
    Adiabatic centroid within the footprint's xy extent (+- a couple of
    element widths, since a nonzero tol_mm necessarily dilates the detected
    region a little past the true edge) and at the casting's top z; total
    patch area within 30% of the chill's analytic contact-face area. A
    large-relative-to-tol footprint (40x40mm chill on a 60x50x15mm base) is
    used so the fixed ~2mm overlap gap doesn't force a tol_mm whose
    Minkowski-dilation ring is a big fraction of a small footprint's area
    (verified numerically while writing this test: a 10x10mm footprint blows
    well past 30% at any tol_mm large enough to bridge a 2mm gap)."""
    base_size = (60.0, 50.0, 15.0)
    chill_size = (40.0, 40.0, 6.0)
    out_msh, assembly_step, casting_id, chill_id, _result = _build_chill_assembly(
        tmp_path, base_size=base_size, chill_size=chill_size, overlap_mm=2.0,
        target_nodes=3000)

    mesh_before = parse_msh(out_msh)
    elem_est = _elem_est_mm(mesh_before)
    tol_mm = 2.5   # > the 2mm overlap gap, small relative to the 40x40 footprint

    tag_result = tag_contact_patches(out_msh, assembly_step, [chill_id], tol_mm=tol_mm)

    assert tag_result["n_adiab_tris"] > 0
    chill_contact_area = chill_size[0] * chill_size[1]   # dx * dy of the chill footprint
    assert tag_result["per_body_patch_area_mm2"][chill_id] == pytest.approx(
        chill_contact_area, rel=0.30)

    mesh_after = parse_msh(out_msh)
    adiab = mesh_after.tris["Adiabatic"]
    assert adiab.shape[0] == tag_result["n_adiab_tris"]
    centroids = mesh_after.node_xyz[adiab].mean(axis=1)

    # Footprint is centered on the base's top face: x in [10, 50], y in [5, 45]
    # (base 60x50, chill 40x40 centered -> (60-40)/2=10, (50-40)/2=5).
    margin = tol_mm + elem_est
    fx_lo, fx_hi = (base_size[0] - chill_size[0]) / 2.0, (base_size[0] + chill_size[0]) / 2.0
    fy_lo, fy_hi = (base_size[1] - chill_size[1]) / 2.0, (base_size[1] + chill_size[1]) / 2.0
    assert np.all(centroids[:, 0] >= fx_lo - margin)
    assert np.all(centroids[:, 0] <= fx_hi + margin)
    assert np.all(centroids[:, 1] >= fy_lo - margin)
    assert np.all(centroids[:, 1] <= fy_hi + margin)
    # at the casting's top face
    assert np.allclose(centroids[:, 2], base_size[2], atol=margin)


def test_tag_contact_patches_nodes_by_solid_matches_adiabatic_group(tmp_path):
    """Part-B roles task (2026-08-03): `nodes_by_solid` -- the in-memory
    per-solid contact-patch node-id mapping `tag_contact_patches` now also
    returns -- must, for a single role body, exactly equal the node ids of
    the "Adiabatic" group the SAME call wrote to `out_msh` (with only one
    body, that body's own `close` mask IS the Adiabatic union)."""
    out_msh, assembly_step, casting_id, chill_id, _result = _build_chill_assembly(
        tmp_path, overlap_mm=2.0, target_nodes=500)

    tag_result = tag_contact_patches(out_msh, assembly_step, [chill_id], tol_mm=3.0)

    assert set(tag_result["nodes_by_solid"]) == {chill_id}
    mesh_after = parse_msh(out_msh)
    adiab_nodes = sorted(set(mesh_after.tris["Adiabatic"].reshape(-1).tolist()))
    assert tag_result["nodes_by_solid"][chill_id] == adiab_nodes
    # sorted, unique, and does not mutate the .msh/node-order contract --
    # covered independently by test_tag_contact_patches_preserves_node_
    # order_and_tets; here just confirm it's genuinely sorted+unique.
    assert tag_result["nodes_by_solid"][chill_id] == sorted(
        set(tag_result["nodes_by_solid"][chill_id]))


def test_tag_contact_patches_small_clearance_still_found(tmp_path):
    """0.1mm CLEARANCE (overlap_mm=-0.1, a hair short of touching) with
    tol_mm >= 0.5mm must still find the patch."""
    out_msh, assembly_step, casting_id, chill_id, _result = _build_chill_assembly(
        tmp_path, overlap_mm=-0.1, target_nodes=500)

    tag_result = tag_contact_patches(out_msh, assembly_step, [chill_id], tol_mm=0.5)

    assert tag_result["n_adiab_tris"] > 0
    assert tag_result["per_body_patch_area_mm2"][chill_id] > 0.0


def test_tag_contact_patches_far_body_raises(tmp_path):
    """A body 10mm away with tol_mm=1mm must raise -- never a silent
    all-sand prediction for a body the caller explicitly assigned a role
    to."""
    out_msh, assembly_step, casting_id, chill_id, _result = _build_chill_assembly(
        tmp_path, overlap_mm=-10.0, target_nodes=500)

    with pytest.raises(ValueError, match=f"body {chill_id} does not touch"):
        tag_contact_patches(out_msh, assembly_step, [chill_id], tol_mm=1.0)


def test_tag_contact_patches_preserves_node_order_and_tets(tmp_path):
    """The rewritten mesh's node ids/coords and tet connectivity must be
    EXACTLY unchanged -- features/geodesics depend on it (see the module
    docstring's node-order caution). Parses before AND after, independently
    of `tag_contact_patches`'s own internal round-trip assertion."""
    out_msh, assembly_step, casting_id, chill_id, _result = _build_chill_assembly(
        tmp_path, overlap_mm=2.0, target_nodes=500)

    mesh_before = parse_msh(out_msh)

    tag_contact_patches(out_msh, assembly_step, [chill_id], tol_mm=3.0)

    mesh_after = parse_msh(out_msh)
    assert np.allclose(mesh_after.node_xyz, mesh_before.node_xyz)
    assert mesh_after.tets.shape == mesh_before.tets.shape
    assert np.array_equal(mesh_after.tets, mesh_before.tets)
    # sanity: some Cooling triangles actually migrated to Adiabatic
    assert "Adiabatic" in mesh_after.tris
    assert mesh_after.tris["Cooling"].shape[0] < mesh_before.tris["Cooling"].shape[0]
    assert (mesh_before.tris["Cooling"].shape[0]
            == mesh_after.tris["Cooling"].shape[0] + mesh_after.tris["Adiabatic"].shape[0])


def test_tag_contact_patches_unknown_role_solid_id_raises(tmp_path):
    """A role `solid_id` that doesn't exist in `step_path` at all must fail
    loudly (whatever form that takes -- gmsh itself raises when asked for
    the boundary of a nonexistent entity), never silently produce an empty
    or wrong patch."""
    out_msh, assembly_step, casting_id, chill_id, _result = _build_chill_assembly(tmp_path)

    with pytest.raises(Exception):
        tag_contact_patches(out_msh, assembly_step, [9999], tol_mm=3.0)


def test_heal_and_mesh_keep_solid_id_meshes_small_chill_from_assembly(tmp_path):
    """Part B Task 3's other change: `heal_and_mesh(..., keep_solid_id=...)`
    meshes the SMALL (chill) solid out of a composed multi-body assembly
    (n_nodes/volume consistent with its own size), while the largest-solid
    default on the SAME input is untouched (still meshes the big casting)."""
    base_step = tmp_path / "base.step"
    _make_one_box_step(base_step, 30.0, 20.0, 10.0)
    assembly_step = tmp_path / "assembly.step"
    compose_assembly(str(base_step),
                      [{"shape": "box", "size": (10.0, 10.0, 6.0), "at_face": "+z"}],
                      str(assembly_step))

    rows = list_solids(str(assembly_step))
    casting_row = max(rows, key=lambda r: r["volume_mm3"])
    chill_row = min(rows, key=lambda r: r["volume_mm3"])

    out_msh_chill = tmp_path / "chill.msh"
    result_chill = heal_and_mesh(str(assembly_step), str(out_msh_chill), target_nodes=300,
                                  keep_solid_id=chill_row["solid_id"])
    assert result_chill["ok"] is True
    assert result_chill["volume_mm3"] == pytest.approx(chill_row["volume_mm3"], rel=0.05)

    out_msh_default = tmp_path / "default.msh"
    result_default = heal_and_mesh(str(assembly_step), str(out_msh_default), target_nodes=300)
    assert result_default["ok"] is True
    assert result_default["volume_mm3"] == pytest.approx(casting_row["volume_mm3"], rel=0.05)


# ---------------------------------------------------------------------- #
# Code-review follow-ups (2026-07-29): total-Cooling-consumption guard,
# per_body_patch_area_mm2's documented double-counting behavior, and the
# clean (no bare StopIteration) too-few-flags path in
# `_rewrite_msh_adiabatic`.
# ---------------------------------------------------------------------- #
def test_tag_contact_patches_enveloping_body_raises_no_cooling_remains(tmp_path):
    """A role body that tightly ENVELOPES the entire casting (every Cooling
    triangle ends up within tol_mm of it) must raise loudly -- if every
    triangle were silently relabeled Adiabatic, `geodesic_d_cool` (see
    `src.features`) would have no "Cooling" surface left to measure
    distance to and would silently degrade to bbox_diag-derived constants
    (confidently wrong physics, not a loud failure)."""
    base_step = tmp_path / "base.step"
    _make_one_box_step(base_step, 30.0, 20.0, 10.0)

    assembly_step = tmp_path / "assembly.step"
    # A box attached at "+z" with overlap_mm >= its own along-normal size
    # sinks past the base's bottom too, and a face-plane size well past the
    # base's own x/y extent -- the result envelopes the base on all 6 sides
    # with a uniform ~5mm margin (worked out analytically: dz=20,
    # overlap_mm=15 -> z in [-5, 15] vs the base's [0, 10]; dx=40, dy=30
    # centered on the base's own center -> x in [-5, 35] vs [0, 30], y in
    # [-5, 25] vs [0, 20]).
    envelope = {"shape": "box", "size": (40.0, 30.0, 20.0), "at_face": "+z", "overlap_mm": 15.0}
    compose_assembly(str(base_step), [envelope], str(assembly_step))

    rows = list_solids(str(assembly_step))
    casting_id = max(rows, key=lambda r: r["volume_mm3"])["solid_id"]
    envelope_id = min(rows, key=lambda r: r["volume_mm3"])["solid_id"]

    out_msh = tmp_path / "casting.msh"
    result = heal_and_mesh(str(assembly_step), str(out_msh), target_nodes=500,
                            keep_solid_id=casting_id)
    assert result["ok"] is True

    with pytest.raises(ValueError, match="cover the entire casting surface"):
        tag_contact_patches(str(out_msh), str(assembly_step), [envelope_id], tol_mm=8.0)


def test_tag_contact_patches_two_bodies_double_count_area_single_count_tris(tmp_path):
    """Two role bodies whose tolerance zones overlap on the casting surface:
    `per_body_patch_area_mm2` sums each body's own area INDEPENDENTLY (the
    shared triangles counted once per body -- documented double-counting),
    while `n_adiab_tris` (and the actual rewritten "Adiabatic" group) counts
    each such triangle exactly once, via the union mask."""
    base_step = tmp_path / "base.step"
    _make_one_box_step(base_step, 60.0, 50.0, 15.0)

    assembly_step = tmp_path / "assembly.step"
    # Two 40x40x6 chills at "+z", offset +-10mm in x -- footprints
    # [0,40] and [20,60] in x (both y in [5,45]) overlap in a 20mm-wide
    # x in [20,40] band.
    attachments = [
        {"shape": "box", "size": (40.0, 40.0, 6.0), "at_face": "+z", "offset": (-10.0, 0.0)},
        {"shape": "box", "size": (40.0, 40.0, 6.0), "at_face": "+z", "offset": (10.0, 0.0)},
    ]
    compose_assembly(str(base_step), attachments, str(assembly_step))

    rows = list_solids(str(assembly_step))
    casting_id = max(rows, key=lambda r: r["volume_mm3"])["solid_id"]
    role_ids = sorted(r["solid_id"] for r in rows if r["solid_id"] != casting_id)
    assert len(role_ids) == 2

    out_msh = tmp_path / "casting.msh"
    result = heal_and_mesh(str(assembly_step), str(out_msh), target_nodes=3000,
                            keep_solid_id=casting_id)
    assert result["ok"] is True

    tag_result = tag_contact_patches(str(out_msh), str(assembly_step), role_ids, tol_mm=2.5)

    areas = tag_result["per_body_patch_area_mm2"]
    assert set(areas) == set(role_ids)
    for solid_id in role_ids:
        assert areas[solid_id] > 0.0
    # symmetric footprints (mirrored offsets) -> roughly equal per-body areas
    assert areas[role_ids[0]] == pytest.approx(areas[role_ids[1]], rel=0.05)

    mesh_after = parse_msh(str(out_msh))
    adiab = mesh_after.tris["Adiabatic"]
    assert adiab.shape[0] == tag_result["n_adiab_tris"]

    true_union_area = float(_triangle_areas(mesh_after.node_xyz, adiab).sum())
    # The overlap region's triangles are counted in BOTH bodies' areas, so
    # the sum across bodies must exceed the true (single-counted) union
    # area -- pins the documented double-counting behavior.
    assert sum(areas.values()) > true_union_area * 1.10

    # nodes_by_solid (Part-B roles task): each body's own node set may share
    # nodes with the other's (the overlap band's shared triangles), but the
    # UNION across both bodies' node sets must equal exactly the true
    # Adiabatic node set -- no node is dropped or invented.
    assert set(tag_result["nodes_by_solid"]) == set(role_ids)
    nodes_union = set()
    for solid_id in role_ids:
        nodes_union.update(tag_result["nodes_by_solid"][solid_id])
    assert nodes_union == set(adiab.reshape(-1).tolist())


def test_rewrite_msh_adiabatic_too_few_flags_raises_clean_assertion_error(tmp_path):
    """A flag array SHORTER than the number of Cooling-tagged triangle
    elements in the file must raise a clean `AssertionError` -- not a bare
    `StopIteration` (the old `next(flags)`-inside-the-loop bug: exhausting
    an iterator this way raises `StopIteration`, which is not a deliberate,
    documented failure mode)."""
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30.0, 20.0, 10.0)
    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=300)
    assert result["ok"] is True

    n_cooling = parse_msh(str(out_msh)).tris["Cooling"].shape[0]
    too_few = np.zeros(n_cooling - 1, dtype=bool)   # one short

    with pytest.raises(AssertionError, match="fewer"):
        _rewrite_msh_adiabatic(str(out_msh), too_few)


def test_rewrite_msh_adiabatic_too_many_flags_raises_clean_assertion_error(tmp_path):
    step = tmp_path / "box.step"
    _make_one_box_step(step, 30.0, 20.0, 10.0)
    out_msh = tmp_path / "box.msh"
    result = heal_and_mesh(str(step), str(out_msh), target_nodes=300)
    assert result["ok"] is True

    n_cooling = parse_msh(str(out_msh)).tris["Cooling"].shape[0]
    too_many = np.zeros(n_cooling + 1, dtype=bool)   # one extra

    with pytest.raises(AssertionError, match="more flag"):
        _rewrite_msh_adiabatic(str(out_msh), too_many)
