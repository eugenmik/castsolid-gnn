"""Multi-sleeve resolution (SP4 spec ADDENDUM).

Real castings carry several sleeves; the pre-SP4 code accepted exactly one.
These tests pin the three cases: identical sleeves resolve exactly, differing
sleeves need explicit weights, and mixed sleeve ROLES still raise.
"""
import pytest

from src.assembly import validate_bodies

CASTING = {"solid_id": 1, "role": "casting"}


def _exo(solid_id, sleeve):
    return {"solid_id": solid_id, "role": "exo_sleeve", "params": {"sleeve": sleeve}}


def test_three_identical_sleeves_resolve_exactly():
    plan = validate_bodies([CASTING, _exo(2, "SD_4_7K"), _exo(3, "SD_4_7K"),
                            _exo(4, "SD_4_7K")])
    assert plan.bc_config == "exo_sleeve"
    assert sorted(plan.role_solid_ids) == [2, 3, 4]
    assert plan.sleeve_MEF == pytest.approx(1.547)
    assert plan.mef_weighted is False
    assert plan.mef_spread == pytest.approx(0.0)
    assert plan.exo_q0 > 0.0
    assert plan.role_by_solid == {2: "exo_sleeve", 3: "exo_sleeve", 4: "exo_sleeve"}


def test_differing_mef_without_weights_raises():
    with pytest.raises(ValueError, match="differing MEF"):
        validate_bodies([CASTING, _exo(2, "SD_4_7K"), _exo(3, "SD_5_8K")])


def test_differing_mef_error_points_at_the_two_pass_path():
    """Review I3: `predict_casting` cannot supply `mef_weights` (contact-
    patch areas don't exist until after meshing/tagging), so the message a
    user actually sees must not just name the missing parameter -- it must
    say where areas come from and which callers already do the two-pass
    resolution correctly."""
    with pytest.raises(ValueError, match="tag_contact_patches") as exc:
        validate_bodies([CASTING, _exo(2, "SD_4_7K"), _exo(3, "SD_5_8K")])
    msg = str(exc.value)
    assert "tools/sp4/prep.py" in msg
    assert "tools/sp4/score.py" in msg


def test_differing_mef_area_weighted():
    # SD_4_7K MEF 1.547 over 300 mm2, SD_5_8K MEF 1.579 over 100 mm2
    # (2026-08-09 sleeve-mef-correction: catalog_klx/catalog_geom, was
    # 1.49/1.63 before the fix) -> (1.547*300 + 1.579*100) / 400 = 1.555
    plan = validate_bodies([CASTING, _exo(2, "SD_4_7K"), _exo(3, "SD_5_8K")],
                           mef_weights={2: 300.0, 3: 100.0})
    assert plan.sleeve_MEF == pytest.approx(1.555)
    assert plan.mef_weighted is True
    assert plan.mef_spread == pytest.approx(1.579 - 1.547)
    assert sorted(plan.role_solid_ids) == [2, 3]


def test_weights_missing_a_sleeve_raises():
    with pytest.raises(ValueError, match="mef_weights"):
        validate_bodies([CASTING, _exo(2, "SD_4_7K"), _exo(3, "SD_5_8K")],
                        mef_weights={2: 300.0})


def test_nonpositive_weight_raises():
    with pytest.raises(ValueError, match="positive"):
        validate_bodies([CASTING, _exo(2, "SD_4_7K"), _exo(3, "SD_5_8K")],
                        mef_weights={2: 300.0, 3: 0.0})


def test_mixed_sleeve_roles_still_raise():
    bodies = [CASTING, _exo(2, "SD_4_7K"),
              {"solid_id": 3, "role": "insulating_sleeve",
               "params": {"sleeve": "FOSECO_V90L"}}]
    with pytest.raises(ValueError, match="one role"):
        validate_bodies(bodies)


def test_single_sleeve_unchanged():
    """The pre-SP4 single-sleeve path must be byte-identical."""
    plan = validate_bodies([CASTING, _exo(2, "SD_4_7K")])
    assert plan.role_solid_ids == [2]
    assert plan.sleeve_MEF == pytest.approx(1.547)
    assert plan.mef_weighted is False
    assert plan.mef_spread == pytest.approx(0.0)


def test_manual_mef_override_across_several_sleeves():
    bodies = [CASTING,
              {"solid_id": 2, "role": "exo_sleeve", "params": {"MEF": 1.49}},
              {"solid_id": 3, "role": "exo_sleeve", "params": {"MEF": 1.49}}]
    plan = validate_bodies(bodies)
    assert plan.sleeve_MEF == pytest.approx(1.49)
    assert plan.mef_weighted is False


def test_chills_path_unchanged():
    bodies = [CASTING,
              {"solid_id": 2, "role": "chill", "params": {"material": "cast_iron"}},
              {"solid_id": 3, "role": "chill", "params": {"material": "cast_iron"}}]
    plan = validate_bodies(bodies)
    assert plan.bc_config == "chills"
    assert plan.n_chills == 2
    assert plan.mef_weighted is False
