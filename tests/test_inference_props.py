import json, pytest
from src.inference_props import resolve_props, list_alloys, list_molds, DEFAULT_ALLOY, \
    resolve_alloy_family
from src.shrinkage import SHRINKAGE_PCT

REQ = {"k", "rho", "cp", "L_kJ_kg", "TL", "TS", "Tpour", "mold_effusivity"}

def test_default_is_gjs400_green_sand():
    p = resolve_props()
    assert set(p) == REQ and all(isinstance(v, float) for v in p.values())
    assert p["TL"] == 1160.0 and p["TS"] == 1120.0          # EN-GJS-400-15 row
    assert p["Tpour"] == pytest.approx((1250 + 1270) / 2)    # midpoint rule
    assert p["mold_effusivity"] == pytest.approx(1018.0)     # molds.csv, NOT 1250

def test_named_alloy_and_lists():
    assert DEFAULT_ALLOY in list_alloys()
    assert list_alloys() == ["EN-GJS-400-15", "G20Mn5"]
    assert list_molds() == ["green_sand"]
    p = resolve_props(alloy="G20Mn5")
    assert p["TS"] == 1495.0

def test_unknown_names_raise_listing_options():
    with pytest.raises(ValueError, match="EN-GJS-400-15"):
        resolve_props(alloy="unobtainium")
    with pytest.raises(ValueError, match="green_sand"):
        resolve_props(mold="vacuum")

def test_props_json_override_and_missing_keys(tmp_path):
    full = dict(k=30.0, rho=7000.0, cp=500.0, L_kJ_kg=220.0, TL=1200.0, TS=1120.0,
                Tpour=1350.0, mold_effusivity=900.0)
    f = tmp_path / "p.json"; f.write_text(json.dumps(full))
    assert resolve_props(props_json=str(f)) == full
    f2 = tmp_path / "p2.json"; f2.write_text(json.dumps({"k": 30.0}))
    with pytest.raises(ValueError, match="L_kJ_kg"):
        resolve_props(props_json=str(f2))


# --------------------------------------------------------------------------- #
# Review finding I2 (final whole-branch review, 2026-08-01): a `--props`
# custom-JSON override with TL <= TS was previously unvalidated -- silently
# collapsing src.shrinkage.niyama_proxy to 0 everywhere (100% flagged).
# --------------------------------------------------------------------------- #
def test_props_json_tl_less_equal_ts_raises(tmp_path):
    inverted = dict(k=30.0, rho=7000.0, cp=500.0, L_kJ_kg=220.0, TL=1100.0, TS=1200.0,
                    Tpour=1350.0, mold_effusivity=900.0)
    f = tmp_path / "inverted.json"; f.write_text(json.dumps(inverted))
    with pytest.raises(ValueError, match=r"TL.*1100.*TS.*1200|TL.*must be strictly greater"):
        resolve_props(props_json=str(f))


def test_props_json_tl_equal_ts_raises(tmp_path):
    # degenerate (TL == TS), not just inverted -- niyama_proxy's tdot_proxy
    # would be exactly (TL-TS)/t_s == 0 for every node either way.
    degenerate = dict(k=30.0, rho=7000.0, cp=500.0, L_kJ_kg=220.0, TL=1150.0, TS=1150.0,
                      Tpour=1350.0, mold_effusivity=900.0)
    f = tmp_path / "degenerate.json"; f.write_text(json.dumps(degenerate))
    with pytest.raises(ValueError, match="TL"):
        resolve_props(props_json=str(f))


def test_named_alloy_tl_ts_still_valid_after_i2_guard():
    # sanity: the new guard must not break any real named grade (already
    # validated by tools.doe.load_properties.load_materials's own TS<TL<Tpour
    # check at CSV-load time -- this just proves resolve_props's OWN check
    # doesn't double-reject a legitimately-ordered grade).
    for grade in list_alloys():
        p = resolve_props(alloy=grade)
        assert p["TL"] > p["TS"]


# --------------------------------------------------------------------------- #
# resolve_alloy_family (SP2 shrinkage Task 4, controller directive 2):
# resolve_props's returned 8-key dict has no family field, so shrinkage
# screening maps the ALLOY GRADE NAME to one of src.shrinkage.SHRINKAGE_PCT's
# family keys via a small name-prefix helper, rather than crashing or
# silently guessing on an unrecognized grade.
# --------------------------------------------------------------------------- #
def test_resolve_alloy_family_known_prefixes():
    assert resolve_alloy_family("EN-GJL-250") == ("grey_iron", False)
    assert resolve_alloy_family("EN-GJL-200") == ("grey_iron", False)
    assert resolve_alloy_family("EN-GJS-400-15") == ("ductile_iron", False)
    assert resolve_alloy_family("GS-38 (GE200)") == ("low_alloy_steel", False)
    assert resolve_alloy_family("GS-45 (GP240GH)") == ("low_alloy_steel", False)
    assert resolve_alloy_family("G20Mn5") == ("low_alloy_steel", False)
    assert resolve_alloy_family("G17CrMo5-5") == ("low_alloy_steel", False)


def test_resolve_alloy_family_default_alloy_when_none():
    # None -> DEFAULT_ALLOY (EN-GJS-400-15) -> ductile_iron, not assumed.
    assert resolve_alloy_family(None) == ("ductile_iron", False)


def test_resolve_alloy_family_unknown_prefix_falls_back_loud():
    family, assumed = resolve_alloy_family("some_unknown_grade")
    assert family == "steel" and assumed is True


def test_resolve_alloy_family_props_json_always_assumed_steel():
    # A custom --props JSON bypasses the named-material table entirely -- no
    # grade name to map from -- so the documented fallback is "steel",
    # flagged as assumed regardless of what `alloy` happens to be.
    family, assumed = resolve_alloy_family("EN-GJL-250", props_json="/some/path.json")
    assert family == "steel" and assumed is True


def test_resolve_alloy_family_every_mapped_family_is_a_known_shrinkage_pct_key():
    for grade in ("EN-GJL-250", "EN-GJS-400-15", "GS-38 (GE200)", "unknown"):
        family, _ = resolve_alloy_family(grade)
        assert family in SHRINKAGE_PCT


def test_resolve_alloy_family_covers_every_catalog_grade_exactly():
    """Review round 1, finding 6: a data-driven guard over the REAL property
    DB (`data/properties/materials.csv` via `list_alloys()`) -- every grade
    actually on file today must be an EXACT prefix match (`assumed is
    False`), not a fallback. This is the loud-failure trip-wire for a
    FUTURE grade added to the catalog under a name none of
    `_ALLOY_FAMILY_PREFIXES` recognizes: that grade would otherwise silently
    default to "steel" (a plausible-looking but wrong family for, say, a new
    aluminum or a differently-prefixed steel grade) with no test ever
    catching it."""
    for grade in list_alloys():
        family, assumed = resolve_alloy_family(grade)
        assert assumed is False, (
            f"catalog grade {grade!r} fell back to the 'assumed' family map "
            f"(resolved family={family!r}) -- add a matching prefix to "
            f"src.inference_props._ALLOY_FAMILY_PREFIXES")
        assert family in SHRINKAGE_PCT
