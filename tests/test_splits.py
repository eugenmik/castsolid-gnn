import numpy as np
from src.splits import build_splits, dims_of, HELDOUT_FAMILIES


def _meta(stems):
    import re
    prefix = np.array([s.split("_")[0] for s in stems])
    n_ad = np.array([len(re.findall(r"_[+-][XYZ]", s)) for s in stems])
    return {"stem": np.array(stems), "prefix": prefix,
            "n_adiab": n_ad, "is_fully_cooled": n_ad == 0}


def test_dims_of():
    assert dims_of("BC_06_08_14_+X_+Z") == ("06", "08", "14")
    assert dims_of("CYL_06") == ("06",)


def test_splits_are_disjoint_and_family_held_out():
    stems = [f"BC_{d:02d}_08_10" for d in range(6, 40, 2)]                # interp pool
    stems += [f"BC_{d:02d}_08_10_+X" for d in range(6, 40, 2)]            # adiab variants
    stems += ["BO_10_10", "T45_10_20", "WH_10", "RR_10_10"]              # held-out families
    m = _meta(stems)
    sp = build_splits(m, seed=0, test_frac=0.2)

    # family test = exactly the held-out-family stems
    fam_test = set(sp["family"]["test"])
    assert all(s.split("_")[0] in HELDOUT_FAMILIES for s in fam_test)
    assert not any(s.split("_")[0] in HELDOUT_FAMILIES for s in sp["family"]["train"])

    # interpolation train/test disjoint, no held-out families leak in
    itr, ite = set(sp["interpolation"]["train"]), set(sp["interpolation"]["test"])
    assert itr.isdisjoint(ite)
    assert not any(s.split("_")[0] in HELDOUT_FAMILIES for s in itr | ite)

    # fully_cooled test contains only 0-adiabatic stems
    assert all("_+" not in s and "_-" not in s for s in sp["fully_cooled"]["test"])
