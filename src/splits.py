"""Reproducible train/test splits over the feature cache.

Three splits, each with an explicit train/test stem list:
  interpolation  — hold out (prefix,dims) groups (interpolate within families)
  family         — hold out whole geometry prefixes (generalize to new shapes)
  fully_cooled   — evaluate only on 0-adiabatic castings (real-inference regime)
"""
import json
import re

import numpy as np

HELDOUT_FAMILIES = ["BO", "T45", "WH", "RR"]
_NUM_RE = re.compile(r"^\d+$")


def dims_of(stem: str) -> tuple:
    return tuple(t for t in stem.split("_")[1:] if _NUM_RE.match(t))


def build_splits(meta, seed: int = 0, test_frac: float = 0.1) -> dict:
    stems = list(map(str, meta["stem"]))
    prefix = {s: p for s, p in zip(stems, map(str, meta["prefix"]))}
    fully = {s: bool(v) for s, v in zip(stems, meta["is_fully_cooled"])}

    non_family = [s for s in stems if prefix[s] not in HELDOUT_FAMILIES]
    family_test = [s for s in stems if prefix[s] in HELDOUT_FAMILIES]

    # interpolation: hold out groups keyed by (prefix, dims)
    groups = {}
    for s in non_family:
        groups.setdefault((prefix[s], dims_of(s)), []).append(s)
    keys = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_frac)))
    test_keys = set(keys[:n_test])

    interp_train, interp_test = [], []
    for k, members in groups.items():
        (interp_test if k in test_keys else interp_train).extend(members)

    # fully_cooled: restrict to 0-adiabatic; test = fully-cooled in the held-out groups
    fc_train = [s for s in interp_train if fully[s]]
    fc_test = [s for s in interp_test if fully[s]]

    return {
        "interpolation": {"train": sorted(interp_train), "test": sorted(interp_test)},
        "family": {"train": sorted(non_family), "test": sorted(family_test)},
        "fully_cooled": {"train": sorted(fc_train), "test": sorted(fc_test)},
    }


def save_splits(splits: dict, path: str):
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
