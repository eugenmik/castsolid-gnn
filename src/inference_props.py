"""Conditioning resolver for v4 multi-alloy inference.

Turns a named alloy grade + mold system (or a raw-props JSON override) into
the 8-key property dict consumed by `src.features.alloy_g_vector` and the
FT2b-conditioned model/GBM:

    {k, rho, cp, L_kJ_kg, TL, TS, Tpour, mold_effusivity}

Named alloys come from `tools.doe.load_properties.load_materials()` (reads
`data/properties/materials.csv`); mold effusivities come from
`data/properties/molds.csv`. A `props_json` path bypasses both tables
entirely and must already contain all 8 keys.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.doe.load_properties import load_materials

DEFAULT_ALLOY = "EN-GJS-400-15"

_REQUIRED_KEYS = ("k", "rho", "cp", "L_kJ_kg", "TL", "TS", "Tpour", "mold_effusivity")

_MATERIALS_PATH = "data/properties/materials.csv"
_MOLDS_PATH = "data/properties/molds.csv"

_materials_cache: dict[str, object] | None = None
_molds_cache: dict[str, float] | None = None


def _materials() -> dict[str, object]:
    """Module-level cached {material_id: Material}."""
    global _materials_cache
    if _materials_cache is None:
        _materials_cache = {m.material_id: m for m in load_materials(_MATERIALS_PATH)}
    return _materials_cache


def _molds() -> dict[str, float]:
    """Module-level cached {mold_system: effusivity_W_s05_m2K}."""
    global _molds_cache
    if _molds_cache is None:
        molds: dict[str, float] = {}
        p = Path(_MOLDS_PATH)
        if p.exists():
            with p.open(newline="") as f:
                for row in csv.DictReader(f):
                    system = row.get("mold_system")
                    raw = row.get("effusivity_W_s05_m2K")
                    if system and raw not in (None, ""):
                        molds[system] = float(raw)
        _molds_cache = molds
    return _molds_cache


def list_alloys() -> list[str]:
    """Named material_ids available (from `data/properties/materials.csv`)."""
    return list(_materials().keys())


def list_molds() -> list[str]:
    """Named mold systems available (from `data/properties/molds.csv`)."""
    return list(_molds().keys())


def _validate_tl_ts(props: dict, source: str) -> None:
    """Review finding I2 (final whole-branch review, 2026-08-01): `TL <= TS`
    is physically inverted (liquidus at or below solidus) and was previously
    UNvalidated for a `--props` custom-JSON override -- `resolve_props` only
    checked key *presence*, never ordering. `src.shrinkage.niyama_proxy`
    clamps its `tdot_proxy` term to `>= 0.0`, so an inverted pair silently
    yields `Ny == 0` for every node, and `porosity_risk(mode="percentile")`
    then flags 100% of nodes as at-risk -- no warning, no exception, just a
    self-contradictory "100% flagged (bottom decile by construction)"
    summary line. Named grades are protected by `tools.doe.load_properties.
    load_materials`'s own `TS < TL < Tpour` check at CSV-load time, but that
    guard never runs for a `--props` override (it bypasses the materials
    table entirely) -- and SP2 makes shrinkage screening default-ON, so this
    path is live in production, not dead code. Raises loudly, naming both
    values and which of `props_json`/`alloy` triggered it.
    """
    TL, TS = props["TL"], props["TS"]
    if TL <= TS:
        raise ValueError(
            f"resolve_props ({source}): TL ({TL!r}) must be strictly greater than "
            f"TS ({TS!r}) -- an inverted or degenerate liquidus/solidus pair makes "
            f"the Niyama proxy (src.shrinkage.niyama_proxy) collapse to 0 "
            f"everywhere, silently flagging 100% of nodes as at-risk")


def resolve_props(alloy: str | None = None, mold: str = "green_sand",
                   props_json: str | None = None) -> dict:
    """Resolve the 8-key conditioning-property dict.

    `props_json`, if given, overrides `alloy`/`mold` entirely: the file must
    be a JSON object containing all 8 required keys (missing keys raise
    ValueError naming them).

    Otherwise resolves `alloy` (default `DEFAULT_ALLOY` if None) against the
    named-material table and `mold` against the mold-effusivity table;
    unknown names raise ValueError listing the available options.

    Both branches validate `TL > TS` (review finding I2) before returning --
    a `--props` override with an inverted or degenerate pair raises a loud
    `ValueError` naming both values, rather than silently degrading the
    shrinkage screening's Niyama proxy to "0 everywhere / 100% flagged".
    """
    if props_json is not None:
        with open(props_json) as f:
            data = json.load(f)
        missing = [k for k in _REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(
                f"props_json '{props_json}' is missing required key(s): {missing} "
                f"(required: {list(_REQUIRED_KEYS)})")
        props = {k: float(data[k]) for k in _REQUIRED_KEYS}
        _validate_tl_ts(props, f"props_json={props_json!r}")
        return props

    alloy_id = alloy if alloy is not None else DEFAULT_ALLOY
    materials = _materials()
    if alloy_id not in materials:
        raise ValueError(
            f"unknown alloy '{alloy_id}'; available alloys: {sorted(materials.keys())}")
    material = materials[alloy_id]

    molds = _molds()
    if mold not in molds:
        raise ValueError(
            f"unknown mold system '{mold}'; available mold systems: {sorted(molds.keys())}")
    mold_effusivity = molds[mold]

    props = {
        "k": float(material.k),
        "rho": float(material.rho),
        "cp": float(material.cp),
        "L_kJ_kg": float(material.L),
        "TL": float(material.TL),
        "TS": float(material.TS),
        "Tpour": float(material.Tpour),
        "mold_effusivity": float(mold_effusivity),
    }
    _validate_tl_ts(props, f"alloy={alloy_id!r}")
    return props


# --------------------------------------------------------------------------- #
# resolve_alloy_family (SP2 shrinkage Task 4, controller directive 2):
# `resolve_props`'s returned 8-key dict has no alloy-family field -- it is
# BUILT from `data/properties/materials.csv`'s own `family` column
# (`tools.doe.load_properties.load_materials` stores it on `Material.
# alloy_family`), but that value never makes it into `resolve_props`'s
# return dict, and `src.shrinkage.screen_casting` needs a family key from
# `src.shrinkage.SHRINKAGE_PCT` to look up the alloy's volumetric shrinkage
# percentage. Rather than reaching back into the materials table (which
# doesn't exist at all for a `--props` custom-JSON override), this maps the
# alloy GRADE NAME directly by prefix -- every named grade in
# `materials.csv` today falls into one of `SHRINKAGE_PCT`'s 4 keys via this
# same prefix convention (EN-GJL-* -> grey_iron, EN-GJS-* -> ductile_iron,
# GS-*/G17*/G20* -> low_alloy_steel), so this is not a guess for any grade
# actually in the property DB -- only for a name this helper has never seen
# (a future grade, a typo, or a `--props` override with no named grade at
# all) does it fall back, loudly (the caller records `alloy_family_assumed`
# rather than silently proceeding as if the mapping were exact).
# --------------------------------------------------------------------------- #
_ALLOY_FAMILY_PREFIXES = (
    ("EN-GJS", "ductile_iron"),
    ("EN-GJL", "grey_iron"),
    ("GS-", "low_alloy_steel"),
    ("G17", "low_alloy_steel"),
    ("G20", "low_alloy_steel"),
)

# The conservative (highest-shrinkage-percentage) fallback family for a grade
# name this helper doesn't recognize, or for a `--props` custom-JSON override
# (no grade name to map from at all) -- see the module comment above.
_UNKNOWN_ALLOY_FAMILY_FALLBACK = "steel"


def resolve_alloy_family(alloy: str | None, props_json: str | None = None) -> tuple[str, bool]:
    """Map an alloy grade name to one of `src.shrinkage.SHRINKAGE_PCT`'s
    family keys, for the shrinkage-screening integration.

    Returns `(family, assumed)`: `assumed` is `True` when the mapping is a
    documented fallback rather than an exact name-prefix match -- either
    because `props_json` was given (a raw-properties override has no named
    grade to map from) or because `alloy` (or `DEFAULT_ALLOY`, if `alloy`
    is `None`) doesn't start with any of the known prefixes. The caller is
    expected to surface `assumed` (e.g. as `alloy_family_assumed` in the
    shrinkage result) rather than silently treat a defaulted family as if it
    were an exact match.
    """
    if props_json is not None:
        return _UNKNOWN_ALLOY_FAMILY_FALLBACK, True

    label = alloy if alloy is not None else DEFAULT_ALLOY
    for prefix, family in _ALLOY_FAMILY_PREFIXES:
        if label.startswith(prefix):
            return family, False
    return _UNKNOWN_ALLOY_FAMILY_FALLBACK, True
