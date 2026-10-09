"""Alloy and mold properties.

Turns a named alloy and mold (rows of `data/properties/materials.csv` and
`molds.csv`), or a JSON file with raw values, into the 8-key dict the model
reads:

    {k, rho, cp, L_kJ_kg, TL, TS, Tpour, mold_effusivity}

Units: W/(m K), kg/m^3, J/(kg K), kJ/kg, deg C, W s^0.5/(m^2 K). The pouring
temperature of a named alloy is the midpoint of its tabulated range.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ALLOY = "EN-GJS-400-15"
DEFAULT_MOLD = "green_sand"

_REQUIRED_KEYS = ("k", "rho", "cp", "L_kJ_kg", "TL", "TS", "Tpour", "mold_effusivity")

_MATERIALS_PATH = "data/properties/materials.csv"
_MOLDS_PATH = "data/properties/molds.csv"

# Family used for shrinkage severity when the alloy family is unknown (a raw
# property file has no grade). Steel has the largest shrinkage, so this errs
# on the side of flagging.
_UNKNOWN_ALLOY_FAMILY_FALLBACK = "steel"


@dataclass(frozen=True)
class Material:
    material_id: str
    alloy_family: str
    k: float
    rho: float
    cp: float
    L: float
    TL: float
    TS: float
    Tpour: float


def load_materials(path: str = _MATERIALS_PATH) -> list[Material]:
    """Rows of the materials table; rejects rows that are not TS < TL < Tpour."""
    materials: list[Material] = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            material_id = row["material_id"]
            TL = float(row["TL_C"])
            TS = float(row["TS_C"])
            tpour = 0.5 * (float(row["Tpour_lo_C"]) + float(row["Tpour_hi_C"]))
            if not TS < TL < tpour:
                raise ValueError(
                    f"material {material_id!r}: expected TS < TL < Tpour, "
                    f"got {TS}, {TL}, {tpour}")
            materials.append(Material(
                material_id=material_id,
                alloy_family=row["family"],
                k=float(row["k_W_mK"]),
                rho=float(row["rho_kg_m3"]),
                cp=float(row["cp_J_kgK"]),
                L=float(row["L_kJ_kg"]),
                TL=TL,
                TS=TS,
                Tpour=tpour,
            ))
    return materials


def load_molds(path: str = _MOLDS_PATH) -> dict[str, float]:
    """{mold_system: effusivity} from the molds table."""
    with Path(path).open(newline="") as f:
        return {row["mold_system"]: float(row["effusivity_W_s05_m2K"])
                for row in csv.DictReader(f)}


_materials_cache: dict[str, Material] | None = None
_molds_cache: dict[str, float] | None = None


def _materials() -> dict[str, Material]:
    global _materials_cache
    if _materials_cache is None:
        _materials_cache = {m.material_id: m for m in load_materials()}
    return _materials_cache


def _molds() -> dict[str, float]:
    global _molds_cache
    if _molds_cache is None:
        _molds_cache = load_molds()
    return _molds_cache


def list_alloys() -> list[str]:
    return list(_materials().keys())


def list_molds() -> list[str]:
    return list(_molds().keys())


def _validate_tl_ts(props: dict, source: str) -> None:
    """An inverted or equal liquidus/solidus pair would make the Niyama proxy
    zero everywhere and flag every node, so it is rejected up front."""
    TL, TS = props["TL"], props["TS"]
    if TL <= TS:
        raise ValueError(
            f"resolve_props ({source}): TL ({TL!r}) must be strictly greater than "
            f"TS ({TS!r})")


def resolve_props(alloy: str | None = None, mold: str = DEFAULT_MOLD,
                  props_json: str | None = None) -> dict:
    """The 8-key property dict.

    `props_json` replaces `alloy` and `mold`; the file must hold all 8 keys.
    Otherwise `alloy` (default `DEFAULT_ALLOY`) and `mold` are looked up in
    the tables, and unknown names raise ValueError listing the options."""
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

    props = {
        "k": float(material.k),
        "rho": float(material.rho),
        "cp": float(material.cp),
        "L_kJ_kg": float(material.L),
        "TL": float(material.TL),
        "TS": float(material.TS),
        "Tpour": float(material.Tpour),
        "mold_effusivity": float(molds[mold]),
    }
    _validate_tl_ts(props, f"alloy={alloy_id!r}")
    return props


def resolve_alloy_family(alloy: str | None, props_json: str | None = None) -> tuple[str, bool]:
    """(family, assumed) for shrinkage severity. The family comes from the
    materials table; `assumed` is True when it had to fall back to steel
    (a raw property file, or an alloy that is not in the table)."""
    if props_json is not None:
        return _UNKNOWN_ALLOY_FAMILY_FALLBACK, True
    material = _materials().get(alloy if alloy is not None else DEFAULT_ALLOY)
    if material is None:
        return _UNKNOWN_ALLOY_FAMILY_FALLBACK, True
    return material.alloy_family, False
