"""Load the public material, mold, chill and sleeve-calibration tables."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


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


_MATERIAL_REQUIRED_FIELDS = [
    "k_W_mK", "rho_kg_m3", "cp_J_kgK", "L_kJ_kg",
    "TL_C", "TS_C", "Tpour_lo_C", "Tpour_hi_C",
]


def _is_missing(value) -> bool:
    value = (value or "").strip()
    return value == "" or value.upper() == "OWNER"


def load_materials(path: str = "data/properties/materials.csv") -> list[Material]:
    """Return complete material rows and reject physically inverted ranges."""
    table = Path(path)
    if not table.exists():
        print(f"[load_properties] WARNING: materials file not found: {path}")
        return []

    materials: list[Material] = []
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            material_id = row.get("material_id") or "<unknown>"
            missing = [key for key in _MATERIAL_REQUIRED_FIELDS
                       if _is_missing(row.get(key))]
            if missing:
                print(f"[load_properties] WARNING: skipping {material_id!r}; "
                      f"missing fields: {missing}")
                continue
            TL = float(row["TL_C"])
            TS = float(row["TS_C"])
            tpour = 0.5 * (float(row["Tpour_lo_C"]) + float(row["Tpour_hi_C"]))
            if not TS < TL < tpour:
                raise ValueError(
                    f"material {material_id!r}: expected TS < TL < Tpour, "
                    f"got {TS}, {TL}, {tpour}")
            materials.append(Material(
                material_id=material_id,
                alloy_family=row.get("family", "") or "",
                k=float(row["k_W_mK"]),
                rho=float(row["rho_kg_m3"]),
                cp=float(row["cp_J_kgK"]),
                L=float(row["L_kJ_kg"]),
                TL=TL,
                TS=TS,
                Tpour=tpour,
            ))
    return materials


def _load_effusivity_dict(path: str, key_column: str) -> dict[str, float]:
    table = Path(path)
    if not table.exists():
        print(f"[load_properties] WARNING: file not found: {path}")
        return {}
    values: dict[str, float] = {}
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get(key_column)
            raw = row.get("effusivity_W_s05_m2K")
            if key and not _is_missing(raw):
                values[key] = float(raw)
    return values


def load_mold_effusivities(path: str = "data/properties/molds.csv") -> dict[str, float]:
    return _load_effusivity_dict(path, "mold_system")


def load_chill_effusivities(path: str = "data/properties/chills.csv") -> dict[str, float]:
    return _load_effusivity_dict(path, "material")


def resolve_exo_q0_by_mef(
    mef: float,
    csv_path: str = "data/properties/exo_calibration.csv",
    tol: float = 0.05,
) -> float:
    """Resolve an exothermic heat-flux input from the nearest calibrated MEF."""
    table = Path(csv_path)
    if not table.exists():
        raise ValueError(f"exothermic calibration table does not exist: {csv_path}")
    with table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"exothermic calibration table has no rows: {csv_path}")
    best = min(rows, key=lambda row: abs(float(row["MEF"]) - mef))
    difference = abs(float(best["MEF"]) - mef)
    if difference > tol:
        raise ValueError(
            f"no calibrated MEF within {tol} of {mef}; "
            f"nearest={best['MEF']}, difference={difference:.4f}")
    return float(best["q0_W_m2"])
