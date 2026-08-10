"""Multi-body STEP assembly tooling -- Part B Task 1.

Two pieces:
  - `list_solids(step_path)` -- the assembly inspector: fresh gmsh/OCC
    session, imports a (possibly multi-solid) STEP file with entity names
    preserved (`Geometry.OCCImportLabels=1`, set BEFORE `importShapes` --
    setting it after import is a no-op), and reports one row per 3-D solid
    (id/name/volume/bbox/center-of-mass), sorted by volume descending. This
    is the data source for the later "role menu" (casting / chill /
    insulating sleeve / exothermic sleeve) that a human or heuristic
    assigns per solid.
  - `compose_assembly(base_step, attachments, out_step)` -- a test/fixture
    helper (used by this task's own tests and by all later Part B fixtures,
    e.g. proximity classification, BC-patch tagging) that builds a
    synthetic multi-body STEP: import a base part, then place one OCC
    primitive (box/cylinder) per attachment spec at one of the base's
    bounding-box faces. NO boolean operations are performed -- the
    attachment is written as an independent solid, deliberately overlapping
    the base by `overlap_mm` (default 2.0mm) along the face normal. This is
    intentional, not a bug: downstream, these bodies are never meshed
    together (only the casting solid is meshed); the attachments exist
    purely so their bounding boxes/faces can drive proximity/contact
    classification against the casting, and a few mm of geometric overlap
    is harmless for that purpose while guaranteeing the two solids actually
    register as "touching" rather than floating a hair apart.

All lengths in mm, all gmsh sessions are fresh (`gmsh.initialize()` /
`gmsh.finalize()` in a try/finally per call) -- never share state across
calls, matching the pattern in `tools/doe/prep_parts.py` /
`src/step_mesh.py`.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_VALID_SHAPES = {"box", "cylinder"}


# --------------------------------------------------------------------------
# Assembly inspection
# --------------------------------------------------------------------------
def list_solids(step_path: str) -> list[dict]:
    """Import `step_path` in a fresh gmsh/OCC session and return one dict
    per 3-D solid entity, sorted by `volume_mm3` descending (largest first
    -- in practice the casting itself, since attached chills/sleeves are
    normally much smaller):

        {"solid_id": <gmsh tag, int>, "name": <STEP entity name or "">,
         "volume_mm3": <float>, "bbox": [xmin, ymin, zmin, xmax, ymax, zmax],
         "com": [cx, cy, cz]}

    `Geometry.OCCImportLabels` MUST be set to 1 BEFORE `importShapes` --
    that's the only way gmsh preserves STEP entity names (product names
    from the CAD export) for `gmsh.model.getEntityName` to read; setting it
    afterwards has no effect on an already-imported model. Solids with no
    STEP name (or on any importer that doesn't preserve one) get `name=""`
    rather than raising -- callers decide how to label an unnamed solid in
    a role menu.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
        gmsh.model.add("assembly_inspect")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()

        rows: list[dict] = []
        for (_dim, tag) in gmsh.model.getEntities(3):
            name = gmsh.model.getEntityName(3, tag) or ""
            volume_mm3 = float(gmsh.model.occ.getMass(3, tag))
            bbox = list(gmsh.model.getBoundingBox(3, tag))
            com = list(gmsh.model.occ.getCenterOfMass(3, tag))
            rows.append({
                "solid_id": tag,
                "name": name,
                "volume_mm3": volume_mm3,
                "bbox": bbox,
                "com": com,
            })
        rows.sort(key=lambda r: r["volume_mm3"], reverse=True)
        return rows
    finally:
        gmsh.finalize()


# --------------------------------------------------------------------------
# Assembly composer (test/fixture helper)
# --------------------------------------------------------------------------
def _parse_face(at_face: str) -> tuple[int, float]:
    """`"+x"/"-x"/"+y"/"-y"/"+z"/"-z"` -> (axis_index, sign)."""
    if len(at_face) != 2 or at_face[0] not in "+-" or at_face[1] not in _AXIS_INDEX:
        raise ValueError(
            f"invalid at_face {at_face!r}; expected one of +x -x +y -y +z -z")
    return _AXIS_INDEX[at_face[1]], (1.0 if at_face[0] == "+" else -1.0)


def _place_attachment(att: dict, bbox_min: Sequence[float], bbox_max: Sequence[float],
                       center: Sequence[float]) -> None:
    """Add ONE primitive (box or cylinder) to the currently-open OCC model,
    positioned at the base's bbox face named by `att["at_face"]`, sunk
    `att.get("overlap_mm", 2.0)` INTO the base along the face normal (see
    module docstring for why a positive overlap is intentional). Does not
    synchronize/write -- the caller does that once after all attachments
    are placed."""
    import gmsh

    shape = att.get("shape")
    if shape not in _VALID_SHAPES:
        raise ValueError(f"invalid shape {shape!r}; expected one of {sorted(_VALID_SHAPES)}")
    if "size" not in att:
        raise ValueError(f"attachment missing required 'size': {att!r}")
    if "at_face" not in att:
        raise ValueError(f"attachment missing required 'at_face': {att!r}")

    axis, sign = _parse_face(att["at_face"])
    overlap_mm = float(att.get("overlap_mm", 2.0))
    du, dv = att.get("offset", (0.0, 0.0))
    perp_axes = [i for i in range(3) if i != axis]
    face_coord = bbox_max[axis] if sign > 0 else bbox_min[axis]

    if shape == "box":
        dx, dy, dz = att["size"]
        size_by_axis = [dx, dy, dz]
        perp_size = size_by_axis[axis]

        if sign > 0:
            lo_axis = face_coord - overlap_mm
        else:
            lo_axis = face_coord + overlap_mm - perp_size

        coords = [0.0, 0.0, 0.0]
        extents = [0.0, 0.0, 0.0]
        coords[axis] = lo_axis
        extents[axis] = perp_size
        for i, u in zip(perp_axes, (du, dv)):
            extents[i] = size_by_axis[i]
            coords[i] = center[i] - size_by_axis[i] / 2.0 + u

        gmsh.model.occ.addBox(coords[0], coords[1], coords[2],
                               extents[0], extents[1], extents[2])
    else:  # cylinder
        radius, height = att["size"]
        base_point = [0.0, 0.0, 0.0]
        axis_vec = [0.0, 0.0, 0.0]
        base_point[axis] = face_coord - sign * overlap_mm
        axis_vec[axis] = sign * height
        for i, u in zip(perp_axes, (du, dv)):
            base_point[i] = center[i] + u

        gmsh.model.occ.addCylinder(base_point[0], base_point[1], base_point[2],
                                    axis_vec[0], axis_vec[1], axis_vec[2], radius)


def compose_assembly(base_step: str, attachments: list[dict], out_step: str) -> int:
    """Build a synthetic multi-body STEP: import `base_step`, then add one
    OCC primitive per `attachments` entry (see module docstring for the
    overlap/no-boolean-ops contract), and write the result (all solids,
    unmerged) to `out_step`. Returns the number of 3-D solids written
    (1 + len(attachments) in the normal case where `base_step` itself is
    single-solid).

    Each attachment dict:
        {"shape": "box" | "cylinder",
         "size": (dx, dy, dz) for a box, (radius, height) for a cylinder,
         "at_face": "+x" | "-x" | "+y" | "-y" | "+z" | "-z",
         "overlap_mm": float, default 2.0,
         "offset": (du, dv) in the face plane, default (0.0, 0.0) (centered)}

    For a box, `size[axis]` (the dimension along the face normal) is the
    part that gets sunk `overlap_mm` into the base; the other two `size`
    entries lie in the face plane. For a cylinder, the axis runs along the
    face normal (`height`) and `radius` is in the face plane. `offset`
    shifts the attachment's center within the face plane, in the same
    (u, v) order as the two non-normal world axes (e.g. for "+z"/"-z",
    (u, v) = (x, y); for "+x"/"-x", (u, v) = (y, z); for "+y"/"-y",
    (u, v) = (x, z)).

    NOTE for `tag_contact_patches` callers: a positive `overlap_mm` is a
    fixed geometric gap that `tol_mm` must exceed just to detect the patch
    at all, and any `tol_mm` beyond that dilates the classified patch past
    the true footprint edge (see `tag_contact_patches`'s CAVEAT). A
    synthetic fixture whose `overlap_mm` is large relative to its footprint
    size will over-classify noticeably; real assemblies (near-zero contact
    or a sub-mm clearance, not several mm sunk in) only need a small
    `tol_mm`, where this dilation is negligible.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("assembly_compose")
        gmsh.model.occ.importShapes(base_step)
        gmsh.model.occ.synchronize()

        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        bbox_min = [xmin, ymin, zmin]
        bbox_max = [xmax, ymax, zmax]
        center = [(bbox_min[i] + bbox_max[i]) / 2.0 for i in range(3)]

        for att in attachments:
            _place_attachment(att, bbox_min, bbox_max, center)

        gmsh.model.occ.synchronize()
        gmsh.write(out_step)
        return len(gmsh.model.getEntities(3))
    finally:
        gmsh.finalize()


# --------------------------------------------------------------------------
# Body spec -- Part B Task 2: user-assigned solid roles -> the exact BC
# parameters the v4 model was trained on (BcPlan), via the same registries
# `tools/doe/doe.py` used to build the training manifest.
# --------------------------------------------------------------------------
_VALID_ROLES = {"casting", "chill", "insulating_sleeve", "exo_sleeve", "ignore"}
_SLEEVE_ROLES = {"insulating_sleeve", "exo_sleeve"}
_DEFAULT_CHILL_MATERIAL = "cast_iron"


@dataclass
class BcPlan:
    """The resolved boundary-condition plan for one casting body, in the
    exact shape `_node_features(version=4)` and `gbm_v2_total_seconds`
    expect (see `feature_spec`/`gbm_bc_spec`). Produced by `validate_bodies`
    -- never construct this by hand from unvalidated user input."""
    bc_config: str            # "none" | "chills" | "insulating_sleeve" | "exo_sleeve"
    casting_solid_id: int
    role_solid_ids: list[int] = field(default_factory=list)  # bodies forming the patch
    n_chills: int = 0
    chill_effusivity: float = 0.0
    sleeve_MEF: float = 0.0
    exo_q0: float = 0.0       # W/m^2 (training units)
    # Part-B roles task (2026-08-03): per-BODY role, `{solid_id: role}` for
    # every id in `role_solid_ids` (e.g. all-"chill" for the chills branch,
    # the single sleeve id -> its bc_config for the sleeve branch, empty for
    # "none"). Lets a caller (e.g. `predict_casting`'s feed-node selection)
    # decide per BODY ROLE rather than per global `bc_config` -- necessary
    # groundwork for a future mixed-role regime even though `validate_bodies`
    # itself still only ever produces a single-role mapping today.
    role_by_solid: dict[int, str] = field(default_factory=dict)
    # SP4 ADDENDUM (2026-08-08): several sleeve bodies are the norm on real
    # castings. When they all resolve to one MEF it is used exactly and these
    # stay at their defaults. When they differ, `sleeve_MEF` is the
    # contact-area-weighted mean, `mef_weighted` is True, and `mef_spread`
    # carries max(MEF) - min(MEF) so the report can state the spread the
    # single scalar channel is standing in for rather than hiding it.
    mef_weighted: bool = False
    mef_spread: float = 0.0

    def feature_spec(self, mold_effusivity: float) -> dict:
        """The `spec` dict `src.features._node_features(version=4, spec=...)`
        reads -- keys match `Global Constraints` in
        `docs/superpowers/plans/2026-07-29-part-b-multibody.md` exactly."""
        return {
            "mold_effusivity": mold_effusivity,
            "bc_config": self.bc_config,
            "chill_effusivity": self.chill_effusivity,
            "sleeve_MEF": self.sleeve_MEF,
            "exo_q0": self.exo_q0,
        }

    def gbm_bc_spec(self) -> dict:
        """The `bc_spec` dict `src.gbm_ft2b.gbm_v2_total_seconds` reads."""
        return {
            "bc_config": self.bc_config,
            "n_chills": self.n_chills,
            "sleeve_MEF": self.sleeve_MEF,
            "exo_q0_MW": self.exo_q0 / 1e6,
        }


def _load_sleeve_registry(csv_path: str) -> dict[str, dict]:
    """{sleeve name: {"MEF": float, "type": str}} from
    `tools/sleeves/sleeve_registry.csv` -- `type` is "exo" or "insulating"
    and is used to cross-check a registry-resolved sleeve's physics against
    the declared role (see `validate_bodies`)."""
    out: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["name"]] = {"MEF": float(row["MEF"]), "type": row["type"]}
    return out


# Declared role -> the sleeve_registry.csv `type` value it requires when the
# sleeve is resolved BY REGISTRY NAME. Without this check, a registry
# collision (two products landing on the same MEF, e.g. FOSECO_V88E
# type=insulating MEF=1.45 vs the exo_calibration.csv row at MEF=1.45) lets
# a plain insulating sleeve silently resolve a confident nonzero exo_q0 --
# a silent-wrong-physics path, not merely a cosmetic mismatch.
_ROLE_TO_SLEEVE_TYPE = {"exo_sleeve": "exo", "insulating_sleeve": "insulating"}

# Two sleeves whose MEFs differ by less than this are treated as identical --
# the registry stores MEF to 2 decimals, so anything below it is float noise,
# not a real difference in sleeve product.
_MEF_AGREEMENT_TOL = 1e-6


def _resolve_chill_material(params: dict | None) -> str:
    params = params or {}
    return (params.get("material") or _DEFAULT_CHILL_MATERIAL).strip()


def validate_bodies(bodies: list[dict],
                     chills_csv: str = "data/properties/chills.csv",
                     sleeve_registry_csv: str = "tools/sleeves/sleeve_registry.csv",
                     exo_calibration_csv: str = "data/properties/exo_calibration.csv",
                     tol: float = 0.05,
                     mef_weights: dict[int, float] | None = None) -> BcPlan:
    """Validate a user-assigned role list and resolve it into a `BcPlan`.

    `bodies` rows: `{"solid_id": int, "role": <see _VALID_ROLES>,
    "params": dict (optional, role-dependent)}`. `ignore` rows are dropped
    before anything else. Enforces the training-regime restriction (owner,
    2026-07-29): EXACTLY one casting; then either >=1 `chill` bodies
    (bc_config="chills") OR one or more sleeve bodies that share a single role
    (`insulating_sleeve`/`exo_sleeve`, bc_config = that role; mixing the two
    roles raises) OR neither (bc_config="none").
    Chills+sleeve mixed, two sleeves, or a casting count != 1 all raise a
    loud `ValueError` explaining the restriction -- the trained model has no
    BC channel for anything else, so silently coercing one regime into
    another would produce a confidently wrong prediction.

    Chill params: `{"material": str}` (default "cast_iron"); an unknown
    material raises `ValueError` listing `load_chill_effusivities()`'s known
    materials. Multiple chills MUST share one material -- the trained model
    has a single scalar `chill_effusivity` channel, not one per chill, so
    differing materials raise `ValueError` (single chill_effusivity per
    training convention).

    Sleeve params: exactly one of `{"sleeve": <sleeve_registry.csv name>}`
    (MEF looked up in the registry) or `{"MEF": float}` (manual override) --
    both or neither present raises `ValueError`. An unknown registry name
    raises `ValueError` listing the known names. When resolved BY NAME, the
    registry row's `type` column ("exo" or "insulating") MUST match the
    declared role (`exo_sleeve` requires type=="exo", `insulating_sleeve`
    requires type=="insulating") -- a mismatch raises `ValueError` naming
    both, since two products can share the same MEF (a registry collision)
    while being physically different (an insulating sleeve has no
    exothermic heat release), and resolving one under the other's role
    would silently produce a confidently wrong prediction. Manual
    `{"MEF": float}` overrides BYPASS this check entirely (no registry row
    to check against) -- this is expert mode: the caller is asserting the
    physics directly and is trusted to pick the right role. For `bc_config
    == "exo_sleeve"`, the resolved MEF is turned into a calibrated `exo_q0`
    via `tools.doe.load_properties.resolve_exo_q0_by_mef` (same nearest-MEF
    lookup `src.build_dataset._resolve_exo_q0_v4` uses for training data --
    raises `ValueError` if no calibration point is within `tol`).

    Several sleeves (SP4 ADDENDUM, 2026-08-08): real castings carry 3-8
    sleeves, so N bodies of one role are accepted, mirroring the chills
    branch. When every sleeve resolves to the same MEF it is used exactly
    (`mef_weighted=False`, `mef_spread=0.0`). When they differ -- e.g. an
    assembly with two sleeve sizes -- the model's single scalar channel
    cannot hold both, so `mef_weights` (contact-patch area per solid_id,
    exactly `tag_contact_patches`'s `per_body_patch_area_mm2`) MUST be
    supplied and the effective MEF is their area-weighted mean, with
    `mef_weighted=True` and `mef_spread` recording what was approximated.
    Omitting the weights on a mixed set raises rather than silently picking
    one sleeve's MEF or an unweighted mean.
    """
    from tools.doe.load_properties import (load_chill_effusivities,
                                            resolve_exo_q0_by_mef)

    active = [b for b in bodies if (b.get("role") or "") != "ignore"]

    for b in active:
        role = b.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(
                f"body solid_id={b.get('solid_id')!r}: unknown role {role!r} "
                f"(expected one of {sorted(_VALID_ROLES)})")

    castings = [b for b in active if b["role"] == "casting"]
    chills = [b for b in active if b["role"] == "chill"]
    sleeves = [b for b in active if b["role"] in _SLEEVE_ROLES]

    if len(castings) != 1:
        raise ValueError(
            f"training-regime restriction: expected exactly one body with "
            f"role='casting', found {len(castings)}")
    casting_solid_id = castings[0]["solid_id"]

    if chills and sleeves:
        raise ValueError(
            "training-regime restriction: chills and a sleeve cannot be "
            "combined -- the trained model has no mixed chills+sleeve BC "
            "channel in the current training regimes")
    if sleeves:
        sleeve_roles = {b["role"] for b in sleeves}
        if len(sleeve_roles) > 1:
            raise ValueError(
                f"training-regime restriction: all sleeve bodies must share "
                f"one role, got {sorted(sleeve_roles)} -- the trained model "
                f"has a single sleeve BC channel and cannot represent an "
                f"insulating and an exothermic sleeve at once")

    if chills:
        materials = {_resolve_chill_material(b.get("params")) for b in chills}
        effs = load_chill_effusivities(chills_csv)
        for material in materials:
            if material not in effs:
                raise ValueError(
                    f"unknown chill material {material!r} (known: "
                    f"{sorted(effs)})")
        if len(materials) > 1:
            raise ValueError(
                f"training-regime restriction: all chills must share one "
                f"material (single chill_effusivity per training "
                f"convention), got {sorted(materials)}")
        material = next(iter(materials))
        chill_ids = [b["solid_id"] for b in chills]
        return BcPlan(
            bc_config="chills",
            casting_solid_id=casting_solid_id,
            role_solid_ids=chill_ids,
            n_chills=len(chills),
            chill_effusivity=float(effs[material]),
            role_by_solid={sid: "chill" for sid in chill_ids},
        )

    if sleeves:
        bc_config = sleeves[0]["role"]
        registry = None
        mef_by_solid: dict[int, float] = {}
        for sleeve in sleeves:
            params = sleeve.get("params") or {}
            has_name = "sleeve" in params
            has_mef = "MEF" in params
            if has_name and has_mef:
                raise ValueError(
                    f"sleeve solid_id={sleeve['solid_id']!r}: specify either "
                    f"'sleeve' (registry name) or 'MEF' (manual override), not "
                    f"both")
            if not has_name and not has_mef:
                raise ValueError(
                    f"sleeve solid_id={sleeve['solid_id']!r}: params must "
                    f"contain either 'sleeve' (registry name) or 'MEF' (manual "
                    f"override)")
            if has_name:
                if registry is None:
                    registry = _load_sleeve_registry(sleeve_registry_csv)
                name = params["sleeve"]
                if name not in registry:
                    raise ValueError(
                        f"unknown sleeve name {name!r} (known: "
                        f"{sorted(registry)})")
                entry = registry[name]
                expected_type = _ROLE_TO_SLEEVE_TYPE[bc_config]
                if entry["type"] != expected_type:
                    raise ValueError(
                        f"sleeve {name!r} is type={entry['type']} but was "
                        f"assigned role {bc_config} (expected "
                        f"type={expected_type!r})")
                mef_by_solid[sleeve["solid_id"]] = float(entry["MEF"])
            else:
                mef_by_solid[sleeve["solid_id"]] = float(params["MEF"])

        sleeve_ids = [b["solid_id"] for b in sleeves]
        mefs = [mef_by_solid[sid] for sid in sleeve_ids]
        spread = float(max(mefs) - min(mefs))
        if spread <= _MEF_AGREEMENT_TOL:
            sleeve_mef = float(mefs[0])
            weighted = False
            spread = 0.0
        else:
            if mef_weights is None:
                raise ValueError(
                    f"sleeve bodies {sleeve_ids} resolve to differing MEF "
                    f"{sorted(set(mefs))} (spread {spread:.4f}) -- the model has "
                    f"a single scalar sleeve_MEF channel, so a mixed set needs "
                    f"explicit `mef_weights` (contact-patch area per solid_id, "
                    f"as returned by `tag_contact_patches`'s "
                    f"'per_body_patch_area_mm2'). Refusing to guess a "
                    f"weighting. Contact-patch areas only exist after meshing "
                    f"and tagging, so a caller that only has `bodies` (e.g. "
                    f"`predict_casting`'s CLI/webapp entry point) cannot supply "
                    f"them here -- mesh and tag the part first (`get_or_mesh` "
                    f"+ `tag_contact_patches`), then call `validate_bodies` "
                    f"again with `mef_weights=per_body_patch_area_mm2`, as "
                    f"`tools/sp4/prep.py` and `tools/sp4/score.py` do for the "
                    f"SP4 gold set.")
            missing = [sid for sid in sleeve_ids if sid not in mef_weights]
            if missing:
                raise ValueError(
                    f"mef_weights is missing solid_id(s) {missing} -- every "
                    f"sleeve body needs a weight, got keys "
                    f"{sorted(mef_weights)}")
            weights = [float(mef_weights[sid]) for sid in sleeve_ids]
            if any(w <= 0.0 for w in weights):
                raise ValueError(
                    f"mef_weights must all be positive contact areas, got "
                    f"{dict(zip(sleeve_ids, weights))}")
            total_w = sum(weights)
            sleeve_mef = float(sum(m * w for m, w in zip(mefs, weights)) / total_w)
            weighted = True

        exo_q0 = 0.0
        if bc_config == "exo_sleeve":
            exo_q0 = resolve_exo_q0_by_mef(sleeve_mef, exo_calibration_csv, tol)

        return BcPlan(
            bc_config=bc_config,
            casting_solid_id=casting_solid_id,
            role_solid_ids=sleeve_ids,
            sleeve_MEF=sleeve_mef,
            exo_q0=exo_q0,
            role_by_solid={sid: bc_config for sid in sleeve_ids},
            mef_weighted=weighted,
            mef_spread=spread,
        )

    return BcPlan(bc_config="none", casting_solid_id=casting_solid_id, role_solid_ids=[])


# --------------------------------------------------------------------------
# Contact-patch tagging -- Part B Task 3: mesh the user-designated casting
# solid, classify its surface triangles by proximity to the role bodies'
# BReps, and rewrite the mesh's facet groups so the contact patch becomes
# "Adiabatic" (the exact BC-patch convention `_node_features(version=4)` /
# `fenicsx_solver.solve_case` read -- see the module docstring's B3 design
# reference).
# --------------------------------------------------------------------------
_ADIABATIC_PHYS_TAG = 3   # alongside prep_parts._VOLUME_PHYS_TAG=1 / _COOLING_PHYS_TAG=2


def _triangle_areas(node_xyz: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """(M,) triangle areas (mm^2) via the standard half cross-product
    magnitude -- `tris` is (M, 3) 0-based vertex indices into `node_xyz`."""
    p0 = node_xyz[tris[:, 0]]
    p1 = node_xyz[tris[:, 1]]
    p2 = node_xyz[tris[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _sample_solid_surface_points(solid_id: int) -> np.ndarray:
    """Assumes a fresh gmsh/OCC session already has the model imported,
    synchronized, AND 2-D-meshed (`gmsh.model.mesh.generate(2)` already ran
    -- done ONCE per `tag_contact_patches` call, not per body, since it
    covers every solid's boundary surfaces in one pass). Returns (P, 3) the
    coordinates of every mesh node lying on any boundary surface
    (`gmsh.model.getBoundary`) of 3-D entity `solid_id`."""
    import gmsh

    boundary = gmsh.model.getBoundary([(3, solid_id)], oriented=False)
    surf_tags = [t for (d, t) in boundary if d == 2]
    if not surf_tags:
        raise ValueError(f"role body {solid_id}: no boundary surfaces found (not a 3-D solid?)")

    pts = []
    for surf_tag in surf_tags:
        _node_tags, coords, _param = gmsh.model.mesh.getNodes(2, surf_tag, includeBoundary=True)
        pts.append(np.asarray(coords, dtype=np.float64).reshape(-1, 3))
    points = np.concatenate(pts, axis=0)
    if points.shape[0] == 0:
        raise ValueError(f"role body {solid_id}: surface sampling yielded 0 points")
    return points


def _distances_to_solid(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(M,) nearest-neighbor distance from each row of `centroids` (M, 3, mm)
    to the sampled boundary point cloud `points` (P, 3, mm) of one role
    solid, via `scipy.spatial.cKDTree`.

    WHY sampled points + KDTree, not `gmsh.model.getClosestPoint` directly
    on the centroids: probed first (as the brief suggested trying), and
    found to be not just slower but WRONG for axis-aligned primitive faces
    (the chill/sleeve boxes and cylinders this module builds and that real
    role bodies resemble) -- `getClosestPoint`'s orthogonal projection lands
    on the face's UNDERLYING (untrimmed) surface, not the actual bounded
    patch (`isInside()` on the returned point came back False in every case
    probed), so a query point far outside a face's true extent can still
    project to a deceptively small distance whenever it happens to share
    that face's plane (e.g. a base box's OWN top face reported ~2mm to a
    chill's bottom-face plane from a corner ~15mm from the chill's actual
    footprint). Meshing the role solids' real (trimmed) boundary at a size
    tied to `tol_mm` and taking nearest-neighbor distance to those sampled
    points respects the true finite geometry and reproduces the expected
    footprint exactly in this task's synthetic fixtures (measured
    corner-case distance corrected from a wrong ~2mm to the correct
    ~11.2mm). Timing (composer fixture, `Mesh.MeshSizeMax = tol_mm/2`,
    2-D-meshing the WHOLE imported model once): 0.16-0.9s total for
    408-11019-node castings (656-8328 Cooling triangles) -- see
    task-3-report.md for the full measurement."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    dist, _ = tree.query(centroids)
    return np.asarray(dist, dtype=np.float64)


def _parse_physical_names_block(lines: list[str], start_idx: int):
    """`lines[start_idx]` must be the bare `"$PhysicalNames"` marker line.
    Returns `(count_idx, entries, end_idx)`: `count_idx` is the index of the
    entry-count line, `entries` is a list of `(dim, tag, name)` (name
    unquoted), and `end_idx` is the index of the `"$EndPhysicalNames"`
    line."""
    count_idx = start_idx + 1
    cnt = int(lines[count_idx].strip())
    entries = []
    for k in range(cnt):
        dim_s, tag_s, name_s = lines[count_idx + 1 + k].split(maxsplit=2)
        entries.append((int(dim_s), int(tag_s), name_s.strip().strip('"')))
    end_idx = count_idx + 1 + cnt
    return count_idx, entries, end_idx


def rewrite_msh_adiabatic(msh_path: str, is_adiabatic: np.ndarray) -> None:
    """Rewrite `msh_path` in place so that the Cooling-tagged surface
    triangles flagged True in `is_adiabatic` (same length AND order as
    `src.msh_parse.parse_msh(msh_path).tris["Cooling"]` -- both this
    function's own `$Elements` scan and `parse_msh` walk the file's
    triangle elements in on-disk order, so the two arrays line up
    positionally) move to a new "Adiabatic" physical group (pinned tag
    `_ADIABATIC_PHYS_TAG`, reusing an existing "Adiabatic" group's tag if
    the mesh already has one).

    Deliberately edits the raw text rather than reimporting through gmsh:
    `$Nodes` is never touched and every `$Elements` line is copied through
    unchanged except the physical-group tag (first tag) of the flagged
    triangle lines -- this is what guarantees node ids/coords and tet
    connectivity survive EXACTLY (features/geodesics depend on it; verified
    by the caller re-parsing and comparing `node_xyz`/`tets`). Raises
    loudly (`ValueError`/`AssertionError`) on any structural surprise --
    missing sections, a malformed `$PhysicalNames` block, tag-3 already
    bound to a different name, or an `is_adiabatic` length that doesn't
    match the number of Cooling-tagged triangle elements actually found --
    rather than silently writing a mismatched mesh."""
    from src.msh_parse import _TRI_TYPE

    lines = Path(msh_path).read_text().splitlines()

    try:
        phn_idx = lines.index("$PhysicalNames")
    except ValueError:
        raise ValueError(f"{msh_path}: missing $PhysicalNames section")
    count_idx, entries, end_idx = _parse_physical_names_block(lines, phn_idx)
    if lines[end_idx].strip() != "$EndPhysicalNames":
        raise ValueError(f"{msh_path}: malformed $PhysicalNames block (no $EndPhysicalNames "
                          f"at the expected offset)")

    cooling_entries = [e for e in entries if e[0] == 2 and e[2] == "Cooling"]
    if not cooling_entries:
        raise ValueError(f"{msh_path}: no dim=2 'Cooling' physical group found")
    cooling_tag = cooling_entries[0][1]

    existing_adiabatic = [e for e in entries if e[0] == 2 and e[2] == "Adiabatic"]
    if existing_adiabatic:
        adiabatic_tag = existing_adiabatic[0][1]
        new_entries = entries
    else:
        collision = [e for e in entries if e[0] == 2 and e[1] == _ADIABATIC_PHYS_TAG]
        if collision:
            raise ValueError(f"{msh_path}: physical tag {_ADIABATIC_PHYS_TAG} is already "
                              f"bound to {collision[0][2]!r}; cannot pin 'Adiabatic' there")
        adiabatic_tag = _ADIABATIC_PHYS_TAG
        new_entries = entries + [(2, adiabatic_tag, "Adiabatic")]

    new_phys_lines = [str(len(new_entries))] + [
        f'{d} {t} "{nm}"' for (d, t, nm) in new_entries
    ]

    try:
        elem_idx = lines.index("$Elements")
    except ValueError:
        raise ValueError(f"{msh_path}: missing $Elements section")
    elem_count_idx = elem_idx + 1
    elem_cnt = int(lines[elem_count_idx].strip())
    elem_lines = lines[elem_count_idx + 1: elem_count_idx + 1 + elem_cnt]

    flags = is_adiabatic.tolist()
    n_flags = len(flags)
    flag_i = 0
    new_elem_lines = []
    for ln in elem_lines:
        parts = ln.split()
        etype = int(parts[1])
        if etype == _TRI_TYPE and int(parts[3]) == cooling_tag:
            if flag_i >= n_flags:
                raise AssertionError(
                    f"{msh_path}: is_adiabatic has only {n_flags} flag(s) -- fewer than the "
                    f"Cooling-tagged triangle elements found in $Elements -- refusing to write "
                    f"a mismatched mesh")
            if flags[flag_i]:
                parts[3] = str(adiabatic_tag)
                ln = " ".join(parts)
            flag_i += 1
        new_elem_lines.append(ln)
    if flag_i != n_flags:
        raise AssertionError(
            f"{msh_path}: is_adiabatic has {n_flags - flag_i} more flag(s) than Cooling-tagged "
            f"triangle elements found in $Elements -- refusing to write a mismatched mesh")

    new_lines = (
        lines[:count_idx] + new_phys_lines + lines[end_idx:elem_count_idx]
        + [str(elem_cnt)] + new_elem_lines + lines[elem_count_idx + 1 + elem_cnt:]
    )
    Path(msh_path).write_text("\n".join(new_lines) + "\n")


# Back-compat alias -- this function was originally private (`_rewrite_msh_
# adiabatic`); FT3's `tools/doe/virtual_patches.py` reuses it under the
# promoted public name `rewrite_msh_adiabatic`, but existing call sites/tests
# (this module's own `tag_contact_patches`, `tests/test_assembly.py`) still
# reference the old private name, so both names resolve to the same function.
_rewrite_msh_adiabatic = rewrite_msh_adiabatic


def tag_contact_patches(msh_path: str, step_path: str, role_solid_ids: Sequence[int],
                         tol_mm: float | None = None) -> dict:
    """Classify the casting mesh's (`msh_path`) "Cooling" surface triangles
    by proximity to each of `role_solid_ids`'s BReps in `step_path` (a fresh
    gmsh/OCC session, distinct from the casting mesh's own session --
    `step_path` is the ORIGINAL multi-body assembly STEP the role bodies'
    tags came from, e.g. `list_solids`' `solid_id`, not the reduced
    single-solid STEP the casting was meshed from); triangles within
    `tol_mm` of ANY role body move from "Cooling" to a new "Adiabatic"
    facet group (see `_rewrite_msh_adiabatic`), rewriting `msh_path` IN
    PLACE. Distance-to-body samples each role solid's real (trimmed)
    boundary surfaces into a point cloud (one shared `gmsh.model.mesh.
    generate(2)` pass over the whole imported model, sized off `tol_mm`)
    and takes the nearest-neighbor distance via `scipy.spatial.cKDTree`
    (`_sample_solid_surface_points` + `_distances_to_solid`) -- see
    `_distances_to_solid`'s docstring for why a direct
    `gmsh.model.getClosestPoint` probe was rejected (not just slower, but
    wrong for bounded planar faces).

    `tol_mm`, if `None` (default), is `0.5 * elem_est` where `elem_est =
    bbox_diag / n_nodes**(1/3)` of the casting mesh (all nodes, matching
    `heal_and_mesh`'s own whole-model bbox convention) -- a rough local
    element-size estimate, so the tolerance scales with mesh density rather
    than being a fixed physical distance that's too tight for a coarse mesh
    or too loose for a fine one.

    CAVEAT -- tol_mm dilates the detected patch past the true contact
    footprint: since classification is "distance to body <= tol_mm", any
    positive tol_mm grows the patch outward from the geometric contact edge
    by roughly `perimeter * tol_mm + pi * tol_mm**2` (a rectangular-
    footprint Minkowski-sum estimate). This matters most for the SYNTHETIC
    fixtures `compose_assembly` builds with a large POSITIVE `overlap_mm`
    (sunk INTO the base -- see that function's docstring): `tol_mm` must
    exceed `overlap_mm` just to detect the patch at all, and when
    `overlap_mm` is a large fraction of the footprint size, the resulting
    dilation can be a large fraction of the footprint's own area (measured
    up to +140% on a 10x10mm footprint with a 2mm overlap -- see
    task-3-report.md). Real assemblies -- where role bodies sit at
    near-zero contact or a sub-mm clearance rather than several mm sunk in
    -- only need a small `tol_mm` to bridge that gap, so this dilation
    stays a small fraction of a realistic contact-patch area; it is a
    synthetic-fixture-scale concern, not a production one.

    A role body whose patch comes out EMPTY (no Cooling triangle centroid
    within `tol_mm`) raises `ValueError` naming that body and `tol_mm` --
    never a silent all-sand prediction for a body the caller explicitly
    assigned a role to. Symmetrically, if EVERY Cooling triangle ends up
    classified Adiabatic (role bodies + tol_mm cover the entire casting
    surface), this raises `ValueError` too, rather than silently handing
    downstream feature code (`geodesic_d_cool` et al., see
    `src.features`) a casting mesh with no cooling surface at all to
    measure geodesic distance to -- which would silently degrade to
    bbox_diag-derived constants (confidently wrong physics, not a loud
    failure).

    Returns `{"per_body_patch_area_mm2": {solid_id: float}, "n_adiab_tris":
    int, "nodes_by_solid": {solid_id: sorted list[int]}}`.
    `per_body_patch_area_mm2` sums each body's own classified
    triangle areas INDEPENDENTLY -- a triangle within `tol_mm` of more than
    one role body (their tolerance zones overlap) is counted once per such
    body, i.e. DOUBLE-COUNTED across `per_body_patch_area_mm2`'s entries,
    even though it is relabeled to "Adiabatic" exactly once; `n_adiab_tris`
    counts each such triangle once (via the union `is_adiabatic` mask), so
    `sum(per_body_patch_area_mm2.values())` can exceed the true total
    Adiabatic area when role bodies' patches overlap.

    `nodes_by_solid` (Part-B roles task, 2026-08-03): the sorted-unique node
    ids (0-based, into `mesh.node_xyz` -- the SAME indexing as every other
    node id this module and `src.msh_parse` use) of each role body's own
    classified triangles -- computed from the SAME per-body `close` mask as
    `per_body_patch_area_mm2`, so it shares its double-counting property (a
    triangle within `tol_mm` of more than one body contributes its nodes to
    EVERY such body's entry). This is an IN-MEMORY-only piece of metadata:
    it is never written into `msh_path` itself (the `.msh` physical groups,
    node order, and this function's cache-relevant hashes/tags are all
    unchanged by its addition) -- callers needing a role-aware node
    selection (e.g. `predict_casting`'s shrinkage-screening feed-node logic)
    combine this per-SOLID mapping with their own solid_id -> role lookup
    (e.g. `BcPlan.role_by_solid`) themselves; `tag_contact_patches` has no
    opinion on role semantics, only geometry. After rewriting,
    re-parses `msh_path` and asserts `node_xyz` is unchanged (`np.allclose`)
    and `tets` is unchanged (`np.array_equal`) -- the node-order/tet-
    connectivity invariant features/geodesics depend on (see the module
    docstring's caution and `_rewrite_msh_adiabatic`)."""
    from src.msh_parse import parse_msh
    import gmsh

    mesh = parse_msh(msh_path)
    cooling_tris = mesh.tris.get("Cooling")
    if cooling_tris is None or cooling_tris.shape[0] == 0:
        raise ValueError(f"{msh_path}: no 'Cooling' surface triangles to classify")

    centroids = mesh.node_xyz[cooling_tris].mean(axis=1)
    tri_areas = _triangle_areas(mesh.node_xyz, cooling_tris)

    if tol_mm is None:
        bbox_min = mesh.node_xyz.min(axis=0)
        bbox_max = mesh.node_xyz.max(axis=0)
        bbox_diag_mm = float(np.linalg.norm(bbox_max - bbox_min))
        n_nodes = mesh.node_xyz.shape[0]
        elem_est = bbox_diag_mm / (n_nodes ** (1.0 / 3.0))
        tol_mm = 0.5 * elem_est

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("tag_contact_patches")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()

        # One shared 2-D surface mesh (not per-body) -- covers every role
        # solid's boundary in a single pass. Sized off tol_mm/2 so sampled
        # points are dense enough relative to the tolerance to not miss or
        # blur the true footprint.
        sample_size_mm = max(tol_mm / 2.0, 1e-3)
        gmsh.option.setNumber("Mesh.MeshSizeMin", sample_size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeMax", sample_size_mm)
        gmsh.model.mesh.generate(2)

        dist_by_body = {}
        for solid_id in role_solid_ids:
            points = _sample_solid_surface_points(solid_id)
            dist_by_body[solid_id] = _distances_to_solid(points, centroids)
    finally:
        gmsh.finalize()

    is_adiabatic = np.zeros(cooling_tris.shape[0], dtype=bool)
    per_body_patch_area_mm2: dict[int, float] = {}
    nodes_by_solid: dict[int, list[int]] = {}
    for solid_id, dist in dist_by_body.items():
        close = dist <= tol_mm
        area = float(tri_areas[close].sum())
        if area <= 0.0:
            raise ValueError(f"body {solid_id} does not touch the casting within "
                              f"tol={tol_mm:.4f}mm")
        per_body_patch_area_mm2[solid_id] = area
        nodes_by_solid[solid_id] = sorted(
            int(n) for n in np.unique(cooling_tris[close].reshape(-1)))
        is_adiabatic |= close

    if bool(is_adiabatic.all()):
        raise ValueError(
            "role bodies cover the entire casting surface; no cooling surface remains "
            f"-- check roles/tol (tol={tol_mm:.4f}mm)")

    rewrite_msh_adiabatic(msh_path, is_adiabatic)

    mesh_after = parse_msh(msh_path)
    if not np.allclose(mesh_after.node_xyz, mesh.node_xyz):
        raise AssertionError(f"{msh_path}: node_xyz changed after Adiabatic rewrite "
                              f"(node order/coords must be byte-identical)")
    if mesh_after.tets.shape != mesh.tets.shape or not np.array_equal(mesh_after.tets, mesh.tets):
        raise AssertionError(f"{msh_path}: tets changed after Adiabatic rewrite "
                              f"(tet connectivity must be byte-identical)")

    return {"per_body_patch_area_mm2": per_body_patch_area_mm2,
            "n_adiab_tris": int(is_adiabatic.sum()),
            "nodes_by_solid": nodes_by_solid}


# --------------------------------------------------------------------------
# CLI printer
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="List the 3-D solids in a (possibly multi-body) STEP file.")
    ap.add_argument("step_path", help="path to a .step/.stp file")
    args = ap.parse_args(argv)

    rows = list_solids(args.step_path)
    header = f"{'solid_id':>8}  {'name':<24}  {'volume_mm3':>14}  {'bbox (mm)':<48}  {'com (mm)':<30}"
    print(header)
    print("-" * len(header))
    for row in rows:
        bbox_str = "[" + ", ".join(f"{v:.1f}" for v in row["bbox"]) + "]"
        com_str = "[" + ", ".join(f"{v:.1f}" for v in row["com"]) + "]"
        name = row["name"] or "<unnamed>"
        print(f"{row['solid_id']:>8}  {name:<24}  {row['volume_mm3']:>14.2f}  "
              f"{bbox_str:<48}  {com_str:<30}")
    print(f"\n{len(rows)} solid(s) total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
