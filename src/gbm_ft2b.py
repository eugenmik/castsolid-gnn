"""FT2b Task 6: multi-alloy GBM scale-head refit.

The production predictor is a HYBRID: GNN field *shape* x a per-graph
*scale* from a HistGradientBoostingRegressor (see `src/hybrid.py`,
`runs/baseline_v2/hgb_A.joblib`). That GBM was trained on `src/features.py`
v2 physics features from the SINGLE-alloy dataset and knows nothing about
material properties, so it cannot generalize its scale prediction across
alloys/mold systems.

This module refits a NEW scale-head GBM directly on raw material properties
+ geometry scale + mold-system effusivity -- exactly the quantities that
determine Chvorinov-style total solidification time -- resolved per training
run_id from the FT2a/FT2b DOE manifest via the SAME resolution path used to
build the v4 feature cache (`src.build_dataset._resolve_material_props_v4`).

Feature row layout (see FEATURE_NAMES): [k, rho, cp, L_kJ_kg, TL, TS, Tpour,
V_over_A, mold_effusivity] (9 material/geometry columns) + [bc_none, bc_chills,
bc_insulating_sleeve, bc_exo_sleeve] (4 BC one-hot from manifest.bc_config) +
[n_chills, sleeve_MEF, exo_q0_MW] (3 BC scalar fields) = 16 columns. `Tpour` IS a
per-material property in this project's DB (`tools.doe.load_properties.load_materials`
midpoints Tpour_lo_C/Tpour_hi_C for the 12 named anchor grades; the LHS-sampled
synthetic alloys carry their own `Tpour_C` in `materials_synthetic.csv`) -- so it is
resolved per material_id, not hardcoded as a constant.

Target: per-graph `log1p(y_raw.max())` -- the total solidification time
(scale), read straight from the v4 graph cache (`y_raw`, physical seconds).
"""
from __future__ import annotations

import csv
import os
import os.path as osp

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src.build_dataset import (FEATURE_NAMES_V4, _resolve_exo_q0_v4,
                               _resolve_material_props_v4)
from src.features import FEATURE_NAMES as _NODE_FEATURE_NAMES

# volume_mm3/surface_area_mm2 are constant-per-node columns of the v2/v3/v4
# node-feature matrix X (see src/features.py FEATURE_NAMES and the v4
# comment there: "v3's 13 columns UNCHANGED (X[:, :9] == v2, X[:, :13] ==
# v3)" -- so these indices are valid for the v4 cache too). Verified live
# against src/build_dataset.py:622-623, which reads the SAME two indices
# out of the SAME FEATURE_NAMES list to get volume/area for the Chvorinov
# t_ref fit:
#     volume = float(X[0, FEATURE_NAMES.index("volume_mm3")])
#     area = float(X[0, FEATURE_NAMES.index("surface_area_mm2")])
_VOL_IDX = _NODE_FEATURE_NAMES.index("volume_mm3")
_AREA_IDX = _NODE_FEATURE_NAMES.index("surface_area_mm2")

# bc_area_frac (soft-anchor Task 2, std head v2 17th feature) column indices --
# resolved off `src.build_dataset.FEATURE_NAMES_V4` (never hardcoded silently)
# with a cross-check assertion against the brief's stated indices
# (is_cooling=3, is_adiabatic=4): if FEATURE_NAMES_V4 ever changes upstream,
# this fails loudly instead of silently reading the wrong columns.
_IS_COOLING_IDX = FEATURE_NAMES_V4.index("is_cooling")
_IS_ADIABATIC_IDX = FEATURE_NAMES_V4.index("is_adiabatic")
assert (_IS_COOLING_IDX, _IS_ADIABATIC_IDX) == (3, 4), (
    f"bc_area_frac column indices drifted from the Global Constraints formula's "
    f"is_cooling=3/is_adiabatic=4 (FEATURE_NAMES_V4 now gives "
    f"is_cooling={_IS_COOLING_IDX}, is_adiabatic={_IS_ADIABATIC_IDX}) -- "
    f"FEATURE_NAMES_V4 changed upstream; update this module before trusting "
    f"bc_area_frac_from_graph")

# GBM row feature layout -- exposed for the Task 8 evaluator so it can
# rebuild rows the exact same way at inference time.
#
# BC-aware extension (exo-fix Task 3): the original 9 material/geometry
# columns are blind to boundary conditions (chills/sleeves change total
# solidification time independently of material/geometry), so the scale
# head could not tell a chilled casting from a plain sand one. The 7 new
# columns make it BC-aware: a one-hot over the manifest's `bc_config`
# values (`none`/`chills`/`insulating_sleeve`/`exo_sleeve` -- the only
# values `tools/doe/doe.py` ever writes), plus the raw `n_chills` and
# `sleeve_MEF` manifest fields, plus the calibrated exothermic-sleeve heat
# release `exo_q0_MW` (0.0 for non-exo rows, by `_resolve_exo_q0_v4`'s own
# contract).
FEATURE_NAMES = [
    "k", "rho", "cp", "L_kJ_kg", "TL", "TS", "Tpour",
    "V_over_A", "mold_effusivity",
    "bc_none", "bc_chills", "bc_insulating_sleeve", "bc_exo_sleeve",
    "n_chills", "sleeve_MEF", "exo_q0_MW",
]

# The only bc_config values `tools/doe/doe.py` ever writes to the manifest
# (verified live against doe_out/manifest.csv: Counter gives exactly these
# 4 keys). An empty string means "no BC row was ever assigned" and is
# treated as "none" (mirrors `_resolve_exo_q0_v4`'s own
# `row.get("bc_config") or "none"` convention). Anything else is an
# unrecognized/typo'd value -- raise loudly rather than silently ignoring it
# in the one-hot.
_BC_VALUES = ("none", "chills", "insulating_sleeve", "exo_sleeve")

_HGB_DEFAULTS = dict(learning_rate=0.1, max_leaf_nodes=63,
                      l2_regularization=1.0, random_state=0)

# HGB's own early_stopping carves off validation_fraction (default 0.1) of
# whatever it's given -- on tiny row counts (e.g. a ~20-stem unit test) that
# leaves 1-2 validation rows, which stops training almost immediately and
# badly underfits. Only turn early stopping on once there's enough data for
# it to be a meaningful signal; production refits (thousands of stems) still
# get it.
_MIN_ROWS_FOR_EARLY_STOPPING = 50


def _load_manifest_rows(manifest_path: str) -> dict:
    rows = {}
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["run_id"]] = row
    return rows


def _num_or_zero(raw, field: str, stem: str) -> float:
    """float(raw), except '' / None -> 0.0 (the brief's explicit rule for
    n_chills/sleeve_MEF, mirroring how the manifest encodes N/A for rows
    with no chill/sleeve BC). A present-but-non-numeric value still raises
    loudly (e.g. a typo) -- only blank is treated as "not applicable"."""
    if raw is None or str(raw).strip() == "":
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"run_id={stem!r}: {field}={raw!r} is not a number") from e


def _row_features_and_graph(stem: str, manifest_rows: dict, manifest_dir: str,
                             features_dir: str) -> tuple[np.ndarray, "np.lib.npyio.NpzFile"]:
    """Shared row-assembly: one BC-aware feature row (see `FEATURE_NAMES`) +
    the loaded graph-cache npz. Factored out of `_row_features` so the
    std-target head (`_row_features_std`) reuses the IDENTICAL X (same
    material/BC resolution, same I/O) and swaps only the target it derives
    from the already-open graph -- never re-deriving or duplicating the
    resolution logic.

    Loud-fails (raises) on anything unresolvable -- a missing manifest row,
    a missing/unresolvable material_id, a missing mold_effusivity, an
    unrecognized bc_config value, an uncalibrated exo_sleeve row, or a
    missing graph-cache file -- NEVER silently drops or fabricates a value.
    The one EXCEPTION (per the brief) is n_chills/sleeve_MEF: '' -> 0.0,
    mirroring how the manifest encodes "no chill/sleeve BC" for this row.
    """
    row = manifest_rows.get(stem)
    if row is None:
        raise KeyError(f"stem {stem!r} not found in manifest ({manifest_dir!r})")

    material_id = (row.get("material_id") or "").strip()
    if not material_id:
        raise KeyError(f"run_id={stem!r}: manifest row has no material_id")
    # Raises KeyError if unresolvable via any of the 3 fallback sources
    # (owner-verified CSV / materials_synthetic.csv / named anchor grades) --
    # see src.build_dataset._resolve_material_props_v4's own docstring.
    props = _resolve_material_props_v4(material_id, manifest_dir)

    mold_eff_raw = row.get("mold_effusivity")
    if mold_eff_raw is None or str(mold_eff_raw).strip() == "":
        raise KeyError(f"run_id={stem!r}: manifest row has no mold_effusivity")
    mold_effusivity = float(mold_eff_raw)

    bc_config = (row.get("bc_config") or "").strip() or "none"
    if bc_config not in _BC_VALUES:
        raise ValueError(
            f"run_id={stem!r}: unrecognized bc_config={bc_config!r} "
            f"(expected one of {_BC_VALUES})")
    one_hot = [1.0 if bc_config == v else 0.0 for v in _BC_VALUES]

    n_chills = _num_or_zero(row.get("n_chills"), "n_chills", stem)
    sleeve_MEF = _num_or_zero(row.get("sleeve_MEF"), "sleeve_MEF", stem)

    # 0.0 for non-exo rows by its own contract; raises loudly for
    # bc_config=="exo_sleeve" rows with no calibrated MEF match -- see
    # src.build_dataset._resolve_exo_q0_v4's docstring.
    exo_q0_MW = _resolve_exo_q0_v4(row) / 1e6

    gpath = osp.join(features_dir, "graphs", f"{stem}.npz")
    if not osp.exists(gpath):
        raise FileNotFoundError(f"graph cache missing for stem={stem!r}: {gpath}")
    d = np.load(gpath)
    volume = float(d["X"][0, _VOL_IDX])
    area = float(d["X"][0, _AREA_IDX])
    if area <= 0:
        raise ValueError(f"run_id={stem!r}: non-positive surface_area_mm2={area}")
    v_over_a = volume / area

    x = np.array([
        props["k"], props["rho"], props["cp"], props["L_kJ_kg"],
        props["TL"], props["TS"], props["Tpour"],
        v_over_a, mold_effusivity,
        *one_hot,
        n_chills, sleeve_MEF, exo_q0_MW,
    ], dtype=np.float64)
    return x, d


def _row_features(stem: str, manifest_rows: dict, manifest_dir: str,
                   features_dir: str) -> tuple[np.ndarray, float]:
    """One BC-aware feature row (see `FEATURE_NAMES`) + log1p(y_global).

    See `_row_features_and_graph` for the loud-fail contract this shares.
    """
    x, d = _row_features_and_graph(stem, manifest_rows, manifest_dir, features_dir)
    y_raw = d["y_raw"] if "y_raw" in d.files else d["y"]
    y_log = float(np.log1p(float(np.max(y_raw))))
    return x, y_log


def _row_features_std(stem: str, manifest_rows: dict, manifest_dir: str,
                       features_dir: str) -> tuple[np.ndarray, float]:
    """The SAME BC-aware feature row as `_row_features` (see
    `_row_features_and_graph`) + log1p(std(y_nodal)) -- the per-graph
    field-spread target for the GBM std head (variance-recomposition Task 2).

    `y_nodal` is the graph npz's `y` field (per-node solidification time,
    seconds) -- raises loudly (KeyError) if a requested stem's graph cache
    has no `y`, never silently skips it.
    """
    x, d = _row_features_and_graph(stem, manifest_rows, manifest_dir, features_dir)
    if "y" not in d.files:
        raise KeyError(f"run_id={stem!r}: graph cache in {features_dir!r} has no 'y' field")
    y_std_log = float(np.log1p(float(np.std(d["y"]))))
    return x, y_std_log


# std head v2 (soft-anchor Task 2): the v1 row (`FEATURE_NAMES`, 16 columns)
# + one graph-derived 17th column, `bc_area_frac`. Kept as its own module
# constant (rather than mutating `FEATURE_NAMES`) so the 16-feature total
# head stays UNTOUCHED -- see `std_feature_row`'s width-dispatch contract.
FEATURE_NAMES_STD_V2 = FEATURE_NAMES + ["bc_area_frac"]


def bc_area_frac_from_graph(graph) -> float:
    """The std-head-v2 17th feature (Global Constraints formula, soft-anchor
    spec Sec 2): the fraction of BC-tagged surface nodes that are adiabatic
    (vs. cooling) -- `sum(is_adiabatic) / max(sum(is_adiabatic) +
    sum(is_cooling), 1)`, read straight off the graph's node feature matrix
    `graph["X"]` at `_IS_ADIABATIC_IDX`/`_IS_COOLING_IDX` (columns 4/3 of
    `FEATURE_NAMES_V4`, cross-checked at import time above -- never
    hardcoded silently a second time here).

    `graph` is whatever `_row_features_and_graph` returns as its second
    element (an open graph-cache npz) -- or, for tests, any object
    supporting `graph["X"]` indexing (e.g. a plain dict).

    A node-count proxy for the true tri-area fraction (no per-node area
    weight is available in `X`), acceptable per the spec. All-sand rows
    (no adiabatic surface at all, e.g. the fully-cooled regime) -> exactly
    0.0, since the numerator is 0 regardless of the denominator; a graph
    with neither cooling nor adiabatic nodes tagged also -> 0.0 (the `max(
    ..., 1)` floor avoids a 0/0 division).
    """
    X = graph["X"]
    n_adiabatic = float(np.sum(X[:, _IS_ADIABATIC_IDX]))
    n_cooling = float(np.sum(X[:, _IS_COOLING_IDX]))
    return n_adiabatic / max(n_adiabatic + n_cooling, 1.0)


def std_feature_row(stem: str, manifest_rows: dict, manifest_dir: str,
                    features_dir: str, n_features: int) -> np.ndarray:
    """Width-dispatching feature-row builder for the std head (soft-anchor
    Task 2) -- lets a caller (Task 3's `predict_casting`/`evaluate_ft2b`)
    build the right-width row for whichever std-head model it loaded,
    keyed off that model's own `n_features_in_`, without duplicating
    `_row_features_and_graph`'s resolution logic.

    - `n_features == 16` -> the v1 row (`_row_features_and_graph`'s X,
      unchanged -- byte-identical to `_row_features`'s own row).
    - `n_features == 17` -> the v1 row + `bc_area_frac_from_graph` of the
      SAME already-open graph appended (never a second graph load).
    - anything else -> a loud `ValueError` naming the two supported widths
      (never a silent truncation/pad).
    """
    if n_features not in (16, 17):
        raise ValueError(
            f"std_feature_row: unsupported n_features={n_features!r} for "
            f"stem={stem!r} -- the std head only supports 16 (v1) or 17 "
            f"(v2, v1 + bc_area_frac) feature columns")

    x, d = _row_features_and_graph(stem, manifest_rows, manifest_dir, features_dir)
    if n_features == 16:
        return x
    area_frac = bc_area_frac_from_graph(d)
    return np.concatenate([x, [area_frac]])


def build_v2_feature_row(props: dict, v_over_a: float,
                         bc_spec: dict | None = None) -> np.ndarray:
    """One BC-aware v2-total-GBM feature row (`FEATURE_NAMES` order) -- the
    bare-mesh-inference counterpart to `_row_features_and_graph`'s
    manifest/graph-cache-driven row (that one resolves `props`/`v_over_a`
    from a manifest row + graph cache; this one takes them straight from
    the caller, for real-part inference where there is no manifest row).

    Public (no leading underscore) and factored out of
    `gbm_v2_total_seconds` -- rather than duplicated wherever a caller needs
    the total head's row (`src.predict_casting.predict_field`'s optional std
    head, variance-recomposition Task 3) -- so the total head
    (`gbm_v2_total_seconds`) and any std head scored from the SAME row
    (`predict_std_seconds`) can never silently desync on a `FEATURE_NAMES`
    reorder. Mirrors `_row_features_and_graph`'s own "so the two heads can
    never drift apart" rationale (Task 2).

    Builds the row from `props` (Task 1's `src.inference_props.
    resolve_props()` 8-key dict: k/rho/cp/L_kJ_kg/TL/TS/Tpour/
    mold_effusivity), `v_over_a` (volume/surface-area, mm), and `bc_spec`:
        - `None` (the default, "plain sand mold" case) -> `bc_config`
          "none" one-hot, `n_chills`/`sleeve_MEF`/`exo_q0_MW` all 0.0.
        - a dict with keys `bc_config`/`n_chills`/`sleeve_MEF`/`exo_q0_MW`
          (Part B's BC-aware inference entry point) -> that `bc_config`'s
          one-hot + its 3 raw scalar fields, verbatim.

    An unrecognized `bc_config` raises `ValueError` (mirrors `_row_features`'
    own convention for a manifest row with a typo'd `bc_config`).
    """
    if bc_spec is None:
        bc_config = "none"
        n_chills = 0.0
        sleeve_MEF = 0.0
        exo_q0_MW = 0.0
    else:
        bc_config = bc_spec["bc_config"]
        n_chills = float(bc_spec["n_chills"])
        sleeve_MEF = float(bc_spec["sleeve_MEF"])
        exo_q0_MW = float(bc_spec["exo_q0_MW"])

    if bc_config not in _BC_VALUES:
        raise ValueError(
            f"unrecognized bc_config={bc_config!r} (expected one of {_BC_VALUES})")
    one_hot = [1.0 if bc_config == v else 0.0 for v in _BC_VALUES]

    return np.array([
        props["k"], props["rho"], props["cp"], props["L_kJ_kg"],
        props["TL"], props["TS"], props["Tpour"],
        float(v_over_a), props["mold_effusivity"],
        *one_hot,
        n_chills, sleeve_MEF, exo_q0_MW,
    ], dtype=np.float64)


def gbm_v2_total_seconds(hgb, props: dict, v_over_a: float,
                         bc_spec: dict | None = None) -> float:
    """Bare-mesh inference entry point: one feature row -> total seconds.

    Builds the row via `build_v2_feature_row` (see that function for the
    full `props`/`v_over_a`/`bc_spec` contract) and returns
    `float(np.expm1(hgb.predict(row)[0]))` -- physical seconds, inverting
    the same `log1p(y_raw.max())` target `_row_features`/`refit_gbm` train
    on.
    """
    x = build_v2_feature_row(props, v_over_a, bc_spec)
    log_pred = float(hgb.predict(x.reshape(1, -1))[0])
    return float(np.expm1(log_pred))


def _assemble_matrix(stems, manifest_path: str, features_dir: str):
    """[len(stems), len(FEATURE_NAMES)] feature matrix + log1p(y_global) target vector.

    Raises (does not skip) on any row that cannot be resolved -- see
    `_row_features`.
    """
    manifest_dir = osp.dirname(osp.abspath(manifest_path)) or "."
    manifest_rows = _load_manifest_rows(manifest_path)

    X_rows, y_rows, used = [], [], []
    for stem in stems:
        x, y_log = _row_features(stem, manifest_rows, manifest_dir, features_dir)
        X_rows.append(x)
        y_rows.append(y_log)
        used.append(stem)

    X = np.stack(X_rows) if X_rows else np.empty((0, len(FEATURE_NAMES)))
    y = np.array(y_rows, dtype=np.float64)
    return X, y, used


def _assemble_matrix_std(stems, manifest_path: str, features_dir: str):
    """[len(stems), len(FEATURE_NAMES)] feature matrix + log1p(std(y_nodal))
    target vector -- the std-head counterpart to `_assemble_matrix`. Same X
    (via `_row_features_std` -> `_row_features_and_graph`), different target.

    Raises (does not skip) on any row that cannot be resolved -- see
    `_row_features_std`.
    """
    manifest_dir = osp.dirname(osp.abspath(manifest_path)) or "."
    manifest_rows = _load_manifest_rows(manifest_path)

    X_rows, y_rows, used = [], [], []
    for stem in stems:
        x, y_log = _row_features_std(stem, manifest_rows, manifest_dir, features_dir)
        X_rows.append(x)
        y_rows.append(y_log)
        used.append(stem)

    X = np.stack(X_rows) if X_rows else np.empty((0, len(FEATURE_NAMES)))
    y = np.array(y_rows, dtype=np.float64)
    return X, y, used


def _fit_hgb_with_metrics(X: np.ndarray, y: np.ndarray, out_path: str,
                           max_iter: int, holdout_frac: float,
                           min_samples_leaf: int, random_state: int,
                           caller: str, loss_kwargs: dict | None = None) -> dict:
    """Shared HGB-fit/holdout/save logic behind `refit_gbm`, `refit_gbm_std`,
    and `refit_gbm_std_v2` -- identical modeling code, only the assembled
    (X, y) (and, for std v2, the loss) differ between heads. Factored out so
    the heads can never drift apart in their fitting/metrics logic;
    `refit_gbm`'s and `refit_gbm_std`'s own public behavior is unchanged by
    this factoring (see the guard tests in `tests/test_gbm_ft2b.py`).

    `min_samples_leaf` (sklearn's HGB default is 20) must be turned down for
    very small row counts (e.g. a ~20-row unit test) -- HGB cannot split a
    node at all once a candidate split would leave either side with fewer
    than `min_samples_leaf` samples, so with the sklearn default and only
    ~20 training rows every tree degenerates to a single leaf (the
    unconditional mean). Production refits (thousands of stems) should keep
    the default.

    `loss_kwargs` (soft-anchor Task 2): optional extra
    `HistGradientBoostingRegressor` constructor kwargs (e.g.
    `{"loss": "quantile", "quantile": 0.5}` for the std-head-v2 median
    objective), merged into BOTH the holdout probe and the final model so a
    caller never has to duplicate this fit/holdout/save logic just to swap
    the loss. `None` (the default) reproduces the exact pre-existing
    behavior (sklearn's default squared-error loss) -- `refit_gbm`'s and
    `refit_gbm_std`'s own output is therefore byte-identical to before this
    parameter existed.

    Returns a dict of finite metrics: `n_train`, `train_mae_log1p`,
    `train_mae_seconds`, and (when there are enough rows to spare a
    holdout split) `holdout_mae_log1p`.
    """
    if X.shape[0] == 0:
        raise ValueError(f"{caller}: no training stems resolved -- nothing to fit")

    metrics: dict = {"n_train": X.shape[0]}

    # Bonus: a quick internal holdout split for a generalization signal.
    # Skipped for tiny row counts (both HGB's own internal validation split
    # and this outer split would starve each other of data).
    if X.shape[0] >= 10 and holdout_frac > 0:
        X_tr, X_ho, y_tr, y_ho = train_test_split(
            X, y, test_size=holdout_frac, random_state=random_state)
        probe_params = dict(_HGB_DEFAULTS)
        probe_params["random_state"] = random_state
        if loss_kwargs:
            probe_params.update(loss_kwargs)
        probe = HistGradientBoostingRegressor(
            max_iter=max_iter,
            early_stopping=(len(X_tr) >= _MIN_ROWS_FOR_EARLY_STOPPING),
            validation_fraction=0.1, min_samples_leaf=min_samples_leaf,
            **probe_params)
        probe.fit(X_tr, y_tr)
        ho_mae = float(np.mean(np.abs(probe.predict(X_ho) - y_ho)))
        if np.isfinite(ho_mae):
            metrics["holdout_mae_log1p"] = ho_mae

    # Final production model: fit on ALL resolved training rows.
    model_params = dict(_HGB_DEFAULTS)
    model_params["random_state"] = random_state
    if loss_kwargs:
        model_params.update(loss_kwargs)
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        early_stopping=(X.shape[0] >= _MIN_ROWS_FOR_EARLY_STOPPING),
        validation_fraction=0.1, min_samples_leaf=min_samples_leaf,
        **model_params)
    model.fit(X, y)

    train_pred = model.predict(X)
    metrics["train_mae_log1p"] = float(np.mean(np.abs(train_pred - y)))
    metrics["train_mae_seconds"] = float(
        np.mean(np.abs(np.expm1(train_pred) - np.expm1(y))))

    out_dir = osp.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, out_path)
    metrics["out_path"] = out_path

    return metrics


def refit_gbm(features_dir: str, train_stems, manifest_path: str, out_path: str,
              max_iter: int = 300, holdout_frac: float = 0.2,
              min_samples_leaf: int = 20, random_state: int = 0) -> dict:
    """Refit the multi-alloy GBM scale-head (total solidification time target,
    `log1p(y_global)`) and save it to `out_path`. See `_fit_hgb_with_metrics`
    for the shared fitting/metrics logic.

    Returns a dict of finite metrics: `n_train`, `train_mae_log1p`,
    `train_mae_seconds`, and (when there are enough rows to spare a
    holdout split) `holdout_mae_log1p`.
    """
    X, y, used = _assemble_matrix(train_stems, manifest_path, features_dir)
    return _fit_hgb_with_metrics(
        X, y, out_path, max_iter=max_iter, holdout_frac=holdout_frac,
        min_samples_leaf=min_samples_leaf, random_state=random_state,
        caller="refit_gbm")


def refit_gbm_std(features_dir: str, train_stems, manifest_path: str, out_path: str,
                   max_iter: int = 300, holdout_frac: float = 0.2,
                   min_samples_leaf: int = 20, random_state: int = 0) -> dict:
    """Refit the GBM std-head (per-row field-spread target,
    `log1p(std(y_nodal))`) and save it to `out_path` -- the counterpart to
    `refit_gbm`'s total-time head, for variance-preserving hybrid
    recomposition (see `src.hybrid.hybrid_field_affine`, Task 1).

    Consumes the SAME feature row layout (`FEATURE_NAMES`) as `refit_gbm`,
    built via `_assemble_matrix_std` (which reuses `_row_features_and_graph`
    -- identical material/BC resolution, only the target differs). See
    `_fit_hgb_with_metrics` for the shared fitting/metrics logic.

    Returns a dict of finite metrics: `n_train`, `train_mae_log1p`,
    `train_mae_seconds`, and (when there are enough rows to spare a
    holdout split) `holdout_mae_log1p`.
    """
    X, y, used = _assemble_matrix_std(train_stems, manifest_path, features_dir)
    return _fit_hgb_with_metrics(
        X, y, out_path, max_iter=max_iter, holdout_frac=holdout_frac,
        min_samples_leaf=min_samples_leaf, random_state=random_state,
        caller="refit_gbm_std")


def _assemble_matrix_std_v2(stems, manifest_path: str, features_dir: str):
    """[len(stems), 17] feature matrix (v1 row + `bc_area_frac`) +
    `log1p(std(y_nodal))` target vector -- the std-head-v2 counterpart to
    `_assemble_matrix_std`. Reuses `_row_features_and_graph` for the shared
    v1 row AND derives `bc_area_frac_from_graph` off that SAME already-open
    graph (never a second graph load per row, mirroring
    `_row_features_and_graph`'s own "so the two heads can never drift
    apart" rationale).

    Raises (does not skip) on any row that cannot be resolved -- see
    `_row_features_and_graph`; also raises `KeyError` if a stem's graph
    cache has no `y` field (mirrors `_row_features_std`).
    """
    manifest_dir = osp.dirname(osp.abspath(manifest_path)) or "."
    manifest_rows = _load_manifest_rows(manifest_path)

    X_rows, y_rows, used = [], [], []
    for stem in stems:
        x, d = _row_features_and_graph(stem, manifest_rows, manifest_dir, features_dir)
        if "y" not in d.files:
            raise KeyError(f"run_id={stem!r}: graph cache in {features_dir!r} has no 'y' field")
        area_frac = bc_area_frac_from_graph(d)
        x17 = np.concatenate([x, [area_frac]])
        y_log = float(np.log1p(float(np.std(d["y"]))))
        X_rows.append(x17)
        y_rows.append(y_log)
        used.append(stem)

    X = np.stack(X_rows) if X_rows else np.empty((0, len(FEATURE_NAMES_STD_V2)))
    y = np.array(y_rows, dtype=np.float64)
    return X, y, used


def refit_gbm_std_v2(features_dir: str, train_stems, manifest_path: str, out_path: str,
                     max_iter: int = 300, holdout_frac: float = 0.2,
                     min_samples_leaf: int = 20, random_state: int = 0) -> dict:
    """Refit the std-head v2 (soft-anchor Task 2): the SAME per-row
    field-spread target as `refit_gbm_std` (`log1p(std(y_nodal))`), but on
    17 features (`FEATURE_NAMES_STD_V2` = the v1 16 + graph-derived
    `bc_area_frac`) and with a median (quantile=0.5) loss instead of squared
    error -- tail-robust against the per-stem noise that squared-error
    over-weights (see the soft-anchor design spec Sec 2).

    Built via `_assemble_matrix_std_v2` (identical material/BC/graph
    resolution to `_assemble_matrix_std`, plus the extra column). See
    `_fit_hgb_with_metrics` for the shared fitting/metrics logic; this is
    the one caller that passes `loss_kwargs`.

    Returns a dict of finite metrics: `n_train`, `train_mae_log1p`,
    `train_mae_seconds`, and (when there are enough rows to spare a
    holdout split) `holdout_mae_log1p`.
    """
    X, y, used = _assemble_matrix_std_v2(train_stems, manifest_path, features_dir)
    return _fit_hgb_with_metrics(
        X, y, out_path, max_iter=max_iter, holdout_frac=holdout_frac,
        min_samples_leaf=min_samples_leaf, random_state=random_state,
        caller="refit_gbm_std_v2", loss_kwargs={"loss": "quantile", "quantile": 0.5})


def predict_std_seconds(model, X_row) -> float:
    """One feature row (`FEATURE_NAMES` order, shape `(len(FEATURE_NAMES),)`
    or `(1, len(FEATURE_NAMES))`) -> predicted std(y_nodal) in seconds.

    Inverts the `log1p(std(y_nodal))` target `refit_gbm_std` trains on via
    `expm1`, floored at 0.0 (a predicted std can never be physically
    negative; `expm1` of a very negative log-prediction could underflow to
    a tiny negative float).
    """
    x = np.asarray(X_row, dtype=np.float64).reshape(1, -1)
    log_pred = float(model.predict(x)[0])
    return max(0.0, float(np.expm1(log_pred)))
