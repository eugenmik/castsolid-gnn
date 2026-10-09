"""Real-casting inference: hybrid field prediction + VTU output.

predict_field combines the GNN field shape with the baseline per-graph scale
(src/hybrid.py). write_vtu exports a ParaView file. load_hybrid loads the
Phase-1 checkpoints. The predict_casting orchestrator (Task 5) ties in meshing.

v4 (film) path (Task 3): `cfg.get("film")` truthy routes `predict_field`
through the multi-alloy/multi-BC stack instead -- v4 node features (BC
spec fixed to a plain sand mold, per this module's bare-mesh-inference
contract: no chills/sleeves info comes from a STEP file), the alloy/mould
`g` vector, the v4 GNN (`src.evaluate_ft2b`), and the v2-total GBM
(`src.gbm_ft2b`) recomposed via the quantile-anchor (`hybrid_field_max_anchor`,
default `q=0.995` -- outlier-robust; `q=1.0` is the original exact-MAX
anchor) -- NOT the SP1 MEAN-anchor, which assumes a genuine per-node
baseline (see `src/evaluate_ft2b.py`'s module docstring for the full
rationale). The non-film path below is functionally unchanged (plus an explicit HGB-dimension guard).

variance-recomposition Task 3: an optional GBM std-head (`src.gbm_ft2b.
refit_gbm_std`'s output, loaded via `load_hybrid`'s `hgb_std_path` /
`predict_casting`'s `--hgb-std`) can be threaded through the film path so
`predict_field` recomposes via `src.hybrid.hybrid_field_affine` (rescales
the GNN field's deviations to match a per-graph SCALE target too, not just
a per-graph LEVEL) instead of `hybrid_field_max_anchor`. `None` (the
default, everywhere) is byte-identical to the pre-Task-3 code.

variance-recomposition amendment 2: `predict_casting`'s `--affine-lambda` /
`predict_field`'s `lambda_shrink` (default `1.0`, forwarded straight to
`hybrid_field_affine`) geometrically shrinks the affine spread target
toward the GNN's own spread -- see `src.hybrid.hybrid_field_affine`'s
docstring for the math/rationale. `1.0` everywhere is byte-identical to
before this argument existed.

soft-anchor Task 3: `hybrid_field_affine` at the std-head call site above
is REPLACED by `src.hybrid.hybrid_field_soft` (a weighted two-target
objective that lets the total anchor itself deviate a bit when that trade
buys a much closer spread match -- see that function's docstring). The
degeneration guarantee built into `hybrid_field_soft` means any case where
the hard affine solve already satisfied both targets recomposes
IDENTICALLY (same array object, same `a`) -- only genuinely stuck cases
(the ones that motivated this track) actually change. `feat["recomposition"]`
is now `"soft"` (was `"affine"`) whenever `hgb_std` is active; the info dict
gains `"m"`/`"target_unreachable"`/`"total_deviation_frac"`/`"j_total_resid"`/
`"j_std_resid"` (see `hybrid_field_soft`'s own docstring for the full key
list) in place of affine's `"a_clamped"`. New `w_total`/`w_std` params
(`predict_field`, `predict_casting`, CLI `--w-total`/`--w-std`; production
defaults `DEFAULT_W_TOTAL=1.0` / `DEFAULT_W_STD=4.0` since the v5.1
promotion) forward straight to `hybrid_field_soft`'s own weights, letting a
caller trade total-anchor fidelity against spread fidelity. The GBM std
head's own row width is now dispatched off the loaded model's
`n_features_in_` (16 -> the v1 row unchanged, 17 -> the v1 row + a
graph-derived `bc_area_frac` 17th column computed from the SAME in-memory
per-node feature array `predict_field` already has -- see
`_std_row_for_predict_field`); anything else raises a loud `ValueError`.
The no-std-head (quantile-anchor) path is completely untouched by any of
this.

variance-recomposition Task 6 (the production flip, 2026-07-31):
`DEFAULT_GNN_CKPT`/`DEFAULT_HGB` now point at the ft3 stack
(`runs/ft3_stage1/ckpt_ep021.pt` x `runs/ft3_gbm/hgb_v3.joblib`) and a new
`DEFAULT_HGB_STD` (now `runs/ft3_gbm/hgb_std_v2.joblib` since the v5.1
promotion) exists. `hgb_std_path`
no longer defaults to plain `None` on `predict_casting`/CLI `--hgb-std`:
both default to the module-level `_USE_DEFAULT` sentinel, which
`load_hybrid` resolves to `DEFAULT_HGB_STD` for a film-capable checkpoint
or to `None` for a non-film one (see `load_hybrid`'s docstring) -- so the
production call, with zero extra flags, now recomposes via
`hybrid_field_affine` (`lambda_shrink=1.0`) instead of the quantile-anchor
(superseded by soft-anchor Task 3 below -- the zero-flag production call
recomposes via `hybrid_field_soft` today; this paragraph is kept as the
Task 6 historical record of when the std head first became the default),
while the SP1 rollback (`--gnn`/`--hgb` pointed at a non-film checkpoint)
is untouched: the sentinel silently resolves to `None` there, exactly as
if `hgb_std_path` had never been passed. The case-insensitive string
`"none"` always disables the std head regardless of checkpoint kind (the
explicit rollback switch to the quantile-anchor, usable on any film
checkpoint, including the new default one) -- passing an explicit path or
loaded model object still loads/uses it unconditionally, exactly as
before Task 6.

learned-heads Task 3 (`--gnn` pointed at a `deriv_heads`-capable checkpoint):
`predict_field`'s film branch also runs `src.evaluate_ft2b.predict_graph_derivs`
and stashes the result in `feat["gradT"]`/`feat["dTdt"]` (K/m / K/s); `predict_casting`
then feeds those into `src.shrinkage.screen_casting`'s exact Niyama branch
instead of the time-only proxy, and relabels the result `niyama_source=
"learned"` (screen_casting's own "exact" label is reserved for a real cached
solver field). Absent entirely for a headless checkpoint -- byte-identical
to the pre-Task-3 proxy-only behavior.

bc-reach Task 4 (cfg-driven v5 feature building at inference): the film
branch's `compute_features_inference` call now reads
`version=cfg.get("feature_version", 4)` instead of hardcoding `version=4`
-- the SAME `cfg.get("feature_version", ...)` pattern the non-film branch
already used (this module, the SP1/non-film path). A v5 (bc-reach)
checkpoint's saved cfg carries `feature_version=5` (`src.train_gnn`'s
`_build_checkpoint_dict`, echoed verbatim from `train(feature_version=...)`,
independent of but in practice equal to `node_in`), so loading one through
`load_hybrid`/`load_ft2b_gnn` and calling `predict_field` now builds the
20-column v5 `X` (`src.features.FEATURE_NAMES_V5`: v4's 15 columns +
`exo_mid`/`exo_long`/`chill_short`/`chill_mid`/`chill_long`) with NO other
change to this module -- `compute_bc_reach_channels` (features v5) already
resolves its chill/exo source patch off the SAME `spec`/`bc_spec` dicts
this module already builds from the all-sand literal, so the reach channels see EXACTLY the same source magnitudes the
training builder (`src.build_dataset.build_v5`) does. A v4 (or absent)
`cfg["feature_version"]` is byte-identical to the pre-Task-4 code (defaults
to `4`, same as before this dispatch existed)."""
import hashlib
import json
import logging
import os.path as osp
import numpy as np
import torch
import joblib

logger = logging.getLogger(__name__)

from src.features import (compute_features_inference, FEATURE_NAMES, FEATURE_NAMES_V4,
                          alloy_g_vector)
from src.dataload import standardize
from src.baseline import predict_seconds
from src.evaluate_gnn import predict_graph_seconds, load_gnn, N_BASELINE_FEATURES
from src.evaluate_ft2b import predict_graph_seconds as predict_graph_seconds_v4, load_ft2b_gnn
from src.evaluate_ft2b import predict_graph_derivs as predict_graph_derivs_v4
from src.gbm_ft2b import (gbm_v2_total_seconds, FEATURE_NAMES as GBM_FEATURE_NAMES,
                          predict_std_seconds, build_v2_feature_row,
                          bc_area_frac_from_graph)
from src.hybrid import hybrid_field, hybrid_field_max_anchor, hybrid_field_soft
from src.inference_props import resolve_props, DEFAULT_ALLOY, resolve_alloy_family
from src.shrinkage import screen_casting
from src.release_io import resolve_artifact, load_gbm, load_gnn_any

N_GBM_V2_FEATURES = len(GBM_FEATURE_NAMES)
_V4_VOL_IDX = FEATURE_NAMES_V4.index("volume_mm3")
_V4_AREA_IDX = FEATURE_NAMES_V4.index("surface_area_mm2")

# Production v5.1 hybrid stack (promoted 2026-07-31, soft-anchor Task 6 -- see
# docs/superpowers/specs/2026-07-31-soft-anchor-design.md's ADDENDUM) -- module
# constants so CLI/kwarg defaults live in one place instead of being duplicated
# string literals. GNN field *shape* x GBM *scale* (total) x GBM *spread* (std,
# std_v2 -- median-loss HGB, 17 features incl. bc_area_frac), recomposed via the
# soft-anchor map (src.hybrid.hybrid_field_soft) at w_total=1.0, w_std=4.0
# (the e_val sweep winner: v2+soft takes gold decisively -- BUKSA 90.2 s/0.700,
# H17 120.1 s/0.895, best-ever MAE on both -- missing only the (owner-revised,
# see tools/ft3_gates.py's DEFAULT_E_VAL_STD_RATIO_FLOOR) e_val exo std_ratio floor).
#
# Rollback #1 (v5, 2026-07-31 same day, pre-soft-anchor-sweep): ckpt_ep021 x
# hgb_v3 x hgb_std_v1 (squared-error, 16 features), w_total=1.0, w_std=1.0 --
# reachable via `--hgb-std runs/ft3_gbm/hgb_std_v1.joblib --w-std 1.0`
# (hgb_std_v1.joblib is left untouched on disk by this promotion).
# Rollback #2 (previous production, 2026-07-28..2026-07-31): ckpt_ep054 x hgb_v2,
# quantile-anchor recomposition (no std head) -- reachable via
# `--gnn runs/ft2b_stage2c/ckpt_ep054.pt --hgb runs/ft2b_gbm/hgb_v2.joblib --hgb-std none`.
# Rollback #3 (SP1, still on disk): runs/sp1_run3_final/best_gnn.pt +
# runs/baseline_v2/hgb_A.joblib, non-film -- reachable via --gnn/--hgb; --hgb-std needs
# no extra flag here, `load_hybrid` auto-resolves the defaulted std head to None for a
# non-film checkpoint (see `load_hybrid`'s docstring / the `_USE_DEFAULT` sentinel below).
DEFAULT_GNN_CKPT = "runs/ft3_stage1/ckpt_ep021.pt"
DEFAULT_HGB = "runs/ft3_gbm/hgb_v3.joblib"
DEFAULT_HGB_STD = "runs/ft3_gbm/hgb_std_v2.joblib"
DEFAULT_W_TOTAL = 1.0
DEFAULT_W_STD = 4.0


class _UseDefaultHgbStd:
    """Sentinel for `predict_casting`'s `hgb_std_path` param / CLI `--hgb-std`:
    means "the caller did not specify" -- `load_hybrid` resolves this to
    `DEFAULT_HGB_STD` when the selected checkpoint is film-capable, or to
    `None` when it isn't (so the SP1 rollback via `--gnn`/`--hgb` needs no
    extra flag). Distinct from an explicit `None` or the string `"none"`
    (case-insensitive), both of which ALWAYS disable the std head -- the
    explicit rollback switch to the quantile-anchor recomposition, usable on
    any film checkpoint including the new default one."""

    def __repr__(self):
        return "<DEFAULT_HGB_STD>"


_USE_DEFAULT = _UseDefaultHgbStd()


def _check_hgb_dim(hgb_model, expected: int, ckpt_kind: str) -> None:
    """Loud, clear failure on a mismatched (film vs. SP1) GNN/HGB pairing.

    `hgb.n_features_in_` is set by sklearn on `.fit()`; a stub without it
    (shouldn't happen for a real fitted estimator) is treated as "unknown"
    and NOT checked, rather than crashing the attribute lookup itself.
    """
    n_have = getattr(hgb_model, "n_features_in_", None)
    if n_have is None or n_have == expected:
        return
    raise ValueError(
        f"predict_field: {ckpt_kind} checkpoint expects an HGB baseline with "
        f"{expected} input features, but hgb_model.n_features_in_={n_have!r} -- "
        f"this looks like the wrong baseline for this checkpoint (a v2-total "
        f"GBM has {N_GBM_V2_FEATURES} features, an SP1 per-node HGB has "
        f"{N_BASELINE_FEATURES}); pass the matching --hgb for this --gnn checkpoint.")


def _std_row_for_predict_field(row_16: np.ndarray, X: np.ndarray, n_features: int
                               ) -> np.ndarray:
    """`predict_field`'s in-memory counterpart to `src.gbm_ft2b.std_feature_row`
    (soft-anchor Task 3): width-dispatches the std-head row off the loaded
    std model's own `n_features_in_`, without a manifest stem or graph-cache
    file to read from -- `predict_field` already has everything both widths
    need in memory.

    - `n_features == 16` -> `row_16` unchanged (the SAME 16-column row
      `predict_field`'s total head already scores -- byte-identical to the
      pre-soft-anchor code).
    - `n_features == 17` -> `row_16` + `src.gbm_ft2b.bc_area_frac_from_graph`'s
      17th column, computed from the SAME in-memory per-node v4 feature
      array `X` (`bc_area_frac_from_graph` only ever indexes `graph["X"]`, so
      a plain `{"X": X}` dict satisfies its contract without a real graph
      npz).
    - anything else -> a loud `ValueError` naming the two supported widths
      (mirrors `std_feature_row`'s own contract).
    """
    if n_features not in (16, 17):
        raise ValueError(
            f"predict_field: unsupported std-head n_features_in_={n_features!r} -- "
            f"the std head only supports 16 (v1) or 17 (v2, v1 + bc_area_frac) "
            f"feature columns")
    if n_features == 16:
        return row_16
    area_frac = bc_area_frac_from_graph({"X": X})
    return np.concatenate([row_16, [area_frac]])


def predict_field(mesh, gnn_model, hgb_model, norm, cfg=None, device=None, tta=0,
                  k=16, baseline_norm=None, props=None, hgb_std=None,
                  lambda_shrink: float = 1.0, w_total: float = DEFAULT_W_TOTAL,
                  w_std: float = DEFAULT_W_STD) -> tuple:
    """The film branch always uses the all-sand boundary spec: the whole
    outer surface cools into the mold, and the model's chill/sleeve input
    channels stay at zero (this release supports the sand-mold regime only).

    `hgb_std` (variance-recomposition Task 3, recomposition swapped to
    `hybrid_field_soft` by soft-anchor Task 3): an optional loaded GBM
    std-head model (`src.gbm_ft2b.refit_gbm_std`/`refit_gbm_std_v2`'s
    output -- e.g. loaded via `load_hybrid`'s `hgb_std_path`). Only
    consulted on the film (v4) path, and only once the total head has
    produced a valid `total`: the 16-column total-head row is built ONCE
    (`src.gbm_ft2b.build_v2_feature_row`, the SAME public row-builder
    `gbm_v2_total_seconds` itself uses, so a `FEATURE_NAMES` reorder can
    never silently desync the two heads) and fed to the total head
    (`hgb_model.predict` directly, mathematically identical to
    `gbm_v2_total_seconds`); the std head is scored on a SEPARATE row,
    width-dispatched off `hgb_std.n_features_in_` (16 -> that same row
    unchanged, 17 -> that row + a graph-derived `bc_area_frac` 17th column,
    via `_std_row_for_predict_field` -- else a loud `ValueError`), then
    recomposed via `src.hybrid.hybrid_field_soft` instead of
    `hybrid_field_max_anchor`.
    `None` (the default) is BYTE-IDENTICAL to the pre-Task-3 code (the
    `total = gbm_v2_total_seconds(...)` / `hybrid_field_max_anchor` path
    below is untouched). When active, the returned `feat` dict gains
    `feat["recomposition"] = "soft"` and `feat["recomposition_info"]`
    (`hybrid_field_soft`'s `info`: `a`/`m`/`target_unreachable`/
    `degenerate_fallback`/`clip_frac`/`total_deviation_frac`/
    `j_total_resid`/`j_std_resid`/`lambda_shrink`/`std_target_effective`)
    -- absent entirely when `hgb_std` is `None`. Ignored entirely by the
    non-film (SP1) path -- SP1 has no scale-head std target.

    `lambda_shrink` (variance-recomposition amendment 2): forwarded straight
    to `hybrid_field_soft` (ignored when `hgb_std` is `None`, same as every
    other soft-anchor-only knob here). `1.0` (the default) is byte-identical
    to before this argument existed.

    `w_total`/`w_std` (soft-anchor Task 3; defaults changed to `DEFAULT_W_TOTAL`/
    `DEFAULT_W_STD` by the soft-anchor Task 6 promotion, 2026-07-31): forwarded
    straight to `src.hybrid.hybrid_field_soft`'s own weights (ignored when
    `hgb_std` is `None`). The defaults here (`1.0`/`4.0`) are the e_val-sweep
    winner recorded in the spec addendum -- NOT `hybrid_field_soft`'s own
    function-level default of `1.0`/`1.0` (that default is unchanged; a
    caller invoking `hybrid_field_soft` directly, bypassing `predict_field`,
    still gets equal weighting unless it passes these explicitly).

    Learned-heads Task 3: when `cfg["deriv_heads"]` is truthy (a heads
    checkpoint, film path only -- SP1 has no deriv heads at all), `feat`
    additionally carries `feat["gradT"]`/`feat["dTdt"]` -- the checkpoint's
    OWN learned per-node gradT (K/m) / dTdt (K/s) predictions
    (`src.evaluate_ft2b.predict_graph_derivs`), on the SAME features the
    shape head just scored. This is a `feat`-dict addition only -- the
    `(pred, feat)` 2-tuple return contract is unchanged, so every existing
    `out, feat = predict_field(...)` call site keeps working unmodified.
    Absent entirely for a headless checkpoint (the default): `"gradT"`/
    `"dTdt"` simply aren't keys of `feat` then, exactly as before this
    feature existed. `predict_casting` reads these (if present) and feeds
    them to `screen_casting`'s exact Niyama branch, labeling the result
    `niyama_source="learned"` (see that function's own docstring)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg or {"k": k, "graph": "knn", "feature_version": 2}
    gnn_model = gnn_model.to(device).eval()

    if cfg.get("film", False):
        if props is None:
            raise ValueError(
                "predict_field: cfg indicates a film (v4) checkpoint but no "
                "`props` (alloy/mould conditioning dict) were given")
        if hgb_model is None:
            raise ValueError(
                "predict_field: a film (v4) checkpoint requires an hgb_model "
                "(the v2-total GBM scale head) -- got None")
        _check_hgb_dim(hgb_model, N_GBM_V2_FEATURES, "film (v4)")

        spec = {"mold_effusivity": props["mold_effusivity"], "bc_config": "none",
                "chill_effusivity": 0.0, "sleeve_MEF": 0.0, "exo_q0": 0.0}
        bc_spec = None
        feat = compute_features_inference(mesh, version=cfg.get("feature_version", 4), spec=spec)
        X, pos = feat["X"], feat["pos"]
        g_vec = alloy_g_vector(props)
        gnn_seconds = predict_graph_seconds_v4(gnn_model, cfg, norm, X, pos,
                                               feat["mesh_edges"], g_vec=g_vec,
                                               device=device)

        # Learned-heads Task 3: a deriv_heads-capable checkpoint additionally
        # predicts per-node gradT (K/m) / dTdt (K/s) off the SAME features --
        # stashed into `feat` (not the return tuple's arity, keeping every
        # existing `out, feat = predict_field(...)` call site unchanged) so
        # `predict_casting` can feed them to `screen_casting`'s exact branch
        # instead of the time-only proxy. Absent entirely (no "gradT"/"dTdt"
        # keys in `feat`) for a headless checkpoint -- byte-identical to the
        # pre-Task-3 code for every existing (non-heads) checkpoint.
        if cfg.get("deriv_heads", False):
            gradT_pred, dTdt_pred = predict_graph_derivs_v4(
                gnn_model, cfg, norm, X, pos, feat["mesh_edges"], g_vec=g_vec,
                device=device)
            feat["gradT"] = gradT_pred
            feat["dTdt"] = dTdt_pred

        v_over_a = float(X[0, _V4_VOL_IDX]) / float(X[0, _V4_AREA_IDX])

        if hgb_std is not None:
            row = build_v2_feature_row(props, v_over_a, bc_spec)
            total = float(np.expm1(float(hgb_model.predict(row.reshape(1, -1))[0])))
            if total <= 0:
                raise ValueError(
                    f"predict_field: the v2-total GBM head returned a non-positive "
                    f"total ({total!r}) for props={props!r} -- the v2-total GBM is "
                    f"broken for this input")
            n_std_features = int(getattr(hgb_std, "n_features_in_", 16))
            std_row = _std_row_for_predict_field(row, X, n_std_features)
            std_seconds = predict_std_seconds(hgb_std, std_row)
            pred, info = hybrid_field_soft(gnn_seconds, total, std_seconds,
                                           lambda_shrink=lambda_shrink,
                                           w_total=w_total, w_std=w_std)
            feat["recomposition"] = "soft"
            feat["recomposition_info"] = info
            return pred, feat

        total = gbm_v2_total_seconds(hgb_model, props, v_over_a, bc_spec=bc_spec)
        if total <= 0:
            raise ValueError(
                f"predict_field: gbm_v2_total_seconds returned a non-positive "
                f"total ({total!r}) for props={props!r} -- the v2-total GBM is "
                f"broken for this input")
        return hybrid_field_max_anchor(gnn_seconds, total), feat

    # --- non-film (SP1) path: functionally unchanged (plus explicit HGB-dimension guard) ---
    if hgb_model is not None:
        _check_hgb_dim(hgb_model, N_BASELINE_FEATURES, "non-film (SP1)")
    feat = compute_features_inference(mesh, version=cfg.get("feature_version", 2))
    X, pos = feat["X"], feat["pos"]
    Xs = standardize(X, norm)
    x = torch.tensor(Xs, dtype=torch.float32)
    pos_t = torch.tensor(pos, dtype=torch.float32)
    me = (torch.tensor(feat["mesh_edges"], dtype=torch.long)
          if cfg.get("graph", "knn") == "union" else None)
    g = predict_graph_seconds(gnn_model, cfg, norm, x, pos_t, me, device=device, tta=tta)
    if hgb_model is None:
        return g, feat
    bn = baseline_norm if baseline_norm is not None else norm
    b = predict_seconds(hgb_model, standardize(X[:, :N_BASELINE_FEATURES], bn), bn)
    return hybrid_field(g, b), feat


def write_vtu(out_path, node_xyz, tets, seconds, top_frac=0.1, extra_point_data=None):
    """`extra_point_data` (SP2 shrinkage Task 4): an optional
    `{field_name: (N,) array}` dict merged into the written `point_data`
    dict alongside the original `solidification_time`/`hotspot` fields --
    e.g. `{"niyama": ..., "porosity_risk": ..., "macro_region_id": ...}`
    from `src.shrinkage.screen_casting`. `None` (the default) is
    byte-identical to the pre-Task-4 contract (only the original two
    fields are written) -- this is a generic extension point, not a
    hardcoded shrinkage-only special case, so any future per-node field
    can reuse it the same way."""
    import meshio
    seconds = np.asarray(seconds, dtype=np.float32)
    n = seconds.shape[0]
    k = max(1, int(round(n * top_frac)))
    thresh = np.sort(seconds)[-k]
    hotspot = (seconds >= thresh).astype(np.int32)
    point_data = {"solidification_time": seconds, "hotspot": hotspot}
    if extra_point_data:
        point_data.update(extra_point_data)
    m = meshio.Mesh(points=np.asarray(node_xyz, dtype=np.float64),
                    cells=[("tetra", np.asarray(tets))],
                    point_data=point_data)
    m.write(out_path)


# --------------------------------------------------------------------------- #
# SP2 shrinkage Task 4: `predict_casting` integration helpers -- deriving
# screen_casting's mesh-topology inputs from the parsed `.msh`, trimming its
# result to a return-dict-friendly shape, and formatting the summary's risk
# block. Kept as free functions (rather than inlined in `predict_casting`)
# so each piece is independently testable and the main function stays
# readable.
# --------------------------------------------------------------------------- #
def _shrinkage_surface_and_feed_nodes(mesh):
    """Controller directive 1 (owner-adjudicated 2026-08-01): `surface_nodes`
    = unique nodes of ALL surface triangles (every physical-group tag the
    mesh carries -- "Cooling" union "Adiabatic" -- matching `screen_casting`'s
    own contract of "the whole free surface", used for sink/feeder
    placement). `feed_nodes` = unique nodes of "Adiabatic"-tagged triangles
    ONLY (riser/sleeve faces, per `src.shrinkage.is_fed`'s documented
    convention) -- empty when the group is absent or empty (a bare sand
    casting with no risers, conservatively "everything unfed")."""
    tri_arrays = [tri for tri in mesh.tris.values() if tri.size]
    surface_nodes = (np.unique(np.concatenate(tri_arrays, axis=0).reshape(-1))
                     if tri_arrays else np.zeros(0, dtype=np.int64))
    adiab_tri = mesh.tris.get("Adiabatic")
    feed_nodes = (np.unique(adiab_tri.reshape(-1))
                 if adiab_tri is not None and adiab_tri.size else np.zeros(0, dtype=np.int64))
    return surface_nodes.astype(np.int64), feed_nodes.astype(np.int64)


def _shrinkage_result_for_return(screen: dict) -> dict:
    """Directive 6: the `screen_casting` result MINUS its bulky per-node
    arrays (`niyama`/`porosity_risk` dropped), plus summary counts +
    `flagged_frac` -- everything else (`macro_shrinkage`/`top_sinks`/
    `feeders`/`material`/`niyama_source`/`per_orientation`) is kept as-is."""
    result = {k: v for k, v in screen.items() if k not in ("niyama", "porosity_risk")}
    risk = screen["porosity_risk"]
    result["flagged_frac"] = float(np.mean(risk)) if risk.size else 0.0
    result["n_macro_regions"] = len(screen["macro_shrinkage"])
    result["n_sinks"] = len(screen["top_sinks"])
    result["n_feeders"] = len(screen["feeders"])
    return result


def _shrinkage_extra_point_data(mesh, screen: dict) -> dict:
    """The 3 VTU point fields directive 4 asks for: `niyama` (float),
    `porosity_risk` (uint8 0/1), and `macro_region_id` (-1 = no macro
    region, else that region's own `region_id`)."""
    n_nodes = mesh.node_xyz.shape[0]
    macro_region_id = np.full(n_nodes, -1, dtype=np.int64)
    for region in screen["macro_shrinkage"]:
        macro_region_id[np.asarray(region["nodes"], dtype=np.int64)] = region["region_id"]
    return {
        "niyama": np.asarray(screen["niyama"], dtype=np.float32),
        "porosity_risk": np.asarray(screen["porosity_risk"], dtype=np.uint8),
        "macro_region_id": macro_region_id,
    }


def _write_shrinkage_summary_block(f, shrinkage) -> None:
    """Directive 5's exact skeleton: a `shrinkage screening:` section with
    niyama_source, flagged-node %, N macro regions (worst severity +
    centroid), N sinks, and feeder recommendations (location + min
    modulus) -- or a single `shrinkage screening: disabled` line.

    Only called when screening was ATTEMPTED (`screen_shrinkage=True`) but
    either produced a result (`shrinkage` is the trimmed dict) or was
    internally skipped (`shrinkage` is `None`, e.g. no surface nodes to
    place a feeder against) -- `screen_shrinkage=False` (the explicit
    opt-out) never calls this at all, so that path stays byte-identical to
    the pre-Task-4 summary (see `predict_casting`'s own docstring)."""
    if shrinkage is None:
        f.write("shrinkage screening: disabled\n")
        return
    f.write("shrinkage screening:\n")
    f.write(f"  niyama_source: {shrinkage['niyama_source']}\n")
    flagged_line = f"  flagged nodes: {shrinkage['flagged_frac'] * 100.0:.1f}%"
    if shrinkage["niyama_source"] == "proxy":
        # Review round 1, finding 4: the proxy path's risk mask is
        # `porosity_risk(mode="percentile")` -- ALWAYS ~the bottom decile by
        # construction (src.shrinkage's default `pct=10.0`), a within-part
        # relative ranking, not an absolute risk level. Without this note a
        # reader could mistake "10.0%" for a measured severity rather than
        # the fixed cutoff it structurally always is on this path.
        flagged_line += " (bottom decile by construction)"
    f.write(flagged_line + "\n")
    macro = shrinkage["macro_shrinkage"]
    if macro:
        # Review finding C1: regions with fewer than 4 nodes
        # (`region["sub_resolution"]`, set by `src.shrinkage.hot_regions`)
        # are still listed here (their volume/severity stay meaningful) but
        # reported SEPARATELY from the well-resolved count -- they never
        # get a feeder recommendation below (their V/A modulus diverges
        # without bound at this resolution; see `feeder_recommendations`'s
        # docstring), so folding them into one undifferentiated N would
        # overstate how many of the N regions actually have a usable
        # feeder placement.
        n_sub_resolution = sum(1 for r in macro if r.get("sub_resolution"))
        region_count = str(len(macro))
        if n_sub_resolution:
            region_count += f" (+{n_sub_resolution} sub-resolution)"
        worst = max(macro, key=lambda r: r["severity"])
        c = worst["centroid"]
        f.write(f"  macro shrinkage regions: {region_count} "
               f"(worst severity={worst['severity']:.4g} mm^3 @ "
               f"({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}))\n")
    else:
        f.write("  macro shrinkage regions: 0\n")
    f.write(f"  sinks: {len(shrinkage['top_sinks'])}\n")
    feeders = shrinkage["feeders"]
    if feeders:
        f.write("  feeder recommendations:\n")
        for feeder in feeders:
            loc = feeder["location"]
            f.write(f"    - location=({loc[0]:.1f},{loc[1]:.1f},{loc[2]:.1f}) mm  "
                   f"min_modulus_mm={feeder['min_modulus_mm']:.4f}\n")
    else:
        f.write("  feeder recommendations: none\n")
    if shrinkage.get("alloy_family_assumed"):
        f.write(f"  alloy_family_assumed: {shrinkage['material']['alloy_family']} "
               f"(alloy grade name not recognized -- conservative fallback)\n")


def _parse_gravity(s: str) -> tuple:
    """CLI `--gravity` parser: `"x,y,z"` -> `(float(x), float(y), float(z))`.
    A small free function so it's unit-testable without going through
    `argparse`/`sys.argv` -- and (review round 1, finding 5) passed directly
    as `argparse`'s own `type=` callable, so a malformed `--gravity` value
    surfaces as argparse's normal `ValueError`-driven usage error (exit
    code 2, "argument --gravity: invalid _parse_gravity value: ...") rather
    than an uncaught traceback deep inside `predict_casting`.

    Rejects the zero vector loudly (`ValueError`): `screen_casting`'s
    `top_sinks`/`feeder_recommendations` normalize `gravity_vec` by its own
    norm and would divide by zero silently otherwise (`up = up / norm`
    guarded only by `norm > 0`, which degrades to a NEVER-normalized,
    physically meaningless direction rather than raising) -- a caller
    typo'ing `0,0,0` deserves a clear error, not a silently-wrong feeder
    placement."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--gravity must be 'x,y,z' (3 comma-separated numbers), got {s!r}")
    vec = tuple(float(p) for p in parts)
    if all(v == 0.0 for v in vec):
        raise ValueError(
            f"--gravity must not be the zero vector (got {s!r}) -- gravity_vec is "
            f"normalized by its own norm for top-sink/feeder placement, and the zero "
            f"vector has no direction")
    return vec


def load_hybrid(gnn_ckpt, hgb_path, device=None,
                baseline_features_dir="data/features_v2", hgb_std_path=None):
    """Load the GNN + (optional) HGB baseline (+ optional GBM std-head) for
    `predict_field`.

    Film (v4) checkpoints load via `src.evaluate_ft2b.load_ft2b_gnn` and skip
    `baseline_norm` entirely (v4's HGB is the v2-total scale GBM, standardized
    by `src.gbm_ft2b` itself -- there is no separate per-node baseline-feature
    norm to load). Non-film checkpoints are loaded exactly as before
    (`src.evaluate_gnn.load_gnn`) -- byte-identical to the pre-v4 code.

    `hgb_std_path` (variance-recomposition Task 3): an optional path to a
    `joblib`-dumped GBM std-head model (`src.gbm_ft2b.refit_gbm_std`'s
    output), loaded exactly like `hgb_path` (`joblib.load` if given, else
    `None`). `None` (the default -- unchanged by Task 6, see below) means
    the returned `hgb_std` is `None` and every other return value / code
    path is byte-identical to before this argument existed.

    `hgb_std_path` may also be (variance-recomposition Task 6):
    - the module-level `_USE_DEFAULT` sentinel: resolved HERE, using the
      GNN checkpoint's own `cfg["film"]` flag (already being determined
      below to pick the loader) -- `DEFAULT_HGB_STD` for a film-capable
      checkpoint, `None` for a non-film one. This is what lets
      `predict_casting`'s own sentinel default reach the new production
      std head with zero flags, while leaving a non-film (SP1) `--gnn`/
      `--hgb` rollback untouched: no attempt is made to load
      `DEFAULT_HGB_STD` at all in that case, so a missing/irrelevant v4-only
      artifact never breaks a non-film call.
    - the case-insensitive string `"none"`: always resolved to `None`,
      regardless of film-ness -- the explicit rollback switch to the
      quantile-anchor recomposition (usable on any film checkpoint,
      including the new default one).
    An explicit path (or an already-loaded model object, via `hgb_path`'s
    own convention) is still loaded/used unconditionally either way, exactly
    as before Task 6 -- `load_hybrid` never judges whether a checkpoint/
    std-head pairing makes sense; that is `predict_casting`'s job (it raises
    if the resulting `hgb_std` is non-`None` for a non-film checkpoint).

    Returns a 6-tuple (was 5, before this argument existed): `(gnn, hgb,
    norm, cfg, baseline_norm, hgb_std)` -- every existing call site was
    updated to the new arity when this argument was added; any NEW call site
    must unpack all 6.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gnn_ckpt, hgb_path = resolve_artifact(gnn_ckpt), resolve_artifact(hgb_path)
    hgb = load_gbm(hgb_path)
    state = load_gnn_any(gnn_ckpt, device)
    is_film = state.get("cfg", {}).get("film", False)

    if hgb_std_path is _USE_DEFAULT:
        hgb_std_path = DEFAULT_HGB_STD if is_film else None
    elif isinstance(hgb_std_path, str) and hgb_std_path.strip().lower() == "none":
        hgb_std_path = None
    hgb_std_path = resolve_artifact(hgb_std_path)
    try:
        hgb_std = load_gbm(hgb_std_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"GBM std head not found: {hgb_std_path!r}. Since the 2026-07-31 v5 "
            f"promotion the std artifact ({DEFAULT_HGB_STD}) is a PRODUCTION "
            f"requirement of the default affine recomposition -- provision it "
            f"alongside the GNN/GBM artifacts, or pass --hgb-std none (API: "
            f"hgb_std_path=None) to fall back to the anchor recomposition.") from e

    if is_film:
        gnn, cfg, norm = load_ft2b_gnn(state, device)
        return gnn, hgb, norm, cfg, None, hgb_std
    gnn, cfg, norm = load_gnn(gnn_ckpt, device)
    from src.dataload import load_norm
    baseline_norm = load_norm(baseline_features_dir) if hgb is not None else None
    return gnn, hgb, norm, cfg, baseline_norm, hgb_std


import os
import argparse

from src.step_mesh import mesh_step
from src.msh_parse import parse_msh
from src import ood as _ood
from src.mesh_cache import get_or_mesh
from tools.doe.prep_parts import DEFAULT_TARGET_NODES


def predict_casting(step_path, out_dir="runs/castings", elem_size=None,
                    mesh_mode="prep",
                    gnn_ckpt=DEFAULT_GNN_CKPT,
                    hgb_path=DEFAULT_HGB,
                    ood_json="data/features_v2/ood_ranges.json", device=None,
                    tta=0, alloy=None, mold="green_sand", props_json=None,
                    target_nodes: int = DEFAULT_TARGET_NODES,
                    hgb_std_path=_USE_DEFAULT,
                    lambda_shrink: float = 1.0, w_total: float = DEFAULT_W_TOTAL,
                    w_std: float = DEFAULT_W_STD, screen_shrinkage: bool = True,
                    gravity: tuple = (0, 0, -1)) -> dict:
    """`mesh_mode`: "prep" (default) routes meshing through
    `tools.doe.prep_parts.heal_and_mesh` -- the SAME largest-solid + OCC
    heal + density-searched pipeline used to prep the training/dataset-D
    parts, so real-part inference and training use one honest meshing
    stack. "raw" keeps the old `src.step_mesh.mesh_step` single-pass
    uniform-density path (escape hatch). An explicit `elem_size` always
    implies "raw" regardless of `mesh_mode` -- density-searching towards a
    node-count target and pinning an exact element size are mutually
    exclusive requests, and the explicit size wins. "prep" mode meshes via
    `src.mesh_cache.get_or_mesh`, so a byte-identical STEP file (same
    `target_nodes`) is served from `runs/mesh_cache/` instead of re-meshed.

    `target_nodes` (app Phase A, 2026-08-01): the node-count target the
    "prep" density search aims at, forwarded to `src.mesh_cache.get_or_mesh`
    (and thus part of the mesh-cache key, so two densities of the same STEP
    cache separately). Defaults to `DEFAULT_TARGET_NODES` (8500), which is
    byte-identical to the previous hardcoded behaviour. The trained density
    band is ~7.5-10k nodes (DECISIONS-2026-07-22 §9); values outside it put
    the mesh outside what the model saw, and callers exposing this to a user
    should bound it there. Passing a non-default value together with an
    explicit `elem_size` raises `ValueError` rather than being ignored --
    `elem_size` forces "raw", which never runs the density search at all.

    `hgb_std_path` (variance-recomposition Task 3; default changed in Task 6;
    recomposition swapped to `hybrid_field_soft` by soft-anchor Task 3): an
    optional path to a `joblib`-dumped GBM std-head model
    (`src.gbm_ft2b.refit_gbm_std`/`refit_gbm_std_v2`'s output, 16 or 17
    features -- `predict_field` width-dispatches off the loaded model's own
    `n_features_in_`), forwarded to `load_hybrid`/`predict_field`. Defaults
    to the module-level `_USE_DEFAULT` sentinel (not `None`, since Task 6):
    `load_hybrid` resolves this to `DEFAULT_HGB_STD` for a film-capable
    checkpoint (the production call, zero extra flags, now recomposes via
    `hybrid_field_soft`) or to `None` for a non-film one (the SP1 rollback
    via `--gnn`/`--hgb` is unaffected -- no attempt is made to load
    `DEFAULT_HGB_STD` for a non-film checkpoint at all). Pass `None`
    explicitly, or the case-insensitive string `"none"` (the CLI-friendly
    form), to force the quantile-anchor recomposition on ANY checkpoint,
    including the new default film one -- this is byte-identical to the
    pre-Task-3 code. Passing an explicit path (or an already-loaded model
    object) still requires a film-capable checkpoint (SP1 has no
    scale-head std target) -- raises `ValueError`
    otherwise, BEFORE meshing. When the soft-anchor recomposition is active,
    the summary file gains a `soft recomposition: a=..., m=...,
    target_unreachable=..., clip_frac=...` line (`target_unreachable`
    surfaces whether the underlying hard (affine) two-target problem the
    soft solve seeds from was itself unreachable at some `a` -- see
    `src.hybrid.hybrid_field_soft`), further appending a trailing
    `total_dev=+x.x%` field (final review I-1, 2026-08-01) whenever
    `abs(info["total_deviation_frac"]) > 0.001` -- signed, one decimal
    (e.g. `total_dev=+4.7%`) -- surfacing the soft anchor's defining
    trade-off (letting the total deviate from the GBM total to buy a closer
    spread match) in the human-readable summary, not just the return dict's
    `recomposition_info`; omitted entirely below that threshold so a healthy
    part's summary stays clean. Otherwise (no active std head) it gains a
    `recomposition: anchor ...` line (film checkpoints only).

    `lambda_shrink` (variance-recomposition amendment 2, CLI `--affine-
    lambda`): forwarded straight to `predict_field`/`hybrid_field_soft`
    (ignored unless `hgb_std_path` is also given). `1.0` (the default) is
    byte-identical to before this argument existed. When active AND
    `lambda_shrink != 1.0`, the summary's recomposition line gains a
    trailing `lambda_shrink=...` field.

    `w_total`/`w_std` (soft-anchor Task 3, CLI `--w-total`/`--w-std`; defaults
    changed to `DEFAULT_W_TOTAL`/`DEFAULT_W_STD` (`1.0`/`4.0`) by the
    soft-anchor Task 6 promotion, 2026-07-31 -- the e_val-sweep winner
    recorded in the spec addendum): forwarded straight to
    `predict_field`/`hybrid_field_soft`'s own weights (ignored unless
    `hgb_std_path` is also given). When either differs from `1.0`/`1.0` (NOT
    from the new defaults -- this compares against `hybrid_field_soft`'s own
    equal-weighting baseline, so the new production default line still
    prints its weights), the summary's recomposition line gains a trailing
    `w_total=..., w_std=...` field.

    `screen_shrinkage` (SP2 shrinkage Task 4, CLI `--no-shrinkage`; learned
    source added by learned-heads Task 3): runs `src.shrinkage.screen_casting`
    on the recomposed hybrid field (`t_s`) AFTER recomposition, BEFORE the
    VTU write -- Niyama (the time-only proxy, UNLESS `gnn_ckpt` is a
    deriv_heads-capable checkpoint, in which case `predict_field` has already
    populated `feat["gradT"]`/`feat["dTdt"]` with the checkpoint's own learned
    per-node predictions, routing `screen_casting` through its exact branch
    instead; the returned `shrinkage["niyama_source"]` is then relabeled
    `"learned"` here -- NOT `screen_casting`'s own `"exact"`, which is
    reserved for a real cached solver field, e.g. Gate C's per-row
    comparison), macro-shrinkage region detection, top-surface sinks, and
    feeder recommendations. `surface_nodes`/`feed_nodes` come straight off
    the parsed mesh's physical-group triangles (`_shrinkage_surface_and_
    feed_nodes` -- ALL surface tris union for `surface_nodes`, "Adiabatic"
    tris only for `feed_nodes`, empty for a bare sand casting with no
    risers); `TL`/`TS` come from the already-resolved `props`; `alloy_family`
    comes from `src.inference_props.resolve_alloy_family` (a name-prefix map
    onto `src.shrinkage.SHRINKAGE_PCT`'s family keys -- an unrecognized grade
    name, or any `--props` custom-JSON override, falls back loudly to
    `"steel"` and is recorded as `alloy_family_assumed` in the result rather
    than crashing or silently mis-scaling severities). `True` (the default)
    adds `niyama`/`porosity_risk`/`macro_region_id` VTU point fields, a
    `shrinkage screening:` block in the summary, and a `"shrinkage"` key in
    the return dict (the `screen_casting` result minus its bulky per-node
    `niyama`/`porosity_risk` arrays, plus `flagged_frac`/`n_macro_regions`/
    `n_sinks`/`n_feeders` counts). The shrinkage dict carries
    `"niyama_confidence": "full"` (this release supports the all-sand regime
    only, where the proxy passed its validation gate). `False` skips all of
    that entirely -- `"shrinkage"` is `None` and the VTU/summary carry no
    extra fields (the `shrinkage screening: disabled` line only appears when
    screening was attempted but internally skipped, e.g. a mesh with no
    surface nodes to place a feeder against, never for this explicit
    opt-out).

    `gravity` (CLI `--gravity x,y,z`, parsed by `_parse_gravity`): forwarded
    straight to `screen_casting`'s `gravity_vec` (top-sink/feeder placement
    direction; conduction/Niyama/macro-region *detection* don't depend on
    it). `(0, 0, -1)` (the default) is the same "gravity along -z" convention
    used elsewhere in this module. Ignored entirely when `screen_shrinkage`
    is `False`.

    Returns a dict that (soft-anchor Task 4, v5 review M-1) always carries
    `"recomposition"` and `"recomposition_info"` alongside the existing keys:
    `"recomposition"` is `"soft"` (the affine two-target solve via
    `hybrid_field_soft` is active -- `hgb_std_path` resolved to a real std
    head), `"anchor"` (a film/v4 checkpoint recomposed via the plain
    quantile-anchor -- `hgb_std_path` resolved to `None`, e.g. via `"none"`),
    or `"sp1"` (a non-film checkpoint -- no recomposition-mode concept
    applies at all). `"recomposition_info"` is `hybrid_field_soft`'s own
    info dict (`a`/`m`/`target_unreachable`/`clip_frac`/
    `total_deviation_frac`/... -- see `predict_field`'s docstring) when
    `"recomposition"` is `"soft"`, else `None` -- there is no equivalent info
    dict on the anchor or SP1 paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    name = osp.splitext(osp.basename(step_path))[0]
    out_msh = osp.join(out_dir, f"{name}.msh")

    # Resolve properties early (cheap, no mesh dependency) so typos/bad alloys
    # fail fast, before potentially lengthy meshing.
    props = resolve_props(alloy=alloy, mold=mold, props_json=props_json)

    effective_mode = "raw" if elem_size is not None else mesh_mode
    if target_nodes != DEFAULT_TARGET_NODES and effective_mode != "prep":
        raise ValueError(
            f"predict_casting: `target_nodes` ({target_nodes}) applies to the "
            f"prep/get_or_mesh density search only, but an explicit "
            f"elem_size={elem_size!r} forces mesh_mode='raw' -- pinning an "
            f"element size and searching towards a node count are mutually "
            f"exclusive requests. Drop one.")
    gnn_ckpt, hgb_path = resolve_artifact(gnn_ckpt), resolve_artifact(hgb_path)
    gnn, hgb, norm, cfg, baseline_norm, hgb_std = load_hybrid(
        gnn_ckpt, hgb_path, device, hgb_std_path=hgb_std_path)
    # Provenance label for the summary txt: the RESOLVED std artifact (the
    # v5.1 stack is defined by it + the weights), mirroring load_hybrid's own
    # sentinel/"none" resolution without re-running it.
    if hgb_std is None:
        hgb_std_path_label = "none"
    elif hgb_std_path is _USE_DEFAULT:
        hgb_std_path_label = resolve_artifact(DEFAULT_HGB_STD)
    else:
        hgb_std_path_label = str(hgb_std_path)
    if hgb_std is not None and not cfg.get("film", False):
        raise ValueError(
            "predict_casting: --hgb-std requires the film-conditioned v4 "
            "stack; the selected checkpoint is not film-capable -- rerun "
            "without --hgb-std or with the v4 checkpoints")

    if effective_mode == "raw":
        raw_info = mesh_step(step_path, out_msh, elem_size)
        mesh = parse_msh(out_msh)
        minfo = {"n_nodes": raw_info["n_nodes"], "n_tets": raw_info["n_tets"],
                "elem_size": raw_info["elem_size"], "bbox_mm": raw_info["bbox_mm"]}
    elif effective_mode == "prep":
        prep_info = get_or_mesh(step_path, out_msh, target_nodes=target_nodes)
        if not prep_info.get("ok", False):
            raise RuntimeError(
                f"predict_casting: prep meshing pipeline (get_or_mesh) failed for "
                f"{step_path!r}: {prep_info.get('notes', '(no notes)')}")
        mesh = parse_msh(out_msh)
        bbox_mm = tuple((mesh.node_xyz.max(0) - mesh.node_xyz.min(0)).tolist())
        cache = prep_info.get("cache", "unknown")
        minfo = {"n_nodes": prep_info["n_nodes"], "n_tets": int(mesh.tets.shape[0]),
                "elem_size": None, "bbox_mm": bbox_mm, "cache": cache}
    else:
        raise ValueError(f"predict_casting: mesh_mode must be 'prep' or 'raw', got {mesh_mode!r}")

    seconds, feat = predict_field(mesh, gnn, hgb, norm, cfg=cfg, device=device, tta=tta,
                                  baseline_norm=baseline_norm, props=props,
                                  hgb_std=hgb_std, lambda_shrink=lambda_shrink,
                                  w_total=w_total, w_std=w_std)

    warnings = []
    ood_json = resolve_artifact(ood_json)
    if osp.exists(ood_json):
        ranges = _ood.load_ranges(ood_json)
        X = feat["X"][:, :N_BASELINE_FEATURES]
        vol = float(X[0, FEATURE_NAMES.index("volume_mm3")])
        gcmax = float(X[:, FEATURE_NAMES.index("geodesic_d_cool")].max())
        diag = float(np.linalg.norm(mesh.node_xyz.max(0) - mesh.node_xyz.min(0)))
        warnings = _ood.ood_flags(vol, diag, gcmax, ranges)

    # SP2 shrinkage Task 4: screen the recomposed hybrid field for shrinkage-
    # porosity risk, AFTER recomposition, BEFORE the VTU write -- see
    # predict_casting's own docstring for the full contract.
    shrinkage = None
    shrinkage_extra_point_data = None
    if screen_shrinkage:
        surface_nodes, feed_nodes = _shrinkage_surface_and_feed_nodes(mesh)
        if surface_nodes.size == 0:
            logger.warning(
                "predict_casting: no surface nodes found on the meshed part -- "
                "shrinkage screening skipped (a feeder location requires a free "
                "surface)")
        else:
            alloy_family, alloy_family_assumed = resolve_alloy_family(alloy, props_json)
            # Learned-heads Task 3: `feat["gradT"]`/`feat["dTdt"]` are only
            # present when `predict_field` ran a deriv_heads-capable (film)
            # checkpoint (see that function's docstring) -- `None`/`None` for
            # every other checkpoint, which routes `screen_casting` through
            # its own time-only proxy branch exactly as before this feature
            # existed.
            learned_gradT = feat.get("gradT")
            learned_dTdt = feat.get("dTdt")
            shrinkage_full = screen_casting(
                pos=mesh.node_xyz, tets=mesh.tets, surface_nodes=surface_nodes,
                t_s=seconds, TL=props["TL"], TS=props["TS"], alloy_family=alloy_family,
                gravity_vec=gravity, feed_nodes=feed_nodes,
                gradT=learned_gradT, dTdt=learned_dTdt)
            shrinkage = _shrinkage_result_for_return(shrinkage_full)
            if learned_gradT is not None and learned_dTdt is not None:
                # `screen_casting` itself only distinguishes "exact" (real
                # cached solver fields, e.g. Gate C's per-row comparison)
                # from "proxy" -- it has no notion of "predicted by a
                # checkpoint at inference time". Relabel HERE, at the
                # inference call site, rather than teaching `screen_casting`
                # a third source string: that keeps `src.shrinkage`'s own
                # tests (which pass real cached gradT/dTdt and expect
                # "exact") untouched, while giving inference-time callers an
                # honest label distinguishing a MODEL PREDICTION from a true
                # solver field.
                shrinkage["niyama_source"] = "learned"
            if alloy_family_assumed:
                shrinkage["alloy_family_assumed"] = True
            # This release covers the all-sand regime only, where the
            # proxy Niyama passed its validation gate.
            shrinkage["niyama_confidence"] = "full"
            shrinkage_extra_point_data = _shrinkage_extra_point_data(mesh, shrinkage_full)

    vtu = osp.join(out_dir, f"{name}.vtu")
    write_vtu(vtu, mesh.node_xyz, mesh.tets, seconds, extra_point_data=shrinkage_extra_point_data)

    # hot-spot summary: top-5 latest-solidifying nodes
    order = np.argsort(seconds)[::-1][:5]
    hot_xyz = mesh.node_xyz[order]
    total = float(seconds.max())
    summary = osp.join(out_dir, f"{name}_summary.txt")
    alloy_label = alloy if alloy is not None else DEFAULT_ALLOY
    if props_json is not None:
        alloy_label = f"custom ({props_json})"
    mold_label = mold if props_json is None else "n/a (props_json)"
    elem_size_label = (f"{minfo['elem_size']:.2f} mm" if minfo["elem_size"] is not None
                       else "n/a (density-searched)")
    cache_label = minfo.get("cache") if effective_mode == "prep" else "n/a (raw mode)"
    with open(summary, "w") as f:
        f.write(f"casting: {name}\n")
        f.write(f"total solidification time: {total:.1f} s\n")
        f.write(f"alloy: {alloy_label}  mold: {mold_label}  gnn: {gnn_ckpt}  hgb: {hgb_path}  "
                f"hgb_std: {hgb_std_path_label}\n")
        f.write(f"mesh_mode: {effective_mode}  cache: {cache_label}  nodes: {minfo['n_nodes']}  "
                f"tets: {minfo['n_tets']}  elem_size: {elem_size_label}  bbox_mm: "
                f"{tuple(round(v,1) for v in minfo['bbox_mm'])}\n")
        if feat.get("recomposition") == "soft":
            info = feat["recomposition_info"]
            line = (f"soft recomposition: a={info['a']:.4f}, m={info['m']:.4f}, "
                   f"target_unreachable={info['target_unreachable']}, "
                   f"clip_frac={info['clip_frac']:.4f}")
            if info.get("lambda_shrink", 1.0) != 1.0:
                line += f", lambda_shrink={info['lambda_shrink']:.4f}"
            if w_total != 1.0 or w_std != 1.0:
                line += f", w_total={w_total:.4f}, w_std={w_std:.4f}"
            total_dev = info.get("total_deviation_frac")
            if total_dev is not None and abs(total_dev) > 0.001:
                line += f", total_dev={total_dev * 100.0:+.1f}%"
            f.write(line + "\n")
        elif cfg.get("film", False):
            # variance-recomposition Task 6: the film path without an active
            # std head (hgb_std_path="none", or a film checkpoint used
            # without one) recomposes via the quantile-anchor -- state that
            # explicitly, mirroring the affine line above, rather than
            # leaving the recomposition mode unstated.
            f.write("recomposition: anchor (quantile, q=0.995)\n")
        if screen_shrinkage:
            _write_shrinkage_summary_block(f, shrinkage)
        f.write("top hot spots (x,y,z mm : t_solidus s):\n")
        for xyz_i, i in zip(hot_xyz, order):
            f.write(f"  ({xyz_i[0]:.1f},{xyz_i[1]:.1f},{xyz_i[2]:.1f}) : {seconds[i]:.1f}\n")
        if warnings:
            f.write("WARNINGS:\n")
            for w in warnings:
                f.write(f"  - {w}\n")

    # soft-anchor Task 4 (v5 review M-1): surface the recomposition mode +
    # its info dict as structured data, mirroring the summary text above --
    # "soft" (hybrid_field_soft active, feat carries its info dict), "anchor"
    # (film checkpoint, quantile-anchor recomposition -- no info dict exists
    # on this path), or "sp1" (non-film checkpoint -- no recomposition
    # concept at all).
    if feat.get("recomposition") == "soft":
        recomposition = "soft"
        recomposition_info = feat["recomposition_info"]
    elif cfg.get("film", False):
        recomposition = "anchor"
        recomposition_info = None
    else:
        recomposition = "sp1"
        recomposition_info = None

    result = {"name": name, "vtu": vtu, "summary": summary, "total_time_s": total,
              "mesh_mode": effective_mode, "n_nodes": minfo["n_nodes"],
              "elem_size": minfo["elem_size"], "ood_warnings": warnings,
              "hotspot_xyz": hot_xyz,
              "recomposition": recomposition,
              "recomposition_info": recomposition_info,
              "shrinkage": shrinkage}
    if "cache" in minfo:
        result["cache"] = minfo["cache"]
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step_path")
    ap.add_argument("--out-dir", default="runs/castings")
    ap.add_argument("--elem-size", type=float, default=None,
                    help="explicit uniform element size (mm) -- implies --mesh-mode raw, "
                         "regardless of --mesh-mode")
    ap.add_argument("--mesh-mode", choices=["prep", "raw"], default="prep",
                    help="'prep' (default): route meshing through the same largest-solid + "
                         "OCC heal + density-searched pipeline used to prep training/dataset-D "
                         "parts (tools.doe.prep_parts.heal_and_mesh), served through the "
                         "on-disk mesh cache (runs/mesh_cache/) so a repeated byte-identical "
                         "STEP file is not re-meshed. 'raw': the old single-pass "
                         "uniform-density src.step_mesh.mesh_step path (escape hatch). Ignored "
                         "(forced to 'raw') when --elem-size is given.")
    ap.add_argument("--gnn", dest="gnn_ckpt", default=DEFAULT_GNN_CKPT,
                    help="GNN checkpoint path (SP1 or a film/v4 checkpoint)")
    ap.add_argument("--hgb", dest="hgb_path", default=DEFAULT_HGB,
                    help="HGB baseline path (SP1 per-node baseline or the v4 v2-total GBM)")
    ap.add_argument("--hgb-std", dest="hgb_std_path", default=_USE_DEFAULT,
                    help="path to an optional GBM std-head model "
                         "(src.gbm_ft2b.refit_gbm_std's output); when active, recomposition "
                         "uses the affine (variance-preserving) map instead of the "
                         "quantile-anchor -- only consulted for film/v4 checkpoints. Default: "
                         f"{DEFAULT_HGB_STD} for a film/v4 checkpoint (auto-disabled, no extra "
                         "flag needed, for a non-film/SP1 checkpoint e.g. via --gnn/--hgb "
                         "rollback); pass 'none' (case-insensitive) to force the old "
                         "quantile-anchor recomposition on a film/v4 stack too")
    ap.add_argument("--affine-lambda", dest="lambda_shrink", type=float, default=1.0,
                    help="geometric-interpolation shrinkage of the affine spread target "
                         "(variance-recomposition amendment 2): 1.0 (default) fully trusts "
                         "--hgb-std's predicted spread; 0.0 targets the GNN's own spread "
                         "(no rescaling); only used together with --hgb-std on film/v4 "
                         "checkpoints")
    ap.add_argument("--w-total", dest="w_total", type=float, default=DEFAULT_W_TOTAL,
                    help="soft-anchor weight on the total-time residual in "
                         "src.hybrid.hybrid_field_soft's objective; default "
                         f"{DEFAULT_W_TOTAL} (the e_val-sweep winner, soft-anchor Task 6 "
                         "promotion, 2026-07-31) -- only used together with --hgb-std")
    ap.add_argument("--w-std", dest="w_std", type=float, default=DEFAULT_W_STD,
                    help="soft-anchor weight on the spread residual in "
                         "src.hybrid.hybrid_field_soft's objective; default "
                         f"{DEFAULT_W_STD} (the e_val-sweep winner, soft-anchor Task 6 "
                         "promotion, 2026-07-31) -- only used together with --hgb-std")
    ap.add_argument("--alloy", default=None,
                    help=f"named alloy grade (default: {DEFAULT_ALLOY}); only used by film/v4 checkpoints")
    ap.add_argument("--mold", default="green_sand",
                    help="named mold system; only used by film/v4 checkpoints")
    ap.add_argument("--props", dest="props_json", default=None,
                    help="path to a JSON file overriding alloy/mold with raw material properties")
    ap.add_argument("--no-shrinkage", dest="screen_shrinkage", action="store_false",
                    help="disable shrinkage-porosity screening (SP2): by default, "
                         "predict_casting runs src.shrinkage.screen_casting on the "
                         "recomposed field and adds niyama/porosity_risk/macro_region_id "
                         "VTU point fields, a 'shrinkage screening:' summary block, and "
                         "a 'shrinkage' return-dict key. This flag skips all of that, "
                         "byte-identical to the pre-shrinkage-screening contract.")
    ap.add_argument("--gravity", dest="gravity", type=_parse_gravity, default=(0, 0, -1),
                    help="gravity direction as 'x,y,z' (default '0,0,-1'), forwarded to "
                         "screen_casting's top-sink/feeder placement; ignored with "
                         "--no-shrinkage. Parsed by _parse_gravity as argparse's own "
                         "type= callable (review round 1, finding 5), so a malformed "
                         "value (wrong arity, non-numeric, or the zero vector) is a "
                         "normal argparse usage error, not a traceback.")
    a = ap.parse_args()
    r = predict_casting(a.step_path, out_dir=a.out_dir, elem_size=a.elem_size,
                        mesh_mode=a.mesh_mode,
                        gnn_ckpt=a.gnn_ckpt, hgb_path=a.hgb_path,
                        alloy=a.alloy, mold=a.mold, props_json=a.props_json,
                        hgb_std_path=a.hgb_std_path,
                        lambda_shrink=a.lambda_shrink, w_total=a.w_total, w_std=a.w_std,
                        screen_shrinkage=a.screen_shrinkage,
                        gravity=a.gravity)
    print(f"{r['name']}: total {r['total_time_s']:.1f}s  mesh_mode {r['mesh_mode']}  "
          f"nodes {r['n_nodes']}  -> {r['vtu']}")
    for w in r["ood_warnings"]:
        print("WARNING:", w)
