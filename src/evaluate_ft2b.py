"""FT2b Task 8: evaluation of the multi-alloy/multi-BC hybrid predictor.

Consumes a trained v4 checkpoint (FiLM, 15 node features, `g`-conditioned,
`node_in=15`) + the refit multi-alloy GBM scale-head (`src.gbm_ft2b`) and
reports, on `interp.test` AND `extrap.test`: per-`alloy_family` and per-
`bc_config` hybrid MAE (physical seconds), hot-spot top-10% overlap, and
`std_ratio = std(pred)/std(true)`; a no-regression comparison against the
SP1 hybrid on the all-sand subset; a Chvorinov-scaling sanity check; and a
variance-collapse flag.

=== y / y_raw / t_ref inversion chain (read before touching this file) ===
`src/build_dataset.py` (`_finalize_row_stage2`, `write_record_v4`) writes,
per v4 graph cache:
    y       = t              (PHYSICAL seconds, per-node solidification time)
    y_raw   = t               (same array, duplicate key for API clarity)
    y_nondim = t / t_ref       (t_ref = C_family * (V/A)^2, Chvorinov reference)
    t_ref   = C_family * (V/A)^2   (scalar, physical seconds)
`build_v4`'s norm_stats.npz computes `y_log_mean`/`y_log_std` from
`log1p(rec["y"])` i.e. `log1p(t)` -- log1p of the PHYSICAL target, exactly
the same convention `features_v2`/`features_v3` use. `SolidGraphDataset`
(`src/gnn_dataset.py`) trains on `standardize(log1p(d["y"]))` i.e.
standardized log1p(PHYSICAL seconds) -- NOT the non-dim `y_nondim` field.
Therefore the GNN's raw output, once unstandardized and expm1'd, is ALREADY
physical seconds -- the exact same inversion `src.evaluate_gnn.
predict_seconds_gnn` performs for SP1. No `t_ref` multiplication is needed
to get physical seconds out of the v4 GNN. `y_nondim`/`t_ref`/`g` are real,
separately-stored cache fields used elsewhere (the Chvorinov sanity check
below reads `t_ref`; `g` feeds FiLM) -- they are NOT an alternate target
space the model was trained on.

=== v4 hybrid recomposition: QUANTILE-anchor (q=0.995), not MEAN-anchor ===
`src.hybrid.hybrid_field(gnn, baseline)` (SP1's convention) re-anchors to the
baseline's MEAN: `pred = gnn - mean(gnn) + mean(baseline)`. That is correct
for SP1 because SP1's HGB baseline predicts a genuine PER-NODE value, so
`mean(baseline)` is a real per-graph mean estimate. The FT2b GBM
(`src.gbm_ft2b`) is different by design (Task 6): it predicts a single
per-graph SCALAR, `log1p(total/max solidification time)` -- see its module
docstring, "Target: per-graph `log1p(y_raw.max())`". Broadcasting that
scalar to every node and then MEAN-anchoring would force
`mean(pred) == gbm_total`, inflating every node by
`(true_max - true_mean)` -- confirmed in review to blow up MAE by ~60x on an
otherwise near-perfect GNN. `_max_anchor_hybrid` (an alias of
`src.hybrid.hybrid_field_max_anchor`, the single source -- see that
function's docstring) anchors to a high QUANTILE of the GNN field instead
(default `q=0.995`; `q=1.0` reproduces the original exact-MAX behavior
bit-for-bit): `pred_i = gnn_i - anchor + gbm_total` where
`anchor = np.quantile(gnn, q)`, which makes `max(pred) ~= gbm_total` (the
quantity the GBM actually predicts) while still preserving the GNN's field
shape (its deviations from that anchor). The anchor moved off the exact
max (Task 7) because it is fragile to a handful of outlier nodes -- a
promotion-gate smoke test found a few inflated GNN log-space predictions
on a real casting dragging ~90% of the field below the zero clip, since the
single max node alone set the whole graph's additive offset; anchoring to
the 99.5th percentile instead is robust to that. This function is used for
EVERY v4 hybrid composition in this module (`_evaluate_stems_v4`, and
therefore every per-family/per-bc metric, the v4 side of `no_regression`,
and `chvorinov`'s `pred_total`). The SP1 side of `no_regression`
(`_sp1_hybrid_predict_stem`) deliberately keeps `src.hybrid.hybrid_field`'s
MEAN-anchor unchanged -- it is SP1's own genuine per-node HGB baseline, not
a broadcast scalar, so the MEAN-anchor is correct there.
"""
from __future__ import annotations

import json
import os
import os.path as osp
from collections import defaultdict

import joblib
import numpy as np
import torch

from src import metrics
from src.baseline import predict_seconds
from src.dataload import load_norm, standardize
from src.evaluate_gnn import predict_seconds_gnn
from src.gbm_ft2b import (_load_manifest_rows, _row_features, predict_std_seconds,
                          std_feature_row)
from src.graph_build import (edge_attr, knn_edge_index, union_edge_attr,
                             union_edge_index)
from src.hybrid import hybrid_field, hybrid_field_soft
from src.hybrid import hybrid_field_max_anchor as _max_anchor_hybrid
from src.model_gnn import SolidGNN

# The HGB v2/SP1 baseline was trained on the 9 v2 features = first 9 columns
# of any newer feature version (ordering contract, SP1 spec §3). The SP1
# production GNN was trained on the 13 v3 features = first 13 columns of the
# v4 cache's X (`src/features.py`: "v4: v3's 13 columns UNCHANGED").
N_V2_FEATURES = 9
N_V3_FEATURES = 13

DEFAULT_OUT_PATH = "runs/ft2b_stage3/eval.json"
DEFAULT_SP1_CKPT = "runs/sp1_run3_final/best_gnn.pt"
DEFAULT_SP1_HGB = "runs/baseline_v2/hgb_A.joblib"
DEFAULT_BASELINE_FEATURES_DIR = "data/features_v2"


def hotspot_overlap(y_true, y_pred, xyz, graph_id, top_frac: float = 0.10,
                    tol_mm: float = 5.0) -> float:
    """Top-`top_frac` latest-solidifying-node overlap, within `tol_mm`.

    Thin, deliberately-not-reimplemented wrapper over
    `src.metrics.hotspot_accuracy` -- the README-documented hot-spot metric
    ("overlap of predicted vs. true top-10% latest-solidifying nodes, within
    a tolerance"). Exposed under this module's own name so callers of
    `evaluate_ft2b` don't need to reach into `src.metrics` directly.
    """
    return metrics.hotspot_accuracy(y_true, y_pred, xyz, graph_id,
                                    top_frac=top_frac, tol_mm=tol_mm)


# --------------------------------------------------------------------------- #
# GNN loading / prediction (v4-aware: film + g)
# --------------------------------------------------------------------------- #
def load_ft2b_gnn(ckpt, device):
    """Load a `{model, cfg, norm}` checkpoint into a `SolidGNN` (film-aware).

    `ckpt` may be a path (loaded via `torch.load`) or an already-loaded
    checkpoint dict -- convenient for tests that build a ckpt in-process.

    Learned-heads Task 3: `cfg.get("deriv_heads")` truthy builds the model
    with `deriv_heads=True` (the `SolidGNN` gains `grad_head`/`dtdt_head`,
    see `src/model_gnn.py`) before the strict `load_state_dict` below -- a
    heads checkpoint's saved `state["model"]` already carries the two heads'
    weights (`src/train_heads.py` saves the full `model.state_dict()` of a
    `deriv_heads=True` model), so this is still a plain strict load, not
    `attach_deriv_heads` (that helper is only for loading a HEADLESS
    state_dict into a heads-shaped model, e.g. inside the trainer). A
    headless checkpoint (`cfg.get("deriv_heads")` absent/falsy, the default)
    builds `deriv_heads=False` exactly as before -- byte-identical path.
    """
    state = torch.load(ckpt, map_location=device, weights_only=False) \
        if isinstance(ckpt, (str, os.PathLike)) else ckpt
    cfg, norm = state["cfg"], state["norm"]
    model = SolidGNN(node_in=cfg.get("node_in", 15), edge_in=cfg.get("edge_in", 4),
                     hidden=cfg["hidden"], n_blocks=cfg["n_blocks"],
                     two_head=cfg.get("two_head", False),
                     film=cfg.get("film", False), g_dim=cfg.get("g_dim", 5),
                     deriv_heads=cfg.get("deriv_heads", False)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg, norm


def _build_edges(pos, cfg, mesh_edges=None):
    if cfg.get("graph", "knn") == "union":
        if mesh_edges is None:
            raise ValueError("checkpoint expects a union graph but mesh_edges is missing")
        ei, is_mesh = union_edge_index(pos, mesh_edges, k=cfg["k"])
        return ei, union_edge_attr(pos, ei, is_mesh)
    ei = knn_edge_index(pos, k=cfg["k"])
    return ei, edge_attr(pos, ei)


def _run_model(model, x, ei, ea, zeros, g, device):
    """Shared forward-pass plumbing for `predict_graph_seconds`/
    `predict_graph_derivs` (learned-heads Task 3): moves the already-built
    tensors to `device` and calls the model once. Returns whatever tuple the
    model itself returns -- 2-tuple `(y_nodal, y_global)` for a headless
    model, 3-tuple `(y_nodal, y_global, deriv)` for a `deriv_heads=True` one
    -- so callers must index (`out[0]`), never unpack-by-position, to stay
    correct for both."""
    with torch.no_grad():
        return model(x.to(device), ei.to(device), ea.to(device), zeros, g=g)


def _prepare_graph_inputs(cfg, norm, X, pos_np, mesh_edges, g_vec, device):
    """Shared tensor-building for `predict_graph_seconds`/`predict_graph_derivs`
    (learned-heads Task 3): standardizes `X`, builds the edge set per `cfg`,
    and the FiLM `g` vector (if applicable). Returns `(x, ei, ea, zeros, g)`,
    all ready to feed straight into `_run_model`."""
    fmean = torch.tensor(norm["feat_mean"], dtype=torch.float32)
    fstd = torch.tensor(norm["feat_std"], dtype=torch.float32)
    x = (torch.tensor(X, dtype=torch.float32) - fmean) / fstd
    pos = torch.tensor(pos_np, dtype=torch.float32)
    me = torch.tensor(mesh_edges, dtype=torch.long) if mesh_edges is not None else None
    ei, ea = _build_edges(pos, cfg, me)
    zeros = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    g = None
    if cfg.get("film", False) and g_vec is not None:
        g = torch.tensor(np.asarray(g_vec), dtype=torch.float32).reshape(1, -1).to(device)
    return x, ei, ea, zeros, g


def predict_graph_seconds(model, cfg, norm, X, pos_np, mesh_edges=None, g_vec=None,
                          device="cpu") -> np.ndarray:
    """One graph's per-node hybrid-shape input -> physical-seconds GNN prediction.

    `g_vec`: the cache's raw (5,) alloy/mould vector, forwarded un-standardized
    (mirrors `SolidGraphDataset`, which does not standardize `g`) -- only used
    when `cfg["film"]` is set.

    Learned-heads Task 3: `model` may be a `deriv_heads=True` `SolidGNN`
    (forward returns a 3-tuple `(y_nodal, y_global, deriv)` instead of the
    headless 2-tuple) -- this function's OWN return contract is unchanged
    either way (always just the `(N,)` physical-seconds array from
    `y_nodal`); the extra `deriv` output (if any) is simply ignored here.
    Use the sibling `predict_graph_derivs` to get the derivative heads'
    predictions.
    """
    x, ei, ea, zeros, g = _prepare_graph_inputs(cfg, norm, X, pos_np, mesh_edges, g_vec, device)
    out = _run_model(model, x, ei, ea, zeros, g, device)
    yn = out[0]
    return predict_seconds_gnn(yn.cpu().numpy(), norm)


def predict_graph_derivs(model, cfg, norm, X, pos_np, mesh_edges=None, g_vec=None,
                         device="cpu"):
    """One graph's per-node hybrid-shape input -> physical-units learned
    gradT (K/m) / dTdt (K/s), the two `deriv_heads` outputs (learned-heads
    Task 3). Sibling of `predict_graph_seconds`, sharing its exact tensor
    construction (standardized `X`, `cfg`-selected edge set, FiLM `g`) --
    only the output side differs.

    `model` MUST be a `deriv_heads=True` SolidGNN (forward returns a 3-tuple
    `(y_nodal, y_global, deriv)`, `deriv` shape `(N, 2)` = standardized
    `[log1p(gradT), log1p(|dTdt|)]`) -- a headless (2-tuple) model raises a
    loud `ValueError` naming the mismatch rather than crashing on a tuple-
    index error deeper in the call, or silently returning garbage.

    `norm` must carry `deriv_log_mean`/`deriv_log_std` (each shape `(2,)`,
    written by `src.train_heads.train_deriv_heads` into the checkpoint's
    `norm` dict) -- the per-head log-space standardization stats the model
    was trained against.

    Inverse transform (pinned exactly, mirrors `predict_seconds_gnn`'s own
    `pred_log = pred_std_log * std + mean; expm1`, applied per-head):
    `deriv_std * z + deriv_mean -> expm1 -> clip(min=0)`. Both outputs are
    non-negative by construction: `gradT` is a gradient MAGNITUDE (never
    signed), and `dTdt` is returned as a positive magnitude in K/s -- this
    matches the cache's own `dTdt` convention (`tools/doe/crossing_capture.py`
    stores `dTdt_at = (T_prev - T_now) / dt`, i.e. a positive cooling rate by
    construction on a monotonically-cooling casting; `src.train_heads`'s
    `DerivGraphDataset` additionally takes `.abs()` defensively before
    log-standardizing) -- and `src.shrinkage.niyama_exact` itself also takes
    `np.abs(dTdt_K_per_s)` internally, so this function's non-negative
    `dTdt` is compatible either way.

    Returns `(gradT_K_per_m, dTdt_K_per_s)`, each an `(N,)` float64 array.
    """
    x, ei, ea, zeros, g = _prepare_graph_inputs(cfg, norm, X, pos_np, mesh_edges, g_vec, device)
    out = _run_model(model, x, ei, ea, zeros, g, device)
    if len(out) != 3:
        raise ValueError(
            "predict_graph_derivs: model did not return a deriv-heads output "
            f"(expected a 3-tuple (y_nodal, y_global, deriv), got a "
            f"{len(out)}-tuple) -- pass a deriv_heads=True SolidGNN / a heads "
            "checkpoint (cfg['deriv_heads'] truthy, loaded via load_ft2b_gnn)")
    deriv = out[2].cpu().numpy().astype(np.float64)
    mean = np.asarray(norm["deriv_log_mean"], dtype=np.float64)
    std = np.asarray(norm["deriv_log_std"], dtype=np.float64)
    log_vals = deriv * std + mean
    vals = np.clip(np.expm1(log_vals), 0.0, None)
    return vals[:, 0], vals[:, 1]


# --------------------------------------------------------------------------- #
# Per-stem v4 hybrid prediction (GNN shape x GBM total-time scale)
# --------------------------------------------------------------------------- #
def _gbm_seconds_for_stem(gbm, stem, manifest_rows, manifest_dir, features_dir) -> float:
    x_row, _y_log_true = _row_features(stem, manifest_rows, manifest_dir, features_dir)
    log_pred = float(gbm.predict(x_row.reshape(1, -1))[0])
    return float(np.clip(np.expm1(log_pred), 0.0, None))


def _evaluate_stems_v4(model, cfg, norm, gbm, stems, manifest_rows, manifest_dir,
                       features_dir, device, hgb_std=None, lambda_shrink: float = 1.0,
                       w_total: float = 1.0, w_std: float = 1.0
                       ) -> list[dict]:
    """One record per stem: per-node y_true/y_hybrid/xyz + per-stem metadata
    (alloy_family, bc_config, t_ref, predicted total time).

    `hgb_std` (variance-recomposition Task 3): an optional loaded GBM
    std-head model (`src.gbm_ft2b.refit_gbm_std`'s output). `None` (the
    default) is byte-identical to before this argument existed --
    `_max_anchor_hybrid` (quantile-anchor). When given, each stem's row
    features are rebuilt via `src.gbm_ft2b.std_feature_row` (soft-anchor
    Task 3: width-dispatched off `hgb_std`'s own `n_features_in_`, 16 or 17
    -- a SEPARATE call from the one `_gbm_seconds_for_stem` makes internally
    for the total, kept separate, rather than refactored to share one call,
    so existing tests that monkeypatch `_gbm_seconds_for_stem` wholesale
    keep working unmodified) and recomposition uses
    `src.hybrid.hybrid_field_soft` (soft-anchor Task 3; was
    `hybrid_field_affine`) instead; each such record also carries
    `recomposition_info` (that function's `info` dict, now carrying
    `"m"`/`"target_unreachable"`/... in place of affine's `"a_clamped"`).

    `lambda_shrink` (variance-recomposition amendment 2): forwarded straight
    to `hybrid_field_soft` (ignored when `hgb_std` is `None`, same as every
    other soft-only knob here). `1.0` (the default) is byte-identical to
    before this argument existed.

    `w_total`/`w_std` (soft-anchor Task 3): forwarded straight to
    `hybrid_field_soft`'s own weights (ignored when `hgb_std` is `None`).
    `1.0`/`1.0` (the defaults) reproduce `hybrid_field_soft`'s own default
    weighting."""
    records = []
    for stem in stems:
        d = np.load(osp.join(features_dir, "graphs", f"{stem}.npz"))
        X, pos_np = d["X"], d["pos"]
        y_true = d["y_raw"] if "y_raw" in d.files else d["y"]
        mesh_edges = d["mesh_edges"] if "mesh_edges" in d.files else None
        g_vec = d["g"] if "g" in d.files else None
        gnn_seconds = predict_graph_seconds(model, cfg, norm, X, pos_np, mesh_edges,
                                            g_vec, device=device)
        gbm_seconds = _gbm_seconds_for_stem(gbm, stem, manifest_rows, manifest_dir,
                                            features_dir)
        recomp_info = None
        if hgb_std is not None:
            n_std_features = int(getattr(hgb_std, "n_features_in_", 16))
            x_row = std_feature_row(stem, manifest_rows, manifest_dir, features_dir,
                                    n_std_features)
            std_seconds = predict_std_seconds(hgb_std, x_row)
            hyb, recomp_info = hybrid_field_soft(gnn_seconds, gbm_seconds, std_seconds,
                                                 lambda_shrink=lambda_shrink,
                                                 w_total=w_total, w_std=w_std)
        else:
            hyb = _max_anchor_hybrid(gnn_seconds, gbm_seconds)
        row = manifest_rows.get(stem, {})
        t_ref = float(d["t_ref"]) if "t_ref" in d.files else None
        rec = {
            "stem": stem,
            "family": row.get("alloy_family") or "unknown",
            "bc": row.get("bc_config") or "none",
            "dataset": row.get("dataset") or "",
            "y_true": y_true, "y_hybrid": hyb, "xyz": pos_np,
            "gbm_seconds": gbm_seconds, "t_ref": t_ref,
            "pred_total": float(np.max(hyb)),
        }
        if recomp_info is not None:
            rec["recomposition_info"] = recomp_info
        records.append(rec)
    return records


def _group_metrics(records, key: str) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    out = {}
    for gval, recs in groups.items():
        y = np.concatenate([r["y_true"] for r in recs])
        p = np.concatenate([r["y_hybrid"] for r in recs])
        xyz = np.concatenate([r["xyz"] for r in recs])
        gid = np.concatenate([np.full(r["y_true"].shape[0], i, dtype=np.int64)
                              for i, r in enumerate(recs)])
        out[str(gval)] = {
            "n_stems": len(recs),
            "n_nodes": int(y.shape[0]),
            "mae": metrics.mae(y, p),
            "hotspot_overlap": hotspot_overlap(y, p, xyz, gid),
            "std_ratio": metrics.pred_std_ratio(y, p, gid),
        }
    return out


def _overall_metrics(records) -> dict:
    if not records:
        return {"n_stems": 0, "n_nodes": 0, "mae": 0.0, "hotspot_overlap": 0.0,
                "std_ratio": 0.0}
    y = np.concatenate([r["y_true"] for r in records])
    p = np.concatenate([r["y_hybrid"] for r in records])
    xyz = np.concatenate([r["xyz"] for r in records])
    gid = np.concatenate([np.full(r["y_true"].shape[0], i, dtype=np.int64)
                          for i, r in enumerate(records)])
    return {
        "n_stems": len(records), "n_nodes": int(y.shape[0]),
        "mae": metrics.mae(y, p),
        "hotspot_overlap": hotspot_overlap(y, p, xyz, gid),
        "std_ratio": metrics.pred_std_ratio(y, p, gid),
    }


# --------------------------------------------------------------------------- #
# No-regression check vs the SP1 hybrid
# --------------------------------------------------------------------------- #
# DESIGN CHOICE (documented per Task 8 brief): the no-regression subset is
# manifest dataset=="A" (all-sand, no BC variation -- the closest analog to
# the OLD single-alloy regime SP1 was trained/validated on) intersected with
# the union of interp.test/extrap.test. SP1 itself cannot be run on this
# subset "as originally trained" (SP1 has no v4 cache -- its own
# `data/features_v3` cache is a DIFFERENT, single-alloy dataset with
# different geometries/splits). Instead we run the SP1 GNN + SP1 HGB THE SAME
# WAY they are normally run (`src.evaluate_gnn.evaluate`): standardize with
# SP1's OWN norm stats and feed it whatever 13/9-column feature vector it's
# given. Since `FEATURE_NAMES_V4 == FEATURE_NAMES_V3 + [2 BC cols]` and
# `FEATURE_NAMES_V3 == FEATURE_NAMES + [4 local-modulus/geodesic cols]`
# (`src/features.py`), `v4_X[:, :13]` IS the v3 feature vector and
# `v4_X[:, :9]` IS the v2 feature vector for the SAME geometry -- so this is
# a legitimate "plug the v4 cache's geometry into the SP1 stack" comparison,
# not an apples-to-oranges one. CAVEATS: (1) the v4 A-subset's alloy
# properties vary (synthetic LHS-sampled alloys, not the single fixed alloy
# SP1's target distribution assumes) -- SP1 has no way to know this, so its
# scale prediction (HGB) is evaluated fully out-of-its-training-distribution
# here; a "regression" in this comparison could reflect the alloy variety
# alone, not the FT2b GNN's shape quality. (2) mesh/graph identity is
# guaranteed (same X/pos/mesh_edges array), so any MAE delta is NOT a
# geometry-mismatch artifact. (3) small subset sizes (a handful of A-rows in
# the actual test splits) make this a noisy signal, not a hard gate by
# itself -- read alongside per-family/per-bc numbers above.
def _no_regression_subset(splits_d: dict, manifest_rows: dict) -> list[str]:
    test_stems = set(splits_d["interp"]["test"]) | set(splits_d["extrap"]["test"])
    return sorted(s for s in test_stems if manifest_rows.get(s, {}).get("dataset") == "A")


def _sp1_hybrid_predict_stem(stem, sp1_model, sp1_cfg, sp1_norm, hgb, hgb_norm,
                             features_dir, device):
    d = np.load(osp.join(features_dir, "graphs", f"{stem}.npz"))
    X, pos_np = d["X"], d["pos"]
    y_true = d["y_raw"] if "y_raw" in d.files else d["y"]
    mesh_edges = d["mesh_edges"] if "mesh_edges" in d.files else None
    X13 = X[:, :N_V3_FEATURES]
    gnn_seconds = predict_graph_seconds(sp1_model, sp1_cfg, sp1_norm, X13, pos_np,
                                        mesh_edges, g_vec=None, device=device)
    Xb = standardize(X[:, :N_V2_FEATURES], hgb_norm)
    baseline_seconds = predict_seconds(hgb, Xb, hgb_norm)
    hyb = hybrid_field(gnn_seconds, baseline_seconds)
    return y_true, hyb, pos_np


def _no_regression_check(splits_d, manifest_rows, features_dir, v4_records_by_stem,
                         sp1_ckpt, sp1_hgb, baseline_features_dir, device,
                         tolerance: float = 1.15) -> dict:
    subset = _no_regression_subset(splits_d, manifest_rows)
    note = ("subset = manifest dataset=='A' (all-sand) stems present in "
            "interp.test | extrap.test; SP1 hybrid run on the SAME v4-cache "
            "geometry via v4_X[:, :13] (v3 features) / v4_X[:, :9] (v2 "
            "features) fed through SP1's own norm stats -- see this module's "
            "`_no_regression_subset`/`_no_regression_check` docstring for the "
            "full design rationale + caveats (alloy variety is OOD for SP1's "
            "HGB; small-subset noise).")
    if not subset:
        zero = {"n_stems": 0, "mae": 0.0, "hotspot_overlap": 0.0, "std_ratio": 0.0}
        return {"n_stems": 0, "v4_hybrid": zero, "sp1_hybrid": dict(zero),
                "mae_ratio": 0.0, "regression_flag": False,
                "note": note + " EMPTY SUBSET: no dataset=='A' test stems found."}

    v4_recs = [v4_records_by_stem[s] for s in subset if s in v4_records_by_stem]
    v4_metrics = _overall_metrics(v4_recs)

    sp1_model, sp1_cfg, sp1_norm = load_ft2b_gnn(sp1_ckpt, device)
    hgb = joblib.load(sp1_hgb) if isinstance(sp1_hgb, (str, os.PathLike)) else sp1_hgb
    hgb_norm = load_norm(baseline_features_dir)

    ys, ps, xyzs = [], [], []
    for stem in subset:
        y_true, hyb, pos_np = _sp1_hybrid_predict_stem(
            stem, sp1_model, sp1_cfg, sp1_norm, hgb, hgb_norm, features_dir, device)
        ys.append(y_true); ps.append(hyb); xyzs.append(pos_np)
    y = np.concatenate(ys); p = np.concatenate(ps); xyz = np.concatenate(xyzs)
    gid = np.concatenate([np.full(ys[i].shape[0], i, dtype=np.int64) for i in range(len(ys))])
    sp1_metrics = {
        "n_stems": len(subset), "n_nodes": int(y.shape[0]),
        "mae": metrics.mae(y, p),
        "hotspot_overlap": hotspot_overlap(y, p, xyz, gid),
        "std_ratio": metrics.pred_std_ratio(y, p, gid),
    }

    mae_ratio = (v4_metrics["mae"] / sp1_metrics["mae"]) if sp1_metrics["mae"] > 1e-9 else 0.0
    return {
        "n_stems": len(subset),
        "v4_hybrid": v4_metrics,
        "sp1_hybrid": sp1_metrics,
        "mae_ratio": float(mae_ratio),
        "regression_flag": bool(v4_metrics["mae"] > sp1_metrics["mae"] * tolerance),
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Chvorinov-scaling sanity check
# --------------------------------------------------------------------------- #
def _linfit(x: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares y = slope*x + intercept + R^2. Always finite: falls
    back to a degenerate-but-finite {0,0,0} fit when there are fewer than 2
    points or `x` is (numerically) constant, rather than raising or NaN'ing."""
    n = len(x)
    if n < 2 or np.ptp(x) < 1e-9:
        return {"n": int(n), "slope": 0.0, "intercept": 0.0, "r2": 0.0}
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"n": int(n), "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def _chvorinov_check(records: list[dict]) -> dict:
    """Regress the model's predicted total solidification time (`pred_total`
    = max of the per-node hybrid field, per stem) against each stem's cached
    analytic Chvorinov reference `t_ref = C_family * (V/A)^2`
    (`src/build_dataset.py::_finalize_row_stage2`). A clean per-family fit
    (slope near 1, high R^2) means the scale head still tracks Chvorinov
    scaling after fine-tuning; reported per family AND overall."""
    usable = [r for r in records if r["t_ref"] is not None and np.isfinite(r["t_ref"])]
    groups = defaultdict(list)
    for r in usable:
        groups[r["family"]].append(r)
    out = {}
    for fam, recs in groups.items():
        x = np.array([r["t_ref"] for r in recs], dtype=np.float64)
        y = np.array([r["pred_total"] for r in recs], dtype=np.float64)
        out[str(fam)] = _linfit(x, y)
    x_all = np.array([r["t_ref"] for r in usable], dtype=np.float64)
    y_all = np.array([r["pred_total"] for r in usable], dtype=np.float64)
    out["overall"] = _linfit(x_all, y_all)
    return out


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def _load_splits(splits) -> dict:
    if isinstance(splits, (str, os.PathLike)):
        with open(splits) as f:
            return json.load(f)
    return splits


def evaluate_ft2b(ckpt, gbm, features_dir, manifest, splits,
                  sp1_ckpt=DEFAULT_SP1_CKPT, sp1_hgb=DEFAULT_SP1_HGB,
                  baseline_features_dir=DEFAULT_BASELINE_FEATURES_DIR,
                  out_path=DEFAULT_OUT_PATH,
                  variance_collapse_threshold: float = 0.9,
                  no_regression_tolerance: float = 1.15,
                  hgb_std=None,
                  lambda_shrink: float = 1.0,
                  w_total: float = 1.0,
                  w_std: float = 1.0,
                  device=None) -> dict:
    """Evaluate a trained FT2b v4 checkpoint + refit GBM hybrid predictor.

    Args:
        ckpt: path to the v4 GNN checkpoint (`{model, cfg, norm}`), or an
            already-loaded checkpoint dict.
        gbm: path to the refit multi-alloy GBM (`src.gbm_ft2b.refit_gbm`'s
            `out_path`), or an already-loaded estimator.
        features_dir: v4 feature cache directory (`graphs/<stem>.npz`).
        manifest: path to the DOE manifest CSV (`run_id`==stem; needs
            `dataset`/`alloy_family`/`bc_config`/`material_id`/
            `mold_effusivity` columns).
        splits: path to `features_dir/splits.json`-shaped JSON, or an
            already-loaded `{"interp": {"train","test"}, "extrap":
            {"train","test"}}` dict.
        sp1_ckpt / sp1_hgb / baseline_features_dir: the SP1 production
            hybrid stack, for the no-regression comparison.
        out_path: where to write the result as JSON (set `None`/falsy to
            skip writing).
        variance_collapse_threshold: any per-family/per-bc `std_ratio` below
            this trips `variance_collapse_flag`.
        no_regression_tolerance: the v4 hybrid may be at most this factor
            worse (MAE) than the SP1 hybrid on the all-sand subset before
            `no_regression.regression_flag` trips.
        hgb_std (variance-recomposition Task 3): an optional path to (or
            already-loaded) GBM std-head model
            (`src.gbm_ft2b.refit_gbm_std`'s output). `None` (the default)
            keeps every v4 hybrid composition in this module on the
            quantile-anchor (`src.hybrid.hybrid_field_max_anchor`) --
            byte-identical to before this argument existed. When given,
            every v4 composition (`_evaluate_stems_v4`, and therefore every
            per-family/per-bc metric, the v4 side of `no_regression`, and
            `chvorinov`'s `pred_total`) uses the soft-anchor (soft-anchor
            Task 3; was affine) recomposition instead
            (`src.hybrid.hybrid_field_soft`); the SP1 side of
            `no_regression` is unaffected (SP1 has no std-head analog).
        lambda_shrink (variance-recomposition amendment 2): forwarded
            straight to `_evaluate_stems_v4`/`hybrid_field_soft` (ignored
            when `hgb_std` is `None`). `1.0` (the default) is byte-identical
            to before this argument existed.
        w_total / w_std (soft-anchor Task 3): forwarded straight to
            `_evaluate_stems_v4`/`hybrid_field_soft`'s own weights (ignored
            when `hgb_std` is `None`). `1.0`/`1.0` (the defaults) reproduce
            `hybrid_field_soft`'s own default weighting.
        device: "cuda"/"cpu"; default picks cuda if available.

    Returns: nested dict with `interp`/`extrap` (each `per_alloy_family`,
        `per_bc_config`, `overall`), `no_regression`, `chvorinov`,
        `variance_collapse_flag`, `recomposition` ("soft" iff `hgb_std`
        was given, else "anchor").
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, norm = load_ft2b_gnn(ckpt, device)
    gbm_model = joblib.load(gbm) if isinstance(gbm, (str, os.PathLike)) else gbm
    hgb_std_model = joblib.load(hgb_std) if isinstance(hgb_std, (str, os.PathLike)) else hgb_std
    manifest_dir = osp.dirname(osp.abspath(manifest)) or "."
    manifest_rows = _load_manifest_rows(manifest)
    splits_d = _load_splits(splits)

    result: dict = {}
    records_by_stem: dict = {}
    for split_name in ("interp", "extrap"):
        test_stems = splits_d[split_name]["test"]
        records = _evaluate_stems_v4(model, cfg, norm, gbm_model, test_stems,
                                     manifest_rows, manifest_dir, features_dir, device,
                                     hgb_std=hgb_std_model, lambda_shrink=lambda_shrink,
                                     w_total=w_total, w_std=w_std)
        for r in records:
            records_by_stem[r["stem"]] = r
        result[split_name] = {
            "n_stems": len(records),
            "per_alloy_family": _group_metrics(records, "family"),
            "per_bc_config": _group_metrics(records, "bc"),
            "overall": _overall_metrics(records),
        }

    variance_collapse = False
    for split_res in result.values():
        for grouping in ("per_alloy_family", "per_bc_config"):
            for m in split_res[grouping].values():
                if m["std_ratio"] < variance_collapse_threshold:
                    variance_collapse = True
    result["variance_collapse_flag"] = variance_collapse

    result["no_regression"] = _no_regression_check(
        splits_d, manifest_rows, features_dir, records_by_stem,
        sp1_ckpt, sp1_hgb, baseline_features_dir, device,
        tolerance=no_regression_tolerance)

    result["chvorinov"] = _chvorinov_check(list(records_by_stem.values()))

    result["recomposition"] = "soft" if hgb_std_model is not None else "anchor"

    if out_path:
        out_dir = osp.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    return result
