"""FT2b v4-inference Task 5: D-parts v4-vs-SP1 comparison.

Before promoting the v4 hybrid (FiLM GNN + multi-alloy GBM, `src.evaluate_ft2b`)
over the SP1 production hybrid (`runs/sp1_run3_final/best_gnn.pt` +
`runs/baseline_v2/hgb_A.joblib`), both stacks are compared head-to-head on
the real-part "dataset D" geometries (`tools/doe/prep_parts.py`'s cleaned
STEP parts) that have a FEniCSx-ground-truth v4 feature cache. This is the
*same* geometry fed through *both* stacks -- exactly the `_no_regression_check`
technique in `src.evaluate_ft2b` (v4_X[:, :13]/[:9] through SP1's own norm
stats), just scoped to dataset=="D" instead of dataset=="A", and reported
per-stem as well as aggregated.

Reused, not reimplemented: `load_ft2b_gnn`/`predict_graph_seconds` (GNN
loading+inference, film-aware), `_evaluate_stems_v4` (v4 quantile-anchor hybrid
per stem), `_sp1_hybrid_predict_stem` (SP1 mean-anchor hybrid on the v4
cache's geometry), `hotspot_overlap` -- all from `src.evaluate_ft2b`, the
single source of truth for both compositions (see that module's docstring
for the quantile-anchor-vs-mean-anchor rationale).
"""
from __future__ import annotations

import json
import os
import os.path as osp

import joblib
import numpy as np
import torch

from src import evaluate_ft2b as ft2b
from src import metrics
from src.dataload import load_norm
from src.gbm_ft2b import _load_manifest_rows


def _d_stems(manifest_rows: dict, features_dir: str) -> tuple[list[str], int]:
    """Sorted `run_id`s with `dataset=="D"` AND a v4 graph cache on disk, plus
    the raw count of `dataset=="D"` manifest rows (cache present or not) --
    the latter lets callers tell "no D rows in the manifest at all" apart
    from "D rows exist but none have been built into the v4 cache yet"."""
    d_rows = [stem for stem, row in manifest_rows.items() if row.get("dataset") == "D"]
    out = [s for s in d_rows if osp.exists(osp.join(features_dir, "graphs", f"{s}.npz"))]
    return sorted(out), len(d_rows)


def _stem_metrics(y_true: np.ndarray, y_pred: np.ndarray, xyz: np.ndarray) -> dict:
    gid = np.zeros(y_true.shape[0], dtype=np.int64)
    return {
        "mae": metrics.mae(y_true, y_pred),
        "hotspot": ft2b.hotspot_overlap(y_true, y_pred, xyz, gid),
        "std_ratio": metrics.pred_std_ratio(y_true, y_pred, gid),
    }


def _aggregate_metrics(records: list[dict]) -> dict:
    """Node-weighted aggregate over `{"y_true", "y_hybrid", "xyz"}` records.

    Delegates to `ft2b._overall_metrics` (single source of truth for the
    node-weighted concat/gid glue -- don't re-derive it here and risk it
    drifting out of sync), just renaming `hotspot_overlap` -> `hotspot` to
    match this module's own key convention."""
    agg = ft2b._overall_metrics(records)
    return {"mae": agg["mae"], "hotspot": agg["hotspot_overlap"],
            "std_ratio": agg["std_ratio"]}


def compare_d_parts(v4_ckpt, v4_gbm,
                    sp1_ckpt=ft2b.DEFAULT_SP1_CKPT, sp1_hgb=ft2b.DEFAULT_SP1_HGB,
                    features_dir="data/features_v4", manifest="doe_out/manifest.csv",
                    out_json="runs/v4_inference/d_parts_comparison.json",
                    baseline_features_dir=ft2b.DEFAULT_BASELINE_FEATURES_DIR,
                    hgb_std=None,
                    lambda_shrink: float = 1.0,
                    w_total: float = 1.0,
                    w_std: float = 1.0,
                    device=None) -> dict:
    """Head-to-head v4-hybrid vs SP1-hybrid comparison on dataset D stems.

    Args:
        v4_ckpt: path to the v4 GNN checkpoint, or an already-loaded
            checkpoint dict (`ft2b.load_ft2b_gnn`'s own convention).
        v4_gbm: path to the refit multi-alloy GBM (`src.gbm_ft2b.refit_gbm`),
            or an already-loaded estimator.
        sp1_ckpt / sp1_hgb: the SP1 production hybrid stack.
        features_dir: v4 feature cache directory (`graphs/<stem>.npz`).
        manifest: DOE manifest CSV (`run_id`==stem; needs `dataset`/
            `geometry_id`/`alloy_family`/`bc_config` columns).
        out_json: where to write the result as JSON (falsy skips the write;
            the parent directory is created when writing).
        baseline_features_dir: SP1 HGB norm stats (`norm_stats.npz`, 9 v2
            features) -- same default as `evaluate_ft2b`.
        hgb_std (variance-recomposition Task 3): an optional path to (or
            already-loaded) GBM std-head model
            (`src.gbm_ft2b.refit_gbm_std`'s output), forwarded ONLY to the
            v4 side's `ft2b._evaluate_stems_v4` (the SP1 side has no
            std-head analog and is unaffected). `None` (the default) is
            byte-identical to before this argument existed (the v4 side
            stays on the quantile-anchor). The returned dict gains a
            top-level `"recomposition": "soft"|"anchor"` field (soft-anchor
            Task 3: was `"affine"|"anchor"` -- `ft2b._evaluate_stems_v4` now
            recomposes via `src.hybrid.hybrid_field_soft` instead of
            `hybrid_field_affine` whenever `hgb_std` is given).
        lambda_shrink (variance-recomposition amendment 2): forwarded
            straight to the v4 side's `ft2b._evaluate_stems_v4` (ignored
            when `hgb_std` is `None`). `1.0` (the default) is byte-identical
            to before this argument existed.
        w_total / w_std (soft-anchor Task 3): forwarded straight to the v4
            side's `ft2b._evaluate_stems_v4`/`hybrid_field_soft`'s own
            weights (ignored when `hgb_std` is `None`). `1.0`/`1.0` (the
            defaults) reproduce `hybrid_field_soft`'s own default weighting.
        device: "cuda"/"cpu"; default picks cuda if available.

    Returns: `{"rows": [{"stem", "geometry_id", "alloy_family",
        "v4": {"mae", "hotspot", "std_ratio"}, "sp1": {...}}, ...],
        "v4": {...aggregate...}, "sp1": {...aggregate...}, "n_stems",
        "n_d_stems_in_manifest", "verdict": {"mae_ok", "hotspot_ok"[, "note"]},
        "recomposition"}`
        where `mae_ok = v4.mae <= sp1.mae` and `hotspot_ok = v4.hotspot >=
        sp1.hotspot` (both computed on the node-weighted aggregates,
        mirroring `evaluate_ft2b`'s `_overall_metrics`) -- EXCEPT when
        `n_stems == 0`, which forces BOTH verdict fields to `False` (an
        empty comparison is not evidence of no-regression: `0.0 <= 0.0` and
        `0.0 >= 0.0` would otherwise vacuously pass) and adds a `note`
        explaining why, mentioning `n_stems`/`n_d_stems_in_manifest`.
        `n_d_stems_in_manifest` is the raw count of `dataset=="D"` manifest
        rows (cache present or not), for skip-visibility -- `n_stems` (<=
        it) is how many of those actually had a `features_dir` graph cache
        and were compared.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest_dir = osp.dirname(osp.abspath(manifest)) or "."
    manifest_rows = _load_manifest_rows(manifest)
    stems, n_d_in_manifest = _d_stems(manifest_rows, features_dir)

    # -- v4 hybrid (GNN shape, quantile-anchor GBM scale -- or, when `hgb_std`
    # is given, the soft-anchor recomposition) -- exactly `evaluate_ft2b`'s
    # own per-stem composition, reused verbatim.
    v4_model, v4_cfg, v4_norm = ft2b.load_ft2b_gnn(v4_ckpt, device)
    v4_gbm_model = joblib.load(v4_gbm) if isinstance(v4_gbm, (str, os.PathLike)) else v4_gbm
    hgb_std_model = joblib.load(hgb_std) if isinstance(hgb_std, (str, os.PathLike)) else hgb_std
    v4_records = ft2b._evaluate_stems_v4(v4_model, v4_cfg, v4_norm, v4_gbm_model, stems,
                                        manifest_rows, manifest_dir, features_dir, device,
                                        hgb_std=hgb_std_model, lambda_shrink=lambda_shrink,
                                        w_total=w_total, w_std=w_std)
    v4_by_stem = {r["stem"]: r for r in v4_records}

    # -- SP1 hybrid (GNN shape, mean-anchor HGB scale) on the SAME v4-cache
    # geometry, fed through SP1's own norm stats -- `_sp1_hybrid_predict_stem`'s
    # own contract (see `evaluate_ft2b`'s `_no_regression_check` docstring for
    # the full rationale/caveats of this "plug the v4 cache into the SP1
    # stack" technique).
    sp1_model, sp1_cfg, sp1_norm = ft2b.load_ft2b_gnn(sp1_ckpt, device)
    hgb = joblib.load(sp1_hgb) if isinstance(sp1_hgb, (str, os.PathLike)) else sp1_hgb
    hgb_norm = load_norm(baseline_features_dir)
    sp1_by_stem = {}
    for stem in stems:
        y_true, hyb, pos_np = ft2b._sp1_hybrid_predict_stem(
            stem, sp1_model, sp1_cfg, sp1_norm, hgb, hgb_norm, features_dir, device)
        sp1_by_stem[stem] = {"y_true": y_true, "y_hybrid": hyb, "xyz": pos_np}

    rows = []
    for stem in stems:
        v4r, sp1r = v4_by_stem[stem], sp1_by_stem[stem]
        row = manifest_rows.get(stem, {})
        rows.append({
            "stem": stem,
            "geometry_id": row.get("geometry_id") or "",
            "alloy_family": v4r.get("family") or "unknown",
            "v4": _stem_metrics(v4r["y_true"], v4r["y_hybrid"], v4r["xyz"]),
            "sp1": _stem_metrics(sp1r["y_true"], sp1r["y_hybrid"], sp1r["xyz"]),
        })

    v4_agg = _aggregate_metrics([v4_by_stem[s] for s in stems])
    sp1_agg = _aggregate_metrics([sp1_by_stem[s] for s in stems])

    if stems:
        verdict = {
            "mae_ok": bool(v4_agg["mae"] <= sp1_agg["mae"]),
            "hotspot_ok": bool(v4_agg["hotspot"] >= sp1_agg["hotspot"]),
        }
    else:
        # An empty comparison must never read as a pass: with no records,
        # both aggregates are all-zero, so a naive `0.0 <= 0.0` /
        # `0.0 >= 0.0` would vacuously satisfy both gates. Force an
        # explicit fail instead, with a note pointing at WHY (0 of
        # n_d_in_manifest dataset=="D" manifest rows had a cached graph).
        verdict = {
            "mae_ok": False,
            "hotspot_ok": False,
            "note": (f"no D stems compared (n_stems=0 of "
                     f"n_d_stems_in_manifest={n_d_in_manifest} dataset==\"D\" "
                     "manifest rows have a features_dir graph cache) -- "
                     "vacuous run, verdict forced to fail rather than "
                     "trivially passing on empty aggregates"),
        }

    result = {
        "rows": rows,
        "v4": v4_agg,
        "sp1": sp1_agg,
        "n_stems": len(stems),
        "n_d_stems_in_manifest": n_d_in_manifest,
        "verdict": verdict,
        "recomposition": "soft" if hgb_std_model is not None else "anchor",
    }

    if out_json:
        out_dir = osp.dirname(out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

    return result
