"""FT2b Task 7: gate-metric checkpoint selector.

`src.train_gnn.train(..., save_every_epoch=True)` (Task 6) writes one
checkpoint per epoch (`<out_dir>/ckpt_ep{ep:03d}.pt`, payload
`{model, norm, cfg}`). Picking among them by validation loss alone can pick a
variance-collapsed checkpoint (near-constant predictions can still have low
smooth-L1 loss on a small/skewed val set) -- exactly the failure mode the
FT2b gate metrics (`src.evaluate_ft2b`'s `std_ratio`/`hotspot_overlap`) exist
to catch. This module re-ranks the per-epoch checkpoints by those SAME gate
metrics instead.

=== Why no GBM is needed here (key insight from the Task 7 brief) ===
`src.evaluate_ft2b._max_anchor_hybrid` recomposes the hybrid prediction as
`pred_i = gnn_i - anchor + gbm_total`, i.e. a per-graph ADDITIVE SHIFT of
the GNN field by a constant (`gbm_total - anchor`, the same shift for every
node in that graph), where `anchor = np.quantile(gnn, q)` (default
`q=0.995`, a high quantile rather than the exact max -- see
`src.hybrid.hybrid_field_max_anchor`'s docstring for why: the exact-max
anchor is fragile to a handful of outlier nodes). Both metrics this
selector ranks by are shift-invariant per graph regardless of which
quantile the anchor uses:
  - `std_ratio = std(pred)/std(true)` (`src.metrics.pred_std_ratio`): std of a
    per-graph-shifted array is unchanged by the shift.
  - `hotspot_overlap` (`src.metrics.hotspot_accuracy`): picks the top-`k`
    LATEST-solidifying nodes by rank; a per-graph constant shift does not
    reorder nodes within a graph, so the top-k set (and thus the overlap) is
    unchanged.
So GNN-only seconds and the eventual hybrid produce IDENTICAL std_ratio/
hotspot_overlap numbers -- this selector never needs the GBM (which also
means it never needs the DOE manifest's alloy/BC feature-row resolution that
`src.gbm_ft2b._row_features` performs; only `alloy_family`/`bc_config` for
grouping are read from the manifest).

=== Reuse, not reimplementation ===
Everything gate-metric-shaped is imported from `src.evaluate_ft2b`, which is
already exercised by `tests/test_evaluate_ft2b.py`:
  - `load_ft2b_gnn` -- checkpoint -> `(model, cfg, norm)`, film-aware.
  - `predict_graph_seconds` -- one graph's cached `X`/`pos`/`mesh_edges`/`g`
    -> physical-seconds GNN prediction (the ckpt's own norm-inversion chain).
  - `_group_metrics` / `_overall_metrics` -- the EXACT per-group (and
    overall) `{mae, hotspot_overlap, std_ratio}` computation `evaluate_ft2b`
    uses for its own `eval.json` (pool all nodes of every stem in the group,
    build a per-stem `graph_id` array, hand the pooled arrays to
    `src.metrics.pred_std_ratio`/`hotspot_overlap`). This selector builds the
    same `{"family", "bc", "y_true", "y_hybrid", "xyz"}`-shaped records (with
    `y_hybrid` set to the GNN-only seconds, per the shift-invariance above)
    and calls those two functions UNCHANGED, so its numbers are directly
    comparable to `evaluate_ft2b`'s `eval.json` for the same stems/ckpt.
  - `src.gbm_ft2b._load_manifest_rows` -- manifest CSV -> `{run_id: row}`.

=== Ranking (fix pass -- two-sided calibration, not one-sided "biggest std") ===
For each checkpoint, pool `per_alloy_family` and `per_bc_config` group
metrics (`bc_config UNION alloy_family`, i.e. every group from BOTH
partitions, not their cross product) into one flat collection, then rank
checkpoints by, in order:
  1. FEWEST of those groups with `std_ratio < std_ratio_threshold` (default
     0.9, matching `evaluate_ft2b`'s own `variance_collapse_threshold`
     default) -- fewer variance-collapsed groups is better.
  2. SMALLEST `calib = max over groups of abs(log(std_ratio))` -- a
     TWO-SIDED mis-calibration score. `std_ratio` measures the same
     miscalibration whether it collapses (std_ratio << 1) or over-disperses
     (std_ratio >> 1); `log` makes the two directions symmetric
     (`log(0.5) == -log(2)`) so neither a collapsed nor an inflated group can
     win by having a numerically larger raw ratio.
     ORIGINAL (WRONG) CRITERION, replaced here: "largest minimum group
     std_ratio" -- that criterion is UNBOUNDED ABOVE and actively rewards
     over-dispersion (a checkpoint where every group is 2.3x over-dispersed
     beats one where every group is a well-calibrated ~0.95, because 2.3 >
     0.95 as a raw ratio). Confirmed on a real 64-checkpoint run: the old
     criterion picked ep019 (min_group_std_ratio=2.34, EVERY group
     over-dispersed) over ep063 (all groups >= 0.9, calibrated). `calib`
     fixes this: ep019's `calib = abs(log(2.34)) ~= 0.85`; a calibrated
     ep063 with all ratios near 1 has `calib` near 0 and wins.
  3. HIGHEST overall `hotspot_overlap` (pooled over every given stem).
  4. Deterministic tie-break: LOWEST epoch number (prefers the earlier,
     cheaper-to-have-trained checkpoint when the gate metrics are tied).

`min_group_std_ratio`/`max_group_std_ratio` are still computed and stored on
every ranking row (for human inspection / plotting the calibration spread)
-- they are just no longer part of the sort key beyond criterion (1)'s
threshold count.

=== `select_ckpt` vs `rerank_selection` ===
`select_ckpt` runs the (slow -- GPU forward pass per checkpoint) evaluation
and produces `selection.json`. `rerank_selection` re-applies the ranking
rule above to an ALREADY-WRITTEN `selection.json` (the per-group numbers for
every checkpoint are already in that file) with NO forward passes --
needed because a real run over many checkpoints takes hours, and a ranking
bugfix must not force re-running it. Both route through the same
`_rank_entries` so the two can never silently diverge.
"""
from __future__ import annotations

import glob
import json
import math
import os
import os.path as osp
import re

import numpy as np
import torch

from src.evaluate_ft2b import (_group_metrics, _overall_metrics,
                               load_ft2b_gnn, predict_graph_seconds)
from src.gbm_ft2b import _load_manifest_rows

DEFAULT_STD_RATIO_THRESHOLD = 0.9

# Floor applied to std_ratio before taking log(), so a fully-collapsed group
# (std_ratio == 0.0 exactly) yields a large-but-finite `calib` instead of
# `inf` (which `json.dump` would emit as the non-standard token `Infinity`).
_CALIB_STD_RATIO_FLOOR = 1e-6

_CKPT_EPOCH_RE = re.compile(r"ckpt_ep(\d+)\.pt$")


def _epoch_of(ckpt_path: str) -> int:
    m = _CKPT_EPOCH_RE.search(osp.basename(ckpt_path))
    if not m:
        raise ValueError(f"not a ckpt_ep###.pt path: {ckpt_path!r}")
    return int(m.group(1))


def _list_ckpts(ckpt_dir: str) -> list[str]:
    paths = glob.glob(osp.join(ckpt_dir, "ckpt_ep*.pt"))
    return sorted(paths, key=_epoch_of)


def _default_stems(features_dir: str) -> list[str]:
    with open(osp.join(features_dir, "splits.json")) as f:
        splits_d = json.load(f)
    return list(splits_d["extrap"]["test"])


def _gnn_only_records(model, cfg, norm, stems, manifest_rows, features_dir,
                      device) -> list[dict]:
    """One record per stem: `{family, bc, y_true, y_hybrid, xyz}`, shaped
    exactly for `src.evaluate_ft2b._group_metrics`/`_overall_metrics` --
    `y_hybrid` here holds the GNN-ONLY seconds prediction (see this module's
    docstring for why that's metric-equivalent to the eventual hybrid)."""
    records = []
    for stem in stems:
        d = np.load(osp.join(features_dir, "graphs", f"{stem}.npz"))
        X, pos_np = d["X"], d["pos"]
        y_true = d["y_raw"] if "y_raw" in d.files else d["y"]
        mesh_edges = d["mesh_edges"] if "mesh_edges" in d.files else None
        g_vec = d["g"] if "g" in d.files else None
        gnn_seconds = predict_graph_seconds(model, cfg, norm, X, pos_np, mesh_edges,
                                            g_vec, device=device)
        row = manifest_rows.get(stem, {})
        records.append({
            "stem": stem,
            "family": row.get("alloy_family") or "unknown",
            "bc": row.get("bc_config") or "none",
            "y_true": y_true, "y_hybrid": gnn_seconds, "xyz": pos_np,
        })
    return records


def _combined_groups(per_family: dict, per_bc: dict) -> dict:
    """Union (not cross product) of the two partitions' groups, disambiguated
    by a `"family:"`/`"bc:"` prefix so same-named groups in each partition
    (e.g. a `bc_config` and an `alloy_family` both happening to be `"none"`)
    don't collide."""
    out = {}
    for k, v in per_family.items():
        out[f"family:{k}"] = v
    for k, v in per_bc.items():
        out[f"bc:{k}"] = v
    return out


def _calib_score(std_ratio: float) -> float:
    """Two-sided mis-calibration score for one group's `std_ratio`.

    `abs(log(std_ratio))`, floored at `_CALIB_STD_RATIO_FLOOR` before the log
    so an exact-zero (fully collapsed) ratio gives a large finite number
    instead of `inf`. 0 at perfect calibration (`std_ratio == 1`), symmetric
    between collapse and over-dispersion (`calib(0.5) == calib(2.0)`)."""
    r = max(float(std_ratio), _CALIB_STD_RATIO_FLOOR)
    return abs(math.log(r))


def _rank_entries(entries: list[dict], std_ratio_threshold: float) -> list[dict]:
    """Shared scoring + sort for both `select_ckpt` and `rerank_selection`.

    `entries`: one dict per checkpoint, each with (at least)
    `{"ckpt", "epoch", "overall", "per_alloy_family", "per_bc_config"}` --
    exactly what both callers already have (freshly computed, or loaded back
    out of a previously-written `selection.json`). Recomputes the derived
    scoring fields (`n_fail_groups`/`min_group_std_ratio`/
    `max_group_std_ratio`/`calib`/`overall_hotspot`) from the stored
    per-group numbers -- no forward pass, no re-evaluation -- and returns a
    NEW list of entries (original dicts are not mutated), sorted best-first
    per this module's ranking rule (see module docstring)."""
    scored = []
    for e in entries:
        combined = _combined_groups(e["per_alloy_family"], e["per_bc_config"])
        std_ratios = [m["std_ratio"] for m in combined.values()]
        n_fail_groups = sum(1 for r in std_ratios if r < std_ratio_threshold)
        min_group_std_ratio = float(min(std_ratios)) if std_ratios else 0.0
        max_group_std_ratio = float(max(std_ratios)) if std_ratios else 0.0
        calib = max((_calib_score(r) for r in std_ratios), default=0.0)
        overall_hotspot = float(e["overall"]["hotspot_overlap"])

        new_entry = dict(e)
        new_entry.update({
            "n_fail_groups": int(n_fail_groups),
            "min_group_std_ratio": min_group_std_ratio,
            "max_group_std_ratio": max_group_std_ratio,
            "calib": float(calib),
            "overall_hotspot": overall_hotspot,
        })
        scored.append(new_entry)

    scored.sort(key=lambda r: (r["n_fail_groups"], r["calib"],
                               -r["overall_hotspot"], r["epoch"]))
    return scored


def rerank_selection(json_path, out_json=None,
                     std_ratio_threshold: float = DEFAULT_STD_RATIO_THRESHOLD) -> dict:
    """Re-rank an ALREADY-WRITTEN `select_ckpt` output (`json_path`) with the
    current ranking rule, WITHOUT re-running any GNN forward pass -- every
    checkpoint's per-group numbers are already stored in `json_path`'s
    `"ranking"` entries (`per_alloy_family`/`per_bc_config`/`overall`), so
    this only recomputes the derived scoring fields and re-sorts. Exists so a
    ranking-rule bugfix never forces re-running a real (hours-long,
    many-checkpoint) `select_ckpt` call.

    Args:
        json_path: path to a JSON file shaped like `select_ckpt`'s return
            value (`{"best", "ranking": [...]}`) -- typically its own
            previous `out_json` output.
        out_json: path to write the re-ranked result (falsy skips writing).
        std_ratio_threshold: same meaning as `select_ckpt`'s.

    Returns: `{"best": <ckpt path>, "ranking": [...]}`, re-sorted.
    """
    with open(json_path) as f:
        data = json.load(f)
    ranking = _rank_entries(data["ranking"], std_ratio_threshold)
    result = {"best": ranking[0]["ckpt"], "ranking": ranking}

    if out_json:
        out_dir = osp.dirname(out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

    return result


def select_ckpt(ckpt_dir, features_dir, manifest, stems=None, out_json=None,
                device=None, std_ratio_threshold: float = DEFAULT_STD_RATIO_THRESHOLD) -> dict:
    """Rank every `<ckpt_dir>/ckpt_ep*.pt` checkpoint by the gate metrics
    (per-group `std_ratio` calibration + `hotspot_overlap`), NOT val loss.

    Args:
        ckpt_dir: directory containing `ckpt_ep{ep:03d}.pt` files (Task 6's
            `train(..., save_every_epoch=True)` output).
        features_dir: v4 feature cache directory (`graphs/<stem>.npz` +
            `splits.json`, used only when `stems` is None).
        manifest: path to the DOE manifest CSV (`run_id`==stem; needs
            `alloy_family`/`bc_config` columns).
        stems: stems to evaluate on. Default (None): `splits.json`'s
            `extrap.test` list (the OOD-generalization split -- the one the
            gate metrics matter most on).
        out_json: path to write the ranked result as JSON (falsy skips
            writing).
        device: "cuda"/"cpu"; default picks cuda if available.
        std_ratio_threshold: a group's `std_ratio` below this counts as
            "failing" for ranking criterion (1). Default 0.9, matching
            `evaluate_ft2b`'s own `variance_collapse_threshold` default.

    Returns: `{"best": <ckpt path>, "ranking": [<per-ckpt dict>, ...]}`,
        `ranking` sorted best-first per this module's docstring's 4-step
        ranking rule. Each ranking entry also carries `per_alloy_family`/
        `per_bc_config`/`overall` (same shape as `evaluate_ft2b`'s `eval.json`
        groups) so the numbers are directly inspectable/comparable.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = _list_ckpts(ckpt_dir)
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_ep*.pt files found in {ckpt_dir!r}")

    if stems is None:
        stems = _default_stems(features_dir)
    manifest_rows = _load_manifest_rows(manifest)

    raw_entries = []
    n = len(ckpts)
    for i, ckpt_path in enumerate(ckpts):
        epoch = _epoch_of(ckpt_path)
        print(f"[select_ckpt] ({i + 1}/{n}) evaluating {osp.basename(ckpt_path)} "
              f"on {len(stems)} stems ...", flush=True)
        model, cfg, norm = load_ft2b_gnn(ckpt_path, device)
        records = _gnn_only_records(model, cfg, norm, stems, manifest_rows,
                                    features_dir, device)
        entry = {
            "ckpt": ckpt_path,
            "epoch": epoch,
            "overall": _overall_metrics(records),
            "per_alloy_family": _group_metrics(records, "family"),
            "per_bc_config": _group_metrics(records, "bc"),
        }
        # Per-entry pass through the shared scorer just for the progress
        # print -- `select_ckpt`'s own final ranking below re-derives this
        # (identically) via `_rank_entries` over the whole batch, so the two
        # can never disagree.
        scored = _rank_entries([entry], std_ratio_threshold)[0]
        print(f"[select_ckpt]   epoch {epoch:3d}: n_fail_groups={scored['n_fail_groups']} "
              f"calib={scored['calib']:.4f} "
              f"min_group_std_ratio={scored['min_group_std_ratio']:.4f} "
              f"max_group_std_ratio={scored['max_group_std_ratio']:.4f} "
              f"overall_hotspot={scored['overall_hotspot']:.4f}", flush=True)
        raw_entries.append(entry)

    ranking = _rank_entries(raw_entries, std_ratio_threshold)
    result = {"best": ranking[0]["ckpt"], "ranking": ranking}

    if out_json:
        out_dir = osp.dirname(out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

    return result
