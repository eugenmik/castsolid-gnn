"""FT2b Task 7: 3-stage staged-training driver (scale -> BC -> calibrate).

Orchestrates the already-built FT2b pieces on top of the SP1 production GNN:
  - Task 4 (`src.warm_start`): zero-init FiLM/BC-column warm-start (+ the
    stage-chaining `build_or_load_v4` entry point, used implicitly here since
    `src.train_gnn.train`'s `init_ckpt` calls it).
  - Task 5 (`src.train_gnn`): `train(...)`'s g/FiLM/freeze hooks.
  - Task 6 (`src.gbm_ft2b`): `refit_gbm(...)`, the multi-alloy scale-head GBM.

Three stages, each warm-starting from the previous stage's own checkpoint
(NOT always from the original SP1 checkpoint -- see `src.warm_start.
build_or_load_v4` for how `train()`'s `init_ckpt` handles both a genuine SP1
source and an already-v4 FT2b source):

  1. **scale/alloy adaptation** -- train stems = the v4 train pool restricted
     to `dataset == "A"` (all-sand, no BC variation). FiLM-only trainable
     (`freeze_backbone=True`), warm-started from `sp1_ckpt`, `lr=3e-4`. Also
     refits the scale-head GBM on the same stems -> `runs/ft2b_gbm/hgb.joblib`
     (the artifact Task 8/9 evaluate against by default).
  2. **BC generalization** -- train stems = train pool with
     `dataset in {A, B, C}` (chills / insulating & exothermic sleeves added).
     Full backbone unfrozen, chained from stage 1's checkpoint, `lr=1e-4`.
  3. **calibration** -- train stems = train pool rows that are either
     named-grade (`alloy_family != "synthetic"`) or `dataset == "D"` (real
     parts). Only the last 2 processor blocks + `scale_head` trainable
     (`freeze_mode="all_but_last2"`), chained from stage 2's checkpoint,
     `lr=1e-4`. Refits the GBM again on the calibration stems into a
     SEPARATE artifact, `runs/ft2b_gbm/hgb_stage3.joblib` -- `hgb.joblib`
     from stage 1 is intentionally left untouched; Task 8/9's evaluation
     decides which of the two GBMs the final hybrid predictor should use.

Reality check this driver must survive: in the actual FT2a/FT2b DOE
manifest + `data/features_v4/splits.json`, named-grade and dataset-D rows
are ENTIRELY reserved for validation (`val_interp`/`val_extrap`) -- none are
in the train pool. Stage 3 can therefore legitimately have zero train stems.
When that happens this driver logs loudly and SKIPS stage 3 (no crash, no
silent no-op train() call, no stage-3 GBM refit): `paths["final_ckpt"]` falls
back to stage 2's checkpoint.

All early-stopping/checkpoint-selection across all 3 stages uses the same
`extrap.test` split (`early_stop_val_stems`), per the FINETUNING plan.

Artifact isolation: this module writes ONLY under
`runs/ft2b_stage{1,2,3}/` and `runs/ft2b_gbm/` (all nested under `out_root`,
default `"runs"`).
"""
from __future__ import annotations

import csv
import json
import logging
import os
import os.path as osp

import torch

from src.gbm_ft2b import refit_gbm
from src.train_gnn import train

logger = logging.getLogger(__name__)

DEFAULT_OUT_ROOT = "runs"

# Loss config the FT2b plan mandates for every stage (train()'s own defaults
# are graph="knn", global_weight=0.0 -- both wrong for FT2b, so pinned here
# rather than relying on train()'s defaults ever staying in sync).
_LOSS_DEFAULTS = {"graph": "union", "global_weight": 0.3, "feature_version": 4}

_STAGE1_DATASETS = {"A"}
_STAGE2_DATASETS = {"A", "B", "C"}


def _load_manifest_rows(manifest_path: str) -> dict:
    with open(manifest_path, newline="") as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def _load_splits(features_dir: str) -> dict:
    with open(osp.join(features_dir, "splits.json")) as f:
        return json.load(f)


def _stage1_stems(train_pool, manifest_rows, manifest_path: str = None) -> list:
    """dataset == A (all-sand, no BC/sleeve/chill variation).

    Raises KeyError if any stem in train_pool is missing from manifest_rows.
    """
    missing = [s for s in train_pool if s not in manifest_rows]
    if missing:
        msg = f"Missing from manifest"
        if manifest_path:
            msg += f" {manifest_path}"
        msg += f": {missing}"
        raise KeyError(msg)
    return [s for s in train_pool
            if manifest_rows[s].get("dataset") in _STAGE1_DATASETS]


def _stage2_stems(train_pool, manifest_rows, manifest_path: str = None) -> list:
    """dataset in {A, B, C} (adds chills + sleeve BC variation).

    Raises KeyError if any stem in train_pool is missing from manifest_rows.
    """
    missing = [s for s in train_pool if s not in manifest_rows]
    if missing:
        msg = f"Missing from manifest"
        if manifest_path:
            msg += f" {manifest_path}"
        msg += f": {missing}"
        raise KeyError(msg)
    return [s for s in train_pool
            if manifest_rows[s].get("dataset") in _STAGE2_DATASETS]


def _stage3_stems(train_pool, manifest_rows, manifest_path: str = None) -> list:
    """Named-grade rows (any dataset) or dataset == D (real parts).

    Raises KeyError if any stem in train_pool is missing from manifest_rows.
    """
    missing = [s for s in train_pool if s not in manifest_rows]
    if missing:
        msg = f"Missing from manifest"
        if manifest_path:
            msg += f" {manifest_path}"
        msg += f": {missing}"
        raise KeyError(msg)
    stems = []
    for s in train_pool:
        row = manifest_rows[s]
        if row.get("alloy_family", "synthetic") != "synthetic" or row.get("dataset") == "D":
            stems.append(s)
    return stems


def run_ft2b(features_dir, manifest, sp1_ckpt, out_root: str = DEFAULT_OUT_ROOT,
             epochs: int = 200, lr_stage1: float = 3e-4, lr_stage2: float = 1e-4,
             lr_stage3: float = 1e-4, train_kwargs: dict | None = None,
             gbm_kwargs: dict | None = None) -> dict:
    """Run the 3-stage FT2b staged training, returning a dict of output paths.

    Args:
        features_dir: v4 feature cache directory (must contain `splits.json`
            with `{"interp": {"train": [...], "test": [...]},
            "extrap": {"train": [...], "test": [...]}}`; `interp.train` ==
            `extrap.train`, the single training pool).
        manifest: path to the DOE manifest CSV (`run_id` == stem; needs
            `dataset` and `alloy_family` columns to partition stems by stage).
        sp1_ckpt: path to the production SP1 checkpoint (node_in=13, no
            FiLM) that stage 1 warm-starts from.
        out_root: base directory; stage dirs and the GBM dir are created
            under it (`<out_root>/ft2b_stage{1,2,3}/`, `<out_root>/ft2b_gbm/`).
        epochs: shared epoch budget passed to every `train()` call (early
            stopping/patience are `train()`'s own defaults unless overridden
            via `train_kwargs`).
        lr_stage{1,2,3}: per-stage learning rate.
        train_kwargs: extra kwargs forwarded to every `train()` call (e.g.
            `n_blocks`, `hidden`, `batch_size`, `patience`, `num_workers`,
            `device`) -- must NOT include `lr` (set via `lr_stage*` instead)
            or the stage-specific wiring args (`init_ckpt`, `freeze_backbone`,
            `freeze_mode`, `out_dir`, `early_stop_val_stems`, `film`), which
            this function always sets itself. `graph`/`global_weight`/
            `feature_version` default to the FT2b plan's values but may be
            overridden here.
        gbm_kwargs: extra kwargs forwarded to every `refit_gbm()` call (e.g.
            `max_iter`, `min_samples_leaf`, `holdout_frac`).

    Returns:
        dict with `stage{1,2,3}_dir`, `stage{1,2,3}_ckpt` (stage 3's may be
        `None` if it was skipped -- see module docstring), `final_ckpt` (the
        last stage actually run), `gbm_stage1_path`/`gbm_stage1_metrics`,
        and `gbm_stage3_path`/`gbm_stage3_metrics` (also `None` if stage 3
        was skipped).
    """
    train_kwargs = dict(train_kwargs or {})
    for k, v in _LOSS_DEFAULTS.items():
        train_kwargs.setdefault(k, v)
    train_kwargs.setdefault("epochs", epochs)
    gbm_kwargs = dict(gbm_kwargs or {})

    manifest_rows = _load_manifest_rows(manifest)
    splits = _load_splits(features_dir)
    train_pool_interp = splits["interp"]["train"]
    train_pool_extrap = splits["extrap"]["train"]

    # Guard: interp and extrap train pools must be identical
    if train_pool_interp != train_pool_extrap:
        raise ValueError(
            f"interp['train'] and extrap['train'] must be identical "
            f"(single training pool assumption). Found {len(train_pool_interp)} "
            f"interp and {len(train_pool_extrap)} extrap stems with "
            f"{len(set(train_pool_interp) - set(train_pool_extrap))} only-in-interp "
            f"and {len(set(train_pool_extrap) - set(train_pool_interp))} only-in-extrap.")

    train_pool = train_pool_interp
    val_stems = splits["interp"]["test"]
    early_stop_val_stems = splits["extrap"]["test"]

    gbm_dir = osp.join(out_root, "ft2b_gbm")
    os.makedirs(gbm_dir, exist_ok=True)

    paths: dict = {}

    # ---- Stage 1: scale/alloy adaptation (FiLM only, warm-start from SP1) ----
    stage1_dir = osp.join(out_root, "ft2b_stage1")
    stage1_stems = _stage1_stems(train_pool, manifest_rows, manifest_path=manifest)
    logger.info("FT2b stage 1 (scale/alloy, dataset=A): %d train stems", len(stage1_stems))
    if not stage1_stems:
        raise RuntimeError(
            "FT2b stage 1: 0 train stems resolved for dataset=A in the train "
            "pool -- cannot proceed (check the manifest/splits inputs)")
    stage1_result = train(
        features_dir, stage1_stems, val_stems, out_dir=stage1_dir, lr=lr_stage1,
        film=True, init_ckpt=sp1_ckpt, freeze_backbone=True,
        early_stop_val_stems=early_stop_val_stems, **train_kwargs,
    )
    paths["stage1_dir"] = stage1_dir
    paths["stage1_ckpt"] = stage1_result["ckpt"]
    paths["final_ckpt"] = paths["stage1_ckpt"]

    hgb_path = osp.join(gbm_dir, "hgb.joblib")
    paths["gbm_stage1_metrics"] = refit_gbm(
        features_dir=features_dir, train_stems=stage1_stems,
        manifest_path=manifest, out_path=hgb_path, **gbm_kwargs)
    paths["gbm_stage1_path"] = hgb_path

    # ---- Stage 2: BC generalization (full backbone, chained from stage 1) ----
    stage2_dir = osp.join(out_root, "ft2b_stage2")
    stage2_stems = _stage2_stems(train_pool, manifest_rows, manifest_path=manifest)
    logger.info("FT2b stage 2 (BC, dataset in A/B/C): %d train stems", len(stage2_stems))
    if not stage2_stems:
        raise RuntimeError(
            "FT2b stage 2: 0 train stems resolved for dataset in {A,B,C} in "
            "the train pool -- cannot proceed (check the manifest/splits inputs)")
    stage2_result = train(
        features_dir, stage2_stems, val_stems, out_dir=stage2_dir, lr=lr_stage2,
        film=True, init_ckpt=paths["stage1_ckpt"], freeze_backbone=False,
        early_stop_val_stems=early_stop_val_stems, **train_kwargs,
    )
    paths["stage2_dir"] = stage2_dir
    paths["stage2_ckpt"] = stage2_result["ckpt"]
    paths["final_ckpt"] = paths["stage2_ckpt"]

    # ---- Stage 3: calibration (named-grade + real parts) -- may be empty ----
    stage3_stems = _stage3_stems(train_pool, manifest_rows, manifest_path=manifest)
    logger.info("FT2b stage 3 (calibrate, named-grade + dataset=D): %d train stems",
                len(stage3_stems))
    if not stage3_stems:
        logger.warning(
            "FT2b stage 3 SKIPPED: 0 train stems resolved for named-grade/"
            "dataset-D rows in the train pool (they may be entirely held out "
            "for validation in this manifest/splits, as in the production "
            "FT2a v4 cache). Final checkpoint stays stage 2's: %s",
            paths["stage2_ckpt"])
        paths["stage3_dir"] = None
        paths["stage3_ckpt"] = None
        paths["gbm_stage3_path"] = None
        paths["gbm_stage3_metrics"] = None
        return paths

    stage3_dir = osp.join(out_root, "ft2b_stage3")
    stage3_result = train(
        features_dir, stage3_stems, val_stems, out_dir=stage3_dir, lr=lr_stage3,
        film=True, init_ckpt=paths["stage2_ckpt"], freeze_mode="all_but_last2",
        early_stop_val_stems=early_stop_val_stems, **train_kwargs,
    )
    paths["stage3_dir"] = stage3_dir
    paths["stage3_ckpt"] = stage3_result["ckpt"]
    paths["final_ckpt"] = paths["stage3_ckpt"]

    hgb_stage3_path = osp.join(gbm_dir, "hgb_stage3.joblib")
    paths["gbm_stage3_metrics"] = refit_gbm(
        features_dir=features_dir, train_stems=stage3_stems,
        manifest_path=manifest, out_path=hgb_stage3_path, **gbm_kwargs)
    paths["gbm_stage3_path"] = hgb_stage3_path

    return paths


def run_stage2b(features_dir, manifest, stage1_ckpt, out_dir: str = "runs/ft2b_stage2b",
                 oversample_exo: int = 3, lr: float = 1e-4, epochs: int = 200,
                 train_kwargs: dict | None = None) -> dict:
    """Exo-channel remediation: repeat FT2b stage 2 (BC generalization) on the
    fixed feature cache, oversampling exo-sleeve stems so the (previously
    corrupted) exothermic-sleeve BC channel gets proportionally more gradient
    signal than its raw row count in the train pool would give it.

    Same stem selection as `run_ft2b`'s stage 2 (`_stage2_stems`: dataset in
    {A, B, C}), then the train list actually passed to `train()` is that
    stem list PLUS `(oversample_exo - 1)` extra copies of every stem whose
    manifest row has `bc_config == "exo_sleeve"` (duplicates appended after
    the base list, so ordering is deterministic).

    Chains from `stage1_ckpt` (a stage-1 FT2b checkpoint, NOT the original
    SP1 checkpoint), full backbone unfrozen (`freeze_backbone=False`), same
    as `run_ft2b`'s stage 2. Early-stop/checkpoint-selection uses
    `extrap.test`, same as every other FT2b stage.

    Args:
        features_dir: v4 feature cache directory (must contain `splits.json`
            with `interp.train == extrap.train`, the single training pool).
        manifest: path to the DOE manifest CSV (`run_id` == stem; needs
            `dataset` and `bc_config` columns).
        stage1_ckpt: path to the checkpoint this run warm-starts/chains from.
        out_dir: directory `train()` writes its checkpoint into.
        oversample_exo: total multiplicity for each exo-sleeve stem in the
            train list (1 = no oversampling; the base stem is always
            present once, plus `oversample_exo - 1` duplicates).
        lr: learning rate passed to `train()`.
        epochs: epoch budget passed to `train()` (overridden by
            `train_kwargs["epochs"]` if given there).
        train_kwargs: extra kwargs forwarded to `train()` -- must NOT
            include `lr` or the stage-wiring args (`init_ckpt`,
            `freeze_backbone`, `out_dir`, `early_stop_val_stems`, `film`,
            `val_stems`), which this function always sets itself. `epochs` IS
            overridable via train_kwargs["epochs"] (setdefault).
            `graph`/`global_weight`/`feature_version` default to the FT2b
            plan's values (`_LOSS_DEFAULTS`) but may be overridden here.

    Returns:
        dict with `ckpt` (the trained checkpoint path), `best_val` (from
        `train()`'s return), `n_train` (length of the train list, including
        oversampled duplicates), and `n_exo` (count of distinct exo-sleeve
        stems that were oversampled).
    """
    train_kwargs = dict(train_kwargs or {})
    for k, v in _LOSS_DEFAULTS.items():
        train_kwargs.setdefault(k, v)
    train_kwargs.setdefault("epochs", epochs)

    manifest_rows = _load_manifest_rows(manifest)
    splits = _load_splits(features_dir)
    train_pool_interp = splits["interp"]["train"]
    train_pool_extrap = splits["extrap"]["train"]

    # Guard: interp and extrap train pools must be identical
    if train_pool_interp != train_pool_extrap:
        raise ValueError(
            f"interp['train'] and extrap['train'] must be identical "
            f"(single training pool assumption). Found {len(train_pool_interp)} "
            f"interp and {len(train_pool_extrap)} extrap stems with "
            f"{len(set(train_pool_interp) - set(train_pool_extrap))} only-in-interp "
            f"and {len(set(train_pool_extrap) - set(train_pool_interp))} only-in-extrap.")

    train_pool = train_pool_interp
    val_stems = splits["interp"]["test"]
    early_stop_val_stems = splits["extrap"]["test"]

    stage2_stems = _stage2_stems(train_pool, manifest_rows, manifest_path=manifest)
    exo_stems = [s for s in stage2_stems if manifest_rows[s].get("bc_config") == "exo_sleeve"]
    train_list = list(stage2_stems) + exo_stems * (oversample_exo - 1)
    logger.info(
        "FT2b stage 2b (exo-oversample, dataset in A/B/C): %d train stems "
        "(%d distinct exo-sleeve, oversampled x%d)",
        len(train_list), len(exo_stems), oversample_exo)

    result = train(
        features_dir, train_list, val_stems=val_stems, out_dir=out_dir, lr=lr,
        film=True, init_ckpt=stage1_ckpt, freeze_backbone=False,
        early_stop_val_stems=early_stop_val_stems, **train_kwargs,
    )

    return {
        "ckpt": result["ckpt"],
        "best_val": result["best_val"],
        "n_train": len(train_list),
        "n_exo": len(exo_stems),
    }


def run_ft3(features_dir, manifest, base_ckpt: str = "runs/ft2b_stage2c/ckpt_ep054.pt",
            out_dir: str = "runs/ft3_stage1", oversample_e: int = 3, lr: float = 5e-5,
            epochs: int = 200, train_kwargs: dict | None = None) -> dict:
    """FT3: frozen-norm fine-tune of the production v4 checkpoint onto the new
    dataset-E (multi-body assembly) rows, mixed into the existing FT2b stage-2
    train pool (dataset in {A, B, C}) with dataset-E stems oversampled.

    Loads `base_ckpt`'s own `norm` (the production checkpoint's frozen
    feature/target normalization -- recomputing norm_stats from the mixed
    A+B+C+E pool would shift every standardization the checkpoint was
    trained with) and forwards it to `train()` as `norm_override`, so every
    SolidGraphDataset construction inside `train()` (train/val/early-stop-val)
    standardizes against the SAME stats the checkpoint already expects.

    Train stems = `_stage2_stems` (A+B+C ∩ train pool, same selection as
    FT2b stage 2/2b) PLUS dataset-E stems in the train pool, each repeated
    `oversample_e` times. Unlike `run_stage2b`'s exo-sleeve subset (already
    present once inside its A/B/C stage-2 selection, so it tops up with
    `oversample_exo - 1` extra copies), dataset "E" is NOT one of
    `_stage2_stems`'s selected datasets ({A, B, C}) -- so E stems are
    entirely new content here, added as a straight `x oversample_e` (not
    `x (oversample_e - 1)`).

    Chains from `base_ckpt` (`init_ckpt=base_ckpt`), full backbone unfrozen
    (`freeze_backbone=False` -- FT3 fine-tunes the whole model, not just
    FiLM), `film=True`, `save_every_epoch=True` (per-epoch checkpoints for
    gate-metric checkpoint selection, mirroring the exo-fix stage-2c
    precedent). Early-stop/checkpoint-selection uses `extrap.test`, same as
    every other FT2b/FT3 stage; the `run_ft2b`/`run_stage2b` interp==extrap
    train-pool guard applies here too (Task 4's `extend_splits_with_e` keeps
    it true by extending both pools equally with E-train stems).

    Args:
        features_dir: v4 feature cache directory (`splits.json` with
            `interp.train == extrap.train`, the single training pool,
            extended with dataset-E train stems by Task 4's
            `extend_splits_with_e`).
        manifest: path to the DOE manifest CSV (`run_id` == stem; needs a
            `dataset` column, with `"E"` rows for the new assembly castings).
        base_ckpt: the production checkpoint this run warm-starts/chains
            from AND whose `norm` is frozen for training (default: the FT2b
            exo-fix production checkpoint, `runs/ft2b_stage2c/ckpt_ep054.pt`).
        out_dir: directory `train()` writes its checkpoints into.
        oversample_e: total multiplicity for each dataset-E train stem in the
            train list (1 = no oversampling).
        lr: learning rate passed to `train()`.
        epochs: epoch budget passed to `train()` (overridden by
            `train_kwargs["epochs"]` if given there).
        train_kwargs: extra kwargs forwarded to `train()` -- must NOT include
            `lr`, `norm_override`, or the stage-wiring args (`init_ckpt`,
            `freeze_backbone`, `out_dir`, `early_stop_val_stems`, `film`,
            `val_stems`, `save_every_epoch`), which this function always sets
            itself. `epochs` IS overridable via `train_kwargs["epochs"]`
            (setdefault); same for `batch_size` (default 2 to prevent OOM: train()'s
            own default batch_size=8 OOMs the RTX 3060 12GB, incident 2026-07-31 FT3
            execution) and `num_workers` (default 4). Explicit caller values in
            `train_kwargs` override all defaults. `graph`/`global_weight`/
            `feature_version` default to the FT2b plan's values (`_LOSS_DEFAULTS`)
            but may be overridden here.

    Returns:
        dict with `ckpt` (the trained checkpoint path), `best_val` (from
        `train()`'s return), `n_train` (length of the train list, including
        oversampled E duplicates), and `n_e` (count of distinct dataset-E
        stems that were oversampled).
    """
    train_kwargs = dict(train_kwargs or {})
    train_kwargs.setdefault("batch_size", 2)
    train_kwargs.setdefault("num_workers", 4)
    for k, v in _LOSS_DEFAULTS.items():
        train_kwargs.setdefault(k, v)
    train_kwargs.setdefault("epochs", epochs)

    base_norm = torch.load(base_ckpt, map_location="cpu")["norm"]

    manifest_rows = _load_manifest_rows(manifest)
    splits = _load_splits(features_dir)
    train_pool_interp = splits["interp"]["train"]
    train_pool_extrap = splits["extrap"]["train"]

    # Guard: interp and extrap train pools must be identical
    if train_pool_interp != train_pool_extrap:
        raise ValueError(
            f"interp['train'] and extrap['train'] must be identical "
            f"(single training pool assumption). Found {len(train_pool_interp)} "
            f"interp and {len(train_pool_extrap)} extrap stems with "
            f"{len(set(train_pool_interp) - set(train_pool_extrap))} only-in-interp "
            f"and {len(set(train_pool_extrap) - set(train_pool_interp))} only-in-extrap.")

    train_pool = train_pool_interp
    val_stems = splits["interp"]["test"]
    early_stop_val_stems = splits["extrap"]["test"]

    stage2_stems = _stage2_stems(train_pool, manifest_rows, manifest_path=manifest)
    e_train_stems = [s for s in train_pool if manifest_rows[s].get("dataset") == "E"]
    train_list = list(stage2_stems) + e_train_stems * oversample_e
    logger.info(
        "FT3 (assembly fine-tune, dataset A/B/C + E oversampled x%d): %d train "
        "stems (%d distinct dataset-E)", oversample_e, len(train_list), len(e_train_stems))

    result = train(
        features_dir, train_list, val_stems=val_stems, out_dir=out_dir, lr=lr,
        film=True, init_ckpt=base_ckpt, freeze_backbone=False,
        early_stop_val_stems=early_stop_val_stems, save_every_epoch=True,
        norm_override=base_norm, **train_kwargs,
    )

    return {
        "ckpt": result["ckpt"],
        "best_val": result["best_val"],
        "n_train": len(train_list),
        "n_e": len(e_train_stems),
    }


# Loss config for bc-reach (Task 3): same graph/global_weight convention as
# FT2b/FT3, but `feature_version=5` (the 20-column BC-reach cache, Task 1/2)
# instead of FT2b/FT3's 4.
_BCR_LOSS_DEFAULTS = {"graph": "union", "global_weight": 0.3, "feature_version": 5}


def run_bcr(features_dir, manifest, base_ckpt: str = "runs/ft3_stage1/ckpt_ep021.pt",
            out_dir: str = "runs/bcr_stage1", oversample_e: int = 3, lr: float = 5e-5,
            epochs: int = 200, train_kwargs: dict | None = None) -> dict:
    """bc-reach: warm-started fine-tune of the production v4 checkpoint onto
    the v5 (20-column, BC-reach) feature cache -- node_in 15 -> 20.

    Same stem assembly as `run_ft3` (train stems = `_stage2_stems`, A+B+C ∩
    train pool, PLUS dataset-E train-pool stems each repeated `oversample_e`
    times), the same `film=True`/`freeze_backbone=False`/`save_every_epoch=True`
    wiring, and the same `extrap.test` early-stop/checkpoint-selection
    convention as every other FT2b/FT3/bc-reach stage.

    UNLIKE `run_ft3`, this does NOT pass `norm_override`: `run_ft3` freezes
    the fine-tune onto its `base_ckpt`'s own (15-wide) norm because it keeps
    `node_in` unchanged (15 -> 15). Here `node_in` genuinely changes (15 ->
    20, Task 1/2's 5 new BC-reach columns), so `base_ckpt`'s 15-wide norm
    stats cannot standardize the v5 cache's 20-wide `X` at all -- `train()`
    is left to load `features_dir`'s OWN fresh `norm_stats.npz` (Task 2's
    `build_v5` output), the default `norm_override=None` behavior.

    `train()`'s own `init_ckpt` dispatch (see its docstring) routes this
    warm-start through `src.warm_start.build_or_load_v5` instead of
    `build_or_load_v4`, because the actual v5 cache data has `node_in=20`:
    `base_ckpt` (`node_in=15`, film=True -- e.g. the production
    `ckpt_ep021.pt`) has its node encoder's first Linear rebuilt for
    `in_features=20` (existing 15 columns' weights copied, the 5 new
    trailing columns zero-init, bias copied verbatim); every other weight
    (edge_encoder, processor blocks, FiLM, decoder, scale_head) is
    strict-copied. The resulting checkpoint's `cfg` therefore carries
    `node_in=20` (computed by `train()` from the actual data, same as every
    other stage) and `feature_version=5` (via `_BCR_LOSS_DEFAULTS` below).

    Args:
        features_dir: v5 feature cache directory (`splits.json` with
            `interp.train == extrap.train`, the single training pool --
            Task 2's `build_v5` copies this verbatim from the v4 cache, so
            it already carries the same E-train stems FT3's
            `extend_splits_with_e` added).
        manifest: path to the DOE manifest CSV (`run_id` == stem; needs a
            `dataset` column, with `"E"` rows for the assembly castings).
        base_ckpt: the v4 (node_in=15, film=True) checkpoint this run warm-
            starts from (default: the production FT3 checkpoint,
            `runs/ft3_stage1/ckpt_ep021.pt`).
        out_dir: directory `train()` writes its checkpoints into.
        oversample_e: total multiplicity for each dataset-E train stem in the
            train list (1 = no oversampling), same convention as `run_ft3`.
        lr: learning rate passed to `train()` (default `5e-5`, per the
            bc-reach plan).
        epochs: epoch budget passed to `train()` (overridden by
            `train_kwargs["epochs"]` if given there); `train()`'s own
            `patience=25` default (checked against `extrap.test`, via
            `early_stop_val_stems`) is left untouched, per the bc-reach plan
            ("epochs<=200 patience 25 on extrap.test").
        train_kwargs: extra kwargs forwarded to `train()` -- must NOT
            include `lr`, `norm_override`, or the stage-wiring args
            (`init_ckpt`, `freeze_backbone`, `out_dir`,
            `early_stop_val_stems`, `film`, `val_stems`, `save_every_epoch`),
            which this function always sets itself. `epochs` IS overridable
            via `train_kwargs["epochs"]` (setdefault); same for `batch_size`
            (default 2, mirroring FT3's OOM-safe pin) and `num_workers`
            (default 4). Explicit caller values in `train_kwargs` override
            all defaults. `graph`/`global_weight`/`feature_version` default
            to `_BCR_LOSS_DEFAULTS` but may be overridden here. Every other
            disallowed key above fails loudly on its own (a plain `TypeError`
            from `train()`'s call below, "got multiple values for keyword
            argument") since this function also passes it explicitly;
            `norm_override` has no such corresponding explicit call kwarg to
            collide with (this function never passes it at all), so it is
            guarded by an explicit `ValueError` instead (review round 1, F1).

    Returns:
        dict with `ckpt` (the trained checkpoint path), `best_val` (from
        `train()`'s return), `n_train` (length of the train list, including
        oversampled E duplicates), and `n_e` (count of distinct dataset-E
        stems that were oversampled).
    """
    train_kwargs = dict(train_kwargs or {})
    train_kwargs.setdefault("batch_size", 2)
    train_kwargs.setdefault("num_workers", 4)
    for k, v in _BCR_LOSS_DEFAULTS.items():
        train_kwargs.setdefault(k, v)
    train_kwargs.setdefault("epochs", epochs)
    # F1 (review round 1): every OTHER pinned kwarg (init_ckpt, freeze_backbone,
    # out_dir, early_stop_val_stems, film, val_stems, save_every_epoch, lr) already
    # fails loudly if train_kwargs duplicates it -- train()'s call below raises a
    # plain TypeError ("got multiple values for keyword argument"). norm_override
    # is the ONE pinned divergence from run_ft3 (this driver deliberately never
    # passes it -- v5's fresh norm_stats.npz, not the 15-wide base_ckpt's, must be
    # what train() standardizes against) that has NO corresponding call kwarg to
    # collide with, so it would otherwise slip through train_kwargs silently and
    # freeze the wrong (15-wide, base_ckpt) normalization onto the v5 run. Guard
    # it explicitly, matching the loud-fail convention every other pinned kwarg
    # already gets for free.
    if "norm_override" in train_kwargs:
        raise ValueError(
            "run_bcr: train_kwargs must not include 'norm_override' -- v5 uses "
            "its own fresh norm_stats.npz (Task 2's build_v5), never a frozen "
            "norm from base_ckpt (unlike run_ft3's frozen-norm fine-tune).")

    manifest_rows = _load_manifest_rows(manifest)
    splits = _load_splits(features_dir)
    train_pool_interp = splits["interp"]["train"]
    train_pool_extrap = splits["extrap"]["train"]

    # Guard: interp and extrap train pools must be identical
    if train_pool_interp != train_pool_extrap:
        raise ValueError(
            f"interp['train'] and extrap['train'] must be identical "
            f"(single training pool assumption). Found {len(train_pool_interp)} "
            f"interp and {len(train_pool_extrap)} extrap stems with "
            f"{len(set(train_pool_interp) - set(train_pool_extrap))} only-in-interp "
            f"and {len(set(train_pool_extrap) - set(train_pool_interp))} only-in-extrap.")

    train_pool = train_pool_interp
    val_stems = splits["interp"]["test"]
    early_stop_val_stems = splits["extrap"]["test"]

    stage2_stems = _stage2_stems(train_pool, manifest_rows, manifest_path=manifest)
    e_train_stems = [s for s in train_pool if manifest_rows[s].get("dataset") == "E"]
    train_list = list(stage2_stems) + e_train_stems * oversample_e
    logger.info(
        "bc-reach (node_in 15->20, dataset A/B/C + E oversampled x%d): %d train "
        "stems (%d distinct dataset-E)", oversample_e, len(train_list), len(e_train_stems))

    result = train(
        features_dir, train_list, val_stems=val_stems, out_dir=out_dir, lr=lr,
        film=True, init_ckpt=base_ckpt, freeze_backbone=False,
        early_stop_val_stems=early_stop_val_stems, save_every_epoch=True,
        **train_kwargs,
    )

    return {
        "ckpt": result["ckpt"],
        "best_val": result["best_val"],
        "n_train": len(train_list),
        "n_e": len(e_train_stems),
    }
