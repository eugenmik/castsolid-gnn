"""Reprocess raw .pt + paired .msh into cached per-node feature files.

Writes one .npz per casting under <data_root>/features/graphs/ plus a global
meta.npz and norm_stats.npz. Parallelised across CPU cores.
"""
import os
import os.path as osp
import re
import sys
from glob import glob

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))
from src.msh_parse import parse_msh
from src.features import compute_features, FEATURE_NAMES

_FACE_RE = re.compile(r'_[+-][XYZ]')
_MSH_SETS = ("msh_none", "msh_half", "msh_all")

# Canonical raw temperature backup (dolfinx-DOF-order snapshots_nodal), on the
# large Media drive. Used when data/pt/*.pt has been deleted to free space.
# The .pt temps are byte-identical to these npz (verified). Override if moved.
RAW_NPZ_DIRS = ("/home/eugen/Media/MIGRATION/Debian/casting_gnn/data/raw",)


def find_msh(stem: str, data_root: str = "data"):
    for s in _MSH_SETS:
        hits = glob(osp.join(data_root, s, "*", f"{stem}.msh"))
        if hits:
            return hits[0]
    return None


def _load_raw_temps(stem: str, data_root: str = "data"):
    """Return (temps_dof (T,N) float32, times (T,) float32) or None.

    Prefers data/pt/<stem>.pt; falls back to the canonical raw npz backup so the
    pipeline still rebuilds after data/pt is deleted."""
    ptp = osp.join(data_root, "pt", f"{stem}.pt")
    if osp.exists(ptp):
        raw = torch.load(ptp, weights_only=False)
        return raw["temps"].numpy().astype(np.float32), raw["times"].numpy().astype(np.float32)
    for d in RAW_NPZ_DIRS:
        hits = glob(osp.join(d, "*", f"{stem}.npz"))
        if hits:
            z = np.load(hits[0])
            return z["snapshots_nodal"].astype(np.float32), z["times"].astype(np.float32)
    return None


def process_one(stem: str, data_root: str = "data"):
    mpath = find_msh(stem, data_root)
    if mpath is None:
        print(f"  WARN missing msh for {stem}")
        return None
    src = _load_raw_temps(stem, data_root)
    if src is None:
        print(f"  WARN no raw temps for {stem} (.pt or npz backup)")
        return None
    temps_dof, times = src
    mesh = parse_msh(mpath)
    n_pt = temps_dof.shape[1]
    if mesh.node_xyz.shape[0] != n_pt:
        print(f"  WARN node count mismatch {stem}: msh={mesh.node_xyz.shape[0]} raw={n_pt}")
        return None

    # Apply the node-ordering recovery permutation: the raw temps are in dolfinx
    # DOF order, which is scrambled relative to the gmsh node order used by pos/
    # mesh labels. perm g maps DOF->gmsh node: temps_gmsh[:, g] = temps_dof.
    # Files without a validated perm are excluded (their per-node target is
    # unrecoverable). See docs spec §1.5.
    perm_path = osp.join(data_root, "features", "perms", f"{stem}.npy")
    if not osp.exists(perm_path):
        print(f"  WARN no recovery perm for {stem} (excluded)")
        return None
    g = np.load(perm_path)
    if g.shape[0] != n_pt:
        print(f"  WARN perm/temps size mismatch {stem}: perm={g.shape[0]} temps_N={n_pt}")
        return None
    temps = np.empty_like(temps_dof)
    temps[:, g] = temps_dof                 # reorder into gmsh node order

    feat = compute_features(mesh, temps, times)

    n_adiab = len(_FACE_RE.findall(stem))
    prefix = stem.split("_")[0]
    return {
        "stem": stem,
        "X": feat["X"],
        "y": feat["y"],
        "pos": feat["pos"],
        "y_global": feat["y_global"],
        "prefix": prefix,
        "n_adiab": n_adiab,
        "is_fully_cooled": n_adiab == 0,
        "n_nodes": n_pt,
    }


import argparse
from concurrent.futures import ProcessPoolExecutor


def write_record(rec, out_dir):
    gdir = osp.join(out_dir, "graphs")
    os.makedirs(gdir, exist_ok=True)
    np.savez(osp.join(gdir, f"{rec['stem']}.npz"), X=rec["X"], y=rec["y"], pos=rec["pos"])


def _all_stems(data_root):
    """Enumerate casting stems. Prefer data/pt; after it is deleted, fall back to
    the recovery perms (the exact recoverable set), then the raw npz backup."""
    stems = [osp.splitext(osp.basename(p))[0]
             for p in glob(osp.join(data_root, "pt", "*.pt"))]
    if not stems:
        stems = [osp.splitext(osp.basename(p))[0]
                 for p in glob(osp.join(data_root, "features", "perms", "*.npy"))]
    if not stems:
        for d in RAW_NPZ_DIRS:
            stems = [osp.splitext(osp.basename(p))[0]
                     for p in glob(osp.join(d, "*", "*.npz"))]
            if stems:
                break
    return sorted(stems)


def build(data_root="data", out_dir="data/features", n_workers=None, limit=None):
    stems = _all_stems(data_root)
    if limit:
        stems = stems[:limit]
    os.makedirs(out_dir, exist_ok=True)

    # accumulators for global normalization (Welford-free: sum, sumsq, count)
    fsum = np.zeros(len(FEATURE_NAMES)); fsq = np.zeros(len(FEATURE_NAMES)); fcnt = 0
    ylsum = ylsq = ycnt = 0.0
    meta = {k: [] for k in ("stem", "prefix", "n_adiab", "is_fully_cooled",
                            "y_global", "n_nodes")}

    def _accumulate(rec):
        nonlocal fsum, fsq, fcnt, ylsum, ylsq, ycnt
        write_record(rec, out_dir)
        X = rec["X"].astype(np.float64)
        fsum += X.sum(0); fsq += (X ** 2).sum(0); fcnt += X.shape[0]
        yl = np.log1p(rec["y"].astype(np.float64))
        ylsum += yl.sum(); ylsq += (yl ** 2).sum(); ycnt += yl.size
        for k in meta:
            meta[k].append(rec[k])

    worker = process_one
    if n_workers == 1:
        for i, s in enumerate(stems):
            rec = worker(s, data_root)
            if rec is not None:
                _accumulate(rec)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(stems)}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(worker, s, data_root): s for s in stems}
            done = 0
            for fut in futs:
                rec = fut.result()
                done += 1
                if rec is not None:
                    _accumulate(rec)
                if done % 500 == 0:
                    print(f"  {done}/{len(stems)}")

    feat_mean = fsum / max(fcnt, 1)
    feat_std = np.sqrt(np.clip(fsq / max(fcnt, 1) - feat_mean ** 2, 1e-12, None))
    y_log_mean = ylsum / max(ycnt, 1)
    y_log_std = float(np.sqrt(max(ylsq / max(ycnt, 1) - y_log_mean ** 2, 1e-12)))

    np.savez(osp.join(out_dir, "meta.npz"),
             stem=np.array(meta["stem"]),
             prefix=np.array(meta["prefix"]),
             n_adiab=np.array(meta["n_adiab"]),
             is_fully_cooled=np.array(meta["is_fully_cooled"]),
             y_global=np.array(meta["y_global"], dtype=np.float32),
             n_nodes=np.array(meta["n_nodes"]),
             feature_names=np.array(FEATURE_NAMES))
    np.savez(osp.join(out_dir, "norm_stats.npz"),
             feat_mean=feat_mean, feat_std=feat_std,
             y_log_mean=np.float64(y_log_mean), y_log_std=np.float64(y_log_std))
    print(f"Built {len(meta['stem'])} graphs → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="data/features")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build(a.data_root, a.out_dir, a.workers, a.limit)
