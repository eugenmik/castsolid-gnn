"""Out-of-distribution size check: compare a casting's scale quantities against
the training-set maxima so predictions beyond the training envelope are flagged."""
import json
import os.path as osp
from glob import glob

import numpy as np

from src.features import FEATURE_NAMES

_VOL = FEATURE_NAMES.index("volume_mm3")
_GC = FEATURE_NAMES.index("geodesic_d_cool")


def build_ood_ranges(features_dir: str) -> dict:
    vol_max = bbox_max = gc_max = 0.0
    for p in glob(osp.join(features_dir, "graphs", "*.npz")):
        d = np.load(p)
        X, pos = d["X"], d["pos"]
        vol_max = max(vol_max, float(X[0, _VOL]))
        gc_max = max(gc_max, float(X[:, _GC].max()))
        diag = float(np.linalg.norm(pos.max(0) - pos.min(0)))
        bbox_max = max(bbox_max, diag)
    return {"volume_mm3_max": vol_max, "bbox_diag_max": bbox_max,
            "geodesic_d_cool_max": gc_max}


def save_ranges(ranges: dict, path: str):
    with open(path, "w") as f:
        json.dump(ranges, f, indent=2)


def load_ranges(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def ood_flags(volume_mm3, bbox_diag, geodesic_d_cool_max, ranges, tol=1.05) -> list:
    w = []
    if volume_mm3 > ranges["volume_mm3_max"] * tol:
        w.append(f"casting volume {volume_mm3:.0f} mm^3 exceeds training max "
                 f"{ranges['volume_mm3_max']:.0f} mm^3 — prediction unreliable")
    if bbox_diag > ranges["bbox_diag_max"] * tol:
        w.append(f"bounding-box diagonal {bbox_diag:.0f} mm exceeds training max "
                 f"{ranges['bbox_diag_max']:.0f} mm — prediction unreliable")
    if geodesic_d_cool_max > ranges["geodesic_d_cool_max"] * tol:
        w.append(f"max distance-to-cooling {geodesic_d_cool_max:.1f} mm exceeds training max "
                 f"{ranges['geodesic_d_cool_max']:.1f} mm — prediction unreliable")
    return w
