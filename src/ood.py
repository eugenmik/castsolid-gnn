"""Out-of-distribution size check: compare a casting's volume, bounding-box
diagonal and largest distance to cooling with the training maxima
(`weights/ood_ranges.json`) and warn when any of them is exceeded."""
import json


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
