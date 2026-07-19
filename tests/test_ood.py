import numpy as np
from src.ood import build_ood_ranges, ood_flags, save_ranges, load_ranges


def _cache(tmp_path):
    g = tmp_path / "graphs"; g.mkdir(parents=True)
    # FEATURE_NAMES index: volume_mm3=6, geodesic_d_cool=0
    for name, vol, gc, extent in [("A", 100.0, 5.0, 10.0), ("B", 400.0, 9.0, 40.0)]:
        N = 8
        X = np.zeros((N, 9), np.float32); X[:, 6] = vol; X[:, 0] = np.linspace(0, gc, N)
        pos = np.zeros((N, 3), np.float32); pos[-1] = extent   # bbox diag ~ extent*sqrt3
        np.savez(g / f"{name}.npz", X=X, y=np.ones(N, np.float32), pos=pos)
    return tmp_path


def test_build_and_flags(tmp_path):
    root = _cache(tmp_path)
    r = build_ood_ranges(str(root))
    assert abs(r["volume_mm3_max"] - 400.0) < 1e-6
    assert abs(r["geodesic_d_cool_max"] - 9.0) < 1e-6
    assert r["bbox_diag_max"] > 40.0    # sqrt(3)*40

    # inside range -> no warnings
    assert ood_flags(300.0, 50.0, 8.0, r) == []
    # volume far above max -> a warning mentioning volume
    w = ood_flags(10000.0, 50.0, 8.0, r)
    assert len(w) == 1 and "volume" in w[0].lower()


def test_roundtrip(tmp_path):
    r = {"volume_mm3_max": 1.0, "bbox_diag_max": 2.0, "geodesic_d_cool_max": 3.0}
    p = tmp_path / "r.json"; save_ranges(r, str(p))
    assert load_ranges(str(p)) == r
