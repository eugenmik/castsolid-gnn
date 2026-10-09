import json

from src.ood import load_ranges, ood_flags

RANGES = {"volume_mm3_max": 400.0, "bbox_diag_max": 69.3, "geodesic_d_cool_max": 9.0}


def test_load_ranges(tmp_path):
    p = tmp_path / "ood.json"
    p.write_text(json.dumps(RANGES))
    assert load_ranges(str(p)) == RANGES


def test_flags_only_beyond_tolerance():
    assert ood_flags(400.0, 69.3, 9.0, RANGES) == []
    assert ood_flags(410.0, 70.0, 9.3, RANGES) == []          # within the 5% tolerance
    w = ood_flags(1000.0, 200.0, 30.0, RANGES)
    assert len(w) == 3
    assert "volume" in w[0] and "bounding-box" in w[1] and "distance-to-cooling" in w[2]
