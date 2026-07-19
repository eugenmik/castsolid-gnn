import os

import numpy as np
import pyvista as pv
import pytest
from src.viz import (load_prediction, time_range, liquid_region, hotspot_points,
                     points_above_time, solidification_animation)


def _make_vtu(path):
    # 5 points, 2 tetra sharing face (1,2,3); solidification_time gradient 0..40
    pts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10], [10, 10, 10]], float)
    cells = np.hstack([[4, 0, 1, 2, 3], [4, 1, 2, 3, 4]])
    ctypes = np.array([pv.CellType.TETRA, pv.CellType.TETRA])
    g = pv.UnstructuredGrid(cells, ctypes, pts)
    g.point_data["solidification_time"] = np.array([0, 10, 20, 30, 40], float)
    g.point_data["hotspot"] = np.array([0, 0, 0, 0, 1], np.int32)
    g.save(path)
    return g


def test_load_and_time_range(tmp_path):
    p = str(tmp_path / "c.vtu"); _make_vtu(p)
    g = load_prediction(p)
    assert "solidification_time" in g.point_data and "hotspot" in g.point_data
    assert time_range(g) == (0.0, 40.0)


def test_load_rejects_missing_arrays(tmp_path):
    p = str(tmp_path / "bad.vtu")
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    g = pv.UnstructuredGrid(np.hstack([[4, 0, 1, 2, 3]]), np.array([pv.CellType.TETRA]), pts)
    g.save(p)
    with pytest.raises(ValueError):
        load_prediction(p)


def test_liquid_region_thresholds(tmp_path):
    p = str(tmp_path / "c.vtu"); _make_vtu(p)
    g = load_prediction(p)
    assert liquid_region(g, 0.0).n_cells == g.n_cells       # nothing below tmin -> all kept
    assert liquid_region(g, 41.0).n_cells == 0              # above tmax -> empty
    mid = liquid_region(g, 25.0)
    assert 0 < mid.n_points <= g.n_points
    assert float(mid.point_data["solidification_time"].max()) == 40.0


def test_hotspot_points(tmp_path):
    p = str(tmp_path / "c.vtu"); _make_vtu(p)
    g = load_prediction(p)
    hp = hotspot_points(g)
    assert hp.n_points == 1
    np.testing.assert_allclose(hp.points[0], [10, 10, 10])


def test_points_above_time(tmp_path):
    p = str(tmp_path / "c.vtu"); _make_vtu(p)
    g = load_prediction(p)
    # times are 0, 10, 20, 30, 40 -> cutoff 25 keeps the two points >= 25
    above = points_above_time(g, 25.0)
    assert above.n_points == 2
    np.testing.assert_allclose(sorted(above.point_data["solidification_time"]),
                               [30.0, 40.0])
    # cutoff above the max -> nothing left
    assert points_above_time(g, 41.0).n_points == 0
    # cutoff at the min -> everything kept
    assert points_above_time(g, 0.0).n_points == g.n_points


def test_solidification_animation_renders_gif(tmp_path):
    p = str(tmp_path / "c.vtu"); _make_vtu(p)
    g = load_prediction(p)
    out = str(tmp_path / "anim.gif")
    result = solidification_animation(g, out, n_frames=3, fps=5)
    assert result == out
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_solidification_animation_with_camera_position(tmp_path):
    # Explicit camera_position (as captured from a live view) should be
    # accepted in place of reset_camera(), and rendering should still succeed.
    p = str(tmp_path / "c.vtu"); _make_vtu(p)
    g = load_prediction(p)
    out = str(tmp_path / "anim_cam.gif")
    cam = [(10, 10, 10), (0, 0, 0), (0, 0, 1)]
    result = solidification_animation(g, out, n_frames=3, fps=5, camera_position=cam)
    assert result == out
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
