import numpy as np
import pyvista as pv
from trame.app import get_server

from src.webapp import build_app, render_prediction

SCALAR_BAR_TITLE = "Solidification Time, s"


def _make_grid():
    # Same tiny synthetic grid shape as tests/test_viz.py: 5 points, 2 tetra
    # sharing a face, solidification_time gradient 0..40.
    pts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10], [10, 10, 10]], float)
    cells = np.hstack([[4, 0, 1, 2, 3], [4, 1, 2, 3, 4]])
    ctypes = np.array([pv.CellType.TETRA, pv.CellType.TETRA])
    g = pv.UnstructuredGrid(cells, ctypes, pts)
    g.point_data["solidification_time"] = np.array([0, 10, 20, 30, 40], float)
    g.point_data["hotspot"] = np.array([0, 0, 0, 0, 1], np.int32)
    return g


def test_scalar_bar_lifecycle_across_render_clear_render():
    server = build_app(get_server("nc_test_render", client_type="vue3"))
    app = server.app
    grid = _make_grid()

    # First load -> exactly one scalar bar, titled correctly.
    render_prediction(app, grid, "summary one")
    plotter = app["plotter"]
    assert SCALAR_BAR_TITLE in plotter.scalar_bars
    assert len(plotter.scalar_bars.keys()) == 1

    # Clear -> no scalar bar left behind.
    server.controller.clear_casting()
    assert SCALAR_BAR_TITLE not in plotter.scalar_bars
    assert len(plotter.scalar_bars.keys()) == 0

    # Second load after Clear -> legend comes back (this is the regression:
    # it used to be lost permanently after a Clear + reload cycle).
    render_prediction(app, grid, "summary two")
    assert SCALAR_BAR_TITLE in plotter.scalar_bars
    assert len(plotter.scalar_bars.keys()) == 1
    assert server.state.loaded is True
    assert server.state.summary == "summary two"


def test_tau_change_does_not_duplicate_or_drop_scalar_bar():
    server = build_app(get_server("nc_test_render", client_type="vue3"))
    app = server.app
    grid = _make_grid()
    render_prediction(app, grid, "summary")
    plotter = app["plotter"]

    # Scrub the tau slider a few times; the ghost actor (not touched by tau)
    # must remain the sole owner of the one scalar bar.
    for tau in (10.0, 20.0, 30.0, 0.0):
        server.state.tau = tau
        server.state.flush()
        assert SCALAR_BAR_TITLE in plotter.scalar_bars
        assert len(plotter.scalar_bars.keys()) == 1
