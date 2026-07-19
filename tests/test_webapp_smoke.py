import numpy as np
from trame.app import get_server

from src.webapp import build_app, summary_text


def test_summary_text_formats_result():
    r = {"name": "part", "total_time_s": 42.5, "n_nodes": 1000,
         "hotspot_xyz": np.array([[1.0, 2.0, 3.0]]), "ood_warnings": ["too big"]}
    s = summary_text(r)
    assert "42.5" in s and "part" in s and "too big" in s


def test_build_app_constructs_without_serving():
    server = build_app(get_server("nc_test_smoke", client_type="vue3"))  # must NOT block / start a server
    # trame server exposes reactive state; our initial keys must exist
    assert server.state.tau is not None
    for key in ("tau", "tmin", "tmax", "show_hotspots", "summary", "loaded"):
        assert key in server.state.to_dict()
    assert server.state.loaded is False


def test_build_app_animation_state_and_controller():
    server = build_app(get_server("nc_test_smoke_anim", client_type="vue3"))
    state_dict = server.state.to_dict()
    for key in ("animating", "anim_ready", "anim_uri", "anim_name"):
        assert key in state_dict
    assert server.state.anim_ready is False
    assert server.state.anim_uri == ""
    assert callable(server.controller.generate_animation)
