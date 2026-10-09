"""predict_casting orchestration with meshing and the model replaced by fakes,
plus an end-to-end run on a synthetic STEP with the published weights
(enabled with CASTSOLID_HUB_TESTS=1)."""
import os
import shutil

import meshio
import numpy as np
import pytest

import src.predict_casting as pc
from src.features import FEATURE_NAMES
from src.msh_parse import parse_msh


class _FakeModel:
    version = "v5.1"
    ood_ranges_path = None

    def predict_with_props(self, mesh, props):
        n = mesh.node_xyz.shape[0]
        X = np.zeros((n, len(FEATURE_NAMES)), np.float32)
        X[:, FEATURE_NAMES.index("volume_mm3")] = 1000.0 / 6.0
        seconds = np.arange(n, dtype=np.float64) + 1.0
        info = {"a": 1.0, "m": 0.0, "target_unreachable": False, "clip_frac": 0.0,
                "total_deviation_frac": 0.0}
        return seconds, {"X": X, "total_s": float(n), "spread_s": 1.0, "recomposition": info}


def _fake_get_or_mesh(mini_msh_file, calls):
    def _fn(step_path, out_msh, target_nodes=None, cache_dir=None):
        calls.append(target_nodes)
        shutil.copy(mini_msh_file, out_msh)
        return {"ok": True, "n_nodes": 5, "cache": "miss"}
    return _fn


def test_writes_vtu_and_summary(monkeypatch, mini_msh_file, tmp_path):
    calls = []
    monkeypatch.setattr(pc, "get_or_mesh", _fake_get_or_mesh(mini_msh_file, calls))
    r = pc.predict_casting("part.step", out_dir=str(tmp_path), model=_FakeModel(),
                           ood_json="missing.json")
    assert calls == [pc.DEFAULT_TARGET_NODES]
    assert r["total_time_s"] == 5.0 and r["n_nodes"] == 5
    assert r["hotspot_xyz"].shape == (5, 3)
    vtu = meshio.read(r["vtu"])
    assert {"solidification_time", "hotspot", "niyama", "porosity_risk",
            "macro_region_id"} <= set(vtu.point_data)
    text = open(r["summary"]).read()
    assert "total solidification time: 5.0 s" in text
    assert "alloy: EN-GJS-400-15  mold: green_sand  model: v5.1" in text
    assert "shrinkage screening:" in text
    assert r["shrinkage"]["niyama_source"] == "proxy"


def test_bad_alloy_fails_before_meshing(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("meshing must not run")
    monkeypatch.setattr(pc, "get_or_mesh", boom)
    with pytest.raises(ValueError, match="unknown alloy"):
        pc.predict_casting("part.step", out_dir=str(tmp_path), alloy="nope",
                           model=_FakeModel())


def test_no_shrinkage_and_ood_warning(monkeypatch, mini_msh_file, tmp_path):
    monkeypatch.setattr(pc, "get_or_mesh", _fake_get_or_mesh(mini_msh_file, []))
    ood = tmp_path / "ood.json"
    ood.write_text('{"volume_mm3_max": 1.0, "bbox_diag_max": 1000.0, '
                   '"geodesic_d_cool_max": 1000.0}')
    r = pc.predict_casting("part.step", out_dir=str(tmp_path), model=_FakeModel(),
                           screen_shrinkage=False, ood_json=str(ood))
    assert r["shrinkage"] is None
    assert len(r["ood_warnings"]) == 1 and "volume" in r["ood_warnings"][0]
    assert "porosity_risk" not in meshio.read(r["vtu"]).point_data


def test_elem_size_and_target_nodes_are_exclusive(tmp_path):
    with pytest.raises(ValueError):
        pc.predict_casting("part.step", out_dir=str(tmp_path), elem_size=4.0,
                           target_nodes=9000, model=_FakeModel())


def test_parse_gravity():
    assert pc._parse_gravity("0, 0, -1") == (0.0, 0.0, -1.0)
    for bad in ("1,2", "0,0,0", "a,b,c"):
        with pytest.raises(ValueError):
            pc._parse_gravity(bad)


def test_runs_with_a_local_weights_folder(monkeypatch, mini_msh_file, tiny_weights_dir,
                                         tmp_path):
    monkeypatch.setattr(pc, "get_or_mesh", _fake_get_or_mesh(mini_msh_file, []))
    r = pc.predict_casting("part.step", out_dir=str(tmp_path), weights_dir=tiny_weights_dir,
                           device="cpu")
    assert r["ood_warnings"] == [] and r["total_time_s"] > 0
    assert "model: test" in open(r["summary"]).read()


@pytest.mark.skipif(not os.environ.get("CASTSOLID_HUB_TESTS"),
                    reason="set CASTSOLID_HUB_TESTS=1 to download the published weights")
def test_end_to_end_on_a_box(tmp_path):
    import gmsh
    step = tmp_path / "box.step"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 120, 80, 40)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step))
    finally:
        gmsh.finalize()
    r = pc.predict_casting(str(step), out_dir=str(tmp_path / "out"),
                           cache_dir=str(tmp_path / "cache"), device="cpu")
    t = meshio.read(r["vtu"]).point_data["solidification_time"]
    assert np.all(np.isfinite(t)) and t.min() >= 0 and r["total_time_s"] > 0
    assert set(parse_msh(r["msh"]).tris) == {"Cooling"}
    # the hottest point of a box sits near its centre
    assert np.linalg.norm(r["hotspot_xyz"][0] - np.array([60, 40, 20])) < 25
