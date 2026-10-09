"""CastSolidModel: loading from a folder or the Hub, and predicting on a mesh."""
import os
import shutil

import numpy as np
import pytest

import src.model_api as api
from src.model_api import CastSolidModel


def test_hotspots_matches_top_fraction():
    t = np.arange(100, dtype=np.float32)
    mask = CastSolidModel.hotspots(t)
    assert mask.sum() == 10 and mask[-10:].all()


def test_predict_from_local_folder(tiny_weights_dir, mini_msh_file):
    model = CastSolidModel(tiny_weights_dir, device="cpu")
    assert model.version == "test" and model.w_std == 4.0
    t_iron = model.predict(mini_msh_file)
    t_steel = model.predict(mini_msh_file, alloy="G20Mn5")
    assert t_iron.shape == t_steel.shape == (5,)
    assert np.all(np.isfinite(t_iron)) and np.all(t_iron >= 0)
    assert not np.allclose(t_iron, t_steel)


def test_from_pretrained_downloads_config_and_listed_files(tiny_weights_dir, tmp_path,
                                                            monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    requested = []

    def fake_download(repo_id, filename, revision=None, cache_dir=None):
        requested.append((repo_id, filename, revision))
        shutil.copy(os.path.join(tiny_weights_dir, filename), snapshot / filename)
        return str(snapshot / filename)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    model = CastSolidModel.from_pretrained(device="cpu")
    assert requested[0] == (api.HF_REPO_ID, "config.json", api.HF_REVISION)
    assert {f for _, f, _ in requested} == {"config.json", "gnn.safetensors",
                                            "total.skops", "spread.skops", "ood.json"}
    assert model.weights_dir == str(snapshot)


@pytest.mark.skipif(not os.environ.get("CASTSOLID_HUB_TESTS"),
                    reason="set CASTSOLID_HUB_TESTS=1 to download the published weights")
def test_published_weights_load_and_predict(mini_msh_file):
    model = CastSolidModel.from_pretrained(device="cpu")
    assert model.version == "v5.1"
    t = model.predict(mini_msh_file)
    assert t.shape == (5,) and np.all(np.isfinite(t))
