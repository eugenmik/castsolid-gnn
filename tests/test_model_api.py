"""CastSolidModel: mesh in, per-node seconds out (needs the published weights)."""
import os

import numpy as np
import pytest

from src.model_api import CastSolidModel

WEIGHTS = "weights/gnn_v5.1.safetensors"


def test_hotspots_matches_top_fraction():
    t = np.arange(100, dtype=np.float32)
    mask = CastSolidModel.hotspots(t)
    assert mask.sum() == 10 and mask[-10:].all()


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="published weights not present")
def test_predict_on_mesh(mini_msh_file):
    model = CastSolidModel(device="cpu")
    t_iron = model.predict(mini_msh_file)
    t_steel = model.predict(mini_msh_file, alloy="G20Mn5")
    assert t_iron.shape == t_steel.shape and t_iron.ndim == 1
    assert np.all(np.isfinite(t_iron)) and np.all(t_iron >= 0)
    assert not np.allclose(t_iron, t_steel)
