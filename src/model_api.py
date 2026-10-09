"""Mesh-in, field-out entry point for the published v5.1 model.

    from src.model_api import CastSolidModel
    model = CastSolidModel()                       # loads weights/ once
    seconds = model.predict("part.msh", alloy="G20Mn5")

`predict` takes a gmsh 2.2 ASCII tetrahedral mesh (path or a parsed
`src.msh_parse.Mesh`) and returns the solidification time in seconds for every
node, in the mesh's node order. Getting from a STEP file to such a mesh (CAD
healing, density-targeted meshing) is the job of `src.predict_casting`.
"""
from __future__ import annotations

import os

import numpy as np

from src.inference_props import resolve_props
from src.msh_parse import Mesh, parse_msh
from src.predict_casting import DEFAULT_W_STD, DEFAULT_W_TOTAL, load_hybrid, predict_field


class CastSolidModel:
    def __init__(self, weights_dir: str = "weights", device: str | None = None):
        w = lambda name: os.path.join(weights_dir, name)
        self.device = device
        (self.gnn, self.gbm_total, self.norm, self.cfg, _,
         self.gbm_std) = load_hybrid(w("gnn_v5.1.safetensors"), w("gbm_total_v3.skops"),
                                     device, hgb_std_path=w("gbm_std_v2.skops"))
        self.device = device or next(self.gnn.parameters()).device.type

    def predict(self, mesh: str | Mesh, alloy: str | None = None, mold: str = "green_sand",
                props_json: str | None = None) -> np.ndarray:
        """Per-node solidification time (s) for `mesh`.

        `alloy`/`mold` name rows of `data/properties/` (default EN-GJS-400-15
        in green sand); `props_json` replaces both with a custom property set.
        """
        if not isinstance(mesh, Mesh):
            mesh = parse_msh(os.fspath(mesh))
        props = resolve_props(alloy=alloy, mold=mold, props_json=props_json)
        seconds, _ = predict_field(mesh, self.gnn, self.gbm_total, self.norm, cfg=self.cfg,
                                   device=self.device, props=props, hgb_std=self.gbm_std,
                                   w_total=DEFAULT_W_TOTAL, w_std=DEFAULT_W_STD)
        return np.asarray(seconds)

    @staticmethod
    def hotspots(seconds: np.ndarray, top_frac: float = 0.1) -> np.ndarray:
        """Boolean mask of the `top_frac` latest-solidifying nodes (the same
        rule as the `hotspot` field `src.predict_casting.write_vtu` writes)."""
        seconds = np.asarray(seconds, dtype=np.float32)
        k = max(1, int(round(seconds.shape[0] * top_frac)))
        return seconds >= np.sort(seconds)[-k]
