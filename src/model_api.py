"""Mesh in, solidification-time field out.

    from src.model_api import CastSolidModel
    model = CastSolidModel()                       # loads weights/ once
    seconds = model.predict("part.msh", alloy="G20Mn5")

`predict` takes a gmsh 2.2 ASCII tetrahedral mesh (a path or a parsed
`src.msh_parse.Mesh`) whose cooling surface is tagged "Cooling", and returns
the solidification time in seconds for every node, in the mesh's node order.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from src import gbm_heads
from src.features import alloy_g_vector, node_features
from src.hybrid import hybrid_field_soft
from src.inference_props import DEFAULT_MOLD, resolve_props
from src.model_gnn import load_gnn, predict_graph_seconds
from src.msh_parse import Mesh, parse_msh
from src.release_io import load_gbm, load_gnn_state

# Recomposition weights of the published model (see src/hybrid.py).
W_TOTAL = 1.0
W_STD = 4.0

GNN_FILE = "gnn_v5.1.safetensors"
TOTAL_HEAD_FILE = "gbm_total_v3.skops"
SPREAD_HEAD_FILE = "gbm_std_v2.skops"


class CastSolidModel:
    def __init__(self, weights_dir: str = "weights", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = load_gnn_state(os.path.join(weights_dir, GNN_FILE), self.device)
        self.gnn, self.cfg, self.norm = load_gnn(state, self.device)
        self.total_head = load_gbm(os.path.join(weights_dir, TOTAL_HEAD_FILE))
        self.spread_head = load_gbm(os.path.join(weights_dir, SPREAD_HEAD_FILE))
        gbm_heads.check_head(self.total_head, gbm_heads.N_TOTAL_FEATURES, "total-time")
        gbm_heads.check_head(self.spread_head, gbm_heads.N_SPREAD_FEATURES, "spread")

    def predict(self, mesh: str | Mesh, alloy: str | None = None, mold: str = DEFAULT_MOLD,
                props_json: str | None = None) -> np.ndarray:
        """Per-node solidification time (s).

        `alloy`/`mold` name rows of `data/properties/` (default EN-GJS-400-15
        in green sand); `props_json` replaces both with your own values."""
        props = resolve_props(alloy=alloy, mold=mold, props_json=props_json)
        return self.predict_with_props(mesh, props)[0]

    def predict_with_props(self, mesh: str | Mesh, props: dict) -> tuple[np.ndarray, dict]:
        """(seconds, details) for an already-resolved property dict.

        `details` holds the node feature matrix `X`, the heads' `total_s` and
        `spread_s`, and the recomposition diagnostics `recomposition`."""
        if not isinstance(mesh, Mesh):
            mesh = parse_msh(os.fspath(mesh))
        feat = node_features(mesh, props["mold_effusivity"])
        X = feat["X"]
        gnn_seconds = predict_graph_seconds(self.gnn, self.cfg, self.norm, X, feat["pos"],
                                            feat["mesh_edges"], alloy_g_vector(props),
                                            device=self.device)
        row = gbm_heads.total_row(props, X)
        total = gbm_heads.predict_total_seconds(self.total_head, row)
        spread = gbm_heads.predict_spread_seconds(self.spread_head,
                                                  gbm_heads.spread_row(row, X))
        seconds, info = hybrid_field_soft(gnn_seconds, total, spread,
                                          w_total=W_TOTAL, w_std=W_STD)
        return seconds, {"X": X, "total_s": total, "spread_s": spread,
                         "recomposition": info}

    @staticmethod
    def hotspots(seconds: np.ndarray, top_frac: float = 0.1) -> np.ndarray:
        """Boolean mask of the `top_frac` latest-solidifying nodes."""
        seconds = np.asarray(seconds, dtype=np.float32)
        k = max(1, int(round(seconds.shape[0] * top_frac)))
        return seconds >= np.sort(seconds)[-k]
