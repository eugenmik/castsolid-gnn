"""PyG dataset over the v2/v3 feature cache. Builds either a kNN graph or the
SP1 union graph (mesh edges union kNN, `graph="union"`, read from the cache's
`mesh_edges`) with mm edge attributes on load; optional random-rotation
augmentation. Features and the log1p target are standardized with the cache's
global stats."""
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.graph_build import (knn_edge_index, edge_attr, random_rotation_matrix, rotate,
                             union_edge_index, union_edge_attr)
from src.dataload import load_norm


class SolidGraphDataset(Dataset):
    def __init__(self, features_dir, stems, k=16, augment=False, norm=None, graph="knn"):
        self.dir = features_dir
        self.stems = list(stems)
        self.k = k
        self.augment = augment
        self.graph = graph
        self.norm = norm if norm is not None else load_norm(features_dir)
        self.fmean = torch.tensor(self.norm["feat_mean"], dtype=torch.float32)
        self.fstd = torch.tensor(self.norm["feat_std"], dtype=torch.float32)
        self.ym = float(self.norm["y_log_mean"]); self.ys = float(self.norm["y_log_std"])

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        d = np.load(osp.join(self.dir, "graphs", f"{self.stems[i]}.npz"))
        x = (torch.tensor(d["X"], dtype=torch.float32) - self.fmean) / self.fstd
        pos = torch.tensor(d["pos"], dtype=torch.float32)
        y_log = torch.log1p(torch.tensor(d["y"], dtype=torch.float32))
        y = (y_log - self.ym) / self.ys
        g = torch.tensor(d["g"], dtype=torch.float).reshape(1, -1) if "g" in d.files else None
        if self.augment:
            pos = rotate(pos, random_rotation_matrix())
        if self.graph == "union":
            if "mesh_edges" not in d.files:
                raise KeyError(f"{self.stems[i]}: cache lacks mesh_edges — "
                               f"rebuild with build_dataset --feature-version 3")
            me = torch.tensor(d["mesh_edges"], dtype=torch.long)
            ei, is_mesh = union_edge_index(pos, me, k=self.k)
            ea = union_edge_attr(pos, ei, is_mesh)
        else:
            ei = knn_edge_index(pos, k=self.k)
            ea = edge_attr(pos, ei)
        return Data(x=x, edge_index=ei, edge_attr=ea, y=y, pos=pos, g=g)
