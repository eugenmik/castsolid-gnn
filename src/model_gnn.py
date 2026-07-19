"""MeshGraphNet-style encode-process-decode GNN predicting standardized
log1p(solidification_time) per node; global scalar = max over nodes."""
import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class ProcessorBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.edge_mlp = _mlp(3 * h, h, h)
        self.edge_norm = nn.LayerNorm(h)
        self.node_mlp = _mlp(2 * h, h, h)
        self.node_norm = nn.LayerNorm(h)

    def forward(self, hnode, edge_index, e):
        src, dst = edge_index
        e_new = self.edge_norm(self.edge_mlp(torch.cat([hnode[src], hnode[dst], e], dim=-1)) + e)
        agg = scatter(e_new, dst, dim=0, dim_size=hnode.shape[0], reduce="mean")
        h_new = self.node_norm(self.node_mlp(torch.cat([hnode, agg], dim=-1)) + hnode)
        return h_new, e_new


class SolidGNN(nn.Module):
    def __init__(self, node_in=9, edge_in=4, hidden=128, n_blocks=12):
        super().__init__()
        self.node_encoder = nn.Sequential(_mlp(node_in, hidden, hidden), nn.LayerNorm(hidden))
        self.edge_encoder = nn.Sequential(_mlp(edge_in, hidden, hidden), nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList([ProcessorBlock(hidden) for _ in range(n_blocks)])
        self.decoder = _mlp(hidden, hidden // 2, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for blk in self.blocks:
            h, e = blk(h, edge_index, e)
        y_nodal = self.decoder(h).squeeze(-1)
        y_global = scatter(y_nodal, batch, dim=0, reduce="max")
        return y_nodal, y_global
