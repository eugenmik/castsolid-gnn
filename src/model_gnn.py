"""The graph network: an encode-process-decode GNN in the style of
MeshGraphNets, conditioned on the alloy/mold vector through FiLM.

It predicts the standardized log1p solidification time of every node. The
graph is the union of the tetrahedral mesh edges and each node's k nearest
neighbours; an edge carries its displacement vector (mm), its length and a
flag telling whether it is a mesh edge.
"""
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph
from torch_geometric.utils import scatter, to_undirected


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class ProcessorBlock(nn.Module):
    """One message-passing step with residual edge and node updates."""

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


class FiLM(nn.Module):
    """g -> per-channel (gamma, beta); applied as h * (1 + gamma) + beta."""

    def __init__(self, g_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(g_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2 * hidden))

    def forward(self, h, g_nodes):
        gamma, beta = self.net(g_nodes).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class SolidGNN(nn.Module):
    """12 processor blocks (hidden width 128 in the published model), FiLM
    after each block, and a two-head decoder: a per-node deviation head and
    a graph-level scale head on the mean node embedding."""

    def __init__(self, node_in=15, edge_in=5, hidden=128, n_blocks=12, g_dim=5):
        super().__init__()
        self.node_encoder = nn.Sequential(_mlp(node_in, hidden, hidden), nn.LayerNorm(hidden))
        self.edge_encoder = nn.Sequential(_mlp(edge_in, hidden, hidden), nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList([ProcessorBlock(hidden) for _ in range(n_blocks)])
        self.decoder = _mlp(hidden, hidden // 2, 1)
        self.scale_head = _mlp(hidden, hidden // 2, 1)
        self.films = nn.ModuleList([FiLM(g_dim, hidden) for _ in range(n_blocks)])

    def forward(self, x, edge_index, edge_attr, batch, g):
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        g_nodes = g[batch]
        for i, blk in enumerate(self.blocks):
            h, e = blk(h, edge_index, e)
            h = self.films[i](h, g_nodes)
        y_nodal = self.decoder(h).squeeze(-1)
        dev = y_nodal - scatter(y_nodal, batch, dim=0, reduce="mean")[batch]
        scale = self.scale_head(scatter(h, batch, dim=0, reduce="mean")).squeeze(-1)
        return dev + scale[batch]


def knn_edge_index(pos: torch.Tensor, k: int = 16) -> torch.Tensor:
    ei = knn_graph(pos, k=k, loop=False)      # directed k-nearest
    return to_undirected(ei)                   # symmetric


def union_edge_index(pos: torch.Tensor, mesh_edges: torch.Tensor, k: int = 16):
    """Undirected union of the mesh edges and kNN edges, deduplicated.

    mesh_edges: (M, 2) long tensor of unique node pairs (any orientation).
    Returns (edge_index (2, E) long, is_mesh (E,) float32 flag)."""
    n = pos.shape[0]
    knn = knn_edge_index(pos, k=k).to(torch.long)
    me = mesh_edges.t().to(torch.long)
    me = torch.cat([me, me.flip(0)], dim=1)                    # both directions
    mesh_key = me[0] * n + me[1]
    keys = torch.unique(torch.cat([mesh_key, knn[0] * n + knn[1]]))
    ei = torch.stack([keys // n, keys % n])
    is_mesh = torch.isin(keys, mesh_key).to(torch.float32)
    return ei, is_mesh


def union_edge_attr(pos: torch.Tensor, edge_index: torch.Tensor,
                    is_mesh: torch.Tensor) -> torch.Tensor:
    """(E, 5): displacement (3), length, mesh-edge flag."""
    src, dst = edge_index
    d = pos[dst] - pos[src]
    return torch.cat([d, d.norm(dim=1, keepdim=True), is_mesh.unsqueeze(1)], dim=1)


def load_gnn(state: dict, device):
    """Build a `SolidGNN` from a `{model, norm, cfg}` dict (see
    `src.release_io.load_gnn_state`). Returns (model, cfg, norm)."""
    cfg, norm = state["cfg"], state["norm"]
    model = SolidGNN(node_in=cfg["node_in"], edge_in=cfg["edge_in"], hidden=cfg["hidden"],
                     n_blocks=cfg["n_blocks"], g_dim=cfg.get("g_dim", 5)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg, norm


def predict_graph_seconds(model, cfg, norm, X, pos_np, mesh_edges, g_vec,
                          device="cpu") -> np.ndarray:
    """Per-node solidification time (s) predicted by the GNN alone.

    `X` is the raw (N, 15) feature matrix, `g_vec` the raw (5,) alloy/mold
    vector; standardization uses the statistics stored with the weights."""
    fmean = torch.tensor(norm["feat_mean"], dtype=torch.float32)
    fstd = torch.tensor(norm["feat_std"], dtype=torch.float32)
    x = (torch.tensor(X, dtype=torch.float32) - fmean) / fstd
    pos = torch.tensor(pos_np, dtype=torch.float32)
    me = torch.tensor(mesh_edges, dtype=torch.long)
    ei, is_mesh = union_edge_index(pos, me, k=cfg["k"])
    ea = union_edge_attr(pos, ei, is_mesh)
    zeros = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    g = torch.tensor(np.asarray(g_vec), dtype=torch.float32).reshape(1, -1).to(device)
    with torch.no_grad():
        yn = model(x.to(device), ei.to(device), ea.to(device), zeros, g)
    pred_log = yn.cpu().numpy() * norm["y_log_std"] + norm["y_log_mean"]
    return np.clip(np.expm1(pred_log), 0.0, None)
