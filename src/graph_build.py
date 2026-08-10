"""Mesh-agnostic graph construction: kNN edges with millimetre edge attributes,
and random 3D-rotation augmentation (physics is rotation-invariant)."""
import torch
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected


def knn_edge_index(pos: torch.Tensor, k: int = 16) -> torch.Tensor:
    ei = knn_graph(pos, k=k, loop=False)      # directed k-nearest
    return to_undirected(ei)                   # symmetric


def edge_attr(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    d = pos[dst] - pos[src]                     # (E,3) mm
    return torch.cat([d, d.norm(dim=1, keepdim=True)], dim=1)   # (E,4)


def random_rotation_matrix(seed: int | None = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    # QR of a random Gaussian matrix gives a Haar-random orthogonal matrix
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))       # fix signs
    if torch.det(q) < 0:                        # ensure proper rotation (det +1)
        q[:, 0] = -q[:, 0]
    return q


def rotate(pos: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return pos @ R.to(pos.device, pos.dtype).t()


def union_edge_index(pos: torch.Tensor, mesh_edges: torch.Tensor, k: int = 16):
    """Undirected union of tet-mesh edges and kNN edges, deduplicated.

    mesh_edges: (M,2) long tensor of unique node pairs (any orientation).
    Returns (edge_index (2,E) long, is_mesh (E,) float32 flag)."""
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
    return torch.cat([edge_attr(pos, edge_index), is_mesh.unsqueeze(1)], dim=1)  # (E,5)
