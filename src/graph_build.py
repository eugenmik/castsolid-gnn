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
