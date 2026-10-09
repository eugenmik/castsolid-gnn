"""Pure-geometry helpers: triangle areas, tet volumes, nearest-surface
distances (via KD-tree), and boundary-node masks. All lengths in mm."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def triangle_areas(node_xyz: np.ndarray, tri: np.ndarray) -> np.ndarray:
    if tri.shape[0] == 0:
        return np.zeros(0)
    a = node_xyz[tri[:, 0]]
    b = node_xyz[tri[:, 1]]
    c = node_xyz[tri[:, 2]]
    cross = np.cross(b - a, c - a)
    return 0.5 * np.linalg.norm(cross, axis=1)


def total_area(node_xyz: np.ndarray, tri: np.ndarray) -> float:
    return float(triangle_areas(node_xyz, tri).sum())


def tet_volumes(node_xyz: np.ndarray, tets: np.ndarray) -> np.ndarray:
    if tets.shape[0] == 0:
        return np.zeros(0)
    a = node_xyz[tets[:, 0]]
    b = node_xyz[tets[:, 1]]
    c = node_xyz[tets[:, 2]]
    d = node_xyz[tets[:, 3]]
    return np.abs(np.einsum('ij,ij->i', b - a, np.cross(c - a, d - a))) / 6.0


def mesh_volume(node_xyz: np.ndarray, tets: np.ndarray) -> float:
    if tets.shape[0] == 0:
        return 0.0
    return float(tet_volumes(node_xyz, tets).sum())


def nearest_distance(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to the nearest target point.
    Returns +inf for every point when there are no targets."""
    if targets.shape[0] == 0:
        return np.full(points.shape[0], np.inf)
    tree = cKDTree(targets)
    d, _ = tree.query(points, k=1)
    return d


def boundary_node_mask(n_nodes: int, tri: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_nodes, dtype=bool)
    if tri.shape[0]:
        mask[np.unique(tri.reshape(-1))] = True
    return mask


_TET_EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def edges_from_tets(tets: np.ndarray) -> np.ndarray:
    if tets.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)
    e = np.concatenate([tets[:, [a, b]] for a, b in _TET_EDGE_PAIRS], axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0).astype(np.int64)


def _edge_csr(n_nodes: int, edges: np.ndarray, edge_len: np.ndarray):
    r = np.concatenate([edges[:, 0], edges[:, 1]])
    c = np.concatenate([edges[:, 1], edges[:, 0]])
    w = np.concatenate([edge_len, edge_len])
    return coo_matrix((w, (r, c)), shape=(n_nodes, n_nodes)).tocsr()


def geodesic_distance(n_nodes: int, edges: np.ndarray, edge_len: np.ndarray,
                      source_idx: np.ndarray) -> np.ndarray:
    if source_idx.shape[0] == 0 or edges.shape[0] == 0:
        return np.full(n_nodes, np.inf)
    g = _edge_csr(n_nodes, edges, edge_len)
    d = dijkstra(g, directed=False, indices=source_idx, min_only=True)
    return d




def _local_modulus_for_query(tets: np.ndarray, cool_tri: np.ndarray, radii: np.ndarray,
                             chunk: int, query_idx: np.ndarray, g, tvol: np.ndarray,
                             tarea: np.ndarray, bbox_diag: float) -> np.ndarray:
    """Exact V/A geodesic-ball modulus, computed only for the nodes in
    `query_idx` (as Dijkstra sources). Returns (len(query_idx), len(radii))."""
    rmax = float(radii.max())
    vals = np.full((query_idx.shape[0], len(radii)), bbox_diag, dtype=np.float64)
    for start in range(0, query_idx.shape[0], chunk):
        sel = query_idx[start:start + chunk]
        dmat = dijkstra(g, directed=False, indices=sel, limit=rmax)   # (c, N)
        for j, r in enumerate(radii):
            in_ball = dmat <= r                                        # (c, N)
            vol = in_ball[:, tets].all(axis=2) @ tvol                  # (c,)
            area = (in_ball[:, cool_tri].all(axis=2) @ tarea
                    if cool_tri.shape[0] else np.zeros(len(sel)))
            vals[start:start + len(sel), j] = np.where(
                area > 0, vol / np.maximum(area, 1e-12), bbox_diag)
    return vals




def local_modulus(node_xyz: np.ndarray, tets: np.ndarray, cool_tri: np.ndarray,
                  edges: np.ndarray, edge_len: np.ndarray, radii,
                  chunk: int = 128) -> np.ndarray:
    """Local cooling modulus per node: V/A over geodesic balls (mm).

    For each node and radius r: (sum of volumes of tets whose 4 nodes all lie
    within geodesic distance r of the node) / (area of cooling triangles whose
    3 nodes all lie within r). Balls with no cooling area get the bbox-diagonal
    sentinel (consistent with the other distance sentinels)."""
    N = node_xyz.shape[0]
    radii = np.asarray(radii, dtype=float)
    bbox_diag = float(np.linalg.norm(node_xyz.max(0) - node_xyz.min(0)))
    out = np.full((N, len(radii)), bbox_diag, dtype=np.float64)
    if edges.shape[0] == 0 or tets.shape[0] == 0:
        return out.astype(np.float32)
    tvol = tet_volumes(node_xyz, tets)
    tarea = triangle_areas(node_xyz, cool_tri)
    g = _edge_csr(N, edges, edge_len)
    query_idx = np.arange(N, dtype=np.int64)
    out[query_idx] = _local_modulus_for_query(tets, cool_tri, radii, chunk, query_idx, g,
                                              tvol, tarea, bbox_diag)
    return out.astype(np.float32)
