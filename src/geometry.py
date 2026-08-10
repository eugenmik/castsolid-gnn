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


def geodesic_diameter_two_sweep(n_nodes: int, edges: np.ndarray,
                                edge_len: np.ndarray, start: int = 0) -> float:
    """Two-sweep (double-Dijkstra) approximation of the mesh's geodesic
    diameter, over the SAME mesh-edge graph `geodesic_distance` already uses
    (not a per-BC-patch quantity -- patch-independent, through-metal, a
    single per-part constant): sweep 1 finds the farthest node from `start`;
    sweep 2 finds the farthest node from THAT node; the sweep-2 distance is
    returned as D_geo. This is the standard double-sweep heuristic for graph
    diameter (exact on trees, a lower bound in general, but a good practical
    estimate for a roughly part-shaped mesh graph).

    `start` is a fixed node INDEX (default 0), not searched or randomized --
    deterministic and reproducible for the same mesh (same node ordering).

    Returns 0.0 (never NaN/inf) for an edgeless mesh, or if either sweep's
    farthest reachable node has zero finite distance (degenerate/single-node
    or fully-disconnected mesh) -- callers must treat 0.0 as "no valid
    part-normalized length available" rather than a real zero-size part.

    CAVEAT (bc-reach re-review N2): the two-sweep heuristic is a LOWER
    BOUND on the true graph diameter in general (exact only on trees), and
    is start-sensitive -- on a BUKSA-like real casting mesh, re-running
    with different random start nodes measured ~15% spread in the reported
    D_geo across starts. Fixing `start=0` makes the result perfectly
    DETERMINISTIC for a given mesh (bit-identical across repeated calls,
    verified in tests/test_geometry.py), which is what the cache-build and
    inference paths both rely on (the same mesh must always yield the same
    D_geo) -- but it does NOT make the estimate stable under a RE-MESH of
    the same part (a different tetrahedralization can shift which node
    lands at index 0, and therefore the estimate, by roughly that same
    ~15%). Closing that gap would need iterating sweeps to convergence
    (repeat from the new farthest node until the distance stops growing)
    or a multi-start-and-take-max scheme -- deliberately NOT done here:
    the part-normalized reach channels only need a stable-per-mesh
    O(part-size) scale, not a tight diameter bound, and re-mesh stability
    was judged out of scope for this task."""
    if edges.shape[0] == 0 or n_nodes == 0:
        return 0.0
    start = int(min(max(start, 0), n_nodes - 1))
    d1 = geodesic_distance(n_nodes, edges, edge_len, np.array([start], dtype=np.int64))
    finite1 = np.isfinite(d1)
    if not finite1.any():
        return 0.0
    far1 = int(np.flatnonzero(finite1)[np.argmax(d1[finite1])])
    d2 = geodesic_distance(n_nodes, edges, edge_len, np.array([far1], dtype=np.int64))
    finite2 = d2[np.isfinite(d2)]
    return float(finite2.max()) if finite2.size else 0.0


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


_IDW_K = 4             # neighbors for inverse-distance-weighted subsample fill
_IDW_EPS = 1e-9        # distance floor to avoid divide-by-zero


def _idw_geodesic_fill(vals: np.ndarray, dist_from_computed: np.ndarray,
                       missing: np.ndarray, bbox_diag: float,
                       k: int = _IDW_K) -> np.ndarray:
    """Inverse-distance-weighted fill for `missing` nodes from the values
    (`vals`, one row per computed node) at the computed/query nodes, using
    geodesic (mesh-edge) distances (`dist_from_computed`, shape
    (n_computed, N)) to the k nearest computed nodes. Smoother than a
    single-nearest-node copy for a continuously-varying field. Nodes
    unreachable from every computed node (inf distance -- disconnected mesh
    component) fall back to the bbox-diagonal sentinel."""
    n_computed = vals.shape[0]
    k_eff = max(1, min(k, n_computed))
    sub = dist_from_computed[:, missing]                        # (n_computed, n_missing)
    part = np.argpartition(sub, k_eff - 1, axis=0)[:k_eff]       # (k_eff, n_missing)
    d = np.take_along_axis(sub, part, axis=0)                   # (k_eff, n_missing)
    finite = np.isfinite(d)
    w = np.where(finite, 1.0 / np.maximum(d, _IDW_EPS), 0.0)     # (k_eff, n_missing)
    wsum = w.sum(axis=0)                                        # (n_missing,)
    reachable = wsum > 0
    w_norm = w / np.where(reachable, wsum, 1.0)                 # (k_eff, n_missing)
    gathered = vals[part]                                       # (k_eff, n_missing, R)
    result = np.einsum('kn,knr->nr', w_norm, gathered)          # (n_missing, R)
    result[~reachable] = bbox_diag
    return result


def local_modulus(node_xyz: np.ndarray, tets: np.ndarray, cool_tri: np.ndarray,
                  edges: np.ndarray, edge_len: np.ndarray, radii,
                  chunk: int = 128, exact: bool = True, subsample=None) -> np.ndarray:
    """Local cooling modulus per node: V/A over geodesic balls (mm).

    For each node and radius r: (sum of volumes of tets whose 4 nodes all lie
    within geodesic distance r of the node) / (area of cooling triangles whose
    3 nodes all lie within r). Balls with no cooling area get the bbox-diagonal
    sentinel (consistent with the other distance sentinels).

    `exact=True` (default): the reference per-node computation -- byte-identical
    to the original (pre-subsample) implementation regardless of `subsample`.

    `exact=False` with `subsample=k` (int): compute the exact ball modulus only
    on every k-th node ("computed" nodes), then fill every other node by
    inverse-distance-weighted (IDW) interpolation over its `_IDW_K` (=4)
    nearest computed nodes, measured by mesh-edge geodesic distance
    (multi-source Dijkstra over `edges`/`edge_len`, reusing `_edge_csr`; see
    `_idw_geodesic_fill`). This trades a small amount of accuracy for roughly
    a `k`-fold cut in the number of expensive geodesic-ball evaluations (the
    dense-mesh bottleneck). If `subsample >= N` (or is None/<=1) this
    degenerates to the exact path.

    HONEST ACCURACY NOTE (see tests/test_local_modulus.py::
    test_subsample_per_radius_column_error and task-2-report.md for the full
    sweep): at the production radii (5/15/40 mm) and subsample=4, per-radius-
    column median relative error on a real ~4k-node training mesh (casting-
    scale bbox) is roughly 0% at r=5mm but ~25-30% at r=15mm and ~6% at
    r=40mm -- i.e. the mid radius does NOT meet a 5% target. This is because
    local_modulus is a ball-membership STEP FUNCTION of geodesic position
    (a tet/triangle is either fully inside a ball or not), not a smoothly-
    varying field -- averaging several dissimilar neighbors (larger IDW k)
    measurably makes the mid-radius error WORSE, not better (single-nearest,
    k=1, beats IDW k=4-8 there); k=4 is used as the best compromise within
    the requested range, not because it minimizes error in isolation.
    CONSEQUENCE: this subsample path is intended for DENSE-mesh inference
    (FT3), where it is a useful order-of-magnitude speed/accuracy trade
    against an otherwise-prohibitive `exact` cost -- it is NOT a validated
    drop-in replacement for `exact=True` in the training-density feature
    cache (FT2a), where `exact` is already fast enough (seconds per mesh)
    and should stay the default. Do not switch `build_dataset`'s v4 feature
    cache to `exact=False` without re-validating per-radius accuracy on the
    actual dense meshes it will be used on."""
    N = node_xyz.shape[0]
    radii = np.asarray(radii, dtype=float)
    bbox_diag = float(np.linalg.norm(node_xyz.max(0) - node_xyz.min(0)))
    out = np.full((N, len(radii)), bbox_diag, dtype=np.float64)
    if edges.shape[0] == 0 or tets.shape[0] == 0:
        return out.astype(np.float32)
    tvol = tet_volumes(node_xyz, tets)
    tarea = triangle_areas(node_xyz, cool_tri)
    g = _edge_csr(N, edges, edge_len)

    use_subsample = (not exact) and subsample is not None and 1 < subsample < N
    query_idx = (np.arange(0, N, subsample, dtype=np.int64) if use_subsample
                 else np.arange(N, dtype=np.int64))

    vals = _local_modulus_for_query(tets, cool_tri, radii, chunk, query_idx, g,
                                    tvol, tarea, bbox_diag)
    out[query_idx] = vals

    if use_subsample and query_idx.shape[0] < N:
        missing = np.setdiff1d(np.arange(N), query_idx, assume_unique=True)
        if missing.shape[0]:
            # inverse-distance-weighted fill from the k nearest computed
            # nodes, along mesh-edge geodesics (smoother than single-nearest)
            dist_from_computed = dijkstra(g, directed=False, indices=query_idx,
                                          min_only=False)              # (M, N)
            out[missing] = _idw_geodesic_fill(vals, dist_from_computed, missing, bbox_diag)

    return out.astype(np.float32)
