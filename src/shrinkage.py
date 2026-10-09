"""Shrinkage screening on a predicted solidification-time field.

Pure numpy/scipy, working on node positions `pos` (mm), tetrahedra `tets` and
the per-node solidification time `t_s` (s). Four outputs:

- A Niyama-style criterion. With only `t_s` available, the identity
  grad(T) = -Tdot * grad(t_s) gives Ny = sqrt(|Tdot|) * |grad(t_s)|, and the
  cooling rate is approximated by the freezing-interval rate
  (TL - TS) / t_s. The result is a relative ranking within one part, so the
  risk mask flags its lowest decile rather than applying absolute
  literature thresholds.
- Macro-shrinkage regions: connected groups of the latest-solidifying nodes
  that no feed source can reach through metal that is still liquid.
- Sinks: regions whose highest point (against gravity) is on the free
  surface, which would draw the surface in rather than void internally.
- Feeder suggestions: a surface point above each region and the minimum
  feeder modulus (1.2 x the region's V/A, Chvorinov's rule).
"""
import logging

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)

# Volumetric solidification shrinkage per alloy family, in percent. Severity
# of a region = shrinkage_pct / 100 * region volume. Grey iron's graphite
# expansion offsets its shrinkage, hence 0 (regions are still reported).
SHRINKAGE_PCT = {
    "steel": 2.5,
    "low_alloy_steel": 2.5,
    "ductile_iron": 0.75,
    "grey_iron": 0.0,
}

# Guards divisions and square roots without moving any physical value.
_EPS = 1e-9


def node_gradient(pos: np.ndarray, tets: np.ndarray, field: np.ndarray) -> np.ndarray:
    """(N, 3) nodal gradient of a P1 field on a tet mesh, in [field units]/mm.

    The gradient is constant per tet (solve J g = df with J the tet's edge
    vectors from vertex 0), then averaged onto the nodes weighted by tet
    volume. Degenerate tets contribute nothing; isolated nodes get zero."""
    pos = np.asarray(pos, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    field = np.asarray(field, dtype=np.float64)
    n_nodes = pos.shape[0]

    if tets.shape[0] == 0:
        return np.zeros((n_nodes, 3))

    p0 = pos[tets[:, 0]]
    p1 = pos[tets[:, 1]]
    p2 = pos[tets[:, 2]]
    p3 = pos[tets[:, 3]]
    f0 = field[tets[:, 0]]
    f1 = field[tets[:, 1]]
    f2 = field[tets[:, 2]]
    f3 = field[tets[:, 3]]

    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0

    jac = np.stack([e1, e2, e3], axis=1)          # (M,3,3), rows = edge vectors
    rhs = np.stack([f1 - f0, f2 - f0, f3 - f0], axis=1)  # (M,3)

    vol = np.abs(np.einsum('ij,ij->i', e1, np.cross(e2, e3))) / 6.0  # (M,)

    grad_tet = np.zeros_like(rhs)
    nondeg = vol > 0
    if nondeg.any():
        grad_tet[nondeg] = np.linalg.solve(jac[nondeg], rhs[nondeg, :, None])[..., 0]

    node_grad_sum = np.zeros((n_nodes, 3))
    node_vol_sum = np.zeros(n_nodes)
    weighted = grad_tet * vol[:, None]
    for k in range(4):
        idx = tets[:, k]
        np.add.at(node_grad_sum, idx, weighted)
        np.add.at(node_vol_sum, idx, vol)

    return node_grad_sum / np.maximum(node_vol_sum, _EPS)[:, None]


def niyama_proxy(pos: np.ndarray, tets: np.ndarray, t_s: np.ndarray,
                 TL: float, TS: float) -> np.ndarray:
    """sqrt((TL - TS) / max(t_s, eps)) * |grad(t_s)|, per node.

    `TL`, `TS` in deg C, `t_s` in s, `pos` in mm. A relative ranking within
    one part; see the module docstring."""
    t_s = np.asarray(t_s, dtype=np.float64)
    grad = node_gradient(pos, tets, t_s)
    grad_mag = np.linalg.norm(grad, axis=1)

    t_safe = np.maximum(t_s, _EPS)
    tdot_proxy = (TL - TS) / t_safe
    return np.sqrt(np.maximum(tdot_proxy, 0.0)) * grad_mag


def porosity_risk(ny: np.ndarray, pct: float = 10.0) -> np.ndarray:
    """Boolean mask of the within-part lowest `pct`% Niyama values
    (`ny <= percentile(ny, pct)`; ties at the threshold are all flagged)."""
    ny = np.asarray(ny, dtype=np.float64)
    threshold = np.percentile(ny, pct)
    return ny <= threshold


# Local vertex-index pairs/triples of a tetrahedron.
_TET_EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_TET_FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def _tet_volumes(pos: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """(M,) unsigned tet volumes in mm^3."""
    if tets.shape[0] == 0:
        return np.zeros(0)
    p0, p1, p2, p3 = pos[tets[:, 0]], pos[tets[:, 1]], pos[tets[:, 2]], pos[tets[:, 3]]
    e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0
    return np.abs(np.einsum('ij,ij->i', e1, np.cross(e2, e3))) / 6.0


def _induced_components(n_nodes: int, tets: np.ndarray, mask: np.ndarray):
    """Connected components of the tet-adjacency graph restricted to `mask`.

    Returns (n_comp, labels) with labels of shape (n_nodes,): the component
    id for masked nodes, -1 elsewhere."""
    idx = np.flatnonzero(mask)
    labels_full = np.full(n_nodes, -1, dtype=np.int64)
    if idx.size == 0:
        return 0, labels_full

    local_of = -np.ones(n_nodes, dtype=np.int64)
    local_of[idx] = np.arange(idx.size)

    rows_list, cols_list = [], []
    for a, b in _TET_EDGE_PAIRS:
        va, vb = tets[:, a], tets[:, b]
        both = mask[va] & mask[vb]
        if both.any():
            rows_list.append(local_of[va[both]])
            cols_list.append(local_of[vb[both]])
    n_local = idx.size
    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.ones(rows.shape[0])
        graph = csr_matrix((data, (rows, cols)), shape=(n_local, n_local))
    else:
        graph = csr_matrix((n_local, n_local))
    n_comp, labels_local = connected_components(graph, directed=False)
    labels_full[idx] = labels_local
    return n_comp, labels_full


def _boundary_faces(tets: np.ndarray) -> np.ndarray:
    """(F, 3) vertex triples of the mesh's outer faces: faces that belong to
    exactly one tet, in their original per-tet vertex order."""
    tets = np.asarray(tets, dtype=np.int64)
    if tets.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)
    face_idx = np.asarray(_TET_FACES, dtype=np.int64)          # (4, 3)
    all_faces = tets[:, face_idx].reshape(-1, 3)                # (4*M, 3), original order
    sorted_faces = np.sort(all_faces, axis=1)
    _, first_idx, counts = np.unique(sorted_faces, axis=0, return_index=True,
                                     return_counts=True)
    boundary = counts == 1
    return all_faces[first_idx[boundary]]


def _region_surface_area(pos: np.ndarray, tets: np.ndarray, nodes: np.ndarray,
                         t_s: np.ndarray, threshold: float,
                         boundary_faces: np.ndarray = None) -> float:
    """Surface area (mm^2) of a hot region, for its V/A modulus.

    The interior part is the iso-surface `t_s == threshold` (the cut that
    selected the region), built by marching tetrahedra on the continuous
    field: a tet with 1 or 3 vertices above the threshold contributes a
    triangle, one with 2 contributes a planar quadrilateral, and crossing
    points sit at their interpolated position on each edge. A tet counts for
    this region if any of its vertices is in `nodes`. Outer faces of the
    mesh are added separately, weighted by the fraction of their 3 vertices
    in the region. `boundary_faces` can be passed in to avoid recomputing
    `_boundary_faces(tets)` for every region."""
    n_nodes = pos.shape[0]
    is_member = np.zeros(n_nodes, dtype=bool)
    is_member[np.asarray(nodes, dtype=np.int64)] = True
    t_s = np.asarray(t_s, dtype=np.float64)
    threshold = float(threshold)

    area = 0.0
    if tets.shape[0] > 0:
        hot_mask = t_s >= threshold
        member_per_tet = hot_mask[tets]
        n_above = member_per_tet.sum(axis=1)
        mixed = np.flatnonzero((n_above >= 1) & (n_above <= 3))
        for ti in mixed:
            tet = tets[ti]
            if not is_member[tet].any():
                continue  # this crossing belongs to a different region
            verts = pos[tet]
            vals = t_s[tet]
            above_local = member_per_tet[ti]
            k = int(n_above[ti])
            if k in (1, 3):
                odd = int(np.flatnonzero(above_local)[0]) if k == 1 else \
                    int(np.flatnonzero(~above_local)[0])
                others = [i for i in range(4) if i != odd]
                pts = []
                for o in others:
                    frac = (threshold - vals[odd]) / (vals[o] - vals[odd])
                    pts.append(verts[odd] + frac * (verts[o] - verts[odd]))
                area += 0.5 * float(np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])))
            else:  # k == 2
                hi = np.flatnonzero(above_local)
                lo = np.flatnonzero(~above_local)

                def _cross_point(h, c, verts=verts, vals=vals):
                    frac = (threshold - vals[h]) / (vals[c] - vals[h])
                    return verts[h] + frac * (verts[c] - verts[h])

                a_pt = _cross_point(hi[0], lo[0])
                b_pt = _cross_point(hi[0], lo[1])
                c_pt = _cross_point(hi[1], lo[1])
                d_pt = _cross_point(hi[1], lo[0])
                area += 0.5 * float(np.linalg.norm(np.cross(c_pt - a_pt, d_pt - b_pt)))

    if boundary_faces is None:
        boundary_faces = _boundary_faces(tets)
    if boundary_faces.shape[0] > 0:
        # Each outer face counts with the fraction of its 3 vertices in
        # the region.
        member_counts = is_member[boundary_faces].sum(axis=1).astype(np.float64)  # (F,)
        frac = member_counts / 3.0
        nonzero = frac > 0
        if nonzero.any():
            v = boundary_faces[nonzero]
            p0f, p1f, p2f = pos[v[:, 0]], pos[v[:, 1]], pos[v[:, 2]]
            tri_areas = 0.5 * np.linalg.norm(np.cross(p1f - p0f, p2f - p0f), axis=1)
            area += float(np.sum(frac[nonzero] * tri_areas))
    return area


def hot_regions(pos: np.ndarray, tets: np.ndarray, t_s: np.ndarray,
                top_frac: float = 0.10) -> list:
    """Connected groups (via tet adjacency) of the top-`top_frac` latest-
    solidifying nodes, sorted by `t_max` descending.

    Each region is a dict: `nodes` (sorted indices), `centroid` (mm),
    `volume_mm3`, `t_max`, `region_id` (its index in this list),
    `hot_threshold` (the percentile cut used) and `sub_resolution` (fewer
    than 4 nodes, too small for a meaningful V/A modulus). The volume sums,
    over all tets touching the region, tet volume times the fraction of its
    vertices that are hot, which converges with mesh refinement."""
    top_frac = float(top_frac)
    if not (0.0 < top_frac <= 1.0):
        raise ValueError(f"hot_regions: top_frac must be in (0, 1], got {top_frac!r}")

    pos = np.asarray(pos, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    t_s = np.asarray(t_s, dtype=np.float64)
    n_nodes = pos.shape[0]

    if n_nodes == 0 or tets.shape[0] == 0:
        return []

    threshold = np.percentile(t_s, 100.0 * (1.0 - top_frac))
    hot_mask = t_s >= threshold
    if not hot_mask.any():
        return []

    n_comp, labels = _induced_components(n_nodes, tets, hot_mask)

    vol = _tet_volumes(pos, tets)
    n_hot_per_tet = hot_mask[tets].sum(axis=1)
    has_hot = n_hot_per_tet > 0
    region_vol = np.zeros(n_comp)
    if has_hot.any():
        # any one hot vertex identifies the tet's (unique) component.
        first_hot_col = np.argmax(hot_mask[tets], axis=1)
        first_hot_node = tets[np.arange(tets.shape[0]), first_hot_col]
        comp_of_tet = labels[first_hot_node]
        weight = vol * (n_hot_per_tet / 4.0)
        np.add.at(region_vol, comp_of_tet[has_hot], weight[has_hot])

    regions = []
    for c in range(n_comp):
        nodes = np.sort(np.flatnonzero(labels == c))
        regions.append({
            "nodes": nodes,
            "centroid": pos[nodes].mean(axis=0),
            "volume_mm3": float(region_vol[c]),
            "t_max": float(t_s[nodes].max()),
        })
    regions.sort(key=lambda r: r["t_max"], reverse=True)
    for i, r in enumerate(regions):
        r["region_id"] = i
        r["hot_threshold"] = float(threshold)
        # Below 4 nodes the iso-surface area and the lumped volume no longer
        # agree, so the V/A modulus is not meaningful.
        r["sub_resolution"] = len(r["nodes"]) < 4
    return regions


def is_fed(region: dict, t_s: np.ndarray, tets: np.ndarray,
          feed_nodes: np.ndarray, corridor_frac: float = 0.95) -> bool:
    """Whether a feed source can still reach the region through liquid metal.

    True iff some region node and some `feed_nodes` node are connected
    through nodes with `t_s >= corridor_frac * region["t_max"]` (a corridor
    that stays liquid until the region itself is nearly solid). Feed nodes
    are adiabatic-tagged surface nodes, for example a riser face. With no
    feed nodes every region is unfed, which is the conservative answer. A
    region with t_max <= 0 is reported unfed."""
    t_s = np.asarray(t_s, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    feed_nodes = np.asarray(feed_nodes, dtype=np.int64)
    region_nodes = np.asarray(region["nodes"], dtype=np.int64)

    if feed_nodes.size == 0:
        return False
    if tets.shape[0] == 0:
        return False
    region_t_max = float(region["t_max"])
    if region_t_max <= 0:
        logger.debug(
            "is_fed: region t_max=%.6g <= 0 -- corridor threshold would be "
            "vacuous (every non-negative t_s qualifies); reporting unfed.",
            region_t_max)
        return False

    n_nodes = t_s.shape[0]
    threshold = corridor_frac * region_t_max
    warm_mask = t_s >= threshold

    _, labels = _induced_components(n_nodes, tets, warm_mask)

    region_labels = set(int(l) for l in labels[region_nodes] if l >= 0)
    if not region_labels:
        return False
    feed_labels = set(int(l) for l in labels[feed_nodes] if l >= 0)
    return len(region_labels & feed_labels) > 0


def macro_shrinkage(pos: np.ndarray, tets: np.ndarray, t_s: np.ndarray,
                    feed_nodes: np.ndarray, shrinkage_pct: float) -> list:
    """The hot regions that are not fed, each with `severity = shrinkage_pct /
    100 * volume_mm3`. `region_id` keeps its value from `hot_regions`."""
    regions = hot_regions(pos, tets, t_s)
    out = []
    for r in regions:
        if is_fed(r, t_s, tets, feed_nodes):
            continue
        severity = shrinkage_pct / 100.0 * r["volume_mm3"]
        out.append({**r, "severity": severity})
    return out


def top_sinks(regions: list, pos: np.ndarray, surface_nodes: np.ndarray,
             gravity_vec) -> list:
    """Regions whose highest node along -gravity is a surface node. Returns
    `{"region_id", "location" (mm), "volume_mm3"}` for each."""
    pos = np.asarray(pos, dtype=np.float64)
    gravity_vec = np.asarray(gravity_vec, dtype=np.float64)
    up = -gravity_vec
    norm = np.linalg.norm(up)
    if norm > 0:
        up = up / norm
    surface_set = set(np.asarray(surface_nodes, dtype=np.int64).tolist())

    sinks = []
    for r in regions:
        nodes = np.asarray(r["nodes"], dtype=np.int64)
        proj = pos[nodes] @ up
        top_node = int(nodes[np.argmax(proj)])
        if top_node in surface_set:
            sinks.append({
                "region_id": r["region_id"],
                "location": pos[top_node].copy(),
                "volume_mm3": r["volume_mm3"],
            })
    return sinks


def feeder_recommendations(regions: list, pos: np.ndarray, tets: np.ndarray,
                           surface_nodes: np.ndarray, gravity_vec, t_s: np.ndarray) -> list:
    """Per region: a feeder `location` and `min_modulus_mm = 1.2 * V/A`.

    The location is the surface node closest to the vertical line (along
    -gravity) through the region's centroid, among surface nodes at or above
    the centroid (all surface nodes if none are). Sub-resolution regions are
    skipped. A region with zero volume or area gets a warning and
    `min_modulus_mm = 0.0`. Raises ValueError if `surface_nodes` is empty."""
    pos = np.asarray(pos, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    t_s = np.asarray(t_s, dtype=np.float64)
    gravity_vec = np.asarray(gravity_vec, dtype=np.float64)
    up = -gravity_vec
    norm = np.linalg.norm(up)
    if norm > 0:
        up = up / norm
    surface_nodes = np.asarray(surface_nodes, dtype=np.int64)
    if surface_nodes.size == 0:
        raise ValueError(
            "feeder_recommendations: surface_nodes is empty -- no candidate "
            "feeder location exists (a mesh with no free surface is not a "
            "valid casting)")
    surface_pos = pos[surface_nodes]
    boundary_faces = _boundary_faces(tets)  # shared by every region

    recs = []
    n_skipped_sub_resolution = 0
    for r in regions:
        if r.get("sub_resolution", False):
            n_skipped_sub_resolution += 1
            continue
        nodes = np.asarray(r["nodes"], dtype=np.int64)
        volume = r["volume_mm3"]
        threshold = r["hot_threshold"]
        area = _region_surface_area(pos, tets, nodes, t_s, threshold,
                                    boundary_faces=boundary_faces)
        if area > 0 and volume > 0:
            region_modulus = volume / area
        else:
            logger.warning(
                "feeder_recommendations: region_id=%s has degenerate "
                "volume/area (V=%.6g mm^3, A=%.6g mm^2) -- min_modulus_mm is "
                "not meaningful and is reported as 0.0",
                r.get("region_id"), volume, area)
            region_modulus = 0.0

        centroid = np.asarray(r["centroid"], dtype=np.float64)
        rel = surface_pos - centroid
        axis_comp = rel @ up
        perp = rel - np.outer(axis_comp, up)
        perp_dist = np.linalg.norm(perp, axis=1)

        above = np.flatnonzero(axis_comp >= 0)
        candidates = above if above.size > 0 else np.arange(surface_pos.shape[0])
        best = candidates[np.argmin(perp_dist[candidates])]

        recs.append({
            "location": surface_pos[best].copy(),
            "min_modulus_mm": 1.2 * region_modulus,
            "region_id": r["region_id"],
        })
    if n_skipped_sub_resolution:
        logger.info(
            "feeder_recommendations: skipped %d sub-resolution region(s) (<4 "
            "nodes) -- V/A modulus is not meaningful at this resolution",
            n_skipped_sub_resolution)
    return recs


def screen_casting(pos: np.ndarray, tets: np.ndarray, surface_nodes: np.ndarray,
                   t_s: np.ndarray, TL: float, TS: float, alloy_family: str,
                   gravity_vec=(0, 0, -1), feed_nodes: np.ndarray = None) -> dict:
    """All four screening outputs for one part.

    `alloy_family` must be a key of `SHRINKAGE_PCT`. `surface_nodes` are all
    surface nodes (for sinks and feeder placement); `feed_nodes` are the
    nodes that can feed a region (None means none).

    Returns {"niyama", "porosity_risk", "niyama_source": "proxy",
    "macro_shrinkage", "top_sinks", "feeders", "material"}."""
    if alloy_family not in SHRINKAGE_PCT:
        raise KeyError(
            f"screen_casting: unknown alloy_family {alloy_family!r} -- known "
            f"families: {sorted(SHRINKAGE_PCT)}"
        )

    pos = np.asarray(pos, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    surface_nodes = np.asarray(surface_nodes, dtype=np.int64)
    t_s = np.asarray(t_s, dtype=np.float64)
    feed_nodes_arr = (np.zeros(0, dtype=np.int64) if feed_nodes is None
                      else np.asarray(feed_nodes, dtype=np.int64))

    ny = niyama_proxy(pos, tets, t_s, TL, TS)
    risk = porosity_risk(ny)

    shrinkage_pct = SHRINKAGE_PCT[alloy_family]
    macro = macro_shrinkage(pos, tets, t_s, feed_nodes_arr, shrinkage_pct)
    sinks = top_sinks(macro, pos, surface_nodes, gravity_vec)
    feeders = feeder_recommendations(macro, pos, tets, surface_nodes, gravity_vec, t_s)

    return {
        "niyama": ny,
        "porosity_risk": risk,
        "niyama_source": "proxy",
        "macro_shrinkage": macro,
        "top_sinks": sinks,
        "feeders": feeders,
        "material": {"alloy_family": alloy_family, "shrinkage_pct": shrinkage_pct},
    }
