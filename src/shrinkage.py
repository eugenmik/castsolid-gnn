"""Field derivatives and the Niyama shrinkage-porosity criterion on tet meshes.

Pure numpy/scipy: no project imports. Mesh arrays (`pos` in mm, `tets` as
int node-index quadruples) are passed in by the caller -- this module knows
nothing about `.msh` parsing, features, or the GNN.

Unit conventions (mirroring the cached raw-solver fields, see
`docs/superpowers/specs/2026-07-24-shrinkage-porosity-screening-design.md`,
ADDENDUM section "GNN-path derived fields"):
  - `pos` -- node coordinates in **mm**.
  - `gradT` -- thermal-gradient magnitude in **K/m** (as cached by the FEniCSx
    solver, see `tools/doe/adapters.py`).
  - `dTdt` -- cooling-rate in **K/s** (as cached).
  - `t_s` / solidification time -- **s**.
  - Niyama number `Ny` -- reported in the literature unit
    **(degC*s)^0.5 / mm** (`G[degC/mm] / sqrt(|Tdot|[degC/s])`); note 1 K
    of *difference* equals 1 degC of difference, so no additive offset is
    needed when converting a gradient/rate, only the K/m -> degC/mm scale
    (divide by 1000).

Two ways to obtain Ny, per the ADDENDUM:
  - **Exact path** (training/validation rows with cached `gradT`/`dTdt`):
    `niyama_exact` computes Ny directly from the solver's own gradient and
    cooling-rate fields, and is compared against literature absolute
    thresholds (`NY_CRIT`, `porosity_risk(mode="absolute")`).
  - **Proxy path** (GNN-inference rows, where only the predicted
    solidification-time field `t_s` exists): the exact identity
    `grad(T) = -Tdot * grad(t_s)` (differentiate `T(x, t_s(x)) = T_S`) gives
    `G = |Tdot| * |grad(t_s)|` and `Ny = G / sqrt(|Tdot|) = sqrt(|Tdot|) *
    |grad(t_s)|`. `Tdot` itself is not available on this path and is proxied
    by the characteristic freezing-interval rate `(TL - TS) / max(t_s, eps)`.
    `niyama_proxy` implements this. Because it is a proxy, it is only a
    *relative* risk ranking within one part -- use
    `porosity_risk(mode="percentile")`, never the absolute thresholds.
"""
import logging

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)

# Shrinkage percentage per alloy family (volumetric solidification shrinkage,
# used as severity = shrinkage_pct/100 * region volume). Grey iron's graphite
# expansion during eutectic solidification largely offsets liquid/solid
# shrinkage, hence 0.0 -- macro-shrinkage regions are still detected/reported
# for grey iron (the geometry-only isolation criterion doesn't depend on
# alloy), just with zero severity.
SHRINKAGE_PCT = {
    "steel": 2.5,
    "low_alloy_steel": 2.5,
    "ductile_iron": 0.75,
    "grey_iron": 0.0,
}

# Absolute Niyama-criterion thresholds (exact path only), (degC*s)^0.5/mm.
# Below this value -> flagged as at-risk of micro-shrinkage porosity.
NY_CRIT = {
    "steel": 1.0,
    "low_alloy_steel": 1.0,
    "grey_iron": 0.75,
    "ductile_iron": 0.75,
}

# Small epsilon guards against division-by-zero / sqrt(0) without perturbing
# any physically meaningful value (cooling rates and solidification times of
# interest are always many orders of magnitude larger than this).
_EPS = 1e-9


def node_gradient(pos: np.ndarray, tets: np.ndarray, field: np.ndarray) -> np.ndarray:
    """(N,3) volume-weighted average of per-tet linear-shape-function gradients.

    For a linear (P1) tetrahedral element the interpolated field is affine
    over the tet, so its gradient is constant per tet. For tet vertices
    x0..x3 with field values f0..f3, the gradient g solves the 3x3 system
        J @ g = rhs,  J[i] = x_{i+1} - x0,  rhs[i] = f_{i+1} - f0   (i=0,1,2)
    i.e. J's rows are the tet's three edge vectors from vertex 0, and rhs is
    the corresponding field differences -- the standard closed form for a
    linear tet element's gradient (no shape-function basis needed beyond
    this, since a P1 gradient is uniquely determined by 3 independent edge
    derivatives).

    The constant per-tet gradient is then averaged onto each of its 4 nodes,
    weighted by tet volume (a volume-weighted average over all tets incident
    to a node), which is the standard nodal-recovery scheme for
    element-constant gradients on unstructured tet meshes.

    Units: `pos` in mm; the returned gradient is in [field units] / mm.

    Degenerate (zero-volume) tets contribute zero weight and zero gradient
    (guarded, never divide by zero); a node touched by no tets (or an empty
    `tets` array) gets a zero gradient.
    """
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


def niyama_exact(gradT_K_per_m: np.ndarray, dTdt_K_per_s: np.ndarray) -> np.ndarray:
    """Ny in (degC*s)^0.5/mm: G[degC/mm] / sqrt(|Tdot|[degC/s]), eps-guarded.

    Exact path: uses the solver's own cached `gradT` (K/m) and `dTdt` (K/s)
    fields directly (see module docstring). 1 K/m = 1e-3 degC/mm (a
    difference/gradient, not an absolute temperature, so no offset).
    """
    gradT_K_per_m = np.asarray(gradT_K_per_m, dtype=np.float64)
    dTdt_K_per_s = np.asarray(dTdt_K_per_s, dtype=np.float64)

    G_degC_per_mm = gradT_K_per_m / 1000.0
    tdot_abs = np.abs(dTdt_K_per_s)
    return G_degC_per_mm / np.sqrt(np.maximum(tdot_abs, _EPS))


def niyama_proxy(pos: np.ndarray, tets: np.ndarray, t_s: np.ndarray,
                 TL: float, TS: float) -> np.ndarray:
    """sqrt((TL-TS)/max(t_s,eps)) * |node_gradient(t_s)| -- the ADDENDUM contract.

    Proxy path: only the (predicted) solidification-time field `t_s` (s) is
    available. Using the identity grad(T) = -Tdot * grad(t_s), a proxy
    cooling rate `Tdot_proxy = (TL - TS) / max(t_s, eps)` (the characteristic
    freezing-interval rate) stands in for the true `Tdot`, giving
    `Ny_proxy = sqrt(Tdot_proxy) * |grad(t_s)|`. `TL`, `TS` in degC; `t_s` in
    s; `pos` in mm (so `grad(t_s)` is in s/mm, matching the exact path's
    (degC*s)^0.5/mm unit once multiplied by sqrt(degC/s)).

    This is a *relative* ranking only, not comparable to the literature
    `NY_CRIT` absolute thresholds -- flag risk within one part via
    `porosity_risk(mode="percentile")`.
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    grad = node_gradient(pos, tets, t_s)
    grad_mag = np.linalg.norm(grad, axis=1)

    t_safe = np.maximum(t_s, _EPS)
    tdot_proxy = (TL - TS) / t_safe
    return np.sqrt(np.maximum(tdot_proxy, 0.0)) * grad_mag


def porosity_risk(ny: np.ndarray, mode: str, alloy_family: str = None,
                  pct: float = 10.0) -> np.ndarray:
    """Boolean at-risk mask from a Niyama field.

    mode="absolute" (exact path): `ny < NY_CRIT[alloy_family]`. `alloy_family`
    must be a key of `NY_CRIT`; an unknown or missing (`None`) family raises
    `KeyError` -- there is no silent default, because guessing an alloy
    family would silently mis-threshold the risk call.

    mode="percentile" (proxy path): flags the within-part bottom-`pct`%
    lowest Niyama values, i.e. `ny <= numpy.percentile(ny, pct)` (numpy's
    default linear-interpolation percentile). Tie handling is deterministic
    and order-independent: every value tied at (or below) the threshold is
    flagged, so if duplicate values straddle the threshold the flagged
    fraction can be slightly *more* than `pct`% (never less, and never
    dependent on array order).
    """
    ny = np.asarray(ny, dtype=np.float64)

    if mode == "absolute":
        crit = NY_CRIT[alloy_family]
        return ny < crit
    elif mode == "percentile":
        threshold = np.percentile(ny, pct)
        return ny <= threshold
    else:
        raise ValueError(f"unknown porosity_risk mode: {mode!r}")


# ---------------------------------------------------------------------------
# Macro-shrinkage regions, top-surface sinks, feeder recommendations.
#
# See docs/superpowers/specs/2026-07-24-shrinkage-porosity-screening-design.md
# section 2-4 and the OWNER-ADJUDICATED correction in
# docs/superpowers/plans/2026-08-01-sp2-shrinkage.md (2026-08-01, review round
# 1, finding 1): a hot spot is a local maximum of solidification time `t_s`
# (last metal to freeze). The ORIGINAL "monotonically non-decreasing t_s path
# to any surface node" feeding-path definition was proved degenerate (a
# region can never escape its own hot-node superlevel-set component under
# that rule, making it tautologically equivalent to "region touches the
# domain surface" -- replacing the whole search with `return False` left
# every test green). The adjudicated definition instead: a region is fed iff
# a node path over tet adjacency connects it to a caller-supplied FEED node
# (e.g. nodes on Adiabatic-tagged riser/sleeve faces -- NOT the generic mold
# surface; a bare sand casting with no risers passes an empty `feed_nodes`,
# so every region is conservatively unfed) with EVERY node on that path
# (endpoints included) at or above `corridor_frac * region["t_max"]` -- a
# warm corridor that stays liquid until the region itself finishes freezing,
# with a 5% relative tolerance. This breaks the old degeneracy because the
# corridor threshold is relative to the REGION's own peak, not the mesh-wide
# hot cut, so a feed node need not itself be part of the region's hot-node
# superlevel set.
#
# Severity scales with the alloy's volumetric shrinkage percentage
# (`SHRINKAGE_PCT`) times the region's volume. A region whose highest point
# against gravity is itself a free surface sinks that surface instead of
# voiding internally. A feeder placed above the region must have modulus
# (V/A, Chvorinov's rule) >= 1.2x the region's own modulus to freeze after it.
# ---------------------------------------------------------------------------

# Local vertex-index pairs/triples of a tetrahedron (0,1,2,3), used to walk a
# tet's 6 edges / 4 faces without re-deriving the combinatorics inline.
_TET_EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_TET_FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def _tet_volumes(pos: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """(M,) unsigned tet volumes in mm^3 (independent copy of the closed form
    used inside `node_gradient`, kept private and separate so Task 1's
    function is never touched)."""
    if tets.shape[0] == 0:
        return np.zeros(0)
    p0, p1, p2, p3 = pos[tets[:, 0]], pos[tets[:, 1]], pos[tets[:, 2]], pos[tets[:, 3]]
    e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0
    return np.abs(np.einsum('ij,ij->i', e1, np.cross(e2, e3))) / 6.0


def _induced_components(n_nodes: int, tets: np.ndarray, mask: np.ndarray):
    """Connected components of the tet-adjacency graph induced on `mask`
    (i.e. vertices = nodes with `mask[node]` True, edges = tet-adjacent pairs
    both satisfying `mask`). Returns `(n_comp, labels)` where `labels` has
    shape `(n_nodes,)`: the component id for masked nodes, `-1` elsewhere.
    Shared by `hot_regions` (mask = the global top-`top_frac` hot cut) and
    `is_fed` (mask = the per-region warm corridor, `t_s >= corridor_frac *
    region_t_max`) -- same graph-labeling primitive, different masks.
    """
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
    """(F, 3) int64 array of the mesh's own boundary-face vertex triples
    (ORIGINAL per-tet vertex order, not a sorted dedup key) -- faces touched
    by exactly one tet, i.e. with nothing "on the other side".

    Review finding I4: `_region_surface_area`'s domain-boundary term used to
    rebuild this WHOLE-MESH structure via a pure-Python double loop over
    every tet's 4 faces, on EVERY call -- but it depends only on `tets`, not
    on any particular region's `nodes`/`threshold`, so `feeder_recommendations`
    (this module's only caller of `_region_surface_area`) was redoing the
    exact same O(n_tets) work once per region. Measured on BUKSA (32138
    tets, 16 regions): 4.74 s total screening time, ~8x the documented
    ~0.5-0.6 s (a figure that had only ever been measured on a single-region
    part). Hoisted here so it runs ONCE per `feeder_recommendations` call,
    vectorized via `np.unique(..., axis=0, return_counts=True)` (both the
    dict-free dedup AND -- inside `_region_surface_area` -- the boundary-term
    summation itself are now array ops, not a Python loop).

    Every tet contributes 4 candidate faces; a face shared by exactly 2 tets
    (an interior face) is discarded, one shared by exactly 1 (nothing on the
    other side) is kept -- a face can never be shared by >2 tets in a
    conforming tet mesh, so `counts == 1` is the correct (and only
    necessary) boundary test."""
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
    """Boundary-surface area (mm^2) of the SAME body `hot_regions`' fraction-
    weighted `volume_mm3` integrates over (review round 2, finding NEW-1: the
    original all-4-vertices-hot rule measured a strictly SMALLER body than
    the fraction-weighted volume, giving an inflated, inconsistent modulus).

    Round-2 history / why this changed again (review round 3): the first fix
    ran the co-area formula on a BINARIZED 0/1 indicator (`nodes` membership
    only). That is an exact calculus identity for a piecewise-linear field,
    but the field it measures is the indicator's own artificial "staircase"
    step (always crossing at each tet edge's midpoint, regardless of where
    the REAL field actually crosses its selection threshold) -- an
    ANISOTROPIC discrete-perimeter measure that plateaus ~24-29% high on
    curved/diagonal boundaries (axis-aligned cuts happened to look fine only
    because a midpoint crossing is coincidentally correct when the true
    level surface IS flat and axis-aligned).

    This version runs marching tetrahedra directly on the CONTINUOUS `t_s`
    field, cut at `threshold` -- literally "the same cut that defines the
    region" (the percentile cutoff `hot_regions` used, stored on each region
    dict as `hot_threshold`), so the interpolated crossing point on each tet
    edge sits at its true fractional position (`(threshold-f_a)/(f_b-f_a)`),
    not an assumed 0.5. Per tet, classified by how many of its 4 vertices
    have `t_s >= threshold` (`k`):
      - k in {0, 4}: no crossing, no contribution.
      - k in {1, 3}: the crossing is a single triangle through the 3 edges
        connecting the "odd one out" vertex to the other 3 (interpolated
        edge points) -- the standard 1-3 marching-tetrahedra case.
      - k == 2: the crossing is a planar quadrilateral through the 4 edges
        connecting each of the 2 "above" vertices to each of the 2 "below"
        vertices (`t_s` is affine per tet, so ANY level set within a tet is
        a plane intersected with the tet -- always exactly planar,
        regardless of the specific values -- computed via the standard
        diagonal cross-product quadrilateral-area formula).
    A tet is attributed to THIS region iff any of its vertices is in
    `nodes` (well-defined: two vertices with `t_s >= threshold` sharing a
    tet are adjacent by `hot_regions`' own definition, so a mixed tet's
    "above" vertices, if any, all belong to exactly one region already).

    Domain-boundary term (KEPT, unchanged, review-verified exact for a
    region whose boundary coincides with the mesh's own outer surface): a
    tet face touching no other tet (nothing "on the other side" for the
    interior marching-tets term to see) is credited separately, weighted by
    the (binary, `nodes`-membership) hot fraction of its own 3 vertices.

    See tests/test_shrinkage.py for the acceptance suite this targets:
    sphere-field area converging toward the analytic 4*pi*R^2 with
    refinement (not plateauing), a diagonal-plane cut and an orientation
    sweep both landing far below the ~29-36% anisotropy the binarized
    co-area formula showed, and the axis-aligned hand-check (now computed
    against a genuine continuous field, not an arbitrarily-thresholded
    binary block) landing near the Chvorinov M=side/6 identity for a cube.

    `boundary_faces` (review finding I4): the mesh's own domain-boundary
    faces (`_boundary_faces(tets)`'s output), region-INDEPENDENT -- a single
    call's worth of work shared by every region in one
    `feeder_recommendations` invocation instead of being rebuilt (a
    pure-Python double loop over ALL tets) on every one of this function's
    calls. `None` (the default) computes it internally -- exactly the old
    behavior, for any direct caller that doesn't have one precomputed.
    """
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
        # Vectorized version of the old per-face Python loop: each boundary
        # face's hot fraction is (# of its 3 vertices in `nodes`) / 3, and
        # only faces with a nonzero fraction contribute a (fraction-
        # weighted) triangle area.
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
    """Connected components (via tet adjacency) of the top-`top_frac`
    latest-solidifying nodes.

    A node is "hot" if `t_s >= percentile(t_s, 100*(1-top_frac))` (inclusive,
    same tie convention as `porosity_risk`'s percentile mode -- may include
    slightly more than `top_frac` when values tie at the threshold).
    Components are the connected components of the graph whose vertices are
    the hot nodes and whose edges are tet-adjacency pairs restricted to hot
    endpoints (two hot nodes sharing a tet).

    Returns a list of dicts sorted by `t_max` descending, each carrying a
    stable `region_id` = its index in THIS sorted list (0-based) -- the id
    downstream functions (`macro_shrinkage`, `top_sinks`,
    `feeder_recommendations`) read back off the dict rather than
    re-deriving positionally, so it stays coherent across filtering:
    `{"nodes": sorted int index array, "centroid": (3,) mm, "volume_mm3":
    float, "t_max": float, "region_id": int, "hot_threshold": float,
    "sub_resolution": bool}`. `sub_resolution` (review finding C1) is `True`
    iff the region has fewer than 4 nodes -- below this floor
    `_region_surface_area`'s iso-surface area and the lumped-nodal
    `volume_mm3` computed below diverge without bound as `t_max` approaches
    `hot_threshold`; `feeder_recommendations` reads this flag back to skip
    such regions rather than report a physically meaningless modulus (see
    its own docstring), while `macro_shrinkage`/`top_sinks` still list them
    unfiltered (their volume/severity/centroid stay meaningful).
    `hot_threshold` is the same percentile cutoff (`t_s >= hot_threshold`)
    used to select every region in THIS call -- `feeder_recommendations`
    (review round 3) reads it back to run marching tetrahedra on the
    continuous `t_s` field at exactly "the cut that defines the region",
    rather than re-deriving or guessing it downstream.

    `volume_mm3` is a fraction-weighted sum over ALL tets touching the
    region: each tet contributes `tet_volume * n_hot_vertices/4` to the
    component of its hot vertices (well-defined -- any two hot vertices
    sharing a tet are adjacent by this function's own definition, so a
    tet's hot vertices, if any, all belong to the same component). This is
    mesh-convergent (equivalent to summing each hot node's lumped nodal
    volume, `sum(tet_volume/4)` over its incident tets) and replaces an
    earlier all-4-vertices-hot rule that measurably undercounted volume
    (35-65% on smooth fields, review round 1 finding 2) by discarding every
    tet straddling the region boundary instead of weighting it.
    """
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
        # Review finding C1: a region with fewer than 4 nodes is below the
        # resolution floor at which `_region_surface_area`'s iso-surface
        # area and the lumped-nodal `volume_mm3` above agree even
        # approximately -- as `t_max` approaches `hot_threshold` the true
        # iso-surface area shrinks toward zero while the lumped volume stays
        # O(h^3), so V/A (the feeder modulus) diverges without bound.
        # `feeder_recommendations` reads this flag back to skip such regions
        # rather than emit a physically-impossible modulus; `volume_mm3`/
        # `severity` remain meaningful (only the surface-area-derived
        # modulus is the part that diverges), so `macro_shrinkage` still
        # lists these regions unfiltered.
        r["sub_resolution"] = len(r["nodes"]) < 4
    return regions


def is_fed(region: dict, t_s: np.ndarray, tets: np.ndarray,
          feed_nodes: np.ndarray, corridor_frac: float = 0.95) -> bool:
    """OWNER-ADJUDICATED SEMANTICS (2026-08-01, review round 1 finding 1 --
    see the module-level comment above for why the original monotone-escape
    definition was degenerate). A region is fed iff some node of `region`
    and some node of `feed_nodes` (e.g. Adiabatic-tagged riser/sleeve nodes;
    an empty `feed_nodes` -- a bare sand casting with no risers -- always
    gives False, the conservative screen) lie in the same connected
    component of the tet-adjacency graph induced on the "warm corridor":
    nodes with `t_s >= corridor_frac * region["t_max"]`.

    The corridor threshold is relative to the REGION's own peak, not the
    mesh-wide hot cut -- a feed node several hops away through nodes just
    below the region's own hot-selection threshold (but still within
    `corridor_frac` of its peak) now correctly counts as fed, which the old
    "escape the hot superlevel set" definition could never recognize.

    Guard (review round 2, finding NEW-2): if `region["t_max"] <= 0`, the
    corridor threshold `corridor_frac * t_max` would be `<= 0`, which is
    vacuously satisfied by every non-negative `t_s` value (solidification
    times are never negative) -- i.e. the "warm corridor" would silently
    become the WHOLE mesh, making every region trivially "fed" through any
    non-empty `feed_nodes` regardless of actual connectivity. This isn't a
    physically meaningful region in the first place (its peak already
    solidified at t=0), so it's reported unfed (`False`) rather than let the
    threshold degenerate.
    """
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
    """Unfed hot regions (`hot_regions` with its default `top_frac`, filtered
    by `is_fed` against `feed_nodes` -- nodes on Adiabatic-tagged riser/sleeve
    faces, NOT the generic mold/cooling surface), each with `severity =
    shrinkage_pct/100 * volume_mm3` added. `region_id` is propagated
    unchanged from `hot_regions` (its index in the FULL sorted list, not
    renumbered by this function's filtering) so ids stay coherent if this
    output is chained into `top_sinks`/`feeder_recommendations`. A bare sand
    casting with no risers passes an empty `feed_nodes` -- every hot region
    is then unfed, the conservative screen (see `is_fed`).
    """
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
    """Regions whose highest point along `-gravity_vec` is itself a surface
    node sink that free surface (rather than voiding internally) -- returns
    `{"region_id": the region dict's own "region_id" (propagated, NOT a
    positional re-index -- see `hot_regions`/`macro_shrinkage`), "location":
    (3,) mm of that surface node, "volume_mm3": the region's own volume}` for
    each qualifying region.
    """
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
    """Per region: `{"location": surface point above the region's centroid
    along -gravity_vec, "min_modulus_mm": 1.2 * region_modulus, "region_id":
    the region dict's own "region_id" (propagated, NOT a positional
    re-index)}`, where `region_modulus = V/A` (Chvorinov's rule), `A` =
    `_region_surface_area` of the region's own tets, cut at the region's own
    `hot_threshold` (review round 3 -- see `_region_surface_area`'s
    docstring; `t_s` is a NEW required parameter added in that round since
    the marching-tetrahedra area computation needs the continuous field, not
    just the region's own node membership).

    `location` is the surface node minimizing perpendicular distance to the
    vertical line (direction `-gravity_vec`) through the centroid, restricted
    to surface nodes at or above the centroid along that direction (falling
    back to all surface nodes if none qualify, e.g. a degenerate/zero-height
    domain) -- a nearest-existing-mesh-node stand-in for "directly above".

    Raises `ValueError` if `surface_nodes` is empty (no candidate feeder
    location exists -- a mesh with no free surface is not a valid casting).
    A region with zero volume or zero boundary area (a degenerate/thin
    region under the fraction-weighted volume rule) gets a logged warning
    and `min_modulus_mm = 0.0` rather than a silently meaningless value.

    Review finding C1 (sub-resolution regions): a region with fewer than 4
    nodes (`region.get("sub_resolution", False)`, set by `hot_regions`) is
    SKIPPED entirely -- no feeder recommendation is produced for it. For such
    a region the iso-surface area shrinks toward zero as `t_max` approaches
    the region's own selection threshold (an unbounded-in-the-limit
    `M = V/A`, since the lumped-nodal `volume_mm3` stays O(h^3) while the
    true iso-surface area vanishes), so `min_modulus_mm` is not a meaningful
    number at this resolution -- reproduced on the branch's own BUKSA smoke
    artifact (`runs/gate_c/smoke/BUKSA_cleaned_summary.txt`): a 1-node
    region's modulus of 185 mm printed a `min_modulus_mm=222` feeder
    recommendation on a 382x317x254 mm part. `macro_shrinkage` still lists
    these regions (their `volume_mm3`/`severity` are fine -- only the
    surface-area-derived modulus is the part that diverges), and the region
    still contributes to `top_sinks`. The number of regions skipped this way
    is logged once (`logger.info`) rather than per-region, so a part with
    many marginal regions doesn't flood the log.

    Review finding I4 (performance): the mesh's boundary-face structure
    (`_boundary_faces(tets)`) is region-INDEPENDENT and is now computed ONCE
    here and passed to every `_region_surface_area` call, instead of being
    rebuilt (a pure-Python double loop over every tet) once per region --
    see `_region_surface_area`'s own docstring for the before/after cost.
    """
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
    boundary_faces = _boundary_faces(tets)  # I4: computed once, shared by every region

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


# ---------------------------------------------------------------------------
# screen_casting -- one-call orchestrator (Task 3). See
# docs/superpowers/specs/2026-07-24-shrinkage-porosity-screening-design.md
# section "Multi-orientation augmentation": the conduction thermal field (and
# hence Niyama, output 1, and macro-shrinkage region *detection*) does not
# depend on gravity -- only the gravity-conditioned outputs (top sinks and
# feeder placement, outputs 2-4 in the spec's numbering) do. `orientations=
# "axes"` sweeps those over the 6 signed axis directions at zero extra
# thermal-solve cost, per the spec.
# ---------------------------------------------------------------------------

# Signed unit axis directions for the `orientations="axes"` sweep, keyed by
# the label used in `per_orientation`.
_AXIS_SWEEP = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


def screen_casting(pos: np.ndarray, tets: np.ndarray, surface_nodes: np.ndarray,
                   t_s: np.ndarray, TL: float, TS: float, alloy_family: str,
                   gravity_vec=(0, 0, -1), gradT: np.ndarray = None, dTdt: np.ndarray = None,
                   orientations: str = None, feed_nodes: np.ndarray = None) -> dict:
    """One-call shrinkage-porosity screening: Niyama (exact when `gradT`+`dTdt`
    are both given, else the time-only proxy), macro-shrinkage region
    detection, top-surface sinks, and feeder recommendations -- the single
    entry point Task 4's `predict_casting` integration is meant to call.

    `feed_nodes` (BRIEF DEVIATION, owner-adjudicated 2026-08-01): the plan's
    signature (`docs/superpowers/plans/2026-08-01-sp2-shrinkage.md` Task 3)
    does not list this parameter, but `macro_shrinkage`/`is_fed` (Task 2,
    review round 1) require an explicit array of ADIABATIC-tagged riser/
    sleeve nodes to know what "feeds" a region -- this is NOT derivable from
    `surface_nodes` (the whole free surface, used only for sink/feeder
    placement -- a different, larger set that includes ordinary mold-cooling
    surface). Omitting it here would either silently misuse `surface_nodes`
    as feed nodes (every region trivially "self-fed" wherever it touches any
    surface, defeating the whole unfed/fed distinction) or hardcode an empty
    array with no way for a caller to ever supply real risers. `feed_nodes=
    None` defaults to an empty array -- the same "bare sand casting, no
    risers, everything conservatively unfed" convention `macro_shrinkage`/
    `is_fed` already document.

    Niyama: uses `niyama_exact` + `porosity_risk(mode="absolute")` whenever
    both `gradT` and `dTdt` are given (`niyama_source="exact"`); otherwise
    `niyama_proxy` + `porosity_risk(mode="percentile")`
    (`niyama_source="proxy"`) -- see the module docstring for why the two
    paths use different risk-thresholding modes (the proxy is only a
    within-part relative ranking, never comparable to the literature
    absolute thresholds).

    Material: `alloy_family` must be a key of `SHRINKAGE_PCT`. An unknown (or
    missing) family raises a loud `KeyError` naming the known families --
    mirroring `porosity_risk`'s own loud-failure style for the identical
    reason: silently defaulting would silently mis-scale every downstream
    severity/threshold number.

    Orientations: `orientations=None` (default) computes `macro_shrinkage`,
    `top_sinks`, and `feeders` once, using `gravity_vec` as given, and
    `per_orientation` is `None`. `orientations="axes"` additionally sweeps
    `top_sinks`/`feeders` (against the SAME `macro_shrinkage` regions --
    region *detection* is gravity-independent, see the module-level comment
    above) over the 6 signed axis directions, populating `per_orientation`
    with one `{"macro_shrinkage", "top_sinks", "feeders"}` entry per axis
    label (`"+x"`, `"-x"`, `"+y"`, `"-y"`, `"+z"`, `"-z"`); the top-level
    `macro_shrinkage`/`top_sinks`/`feeders` keys are still populated from the
    single `gravity_vec` argument regardless of `orientations`, so the
    `"-z"` entry is numerically identical to calling `screen_casting` again
    with `orientations=None, gravity_vec=(0, 0, -1)` (the default) --
    verified in `tests/test_shrinkage.py`.

    Returns `{"niyama": (N,), "porosity_risk": (N,) bool, "niyama_source":
    str, "macro_shrinkage": [...], "top_sinks": [...], "feeders": [...],
    "per_orientation": {axis: {...}} | None, "material": {"alloy_family",
    "shrinkage_pct"}}`.
    """
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

    if gradT is not None and dTdt is not None:
        ny = niyama_exact(gradT, dTdt)
        risk = porosity_risk(ny, mode="absolute", alloy_family=alloy_family)
        niyama_source = "exact"
    else:
        ny = niyama_proxy(pos, tets, t_s, TL, TS)
        risk = porosity_risk(ny, mode="percentile")
        niyama_source = "proxy"

    shrinkage_pct = SHRINKAGE_PCT[alloy_family]
    macro = macro_shrinkage(pos, tets, t_s, feed_nodes_arr, shrinkage_pct)
    sinks = top_sinks(macro, pos, surface_nodes, gravity_vec)
    feeders = feeder_recommendations(macro, pos, tets, surface_nodes, gravity_vec, t_s)

    per_orientation = None
    if orientations == "axes":
        per_orientation = {}
        for label, axis_vec in _AXIS_SWEEP.items():
            per_orientation[label] = {
                "macro_shrinkage": macro,
                "top_sinks": top_sinks(macro, pos, surface_nodes, axis_vec),
                "feeders": feeder_recommendations(macro, pos, tets, surface_nodes,
                                                  axis_vec, t_s),
            }
    elif orientations is not None:
        raise ValueError(
            f"screen_casting: unknown orientations mode {orientations!r} "
            f"(expected None or 'axes')"
        )

    return {
        "niyama": ny,
        "porosity_risk": risk,
        "niyama_source": niyama_source,
        "macro_shrinkage": macro,
        "top_sinks": sinks,
        "feeders": feeders,
        "per_orientation": per_orientation,
        "material": {"alloy_family": alloy_family, "shrinkage_pct": shrinkage_pct},
    }
