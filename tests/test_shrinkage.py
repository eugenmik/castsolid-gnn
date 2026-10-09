import numpy as np
import pytest

from src.shrinkage import (
    node_gradient, niyama_proxy, porosity_risk,
    hot_regions, is_fed, macro_shrinkage, top_sinks, feeder_recommendations,
    SHRINKAGE_PCT, screen_casting, _boundary_faces,
)


# ---------------------------------------------------------------------------
# Structured tet-grid helpers  -- a Kuhn/Freudenthal 6-tet-per-cube
# decomposition of a regular nx*ny*nz node lattice (unit spacing unless
# overridden). This is a conforming triangulation (verified numerically: tet
# volumes sum exactly to the lattice bounding volume, and internal faces
# between adjacent cubes cancel out of the boundary-surface-area count),
# reused across the region/sink/feeder tests below.
# ---------------------------------------------------------------------------

_KUHN_TETS = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 4, 5, 7),
              (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 6, 7)]


def _node_idx(nx, ny):
    return lambda i, j, k: i + j * nx + k * nx * ny


def _grid_mesh(nx, ny, nz, spacing=1.0):
    idx = _node_idx(nx, ny)
    pos = np.zeros((nx * ny * nz, 3))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                pos[idx(i, j, k)] = [i * spacing, j * spacing, k * spacing]

    def local(i, j, k, bit):
        dx, dy, dz = bit & 1, (bit >> 1) & 1, (bit >> 2) & 1
        return idx(i + dx, j + dy, k + dz)

    tets = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                for t in _KUHN_TETS:
                    tets.append([local(i, j, k, b) for b in t])
    return pos, np.array(tets, dtype=np.int64)


def _surface_nodes(nx, ny, nz):
    idx = _node_idx(nx, ny)
    nodes = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if i in (0, nx - 1) or j in (0, ny - 1) or k in (0, nz - 1):
                    nodes.append(idx(i, j, k))
    return np.array(sorted(nodes), dtype=np.int64)


# ---------------------------------------------------------------------------
# (a) node_gradient
# ---------------------------------------------------------------------------

def test_node_gradient_single_regular_tet():
    # unit tet at the origin: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    tets = np.array([[0, 1, 2, 3]])
    field = 2.0 * pos[:, 0]  # f = 2x -> grad = (2, 0, 0) everywhere
    g = node_gradient(pos, tets, field)
    assert g.shape == (4, 3)
    np.testing.assert_allclose(g, np.tile([2.0, 0.0, 0.0], (4, 1)), atol=1e-10)


def test_node_gradient_two_tet_bipyramid():
    # two tets sharing triangular face ABC, apexes D (above) and E (below)
    # A, B, C, D, E -> indices 0..4
    pos = np.array([
        [0, 0, 0],   # A
        [1, 0, 0],   # B
        [0, 1, 0],   # C
        [0, 0, 1],   # D (apex above)
        [0, 0, -1],  # E (apex below)
    ], float)
    tets = np.array([
        [0, 1, 2, 3],  # A B C D
        [0, 1, 2, 4],  # A B C E
    ])
    field = pos[:, 0] + pos[:, 1]  # f = x + y -> grad = (1, 1, 0) everywhere (exact, linear)
    g = node_gradient(pos, tets, field)
    assert g.shape == (5, 3)
    np.testing.assert_allclose(g, np.tile([1.0, 1.0, 0.0], (5, 1)), atol=1e-10)


def test_node_gradient_no_tets_returns_zeros():
    pos = np.zeros((3, 3))
    tets = np.empty((0, 4), dtype=np.int64)
    field = np.zeros(3)
    g = node_gradient(pos, tets, field)
    np.testing.assert_allclose(g, np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# (b) niyama_exact
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# (c) niyama_proxy
# ---------------------------------------------------------------------------

def test_niyama_proxy_linear_slab_hand_computed():
    # single regular tet, t_s = 2 + 2x (linear ramp along x) -> |grad t_s| = 2 exactly
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    tets = np.array([[0, 1, 2, 3]])
    t_s = np.array([2.0, 4.0, 2.0, 2.0])  # node0=2 (x=0), node1=4 (x=1), node2/3 at x=0 -> 2
    TL, TS = 1400.0, 1100.0  # TL - TS = 300

    ny = niyama_proxy(pos, tets, t_s, TL, TS)
    assert ny.shape == (4,)

    # hand computation: sqrt((TL-TS)/t_s) * |grad t_s|, |grad t_s| = 2 at every node
    expected_node0 = np.sqrt(300.0 / 2.0) * 2.0  # ~24.494897...
    expected_node1 = np.sqrt(300.0 / 4.0) * 2.0  # ~17.320508...
    np.testing.assert_allclose(ny[0], expected_node0, rtol=1e-10)
    np.testing.assert_allclose(ny[1], expected_node1, rtol=1e-10)


def test_niyama_proxy_eps_guarded_at_zero_t_s():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    tets = np.array([[0, 1, 2, 3]])
    t_s = np.array([0.0, 2.0, 0.0, 0.0])
    ny = niyama_proxy(pos, tets, t_s, 1400.0, 1100.0)
    assert np.isfinite(ny).all()


# ---------------------------------------------------------------------------
# (d) porosity_risk -- absolute mode
# ---------------------------------------------------------------------------











# ---------------------------------------------------------------------------
# (e) porosity_risk -- percentile mode
# ---------------------------------------------------------------------------

def test_porosity_risk_percentile_exact_bottom_10pct():
    ny = np.arange(20, dtype=float)  # 0..19, no ties
    mask = porosity_risk(ny, pct=10.0)
    assert mask.sum() == 2  # exactly bottom 10% of 20 = 2
    np.testing.assert_array_equal(np.where(mask)[0], [0, 1])


def test_porosity_risk_percentile_exact_bottom_10pct_n10():
    ny = np.arange(10, dtype=float)  # 0..9
    mask = porosity_risk(ny, pct=10.0)
    assert mask.sum() == 1
    np.testing.assert_array_equal(np.where(mask)[0], [0])


def test_porosity_risk_percentile_tie_handling_documented():
    # duplicate values straddling the percentile threshold: both ties are
    # flagged (mask is defined as ny <= percentile(ny, pct), so a tie AT the
    # threshold is always included) -- this can flag slightly MORE than the
    # nominal pct% when duplicates sit at the boundary. Documented behavior,
    # not a bug: deterministic and order-independent.
    ny = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    mask = porosity_risk(ny, pct=10.0)
    assert mask.sum() == 2
    np.testing.assert_array_equal(np.where(mask)[0], [0, 1])




# ---------------------------------------------------------------------------
# (a) hot_regions + is_fed -- isolated boss (unfed) vs. surface-connected
# band (fed), on one 7x7x7 mesh with two disjoint elevated node clusters.
# ---------------------------------------------------------------------------

def _boss_and_band_mesh():
    nx = ny = nz = 7
    pos, tets = _grid_mesh(nx, ny, nz)
    idx = _node_idx(nx, ny)
    n_nodes = pos.shape[0]
    t_s = np.zeros(n_nodes)

    # boss: a fully-interior unit cube {2,3}x{2,3}x{2,3} -- 8 nodes, none on
    # the (0 or 6) domain boundary, and not adjacent to the band below.
    boss_nodes = [idx(i, j, k) for i in (2, 3) for j in (2, 3) for k in (2, 3)]
    # band: a column at x=5,y=5, z=3..6 -- z=6 sits on the top boundary face,
    # so the band directly touches a free surface.
    band_nodes = [idx(5, 5, z) for z in (3, 4, 5, 6)]

    t_s[boss_nodes] = 100.0
    t_s[band_nodes] = 100.0
    n_hot = len(boss_nodes) + len(band_nodes)
    top_frac = n_hot / n_nodes  # sized so the percentile cut lands exactly on {0, 100}

    surface_nodes = _surface_nodes(nx, ny, nz)
    return pos, tets, t_s, top_frac, surface_nodes, set(boss_nodes), set(band_nodes)


def test_hot_regions_finds_boss_and_band_as_separate_components():
    pos, tets, t_s, top_frac, surface_nodes, boss_nodes, band_nodes = _boss_and_band_mesh()
    regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
    assert len(regions) == 2
    node_sets = [set(r["nodes"].tolist()) for r in regions]
    assert boss_nodes in node_sets
    assert band_nodes in node_sets
    for r in regions:
        assert r["t_max"] == 100.0
        np.testing.assert_allclose(r["centroid"], pos[np.array(sorted(r["nodes"]))].mean(axis=0))


def test_hot_regions_sorted_by_t_max_desc():
    pos, tets, t_s, top_frac, surface_nodes, boss_nodes, band_nodes = _boss_and_band_mesh()
    # bump one region's peak so the sort order is unambiguous.
    idx = _node_idx(7, 7)
    t_s = t_s.copy()
    t_s[idx(5, 5, 6)] = 150.0
    regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
    t_maxes = [r["t_max"] for r in regions]
    assert t_maxes == sorted(t_maxes, reverse=True)
    assert t_maxes[0] == 150.0


def test_is_fed_isolated_boss_is_false():
    # feed_nodes = surface_nodes here (this mesh's domain boundary stands in
    # for a riser/feed location) -- the boss never reaches the corridor
    # threshold (0.95*t_max) outside its own 8 nodes, so it can't connect.
    pos, tets, t_s, top_frac, surface_nodes, boss_nodes, band_nodes = _boss_and_band_mesh()
    regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
    boss_region = next(r for r in regions if set(r["nodes"].tolist()) == boss_nodes)
    assert is_fed(boss_region, t_s, tets, feed_nodes=surface_nodes) is False


def test_is_fed_surface_connected_band_is_true():
    # the band's own top node (z=6) is itself in feed_nodes=surface_nodes,
    # so the region trivially self-connects to a feed node.
    pos, tets, t_s, top_frac, surface_nodes, boss_nodes, band_nodes = _boss_and_band_mesh()
    regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
    band_region = next(r for r in regions if set(r["nodes"].tolist()) == band_nodes)
    assert is_fed(band_region, t_s, tets, feed_nodes=surface_nodes) is True


def test_is_fed_empty_feed_nodes_is_always_false():
    pos, tets, t_s, top_frac, surface_nodes, boss_nodes, band_nodes = _boss_and_band_mesh()
    regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
    band_region = next(r for r in regions if set(r["nodes"].tolist()) == band_nodes)
    # bare sand casting convention: no risers -> empty feed_nodes -> unfed,
    # even for a region that DOES touch the domain surface.
    assert is_fed(band_region, t_s, tets, feed_nodes=np.array([], dtype=np.int64)) is False


# --------------------------------------------------------------------------- #
# Marginal ("sub-resolution", <4 nodes) hot regions mix a lumped-nodal V
# (stays O(h^3)) with a true iso-surface A (shrinks toward zero as t_max ->
# hot_threshold), so M = V/A diverges without bound. `hot_regions` marks such regions
# "sub_resolution": True; `feeder_recommendations` skips them (logging the
# count once); `macro_shrinkage` still lists them (volume/severity remain
# meaningful, only the modulus diverges).
# --------------------------------------------------------------------------- #
def _single_isolated_hot_node_mesh():
    """A 7x7x7 grid mesh with exactly ONE hot node (fully interior, no hot
    neighbors) -- the minimal fixture for a 1-node `hot_regions` component."""
    nx = ny = nz = 7
    pos, tets = _grid_mesh(nx, ny, nz)
    idx = _node_idx(nx, ny)
    n_nodes = pos.shape[0]
    t_s = np.zeros(n_nodes)
    hot_node = idx(3, 3, 3)
    t_s[hot_node] = 100.0
    top_frac = 1.0 / n_nodes
    surface_nodes = _surface_nodes(nx, ny, nz)
    return pos, tets, t_s, top_frac, surface_nodes, hot_node


def test_hot_regions_marks_sub_resolution_for_regions_under_4_nodes():
    pos, tets, t_s, top_frac, surface_nodes, hot_node = _single_isolated_hot_node_mesh()
    regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
    assert len(regions) == 1
    assert regions[0]["nodes"].tolist() == [hot_node]
    assert regions[0]["sub_resolution"] is True


def test_hot_regions_does_not_mark_well_resolved_regions_sub_resolution():
    # the boss (8 nodes) and band (4 nodes) fixture -- both >= the 4-node floor.
    pos, tets, t_s, top_frac, surface_nodes, boss_nodes, band_nodes = _boss_and_band_mesh()
    regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
    assert len(regions) == 2
    for r in regions:
        assert r["sub_resolution"] is False


def test_feeder_recommendations_skips_sub_resolution_region_and_logs_count(caplog):
    """A 1-node region must get NO feeder recommendation at all, and the
    skip must be logged (once, with the count) rather than silent."""
    import logging as _logging
    pos, tets, t_s, top_frac, surface_nodes, hot_node = _single_isolated_hot_node_mesh()
    regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
    assert regions[0]["sub_resolution"] is True

    with caplog.at_level(_logging.INFO, logger="src.shrinkage"):
        recs = feeder_recommendations(regions, pos, tets, surface_nodes,
                                      gravity_vec=(0, 0, -1), t_s=t_s)

    assert recs == []
    assert any("skipped 1 sub-resolution region" in rec.message for rec in caplog.records)


def test_feeder_recommendations_mixed_regions_skips_only_sub_resolution_ones(caplog):
    """A regions list mixing well-resolved regions (get a feeder rec each)
    and a sub-resolution one (skipped, not just de-prioritized) -- the
    well-resolved regions' recs must be unaffected by the mix."""
    import logging as _logging
    pos, tets, t_s, top_frac, surface_nodes, boss_nodes, band_nodes = _boss_and_band_mesh()
    regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
    assert all(not r["sub_resolution"] for r in regions)

    sub_region = {
        "nodes": np.array([0]), "centroid": pos[0], "volume_mm3": 1.0,
        "t_max": float(t_s.max()), "region_id": 99,
        "hot_threshold": regions[0]["hot_threshold"], "sub_resolution": True,
    }
    mixed_regions = regions + [sub_region]

    with caplog.at_level(_logging.INFO, logger="src.shrinkage"):
        recs = feeder_recommendations(mixed_regions, pos, tets, surface_nodes,
                                      gravity_vec=(0, 0, -1), t_s=t_s)

    assert len(recs) == len(regions)
    assert {r["region_id"] for r in recs} == {r["region_id"] for r in regions}
    assert any("skipped 1 sub-resolution region" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# the mesh's boundary-face structure is
# region-independent and hoisted out of `_region_surface_area`'s per-call
# work into a standalone `_boundary_faces(tets)` helper, computed once by
# `feeder_recommendations` and shared across every region. A direct check
# of the helper on a hand-countable 2-tet mesh (one shared interior face).
# --------------------------------------------------------------------------- #
def test_boundary_faces_excludes_shared_interior_face():
    # two tets sharing face (0,1,2) -- a triangular bipyramid.
    tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64)
    bf = _boundary_faces(tets)
    assert bf.shape == (6, 3)
    boundary_keys = {tuple(sorted(row.tolist())) for row in bf}
    assert (0, 1, 2) not in boundary_keys  # the shared interior face is excluded
    expected = {(0, 1, 3), (0, 2, 3), (1, 2, 3),
               (0, 1, 4), (0, 2, 4), (1, 2, 4)}
    assert boundary_keys == expected


def test_boundary_faces_empty_tets_gives_empty_array():
    bf = _boundary_faces(np.zeros((0, 4), dtype=np.int64))
    assert bf.shape == (0, 3)


def test_hot_regions_top_frac_out_of_range_raises():
    pos, tets, t_s, top_frac, surface_nodes, boss_nodes, band_nodes = _boss_and_band_mesh()
    with pytest.raises(ValueError):
        hot_regions(pos, tets, t_s, top_frac=0.0)
    with pytest.raises(ValueError):
        hot_regions(pos, tets, t_s, top_frac=1.5)


# ---------------------------------------------------------------------------
# (a2) is_fed corridor semantics: a warm corridor (t within `corridor_frac` of the region's own
# t_max) bridges the region to a feed node several hops away, NOT required
# to be part of the region's own hot-node set. This is the
# REQUIRED test: it must fail both under a `return False` stub (variant 1
# expects True) and under a naive "touch test" that only checks direct
# region-to-feed_node adjacency or membership (variant 1's feed node is 6
# hops from the region, connected only via the intermediate warm corridor).
# ---------------------------------------------------------------------------

def _corridor_chain_mesh():
    """8x2x2 lattice (32 nodes): an x-adjacent chain of nodes (i,0,0),
    i=0..7 (verified consecutively tet-adjacent). Node 6 will be the sole
    hot region (t_max=100); nodes 0-5 are the candidate corridor (0 doubles
    as the feed node); node 7 is the "far cold side" used as a decoy feed
    node in variant 3. Every other mesh node stays at background 0 and
    never participates (present only so this is a valid tet mesh)."""
    nx, ny, nz = 8, 2, 2
    pos, tets = _grid_mesh(nx, ny, nz)
    idx = _node_idx(nx, ny)
    chain = [idx(i, 0, 0) for i in range(nx)]
    n_nodes = pos.shape[0]
    return pos, tets, n_nodes, chain


def test_is_fed_warm_corridor_bridges_region_to_distant_feed_node():
    pos, tets, n_nodes, chain = _corridor_chain_mesh()
    t_s = np.zeros(n_nodes)
    t_s[chain[6]] = 100.0       # region peak
    t_s[chain[0:6]] = 96.0      # corridor >= 0.95*100 = 95 -- stays warm
    region = {"nodes": np.array([chain[6]]), "t_max": 100.0}
    assert is_fed(region, t_s, tets, feed_nodes=np.array([chain[0]])) is True


def test_is_fed_false_when_corridor_cools_below_corridor_frac():
    pos, tets, n_nodes, chain = _corridor_chain_mesh()
    t_s = np.zeros(n_nodes)
    t_s[chain[6]] = 100.0
    t_s[chain[0:6]] = 90.0      # < 95 -- corridor no longer warm enough
    region = {"nodes": np.array([chain[6]]), "t_max": 100.0}
    assert is_fed(region, t_s, tets, feed_nodes=np.array([chain[0]])) is False


def test_is_fed_false_when_feed_node_is_on_far_cold_side():
    pos, tets, n_nodes, chain = _corridor_chain_mesh()
    t_s = np.zeros(n_nodes)
    t_s[chain[6]] = 100.0
    t_s[chain[0:6]] = 96.0      # corridor stays warm...
    region = {"nodes": np.array([chain[6]]), "t_max": 100.0}
    # ...but feed_nodes points to the far (cold, disconnected) side instead.
    assert is_fed(region, t_s, tets, feed_nodes=np.array([chain[7]])) is False


def test_is_fed_region_t_max_zero_is_unfed_not_vacuously_true():
    """a region t_max <= 0 would make the
    corridor threshold (corridor_frac * t_max) <= 0, vacuously satisfied by
    every non-negative t_s -- i.e. the whole (connected) mesh would count as
    "warm", trivially fed via any non-empty feed_nodes regardless of real
    connectivity. Guarded to report unfed instead."""
    pos, tets, n_nodes, chain = _corridor_chain_mesh()
    t_s = np.zeros(n_nodes)  # every node, including all feed candidates, is 0
    region = {"nodes": np.array([chain[6]]), "t_max": 0.0}
    # without the guard, corridor threshold = 0.95*0 = 0 <= every t_s -> the
    # whole mesh (one connected component) would trivially "fed" this region.
    assert is_fed(region, t_s, tets, feed_nodes=np.array([chain[0]])) is False


# ---------------------------------------------------------------------------
# (b) macro_shrinkage -- severity = pct/100 * volume, hand-checked; grey_iron
# (pct=0) still reports the region with zero severity; fed regions excluded.
# ---------------------------------------------------------------------------

def _interior_cube_block_mesh():
    """9x6x5 lattice with a fully-interior 3x3x3 node block (a 2x2x2 mm solid
    cube of 8 sub-cells). Under the fraction-weighted volume rule, volume_mm3 = 27.0 exactly: each of the 27 hot nodes
    is fully interior with a complete 8-cube neighborhood, so its lumped
    nodal volume (sum of tet_volume/4 over incident tets) is exactly 1 mm^3
    on this unit-spacing lattice -- hand-verified numerically (independently
    confirmed: fraction-weighted volume == hot-node COUNT here, since
    summing volume*n_hot/4 over all tets is algebraically identical to
    summing each hot node's own lumped volume). Boundary area is now `_region_surface_area`'s co-area/domain-boundary
    formula -- see that function's docstring -- which for this fixture comes
    to ~49.31 mm^2 (M = V/A ~= 0.5475, within 15% of the idealized "3x3x3 mm
    dilated cube" reference of A=54/M=0.5 derived by hand; an
    exact closed-form match isn't expected for a graded/partial-tet field --
    see `test_feeder_recommendations_modulus_exact` and the mesh-refinement
    drift test below for the actual acceptance checks). The block is exactly
    27/270 = 10% of nodes, matching hot_regions'/macro_shrinkage's default
    top_frac."""
    nx, ny, nz = 9, 6, 5
    pos, tets = _grid_mesh(nx, ny, nz)
    idx = _node_idx(nx, ny)
    n_nodes = pos.shape[0]
    t_s = np.zeros(n_nodes)
    block_nodes = [idx(i, j, k) for i in (3, 4, 5) for j in (1, 2, 3) for k in (1, 2, 3)]
    t_s[block_nodes] = 100.0
    surface_nodes = _surface_nodes(nx, ny, nz)
    assert len(block_nodes) / n_nodes == pytest.approx(0.10)
    assert not any(n in set(surface_nodes.tolist()) for n in block_nodes)
    return pos, tets, t_s, surface_nodes, set(block_nodes)


def test_macro_shrinkage_severity_hand_checked():
    pos, tets, t_s, surface_nodes, block_nodes = _interior_cube_block_mesh()
    # bare-casting convention: no risers -> empty feed_nodes -> conservative
    # (this fully-interior block is unfed regardless, since it never reaches
    # the domain surface anyway -- see is_fed's docstring).
    no_feed = np.array([], dtype=np.int64)
    regions = macro_shrinkage(pos, tets, t_s, no_feed, shrinkage_pct=2.5)
    assert len(regions) == 1
    r = regions[0]
    assert set(r["nodes"].tolist()) == block_nodes
    assert r["volume_mm3"] == pytest.approx(27.0)
    assert r["severity"] == pytest.approx(2.5 / 100.0 * 27.0)


def test_macro_shrinkage_grey_iron_zero_severity_still_reported():
    pos, tets, t_s, surface_nodes, block_nodes = _interior_cube_block_mesh()
    no_feed = np.array([], dtype=np.int64)
    regions = macro_shrinkage(pos, tets, t_s, no_feed,
                              shrinkage_pct=SHRINKAGE_PCT["grey_iron"])
    assert len(regions) == 1
    assert regions[0]["severity"] == pytest.approx(0.0)
    assert regions[0]["volume_mm3"] == pytest.approx(27.0)  # region still fully reported


def _unfed_boss_and_fed_band_10pct_mesh():
    """10x5x4 lattice (200 nodes) with an interior unit-cube boss (8 nodes,
    t_max=100, unfed) and a corner block band (12 nodes, t_max=150, touching
    x=0/y=0/z=0 boundary faces) -- 20/200 = exactly 10%, matching
    macro_shrinkage's (un-parametrized) default top_frac so BOTH regions are
    picked up by its internal hot_regions() call. `feed_nodes=surface_nodes`
    is used by the tests below as a stand-in "this domain boundary carries
    risers" convention: the band's own boundary-touching nodes are literally
    IN `surface_nodes`, so it self-connects trivially (fed); the boss's
    warm corridor (0.95*its own t_max=100 -> threshold 95) never leaves its
    own 8 nodes (all neighbors are background 0), so it cannot reach the
    boundary (unfed) -- hand-verified via the boss/band-touches-surface
    assertions below. Band's t_max is deliberately HIGHER than boss's so
    hot_regions sorts band first (region_id=0) and boss second (region_id=1)
    -- this exposes a positional-reindex bug (region_id coherence) that a
    same-valued t_max tie would hide."""
    nx, ny, nz = 10, 5, 4
    pos, tets = _grid_mesh(nx, ny, nz)
    idx = _node_idx(nx, ny)
    n_nodes = pos.shape[0]
    t_s = np.zeros(n_nodes)

    boss_nodes = [idx(i, j, k) for i in (4, 5) for j in (1, 2) for k in (1, 2)]
    band_nodes = [idx(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1, 2)]
    t_s[boss_nodes] = 100.0
    t_s[band_nodes] = 150.0
    assert (len(boss_nodes) + len(band_nodes)) / n_nodes == pytest.approx(0.10)

    surface_nodes = _surface_nodes(nx, ny, nz)
    surface_set = set(surface_nodes.tolist())
    assert not any(n in surface_set for n in boss_nodes)
    assert any(n in surface_set for n in band_nodes)
    return pos, tets, t_s, surface_nodes, set(boss_nodes), set(band_nodes)


def test_macro_shrinkage_excludes_fed_regions():
    pos, tets, t_s, surface_nodes, boss_nodes, band_nodes = _unfed_boss_and_fed_band_10pct_mesh()
    regions = macro_shrinkage(pos, tets, t_s, feed_nodes=surface_nodes, shrinkage_pct=2.5)
    assert len(regions) == 1
    assert set(regions[0]["nodes"].tolist()) == boss_nodes


def test_region_id_coherence_across_pipeline():
    """region_id must be the STABLE hot_regions index, propagated (not
    positionally re-derived) through macro_shrinkage's filtering and further
    through top_sinks/feeder_recommendations."""
    pos, tets, t_s, surface_nodes, boss_nodes, band_nodes = _unfed_boss_and_fed_band_10pct_mesh()

    raw_regions = hot_regions(pos, tets, t_s)
    assert len(raw_regions) == 2
    boss_raw = next(r for r in raw_regions if set(r["nodes"].tolist()) == boss_nodes)
    band_raw = next(r for r in raw_regions if set(r["nodes"].tolist()) == band_nodes)
    # band's t_max (150) > boss's (100) -> band sorts first.
    assert band_raw["region_id"] == 0
    assert boss_raw["region_id"] == 1

    macro = macro_shrinkage(pos, tets, t_s, feed_nodes=surface_nodes, shrinkage_pct=2.5)
    assert len(macro) == 1  # band filtered out (fed); a naive positional
    # re-index would mislabel the sole survivor as region_id 0.
    assert macro[0]["region_id"] == 1

    sinks = top_sinks(macro, pos, surface_nodes, gravity_vec=(0, 0, -1))
    feeders = feeder_recommendations(macro, pos, tets, surface_nodes, gravity_vec=(0, 0, -1),
                                     t_s=t_s)
    assert all(f["region_id"] == 1 for f in feeders)
    assert all(s["region_id"] == 1 for s in sinks)


def test_macro_shrinkage_verdict_flips_via_genuine_external_corridor():
    """an end-to-end (not just is_fed-level)
    demonstration that a region's fed/unfed verdict can flip because of a
    node that is NOT part of the region (hot_regions' own top-10% cut
    excludes it) and NOT the feed node itself, but sits between
    `0.95*t_max` and the hot cut and bridges to a feed node -- i.e. a
    genuine warm-corridor effect, not the trivial "feed node happens to be
    inside the region" case exercised by `test_macro_shrinkage_excludes_
    fed_regions`. Peak (node 1), bridge (node 3), and feed (node 2) are
    chosen so the feed node is NOT itself tet-adjacent to the region peak
    (hand-verified: nodes 1 and 2 share no tet in this mesh's Kuhn
    decomposition) -- the bridge node is structurally required to connect
    them, unlike a single-hop setup where a direct peak-feed edge could
    make the test pass for the wrong reason. A single unit-cube mesh (N=8)
    is used because hot_regions' un-parametrized default top_frac=0.10 only
    isolates a lone peak node cleanly at small N (verified: at N=12+ the
    corridor value gets pulled into the "hot" selection itself by
    percentile interpolation, defeating the "external, not region-touching"
    setup)."""
    pos, tets = _grid_mesh(2, 2, 2)
    n_nodes = pos.shape[0]
    t_s = np.zeros(n_nodes)
    t_s[1] = 100.0   # sole region peak
    t_s[3] = 96.0    # bridge node: >= 0.95*100=95, but < the 97.2 hot cut
    t_s[2] = 96.0    # feed node: same band; NOT tet-adjacent to node 1 directly

    regions = hot_regions(pos, tets, t_s)  # default top_frac=0.10
    assert len(regions) == 1
    assert regions[0]["nodes"].tolist() == [1]  # bridge/feed excluded from the region itself

    fed_via_corridor = macro_shrinkage(pos, tets, t_s, feed_nodes=np.array([2]),
                                       shrinkage_pct=2.5)
    assert fed_via_corridor == []  # fed -> not reported as macro-shrinkage

    unfed_without_corridor = macro_shrinkage(pos, tets, t_s, feed_nodes=np.array([], dtype=np.int64),
                                             shrinkage_pct=2.5)
    assert len(unfed_without_corridor) == 1
    assert unfed_without_corridor[0]["nodes"].tolist() == [1]

    # cooling the bridge/feed band below 0.95*t_max flips the verdict back
    # to unfed even with the SAME feed_nodes -- confirms the flip is driven
    # by the corridor threshold, not merely feed_nodes' presence.
    t_s_cold_corridor = t_s.copy()
    t_s_cold_corridor[[2, 3]] = 90.0
    cooled = macro_shrinkage(pos, tets, t_s_cold_corridor, feed_nodes=np.array([2]),
                             shrinkage_pct=2.5)
    assert len(cooled) == 1


def test_shrinkage_pct_table():
    assert SHRINKAGE_PCT == {
        "steel": 2.5, "low_alloy_steel": 2.5, "ductile_iron": 0.75, "grey_iron": 0.0,
    }


# ---------------------------------------------------------------------------
# (c) top_sinks -- gravity -z -> sink on the top (z-max) face; rotating
# gravity to -x/-y relocates the sink to the +x/+y face. Single 3x3x3 mesh,
# one hot region spanning a top-face node, an interior bridge node, an
# x-face node, and a y-face node (all connected via tet adjacency through
# the bridge) -- also reused by feeder_recommendations' rotation tests below
#.
# ---------------------------------------------------------------------------

def _rotatable_sink_mesh():
    nx = ny = nz = 3
    pos, tets = _grid_mesh(nx, ny, nz)
    idx = _node_idx(nx, ny)
    n_nodes = pos.shape[0]
    t_s = np.zeros(n_nodes)

    node_top = idx(1, 1, 2)     # z=2 -> top (z-max) surface face
    node_bridge = idx(1, 1, 1)  # interior
    node_xface = idx(2, 1, 1)   # x=2 -> +x surface face
    node_yface = idx(1, 2, 1)   # y=2 -> +y surface face

    hot = [node_top, node_bridge, node_xface, node_yface]
    t_s[hot] = 100.0
    top_frac = len(hot) / n_nodes
    surface_nodes = _surface_nodes(nx, ny, nz)
    regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
    assert len(regions) == 1
    return pos, tets, t_s, regions, surface_nodes, node_top, node_xface, node_yface


def test_top_sinks_default_gravity_hits_top_face():
    pos, tets, t_s, regions, surface_nodes, node_top, node_xface, node_yface = _rotatable_sink_mesh()
    sinks = top_sinks(regions, pos, surface_nodes, gravity_vec=(0, 0, -1))
    assert len(sinks) == 1
    np.testing.assert_allclose(sinks[0]["location"], pos[node_top])


def test_top_sinks_rotated_gravity_relocates_to_x_face():
    pos, tets, t_s, regions, surface_nodes, node_top, node_xface, node_yface = _rotatable_sink_mesh()
    sinks = top_sinks(regions, pos, surface_nodes, gravity_vec=(-1, 0, 0))
    assert len(sinks) == 1
    np.testing.assert_allclose(sinks[0]["location"], pos[node_xface])


# ---------------------------------------------------------------------------
# (d) feeder_recommendations -- min_modulus == 1.2*(V/A); location on the
# top surface above the region's centroid; ValueError on empty surface_nodes;
# a degenerate (zero-volume/area) region logs a warning instead of a silent 0.
#
# `_region_surface_area` now runs marching tetrahedra on the
# CONTINUOUS `t_s` field cut at the region's own `hot_threshold`, replacing
# round 2's co-area formula on a BINARIZED indicator -- the binarized version
# always crosses each tet edge at its midpoint regardless of where the real
# field crosses its OWN selection threshold, which is an exact calculus
# identity for the (artificial) binary step but measures the anisotropic
# discrete perimeter of that step, not the true field's level-surface area:
# it plateaus ~24-29% high on curved/diagonal boundaries (only looking exact
# on an axis-aligned cut, where a midpoint crossing happens to coincide with
# the true one). `feeder_recommendations` gained a required `t_s` parameter
# in this round since the marching-tetrahedra computation needs the
# continuous field, not just region-node membership.
# ---------------------------------------------------------------------------

def _reference_region_area_marching_tets(pos, tets, nodes, t_s, threshold):
    """Independent reimplementation of the marching-tetrahedra + domain-
    boundary-face area formula (mirrors, but does not call,
    `src.shrinkage._region_surface_area` -- a genuine hand-check against the
    same algorithm, not a tautological re-invocation of the code under
    test)."""
    n_nodes = pos.shape[0]
    is_member = np.zeros(n_nodes, dtype=bool)
    is_member[np.asarray(nodes, dtype=np.int64)] = True
    hot_mask = t_s >= threshold

    area = 0.0
    member_per_tet = hot_mask[tets]
    n_above = member_per_tet.sum(axis=1)
    for ti in np.flatnonzero((n_above >= 1) & (n_above <= 3)):
        tet = tets[ti]
        if not is_member[tet].any():
            continue
        verts = pos[tet]
        vals = t_s[tet]
        above = member_per_tet[ti]
        k = int(n_above[ti])
        if k in (1, 3):
            odd = int(np.flatnonzero(above)[0]) if k == 1 else int(np.flatnonzero(~above)[0])
            others = [i for i in range(4) if i != odd]
            pts = []
            for o in others:
                frac = (threshold - vals[odd]) / (vals[o] - vals[odd])
                pts.append(verts[odd] + frac * (verts[o] - verts[odd]))
            area += 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        else:
            hi = np.flatnonzero(above)
            lo = np.flatnonzero(~above)

            def _cp(h, c, verts=verts, vals=vals):
                frac = (threshold - vals[h]) / (vals[c] - vals[h])
                return verts[h] + frac * (verts[c] - verts[h])

            a_pt, b_pt = _cp(hi[0], lo[0]), _cp(hi[0], lo[1])
            c_pt, d_pt = _cp(hi[1], lo[1]), _cp(hi[1], lo[0])
            area += 0.5 * np.linalg.norm(np.cross(c_pt - a_pt, d_pt - b_pt))

    face_count = {}
    for tet in tets:
        for f in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            key = tuple(sorted(tet[list(f)].tolist()))
            face_count[key] = face_count.get(key, 0) + 1
    for key, cnt in face_count.items():
        if cnt == 1:
            v = np.array(key)
            frac = is_member[v].sum() / 3.0
            if frac > 0:
                q0, q1, q2 = pos[v[0]], pos[v[1]], pos[v[2]]
                area += frac * 0.5 * np.linalg.norm(np.cross(q1 - q0, q2 - q0))
    return area


def _interior_chebyshev_cube_mesh():
    """9x6x5 lattice (same domain as `_interior_cube_block_mesh`) with a
    CONTINUOUS Chebyshev-distance field `t_s = -max(|x-cx|,|y-cy|,|z-cz|)`
    centered at (4,2,2) -- the level sets of this field are EXACT axis-
    aligned cubes at any threshold (unlike the binary 0/100 step used
    elsewhere in this file, whose arbitrary node-count-derived threshold
    has no meaningful relationship to a genuine half-integer offset). At
    threshold=-1.5 (exactly bisecting the "distance 1" hot shell and the
    "distance 2" next shell), `t_s >= -1.5` selects the SAME 27-node block
    as `_interior_cube_block_mesh` (hand-verified) -- reusing `hot_regions`
    with top_frac=27/270 reproduces this node set and the correct volume/
    centroid/t_max/region_id, but its OWN `hot_threshold` lands at -1.9 (a
    percentile-interpolation artifact of this exact node-count split, not a
    physically meaningful cut) -- overridden to the analytically clean -1.5
    for this specific hand-check, isolating the area ALGORITHM's behavior at
    a genuinely representative threshold position from `hot_regions`' own
    (separately tested) node-selection quirks."""
    nx, ny, nz = 9, 6, 5
    pos, tets = _grid_mesh(nx, ny, nz)
    center = np.array([4.0, 2.0, 2.0])
    t_s = -np.max(np.abs(pos - center), axis=1)
    surface_nodes = _surface_nodes(nx, ny, nz)

    regions = hot_regions(pos, tets, t_s, top_frac=27.0 / 270.0)
    assert len(regions) == 1
    region = dict(regions[0])
    assert region["volume_mm3"] == pytest.approx(27.0)
    region["hot_threshold"] = -1.5  # see docstring: the clean, symmetric cut
    return pos, tets, t_s, surface_nodes, region


def test_feeder_recommendations_modulus_level_set_consistent():
    """V and A must measure the same body via a genuine
    continuous-field level set (not an artificially-thresholded binary
    block). On this fixture the true level set is an exact 3x3x3mm cube
    (M = side/6 = 0.5 by the Chvorinov identity for any cube); marching
    tetrahedra on a real linear field still chamfers cube corners/edges
    (hand-verified: the same chamfering appears in an independent marching-
    tets reimplementation, not just the shipped code), so M lands near but
    not exactly at 0.5 -- within the stated 15% band."""
    pos, tets, t_s, surface_nodes, region = _interior_chebyshev_cube_mesh()
    recs = feeder_recommendations([region], pos, tets, surface_nodes, gravity_vec=(0, 0, -1),
                                  t_s=t_s)
    assert len(recs) == 1

    expected_area = _reference_region_area_marching_tets(pos, tets, region["nodes"], t_s,
                                                         region["hot_threshold"])
    expected_modulus = region["volume_mm3"] / expected_area
    assert recs[0]["min_modulus_mm"] == pytest.approx(1.2 * expected_modulus, rel=1e-9)
    assert recs[0]["region_id"] == region["region_id"]

    M = recs[0]["min_modulus_mm"] / 1.2
    assert abs(M - 0.5) / 0.5 < 0.15, f"M={M} not within 15% of the Chvorinov M=0.5 cube reference"


def test_feeder_recommendations_location_on_top_surface_above_centroid():
    pos, tets, t_s, surface_nodes, block_nodes = _interior_cube_block_mesh()
    regions = hot_regions(pos, tets, t_s)
    recs = feeder_recommendations(regions, pos, tets, surface_nodes, gravity_vec=(0, 0, -1),
                                  t_s=t_s)
    loc = recs[0]["location"]
    centroid = regions[0]["centroid"]

    # the recommended point must itself be a surface node...
    surface_pos = pos[surface_nodes]
    assert np.any(np.all(np.isclose(surface_pos, loc), axis=1))
    # ...directly above the centroid (matching x,y; z at the domain's top face)
    np.testing.assert_allclose(loc[:2], centroid[:2])
    assert loc[2] == pos[:, 2].max()


def test_feeder_recommendations_gravity_rotation_relocates_location():
    pos, tets, t_s, regions, surface_nodes, node_top, node_xface, node_yface = _rotatable_sink_mesh()

    recs_z = feeder_recommendations(regions, pos, tets, surface_nodes, gravity_vec=(0, 0, -1),
                                    t_s=t_s)
    np.testing.assert_allclose(recs_z[0]["location"], pos[node_top])

    recs_x = feeder_recommendations(regions, pos, tets, surface_nodes, gravity_vec=(-1, 0, 0),
                                    t_s=t_s)
    np.testing.assert_allclose(recs_x[0]["location"], pos[node_xface])

    recs_y = feeder_recommendations(regions, pos, tets, surface_nodes, gravity_vec=(0, -1, 0),
                                    t_s=t_s)
    np.testing.assert_allclose(recs_y[0]["location"], pos[node_yface])


def test_feeder_recommendations_empty_surface_nodes_raises_value_error():
    pos, tets, t_s, surface_nodes, block_nodes = _interior_cube_block_mesh()
    regions = hot_regions(pos, tets, t_s)
    with pytest.raises(ValueError):
        feeder_recommendations(regions, pos, tets, np.array([], dtype=np.int64),
                               gravity_vec=(0, 0, -1), t_s=t_s)


def test_feeder_recommendations_degenerate_region_warns_instead_of_silent_zero(caplog):
    import logging as _logging
    pos, tets, t_s, surface_nodes, block_nodes = _interior_cube_block_mesh()
    # a hand-built region touching no tet at all (an isolated single node
    # from the fixture's background, never part of any hot cluster) has
    # volume_mm3 == 0 and boundary area == 0 -> degenerate modulus.
    degenerate_region = {
        "nodes": np.array([0]),
        "centroid": pos[0],
        "volume_mm3": 0.0,
        "t_max": 0.0,
        "region_id": 0,
        "hot_threshold": t_s.max() + 1.0,  # nothing in the mesh qualifies as "hot"
    }
    with caplog.at_level(_logging.WARNING, logger="src.shrinkage"):
        recs = feeder_recommendations([degenerate_region], pos, tets, surface_nodes,
                                      gravity_vec=(0, 0, -1), t_s=t_s)
    assert recs[0]["min_modulus_mm"] == 0.0
    assert any("degenerate" in rec.message for rec in caplog.records)


def test_feeder_recommendations_region_touching_mesh_skin_exercises_boundary_term(caplog):
    """a dedicated, in-suite exercise of the
    domain-boundary term (a region whose own boundary coincides with the
    mesh's outer surface, so `_region_surface_area`'s "no other side to
    integrate across" correction path actually fires) -- the gravity-
    rotation tests above already use a skin-touching region but only assert
    `location`, never `min_modulus_mm`. Reuses the boss+band fixture (task 2
    round 1): the band directly touches 3 domain-boundary faces."""
    pos, tets, t_s, top_frac, surface_nodes, boss_nodes, band_nodes = _boss_and_band_mesh()
    regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
    band_region = next(r for r in regions if set(r["nodes"].tolist()) == band_nodes)

    with caplog.at_level("WARNING", logger="src.shrinkage"):
        recs = feeder_recommendations([band_region], pos, tets, surface_nodes,
                                      gravity_vec=(0, 0, -1), t_s=t_s)
    assert not any("degenerate" in rec.message for rec in caplog.records)
    assert recs[0]["min_modulus_mm"] > 0.0
    assert np.isfinite(recs[0]["min_modulus_mm"])

    # independent hand-check: the reference reimplementation must agree.
    expected_area = _reference_region_area_marching_tets(
        pos, tets, band_region["nodes"], t_s, band_region["hot_threshold"])
    expected_modulus = band_region["volume_mm3"] / expected_area
    assert recs[0]["min_modulus_mm"] == pytest.approx(1.2 * expected_modulus, rel=1e-9)


def _smooth_blob_mesh(spacing, extent=10.0, top_frac=0.05, center_offset=(0.13, 0.29, 0.07)):
    """A cubic domain of side `extent` mm with a smooth (non-binary-block)
    field `t_s = -|x - center|^2` -- hottest at the center, smoothly falling
    off outward, so the top-`top_frac` hot selection is a roughly spherical
    blob whose boundary genuinely straddles many partial tets at every mesh
    density (unlike the axis-aligned block fixtures elsewhere in this file,
    which are deliberately clean/hand-checkable but not representative of a
    "smooth field" mesh-convergence scenario). `center_offset` is a
    deliberately non-grid-aligned (sub-cell) shift."""
    n = int(round(extent / spacing)) + 1
    pos, tets = _grid_mesh(n, n, n, spacing=spacing)
    center = np.array([extent / 2.0] * 3) + np.array(center_offset)
    t_s = -((pos - center) ** 2).sum(axis=1)
    surface_nodes = _surface_nodes(n, n, n)
    return pos, tets, t_s, top_frac, surface_nodes


def test_feeder_recommendations_modulus_refinement_drift_improves_nonaxis(caplog):
    """moved from an exact-grid-centered blob to an off-grid-centered spherical blob -- a genuinely NON-axis-
    aligned fixture, since the round-2 co-area formula's axis-aligned cuts
    were "the one exact case" and could mask residual anisotropy even on a
    nominally spherical shape if its center happened to sit on a favorable
    grid position at both tested densities. Under the OLD (round-1-shipped)
    all-4-vertices-hot area rule, M = V/A drifted ~-29% between a coarse and
    2x-refined mesh of a smooth field (worse than the pre-fix-round-1
    baseline of -17%). The marching-tetrahedra area (this round) must keep
    drift well under the 15% bound."""
    import logging as _logging

    def modulus_at(spacing):
        pos, tets, t_s, top_frac, surface_nodes = _smooth_blob_mesh(spacing)
        regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
        assert len(regions) == 1
        with caplog.at_level(_logging.WARNING, logger="src.shrinkage"):
            recs = feeder_recommendations(regions, pos, tets, surface_nodes,
                                          gravity_vec=(0, 0, -1), t_s=t_s)
        return recs[0]["min_modulus_mm"] / 1.2

    m_coarse = modulus_at(1.0)
    m_fine = modulus_at(0.5)
    drift = abs(m_fine - m_coarse) / m_coarse
    assert drift < 0.15, f"M drift {drift:.1%} exceeds the 15% bound (coarse={m_coarse}, fine={m_fine})"


# ---------------------------------------------------------------------------
# (g) Acceptance suite: marching-tets area vs. the ANALYTIC
# true value on shapes the old (binarized co-area) formula got badly wrong --
# a sphere (curved, converging with refinement, not plateauing), a diagonal
# plane cut through a box (exact via a convex-hull oracle), and an
# orientation sweep (axis/face-diagonal/space-diagonal/generic normals)
# showing the error stays small and roughly ORIENTATION-INDEPENDENT.
# ---------------------------------------------------------------------------

def _sphere_region(spacing, extent=12.0, top_frac=0.03, center_offset=(0.13, 0.29, 0.07)):
    """A spherical hot blob (off-grid center, see `_smooth_blob_mesh`) with
    its ANALYTIC area/radius derived directly from the threshold `hot_
    regions` actually used (R = sqrt(-threshold) exactly, since the
    continuous field t_s = -|pos-center|^2 has a perfectly spherical level
    set at every threshold, independent of mesh discretization)."""
    n = int(round(extent / spacing)) + 1
    pos, tets = _grid_mesh(n, n, n, spacing=spacing)
    center = np.array([extent / 2.0] * 3) + np.array(center_offset)
    t_s = -((pos - center) ** 2).sum(axis=1)
    threshold = np.percentile(t_s, 100 * (1 - top_frac))
    nodes = np.flatnonzero(t_s >= threshold)
    radius = np.sqrt(-threshold)
    true_area = 4.0 * np.pi * radius ** 2
    return pos, tets, nodes, t_s, threshold, true_area


def test_region_area_sphere_converges_not_plateaus():
    """sphere-field area must converge
    toward the analytic 4*pi*R^2 as the mesh refines (error DECREASING),
    not plateau at the ~24-29% measured for the binarized
    co-area formula."""
    errors = []
    for spacing in (1.0, 0.5, 0.25):
        pos, tets, nodes, t_s, threshold, true_area = _sphere_region(spacing)
        area = _reference_region_area_marching_tets(pos, tets, nodes, t_s, threshold)
        errors.append(abs(area - true_area) / true_area)

    assert errors[0] < 0.10, f"coarsest-mesh sphere error too high: {errors[0]:.1%}"
    assert errors[-1] < errors[0], f"error did not decrease with refinement: {errors}"
    assert errors[-1] < 0.02, f"finest-mesh sphere error too high: {errors[-1]:.1%}"


def _halfspace_box_total_area(box_extent, normal, offset):
    """Analytic total boundary area of `box ∩ {p : normal.p <= offset}` for
    an axis-aligned box `[0,box_extent]^3` -- exact via `scipy.spatial.
    ConvexHull` on the (convex) polytope's vertices: the box corners already
    inside the half-space, plus the half-space plane's intersection points
    with each of the box's 12 edges. Serves as the analytic reference for
    the diagonal-cut and orientation-sweep tests below (hand-verified
    against known closed forms: an axis-aligned cut of a 10mm cube at x=5
    gives exactly 400mm^2 = surface area of a 5x10x10 box; a cut through the
    center along the space diagonal gives 300 [half the original 600mm^2
    surface, split evenly by symmetry] + 3*sqrt(3)/4*10^2 [=129.90, the
    well-known hexagonal cross-section area of a cube bisected
    perpendicular to its space diagonal] = 429.90)."""
    from scipy.spatial import ConvexHull

    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    lo, hi = 0.0, float(box_extent)
    corners = np.array([[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)])
    edges = []
    for xi in range(2):
        for yi in range(2):
            for zi in range(2):
                idx0 = 4 * xi + 2 * yi + zi
                if xi == 0:
                    edges.append((idx0, 4 * 1 + 2 * yi + zi))
                if yi == 0:
                    edges.append((idx0, 4 * xi + 2 * 1 + zi))
                if zi == 0:
                    edges.append((idx0, 4 * xi + 2 * yi + 1))

    pts = [c for c in corners if normal @ c - offset <= 1e-9]
    for i, j in edges:
        vi, vj = corners[i], corners[j]
        si, sj = normal @ vi - offset, normal @ vj - offset
        if si * sj < 0:
            t = si / (si - sj)
            pts.append(vi + t * (vj - vi))
    return ConvexHull(np.array(pts)).area


def _diagonal_cut_region(spacing, normal, extent=10.0, top_frac=0.3):
    pos, tets = _grid_mesh(int(round(extent / spacing)) + 1,
                           int(round(extent / spacing)) + 1,
                           int(round(extent / spacing)) + 1, spacing=spacing)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    t_s = -(pos @ normal)
    threshold = np.percentile(t_s, 100 * (1 - top_frac))
    nodes = np.flatnonzero(t_s >= threshold)
    true_area = _halfspace_box_total_area(extent, normal, offset=-threshold)
    return pos, tets, nodes, t_s, threshold, true_area


def test_region_area_diagonal_plane_cut_error_far_below_old_bias():
    """a half-space region cut by a
    plane along the space diagonal (1,1,1) -- normal not aligned with any
    mesh axis. Error must be far below the reported +36% for the
    prior (binarized co-area) formula."""
    for spacing in (1.0, 0.5):
        pos, tets, nodes, t_s, threshold, true_area = _diagonal_cut_region(spacing, (1, 1, 1))
        area = _reference_region_area_marching_tets(pos, tets, nodes, t_s, threshold)
        err = abs(area - true_area) / true_area
        assert err < 0.10, f"spacing={spacing}: diagonal-cut error {err:.1%} too high"




# ---------------------------------------------------------------------------
# (f) hot_regions volume -- fraction-weighted rule:
# each tet contributes tet_volume * n_hot_vertices/4, replacing an earlier
# all-4-vertices-hot rule that discarded (undercounted) every boundary tet.
# ---------------------------------------------------------------------------

def test_hot_regions_volume_fraction_weighted_partial_tets_hand_checked():
    """An isolated single unit-cube mesh (2x2x2 nodes, spacing 1mm, no
    neighboring cubes -- nothing outside this mesh to contaminate the
    count): corner 7 (the (1,1,1) corner) is background/cold, the other 7
    corners are hot. Every one of the cube's 6 Kuhn tets contains corner 7
    (hand-checked from the tet list), so every tet has exactly 3 of its 4
    vertices hot; each tet has volume 1/6 mm^3 (regular unit cube / 6). Total
    fraction-weighted volume = 6 * (1/6) * (3/4) = 0.75 mm^3 exactly. Under
    the old all-4-vertices rule this region would score 0.0 mm^3 (no tet has
    all 4 vertices hot) -- the undercount this fix targets."""
    pos, tets = _grid_mesh(2, 2, 2)
    t_s = np.full(8, 100.0)
    t_s[7] = 0.0
    top_frac = 7.0 / 8.0
    regions = hot_regions(pos, tets, t_s, top_frac=top_frac)
    assert len(regions) == 1
    assert set(regions[0]["nodes"].tolist()) == {0, 1, 2, 3, 4, 5, 6}
    assert regions[0]["volume_mm3"] == pytest.approx(0.75)


def test_hot_regions_volume_full_mesh_matches_analytic_total():
    """Sanity/regression check: when EVERY node is hot (top_frac=1.0), no
    tet straddles the region boundary (n_hot_vertices == 4 everywhere), so
    the fraction-weighted rule reduces to the plain tet-volume sum, which
    for a unit-spacing (nx-1)x(ny-1)x(nz-1) lattice equals that product
    exactly."""
    nx, ny, nz = 4, 3, 3
    pos, tets = _grid_mesh(nx, ny, nz)
    t_s = np.ones(pos.shape[0])  # uniform -> every node is "hot" at top_frac=1.0
    regions = hot_regions(pos, tets, t_s, top_frac=1.0)
    assert len(regions) == 1
    assert regions[0]["volume_mm3"] == pytest.approx((nx - 1) * (ny - 1) * (nz - 1))


# ---------------------------------------------------------------------------
# (e) determinism -- two runs on the same inputs produce identical results.
# ---------------------------------------------------------------------------

def test_determinism_hot_regions_and_downstream():
    pos, tets, t_s, surface_nodes, block_nodes = _interior_cube_block_mesh()
    no_feed = np.array([], dtype=np.int64)

    regions1 = hot_regions(pos, tets, t_s)
    regions2 = hot_regions(pos, tets, t_s)
    assert len(regions1) == len(regions2) == 1
    np.testing.assert_array_equal(regions1[0]["nodes"], regions2[0]["nodes"])
    np.testing.assert_allclose(regions1[0]["centroid"], regions2[0]["centroid"])
    assert regions1[0]["volume_mm3"] == regions2[0]["volume_mm3"]
    assert regions1[0]["t_max"] == regions2[0]["t_max"]
    assert regions1[0]["region_id"] == regions2[0]["region_id"]

    macro1 = macro_shrinkage(pos, tets, t_s, no_feed, shrinkage_pct=2.5)
    macro2 = macro_shrinkage(pos, tets, t_s, no_feed, shrinkage_pct=2.5)
    assert macro1[0]["severity"] == macro2[0]["severity"]

    feeders1 = feeder_recommendations(regions1, pos, tets, surface_nodes, gravity_vec=(0, 0, -1),
                                      t_s=t_s)
    feeders2 = feeder_recommendations(regions2, pos, tets, surface_nodes, gravity_vec=(0, 0, -1),
                                      t_s=t_s)
    assert feeders1[0]["min_modulus_mm"] == feeders2[0]["min_modulus_mm"]
    np.testing.assert_array_equal(feeders1[0]["location"], feeders2[0]["location"])


# ---------------------------------------------------------------------------
# screen_casting: one call combining the Niyama proxy, macro-shrinkage, top
# sinks and feeder recommendations. `feed_nodes` (adiabatic-face nodes) is
# separate from `surface_nodes` (the whole free surface, used for sink and
# feeder placement); None means no feed source, so every region is unfed.
# ---------------------------------------------------------------------------



def test_screen_casting_proxy_niyama_when_fields_omitted():
    nx = ny = nz = 4
    pos, tets = _grid_mesh(nx, ny, nz)
    t_s = pos[:, 0].copy() + 1.0
    surface_nodes = _surface_nodes(nx, ny, nz)
    TL, TS = 1400.0, 1100.0

    result = screen_casting(pos, tets, surface_nodes, t_s, TL, TS, "grey_iron")

    assert result["niyama_source"] == "proxy"
    expected_ny = niyama_proxy(pos, tets, t_s, TL, TS)
    np.testing.assert_allclose(result["niyama"], expected_ny)
    expected_risk = porosity_risk(expected_ny)
    np.testing.assert_array_equal(result["porosity_risk"], expected_risk)
    assert result["material"] == {"alloy_family": "grey_iron", "shrinkage_pct": 0.0}




def test_screen_casting_unknown_alloy_family_raises_loud_error_naming_known_families():
    pos, tets, t_s, surface_nodes, block_nodes = _interior_cube_block_mesh()
    with pytest.raises(KeyError) as exc_info:
        screen_casting(pos, tets, surface_nodes, t_s, 1400.0, 1100.0, "unobtainium")
    msg = str(exc_info.value)
    assert "unobtainium" in msg
    for family in SHRINKAGE_PCT:
        assert family in msg


def test_screen_casting_feed_nodes_default_is_empty_bare_casting_convention():
    pos, tets, t_s, surface_nodes, boss_nodes, band_nodes = _unfed_boss_and_fed_band_10pct_mesh()
    TL, TS = 1400.0, 1100.0

    default_call = screen_casting(pos, tets, surface_nodes, t_s, TL, TS, "steel")
    explicit_empty = screen_casting(pos, tets, surface_nodes, t_s, TL, TS, "steel",
                                    feed_nodes=np.array([], dtype=np.int64))
    assert len(default_call["macro_shrinkage"]) == len(explicit_empty["macro_shrinkage"]) == 2

    fed_call = screen_casting(pos, tets, surface_nodes, t_s, TL, TS, "steel",
                              feed_nodes=surface_nodes)
    assert len(fed_call["macro_shrinkage"]) == 1  # band fed away via feed_nodes=surface_nodes


def test_region_area_orientation_sweep_spread_far_below_old_bias():
    """The SAME half-space-cut
    construction at several normals (axis-aligned, face-diagonal, space-
    diagonal, and a generic non-symmetric direction) -- the spread of
    relative errors across orientations must be far below the ~29% orientation
    sensitivity of a binarized co-area estimate."""
    orientations = {
        "axis-x": (1, 0, 0),
        "face-diag": (1, 1, 0),
        "space-diag": (1, 1, 1),
        "generic": (2, 1, 0.3),
    }
    errors = {}
    for name, normal in orientations.items():
        pos, tets, nodes, t_s, threshold, true_area = _diagonal_cut_region(0.5, normal)
        area = _reference_region_area_marching_tets(pos, tets, nodes, t_s, threshold)
        errors[name] = abs(area - true_area) / true_area

    spread = max(errors.values()) - min(errors.values())
    assert spread < 0.10, f"orientation-sweep spread {spread:.1%} too high: {errors}"
    assert all(e < 0.10 for e in errors.values()), f"an orientation exceeded 10% error: {errors}"
