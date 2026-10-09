"""max/quantile-anchor hybrid recomposition, in src/hybrid.py.

The exact-max anchor is fragile to a handful of outlier nodes (one
inflated GNN log-space prediction can drag the entire rest of the field
below the zero clip). The anchor moved to a high quantile
(`np.quantile(gnn, q)`, default `q=0.995`); `q=1.0` reproduces the old
exact-max behavior bit-for-bit (`np.quantile(x, 1.0) == x.max()`).
"""
import numpy as np

from src.hybrid import hybrid_field_max_anchor


def test_max_anchor_q1_sets_exact_max_and_preserves_shape():
    """q=1.0 regression pin: reproduces the earlier exact-max behavior."""
    g = np.array([1.0, 4.0, 2.5])
    out = hybrid_field_max_anchor(g, 10.0, q=1.0)
    assert out.max() == 10.0
    assert np.allclose(np.diff(out), np.diff(g))  # additive shift only


def test_max_anchor_default_q_preserves_shape_and_meets_or_exceeds_total():
    """Default q<1: still an additive shift; max(out) >= total (not ==)."""
    g = np.array([1.0, 4.0, 2.5])
    out = hybrid_field_max_anchor(g, 10.0)
    assert out.max() >= 10.0
    assert out.shape == g.shape
    assert np.allclose(np.diff(out), np.diff(g))  # additive shift only


def test_max_anchor_clips_at_zero():
    assert hybrid_field_max_anchor(np.array([0.0, 100.0]), 10.0)[0] == 0.0




def test_max_anchor_q1_reproduces_old_exact_max_on_random_vector():
    """Regression pin: q=1.0 bit-for-bit equals the earlier formula
    `clip(g - g.max() + total, 0, None)` on an arbitrary random vector."""
    rng = np.random.default_rng(42)
    g = rng.normal(loc=50.0, scale=20.0, size=200)
    total = 37.5

    out = hybrid_field_max_anchor(g, total, q=1.0)
    expected = np.clip(g - g.max() + total, 0.0, None)

    assert np.array_equal(out, expected)


def test_max_anchor_default_q_is_robust_to_a_single_outlier_node():
    """The failure mode the promotion gate exposed: one wildly-inflated
    GNN prediction anchored to the exact max drags ~all other nodes below
    the zero clip. With the default quantile anchor (q=0.995), >=99% of a
    1000-node field stays > 0 for the same outlier + total."""
    rng = np.random.default_rng(0)
    bulk = rng.uniform(0.0, 100.0, size=1000)
    g = np.concatenate([bulk, [1e6]])  # one extreme outlier node
    total = 500.0

    out_exact = hybrid_field_max_anchor(g, total, q=1.0)
    out_default = hybrid_field_max_anchor(g, total)

    # Old exact-max behavior: the outlier alone sets the anchor, dragging
    # essentially the whole bulk to the zero clip.
    assert (out_exact[:-1] > 0).mean() < 0.01

    # New default-q behavior: the bulk survives the clip.
    assert (out_default[:-1] > 0).mean() >= 0.99
    assert out_default.shape == g.shape


def test_max_anchor_argsort_unchanged_between_q1_and_default_q():
    """Hot-spot ranking must be unaffected by the anchor choice: both are
    per-graph additive shifts of the same field, so (absent clipping
    collisions) node order is identical regardless of q."""
    rng = np.random.default_rng(7)
    g = rng.uniform(10.0, 200.0, size=500)
    # total large enough that neither anchor choice clips any node at zero
    # (min(g) - anchor + total > 0 for anchor <= max(g)): isolates ordering
    # from the (expected, separately-tested) clipping behavior.
    total = 300.0

    out_q1 = hybrid_field_max_anchor(g, total, q=1.0)
    out_default = hybrid_field_max_anchor(g, total)

    assert np.array_equal(np.argsort(out_q1), np.argsort(out_default))
