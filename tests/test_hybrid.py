import logging

import numpy as np
import pytest
from src.hybrid import hybrid_field_affine, hybrid_field_max_anchor, hybrid_field_soft








def test_hybrid_affine_spread_match():
    # gnn std is exactly 10 (two points, mean 10, deviations -+10)
    gnn = np.array([0.0, 20.0])
    total = 1000.0  # large enough that nothing clips
    pred, info = hybrid_field_affine(gnn, total, std_seconds=20.0, q=0.995)
    assert info["a"] == 2.0
    np.testing.assert_allclose(np.std(pred), 20.0, atol=1e-9)
    assert info["clip_frac"] == 0.0


def test_hybrid_affine_total_anchor():
    gnn = np.array([0.0, 20.0])
    total = 1000.0  # large enough that nothing clips
    pred, info = hybrid_field_affine(gnn, total, std_seconds=20.0, q=0.995)
    assert np.quantile(pred, 0.995) == pytest.approx(total, abs=1e-9)
    assert info["clip_frac"] == 0.0


def test_hybrid_affine_preserves_ranking():
    gnn = np.array([3.0, 1.0, 9.0, 5.0])
    total = 1000.0
    pred, info = hybrid_field_affine(gnn, total, std_seconds=5.0)
    assert info["a"] > 0.0
    assert list(np.argsort(pred)) == list(np.argsort(gnn))


def test_hybrid_affine_clamp_high():
    # gnn std is 10; std_seconds=100 -> a_raw=10.0, clamped down to the a_clamp
    # ceiling 5.0. No clipping occurs at a=5.0 here (total=1000 is large), so
    # this exercises the closed-form path's own clamp handling.
    gnn = np.array([0.0, 20.0])
    total = 1000.0
    pred, info = hybrid_field_affine(gnn, total, std_seconds=100.0)
    assert info["a"] == 5.0
    assert info["a_clamped"] is True


def test_hybrid_affine_degenerate_fallback(caplog):
    gnn = np.array([5.0, 5.0, 5.0])
    total = 100.0
    with caplog.at_level(logging.WARNING):
        pred, info = hybrid_field_affine(gnn, total, std_seconds=20.0, q=0.995)
    expected = hybrid_field_max_anchor(gnn, total, q=0.995)
    np.testing.assert_allclose(pred, expected)
    assert info["degenerate_fallback"] is True
    assert any("degenerate" in rec.message.lower() for rec in caplog.records)


def test_hybrid_affine_clip_negative():
    # gnn std is 50; the naive (pre-amendment) closed form a=std_seconds/s=1.0
    # already clips the low node (raw would be [-89.5, 10.5]), so the
    # clip-aware solver kicks in. With only 2 nodes the post-clip std is capped
    # well below 50 everywhere in the clamp range (max achievable is at
    # a=5.0), so this lands on the a_clamp ceiling.
    gnn = np.array([0.0, 100.0])
    total = 10.0
    pred, info = hybrid_field_affine(gnn, total, std_seconds=50.0)
    np.testing.assert_allclose(pred, [0.0, 12.5])
    assert info["a"] == 5.0
    assert info["a_clamped"] is True
    assert info["clip_frac"] == pytest.approx(0.5)


def test_hybrid_affine_heavy_clip_achievable():
    # Real fields sometimes need heavy scaling to hit the spread head's
    # target; a closed form that ignores clipping
    # entirely, so its implied `a` can leave the delivered std far short of
    # the target. Construct such a case (right-skewed field; the naive
    # closed-form `a` clips ~30% of nodes) and confirm the clip-aware solve
    # still lands on-target.
    g = np.random.default_rng(1).exponential(5.0, size=200)
    s = np.std(g)
    total = 8.0
    target = 0.3 * s

    # sanity: the pre-amendment closed form (a = target/s, unclamped) clips
    # ~30% of nodes here -- this is the regime the amendment targets.
    a_naive = target / s
    centered_naive = a_naive * (g - g.mean())
    m_naive = total - np.quantile(centered_naive, 0.995)
    naive_clip_frac = float((centered_naive + m_naive < 0.0).mean())
    assert 0.25 <= naive_clip_frac <= 0.35

    pred, info = hybrid_field_affine(g, total, target)
    assert info["a_clamped"] is False
    assert np.std(pred) == pytest.approx(target, rel=1e-3)


def test_hybrid_affine_unachievable_due_to_clipping():
    # std_seconds=8.0 exceeds this field's interior PEAK (f rises past
    # a_clipstart to a peak around a~1.78, std~3.82, then falls -- it never
    # reaches 8.0 anywhere in the clamp range). The unreachable case must pick the scanned point CLOSEST to
    # the target (here, at/near the peak) rather than blindly clamping to
    # a_clamp[1]=5.0 -- clamping to 5.0 would give a worse fit (std~3.45,
    # clip_frac 0.7) than the peak (std~3.82, clip_frac 0.4).
    gnn = np.arange(10.0)
    total = 10.5
    pred, info = hybrid_field_affine(gnn, total, std_seconds=8.0)
    assert info["a"] != 5.0  # must NOT blindly clamp to the ceiling
    assert info["a_clamped"] is True
    achieved = np.std(pred)
    assert achieved == pytest.approx(3.816200508029165, rel=1e-6)
    assert achieved < 8.0
    # the peak (this test's outcome) beats the naive hi=5.0 clamp on fit:
    at_hi = np.clip(5.0 * (gnn - gnn.mean() - np.quantile(gnn - gnn.mean(), 0.995)) + total,
                    0.0, None)
    assert abs(achieved - 8.0) < abs(np.std(at_hi) - 8.0)




def test_hybrid_affine_no_clip_matches_pre_amendment_closed_form():
    # Invariant: when nothing clips at the closed-form solution, the amended
    # (clip-aware) function must reproduce the pre-amendment closed form
    # exactly (it takes it as a fast path; bisection is never invoked).
    def pre_amendment_closed_form(gnn_seconds, total_seconds, std_seconds,
                                   q=0.995, a_clamp=(0.3, 5.0)):
        g = np.asarray(gnn_seconds, dtype=np.float64)
        s = float(np.std(g))
        a_raw = float(std_seconds) / s
        a = float(np.clip(a_raw, a_clamp[0], a_clamp[1]))
        centered = a * (g - g.mean())
        m = float(total_seconds) - float(np.quantile(centered, q))
        out = centered + m
        return np.clip(out, 0.0, None), a

    gnn = np.array([3.0, 1.0, 9.0, 5.0])
    total = 1000.0
    std_seconds = 5.0
    pred_new, info_new = hybrid_field_affine(gnn, total, std_seconds)
    pred_old, a_old = pre_amendment_closed_form(gnn, total, std_seconds)
    assert info_new["a_clamped"] is False
    assert info_new["a"] == pytest.approx(a_old, rel=1e-6)
    np.testing.assert_allclose(pred_new, pred_old, rtol=1e-6)












# --------------------------------------------------------------------------- #
# hybrid_field_soft: weighted two-target soft recomposition
# --------------------------------------------------------------------------- #
def _squeezed_case():
    # A field with a small (3%) group of low-value outlier nodes and a tight
    # high-value cluster: the affine solve's hard quantile-anchor pins the
    # top near `total`, so scaling enough to hit a large `std_seconds` clips
    # the low outliers hard -- `a_clamped` fires (the two-target problem is
    # unreachable via a pure affine map). Verified interactively: at these
    # exact parameters the affine solve lands short of the std target
    # (std ~169.8 vs target ~330.1, clip_frac 0.03, a_clamped True) while the
    # soft solve's default-weight optimum (~203.2) is strictly closer, by
    # trading a ~20% total-anchor deviation for less clipping.
    rng = np.random.default_rng(7)
    n = 200
    n_low = int(0.03 * n)
    n_high = n - n_low
    high_vals = 1000.0 + rng.normal(0.0, 2.0, n_high)
    low_vals = rng.uniform(0.0, 50.0, n_low)
    g = np.concatenate([high_vals, low_vals])
    total = 1000.0
    target = 2.0 * float(np.std(g))
    return g, total, target


def test_hybrid_soft_exact_degeneration_matches_affine():
    # No-clip case (from test_hybrid_affine_spread_match/total_anchor): the
    # affine solve already hits both targets essentially exactly, so the soft
    # solve must return that seed unchanged.
    gnn = np.array([0.0, 20.0])
    total = 1000.0
    pred_affine, info_affine = hybrid_field_affine(gnn, total, std_seconds=20.0, q=0.995)
    pred_soft, info_soft = hybrid_field_soft(gnn, total, std_seconds=20.0, q=0.995)
    np.testing.assert_allclose(pred_soft, pred_affine, rtol=1e-9)
    assert abs(info_soft["total_deviation_frac"]) < 1e-6
    assert info_soft["degenerate_fallback"] is False


def test_hybrid_soft_trade_off_engages_on_squeezed_case():
    g, total, target = _squeezed_case()
    pred_affine, info_affine = hybrid_field_affine(g, total, target)
    assert info_affine["a_clamped"] is True  # affine truly cannot reach std_t

    pred_soft, info_soft = hybrid_field_soft(g, total, target)
    d_affine = abs(np.std(pred_affine) - target)
    d_soft = abs(np.std(pred_soft) - target)
    assert d_soft < d_affine  # soft's std is strictly closer to the target
    assert info_soft["total_deviation_frac"] > 0  # the top rose above total


def test_hybrid_soft_weight_monotonicity_on_squeezed_case():
    g, total, target = _squeezed_case()
    pred_hi, info_hi = hybrid_field_soft(g, total, target, w_total=100.0, w_std=1.0)
    pred_lo, info_lo = hybrid_field_soft(g, total, target, w_total=0.01, w_std=1.0)
    assert abs(info_hi["total_deviation_frac"]) <= abs(info_lo["total_deviation_frac"])
    assert np.std(pred_lo) >= np.std(pred_hi)


def test_hybrid_soft_determinism():
    g, total, target = _squeezed_case()
    pred1, info1 = hybrid_field_soft(g, total, target)
    pred2, info2 = hybrid_field_soft(g, total, target)
    np.testing.assert_array_equal(pred1, pred2)
    assert info1 == info2


def test_hybrid_soft_degenerate_flat_field_falls_back(caplog):
    gnn = np.array([5.0, 5.0, 5.0])
    total = 100.0
    with caplog.at_level(logging.WARNING):
        pred, info = hybrid_field_soft(gnn, total, std_seconds=20.0, q=0.995)
    expected = hybrid_field_max_anchor(gnn, total, q=0.995)
    np.testing.assert_allclose(pred, expected)
    assert info["degenerate_fallback"] is True
    assert any("degenerate" in rec.message.lower() for rec in caplog.records)






# --------------------------------------------------------------------------- #
# Edge cases: the a_clamp[0] <= 0 guard and degeneration to the affine solve
# at a realistic scale.
# --------------------------------------------------------------------------- #




def test_hybrid_soft_realistic_scale_exact_degeneration_matches_affine():
    # The exact-degeneration test above uses a 2-node toy field;
    # this adds a realistic-scale (n=5000), right-skewed synthetic field with
    # an EASILY achievable std target (no clipping needed at all), so the
    # affine seed hits both targets and the soft solve must reduce to it
    # exactly (via the j_seed < 1e-12 early return, not by chance).
    rng = np.random.default_rng(11)
    g = rng.lognormal(mean=5.0, sigma=0.4, size=5000)
    total = 5000.0  # large relative to g's own scale -> no clipping at a~1
    target = 0.8 * float(np.std(g))  # well within reach, a_raw < 1 < a_clamp[1]

    pred_affine, info_affine = hybrid_field_affine(g, total, target)
    assert info_affine["a_clamped"] is False
    pred_soft, info_soft = hybrid_field_soft(g, total, target)

    np.testing.assert_allclose(pred_soft, pred_affine, rtol=1e-9)
    assert info_soft["target_unreachable"] is False
    assert abs(info_soft["total_deviation_frac"]) < 1e-6


def test_hybrid_soft_a_clamp_lo_nonpositive_raises():
    gnn = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="a_clamp"):
        hybrid_field_soft(gnn, 10.0, 1.0, a_clamp=(0.0, 5.0))
    with pytest.raises(ValueError, match="a_clamp"):
        hybrid_field_soft(gnn, 10.0, 1.0, a_clamp=(-1.0, 5.0))


def test_hybrid_soft_negative_weight_raises():
    # LOW: a negative weight rewards moving AWAY from that
    # target (minimizing J then maximizes the residual's square) -- must
    # raise loudly rather than silently drive a to a nonsensical optimum.
    gnn = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="w_total|w_std"):
        hybrid_field_soft(gnn, 10.0, 1.0, w_total=-1.0)
    with pytest.raises(ValueError, match="w_total|w_std"):
        hybrid_field_soft(gnn, 10.0, 1.0, w_std=-1.0)


def test_hybrid_soft_both_weights_zero_raises():
    # LOW: w_total == w_std == 0 makes J identically zero
    # everywhere -- no signal for the search to optimize against.
    gnn = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="w_total|w_std"):
        hybrid_field_soft(gnn, 10.0, 1.0, w_total=0.0, w_std=0.0)


def test_hybrid_soft_one_weight_zero_is_allowed():
    # Sanity: only BOTH-zero is forbidden -- a single zero weight (pure
    # single-target optimization) is a legitimate, deliberate use case.
    gnn = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred, info = hybrid_field_soft(gnn, 10.0, 1.0, w_total=0.0, w_std=1.0)
    assert np.all(np.isfinite(pred))
    pred, info = hybrid_field_soft(gnn, 10.0, 1.0, w_total=1.0, w_std=0.0)
    assert np.all(np.isfinite(pred))


def test_hybrid_soft_accepted_optimum_majority_clipped_warns(caplog):
    # The affine path already warns when ITS OWN clip_frac > 0.5; the
    # soft path must emit the analogous warning when the ACCEPTED soft
    # optimum (which can differ from the affine seed) itself clips a
    # majority of the field.
    rng = np.random.default_rng(3)
    g = rng.exponential(1.0, size=500) ** 2
    g = g - g.min()
    total = 3.0
    target = 3.0 * float(np.std(g))
    with caplog.at_level(logging.WARNING):
        pred, info = hybrid_field_soft(g, total, target, w_total=0.1, w_std=10.0)
    assert info["clip_frac"] > 0.5
    assert any("hybrid_field_soft" in rec.message and "majority" in rec.message.lower()
              for rec in caplog.records)




def test_hybrid_soft_info_dict_carries_all_keys():
    g, total, target = _squeezed_case()
    pred, info = hybrid_field_soft(g, total, target)
    expected_keys = {"a", "m", "target_unreachable", "degenerate_fallback", "clip_frac",
                     "total_deviation_frac", "j_total_resid", "j_std_resid",
                     "std_target_effective"}
    assert expected_keys <= set(info.keys())
    # clip_frac must be measured on the FINAL (returned) field.
    a, m = info["a"], info["m"]
    c = g - g.mean()
    final_raw = a * c + m
    assert info["clip_frac"] == pytest.approx(float((final_raw < 0.0).mean()))
