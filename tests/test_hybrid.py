import logging
import os

import numpy as np
import pytest
from src.hybrid import (hybrid_field, hybrid_field_affine, hybrid_field_max_anchor,
                        hybrid_field_soft)


def test_hybrid_takes_shape_from_gnn_level_from_baseline():
    # GNN shape [-1,0,+1] around mean 5; baseline mean 20 (shape irrelevant)
    gnn = np.array([4.0, 5.0, 6.0])          # deviations -1,0,+1
    base = np.array([18.0, 22.0, 20.0])      # mean 20
    out = hybrid_field(gnn, base)
    # deviations preserved from GNN, level set to baseline mean (20)
    np.testing.assert_allclose(out, [19.0, 20.0, 21.0])


def test_hybrid_clips_negative():
    gnn = np.array([0.0, 10.0])              # mean 5, devs -5,+5
    base = np.array([1.0, 1.0])              # mean 1 -> level 1: -4, +6 -> clip -4 to 0
    out = hybrid_field(gnn, base)
    np.testing.assert_allclose(out, [0.0, 6.0])


def test_hybrid_preserves_within_graph_ranking():
    # hot-spot ranking (argsort) must come from the GNN shape, unchanged by re-leveling
    gnn = np.array([3.0, 1.0, 9.0, 5.0])
    base = np.array([100.0, 100.0, 100.0, 100.0])
    out = hybrid_field(gnn, base)
    assert list(np.argsort(out)) == list(np.argsort(gnn))


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
    # ceiling (widened 3.0 -> 5.0 by the 2026-07-31 spec amendment). No clipping
    # occurs at a=5.0 here (total=1000 is large), so this exercises the fast
    # (pre-amendment closed-form) path's own clamp handling.
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
    # Real fields under FT3 gate 4 sometimes need heavy scaling to hit the
    # GBM's std target; the pre-amendment closed form ignores clipping
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
    # reaches 8.0 anywhere in the clamp range). Per the 2026-07-31 review
    # correction, the unreachable case must pick the scanned point CLOSEST to
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


_A_00968 = "data/features_v4/graphs/A_00968.npz"


@pytest.mark.skipif(not os.path.exists(_A_00968),
                    reason="requires the FT2a v4 feature cache "
                           "(data/features_v4/graphs/A_00968.npz) on disk")
def test_hybrid_affine_real_field_returns_smallest_root_not_ceiling():
    # Regression for the 2026-07-31 review finding: on a REAL per-node `y`
    # field, f(a) has an interior peak (~a=2.67, std~420.5) then falls
    # (f(5.0)~390.8). A naive `f(hi) < std_target => clamp` pre-check would
    # wrongly land on a=5.0 (clip_frac~0.48) even though a much-less-clipped
    # true root exists near the peak's rising edge (a~2.6, clip_frac~0.07).
    # The clip-aware solve must find that smaller root, not the ceiling.
    data = np.load(_A_00968)
    y = data["y"].astype(np.float64)
    total = 1000.0
    target = 419.0  # inside the narrow (f(a_clipstart), peak_f) crossing window

    pred, info = hybrid_field_affine(y, total, target)

    assert info["a_clamped"] is False
    assert info["a"] != pytest.approx(5.0)
    assert info["clip_frac"] < 0.3
    assert np.std(pred) == pytest.approx(target, rel=1e-3)


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
# AMENDMENT 2: lambda-shrinkage of the affine spread target (2026-07-31)
# --------------------------------------------------------------------------- #
def test_hybrid_affine_lambda_default_bit_identical_on_clip_heavy_case():
    # lambda_shrink=1.0 (the default) must be byte-for-bit identical to
    # calling hybrid_field_affine with no lambda_shrink argument at all --
    # on a CLIP-HEAVY case (the heavy-clip-achievable field from round 1),
    # to prove the guard skips the new code path even when the solve
    # actually clips (not just in a trivial no-clip case).
    g = np.random.default_rng(1).exponential(5.0, size=200)
    total = 8.0
    target = 0.3 * np.std(g)

    pred_default, info_default = hybrid_field_affine(g, total, target)
    pred_lambda1, info_lambda1 = hybrid_field_affine(g, total, target, lambda_shrink=1.0)

    assert np.array_equal(pred_default, pred_lambda1)
    assert info_default["a"] == info_lambda1["a"]
    assert info_default["clip_frac"] == info_lambda1["clip_frac"]
    assert info_lambda1["lambda_shrink"] == 1.0
    assert info_lambda1["std_target_effective"] == pytest.approx(target, rel=1e-12)


def test_hybrid_affine_lambda_zero_targets_gnn_own_spread():
    # lambda_shrink=0.0 -> std_eff = std(gnn)**1 * std_target**0 = std(gnn)
    # exactly, regardless of std_target's value. In a no-clip case this
    # makes a==1.0 (no spread rescaling at all), so the result must equal
    # hybrid_field_max_anchor's pure quantile-anchor level shift exactly.
    gnn = np.array([3.0, 1.0, 9.0, 5.0])
    total = 1000.0
    pred, info = hybrid_field_affine(gnn, total, std_seconds=999.0, lambda_shrink=0.0)
    expected = hybrid_field_max_anchor(gnn, total)
    assert info["a"] == pytest.approx(1.0, rel=1e-9)
    assert info["a_clamped"] is False
    assert info["std_target_effective"] == pytest.approx(np.std(gnn), rel=1e-9)
    np.testing.assert_allclose(pred, expected, rtol=1e-9)


def test_hybrid_affine_lambda_half_is_geometric_mean_of_stds():
    # lambda_shrink=0.5 -> std_eff = sqrt(std(gnn) * std_target). In a
    # no-clip case the achieved std equals std_eff exactly (fast path).
    gnn = np.array([0.0, 20.0])  # std(gnn) == 10
    total = 1000.0  # large enough that nothing clips
    std_target = 20.0
    s = np.std(gnn)

    pred, info = hybrid_field_affine(gnn, total, std_target, lambda_shrink=0.5)

    expected_std_eff = np.sqrt(s * std_target)
    assert info["std_target_effective"] == pytest.approx(expected_std_eff, rel=1e-9)
    assert info["a_clamped"] is False
    assert np.std(pred) == pytest.approx(expected_std_eff, rel=1e-9)


def test_hybrid_affine_lambda_info_carries_new_keys():
    gnn = np.array([3.0, 1.0, 9.0, 5.0])
    pred, info = hybrid_field_affine(gnn, 1000.0, 5.0, lambda_shrink=0.7)
    assert "lambda_shrink" in info
    assert "std_target_effective" in info
    assert info["lambda_shrink"] == pytest.approx(0.7)
    assert isinstance(info["std_target_effective"], float)


def test_hybrid_affine_lambda_nondefault_degenerate_std_target_falls_back(caplog):
    # std_seconds <= 0 is only inspected when lambda_shrink != 1.0 (the
    # default path never checks its sign) -- with a non-default lambda it
    # must fall back to the quantile-anchor with the same loud-warning
    # pattern as the degenerate-GNN-field guard.
    gnn = np.array([3.0, 1.0, 9.0, 5.0])
    total = 1000.0
    with caplog.at_level(logging.WARNING):
        pred, info = hybrid_field_affine(gnn, total, std_seconds=0.0, lambda_shrink=0.5)
    expected = hybrid_field_max_anchor(gnn, total)
    np.testing.assert_allclose(pred, expected)
    assert info["degenerate_fallback"] is True
    assert info["lambda_shrink"] == 0.5
    assert any("degenerate" in rec.message.lower() for rec in caplog.records)


# --------------------------------------------------------------------------- #
# hybrid_field_soft: weighted two-target soft recomposition (2026-07-31)
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


def test_hybrid_soft_info_dict_carries_all_keys():
    g, total, target = _squeezed_case()
    pred, info = hybrid_field_soft(g, total, target)
    expected_keys = {"a", "m", "target_unreachable", "degenerate_fallback", "clip_frac",
                     "total_deviation_frac", "j_total_resid", "j_std_resid",
                     "lambda_shrink", "std_target_effective"}
    assert expected_keys <= set(info.keys())
    # clip_frac must be measured on the FINAL (returned) field.
    a, m = info["a"], info["m"]
    c = g - g.mean()
    final_raw = a * c + m
    assert info["clip_frac"] == pytest.approx(float((final_raw < 0.0).mean()))


def test_hybrid_soft_lambda_shrink_matches_v5_convention():
    gnn = np.array([0.0, 20.0])
    total = 1000.0
    std_seconds = 20.0
    s = np.std(gnn)
    pred, info = hybrid_field_soft(gnn, total, std_seconds, lambda_shrink=0.5)
    expected = (s ** 0.5) * (std_seconds ** 0.5)
    assert info["std_target_effective"] == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# Review round 1 (2026-07-31) fix coverage: CRITICAL-1 (m-window centered on
# the seed, not the absolute total*[0.7,1.3]), CRITICAL-2 (perf budget), LOW-6
# (a_clamp[0]<=0 guard), LOW-7 (realistic-scale degeneration case).
# --------------------------------------------------------------------------- #
_REAL_FIELD_STEMS = ["A_00968", "A_01541", "C_00681"]
_REAL_FIELD_PATHS = [f"data/features_v4/graphs/{stem}.npz" for stem in _REAL_FIELD_STEMS]


@pytest.mark.parametrize("path", _REAL_FIELD_PATHS)
def test_hybrid_soft_engages_with_interior_optimum_on_real_fields(path):
    # Reviewer-mandated re-verification for CRITICAL-1: on real per-node `y`
    # fields with std_t just past the affine solve's own reach (1.1-1.5x
    # std(g)), the soft solve at DEFAULT weights must engage (not just return
    # the seed) with an INTERIOR `a` (strictly inside a_clamp, not pinned to
    # either boundary) and a single-digit-percent total deviation -- verified
    # interactively before encoding: e.g. A_00968 at 1.3x gives a~1.28 (vs
    # a_clamp (0.3, 5.0)) and total_deviation_frac ~3.8%.
    if not os.path.exists(path):
        pytest.skip(f"requires the FT2a v4 feature cache ({path}) on disk")
    data = np.load(path)
    y = data["y"].astype(np.float64)
    s = float(np.std(y))
    total = float(np.quantile(y, 0.995))

    engaged = False
    for mult in (1.1, 1.2, 1.3, 1.4, 1.5):
        target = mult * s
        pred_affine, info_affine = hybrid_field_affine(y, total, target)
        if not info_affine["a_clamped"]:
            continue  # affine already reaches this target -- nothing to trade
        engaged = True
        pred_soft, info_soft = hybrid_field_soft(y, total, target)
        assert info_soft["target_unreachable"] is True
        a_clamp = (0.3, 5.0)
        assert a_clamp[0] * 1.001 < info_soft["a"] < a_clamp[1] * 0.999, (
            f"expected an interior optimum, got a={info_soft['a']!r}")
        assert abs(info_soft["total_deviation_frac"]) < 0.10
        # the soft solve must never do WORSE than the affine seed on std:
        d_affine = abs(np.std(pred_affine) - target)
        d_soft = abs(np.std(pred_soft) - target)
        assert d_soft <= d_affine + 1e-9
    assert engaged, f"{path}: no mult in (1.1..1.5) made the affine solve a_clamped=True"


def test_hybrid_soft_realistic_scale_exact_degeneration_matches_affine():
    # LOW-7: the original exact-degeneration test used a 2-node toy field;
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
    # Review round 1, LOW: a negative weight rewards moving AWAY from that
    # target (minimizing J then maximizes the residual's square) -- must
    # raise loudly rather than silently drive a to a nonsensical optimum.
    gnn = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="w_total|w_std"):
        hybrid_field_soft(gnn, 10.0, 1.0, w_total=-1.0)
    with pytest.raises(ValueError, match="w_total|w_std"):
        hybrid_field_soft(gnn, 10.0, 1.0, w_std=-1.0)


def test_hybrid_soft_both_weights_zero_raises():
    # Review round 1, LOW: w_total == w_std == 0 makes J identically zero
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
    # LOW-5: the affine path already warns when ITS OWN clip_frac > 0.5; the
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


@pytest.mark.parametrize("path", _REAL_FIELD_PATHS)
def test_hybrid_soft_perf_budget_10k_nodes(path):
    # CRITICAL-2: < 200ms/10k-node budget (generous 2x margin here to avoid
    # CI flakiness while still catching a gross regression). Measured on
    # dev hardware: ~40-60ms per real field at n=5.4k-8k nodes engaged.
    import time
    if not os.path.exists(path):
        pytest.skip(f"requires the FT2a v4 feature cache ({path}) on disk")
    data = np.load(path)
    y = data["y"].astype(np.float64)
    s = float(np.std(y))
    total = float(np.quantile(y, 0.995))
    target = 1.4 * s  # engages the grid search on all three reference fields

    hybrid_field_soft(y, total, target)  # warm-up (import/JIT-adjacent costs)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        hybrid_field_soft(y, total, target)
        times.append(time.perf_counter() - t0)
    budget_s = 0.200 * (len(y) / 10_000) * 2  # scale budget to n, then 2x slack
    assert min(times) < budget_s, f"{path}: {min(times)*1000:.1f}ms exceeds {budget_s*1000:.1f}ms budget"
