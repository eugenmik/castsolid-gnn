"""Hybrid predictor: GNN field *shape* + baseline absolute *scale*.

The GNN captures the within-graph solidification field shape almost perfectly
(per-graph corr ~0.98, hot-spot ~0.88) but has a per-graph absolute-scale bias.
The physics-feature baseline (HGB) predicts the per-graph magnitude well. The
hybrid keeps the GNN's deviations from its own per-graph mean and re-anchors the
per-graph level to the baseline's per-graph mean:

    pred_i = gnn_i - mean_g(gnn) + mean_g(baseline)

This preserves the GNN's hot-spot/variance advantage while fixing the offset,
and beats both parents on MAE and hot-spot across all splits.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def hybrid_field(gnn_seconds: np.ndarray, baseline_seconds: np.ndarray) -> np.ndarray:
    """Combine one casting's per-node predictions (both in seconds, same node order).

    Shape from `gnn_seconds`, absolute level from `baseline_seconds`. Clipped at 0.
    """
    g = np.asarray(gnn_seconds, dtype=np.float64)
    b = np.asarray(baseline_seconds, dtype=np.float64)
    out = g - g.mean() + b.mean()
    return np.clip(out, 0.0, None)


def hybrid_field_max_anchor(gnn_seconds: np.ndarray, total_seconds: float,
                            q: float = 0.995) -> np.ndarray:
    """v4 hybrid recomposition: GNN field *shape* + a per-graph quantile-anchor scale.

    `pred_i = gnn_i - anchor + total`, clipped at 0, where
    `anchor = np.quantile(gnn_seconds, q)`. Unlike `hybrid_field`'s
    MEAN-anchor (correct for a genuine per-node baseline), the FT2b GBM scale
    head (`src.gbm_ft2b`) predicts a single per-graph SCALAR -- the total
    solidification time (the field's true max), not a per-node mean.
    Broadcasting that scalar and MEAN-anchoring would force
    `mean(pred) == total`, inflating every node by `(true_max - true_mean)`.
    Anchoring to a high quantile of the GNN field instead makes
    `max(pred) ~= total` (the quantity actually predicted) while preserving
    the GNN's field shape (its deviations from that anchor).

    `q=1.0` reproduces the previous exact-max behavior bit-for-bit
    (`np.quantile(x, 1.0) == x.max()`): `max(pred) == total` exactly. The
    DEFAULT `q=0.995` exists because the exact-max anchor is fragile to a
    handful of outlier nodes -- a promotion-gate smoke test found a few
    inflated GNN log-space predictions on a real casting dragging ~90% of
    the field below the zero clip, because the single max node set the
    entire graph's offset. Anchoring to the 99.5th percentile instead makes
    the offset robust to a small number of outliers, at the cost that nodes
    ABOVE the anchor (up to ~0.5% of nodes, by construction) may now exceed
    `total` slightly. This does not affect hot-spot ranking or field shape:
    the anchor choice is still a per-graph ADDITIVE CONSTANT shift, so
    within-graph node ordering (`argsort`) is unchanged, and only the
    reported `float(pred.max())` total-time summary can differ slightly
    from `total`. See `src/evaluate_ft2b.py`'s module docstring ("v4 hybrid
    recomposition: QUANTILE-anchor (q=0.995), not MEAN-anchor") for the full anchor-choice
    rationale -- this is the single source; `evaluate_ft2b` imports this
    function under its former private name, and the same default `q` applies
    everywhere (gate evaluation, checkpoint selection, and production
    inference all share this one function).
    """
    g = np.asarray(gnn_seconds, dtype=np.float64)
    anchor = np.quantile(g, q)
    out = g - anchor + float(total_seconds)
    return np.clip(out, 0.0, None)


def hybrid_field_affine(gnn_seconds: np.ndarray, total_seconds: float,
                        std_seconds: float, q: float = 0.995,
                        a_clamp: tuple = (0.3, 5.0),
                        lambda_shrink: float = 1.0) -> tuple:
    """Affine hybrid recomposition: GNN field *shape* rescaled to match TWO scalar
    targets from the GBM scale head -- a per-graph total AND a per-graph spread --
    instead of `hybrid_field_max_anchor`'s single total-only anchor.

    `hybrid_field_max_anchor` only re-anchors the GNN field's level (an additive
    shift); it inherits the GNN's own spread (std) verbatim. FT3 gate 4
    (`runs/ft3_stage1/FT3_GATE_REPORT.md`) found that spread can be badly wrong
    even when the level and hot-spot ranking are fixed: the GNN's std can
    collapse relative to ground truth on regimes it under-fits, and a pure
    level-shift cannot repair that -- e.g. the report's BUKSA case, where a
    refit GBM anchor overshot the true total by ~+190 s because the GNN field's
    shape carried a whole-field bias that a level-only anchor could not absorb.
    This function additionally SCALES the field's deviations from its own mean
    by a factor `a`, then re-anchors so `np.quantile(pred, q) == total_seconds`
    (same quantile-anchor rationale as `hybrid_field_max_anchor` -- robust to a
    handful of outlier nodes, at the cost that nodes above the anchor may
    slightly exceed `total_seconds`). Scaling deviations by a positive constant
    and then shifting by an additive constant is a strictly increasing affine
    map, so within-graph node ordering (`argsort`, hot-spot ranking) is
    unchanged from the raw GNN field -- exactly as for `hybrid_field_max_anchor`.

    AMENDMENT (owner-approved 2026-07-31, see the "AMENDMENT" block in
    `docs/superpowers/specs/2026-07-31-variance-recomposition-design.md`): the
    field is 0-bounded (solidification time can't be negative) but this affine
    map is symmetric around the field's own mean, so on real (skewed, clip-prone)
    parts a naive `a = std_seconds / std(gnn_seconds)` collapses under its own
    clip. The e_val eval matrix found 21-34% of nodes clipping and 33-35% of
    chills/exo stems clamp-saturated at the old `a<=3.0` ceiling, leaving group
    std_ratio at ~0.83 against the 0.9 floor. `a` is therefore solved so the
    POST-CLIP field matches `std_seconds`, not the naive pre-clip ratio:

        c      = gnn_seconds - mean(gnn_seconds)
        d      = c - quantile(c, q)         # constant vector; independent of a
        raw(a) = a * d + total_seconds      # quantile(raw(a), q) == total_seconds for any a>0
        f(a)   = std(clip(raw(a), 0, None))
        a      = solve f(a) == std_seconds over a in a_clamp

    (`raw(a) = a*d + total_seconds` follows from the quantile-anchor property:
    anchoring `a*c` at `total_seconds` gives `m(a) = total_seconds - a*quantile(c, q)`,
    so `a*c + m(a) = a*(c - quantile(c, q)) + total_seconds = a*d + total_seconds`.)

    CORRECTION (2026-07-31 review): `f` is **NOT monotone** on realistic
    fields. Below `a_clipstart = total_seconds / max(-d.min(), eps)` nothing
    clips and `f(a) = a * std(gnn_seconds)` exactly (the pre-amendment closed
    form). PAST `a_clipstart`, clipping starts eating the field's low tail;
    on real (23/25 sampled `data/features_v4` graphs) `y` fields this makes
    `f` RISE to an interior peak just past `a_clipstart` and then FALL as
    growing clip mass collapses variance faster than the surviving nodes'
    growing spread can compensate. A naive `f(hi) < std_target => clamp to
    hi` pre-check is therefore wrong: it can fire even when a perfectly good
    (much-less-clipped) root exists on the RISING part of the curve, close to
    `a_clipstart` -- on real data this false-clamp inflated MAE from 85.1 (at
    the true root) to 359.5 (at the falsely-clamped `a=5.0`), because clamping
    to `hi` clips 61-90% of the field instead of the true root's ~5-30%.

    Solve semantics, in order:
      1. If `a_raw = std_seconds / std(gnn_seconds) <= a_clipstart`: nothing
         clips at `a_raw`, so use the closed form directly (clamped to
         `a_clamp` -- fast path, identical to the pre-amendment closed form
         to floating-point precision).
      2. Otherwise, coarse-scan `a` over `[max(a_clipstart, a_clamp[0]),
         a_clamp[1]]` (64 points) for the FIRST sign change of
         `f(a) - std_seconds`. Because `f` starts below `std_seconds` at
         `a_clipstart` (by construction of this branch) and is scanned
         low-to-high, the first crossing is the SMALLEST root -- i.e. the
         LEAST-clipped `a` that hits the target, always preferred over any
         larger root further out on the peak's falling side. Bisect inside
         that bracket to refine it; `info["a_clamped"]` is `False`.
      3. If no crossing exists anywhere on the scan (`std_seconds` exceeds
         even the interior peak, or the scan range is empty), the target is
         genuinely unreachable: pick the SCANNED POINT whose `f` is closest
         to `std_seconds` (not blindly `a_clamp[1]` -- the peak is often a
         much better fit than the ceiling, exactly the failure mode above)
         and set `info["a_clamped"] = True`.

    `info["clip_frac"]` reports how much of the FINAL field hit the 0 floor,
    so callers can flag graphs where the GBM's total/std pair is inconsistent
    with the GNN's shape; if `clip_frac > 0.5` (a majority of the field
    clipped -- a strong scale/anchor-inconsistency signal) this also logs a
    warning.

    Degenerate fallback: if the GNN field is (numerically) flat, `a` is
    undefined (division by ~0), so this falls back to
    `hybrid_field_max_anchor(gnn_seconds, total_seconds, q=q)` and logs a
    warning; `info["degenerate_fallback"]` is `True` in that case and `a` is
    reported as `1.0` (a no-op scale, since there is no shape to rescale).
    This guard is unaffected by the amendment above -- it is checked before
    any `a`-solve is attempted.

    AMENDMENT 2 (owner-approved 2026-07-31, see the "AMENDMENT 2" block in
    `docs/superpowers/specs/2026-07-31-variance-recomposition-design.md`):
    the corrected solver exposed the next layer of error -- the GBM std
    head's own per-part noise (p10-p90 ~ 0.66-1.36x; BUKSA gold
    over-predicted at ~1.3x) feeds straight into MAE through the spread it
    now honestly hits. `lambda_shrink` (default `1.0`, fully trusting
    `std_seconds`) lets a caller partially discount that noisy target via
    GEOMETRIC interpolation between the GNN's OWN spread and the GBM's
    predicted spread, applied BEFORE the (unchanged) clip-aware solve above:

        std_eff = std(gnn_seconds)**(1 - lambda_shrink) * std_seconds**lambda_shrink

    `lambda_shrink=1.0` reproduces the full-trust behavior exactly (`std_eff
    == std_seconds`) -- this is the ONLY code path change, gated by `if
    lambda_shrink != 1.0:` so the default is untouched byte-for-bit.
    `lambda_shrink=0.0` degenerates to `std_eff == std(gnn_seconds)`, i.e.
    the solve targets the GNN's own (unscaled) spread -- in a no-clip case
    this makes `a == 1.0` and the result IS `hybrid_field_max_anchor`'s
    output exactly (no spread rescaling at all, quantile-anchor-only level
    shift). Intermediate `lambda_shrink` partially shrinks the GBM's spread
    prediction toward the GNN's own, geometric-mean-style. Selection
    protocol (owner): `lambda_shrink` is tuned on `e_val` only (per-BC
    `std_ratio` vs. the 0.9 floor + MAE); the GOLD assemblies stay a pure
    held-out test, evaluated once at the chosen value.

    `std_seconds <= 0` is a degenerate SCALE target (the geometric
    interpolation is undefined/nonsensical for a non-positive base) --
    ONLY WHEN `lambda_shrink != 1.0` (the default path never inspects
    `std_seconds`'s sign, matching pre-amendment-2 behavior exactly), this
    falls back to `hybrid_field_max_anchor` with the same loud-warning
    pattern as the degenerate-GNN-field guard above.

    Returns `(pred, info)` where `info = {"a": float, "a_clamped": bool,
    "degenerate_fallback": bool, "clip_frac": float, "lambda_shrink": float,
    "std_target_effective": float}`. NOTE `a_clamped` semantics since the
    smallest-root correction: True means "the requested std target was
    UNREACHABLE" (closest-fit `a` returned, which may be an interior scan
    point) -- it no longer implies `a` sits exactly on an `a_clamp`
    boundary. `std_target_effective` is `std_eff` above (`== std_seconds`
    when `lambda_shrink == 1.0`).
    """
    g = np.asarray(gnn_seconds, dtype=np.float64)
    s = float(np.std(g))
    lambda_shrink = float(lambda_shrink)
    if s < 1e-9:
        logger.warning("hybrid_field_affine: degenerate flat GNN field (std=%.3g) -- "
                       "falling back to quantile-anchor recomposition", s)
        pred = hybrid_field_max_anchor(g, total_seconds, q=q)
        return pred, {"a": 1.0, "a_clamped": False, "degenerate_fallback": True,
                      "clip_frac": float((pred == 0.0).mean()),
                      "lambda_shrink": lambda_shrink,
                      "std_target_effective": float(std_seconds)}

    std_target = float(std_seconds)
    if lambda_shrink != 1.0:
        if std_target <= 0.0:
            logger.warning("hybrid_field_affine: degenerate std target "
                           "(std_seconds=%.3g, lambda_shrink=%.3g) -- falling back to "
                           "quantile-anchor recomposition", std_target, lambda_shrink)
            pred = hybrid_field_max_anchor(g, total_seconds, q=q)
            return pred, {"a": 1.0, "a_clamped": False, "degenerate_fallback": True,
                          "clip_frac": float((pred == 0.0).mean()),
                          "lambda_shrink": lambda_shrink,
                          "std_target_effective": std_target}
        std_target = (s ** (1.0 - lambda_shrink)) * (std_target ** lambda_shrink)
    std_target_effective = std_target

    lo, hi = float(a_clamp[0]), float(a_clamp[1])
    total = float(total_seconds)

    # `d` and `a_clipstart` depend only on the (fixed) GNN field, the quantile
    # `q`, and `total` -- computed ONCE, not inside any search loop.
    c = g - g.mean()
    shift = float(np.quantile(c, q))
    d = c - shift
    eps = 1e-9
    a_clipstart = total / max(-float(d.min()), eps)

    def _raw_for_a(a):
        return a * d + total

    def _post_clip_std(a):
        return float(np.std(np.clip(_raw_for_a(a), 0.0, None)))

    a_raw = std_target / s

    if a_raw <= a_clipstart:
        # At a_raw <= a_clipstart the SOLUTION a_raw clips nothing, so
        # f(a_raw) = a_raw*s exactly (closed form; identical to the
        # pre-amendment behavior to floating-point precision). NOTE: if
        # a_raw < lo the clamp to lo can still clip (total << spread).
        a = float(np.clip(a_raw, lo, hi))
        a_clamped = bool(a != a_raw)
        raw = _raw_for_a(a)
    else:
        # a_raw would clip. f is not monotone past a_clipstart -- it rises to
        # an interior peak, then falls. Coarse-scan for the smallest (first,
        # least-clipping) root and bisect to refine it.
        lo_search = max(a_clipstart, lo)
        if lo_search >= hi:
            a, a_clamped = hi, True
            raw = _raw_for_a(a)
        else:
            n_scan = 64
            scan_a = np.linspace(lo_search, hi, n_scan)
            scan_f = np.array([_post_clip_std(x) for x in scan_a])
            diffs = scan_f - std_target
            crossing = None
            for i in range(1, n_scan):
                if diffs[i - 1] == 0.0 or diffs[i - 1] * diffs[i] <= 0.0:
                    crossing = i
                    break
            if crossing is not None:
                blo, bhi = float(scan_a[crossing - 1]), float(scan_a[crossing])
                flo = float(diffs[crossing - 1])
                if flo == 0.0:
                    a = blo
                else:
                    for _ in range(50):
                        mid = 0.5 * (blo + bhi)
                        fm = _post_clip_std(mid) - std_target
                        if (fm > 0.0) == (flo > 0.0):
                            blo, flo = mid, fm
                        else:
                            bhi = mid
                        if (bhi - blo) <= 1e-9 * max(abs(bhi), 1e-12):
                            break
                    a = 0.5 * (blo + bhi)
                a_clamped = False
                raw = _raw_for_a(a)
            else:
                # No crossing anywhere on the scan: std_target is unreachable
                # (e.g. it exceeds the interior peak). Pick the scanned point
                # CLOSEST to the target rather than blindly a_clamp[1].
                best_idx = int(np.argmin(np.abs(diffs)))
                a = float(scan_a[best_idx])
                a_clamped = True
                raw = _raw_for_a(a)

    clip_frac = float((raw < 0.0).mean())
    if clip_frac > 0.5:
        logger.warning("hybrid_field_affine: majority of field clipped "
                       "(clip_frac=%.3f) -- GBM total/std may be inconsistent "
                       "with the GNN shape", clip_frac)
    return np.clip(raw, 0.0, None), {"a": a, "a_clamped": a_clamped,
                                     "degenerate_fallback": False, "clip_frac": clip_frac,
                                     "lambda_shrink": lambda_shrink,
                                     "std_target_effective": std_target_effective}


def hybrid_field_soft(gnn_seconds: np.ndarray, total_seconds: float, std_seconds: float,
                      q: float = 0.995, a_clamp: tuple = (0.3, 5.0),
                      w_total: float = 1.0, w_std: float = 1.0,
                      lambda_shrink: float = 1.0) -> tuple:
    """Soft-anchor recomposition: `hybrid_field_affine` generalized to a WEIGHTED
    two-target objective, so the total anchor may itself deviate a few percent
    when that trade buys a much closer spread match.

    WHY (see `docs/superpowers/specs/2026-07-31-soft-anchor-design.md` §1):
    `hybrid_field_affine`'s quantile-anchor is a HARD constraint
    (`quantile(pred, q) == total_seconds` exactly, by construction of its
    `m(a)`), and the field is 0-clipped from below. Those two constraints
    together form a ceiling: on some real parts (e_val exo stems, std_ratio
    0.874 < the 0.9 floor) there is NO `a` in `a_clamp` for which the
    post-clip field can reach `std_seconds` while ALSO satisfying the hard
    top-anchor exactly -- the affine solve already returns the closest-fit
    root, and that is provably the best available under the hard constraint.
    Relaxing the top anchor into a soft (penalized) term recovers a strictly
    larger solution space: allowing `quantile(pred, q)` to sit a bit off
    `total_seconds` can unlock a much less-clipped field whose spread lands
    far closer to `std_seconds`, at a small, bounded cost to the total.

    Objective (both residuals normalized by their own target, so the weights
    are scale-free across parts of wildly different absolute magnitude):

        pred(a, m) = clip(a * (gnn - mean(gnn)) + m, 0, None)
        r_total    = (quantile(pred, q) - total_seconds) / total_seconds
        r_std      = (std(pred) - std_target_eff) / std_target_eff
        J(a, m)    = w_total * r_total**2 + w_std * r_std**2

    where `std_target_eff` is `std_seconds` shrunk toward the GNN's own
    spread by `lambda_shrink` (identical convention to `hybrid_field_affine`;
    delegated to it -- see its AMENDMENT 2 docstring). `a` ranges over
    `a_clamp`; `m` is a free additive constant (no longer tied to the hard
    anchor's `m(a) = total - a*quantile(centered_gnn, q)`, though that
    formula is exactly where the SEED point comes from, below).

    Solver: `hybrid_field_affine`'s own clip-aware smallest-root solve
    supplies the SEED `(a_seed, m_seed)` -- `a_seed` is its returned `a`;
    `m_seed` is recovered from the identity `m(a) = total_seconds -
    a*quantile(gnn - mean(gnn), q)` that its solve always satisfies (true in
    every branch, including both degenerate fallbacks: `hybrid_field_max_anchor`
    is itself `a=1, m = total - quantile(centered_gnn, q)` after the same
    quantile-shift-equivariance identity). A coarse grid -- 64 log-spaced `a`
    over `a_clamp` x 33 linear `m` CENTERED ON THE SEED, `m_seed +
    total_seconds*[-0.3, +0.3]` -- is evaluated (vectorized per row: one
    clip+quantile+std pass per `a`, across all `m` at once), PLUS the single
    exact seed point `(a_seed, m_seed)` folded into the running best (not
    cross-producted against either axis -- that would silently widen the
    search past the intended band). REVIEW FIX (2026-07-31, round 1,
    CRITICAL-1): the window was originally the ABSOLUTE
    `total_seconds*[0.7,1.3]` (centered on `total_seconds`, not on
    `m_seed`) -- measured on 58 real `features_v4` fields, `m_seed/total in
    [0.03, 0.61]`, i.e. `m_seed` was NEVER inside that window, so the grid
    could only ever explore points strictly worse than the seed and the
    solver never engaged on the exo regime it exists for. Centering the
    window on `m_seed` is what actually lets the search reach nearby, less-
    clipped `(a, m)` combinations. ONE 9x9 local refinement grid (log-spaced
    `a` / linear `m`, windowed to one coarse grid cell around whichever point
    -- grid or seed -- is best so far, bounded within the same centered
    window) is then evaluated the same way.

    DEGENERATION GUARANTEE: `J_seed` is computed EXACTLY at `(a_seed, m_seed)`
    before any grid search runs (not interpolated from a grid cell -- it is a
    direct, separate evaluation). A grid or refinement candidate is accepted
    ONLY if it beats the seed by a strict margin (`J_candidate < J_seed -
    1e-12`); otherwise this returns the SEED array unchanged (same object,
    not a recomputation) with the seed's own residuals. This is the guard
    against the v5 track's earlier critical finding that `f` (the affine
    solve's post-clip std, as a function of `a`) is NOT monotone/convex on
    real fields -- the same non-convexity applies to `J(a, m)` here, so the
    grid+refinement scheme never ASSUMES a single interior optimum; it only
    ever accepts a point that is verifiably better than the seed by direct
    evaluation. On any field where the affine seed already satisfies both
    targets (no-clip case), this makes `hybrid_field_soft` numerically
    identical to `hybrid_field_affine` -- and, as a PERFORMANCE consequence
    (REVIEW FIX, CRITICAL-2a), when `J_seed < 1e-12` this returns immediately
    WITHOUT building or evaluating any grid at all (healthy/no-clip parts pay
    only the affine solve's own cost). When the grid does run (REVIEW FIX,
    CRITICAL-2b), its per-`a` quantile term is computed via the closed-form
    `a*shift + m` (exact whenever `clip_frac < 1 - q`, which always holds in
    practice -- see the in-code comment above `_grid_eval`) instead of a per-
    point `np.quantile`/`np.partition` call, and the grid's winning candidate
    is re-evaluated with the exact `_residuals()` helper before the
    accept/reject decision, so no approximation reaches the returned
    `info`/prediction -- this keeps a 10k-node solve inside the
    ~200ms budget even when the search engages.

    `lambda_shrink` and the degenerate-field/degenerate-std-target fallbacks
    are delegated verbatim to `hybrid_field_affine` (same warnings, same
    `hybrid_field_max_anchor` escape hatch) -- when `info["degenerate_fallback"]`
    comes back `True` there is no field shape left to trade off, so this
    returns that fallback's prediction unchanged (no grid search).

    Returns `(pred, info)` where `info = {"a": float, "m": float,
    "target_unreachable": bool, "degenerate_fallback": bool,
    "clip_frac": float, "total_deviation_frac": float, "j_total_resid": float,
    "j_std_resid": float, "lambda_shrink": float,
    "std_target_effective": float}`. `target_unreachable` repurposes
    `hybrid_field_affine`'s `a_clamped` (`True` means the HARD two-target
    problem the affine seed solved was itself unreachable at some `a` -- the
    signal that motivated relaxing to a soft objective in the first place).
    `clip_frac` is measured on the FINAL (returned) field. `total_deviation_frac
    = r_total` of the final field (positive = the top anchor rose above
    `total_seconds`). `j_total_resid`/`j_std_resid` are `J`'s two weighted,
    squared terms at the final `(a, m)` (`J == j_total_resid + j_std_resid`).

    REVIEW FIX (round 1, LOW): `w_total`/`w_std` must both be `>= 0` (a
    negative weight REWARDS moving away from that target instead of
    penalizing deviation from it -- a reviewer-demonstrated `w_total=-1.0`
    silently drove `a` to the `a_clamp` ceiling with a +273% total
    deviation, since minimizing `J` then means MAXIMIZING `r_total**2`) and
    may not both be exactly `0` (the objective `J(a, m)` would be
    identically zero everywhere, leaving the grid/refinement search no
    signal to optimize against at all) -- either violation raises a loud
    `ValueError` before any solve is attempted.
    """
    g = np.asarray(gnn_seconds, dtype=np.float64)
    total = float(total_seconds)
    q = float(q)
    w_total = float(w_total)
    w_std = float(w_std)

    # REVIEW FIX (round 1, LOW): a negative weight would reward the search
    # for moving AWAY from that target rather than penalizing deviation from
    # it (minimizing J then means MAXIMIZING that residual's square), and
    # both-zero weights make J identically zero everywhere -- fail loudly
    # rather than silently returning a meaningless/adversarial optimum.
    if w_total < 0.0 or w_std < 0.0:
        raise ValueError(
            f"hybrid_field_soft: w_total/w_std must be >= 0 (got "
            f"w_total={w_total!r}, w_std={w_std!r}) -- a negative weight would "
            f"reward moving AWAY from that target instead of penalizing "
            f"deviation from it")
    if w_total == 0.0 and w_std == 0.0:
        raise ValueError(
            "hybrid_field_soft: w_total and w_std cannot both be 0 -- the "
            "objective J(a, m) would be identically zero everywhere, leaving "
            "the search no signal to optimize against")

    # REVIEW FIX (LOW-6): np.logspace(log10(a_clamp[0]), ...) below silently
    # produces NaN/-inf for a non-positive lower clamp; fail loudly instead,
    # regardless of whether this call ends up taking the no-grid fast path.
    if float(a_clamp[0]) <= 0.0:
        raise ValueError(f"hybrid_field_soft: a_clamp[0] must be > 0 (got {a_clamp[0]!r}) "
                         "-- the a-grid is log-spaced and cannot include a non-positive "
                         "lower bound")

    pred_seed, info_affine = hybrid_field_affine(g, total, std_seconds, q=q,
                                                 a_clamp=a_clamp,
                                                 lambda_shrink=lambda_shrink)

    # `c`/`shift` recover m(a) = total - a*shift for ANY branch of the affine
    # solve, including both its degenerate fallbacks (each is exactly a=1,
    # m = total - shift, by the quantile-shift-equivariance identity used in
    # `hybrid_field_affine`'s own AMENDMENT derivation).
    c = g - g.mean()
    shift = float(np.quantile(c, q))
    a_seed = float(info_affine["a"])
    m_seed = total - a_seed * shift
    std_target_eff = float(info_affine["std_target_effective"])

    denom_total = total if abs(total) > 1e-9 else (1e-9 if total >= 0.0 else -1e-9)
    denom_std = std_target_eff if abs(std_target_eff) > 1e-9 else 1e-9

    def _residuals(pred):
        r_total = (float(np.quantile(pred, q)) - total) / denom_total
        r_std = (float(np.std(pred)) - std_target_eff) / denom_std
        return r_total, r_std

    def _build_info(final_a, final_m, final_rt, final_rs, clip_frac, target_unreachable):
        if clip_frac > 0.5:
            logger.warning("hybrid_field_soft: majority of the accepted soft-anchor "
                           "field clipped (clip_frac=%.3f) -- GBM total/std may be "
                           "inconsistent with the GNN shape", clip_frac)
        return {
            "a": final_a, "m": final_m, "target_unreachable": target_unreachable,
            "degenerate_fallback": False, "clip_frac": clip_frac,
            "total_deviation_frac": final_rt,
            "j_total_resid": float(w_total * final_rt ** 2),
            "j_std_resid": float(w_std * final_rs ** 2),
            "lambda_shrink": info_affine["lambda_shrink"],
            "std_target_effective": std_target_eff,
        }

    r_total_seed, r_std_seed = _residuals(pred_seed)
    j_seed = w_total * r_total_seed ** 2 + w_std * r_std_seed ** 2

    if info_affine["degenerate_fallback"]:
        return pred_seed, {
            "a": a_seed, "m": m_seed, "target_unreachable": False,
            "degenerate_fallback": True, "clip_frac": info_affine["clip_frac"],
            "total_deviation_frac": r_total_seed,
            "j_total_resid": float(w_total * r_total_seed ** 2),
            "j_std_resid": float(w_std * r_std_seed ** 2),
            "lambda_shrink": info_affine["lambda_shrink"],
            "std_target_effective": std_target_eff,
        }

    target_unreachable = bool(info_affine["a_clamped"])
    seed_clip_frac = float((c * a_seed + m_seed < 0.0).mean())

    # REVIEW FIX (CRITICAL-2a): when the seed already satisfies both targets
    # (J_seed ~ 0), skip the grid/refinement search entirely -- healthy,
    # no-clip parts then pay only the affine solve's own cost (~ms), not the
    # grid's. This is the common case in production (most parts do not need
    # the soft trade-off at all).
    if j_seed < 1e-12:
        return pred_seed, _build_info(a_seed, m_seed, r_total_seed, r_std_seed,
                                      seed_clip_frac, target_unreachable)

    lo, hi = float(a_clamp[0]), float(a_clamp[1])
    n_a, n_m = 64, 33
    a_grid = np.logspace(np.log10(lo), np.log10(hi), n_a)
    # REVIEW FIX (CRITICAL-1): the m-window is CENTERED ON THE SEED
    # (`m_seed + total*[-0.3, +0.3]`), not on the absolute `total*[0.7, 1.3]`
    # -- see the docstring's "REVIEW FIX (2026-07-31, round 1, CRITICAL-1)"
    # note for why the absolute window never contained `m_seed` on real
    # fields and so never let the solver engage.
    m_lo, m_hi = sorted((m_seed - 0.3 * total, m_seed + 0.3 * total))
    m_grid = np.linspace(m_lo, m_hi, n_m)

    # REVIEW FIX (CRITICAL-2b): the grid's hot loop no longer calls
    # `np.quantile`/`np.partition` PER GRID POINT for the top-quantile term.
    # `raw(a, m) = a*c + m` is a POSITIVE-scale-then-shift map, which is
    # order-preserving; clipping at 0 only ever affects the LOW tail. The
    # q-quantile of `clip(raw, 0, None)` therefore equals the q-quantile of
    # the UNCLIPPED `raw` -- which is just `a*shift + m` (`shift =
    # quantile(c, q)`, computed once above) -- exactly, as long as the number
    # of clipped nodes is <= floor((n-1)*q), i.e. clip_frac up to ~q = 0.995
    # (review-verified exact at clip_frac 0.00 / 0.43 / 0.83 on real fields;
    # rel err <= 1.7e-16). The identity only breaks at clip_frac == 1.0, and
    # there it is ONE-SIDED SAFE: the closed form returns the negative raw
    # quantile vs the true 0, so r_total < -1 and J is OVER-estimated --
    # such a point can never be falsely selected for any total > 0. This
    # turns an O(n_nodes log n_nodes)-per-point cost into an O(1)-per-point
    # closed form; the winning candidate is re-scored through the exact
    # `_residuals()` before acceptance regardless. `np.std` still needs the actual clipped per-node values (clip
    # changes variance nontrivially), so it remains the only per-node
    # computation in the search; the search's own (a, m) argmin is therefore
    # APPROXIMATE only in the sense that its objective is now built from an
    # exact-closed-form quantile + an exact std -- i.e. not approximate at
    # all for the quantile term, and exact for std. Regardless, the FINAL
    # accept/reject decision below re-evaluates the winning candidate's
    # residuals via the exact `_residuals()` helper (a single `np.quantile`
    # call, not a grid-wide one) before comparing to `j_seed`, so no
    # approximation ever reaches the returned `info`/decision.
    def _grid_eval(a_vals, m_vals):
        n_av, n_mv = len(a_vals), len(m_vals)
        J = np.empty((n_av, n_mv))
        Rt = np.empty((n_av, n_mv))
        Rs = np.empty((n_av, n_mv))
        for i, av in enumerate(a_vals):
            raw = m_vals[:, None] + av * c[None, :]  # (n_mv, n_nodes)
            clipped = np.clip(raw, 0.0, None)
            qv = av * shift + m_vals
            sv = np.std(clipped, axis=-1)
            rt = (qv - total) / denom_total
            rs = (sv - std_target_eff) / denom_std
            J[i] = w_total * rt ** 2 + w_std * rs ** 2
            Rt[i] = rt
            Rs[i] = rs
        return J, Rt, Rs

    # Coarse 64x33 (a, m) grid, PLUS the single explicit seed point (NOT a
    # cross-product extension of either axis -- crossing m_seed against every
    # `a` would silently widen the m-search far outside the intended window).
    # The seed's own (J_seed, r_total_seed, r_std_seed) were already computed
    # exactly above, so it is folded into the "best so far" comparison
    # directly rather than re-evaluated here.
    J_coarse, Rt_coarse, Rs_coarse = _grid_eval(a_grid, m_grid)
    idx = np.unravel_index(int(np.argmin(J_coarse)), J_coarse.shape)
    best_a, best_m = float(a_grid[idx[0]]), float(m_grid[idx[1]])
    best_J = float(J_coarse[idx])
    best_rt, best_rs = float(Rt_coarse[idx]), float(Rs_coarse[idx])

    if j_seed < best_J:
        best_a, best_m, best_J = a_seed, m_seed, j_seed
        best_rt, best_rs = r_total_seed, r_std_seed

    # One local refinement pass: a finer 9x9 sub-grid windowed to roughly one
    # coarse grid cell around whichever point (grid or seed) is best so far
    # (log-spaced in a, linear in m), bounded within the same centered window.
    log_lo, log_hi = np.log10(lo), np.log10(hi)
    a_step = (log_hi - log_lo) / max(n_a - 1, 1)
    log_best_a = np.log10(best_a)
    ref_log_a = np.clip(np.linspace(log_best_a - a_step, log_best_a + a_step, 9),
                        log_lo, log_hi)
    ref_a = 10.0 ** ref_log_a

    m_step = (m_hi - m_lo) / max(n_m - 1, 1)
    ref_m = np.clip(np.linspace(best_m - m_step, best_m + m_step, 9), m_lo, m_hi)

    J_ref, Rt_ref, Rs_ref = _grid_eval(ref_a, ref_m)
    idx_r = np.unravel_index(int(np.argmin(J_ref)), J_ref.shape)
    if float(J_ref[idx_r]) < best_J:
        best_a, best_m = float(ref_a[idx_r[0]]), float(ref_m[idx_r[1]])
        best_J = float(J_ref[idx_r])
        best_rt, best_rs = float(Rt_ref[idx_r]), float(Rs_ref[idx_r])

    # Re-evaluate the winning candidate's residuals EXACTLY (a single
    # `np.quantile` + `np.std` call, not a grid-wide pass) before the
    # accept/reject decision -- the search above is already exact for
    # everything but relies on the closed-form quantile identity holding, so
    # this is a cheap final confirmation, not a correction of a known bias.
    if (best_a, best_m) != (a_seed, m_seed):
        best_pred = np.clip(best_a * c + best_m, 0.0, None)
        best_rt, best_rs = _residuals(best_pred)
        best_J = w_total * best_rt ** 2 + w_std * best_rs ** 2

    if best_J < j_seed - 1e-12:
        final_a, final_m = best_a, best_m
        final_raw = final_a * c + final_m
        final_pred = np.clip(final_raw, 0.0, None)
        final_rt, final_rs = best_rt, best_rs
    else:
        final_a, final_m = a_seed, m_seed
        final_raw = final_a * c + final_m
        final_pred = pred_seed
        final_rt, final_rs = r_total_seed, r_std_seed

    clip_frac = float((final_raw < 0.0).mean())
    return final_pred, _build_info(final_a, final_m, final_rt, final_rs, clip_frac,
                                   target_unreachable)
