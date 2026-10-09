"""Recomposition: combine the GNN field shape with the two heads' targets.

The GNN predicts the spatial pattern of the field well but is less reliable
about its absolute level and spread. The heads predict the part's total time
(used as the 99.5% quantile of the field) and the standard deviation of the
field. `hybrid_field_soft` maps the GNN field affinely,

    pred(a, m) = clip(a * (gnn - mean(gnn)) + m, 0, None)

and picks (a, m) to minimize

    J = w_total * r_total**2 + w_std * r_std**2,
    r_total = (quantile(pred, q) - total) / total
    r_std   = (std(pred) - std_target) / std_target.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def hybrid_field_max_anchor(gnn_seconds: np.ndarray, total_seconds: float,
                            q: float = 0.995) -> np.ndarray:
    """Shift the GNN field so its q-quantile equals `total_seconds`, clip at 0."""
    g = np.asarray(gnn_seconds, dtype=np.float64)
    anchor = np.quantile(g, q)
    out = g - anchor + float(total_seconds)
    return np.clip(out, 0.0, None)


def hybrid_field_affine(gnn_seconds: np.ndarray, total_seconds: float,
                        std_seconds: float, q: float = 0.995,
                        a_clamp: tuple = (0.3, 5.0)) -> tuple:
    """Hard two-target fit: the q-quantile is anchored exactly at
    `total_seconds` and the scale `a` is solved so that the clipped field's
    standard deviation matches `std_seconds`.

    Anchoring the quantile gives raw(a) = a * d + total with d fixed by the
    GNN field. Below a_clipstart nothing is clipped and a = std/std(gnn) in
    closed form. Past it, the post-clip spread is not monotone in a (it rises
    to an interior peak and then falls), so the smallest root is found by a
    coarse scan followed by bisection; if no root exists the closest scanned
    point is used and `a_clamped` is set. Returns (field, info)."""
    g = np.asarray(gnn_seconds, dtype=np.float64)
    s = float(np.std(g))
    if s < 1e-9:
        logger.warning("hybrid_field_affine: degenerate flat GNN field (std=%.3g) -- "
                       "falling back to quantile-anchor recomposition", s)
        pred = hybrid_field_max_anchor(g, total_seconds, q=q)
        return pred, {"a": 1.0, "a_clamped": False, "degenerate_fallback": True,
                      "clip_frac": float((pred == 0.0).mean()),
                      "std_target_effective": float(std_seconds)}

    std_target = float(std_seconds)
    lo, hi = float(a_clamp[0]), float(a_clamp[1])
    total = float(total_seconds)

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
        # Nothing is clipped at a_raw, so the closed form is exact. The clamp
        # to `lo` can still clip when the total is small next to the spread.
        a = float(np.clip(a_raw, lo, hi))
        a_clamped = bool(a != a_raw)
        raw = _raw_for_a(a)
    else:
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
                # The target exceeds the reachable spread: take the closest
                # scanned point instead of the upper clamp.
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
                                     "std_target_effective": std_target}


def hybrid_field_soft(gnn_seconds: np.ndarray, total_seconds: float, std_seconds: float,
                      q: float = 0.995, a_clamp: tuple = (0.3, 5.0),
                      w_total: float = 1.0, w_std: float = 1.0) -> tuple:
    """Soft two-target fit (see the module docstring for J).

    Seeded by `hybrid_field_affine`. When the seed already meets both
    targets it is returned unchanged; otherwise a 64 x 33 grid over (a, m),
    a 9 x 9 refinement around the best point and an exact re-check decide
    whether letting the total move buys a better spread match. Returns
    (field, info) with info keys a, m, target_unreachable,
    degenerate_fallback, clip_frac, total_deviation_frac, j_total_resid,
    j_std_resid, std_target_effective."""
    g = np.asarray(gnn_seconds, dtype=np.float64)
    total = float(total_seconds)
    q = float(q)
    w_total = float(w_total)
    w_std = float(w_std)

    if w_total < 0.0 or w_std < 0.0:
        raise ValueError(
            f"hybrid_field_soft: w_total/w_std must be >= 0 (got "
            f"w_total={w_total!r}, w_std={w_std!r})")
    if w_total == 0.0 and w_std == 0.0:
        raise ValueError("hybrid_field_soft: w_total and w_std cannot both be 0")
    if float(a_clamp[0]) <= 0.0:
        raise ValueError(f"hybrid_field_soft: a_clamp[0] must be > 0 (got {a_clamp[0]!r}) "
                         "-- the a-grid is log-spaced")

    pred_seed, info_affine = hybrid_field_affine(g, total, std_seconds, q=q, a_clamp=a_clamp)

    # For every branch of the affine solve, m(a) = total - a * quantile(c, q).
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
            "std_target_effective": std_target_eff,
        }

    target_unreachable = bool(info_affine["a_clamped"])
    seed_clip_frac = float((c * a_seed + m_seed < 0.0).mean())

    # Both targets already met: no search needed (the common case).
    if j_seed < 1e-12:
        return pred_seed, _build_info(a_seed, m_seed, r_total_seed, r_std_seed,
                                      seed_clip_frac, target_unreachable)

    lo, hi = float(a_clamp[0]), float(a_clamp[1])
    n_a, n_m = 64, 33
    a_grid = np.logspace(np.log10(lo), np.log10(hi), n_a)
    # The m-window is centred on the seed.
    m_lo, m_hi = sorted((m_seed - 0.3 * total, m_seed + 0.3 * total))
    m_grid = np.linspace(m_lo, m_hi, n_m)

    # raw(a, m) = a*c + m is order-preserving and clipping only touches the
    # low tail, so quantile(clip(raw), q) == a*shift + m in closed form as
    # long as fewer than ~q of the nodes are clipped. Where that fails (all
    # nodes clipped) J is over-estimated, so such a point is never selected.
    # The winning candidate is re-scored exactly below.
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

    J_coarse, Rt_coarse, Rs_coarse = _grid_eval(a_grid, m_grid)
    idx = np.unravel_index(int(np.argmin(J_coarse)), J_coarse.shape)
    best_a, best_m = float(a_grid[idx[0]]), float(m_grid[idx[1]])
    best_J = float(J_coarse[idx])
    best_rt, best_rs = float(Rt_coarse[idx]), float(Rs_coarse[idx])

    if j_seed < best_J:
        best_a, best_m, best_J = a_seed, m_seed, j_seed
        best_rt, best_rs = r_total_seed, r_std_seed

    # One refinement pass on a 9 x 9 sub-grid about one coarse cell wide
    # around the best point so far (log-spaced in a, linear in m).
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

    # Exact re-check of the winner before accepting it over the seed.
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
