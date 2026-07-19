"""Per-node solidification time from temperature snapshots.

For each node, find the first time its temperature crosses T_SOLIDUS
(cooling, high->low) and linearly interpolate the crossing time. Nodes that
never cross use the last simulation time. Ported from the verified logic in
the original dataset.py::_compute_y_nodal, vectorised in NumPy."""
import numpy as np

T_SOLIDUS = 1120.0


def solidification_time(temps: np.ndarray, times: np.ndarray,
                        t_solidus: float = T_SOLIDUS) -> np.ndarray:
    temps = np.asarray(temps, dtype=np.float64)   # (T, N)
    times = np.asarray(times, dtype=np.float64)   # (T,)
    T, N = temps.shape

    below = temps <= t_solidus                     # (T, N)
    any_below = below.any(axis=0)                  # (N,)
    first_s = below.argmax(axis=0)                 # (N,) first True row, 0 if none
    s = np.clip(first_s, 1, None)                  # avoid index -1
    ni = np.arange(N)

    t_hi = temps[s, ni]        # temp at crossing snapshot
    t_lo = temps[s - 1, ni]    # temp at previous snapshot
    dt = t_hi - t_lo
    dt = np.where(np.abs(dt) < 1e-10, -1.0, dt)
    alpha = (t_solidus - t_lo) / dt
    y = times[s - 1] + alpha * (times[s] - times[s - 1])

    # already below at first snapshot -> time[0]
    y = np.where((first_s == 0) & below[0], times[0], y)
    # never crosses -> last time
    y = np.where(~any_below, times[-1], y)
    return y.astype(np.float32)


def global_time(y_nodal: np.ndarray) -> float:
    return float(np.asarray(y_nodal).max())
