"""PyVista helpers for the solidification viewer. Pure/testable: load a
prediction VTU and derive the ghost mesh, the still-liquid region above a time
threshold, and the hot-spot points."""
import os

import numpy as np
import pyvista as pv


def load_prediction(vtu_path: str) -> pv.UnstructuredGrid:
    grid = pv.read(vtu_path)
    for name in ("solidification_time", "hotspot"):
        if name not in grid.point_data:
            raise ValueError(f"VTU missing point array '{name}'")
    return grid


def time_range(grid):
    t = np.asarray(grid.point_data["solidification_time"])
    return float(t.min()), float(t.max())


def liquid_region(grid, tau: float):
    # cells still liquid at time tau (solidification_time >= tau)
    return grid.threshold(tau, scalars="solidification_time")


def hotspot_points(grid):
    mask = np.asarray(grid.point_data["hotspot"]) > 0.5
    return pv.PolyData(np.asarray(grid.points)[mask])


def points_above_time(grid, t: float) -> pv.PolyData:
    """Points whose solidification_time is >= t. Used to thin out the hot-spot
    cloud with a user-controlled cutoff/threshold slider."""
    times = np.asarray(grid.point_data["solidification_time"])
    mask = times >= t
    pts = pv.PolyData(np.asarray(grid.points)[mask])
    pts.point_data["solidification_time"] = times[mask]
    return pts


def solidification_animation(grid, out_path, *, n_frames=40, fps=10,
                             cmap="turbo_r", ghost_opacity=0.15,
                             scalar_bar_args=None, bg="#f7f7f5",
                             window_size=(1280, 720), camera_position=None):
    """Render an animated GIF of progressive solidification: tau sweeps from
    tmin to tmax; already-solidified material (t < tau) shows as a low-opacity
    ghost of the full body, still-liquid material (t >= tau) is opaque and
    colored by solidification time. Each frame shows the color legend and the
    current time. Returns out_path. Off-screen; independent of any live view,
    except that when `camera_position` is given (e.g. captured from a live
    plotter) it is used verbatim so the GIF renders from that exact viewpoint;
    otherwise the camera is auto-framed via reset_camera()."""
    import pyvista as pv
    tmin, tmax = time_range(grid)
    taus = np.linspace(tmin, tmax, int(n_frames))
    p = pv.Plotter(off_screen=True, window_size=list(window_size))
    p.set_background(bg)
    try:
        # permanent ghost of the full body (owns the one legend)
        p.add_mesh(grid, scalars="solidification_time", opacity=ghost_opacity,
                   cmap=cmap, clim=[tmin, tmax], smooth_shading=True,
                   show_scalar_bar=True, scalar_bar_args=dict(scalar_bar_args or {}))
        p.add_mesh(liquid_region(grid, tmin), scalars="solidification_time",
                   cmap=cmap, clim=[tmin, tmax], smooth_shading=True,
                   name="liquid", show_scalar_bar=False)
        if camera_position is not None:
            p.camera_position = camera_position
        else:
            p.reset_camera()
        p.open_gif(out_path, fps=int(fps))
        for tau in taus:
            liq = liquid_region(grid, float(tau))
            if liq.n_cells > 0:
                p.add_mesh(liq, scalars="solidification_time", cmap=cmap,
                           clim=[tmin, tmax], smooth_shading=True,
                           name="liquid", show_scalar_bar=False)
            else:
                p.remove_actor("liquid")  # fully solidified -> only the ghost
            p.add_text(f"t = {float(tau):6.1f} s", name="tlabel",
                       position="upper_left", font_size=14, color="black")
            p.write_frame()
    except Exception:
        if os.path.exists(out_path):
            os.unlink(out_path)  # don't leave a truncated/corrupt gif behind
        raise
    finally:
        p.close()
    return out_path
