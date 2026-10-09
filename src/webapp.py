"""Local trame + PyVista web app: upload a STEP casting, run the Phase 2A hybrid
prediction, and view the solidification field with a time slider (solidified =
ghost, liquid = opaque) and a hot-spot toggle. Run: python -m src.webapp

A multi-solid STEP is meshed through the same prep pipeline as the CLI, which
keeps the largest solid as the casting."""
import os
import os.path as osp

import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import html, vuetify3
from pyvista.trame.ui import plotter_ui

from src import viz
from src.inference_props import DEFAULT_ALLOY, list_alloys, list_molds
from src.predict_casting import predict_casting

DEFAULT_MOLD = "green_sand"

pv.OFF_SCREEN = True

GHOST_OPACITY = 0.15
DEFAULT_HOTSPOT_SIZE = 12.0
HOTSPOT_CUTOFF_PERCENTILE = 90.0

# --- visual identity -------------------------------------------------------
# "Neuralcast" = a neural-network surrogate for a foundry casting process, so
# the palette stays in the same molten-metal family as before but is rebuilt
# for a light, legible workbench: warm parchment chrome, white control
# surfaces, and a near-white VTK stage so the inferno field is what draws the
# eye. The one recurring accent is a deeper burnt-orange (readable at text
# contrast on white, unlike the original pale version). Wired into vuetify's
# theme system so every stock control (slider, checkbox, buttons) inherits it.
# The app bar is a deliberate exception: a dark charcoal strip (independent of
# the light theme) so the header reads as instrument-panel chrome holding just
# the title and a status line, not another content surface.
BG_APP = "#f4f2ee"          # app chrome background (warm parchment)
SURFACE = "#ffffff"         # drawer / card surfaces
BORDER = "#ddd8cf"          # hairline dividers
VTK_BG = "#f7f7f5"          # 3D viewport background -- near-white so the
                            # colored field is the highest-contrast thing on screen
ACCENT_MOLTEN = "#c1440e"   # burnt-orange -- the one recurring accent (readable on white)
TEXT_PRIMARY = "#23262b"
TEXT_MUTED = "#6b7078"
HEADER_BG = "#1c1a17"       # dark charcoal app-bar -- deliberately dark in the light theme
HEADER_TEXT = "#f4f2ee"
HEADER_TEXT_MUTED = "#b6afa1"
FIELD_CMAP = "turbo_r"      # reversed turbo (red -> blue): wide, high-contrast gradient
                            # with far more distinguishable steps than a single-hue map,
                            # so small time differences across the field stay legible
HOTSPOT_COLOR = "#0092b8"   # deep cyan/teal marker: falls in the middle of turbo's
                            # range but is rendered as small spheres (not a field
                            # color), so it still reads as an overlay/alert

_MONO_STACK = ('"JetBrains Mono", ui-monospace, "SFMono-Regular", '
              "Menlo, Consolas, monospace")
_COND_STACK = '"Roboto Condensed", "Arial Narrow", sans-serif'

# Scalar bar (legend) for the solidification-time color field: a vertical bar
# pinned to the right edge of the viewport, with generous title/label padding
# so the title never collides with the tick numbers, and integer tick labels
# (fractional seconds add no information at casting scale).
SCALAR_BAR_ARGS = dict(
    title="Solidification Time, s",
    fmt="%.0f",
    color="black",
    bold=True,
    font_family="arial",
    vertical=True,
    position_x=0.91,
    position_y=0.06,
    width=0.055,
    height=0.80,
    title_font_size=20,
    label_font_size=16,
    n_labels=6,
)

_CUSTOM_CSS = f"""
.neuralcast-title {{
  display: flex;
  align-items: center;
  gap: 10px;
}}
.neuralcast-title-text {{
  font-family: {_COND_STACK};
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
  font-size: 1.05rem;
  color: {HEADER_TEXT};
}}
.neuralcast-status {{
  font-family: {_COND_STACK};
  letter-spacing: .08em;
  text-transform: uppercase;
  font-size: .7rem;
  color: {HEADER_TEXT_MUTED};
  white-space: nowrap;
}}
.neuralcast-drawer-content {{
  overflow-y: auto;
}}
.neuralcast-section-label {{
  font-family: {_COND_STACK};
  letter-spacing: .12em;
  text-transform: uppercase;
  font-size: .7rem;
  color: {TEXT_MUTED};
  margin-bottom: 10px;
  flex: none;
}}
.neuralcast-section {{
  margin-bottom: 4px;
}}
.neuralcast-section + .neuralcast-section {{
  border-top: 1px solid {BORDER};
  padding-top: 20px;
  margin-top: 20px;
}}
.neuralcast-tau-badge {{
  display: inline-block;
  font-family: {_MONO_STACK};
  font-size: .75rem;
  color: {ACCENT_MOLTEN};
  background: {BG_APP};
  border: 1px solid {BORDER};
  border-radius: 4px;
  padding: 3px 10px;
  white-space: nowrap;
}}
.neuralcast-summary-box {{
  min-height: 0;
}}
.neuralcast-summary-box .v-input {{
  flex: 1 1 auto;
  min-height: 0;
}}
.neuralcast-summary-box .v-field__input,
.neuralcast-summary-box textarea {{
  font-family: {_MONO_STACK} !important;
  font-size: .78rem;
  line-height: 1.5;
  color: {TEXT_PRIMARY};
}}
.neuralcast-footer-text {{
  width: 100%;
  text-align: center;
  font-family: {_COND_STACK};
  letter-spacing: .06em;
  font-size: .72rem;
  color: {TEXT_MUTED};
}}
"""


def summary_text(result: dict, alloy: str = None, mold: str = None) -> str:
    lines = [f"casting: {result['name']}",
             f"total solidification time: {result['total_time_s']:.1f} s",
             f"nodes: {result['n_nodes']}"]
    if alloy is not None or mold is not None:
        cond_parts = []
        if alloy is not None:
            cond_parts.append(f"alloy: {alloy}")
        if mold is not None:
            cond_parts.append(f"mold: {mold}")
        lines.append("  " + ", ".join(cond_parts))
    lines.append("top hot spots (x,y,z mm):")
    for xyz in result["hotspot_xyz"]:
        lines.append(f"  ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})")
    if result.get("ood_warnings"):
        lines.append("WARNINGS:")
        lines += [f"  - {w}" for w in result["ood_warnings"]]
    return "\n".join(lines)


INITIAL_SUMMARY = "(upload a STEP file)"


def _run_prediction(step_path, out_dir, alloy, mold):
    """Thin wrapper around `predict_casting` forwarding alloy/mold conditioning.

    Extracted so the upload handler's predict call is unit-testable without
    driving the full trame upload flow (module-level lookup of
    `predict_casting` so tests can monkeypatch `src.webapp.predict_casting`).
    """
    return predict_casting(step_path, out_dir=out_dir, alloy=alloy, mold=mold)


def _rebuild_hotspots(app):
    """(Re)build the hot-spot point actor from the current cutoff/size state.
    Called on load and whenever the size or cutoff slider moves."""
    plotter, server = app["plotter"], app["server"]
    state, ctrl = server.state, server.controller
    if app["grid"] is None:
        return
    pts = viz.points_above_time(app["grid"], float(state.hotspot_cutoff))
    actor = plotter.add_mesh(pts, color=HOTSPOT_COLOR, point_size=float(state.hotspot_size),
                             name="hotspots", render_points_as_spheres=True)
    actor.SetVisibility(bool(state.show_hotspots))
    app["hotspot_actor"] = actor
    ctrl.view_update()


def _remove_scalar_bar(plotter):
    """Drop any existing scalar bar(s), regardless of how many are present.
    Guarded because pyvista raises if none exist (or if a title is required
    with more than one) -- callers just want a clean slate either way."""
    try:
        plotter.remove_scalar_bar()
    except Exception:
        pass
    # Belt-and-suspenders: clear() takes the whole registry in one shot, in
    # case remove_scalar_bar() above only got one of several.
    try:
        plotter.scalar_bars.clear()
    except Exception:
        pass


def _process_upload(app, step_path, alloy, mold):
    """Core upload logic, extracted from the `uploaded_file` state-change
    handler so it is testable without a browser/file-upload round trip:
    run the prediction and render it, keeping the UI alive on failure."""
    server = app["server"]
    state = server.state
    plotter = app["plotter"]
    out_dir = "runs/castings"
    try:
        result = _run_prediction(step_path, out_dir, alloy, mold)
        grid = viz.load_prediction(result["vtu"])
        app["name"] = result["name"]
        render_prediction(app, grid, summary_text(result, alloy=alloy, mold=mold))
    except Exception as e:            # keep the UI alive on bad/unmeshable STEP
        plotter.clear_actors()
        _remove_scalar_bar(plotter)
        app["grid"] = None
        app["hotspot_actor"] = None
        state.loaded = False
        state.summary = f"Prediction failed for {osp.basename(step_path)}:\n{e}"


def render_prediction(app, grid, summary):
    plotter, server = app["plotter"], app["server"]
    state, ctrl = server.state, server.controller
    plotter.clear_actors()
    _remove_scalar_bar(plotter)
    tmin, tmax = viz.time_range(grid)
    app["grid"] = grid
    # The "ghost" actor is the single, permanent owner of the scalar bar/legend:
    # it is added once here and never touched again by the tau slider, so the
    # legend persists across playback without being re-added or dropped. The
    # "liquid" actor (re-added on every tau change) must NEVER own a scalar bar,
    # or scrubbing the slider would repeatedly add/replace it.
    plotter.add_mesh(grid, scalars="solidification_time", opacity=GHOST_OPACITY,
                     cmap=FIELD_CMAP, name="ghost", clim=[tmin, tmax],
                     smooth_shading=True, show_scalar_bar=True,
                     scalar_bar_args=SCALAR_BAR_ARGS)
    plotter.add_mesh(viz.liquid_region(grid, tmin), scalars="solidification_time",
                     cmap=FIELD_CMAP, name="liquid", clim=[tmin, tmax],
                     smooth_shading=True, show_scalar_bar=False)
    state.tmin, state.tmax, state.tau = tmin, tmax, tmin
    times = np.asarray(grid.point_data["solidification_time"])
    state.hotspot_cutoff = float(np.percentile(times, HOTSPOT_CUTOFF_PERCENTILE))
    _rebuild_hotspots(app)
    state.anim_ready = False
    state.anim_uri = ""
    state.summary, state.loaded = summary, True
    plotter.reset_camera()
    ctrl.view_update()


def build_app(server=None):
    if server is None:
        server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller
    plotter = pv.Plotter()
    plotter.set_background(VTK_BG)
    app = {"server": server, "plotter": plotter, "grid": None, "hotspot_actor": None,
           "name": "casting"}
    server.app = app  # exposes the app dict for tests / headless callers that
                      # need render_prediction()/clear_casting() without the UI

    state.trame__title = "Neuralcast Viewer"

    # Custom light vuetify theme (molten-metal accent) applied through
    # Vuetify's own theme system, so every stock control (slider, checkbox,
    # buttons) picks up the palette automatically instead of one-off color
    # props.
    vuetify_config = {
        "theme": {
            "defaultTheme": "neuralcastLight",
            "themes": {
                "neuralcastLight": {
                    "dark": False,
                    "colors": {
                        "background": BG_APP,
                        "surface": SURFACE,
                        "primary": ACCENT_MOLTEN,
                        "secondary": TEXT_MUTED,
                        "on-background": TEXT_PRIMARY,
                        "on-surface": TEXT_PRIMARY,
                    },
                },
            },
        },
    }
    for k, v in dict(tau=0.0, tmin=0.0, tmax=1.0, show_hotspots=False,
                     summary=INITIAL_SUMMARY, loaded=False, uploaded_file=None,
                     loading=False, hotspot_size=DEFAULT_HOTSPOT_SIZE,
                     hotspot_cutoff=0.0, animating=False, anim_ready=False,
                     anim_uri="", anim_name="solidification.gif",
                     alloy=DEFAULT_ALLOY, mold=DEFAULT_MOLD,
                     alloy_options=list_alloys(), mold_options=list_molds(),
                     ).items():
        state.setdefault(k, v)

    @state.change("tau")
    def _on_tau(tau, **kwargs):
        if app["grid"] is None:
            return
        # show_scalar_bar=False: the "ghost" actor owns the one legend (see
        # render_prediction); re-adding it here would duplicate/replace it on
        # every slider tick.
        liq = viz.liquid_region(app["grid"], tau)
        if liq.n_cells > 0:
            plotter.add_mesh(liq, scalars="solidification_time",
                             cmap=FIELD_CMAP, name="liquid", smooth_shading=True,
                             clim=[state.tmin, state.tmax], show_scalar_bar=False)
        else:
            plotter.remove_actor("liquid")
        ctrl.view_update()

    @state.change("show_hotspots")
    def _on_hot(show_hotspots, **kwargs):
        if app["hotspot_actor"] is not None:
            app["hotspot_actor"].SetVisibility(bool(show_hotspots))
            ctrl.view_update()

    @state.change("hotspot_cutoff", "hotspot_size")
    def _on_hotspot_controls(**kwargs):
        _rebuild_hotspots(app)

    @state.change("uploaded_file")
    def _on_upload(uploaded_file, **kwargs):
        if not uploaded_file:
            return
        f = uploaded_file[0] if isinstance(uploaded_file, list) else uploaded_file
        out_dir = "runs/castings"
        os.makedirs(out_dir, exist_ok=True)
        step_path = osp.join(out_dir, f.get("name", "part.step"))
        with open(step_path, "wb") as fh:
            fh.write(f["content"])
        state.loading = True
        state.flush()
        try:
            _process_upload(app, step_path, state.alloy, state.mold)
        finally:
            state.loading = False
            ctrl.view_update()

    def clear_casting():
        """Drop the loaded casting and return to a ready-for-upload state."""
        plotter.clear_actors()
        _remove_scalar_bar(plotter)
        app["grid"] = None
        app["hotspot_actor"] = None
        state.loaded = False
        state.summary = INITIAL_SUMMARY
        state.uploaded_file = None
        state.tau, state.tmin, state.tmax = 0.0, 0.0, 1.0
        state.show_hotspots = False
        state.anim_ready = False
        state.anim_uri = ""
        state.animating = False
        ctrl.view_update()

    # Exposed on the controller so tests (and any future non-UI caller) can
    # invoke the same clear logic the "Clear" button uses.
    ctrl.clear_casting = clear_casting

    def generate_animation():
        """Render the solidification GIF for the loaded casting and expose it
        to the browser as a base64 data URI (preview + download). Blocking, like
        the upload prediction -- fine for this local single-user app."""
        if app["grid"] is None:
            return
        state.animating = True
        state.flush()
        try:
            out_dir = "runs/castings"
            os.makedirs(out_dir, exist_ok=True)
            gif_path = osp.join(out_dir, f"{app['name']}_solidification.gif")
            viz.solidification_animation(app["grid"], gif_path,
                                         cmap=FIELD_CMAP, ghost_opacity=GHOST_OPACITY,
                                         scalar_bar_args=SCALAR_BAR_ARGS, bg=VTK_BG,
                                         camera_position=plotter.camera_position)
            import base64
            with open(gif_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
            state.anim_uri = f"data:image/gif;base64,{b64}"
            state.anim_name = f"{app['name']}_solidification.gif"
            state.anim_ready = True
        except Exception as e:
            state.anim_ready = False
            state.summary = f"Animation failed:\n{e}"
        finally:
            state.animating = False
            ctrl.view_update()
    ctrl.generate_animation = generate_animation

    with SinglePageWithDrawerLayout(server, vuetify_config=vuetify_config,
                                    width=360) as layout:
        html.Style(_CUSTOM_CSS)

        # The nav-icon toggles the drawer, but the drawer is `permanent` (it
        # stays open as the fixed left control panel), so the toggle has no
        # visible effect -- hide it rather than show a button that does nothing.
        layout.icon.hide()

        # Dark, compact app-bar: it holds only the title and a status line, so
        # it is styled as an instrument-panel strip distinct from the light
        # control surfaces below it, and kept slim since it carries no controls.
        layout.toolbar.color = HEADER_BG
        layout.toolbar.density = "compact"
        layout.toolbar.flat = True

        layout.title.set_text(
            '<div class="neuralcast-title">'
            '<span class="neuralcast-title-text">Neuralcast Viewer</span>'
            "</div>"
        )
        with layout.toolbar:
            vuetify3.VSpacer()
            html.Div(
                "{{ loading ? 'Predicting…' : (loaded ? 'Prediction loaded' "
                ": 'Awaiting upload') }}",
                classes="neuralcast-status mr-2",
            )
            # Thin indeterminate progress bar pinned to the bottom edge of the
            # header -- visible only while a prediction is running, so the user
            # can see the model is working during the blocking STEP -> VTU call.
            vuetify3.VProgressLinear(
                indeterminate=True, absolute=True, location="bottom",
                height=3, color=ACCENT_MOLTEN, v_show=("loading",),
            )

        with layout.drawer:
            with html.Div(classes="d-flex flex-column fill-height pa-4 "
                                  "neuralcast-drawer-content"):
                with html.Div(classes="neuralcast-section"):
                    html.Div("Upload", classes="neuralcast-section-label")
                    vuetify3.VSelect(v_model=("alloy",), items=("alloy_options",),
                                     label="Alloy", density="compact",
                                     variant="outlined", hide_details=True,
                                     classes="mb-2")
                    vuetify3.VSelect(v_model=("mold",), items=("mold_options",),
                                     label="Mold system", density="compact",
                                     variant="outlined", hide_details=True,
                                     classes="mb-2")
                    vuetify3.VFileInput(v_model=("uploaded_file", None), label="STEP file",
                                        prepend_inner_icon="mdi-file-upload-outline",
                                        accept=".step,.stp,.STEP,.STP", density="compact",
                                        variant="outlined", hide_details=True)
                    vuetify3.VBtn("Clear", click=clear_casting, variant="text",
                                  size="small", color="secondary", disabled=("!loaded",),
                                  classes="mt-2", block=True)

                with html.Div(classes="neuralcast-section"):
                    html.Div("Playback", classes="neuralcast-section-label")
                    vuetify3.VSlider(v_model=("tau",), min=("tmin",), max=("tmax",),
                                     label="time (s)",
                                     hide_details=True, disabled=("!loaded",),
                                     thumb_label=True)
                    html.Div(
                        "{{ Number(tau).toFixed(1) }} / {{ Number(tmax).toFixed(1) }} s",
                        classes="neuralcast-tau-badge mt-1",
                    )

                with html.Div(classes="neuralcast-section"):
                    html.Div("Hot Spots", classes="neuralcast-section-label")
                    vuetify3.VCheckbox(v_model=("show_hotspots",), label="show hot spots",
                                       hide_details=True, disabled=("!loaded",))
                    vuetify3.VSlider(v_model=("hotspot_size",), min=2, max=40, step=1,
                                     label="size", hide_details=True,
                                     disabled=("!loaded",), thumb_label=True,
                                     classes="mt-4")
                    vuetify3.VSlider(v_model=("hotspot_cutoff",), min=("tmin",),
                                     max=("tmax",), label="cutoff (s)", hide_details=True,
                                     disabled=("!loaded",), thumb_label=True,
                                     classes="mt-2")

                with html.Div(classes="neuralcast-section"):
                    html.Div("Animation", classes="neuralcast-section-label")
                    vuetify3.VBtn("Generate GIF", click=generate_animation,
                                  variant="tonal", size="small", color="primary",
                                  disabled=("!loaded || animating",), block=True,
                                  loading=("animating",))
                    vuetify3.VProgressLinear(indeterminate=True, height=3,
                                             color=ACCENT_MOLTEN, v_show=("animating",),
                                             classes="mt-2")
                    html.Img(src=("anim_uri",), v_show=("anim_ready",),
                             style="max-width:100%; margin-top:10px; "
                                   f"border:1px solid {BORDER}; border-radius:4px;")
                    html.A("Download GIF", href=("anim_uri",), download=("anim_name",),
                           v_show=("anim_ready",),
                           style=f"display:block; margin-top:8px; color:{ACCENT_MOLTEN}; "
                                 "font-size:.8rem; text-decoration:underline;")

                with html.Div(classes="neuralcast-section flex-grow-1 d-flex "
                                      "flex-column neuralcast-summary-box"):
                    html.Div("Summary", classes="neuralcast-section-label")
                    vuetify3.VTextarea(v_model=("summary",), readonly=True,
                                       variant="outlined", density="compact",
                                       hide_details=True, no_resize=True,
                                       classes="flex-grow-1")

        with layout.content:
            with vuetify3.VContainer(fluid=True, classes="fill-height pa-0"):
                # add_menu=True: keep pyvista's default view-toolbar overlay --
                # it gives reset-camera / view-axis / screenshot controls that
                # users want when inspecting the casting.
                # mode="server": render server-side (VTK on the Python process,
                # frames streamed to the browser) rather than client-side
                # (geometry shipped to the browser and rendered there). This
                # keeps the server-side VTK camera in sync with the user's
                # mouse rotations/pans/zooms, so generate_animation() below can
                # read plotter.camera_position and reuse the exact view for
                # the GIF -- with client-side rendering the server-side camera
                # would stay frozen regardless of what the user sees.
                view = plotter_ui(plotter, add_menu=True, mode="server")
                ctrl.view_update = view.update

        # Replace trame's default footer ("Powered by trame" / Kitware credit)
        # with our own.
        layout.footer.clear()
        with layout.footer:
            html.Div("© 2026 Eugen Miknevic", classes="neuralcast-footer-text")

    # NOTE (deviation from brief): trame's `state.to_dict()` only reflects state
    # that has been *flushed*, and flushing is a no-op until the state is marked
    # `ready()` (normally done when a client connects, i.e. inside `.start()`).
    # Since `build_app()` must be unit-constructible without starting the server,
    # we mark the state ready here so the initial `setdefault(...)` values are
    # visible via `to_dict()`/attribute access immediately after construction.
    # This performs no networking (no client is connected yet) and is idempotent
    # (a later real `.start()` calling `ready()` again is a no-op).
    state.ready()

    return server


def main():
    build_app().start(port=8080)


if __name__ == "__main__":
    main()
