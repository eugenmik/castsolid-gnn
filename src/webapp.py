"""Local trame + PyVista web app: upload a STEP casting, run the Phase 2A hybrid
prediction, and view the solidification field with a time slider (solidified =
ghost, liquid = opaque) and a hot-spot toggle. Run: python -m src.webapp

Part B Task 5 adds an assembly mode: a multi-solid STEP upload shows a body
table (role + params per solid) and a role-colored 3-D preview instead of
auto-running the prediction; a single-solid upload keeps the original flow
untouched."""
import csv
import os
import os.path as osp

import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import html, vuetify3
from pyvista.trame.ui import plotter_ui

from src import viz
from src.assembly import list_solids, validate_bodies
from src.inference_props import DEFAULT_ALLOY, list_alloys, list_molds
from src.predict_casting import predict_casting
from tools.doe.load_properties import load_chill_effusivities

DEFAULT_MOLD = "green_sand"
SLEEVE_REGISTRY_CSV = "tools/sleeves/sleeve_registry.csv"

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

# --- assembly mode (Part B Task 5) -----------------------------------------
# One row per solid in a multi-body STEP upload: role dropdown (defaults
# largest solid = casting, everything else = ignore) + role-dependent params
# (chill material / sleeve name) feeding `src.assembly.validate_bodies` and
# `predict_casting(..., bodies=...)`. The 3-D preview colors each solid's
# coarse tessellation by its currently-selected role; "ignore" bodies stay
# the casting's neutral grey but at the same faint GHOST_OPACITY used for the
# solidified-region ghost actor elsewhere in this file.
ROLE_OPTIONS = ["casting", "chill", "insulating_sleeve", "exo_sleeve", "ignore"]
ROLE_COLORS = {
    "casting": "#9e9e9e",
    "chill": "#2196f3",
    "insulating_sleeve": "#ffc107",
    "exo_sleeve": "#f44336",
    "ignore": "#9e9e9e",
}

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


def _run_prediction(step_path, out_dir, alloy, mold, bodies=None):
    """Thin wrapper around `predict_casting` forwarding alloy/mold conditioning
    and (Part B Task 5) an optional multi-body role assignment.

    Extracted so the upload handler's predict call is unit-testable without
    driving the full trame upload flow (module-level lookup of
    `predict_casting` so tests can monkeypatch `src.webapp.predict_casting`).
    `bodies` is only forwarded as a keyword when given (not merely `None`),
    so the pre-Part-B single-solid call shape (no `bodies` kwarg at all)
    stays byte-identical -- some callers (this module's own existing tests)
    monkeypatch `predict_casting` with a fake that doesn't accept `bodies`.
    """
    if bodies is not None:
        return predict_casting(step_path, out_dir=out_dir, alloy=alloy, mold=mold, bodies=bodies)
    return predict_casting(step_path, out_dir=out_dir, alloy=alloy, mold=mold)


def _load_sleeve_names_by_type(csv_path=SLEEVE_REGISTRY_CSV):
    """{"exo": [name, ...], "insulating": [name, ...]} from
    `tools/sleeves/sleeve_registry.csv`'s `name`/`type` columns, sorted --
    the source of the sleeve-name dropdown options for the assembly panel,
    pre-filtered by role so the UI never offers a sleeve `validate_bodies`
    would reject for a type mismatch (see `src.assembly.validate_bodies`'s
    `_ROLE_TO_SLEEVE_TYPE` cross-check)."""
    out: dict[str, list[str]] = {"exo": [], "insulating": []}
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                names = out.setdefault(row["type"], [])
                names.append(row["name"])
    except FileNotFoundError:
        pass
    for names in out.values():
        names.sort()
    return out


def _assemble_bodies_from_state(state_like) -> list:
    """Pure builder: the assembly panel's per-row state (`state_like.
    assembly_bodies`, one dict per solid with at least `solid_id` and
    `role`, plus an optional `params`) -> the plain `bodies` list
    `src.assembly.validate_bodies`/`predict_casting(..., bodies=...)`
    expect (`{"solid_id", "role", "params"}`, UI-only fields like `name`/
    `volume_mm3` dropped). Takes anything with attribute access
    (`state_like.assembly_bodies`) so it is testable with a plain
    `SimpleNamespace`/fake as well as the real trame state -- no gmsh, no
    validation (that's `validate_bodies`'s job, called separately as a
    pre-flight check by the Run-assembly handler)."""
    rows = state_like.assembly_bodies or []
    bodies = []
    for row in rows:
        body = {"solid_id": row["solid_id"], "role": row.get("role", "ignore")}
        params = row.get("params")
        if params:
            body["params"] = params
        bodies.append(body)
    return bodies


def _solid_preview_meshes(step_path):
    """Coarse per-solid surface tessellation for the assembly 3-D preview:
    one fresh gmsh/OCC session imports `step_path` and generates a SINGLE
    2-D surface mesh over the whole model, but sized PER SOLID (each
    solid's own boundary points get their own target size, its OWN bbox
    diagonal / 15, via `gmsh.model.mesh.setSize` + gmsh's default
    `Mesh.MeshSizeExtendFromBoundary` sizing field) -- a large casting
    tessellates at its own coarse scale while a small chill/sleeve still
    gets enough triangles to read as a shape, instead of one uniform size
    forcing every solid into whichever end of the size range is wrong for
    it (a small uniform size on a big casting is a triangle-count blowup;
    a uniform size sized for the casting leaves a small role body a
    handful of degenerate facets). Returns one `(solid_id,
    pyvista.PolyData)` per 3-D solid (its own boundary surfaces' triangles
    + nodes). `solid_id` matches what `list_solids` reports for the SAME
    STEP file, since both walk a fresh `gmsh.model.occ.importShapes` +
    `gmsh.model.getEntities(3)` and gmsh assigns solid tags
    deterministically from import order.

    Real-gmsh, no mocking -- callers should expect this to be slow-ish (a
    full 2-D mesh pass) but bounded for the coarse sizes used here."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("assembly_preview")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()

        solids = gmsh.model.getEntities(3)
        if not solids:
            return []

        size_by_tag = {}
        for (_dim, tag) in solids:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, tag)
            diag = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
            size_by_tag[tag] = max(diag / 15.0, 1e-3)

        # Global bounds only fence in gmsh's implicit sizing AWAY from any
        # explicitly-sized point (e.g. deep in a solid's interior); the
        # per-solid setSize calls below are what actually drive each
        # solid's own coarse scale.
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(size_by_tag.values()))
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(size_by_tag.values()))
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        for (_dim, tag) in solids:
            boundary_pts = gmsh.model.getBoundary([(3, tag)], recursive=True)
            point_dimtags = [(d, t) for (d, t) in boundary_pts if d == 0]
            if point_dimtags:
                gmsh.model.mesh.setSize(point_dimtags, size_by_tag[tag])
        gmsh.model.mesh.generate(2)

        out = []
        for (_dim, tag) in solids:
            boundary = gmsh.model.getBoundary([(3, tag)], oriented=False)
            surf_tags = [t for (d, t) in boundary if d == 2]
            node_xyz_by_id = {}
            tri_blocks = []
            for surf_tag in surf_tags:
                node_tags, coords, _param = gmsh.model.mesh.getNodes(
                    2, surf_tag, includeBoundary=True)
                coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
                for nid, xyz in zip(node_tags, coords):
                    node_xyz_by_id.setdefault(int(nid), xyz)
                elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, surf_tag)
                for etype, enodes in zip(elem_types, elem_node_tags):
                    if etype != 2:   # gmsh type 2 = 3-node triangle
                        continue
                    tri_blocks.append(np.asarray(enodes, dtype=np.int64).reshape(-1, 3))
            if not tri_blocks or not node_xyz_by_id:
                continue
            tris = np.concatenate(tri_blocks, axis=0)
            node_ids = sorted(node_xyz_by_id)
            local_index = {nid: i for i, nid in enumerate(node_ids)}
            points = np.array([node_xyz_by_id[nid] for nid in node_ids], dtype=np.float64)
            local_tris = np.vectorize(local_index.get)(tris)
            faces = np.hstack(
                [np.full((local_tris.shape[0], 1), 3, dtype=np.int64), local_tris])
            out.append((tag, pv.PolyData(points, faces)))
        return out
    finally:
        gmsh.finalize()


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


def _render_assembly_preview(app):
    """(Re)draw every cached assembly-preview mesh (`app["assembly_meshes"]`,
    built once by `_enter_assembly_mode`), colored/opacity'd by each body's
    CURRENT role in `state.assembly_bodies` (see `ROLE_COLORS`/
    `GHOST_OPACITY`). Called on entering assembly mode and again whenever
    `assembly_bodies` changes (role edits) -- cheap, since it only touches
    actors, never re-tessellates."""
    plotter, server = app["plotter"], app["server"]
    state, ctrl = server.state, server.controller
    plotter.clear_actors()
    _remove_scalar_bar(plotter)
    roles = {b["solid_id"]: b.get("role", "ignore") for b in (state.assembly_bodies or [])}
    meshes = app.get("assembly_meshes") or []
    for solid_id, mesh in meshes:
        role = roles.get(solid_id, "ignore")
        color = ROLE_COLORS.get(role, ROLE_COLORS["ignore"])
        opacity = GHOST_OPACITY if role == "ignore" else 1.0
        plotter.add_mesh(mesh, color=color, opacity=opacity,
                         name=f"assembly_solid_{solid_id}", smooth_shading=True)
    if meshes:
        plotter.reset_camera()
    ctrl.view_update()


def _enter_assembly_mode(app, step_path, rows):
    """Switch the viewer into assembly mode for a multi-solid STEP upload:
    builds the role-colored 3-D preview meshes FIRST (`_solid_preview_meshes`
    into `app["assembly_meshes"]`), THEN seeds `state.assembly_bodies` from
    `list_solids`' rows (default roles -- largest solid = casting, everything
    else = ignore, per the brief) and renders (`_render_assembly_preview`).
    This order matters: `state.assembly_bodies` is watched
    (`_on_assembly_bodies_change` -> `_render_assembly_preview`), so setting
    it BEFORE `app["assembly_meshes"]` is refreshed would briefly recolor
    whatever meshes happened to be cached from a PRIOR upload (e.g. a
    multi -> single -> multi upload sequence) against the new roles -- a
    stale-preview flash. A preview failure (e.g. a STEP gmsh can list solids
    for but not re-tessellate) is surfaced in `state.assembly_error` rather
    than raised -- the body table/role panel still works without a 3-D
    preview."""
    server = app["server"]
    state = server.state
    app["grid"] = None
    app["hotspot_actor"] = None
    state.loaded = False
    state.summary = INITIAL_SUMMARY
    state.assembly_step_path = step_path
    state.assembly_error = ""

    try:
        app["assembly_meshes"] = _solid_preview_meshes(step_path)
    except Exception as e:
        app["assembly_meshes"] = []
        state.assembly_error = f"3D preview failed: {e}"

    bodies = []
    for i, row in enumerate(rows):
        bodies.append({
            "solid_id": row["solid_id"],
            "name": row["name"] or f"solid {row['solid_id']}",
            "volume_mm3": row["volume_mm3"],
            "role": "casting" if i == 0 else "ignore",
            "params": {},
        })
    state.assembly_mode = True
    state.assembly_bodies = bodies
    _render_assembly_preview(app)


def _process_upload(app, step_path, alloy, mold):
    """Core upload logic, extracted from the `uploaded_file` state-change
    handler so it is testable without a browser/file-upload round trip:
    inspects `step_path` via `list_solids`; >1 solid enters assembly mode
    (`_enter_assembly_mode` -- role panel + 3-D preview, prediction deferred
    to the Run-assembly button); exactly 1 solid (or a STEP `list_solids`
    itself can't parse -- fails soft here so it falls through to the SAME
    meshing/healing pipeline `predict_casting` always used) runs the
    original single-solid predict-and-render flow, byte-identical to
    pre-Part-B behavior."""
    server = app["server"]
    state = server.state
    try:
        rows = list_solids(step_path)
    except Exception:
        rows = []   # fall back to the pre-Part-B single-body flow untouched

    if len(rows) > 1:
        _enter_assembly_mode(app, step_path, rows)
        return

    # Single-solid (or unlistable) upload: leave no assembly-mode residue
    # behind -- mirrors clear_casting()'s reset of the same four fields, so
    # a LATER multi-solid upload starts from a clean slate rather than
    # _enter_assembly_mode briefly recoloring stale meshes/rows from a
    # previous assembly session (see _enter_assembly_mode's docstring).
    state.assembly_mode = False
    state.assembly_bodies = []
    state.assembly_error = ""
    state.assembly_step_path = None
    app["assembly_meshes"] = []
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
    sleeve_names_by_type = _load_sleeve_names_by_type()
    for k, v in dict(tau=0.0, tmin=0.0, tmax=1.0, show_hotspots=False,
                     summary=INITIAL_SUMMARY, loaded=False, uploaded_file=None,
                     loading=False, hotspot_size=DEFAULT_HOTSPOT_SIZE,
                     hotspot_cutoff=0.0, animating=False, anim_ready=False,
                     anim_uri="", anim_name="solidification.gif",
                     alloy=DEFAULT_ALLOY, mold=DEFAULT_MOLD,
                     alloy_options=list_alloys(), mold_options=list_molds(),
                     # Assembly mode (Part B Task 5): populated by
                     # _enter_assembly_mode() on a multi-solid STEP upload.
                     assembly_mode=False, assembly_bodies=[], assembly_error="",
                     assembly_step_path=None, role_options=ROLE_OPTIONS,
                     chill_material_options=sorted(load_chill_effusivities()),
                     exo_sleeve_options=sleeve_names_by_type["exo"],
                     insulating_sleeve_options=sleeve_names_by_type["insulating"],
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

    @state.change("assembly_bodies")
    def _on_assembly_bodies_change(**kwargs):
        # Fires on every role/param edit in the body table (trame syncs the
        # whole reactive list back on any nested-item mutation) -- just
        # recolor the already-tessellated preview, never re-tessellate.
        if state.assembly_mode:
            _render_assembly_preview(app)

    def run_assembly_prediction():
        """The assembly panel's "Run" button: build the bodies list from the
        current panel state, pre-flight it through `validate_bodies` so a
        training-regime violation (see `src.assembly.validate_bodies`) shows
        up in the panel (`state.assembly_error`) WITHOUT ever calling
        `predict_casting` -- never a crash for a bad role combination.
        `_assemble_bodies_from_state` itself is inside this same try/except:
        a malformed row (e.g. a missing `solid_id`) is a caller/state bug,
        not a training-regime violation, but it should surface in the panel
        exactly the same way -- never a raw traceback, and never a
        `predict_casting` call on a bodies list that failed to even build."""
        try:
            bodies = _assemble_bodies_from_state(state)
            validate_bodies(bodies)
        except Exception as e:
            state.assembly_error = str(e)
            ctrl.view_update()
            return
        state.assembly_error = ""
        state.loading = True
        state.flush()
        try:
            result = _run_prediction(state.assembly_step_path, "runs/castings",
                                     state.alloy, state.mold, bodies=bodies)
            grid = viz.load_prediction(result["vtu"])
            app["name"] = result["name"]
            state.assembly_mode = False
            render_prediction(app, grid, summary_text(result, alloy=state.alloy, mold=state.mold))
        except Exception as e:            # keep the UI alive on a bad assembly/prediction
            plotter.clear_actors()
            _remove_scalar_bar(plotter)
            app["grid"] = None
            app["hotspot_actor"] = None
            state.loaded = False
            state.summary = f"Prediction failed:\n{e}"
        finally:
            state.loading = False
            ctrl.view_update()

    # Exposed on the controller so the Run button (and tests) can invoke it
    # without a browser round trip.
    ctrl.run_assembly_prediction = run_assembly_prediction

    def clear_casting():
        """Drop the loaded casting and return to a ready-for-upload state."""
        plotter.clear_actors()
        _remove_scalar_bar(plotter)
        app["grid"] = None
        app["hotspot_actor"] = None
        app["assembly_meshes"] = []
        state.loaded = False
        state.summary = INITIAL_SUMMARY
        state.uploaded_file = None
        state.tau, state.tmin, state.tmax = 0.0, 0.0, 1.0
        state.show_hotspots = False
        state.anim_ready = False
        state.anim_uri = ""
        state.animating = False
        state.assembly_mode = False
        state.assembly_bodies = []
        state.assembly_error = ""
        state.assembly_step_path = None
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

                # Assembly mode (Part B Task 5): shown instead of an
                # auto-run prediction whenever the STEP upload has >1
                # solid. One row per body -- role dropdown (defaults
                # largest=casting/rest=ignore, set by _enter_assembly_mode)
                # + a role-dependent params select (chill material / sleeve
                # name, the latter pre-filtered by sleeve type so the UI
                # never offers a pairing validate_bodies would reject).
                # `v_model="body.<path>"` (a literal string, not a
                # (name, default) tuple) is trame's standard per-row
                # editable-table idiom: it binds directly to the `body`
                # loop variable from `v_for`, and edits to `assembly_bodies`
                # sync back to the server as a whole (see
                # `_on_assembly_bodies_change`).
                with html.Div(classes="neuralcast-section", v_show=("assembly_mode",)):
                    html.Div("Assembly (multi-body STEP)",
                             classes="neuralcast-section-label")
                    with html.Template(v_for="body in assembly_bodies", key="body.solid_id"):
                        with html.Div(classes="mb-3"):
                            html.Div(
                                "{{ body.name || ('solid ' + body.solid_id) }} "
                                "— {{ Number(body.volume_mm3).toFixed(0) }} mm³",
                                style=f"font-size:.75rem; color:{TEXT_MUTED}; "
                                      "margin-bottom:2px;",
                            )
                            vuetify3.VSelect(v_model="body.role", items=("role_options",),
                                             label="role", density="compact",
                                             variant="outlined", hide_details=True,
                                             classes="mb-1")
                            vuetify3.VSelect(v_model="body.params.material",
                                             items=("chill_material_options",),
                                             label="chill material", density="compact",
                                             variant="outlined", hide_details=True,
                                             v_if="body.role === 'chill'")
                            vuetify3.VSelect(v_model="body.params.sleeve",
                                             items=("exo_sleeve_options",),
                                             label="sleeve (exo)", density="compact",
                                             variant="outlined", hide_details=True,
                                             v_if="body.role === 'exo_sleeve'")
                            vuetify3.VSelect(v_model="body.params.sleeve",
                                             items=("insulating_sleeve_options",),
                                             label="sleeve (insulating)", density="compact",
                                             variant="outlined", hide_details=True,
                                             v_if="body.role === 'insulating_sleeve'")
                    html.Div("{{ assembly_error }}", v_show=("assembly_error",),
                             style=f"color:{ACCENT_MOLTEN}; font-size:.75rem; "
                                   "margin:4px 0 8px;")
                    vuetify3.VBtn("Run Assembly Prediction",
                                  click=ctrl.run_assembly_prediction,
                                  variant="tonal", size="small", color="primary",
                                  disabled=("loading",), loading=("loading",), block=True)

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
