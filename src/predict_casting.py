"""STEP file in, solidification field out.

    python -m src.predict_casting part.step --out-dir out [--alloy G20Mn5]

Steps: mesh the STEP file (healed, density-searched and cached, or at a fixed
element size with --elem-size), run the model, check the training envelope,
screen for shrinkage, and write `<name>.vtu`, `<name>_summary.txt` and the
mesh `<name>.msh` to the output directory. The model weights come from the
Hugging Face Hub unless --weights-dir points at a local copy.
"""
from __future__ import annotations

import argparse
import logging
import os
import os.path as osp

import numpy as np

from src import ood as _ood
from src.features import FEATURE_NAMES
from src.inference_props import DEFAULT_ALLOY, DEFAULT_MOLD, resolve_alloy_family, resolve_props
from src.mesh_cache import get_or_mesh
from src.meshing import DEFAULT_TARGET_NODES, mesh_step
from src.model_api import CastSolidModel
from src.msh_parse import parse_msh
from src.shrinkage import screen_casting

logger = logging.getLogger(__name__)

_VOL_IDX = FEATURE_NAMES.index("volume_mm3")
_GCOOL_IDX = FEATURE_NAMES.index("geodesic_d_cool")


def write_vtu(out_path, node_xyz, tets, seconds, top_frac=0.1, extra_point_data=None):
    """Write the mesh with `solidification_time` (s) and `hotspot` (the
    `top_frac` latest-solidifying nodes) point fields, plus any extras."""
    import meshio
    seconds = np.asarray(seconds, dtype=np.float32)
    n = seconds.shape[0]
    k = max(1, int(round(n * top_frac)))
    thresh = np.sort(seconds)[-k]
    hotspot = (seconds >= thresh).astype(np.int32)
    point_data = {"solidification_time": seconds, "hotspot": hotspot}
    if extra_point_data:
        point_data.update(extra_point_data)
    m = meshio.Mesh(points=np.asarray(node_xyz, dtype=np.float64),
                    cells=[("tetra", np.asarray(tets))],
                    point_data=point_data)
    m.write(out_path)


def _surface_and_feed_nodes(mesh):
    """All tagged surface nodes, and the adiabatic-tagged ones that can feed
    a shrinkage region (none on a STEP meshed by this pipeline)."""
    tri_arrays = [tri for tri in mesh.tris.values() if tri.size]
    surface_nodes = (np.unique(np.concatenate(tri_arrays, axis=0).reshape(-1))
                     if tri_arrays else np.zeros(0, dtype=np.int64))
    adiab_tri = mesh.tris.get("Adiabatic")
    feed_nodes = (np.unique(adiab_tri.reshape(-1))
                  if adiab_tri is not None and adiab_tri.size else np.zeros(0, dtype=np.int64))
    return surface_nodes.astype(np.int64), feed_nodes.astype(np.int64)


def _shrinkage_result_for_return(screen: dict) -> dict:
    """The screening result without its per-node arrays, plus counts."""
    result = {k: v for k, v in screen.items() if k not in ("niyama", "porosity_risk")}
    risk = screen["porosity_risk"]
    result["flagged_frac"] = float(np.mean(risk)) if risk.size else 0.0
    result["n_macro_regions"] = len(screen["macro_shrinkage"])
    result["n_sinks"] = len(screen["top_sinks"])
    result["n_feeders"] = len(screen["feeders"])
    return result


def _shrinkage_point_data(mesh, screen: dict) -> dict:
    n_nodes = mesh.node_xyz.shape[0]
    macro_region_id = np.full(n_nodes, -1, dtype=np.int64)
    for region in screen["macro_shrinkage"]:
        macro_region_id[np.asarray(region["nodes"], dtype=np.int64)] = region["region_id"]
    return {
        "niyama": np.asarray(screen["niyama"], dtype=np.float32),
        "porosity_risk": np.asarray(screen["porosity_risk"], dtype=np.uint8),
        "macro_region_id": macro_region_id,
    }


def _write_shrinkage_summary(f, shrinkage) -> None:
    if shrinkage is None:
        f.write("shrinkage screening: disabled\n")
        return
    f.write("shrinkage screening:\n")
    f.write(f"  niyama_source: {shrinkage['niyama_source']}\n")
    # The risk mask is the lowest Niyama decile of the part, so this
    # percentage is fixed by construction, not a measured severity.
    f.write(f"  flagged nodes: {shrinkage['flagged_frac'] * 100.0:.1f}%"
            f" (bottom decile by construction)\n")
    macro = shrinkage["macro_shrinkage"]
    if macro:
        # Regions under 4 nodes are listed but get no feeder suggestion.
        n_sub_resolution = sum(1 for r in macro if r.get("sub_resolution"))
        region_count = str(len(macro))
        if n_sub_resolution:
            region_count += f" (+{n_sub_resolution} sub-resolution)"
        worst = max(macro, key=lambda r: r["severity"])
        c = worst["centroid"]
        f.write(f"  macro shrinkage regions: {region_count} "
                f"(worst severity={worst['severity']:.4g} mm^3 @ "
                f"({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}))\n")
    else:
        f.write("  macro shrinkage regions: 0\n")
    f.write(f"  sinks: {len(shrinkage['top_sinks'])}\n")
    feeders = shrinkage["feeders"]
    if feeders:
        f.write("  feeder recommendations:\n")
        for feeder in feeders:
            loc = feeder["location"]
            f.write(f"    - location=({loc[0]:.1f},{loc[1]:.1f},{loc[2]:.1f}) mm  "
                    f"min_modulus_mm={feeder['min_modulus_mm']:.4f}\n")
    else:
        f.write("  feeder recommendations: none\n")
    if shrinkage.get("alloy_family_assumed"):
        f.write(f"  alloy_family_assumed: {shrinkage['material']['alloy_family']} "
                f"(alloy not in the materials table -- conservative fallback)\n")


def _mesh(step_path, out_msh, elem_size, target_nodes, cache_dir):
    if elem_size is not None:
        info = mesh_step(step_path, out_msh, elem_size)
        mesh = parse_msh(out_msh)
        return mesh, {"mode": "fixed", "n_nodes": info["n_nodes"], "n_tets": info["n_tets"],
                      "elem_size": info["elem_size"], "cache": "n/a"}
    info = get_or_mesh(step_path, out_msh, target_nodes=target_nodes, cache_dir=cache_dir)
    if not info.get("ok", False):
        raise RuntimeError(f"meshing failed for {step_path!r}: {info.get('notes', '(no notes)')}")
    mesh = parse_msh(out_msh)
    return mesh, {"mode": "density-searched", "n_nodes": info["n_nodes"],
                  "n_tets": int(mesh.tets.shape[0]), "elem_size": None,
                  "cache": info.get("cache", "unknown")}


def predict_casting(step_path, out_dir="runs/castings", alloy=None, mold=DEFAULT_MOLD,
                    props_json=None, elem_size=None,
                    target_nodes: int = DEFAULT_TARGET_NODES,
                    screen_shrinkage: bool = True, gravity: tuple = (0, 0, -1),
                    model: CastSolidModel | None = None, weights_dir: str | None = None,
                    ood_json: str | None = None, cache_dir: str = "runs/mesh_cache",
                    device: str | None = None) -> dict:
    """Mesh `step_path`, predict, and write the result files.

    `alloy`/`mold` name rows of `data/properties/` (default EN-GJS-400-15 in
    green sand); `props_json` replaces both. `elem_size` (mm) meshes at a
    fixed size instead of searching for `target_nodes`. The model is
    downloaded from the Hugging Face Hub unless `weights_dir` names a local
    copy; pass a loaded `model` to reuse it across calls.

    Returns {name, vtu, summary, msh, total_time_s, hotspot_xyz (5 x 3 mm),
    ood_warnings, n_nodes, mesh_mode, elem_size, cache, recomposition_info,
    shrinkage}."""
    if elem_size is not None and target_nodes != DEFAULT_TARGET_NODES:
        raise ValueError("predict_casting: give either elem_size or target_nodes, not both")
    os.makedirs(out_dir, exist_ok=True)
    name = osp.splitext(osp.basename(step_path))[0]
    out_msh = osp.join(out_dir, f"{name}.msh")

    # Resolve properties before meshing so a bad alloy name fails fast.
    props = resolve_props(alloy=alloy, mold=mold, props_json=props_json)
    if model is None:
        model = (CastSolidModel(weights_dir, device=device) if weights_dir
                 else CastSolidModel.from_pretrained(device=device))

    mesh, minfo = _mesh(step_path, out_msh, elem_size, target_nodes, cache_dir)
    seconds, details = model.predict_with_props(mesh, props)
    X = details["X"]

    warnings = []
    ood_json = ood_json or getattr(model, "ood_ranges_path", None)
    if ood_json and osp.exists(ood_json):
        diag = float(np.linalg.norm(mesh.node_xyz.max(0) - mesh.node_xyz.min(0)))
        warnings = _ood.ood_flags(float(X[0, _VOL_IDX]), diag, float(X[:, _GCOOL_IDX].max()),
                                  _ood.load_ranges(ood_json))

    shrinkage = None
    point_data = None
    if screen_shrinkage:
        surface_nodes, feed_nodes = _surface_and_feed_nodes(mesh)
        if surface_nodes.size == 0:
            logger.warning("predict_casting: no surface nodes on the mesh -- "
                           "shrinkage screening skipped")
        else:
            alloy_family, assumed = resolve_alloy_family(alloy, props_json)
            screen = screen_casting(pos=mesh.node_xyz, tets=mesh.tets,
                                    surface_nodes=surface_nodes, t_s=seconds,
                                    TL=props["TL"], TS=props["TS"],
                                    alloy_family=alloy_family, gravity_vec=gravity,
                                    feed_nodes=feed_nodes)
            shrinkage = _shrinkage_result_for_return(screen)
            if assumed:
                shrinkage["alloy_family_assumed"] = True
            point_data = _shrinkage_point_data(mesh, screen)

    vtu = osp.join(out_dir, f"{name}.vtu")
    write_vtu(vtu, mesh.node_xyz, mesh.tets, seconds, extra_point_data=point_data)

    order = np.argsort(seconds)[::-1][:5]
    hot_xyz = mesh.node_xyz[order]
    total = float(seconds.max())
    info = details["recomposition"]
    summary = osp.join(out_dir, f"{name}_summary.txt")
    alloy_label = (f"custom ({props_json})" if props_json is not None
                   else alloy if alloy is not None else DEFAULT_ALLOY)
    mold_label = mold if props_json is None else "n/a (props_json)"
    bbox_mm = tuple(round(float(v), 1) for v in mesh.node_xyz.max(0) - mesh.node_xyz.min(0))
    elem_label = (f"{minfo['elem_size']:.2f} mm" if minfo["elem_size"] is not None
                  else "n/a (density-searched)")
    with open(summary, "w") as f:
        f.write(f"casting: {name}\n")
        f.write(f"total solidification time: {total:.1f} s\n")
        f.write(f"alloy: {alloy_label}  mold: {mold_label}  "
                f"model: {getattr(model, 'version', 'unknown')}\n")
        f.write(f"mesh: {minfo['mode']}  cache: {minfo['cache']}  nodes: {minfo['n_nodes']}  "
                f"tets: {minfo['n_tets']}  elem_size: {elem_label}  bbox_mm: {bbox_mm}\n")
        line = (f"recomposition: a={info['a']:.4f}, m={info['m']:.4f}, "
                f"target_unreachable={info['target_unreachable']}, "
                f"clip_frac={info['clip_frac']:.4f}")
        if abs(info.get("total_deviation_frac", 0.0)) > 0.001:
            line += f", total_dev={info['total_deviation_frac'] * 100.0:+.1f}%"
        f.write(line + "\n")
        if screen_shrinkage:
            _write_shrinkage_summary(f, shrinkage)
        f.write("top hot spots (x,y,z mm : t_solidus s):\n")
        for xyz_i, i in zip(hot_xyz, order):
            f.write(f"  ({xyz_i[0]:.1f},{xyz_i[1]:.1f},{xyz_i[2]:.1f}) : {seconds[i]:.1f}\n")
        if warnings:
            f.write("WARNINGS:\n")
            for w in warnings:
                f.write(f"  - {w}\n")

    return {"name": name, "vtu": vtu, "summary": summary, "msh": out_msh,
            "total_time_s": total, "hotspot_xyz": hot_xyz, "ood_warnings": warnings,
            "n_nodes": minfo["n_nodes"], "mesh_mode": minfo["mode"],
            "elem_size": minfo["elem_size"], "cache": minfo["cache"],
            "recomposition_info": info, "shrinkage": shrinkage}


def _parse_gravity(s: str) -> tuple:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--gravity must be 'x,y,z' (3 comma-separated numbers), got {s!r}")
    vec = tuple(float(p) for p in parts)
    if all(v == 0.0 for v in vec):
        raise ValueError(f"--gravity must not be the zero vector (got {s!r})")
    return vec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step_path")
    ap.add_argument("--out-dir", default="runs/castings")
    ap.add_argument("--alloy", default=None,
                    help=f"alloy from data/properties/materials.csv (default {DEFAULT_ALLOY})")
    ap.add_argument("--mold", default=DEFAULT_MOLD,
                    help=f"mold from data/properties/molds.csv (default {DEFAULT_MOLD})")
    ap.add_argument("--props", dest="props_json", default=None,
                    help="JSON file with k, rho, cp, L_kJ_kg, TL, TS, Tpour, mold_effusivity "
                         "(replaces --alloy and --mold)")
    ap.add_argument("--elem-size", type=float, default=None,
                    help="mesh at this fixed element size in mm instead of searching for "
                         f"about {DEFAULT_TARGET_NODES} nodes")
    ap.add_argument("--gravity", type=_parse_gravity, default=(0, 0, -1),
                    help="gravity direction 'x,y,z' for sinks and feeder placement "
                         "(default 0,0,-1)")
    ap.add_argument("--no-shrinkage", dest="screen_shrinkage", action="store_false",
                    help="skip the shrinkage screening")
    ap.add_argument("--weights-dir", default=None,
                    help="local folder with config.json and the weight files "
                         "(default: download from the Hugging Face Hub)")
    a = ap.parse_args(argv)
    r = predict_casting(a.step_path, out_dir=a.out_dir, alloy=a.alloy, mold=a.mold,
                        props_json=a.props_json, elem_size=a.elem_size,
                        screen_shrinkage=a.screen_shrinkage, gravity=a.gravity,
                        weights_dir=a.weights_dir)
    print(f"{r['name']}: total {r['total_time_s']:.1f}s  nodes {r['n_nodes']}  -> {r['vtu']}")
    for w in r["ood_warnings"]:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
