"""Real-casting inference: hybrid field prediction + VTU output.

predict_field combines the GNN field shape with the baseline per-graph scale
(src/hybrid.py). write_vtu exports a ParaView file. load_hybrid loads the
Phase-1 checkpoints. The predict_casting orchestrator (Task 5) ties in meshing."""
import os.path as osp
import numpy as np
import torch
import joblib

from src.features import compute_features_inference, FEATURE_NAMES
from src.dataload import standardize
from src.baseline import predict_seconds
from src.graph_build import knn_edge_index, edge_attr
from src.model_gnn import SolidGNN
from src.evaluate_gnn import predict_seconds_gnn
from src.hybrid import hybrid_field


def predict_field(mesh, gnn_model, hgb_model, norm, k=16, device=None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feat = compute_features_inference(mesh)
    X, pos = feat["X"], feat["pos"]
    Xs = standardize(X, norm)
    b = predict_seconds(hgb_model, Xs, norm)                 # baseline scale (seconds)

    pos_t = torch.tensor(pos, dtype=torch.float32)
    ei = knn_edge_index(pos_t, k=k)
    ea = edge_attr(pos_t, ei)
    xg = torch.tensor(Xs, dtype=torch.float32)
    gnn_model = gnn_model.to(device).eval()
    with torch.no_grad():
        yn, _ = gnn_model(xg.to(device), ei.to(device), ea.to(device),
                          torch.zeros(xg.shape[0], dtype=torch.long, device=device))
    g = predict_seconds_gnn(yn.cpu().numpy(), norm)           # GNN shape (seconds)
    return hybrid_field(g, b)


def write_vtu(out_path, node_xyz, tets, seconds, top_frac=0.1):
    import meshio
    seconds = np.asarray(seconds, dtype=np.float32)
    n = seconds.shape[0]
    k = max(1, int(round(n * top_frac)))
    thresh = np.sort(seconds)[-k]
    hotspot = (seconds >= thresh).astype(np.int32)
    m = meshio.Mesh(points=np.asarray(node_xyz, dtype=np.float64),
                    cells=[("tetra", np.asarray(tets))],
                    point_data={"solidification_time": seconds, "hotspot": hotspot})
    m.write(out_path)


def load_hybrid(gnn_ckpt, hgb_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(gnn_ckpt, map_location=device, weights_only=False)
    cfg, norm = state["cfg"], state["norm"]
    gnn = SolidGNN(hidden=cfg["hidden"], n_blocks=cfg["n_blocks"]).to(device)
    gnn.load_state_dict(state["model"]); gnn.eval()
    hgb = joblib.load(hgb_path)
    return gnn, hgb, norm, cfg["k"]


import os
import argparse

from src.step_mesh import mesh_step
from src.msh_parse import parse_msh
from src import ood as _ood


def predict_casting(step_path, out_dir="runs/castings", elem_size=None,
                    gnn_ckpt="runs/gnn12/best_gnn.pt",
                    hgb_path="runs/baseline_v2/hgb_A.joblib",
                    ood_json="data/features_v2/ood_ranges.json", device=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    name = osp.splitext(osp.basename(step_path))[0]
    out_msh = osp.join(out_dir, f"{name}.msh")
    minfo = mesh_step(step_path, out_msh, elem_size)
    mesh = parse_msh(out_msh)

    gnn, hgb, norm, k = load_hybrid(gnn_ckpt, hgb_path, device)
    seconds = predict_field(mesh, gnn, hgb, norm, k=k, device=device)

    warnings = []
    if osp.exists(ood_json):
        ranges = _ood.load_ranges(ood_json)
        X = compute_features_inference(mesh)["X"]
        vol = float(X[0, FEATURE_NAMES.index("volume_mm3")])
        gcmax = float(X[:, FEATURE_NAMES.index("geodesic_d_cool")].max())
        diag = float(np.linalg.norm(mesh.node_xyz.max(0) - mesh.node_xyz.min(0)))
        warnings = _ood.ood_flags(vol, diag, gcmax, ranges)

    vtu = osp.join(out_dir, f"{name}.vtu")
    write_vtu(vtu, mesh.node_xyz, mesh.tets, seconds)

    # hot-spot summary: top-5 latest-solidifying nodes
    order = np.argsort(seconds)[::-1][:5]
    hot_xyz = mesh.node_xyz[order]
    total = float(seconds.max())
    summary = osp.join(out_dir, f"{name}_summary.txt")
    with open(summary, "w") as f:
        f.write(f"casting: {name}\n")
        f.write(f"total solidification time: {total:.1f} s\n")
        f.write(f"nodes: {minfo['n_nodes']}  tets: {minfo['n_tets']}  "
                f"elem_size: {minfo['elem_size']:.2f} mm  bbox_mm: "
                f"{tuple(round(v,1) for v in minfo['bbox_mm'])}\n")
        f.write("top hot spots (x,y,z mm : t_solidus s):\n")
        for xyz_i, i in zip(hot_xyz, order):
            f.write(f"  ({xyz_i[0]:.1f},{xyz_i[1]:.1f},{xyz_i[2]:.1f}) : {seconds[i]:.1f}\n")
        if warnings:
            f.write("WARNINGS:\n")
            for w in warnings:
                f.write(f"  - {w}\n")

    return {"name": name, "vtu": vtu, "summary": summary, "total_time_s": total,
            "n_nodes": minfo["n_nodes"], "elem_size": minfo["elem_size"],
            "ood_warnings": warnings, "hotspot_xyz": hot_xyz}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step_path")
    ap.add_argument("--out-dir", default="runs/castings")
    ap.add_argument("--elem-size", type=float, default=None)
    a = ap.parse_args()
    r = predict_casting(a.step_path, out_dir=a.out_dir, elem_size=a.elem_size)
    print(f"{r['name']}: total {r['total_time_s']:.1f}s  nodes {r['n_nodes']}  -> {r['vtu']}")
    for w in r["ood_warnings"]:
        print("WARNING:", w)
