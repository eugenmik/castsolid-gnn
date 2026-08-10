"""Learned-heads Task 2: frozen-backbone trainer for the gradT/dTdt deriv
heads (Task 1's `SolidGNN(deriv_heads=True)` / `attach_deriv_heads`).

Loads a headless production checkpoint (`{model, norm, cfg}`), attaches the
two new heads (`grad_head`/`dtdt_head`), freezes EVERY other parameter, and
trains only the ~8 head tensors to regress standardized log1p(gradT) /
log1p(|dTdt|) targets read straight from the v4 feature cache (each
`data/features_v4/graphs/*.npz` already carries raw per-node `gradT` (K/m)
and `dTdt` (K/s) arrays, see `src/build_dataset.py`). Base weights are never
touched (verified by an assertion here, and by the caller's own
byte-identity check).
"""
import json
import logging
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from src.gnn_dataset import SolidGraphDataset
from src.model_gnn import SolidGNN, attach_deriv_heads

logger = logging.getLogger(__name__)


class DerivGraphDataset(SolidGraphDataset):
    """`SolidGraphDataset` plus a standardized `y_deriv` (N, 2) node target:
    `[(log1p(gradT) - mean) / std, (log1p(|dTdt|) - mean) / std]`, read from
    the same graph cache. Callers must pre-filter `stems` to ones known to
    carry valid `gradT`/`dTdt` (see `_filter_stems_with_deriv`) -- this class
    does not re-validate.
    """

    def __init__(self, features_dir, stems, deriv_mean, deriv_std, k=16,
                 graph="union", norm=None):
        super().__init__(features_dir, stems, k=k, augment=False, norm=norm, graph=graph)
        self.deriv_mean = torch.tensor(np.asarray(deriv_mean), dtype=torch.float32)
        self.deriv_std = torch.tensor(np.asarray(deriv_std), dtype=torch.float32)

    def __getitem__(self, i):
        data = super().__getitem__(i)
        raw = np.load(osp.join(self.dir, "graphs", f"{self.stems[i]}.npz"))
        gradT = torch.tensor(raw["gradT"], dtype=torch.float32).clamp(min=0.0)
        dTdt = torch.tensor(raw["dTdt"], dtype=torch.float32).abs()
        deriv_log = torch.stack([torch.log1p(gradT), torch.log1p(dTdt)], dim=-1)
        data.y_deriv = (deriv_log - self.deriv_mean) / self.deriv_std
        return data


def _stem_has_valid_deriv(features_dir, stem):
    """A stem is usable if its cached graph has finite gradT/dTdt arrays,
    each with exactly one value per node (matching X's row count -- a
    length-mismatched stem would otherwise broadcast silently into the
    loss). A missing `.npz` file is also treated as an exclusion (counted
    and logged by the caller), not a raw crash."""
    path = osp.join(features_dir, "graphs", f"{stem}.npz")
    try:
        d = np.load(path)
    except FileNotFoundError:
        return False
    if "gradT" not in d.files or "dTdt" not in d.files or "X" not in d.files:
        return False
    gradT, dTdt, X = d["gradT"], d["dTdt"], d["X"]
    if gradT.shape != dTdt.shape or gradT.shape[0] != X.shape[0]:
        return False
    return bool(np.all(np.isfinite(gradT)) and np.all(np.isfinite(dTdt)))


def _filter_stems_with_deriv(features_dir, stems):
    """Split `stems` into (ok, n_excluded), logging loudly (never silently
    dropping) every stem missing/invalid gradT/dTdt."""
    ok, excluded = [], []
    for stem in stems:
        if _stem_has_valid_deriv(features_dir, stem):
            ok.append(stem)
        else:
            excluded.append(stem)
    if excluded:
        logger.warning(
            "train_heads: excluding %d/%d stem(s) missing/invalid gradT/dTdt: %s",
            len(excluded), len(stems), excluded)
    return ok, len(excluded)


def _compute_deriv_stats(features_dir, stems):
    """Streamed (single accumulation pass, one stem's arrays in memory at a
    time -- never the whole train-pool's per-node values at once) mean/std
    of `[log1p(gradT), log1p(|dTdt|)]` over every node in `stems`."""
    n = 0
    s1 = np.zeros(2, dtype=np.float64)
    s2 = np.zeros(2, dtype=np.float64)
    for stem in stems:
        d = np.load(osp.join(features_dir, "graphs", f"{stem}.npz"))
        gradT = np.clip(d["gradT"].astype(np.float64), 0.0, None)
        dTdt = np.abs(d["dTdt"].astype(np.float64))
        vals = np.stack([np.log1p(gradT), np.log1p(dTdt)], axis=-1)
        n += vals.shape[0]
        s1 += vals.sum(axis=0)
        s2 += (vals ** 2).sum(axis=0)
    if n == 0:
        raise ValueError("train_heads: no valid stems to compute deriv_log_mean/std from")
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _freeze_all_but_deriv_heads(model):
    """Freeze every parameter except `grad_head.*`/`dtdt_head.*`. Exposed
    standalone (mirrors `train_gnn._freeze_all_but_last2_and_scale_head`) so
    the freeze scope can be unit-tested directly."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("grad_head.") or name.startswith("dtdt_head.")


def _run_epoch_heads(model, loader, device, opt=None):
    train_mode = opt is not None
    model.train(train_mode)
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        g = getattr(batch, "g", None)
        if train_mode:
            opt.zero_grad()
            _, _, deriv = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, g=g)
            loss = F.mse_loss(deriv, batch.y_deriv)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                _, _, deriv = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, g=g)
                loss = F.mse_loss(deriv, batch.y_deriv)
        total += float(loss) * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


def train_deriv_heads(features_dir: str, base_ckpt: str, out_dir: str = "runs/lh_stage1",
                       train_stems=None, val_stems=None, epochs: int = 60, lr: float = 1e-3,
                       batch_size: int = 2, num_workers: int = 4, patience: int = 10,
                       device=None) -> dict:
    """Freeze the entire base model and train only the deriv (gradT/dTdt) heads.

    Args:
        features_dir: v4 feature cache dir (`graphs/*.npz` carry X/pos/y/
            mesh_edges/g plus raw `gradT`/`dTdt`).
        base_ckpt: path to a headless `{model, norm, cfg}` production
            checkpoint (Task 1's `attach_deriv_heads` contract: only
            grad_head/dtdt_head keys may be missing from it).
        out_dir: where `heads_gnn.pt` (best-val checkpoint) and per-epoch
            `ckpt_ep{ep:03d}.pt` snapshots are written.
        train_stems, val_stems: stem lists; default to `splits.json`'s
            `interp.train` / **`interp.test`** (NOT `extrap.test`: per the
            spec's 2026-08-01 Task-2-review correction, `extrap.test` is
            72% of the Gate C evaluation set, so early-stopping/checkpoint
            selection on it would leak into that pre-authorized gate;
            `interp.test` is disjoint from both Gate C evaluation lists
            (`extrap.test` + `e_val`)).
        epochs, lr, batch_size, num_workers, patience: as in `src.train_gnn.train`.
        device: torch device string; default cuda if available else cpu.

    Freezes ALL base parameters (`requires_grad=False`), asserting the
    optimizer's trainable-parameter set is EXACTLY the head tensors
    (`grad_head.*`/`dtdt_head.*`). Computes `deriv_log_mean`/`deriv_log_std`
    over the (filtered) train pool ONCE, streamed, and stores them in the
    saved checkpoint's `norm` dict (base norm + these two (2,) arrays). Loss
    is MSE over the two standardized heads. Stems whose cached graph is
    missing/length-mismatched gradT/dTdt (or has non-finite values), or
    whose `.npz` file is simply absent, are excluded loudly and counted in
    `n_excluded`, rather than silently dropped or crashing the run. The
    graph mode/kNN-k are read verbatim from the base checkpoint's own cfg
    (`cfg["graph"]`/`cfg["k"]`, no divergent defaults -- a cfg missing
    either raises `KeyError` loudly rather than silently training/serving
    with a mismatched graph construction). Saves a checkpoint every epoch
    (`<out_dir>/ckpt_ep{ep:03d}.pt`, gate-metric checkpoint-selection
    precedent from the exo-fix retrain) IN ADDITION to early-stopping on val
    loss and keeping the MSE-best as `<out_dir>/heads_gnn.pt`, with `cfg` =
    base cfg + `{"deriv_heads": True}`.

    Returns:
        {"ckpt": str, "best_val": float, "n_train": int, "n_excluded": int}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    if train_stems is None or val_stems is None:
        with open(osp.join(features_dir, "splits.json")) as f:
            splits = json.load(f)
        if train_stems is None:
            train_stems = splits["interp"]["train"]
        if val_stems is None:
            val_stems = splits["interp"]["test"]

    train_ok, n_exc_train = _filter_stems_with_deriv(features_dir, train_stems)
    val_ok, n_exc_val = _filter_stems_with_deriv(features_dir, val_stems)
    n_excluded = n_exc_train + n_exc_val
    if not train_ok:
        raise ValueError("train_deriv_heads: no train stems with valid gradT/dTdt")
    if not val_ok:
        raise ValueError("train_deriv_heads: no val stems with valid gradT/dTdt")

    deriv_mean, deriv_std = _compute_deriv_stats(features_dir, train_ok)

    base = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    cfg = dict(base["cfg"])
    base_norm = base["norm"]

    model = SolidGNN(node_in=cfg.get("node_in", 15), edge_in=cfg.get("edge_in", 4),
                      hidden=cfg["hidden"], n_blocks=cfg["n_blocks"],
                      two_head=cfg.get("two_head", False), film=cfg.get("film", False),
                      g_dim=cfg.get("g_dim", 5), deriv_heads=True)
    attach_deriv_heads(model, base["model"])
    model = model.to(device)

    _freeze_all_but_deriv_heads(model)

    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    expected_names = {n for n, _ in model.named_parameters()
                       if n.startswith("grad_head.") or n.startswith("dtdt_head.")}
    assert trainable_names == expected_names, (
        f"train_deriv_heads: optimizer param set must be exactly the deriv-head "
        f"tensors -- unexpected trainable: {sorted(trainable_names - expected_names)}, "
        f"missing: {sorted(expected_names - trainable_names)}")
    trainable_params = [p for n, p in model.named_parameters() if n in trainable_names]

    opt = AdamW(trainable_params, lr=lr, weight_decay=1e-5)

    # No divergent .get(default) here: the graph mode/k must match whatever
    # the base checkpoint was actually trained/normalized with, or the deriv
    # heads would be trained (and later served) on a mismatched graph
    # construction. A base cfg missing either key is a loud KeyError.
    graph = cfg["graph"]
    k = cfg["k"]

    tr_ds = DerivGraphDataset(features_dir, train_ok, deriv_mean, deriv_std,
                              k=k, graph=graph, norm=base_norm)
    va_ds = DerivGraphDataset(features_dir, val_ok, deriv_mean, deriv_std,
                              k=k, graph=graph, norm=base_norm)
    trl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, persistent_workers=num_workers > 0)
    val = DataLoader(va_ds, batch_size=batch_size, num_workers=num_workers,
                      persistent_workers=num_workers > 0)

    saved_norm = dict(base_norm)
    saved_norm["deriv_log_mean"] = deriv_mean
    saved_norm["deriv_log_std"] = deriv_std
    saved_cfg = dict(cfg)
    saved_cfg["deriv_heads"] = True

    ckpt_path = osp.join(out_dir, "heads_gnn.pt")
    best, since = float("inf"), 0
    ep = -1
    for ep in range(epochs):
        tr_loss = _run_epoch_heads(model, trl, device, opt=opt)
        vloss = _run_epoch_heads(model, val, device, opt=None)
        print(f"  epoch {ep:3d}  train {tr_loss:.4f}  val {vloss:.4f}", flush=True)

        ckpt_dict = {"model": model.state_dict(), "norm": saved_norm, "cfg": saved_cfg}

        # Per-epoch checkpoint (gate-metric checkpoint-selection precedent:
        # the exo-fix retrain needed to compare 8 candidates), independent
        # of the best-val bookkeeping below.
        torch.save(ckpt_dict, osp.join(out_dir, f"ckpt_ep{ep:03d}.pt"))

        if vloss < best - 1e-6:
            best, since = vloss, 0
            torch.save(ckpt_dict, ckpt_path)
        else:
            since += 1
            if since >= patience:
                break

    return {"ckpt": ckpt_path, "best_val": best, "n_train": len(train_ok),
            "n_excluded": n_excluded}
