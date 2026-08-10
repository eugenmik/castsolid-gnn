"""GNN training: loss, training loop, checkpointing. Trains on standardized
log1p targets with rotation augmentation; validates on a held-out fraction."""
import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


def gnn_loss(pred, target, batch, spatial_weight: float = 0.3, global_weight: float = 0.0):
    base = F.smooth_l1_loss(pred, target)
    pm = scatter(pred, batch, dim=0, reduce="mean")
    tm = scatter(target, batch, dim=0, reduce="mean")
    spatial = F.mse_loss(pred - pm[batch], target - tm[batch])
    glob = F.smooth_l1_loss(pm, tm)
    return base + spatial_weight * spatial + global_weight * glob


import os
import os.path as osp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from src.gnn_dataset import SolidGraphDataset
from src.model_gnn import SolidGNN
from src.warm_start import build_or_load_v4, build_or_load_v5

# bc-reach (Task 1/2): the v5 BC-reach feature cache's per-node width --
# FEATURE_NAMES_V4 (15) + the 5 new multi-scale BC-reach channels
# (exo_mid/exo_long/chill_short/chill_mid/chill_long, `src.features.
# FEATURE_NAMES_V5`). `train()`'s `init_ckpt` dispatch below compares the
# ACTUAL training data's own feature width (`node_in`, computed from the
# built `tr` dataset -- never the checkpoint's) against this constant to
# decide whether to warm-start through `build_or_load_v5` (rebuild the node
# encoder for 20 columns) instead of `build_or_load_v4` (the existing
# 15-column FT2b/FT3 path, unaffected). Named here (review round 1, F3)
# instead of an inline literal `20`, so the pin's provenance is legible at
# the call site.
_V5_NODE_IN = 20


def _freeze_backbone_params(model):
    """Freeze everything except FiLM (used for FT2b stage-1 training: only the
    per-block gamma/beta conditioning is trainable, so warm-started SP1 shape
    knowledge is preserved while the model learns to respond to the material
    g-vector). The new BC input columns live inside node_encoder's first
    Linear, so they are frozen too in stage 1 -- they get unfrozen (along with
    the rest of the backbone) in stage 2 per the FINETUNING plan's stage table.
    """
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("films.")


# Public name for tests/other callers. `train()` below takes a same-named
# `freeze_backbone` bool parameter, which would shadow this at module scope
# inside that function's body -- so `train()` calls `_freeze_backbone_params`
# directly instead of this alias.
freeze_backbone = _freeze_backbone_params


def _freeze_all_but_last2_and_scale_head(model):
    """FT2b stage-3 (calibrate) freeze scope: everything is frozen except the
    LAST 2 processor blocks and the scale head -- the small, late-stage
    capacity that plausibly still needs to adapt to a handful of named-grade
    (steel/ductile-iron/grey-iron) + real-part calibration rows, without
    letting so few rows overwrite the shape knowledge learned in stages 1-2.
    FiLM is deliberately NOT included here (unlike stage 1's `freeze_backbone`,
    which trains FiLM only) -- stage 3 is calibrating the backbone's late
    representation + scale, not re-learning material conditioning.
    """
    n_blocks = len(model.blocks)
    keep_prefixes = tuple(f"blocks.{i}." for i in range(max(n_blocks - 2, 0), n_blocks))
    keep_prefixes = keep_prefixes + ("scale_head.",)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(keep_prefixes)


# Public name, mirroring `freeze_backbone` above.
freeze_all_but_last2_and_scale_head = _freeze_all_but_last2_and_scale_head


def _run_epoch(model, loader, device, opt=None, scaler=None, spatial_weight=0.3, global_weight=0.0):
    train_mode = opt is not None
    model.train(train_mode)
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        g = getattr(batch, "g", None)
        if train_mode:
            opt.zero_grad()
            with torch.autocast(device_type=device.split(":")[0], enabled=scaler is not None):
                yn, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, g=g)
                loss = gnn_loss(yn, batch.y, batch.batch, spatial_weight, global_weight)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
        else:
            with torch.no_grad():
                yn, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, g=g)
                loss = gnn_loss(yn, batch.y, batch.batch, spatial_weight, global_weight)
        total += float(loss) * batch.num_graphs; n += batch.num_graphs
    return total / max(n, 1)


def _build_checkpoint_dict(model, norm, hidden, n_blocks, k, node_in, edge_in,
                            spatial_weight, global_weight, graph, feature_version,
                            two_head, film, g_dim):
    """Build the checkpoint dict payload (same format as best_gnn.pt).

    Args:
        model: the SolidGNN model.
        norm: normalization stats (SolidGraphDataset.norm).
        All other args: hyperparameters/config to store.

    Returns:
        dict with keys {model, norm, cfg}.
    """
    return {
        "model": model.state_dict(),
        "norm": norm,
        "cfg": {
            "hidden": hidden,
            "n_blocks": n_blocks,
            "k": k,
            "node_in": node_in,
            "edge_in": edge_in,
            "spatial_weight": spatial_weight,
            "global_weight": global_weight,
            "graph": graph,
            "feature_version": feature_version,
            "two_head": two_head,
            "film": film,
            "g_dim": g_dim,
        },
    }


def train(features_dir, train_stems, val_stems, epochs=200, k=16, hidden=128,
          n_blocks=12, batch_size=8, lr=1e-3, device=None, out_dir="runs/gnn",
          augment=True, patience=25, spatial_weight=0.3, global_weight=0.0,
          graph="knn", feature_version=2, num_workers=0, two_head=False,
          film=False, g_dim=5, freeze_backbone: bool = False, init_ckpt=None,
          early_stop_val_stems=None, freeze_mode=None, save_every_epoch=False,
          norm_override=None):
    """
    film: build the model with FiLM (per-block gamma/beta from data.g); forced
        True when init_ckpt is given (warm-start is always onto a FiLM model).
    g_dim: FiLM conditioning-vector width (ignored unless film/init_ckpt).
    freeze_backbone: freeze node_encoder/edge_encoder/blocks/decoder/scale_head,
        leaving only FiLM params trainable (FT2b stage-1: shape/scale knowledge
        from warm-start is preserved while FiLM learns to respond to `g`).
        Back-compat boolean from Task 5; equivalent to `freeze_mode="backbone"`.
    freeze_mode: FT2b's more general freeze-scope switch, superset of
        `freeze_backbone`. One of:
          - None (default): use `freeze_backbone`'s boolean behavior (freeze
            everything but FiLM if True, nothing if False) -- unchanged from
            the pre-FT2b/Task-5 code path.
          - "backbone": same as `freeze_backbone=True` (only FiLM trainable).
          - "all_but_last2": only the last 2 processor blocks + scale_head are
            trainable (FT2b stage 3 / calibrate -- see
            `_freeze_all_but_last2_and_scale_head`'s docstring).
        `freeze_mode` takes precedence over `freeze_backbone` when both are given.
    init_ckpt: path to an SP1 checkpoint (node_in=13, no FiLM) OR an earlier
        FT2b-stage checkpoint (node_in=15, FiLM already present -- stage 2
        chains from stage 1's `best_gnn.pt`, stage 3 from stage 2's) to warm-
        start from via `src.warm_start.build_or_load_v4`, instead of a
        randomly initialized model. `build_or_load_v4` dispatches on the
        checkpoint's own `cfg` (splice-from-SP1 vs. direct-load-an-already-v4-
        checkpoint) -- see its docstring. Implies film=True. bc-reach: when
        the ACTUAL training data (`node_in`, computed below from the built
        `tr` dataset's own feature width) equals `_V5_NODE_IN` (20) -- i.e.
        `features_dir` is a v5 (BC-reach) cache -- this dispatches to
        `src.warm_start.build_or_load_v5` instead, which rebuilds the node
        encoder's first Linear for the 5 new trailing columns (node_in 15 ->
        20) from a v4 `init_ckpt` (e.g. the production `ckpt_ep021.pt`), or
        direct-loads an already-v5 checkpoint. Every other `node_in` (in
        particular the existing 15-wide FT2b/FT3 path) is unaffected -- still
        routed through `build_or_load_v4`, byte-identical to before this
        dispatch existed.
        NOTE on normalization: the checkpoint's own `norm` (the SOURCE
        checkpoint's feature/target stats -- SP1's v3 stats for the original
        13->15 warm-start, or a v4 checkpoint's 15-wide stats for the
        15->20 bc-reach one) is NOT reused here -- the training data in
        `features_dir` (whichever cache version it actually is: v3/v4/v5)
        is normalized (by `SolidGraphDataset`) with *its own*
        `norm_stats.npz` (`tr.norm`), since the model always consumes
        THAT cache's own normalized features, not the source checkpoint's.
        The saved checkpoint's `norm` field must match what the *data* was
        actually standardized with, so we always save `tr.norm`, discarding
        the `_sp1_norm` that `build_or_load_v4`/`build_or_load_v5` return
        (that value exists for callers who need to reconcile the two spaces
        explicitly, e.g. a sanity-gate check at warm-start time -- not for
        the trained ckpt).
    early_stop_val_stems: if given, checkpoint-selection/early-stopping use a
        loss computed on this stem list instead of `val_stems` (e.g. a
        val_extrap split held out for OOD generalization signal). When None,
        behavior is identical to the pre-FT2b code path (single val set/loader
        serving both roles).
    save_every_epoch: if True, save a checkpoint to `<out_dir>/ckpt_ep{ep:03d}.pt`
        after EVERY epoch (in the same format as best_gnn.pt: {model, norm, cfg}),
        in addition to the unchanged best_gnn.pt logic. Default False preserves
        byte-identical behavior (no ckpt_ep* files).
    norm_override: if given (a norm dict, e.g. a production checkpoint's own
        `ckpt["norm"]"), freeze normalization to it instead of computing/loading
        the feature cache's own `norm_stats.npz` (FT3: fine-tuning must not
        shift any standardization the production checkpoint was trained
        with). Passed as `norm=norm_override` into `tr`'s SolidGraphDataset
        construction; `va`/`va_es` already reuse `tr.norm` downstream, so all
        three dataset constructions end up standardized with `norm_override`,
        and the saved checkpoint's `norm` field (`tr.norm`) is `norm_override`
        too. Default None preserves byte-identical behavior (tr loads the
        cache's own norm_stats.npz, as before).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    tr = SolidGraphDataset(features_dir, train_stems, k=k, augment=augment, graph=graph,
                            norm=norm_override)
    va = SolidGraphDataset(features_dir, val_stems, k=k, augment=False, norm=tr.norm, graph=graph)
    trl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=num_workers > 0)
    val = DataLoader(va, batch_size=batch_size, num_workers=num_workers, persistent_workers=num_workers > 0)

    if early_stop_val_stems is not None:
        va_es = SolidGraphDataset(features_dir, early_stop_val_stems, k=k, augment=False,
                                   norm=tr.norm, graph=graph)
        val_es = DataLoader(va_es, batch_size=batch_size, num_workers=num_workers,
                             persistent_workers=num_workers > 0)
    else:
        val_es = None

    sample = tr[0]
    node_in, edge_in = sample.x.shape[1], sample.edge_attr.shape[1]

    if init_ckpt is not None:
        film = True
        if node_in == _V5_NODE_IN:
            model, _sp1_norm = build_or_load_v5(init_ckpt, hidden=hidden, n_blocks=n_blocks, g_dim=g_dim)
        else:
            model, _sp1_norm = build_or_load_v4(init_ckpt, hidden=hidden, n_blocks=n_blocks, g_dim=g_dim)
        model = model.to(device)
        # build_or_load_v4 (like build_warm_started_v4 before it) prefers the
        # *source* checkpoint's own cfg (hidden/n_blocks/two_head) over the
        # args passed here (two_head isn't even forwarded to it), so
        # re-derive them from the actual constructed model -- the saved cfg
        # below must describe the model we trained, not just echo back the
        # caller's arguments.
        hidden = model.decoder[0].in_features
        n_blocks = len(model.blocks)
        two_head = model.two_head
    else:
        model = SolidGNN(node_in=node_in, edge_in=edge_in, hidden=hidden, n_blocks=n_blocks,
                          two_head=two_head, film=film, g_dim=g_dim).to(device)

    if freeze_mode == "all_but_last2":
        _freeze_all_but_last2_and_scale_head(model)
    elif freeze_mode == "backbone" or (freeze_mode is None and freeze_backbone):
        _freeze_backbone_params(model)
    elif freeze_mode not in (None,):
        raise ValueError(f"train(): unknown freeze_mode={freeze_mode!r}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable_params, lr=lr, weight_decay=1e-5)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None

    best, best_ep, since = float("inf"), 0, 0
    ckpt = osp.join(out_dir, "best_gnn.pt")
    ep = -1
    for ep in range(epochs):
        tr_loss = _run_epoch(model, trl, device, opt=opt, scaler=scaler, spatial_weight=spatial_weight, global_weight=global_weight)
        # Checkpoint selection/early-stop use shape-only val loss: the global term is
        # noisy at batch=2 and masks shape improvements (Run-1 lesson).
        if val_es is not None:
            vloss = _run_epoch(model, val_es, device, spatial_weight=spatial_weight, global_weight=0.0)
        else:
            vloss = _run_epoch(model, val, device, spatial_weight=spatial_weight, global_weight=0.0)
        sched.step(vloss)
        lr = opt.param_groups[0]["lr"]
        print(f"  epoch {ep:3d}  train {tr_loss:.4f}  val {vloss:.4f}  lr {lr:.2e}", flush=True)

        # Build checkpoint dict (used for both best_gnn.pt and per-epoch saves)
        ckpt_dict = _build_checkpoint_dict(
            model, tr.norm, hidden, n_blocks, k, node_in, edge_in,
            spatial_weight, global_weight, graph, feature_version,
            two_head, film, g_dim
        )

        # Save per-epoch checkpoint if requested
        if save_every_epoch:
            ep_ckpt = osp.join(out_dir, f"ckpt_ep{ep:03d}.pt")
            torch.save(ckpt_dict, ep_ckpt)

        # Save best checkpoint if this is a new best
        if vloss < best - 1e-5:
            best, best_ep, since = vloss, ep, 0
            torch.save(ckpt_dict, ckpt)
        else:
            since += 1
            if since >= patience:
                break
    return {"best_val": best, "epochs_run": ep + 1, "ckpt": ckpt}
