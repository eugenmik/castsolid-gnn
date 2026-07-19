"""GNN training: loss, training loop, checkpointing. Trains on standardized
log1p targets with rotation augmentation; validates on a held-out fraction."""
import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


def gnn_loss(pred, target, batch, spatial_weight: float = 0.3):
    base = F.smooth_l1_loss(pred, target)
    pm = scatter(pred, batch, dim=0, reduce="mean")[batch]
    tm = scatter(target, batch, dim=0, reduce="mean")[batch]
    spatial = F.mse_loss(pred - pm, target - tm)
    return base + spatial_weight * spatial


import os
import os.path as osp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from src.gnn_dataset import SolidGraphDataset
from src.model_gnn import SolidGNN


def _run_epoch(model, loader, device, opt=None, scaler=None, spatial_weight=0.3):
    train_mode = opt is not None
    model.train(train_mode)
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        if train_mode:
            opt.zero_grad()
            with torch.autocast(device_type=device.split(":")[0], enabled=scaler is not None):
                yn, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = gnn_loss(yn, batch.y, batch.batch, spatial_weight)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
        else:
            with torch.no_grad():
                yn, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = gnn_loss(yn, batch.y, batch.batch, spatial_weight)
        total += float(loss) * batch.num_graphs; n += batch.num_graphs
    return total / max(n, 1)


def train(features_dir, train_stems, val_stems, epochs=200, k=16, hidden=128,
          n_blocks=12, batch_size=8, lr=1e-3, device=None, out_dir="runs/gnn",
          augment=True, patience=25):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    tr = SolidGraphDataset(features_dir, train_stems, k=k, augment=augment)
    va = SolidGraphDataset(features_dir, val_stems, k=k, augment=False, norm=tr.norm)
    trl = DataLoader(tr, batch_size=batch_size, shuffle=True)
    val = DataLoader(va, batch_size=batch_size)
    model = SolidGNN(hidden=hidden, n_blocks=n_blocks).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None

    best, best_ep, since = float("inf"), 0, 0
    ckpt = osp.join(out_dir, "best_gnn.pt")
    ep = -1
    for ep in range(epochs):
        tr_loss = _run_epoch(model, trl, device, opt=opt, scaler=scaler)
        vloss = _run_epoch(model, val, device)
        sched.step(vloss)
        lr = opt.param_groups[0]["lr"]
        print(f"  epoch {ep:3d}  train {tr_loss:.4f}  val {vloss:.4f}  lr {lr:.2e}", flush=True)
        if vloss < best - 1e-5:
            best, best_ep, since = vloss, ep, 0
            torch.save({"model": model.state_dict(), "norm": tr.norm,
                        "cfg": {"hidden": hidden, "n_blocks": n_blocks, "k": k}}, ckpt)
        else:
            since += 1
            if since >= patience:
                break
    return {"best_val": best, "epochs_run": ep + 1, "ckpt": ckpt}
