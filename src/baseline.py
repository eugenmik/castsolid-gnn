"""Per-node regression baselines: gradient boosting (primary) and a small MLP.

Both predict standardized log1p(solidification_time) from standardized
physics features. predict_seconds inverts the transform back to seconds.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor


def train_hgb(X_std, y_log, max_iter: int = 300):
    model = HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        random_state=0)
    model.fit(X_std, y_log)          # trained directly on log1p(y)
    return model


class _MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_std, y_log, epochs: int = 30, device: str = "cpu", batch: int = 8192):
    torch.manual_seed(0)
    X = torch.tensor(np.asarray(X_std), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y_log), dtype=torch.float32, device=device)
    model = _MLP(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict_seconds(model, X_std, norm) -> np.ndarray:
    if isinstance(model, nn.Module):
        dev = next(model.parameters()).device
        with torch.no_grad():
            pred_log = model(
                torch.tensor(np.asarray(X_std), dtype=torch.float32, device=dev)
            ).cpu().numpy()
    else:
        pred_log = model.predict(X_std)
    # Both models predict raw log1p(seconds); invert uniformly.
    return np.clip(np.expm1(pred_log), 0.0, None)
