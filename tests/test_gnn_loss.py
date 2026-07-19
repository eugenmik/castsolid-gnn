import torch
from src.train_gnn import gnn_loss


def test_zero_loss_on_perfect_prediction():
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    batch = torch.tensor([0, 0, 1, 1])
    assert gnn_loss(pred, pred.clone(), batch).item() < 1e-6


def test_spatial_term_penalizes_flipped_within_graph():
    # same per-graph mean, opposite within-graph deviation -> spatial term > 0
    target = torch.tensor([0.0, 2.0])           # graph 0, mean 1, devs [-1,+1]
    pred = torch.tensor([2.0, 0.0])             # mean 1, devs [+1,-1]
    batch = torch.tensor([0, 0])
    loss = gnn_loss(pred, target, batch, spatial_weight=1.0)
    assert loss.item() > 0.5
